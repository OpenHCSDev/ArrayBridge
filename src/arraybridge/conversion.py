"""How an array gets from one library and device to another, and what was done.

``ConversionRoute`` leaves are tried in the order they are declared in this
module. ``applies`` reads only the two library declarations and the two device
descriptions, so the same test answers "which mechanism will be tried first" on
a machine without the device and selects the route when the conversion runs.

A plan sees no array. Where a mechanism depends on the particular array (a
DLPack capsule the target cannot read, a NumPy buffer the target cannot wrap),
the record of the conversion names the mechanism that was used and lists the
planned one under ``declined_mechanisms``.

A ``ConversionRecord`` is written when data was moved, copied or wrapped. The
route that leaves the array where it is (``StaysInPlace``) writes none.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, fields
from types import NotImplementedType
from typing import TYPE_CHECKING, ClassVar, TypeGuard, TypeVar

from metaclass_registry import AutoRegisterMeta

from arraybridge.devices import Device

if TYPE_CHECKING:
    from arraybridge.libraries import ArrayLibrary, DLPackImport

logger = logging.getLogger(__name__)

S = TypeVar("S")
T = TypeVar("T")


def _is_array_of_both(data: S, source: ArrayLibrary[S], target: ArrayLibrary[T]) -> TypeGuard[T]:
    """Whether ``data`` of ``source`` is an array of ``target`` too: they are one library."""

    del data
    return source is target


def _imports_dlpack(library: ArrayLibrary[T]) -> TypeGuard[DLPackImport[T]]:
    from arraybridge.libraries import DLPackImport

    return isinstance(library, DLPackImport)


class RouteDeclinedError(Exception):
    """A route that applied by declaration could not carry this particular array."""


class ConversionRoute(ABC, metaclass=AutoRegisterMeta):
    """One mechanism for moving an array between libraries and devices."""

    __registry_key__ = "mechanism"
    __registry__: ClassVar[Mapping[str, type[ConversionRoute]]] = {}
    mechanism: ClassVar[str | None] = None
    writes_a_record: ClassVar[bool] = True
    """Whether a conversion by this route is reported to the observer."""

    @classmethod
    @abstractmethod
    def applies(
        cls,
        source: ArrayLibrary[S],
        source_device: Device,
        target: ArrayLibrary[T],
        target_device: Device,
    ) -> bool:
        """Whether this route is declared for the pair. Touches no device."""

    @classmethod
    @abstractmethod
    def convert(
        cls,
        data: S,
        source: ArrayLibrary[S],
        target: ArrayLibrary[T],
        target_device: Device,
    ) -> T:
        """Carry ``data`` to ``target`` on ``target_device`` or raise ``RouteDeclinedError``."""

    @classmethod
    def taken(
        cls,
        data: S,
        result: T,
        source: ArrayLibrary[S],
        target: ArrayLibrary[T],
    ) -> type[ConversionRoute]:
        """The route that names what ``convert`` did to make ``result``: this one.

        A route whose mechanism can only be known once the conversion is done
        answers with another route here, and the record lists this one as declined.
        """

        del data, result, source, target
        return cls

    @classmethod
    def candidates(
        cls,
        source: ArrayLibrary[S],
        source_device: Device,
        target: ArrayLibrary[T],
        target_device: Device,
    ) -> tuple[type[ConversionRoute], ...]:
        """Every declared route for the pair, first choice first."""

        return tuple(
            route
            for route in cls.__registry__.values()
            if route.applies(source, source_device, target, target_device)
        )

    @classmethod
    def planned(
        cls,
        source: ArrayLibrary[S],
        source_device: Device,
        target: ArrayLibrary[T],
        target_device: Device,
    ) -> type[ConversionRoute]:
        """The route a conversion of this pair will try first.

        It is the route that runs unless the particular array cannot be carried
        by it, which no plan can know; the record then names the one that did.
        """

        return cls.candidates(source, source_device, target, target_device)[0]

    @classmethod
    def run(
        cls,
        data: S,
        source: ArrayLibrary[S],
        target: ArrayLibrary[T],
        target_device: Device,
    ) -> T:
        """Convert through the first candidate route that carries ``data``."""

        observer = ConversionObserver.current
        started = time.perf_counter() if observer.wants_records else 0.0
        source_device = source.device_of(data)
        declined: list[str] = []
        for route in cls.candidates(source, source_device, target, target_device):
            try:
                result = route.convert(data, source, target, target_device)
            except RouteDeclinedError as reason:
                logger.debug("%s declined %s to %s: %s", route.mechanism, source, target, reason)
                declined.append(str(route.mechanism))
                continue
            if observer.wants_records and route.writes_a_record:
                taken = route.taken(data, result, source, target)
                if taken is not route:
                    declined.append(str(route.mechanism))
                observer.record(
                    ConversionRecord(
                        source_library=source.name,
                        source_device=source_device.label,
                        target_library=target.name,
                        target_device=target_device.label,
                        mechanism=str(taken.mechanism),
                        nbytes=target.nbytes_of(result),
                        seconds=time.perf_counter() - started,
                        declined_mechanisms=tuple(declined),
                    )
                )
            return result
        raise LookupError(f"No conversion route from {source} to {target}")


class StaysInPlace(ConversionRoute):
    """Same library, same device: nothing is converted and the input is returned.

    This is the one statement of that rule. ``ConversionRoute.planned`` answers
    with this route, the conversion runs through it, and
    ``ArrayLibrary.move_to_device`` asks ``applies``. No record is written.
    """

    mechanism = "stays in place"
    writes_a_record = False

    @classmethod
    def applies(
        cls,
        source: ArrayLibrary[S],
        source_device: Device,
        target: ArrayLibrary[T],
        target_device: Device,
    ) -> bool:
        return source is target and source_device == target_device

    @classmethod
    def convert(
        cls,
        data: S,
        source: ArrayLibrary[S],
        target: ArrayLibrary[T],
        target_device: Device,
    ) -> T:
        if not _is_array_of_both(data, source, target):
            raise TypeError(f"{source} and {target} are different libraries")
        return data


class DeviceMove(ConversionRoute):
    """Same library, another device: the array stays native and changes device."""

    mechanism = "device move"

    @classmethod
    def applies(
        cls,
        source: ArrayLibrary[S],
        source_device: Device,
        target: ArrayLibrary[T],
        target_device: Device,
    ) -> bool:
        return source is target and not StaysInPlace.applies(
            source, source_device, target, target_device
        )

    @classmethod
    def convert(
        cls,
        data: S,
        source: ArrayLibrary[S],
        target: ArrayLibrary[T],
        target_device: Device,
    ) -> T:
        if not _is_array_of_both(data, source, target):
            raise TypeError(f"{source} and {target} are different libraries")
        return target.move_to_device(data, target_device)


class DLPackExchange(ConversionRoute):
    """Both libraries speak DLPack: the target wraps the source's buffer."""

    mechanism = "dlpack"

    @classmethod
    def applies(
        cls,
        source: ArrayLibrary[S],
        source_device: Device,
        target: ArrayLibrary[T],
        target_device: Device,
    ) -> bool:
        return target.imports_dlpack_from(source_device) and source.exports_dlpack_from(
            source_device
        )

    @classmethod
    def convert(
        cls,
        data: S,
        source: ArrayLibrary[S],
        target: ArrayLibrary[T],
        target_device: Device,
    ) -> T:
        try:
            if not _imports_dlpack(target):
                raise TypeError(f"{target} does not import DLPack")
            payload = source.export_dlpack(data)
            if payload is None:
                raise RouteDeclinedError("the source array has no DLPack export")
            with target.device_scope(target_device):
                result = target.from_dlpack(payload)
            if isinstance(result, NotImplementedType):
                raise RouteDeclinedError("the target cannot import this capsule")
            return target.move_to_device(result, target_device)
        except RouteDeclinedError:
            raise
        except Exception as error:
            logger.warning(
                "DLPack conversion from %s to %s failed: %s. Using CPU roundtrip.",
                source.name,
                target.name,
                error,
            )
            raise RouteDeclinedError(str(error)) from error


class HostExchange(ConversionRoute):
    """Through NumPy: the source hands over a NumPy array and the target takes it."""

    @classmethod
    def convert(
        cls,
        data: S,
        source: ArrayLibrary[S],
        target: ArrayLibrary[T],
        target_device: Device,
    ) -> T:
        return target.unrecorded_from_numpy(source.unrecorded_to_numpy(data), target_device)


class SharedHostBuffer(HostExchange):
    """Both arrays are in main memory and the result is the source's own buffer.

    Declared by the two libraries (``shares_numpy_buffer_on``). Whether one
    particular array is shared is known only afterwards, by comparing where the
    two arrays begin; an exchange that copied is recorded as a ``HostCopy``.
    """

    mechanism = "shared host buffer"

    @classmethod
    def applies(
        cls,
        source: ArrayLibrary[S],
        source_device: Device,
        target: ArrayLibrary[T],
        target_device: Device,
    ) -> bool:
        return source.shares_numpy_buffer_on(source_device) and target.shares_numpy_buffer_on(
            target_device
        )

    @classmethod
    def taken(
        cls,
        data: S,
        result: T,
        source: ArrayLibrary[S],
        target: ArrayLibrary[T],
    ) -> type[ConversionRoute]:
        address = source.host_address_of(data)
        shared = address is not None and address == target.host_address_of(result)
        return cls if shared else HostCopy


class HostCopy(HostExchange):
    """The values are copied: downloaded to NumPy in main memory, then uploaded."""

    mechanism = "host copy"

    @classmethod
    def applies(
        cls,
        source: ArrayLibrary[S],
        source_device: Device,
        target: ArrayLibrary[T],
        target_device: Device,
    ) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class ConversionRecord:
    """One conversion that moved, copied or wrapped data. Plain fields: pickles, dumps to JSON.

    ``mechanism`` names what was done. ``declined_mechanisms`` lists the routes
    tried before it that could not carry this array.
    """

    source_library: str
    source_device: str
    target_library: str
    target_device: str
    mechanism: str
    nbytes: int
    seconds: float
    declined_mechanisms: tuple[str, ...] = ()

    def to_json(self) -> dict[str, str | int | float | list[str]]:
        return {**asdict(self), "declined_mechanisms": list(self.declined_mechanisms)}

    @classmethod
    def from_json(cls, row: Mapping[str, str | int | float | list[str]]) -> ConversionRecord:
        declined = row["declined_mechanisms"]
        assert isinstance(declined, list)
        return cls(
            source_library=str(row["source_library"]),
            source_device=str(row["source_device"]),
            target_library=str(row["target_library"]),
            target_device=str(row["target_device"]),
            mechanism=str(row["mechanism"]),
            nbytes=int(str(row["nbytes"])),
            seconds=float(str(row["seconds"])),
            declined_mechanisms=tuple(declined),
        )


class ObserverOfThisContext(ABCMeta):
    """Gives ``ConversionObserver.current``: read and assigned per context."""

    @property
    def current(cls) -> ConversionObserver:
        """The observer of the running thread or asyncio task."""

        return _observer.get()

    @current.setter
    def current(cls, observer: ConversionObserver) -> None:
        _observer.set(observer)


class ConversionObserver(ABC, metaclass=ObserverOfThisContext):
    """Receives one ``ConversionRecord`` per conversion while it is installed.

    ``ConversionObserver.current`` is the one hook, and it belongs to the
    context (``contextvars``) that reads it. Every thread starts with the
    default, which ignores conversions and asks for no records, so nothing is
    timed or sized. An asyncio task starts with the observer of the code that
    created it. Installing an observer in one thread or task changes nothing
    in another; to observe the conversions of a thread you start, run it in a
    copy of your context: ``Thread(target=contextvars.copy_context().run,
    args=(work,))``.
    """

    wants_records: ClassVar[bool] = True

    @abstractmethod
    def record(self, record: ConversionRecord) -> None:
        """Take one finished conversion."""

    @contextmanager
    def installed(self) -> Iterator[ConversionObserver]:
        """Make this the observer of this context for the duration of a block."""

        previous = self.install()
        try:
            yield self
        finally:
            previous.install()

    def install(self) -> ConversionObserver:
        """Make this the observer of this context and return the one it replaced."""

        previous = _observer.get()
        _observer.set(self)
        return previous


class IgnoreConversions(ConversionObserver):
    """The default observer: conversions are neither measured nor kept."""

    wants_records = False

    def record(self, record: ConversionRecord) -> None:
        del record


_observer: ContextVar[ConversionObserver] = ContextVar(
    "arraybridge conversion observer", default=IgnoreConversions()
)

# A conversion kind is identified by every field of the record except its measurements.
_MEASUREMENTS = frozenset({"nbytes", "seconds"})
_IDENTITY_FIELDS = tuple(
    each.name for each in fields(ConversionRecord) if each.name not in _MEASUREMENTS
)
ConversionKey = tuple[str | tuple[str, ...], ...]
JsonRow = dict[str, str | int | float | list[str]]


def _key_of(record: ConversionRecord) -> ConversionKey:
    return tuple(getattr(record, name) for name in _IDENTITY_FIELDS)


@dataclass(slots=True)
class ConversionTotal:
    """How many conversions of one kind happened, and their bytes and seconds."""

    count: int = 0
    nbytes: int = 0
    seconds: float = 0.0


@dataclass(slots=True)
class ConversionCounters(ConversionObserver):
    """Per-process totals keyed by what identifies a conversion.

    The key is every field of ``ConversionRecord`` but ``nbytes`` and
    ``seconds``: the libraries and devices, the mechanism and the declined
    mechanisms. Counters from worker processes are plain data: pickle them or
    send ``to_json()`` back, then ``merge`` them into the parent's.
    """

    totals: dict[ConversionKey, ConversionTotal] = field(default_factory=dict)

    def record(self, record: ConversionRecord) -> None:
        self._add(_key_of(record), 1, record.nbytes, record.seconds)

    def _add(self, key: ConversionKey, count: int, nbytes: int, seconds: float) -> None:
        total = self.totals.setdefault(key, ConversionTotal())
        total.count += count
        total.nbytes += nbytes
        total.seconds += seconds

    def merge(self, other: ConversionCounters) -> ConversionCounters:
        """Add another process's totals into this one."""

        for key, total in other.totals.items():
            self._add(key, total.count, total.nbytes, total.seconds)
        return self

    def _largest_first(self) -> list[tuple[ConversionKey, ConversionTotal]]:
        return sorted(self.totals.items(), key=lambda item: (-item[1].nbytes, item[0]))

    def rows(self) -> list[JsonRow]:
        """One JSON-ready row per kind of conversion, largest byte total first."""

        return [
            {
                **{
                    name: list(part) if isinstance(part, tuple) else part
                    for name, part in zip(_IDENTITY_FIELDS, key, strict=True)
                },
                "count": total.count,
                "nbytes": total.nbytes,
                "seconds": total.seconds,
            }
            for key, total in self._largest_first()
        ]

    to_json = rows

    @classmethod
    def from_json(cls, rows: list[JsonRow]) -> ConversionCounters:
        counters = cls()
        for row in rows:
            key = tuple(
                tuple(map(str, part)) if isinstance(part, list) else str(part)
                for part in (row[name] for name in _IDENTITY_FIELDS)
            )
            counters._add(
                key, int(str(row["count"])), int(str(row["nbytes"])), float(str(row["seconds"]))
            )
        return counters

    def format_table(self) -> str:
        """The totals as aligned text, one line per kind of conversion."""

        lines = [
            f"{'source':<28} {'target':<28} {'mechanism':<20} "
            f"{'count':>7} {'MB':>10} {'seconds':>9}  declined"
        ]
        for key, total in self._largest_first():
            row = dict(zip(_IDENTITY_FIELDS, key, strict=True))
            declined = row["declined_mechanisms"]
            assert isinstance(declined, tuple)
            lines.append(
                f"{row['source_library']} {row['source_device']}".ljust(29)
                + f"{row['target_library']} {row['target_device']}".ljust(29)
                + f"{row['mechanism']!s:<20} "
                + f"{total.count:>7} {total.nbytes / 1e6:>10.1f} {total.seconds:>9.3f}  "
                + ", ".join(declined)
            )
        return "\n".join(lines)
