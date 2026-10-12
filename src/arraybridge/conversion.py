"""How an array gets from one library and device to another, and what was done.

``ConversionRoute`` leaves are tried in the order they are declared in this
module. ``applies`` reads only the two library declarations and the two device
descriptions, so the same test answers "which mechanism will be used" on a
machine without the device and selects the route when the conversion runs.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, ClassVar, TypeVar

from metaclass_registry import AutoRegisterMeta

from arraybridge.devices import Device

if TYPE_CHECKING:
    from arraybridge.libraries import ArrayLibrary

logger = logging.getLogger(__name__)

S = TypeVar("S")
T = TypeVar("T")


class RouteDeclinedError(Exception):
    """A route that applied by declaration could not carry this particular array."""


class ConversionRoute(ABC, metaclass=AutoRegisterMeta):
    """One mechanism for moving an array between libraries and devices."""

    __registry_key__ = "mechanism"
    __registry__: ClassVar[Mapping[str, type[ConversionRoute]]] = {}
    mechanism: ClassVar[str | None] = None

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
        """The route a conversion of this pair will try first."""

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
            if observer.wants_records and result is not data:
                observer.record(
                    ConversionRecord(
                        source_library=source.name,
                        source_device=source_device.label,
                        target_library=target.name,
                        target_device=target_device.label,
                        mechanism=str(route.mechanism),
                        nbytes=target.nbytes_of(result),
                        seconds=time.perf_counter() - started,
                        declined_mechanisms=tuple(declined),
                    )
                )
            return result
        raise LookupError(f"No conversion route from {source} to {target}")


class DeviceMove(ConversionRoute):
    """Same library: the array stays native and only changes device."""

    mechanism = "device move"

    @classmethod
    def applies(cls, source, source_device, target, target_device) -> bool:
        return source is target

    @classmethod
    def convert(cls, data, source, target, target_device):
        return target.move_to_device(data, target_device)


class DLPackExchange(ConversionRoute):
    """Both libraries speak DLPack: the target wraps the source's buffer."""

    mechanism = "dlpack"

    @classmethod
    def applies(cls, source, source_device, target, target_device) -> bool:
        return target.imports_dlpack and source.exports_dlpack_from(source_device)

    @classmethod
    def convert(cls, data, source, target, target_device):
        try:
            payload = source.export_dlpack(data)
            if payload is None:
                raise RouteDeclinedError("the source array has no DLPack export")
            with target.device_scope(target_device):
                result = target.from_dlpack(payload)
            if result is NotImplemented:
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


class HostCopy(ConversionRoute):
    """Download to a NumPy array in main memory, then upload to the target."""

    mechanism = "host copy"

    @classmethod
    def applies(cls, source, source_device, target, target_device) -> bool:
        return True

    @classmethod
    def convert(cls, data, source, target, target_device):
        return target.unrecorded_from_numpy(source.unrecorded_to_numpy(data), target_device)


@dataclass(frozen=True, slots=True)
class ConversionRecord:
    """One conversion that produced a new array. Plain fields: pickles and dumps to JSON."""

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


class ConversionObserver(ABC):
    """Receives one ``ConversionRecord`` per conversion while it is installed.

    ``ConversionObserver.current`` is the one hook. Its default ignores
    conversions and asks for no records, so nothing is timed or sized.
    """

    current: ClassVar[ConversionObserver]
    wants_records: ClassVar[bool] = True

    @abstractmethod
    def record(self, record: ConversionRecord) -> None:
        """Take one finished conversion."""

    @contextmanager
    def installed(self) -> Iterator[ConversionObserver]:
        """Make this the process's observer for the duration of a block."""

        previous = ConversionObserver.current
        ConversionObserver.current = self
        try:
            yield self
        finally:
            ConversionObserver.current = previous

    def install(self) -> ConversionObserver:
        """Make this the process's observer and return the one it replaced."""

        previous = ConversionObserver.current
        ConversionObserver.current = self
        return previous


class IgnoreConversions(ConversionObserver):
    """The default observer: conversions are neither measured nor kept."""

    wants_records = False

    def record(self, record: ConversionRecord) -> None:
        del record


ConversionObserver.current = IgnoreConversions()

ConversionKey = tuple[str, str, str, str, str]


@dataclass(slots=True)
class ConversionTotal:
    """How many conversions of one kind happened, and their bytes and seconds."""

    count: int = 0
    nbytes: int = 0
    seconds: float = 0.0


@dataclass(slots=True)
class ConversionCounters(ConversionObserver):
    """Per-process totals keyed by source, target and mechanism.

    Counters from worker processes are plain data: pickle them or send
    ``to_json()`` back, then ``merge`` them into the parent's.
    """

    totals: dict[ConversionKey, ConversionTotal] = field(default_factory=dict)

    def record(self, record: ConversionRecord) -> None:
        self._add(
            (
                record.source_library,
                record.source_device,
                record.target_library,
                record.target_device,
                record.mechanism,
            ),
            1,
            record.nbytes,
            record.seconds,
        )

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

    def rows(self) -> list[dict[str, str | int | float]]:
        """One JSON-ready row per kind of conversion, largest byte total first."""

        return [
            {
                "source_library": key[0],
                "source_device": key[1],
                "target_library": key[2],
                "target_device": key[3],
                "mechanism": key[4],
                "count": total.count,
                "nbytes": total.nbytes,
                "seconds": total.seconds,
            }
            for key, total in sorted(
                self.totals.items(), key=lambda item: (-item[1].nbytes, item[0])
            )
        ]

    to_json = rows

    @classmethod
    def from_json(cls, rows: list[dict[str, str | int | float]]) -> ConversionCounters:
        counters = cls()
        for row in rows:
            counters._add(
                (
                    str(row["source_library"]),
                    str(row["source_device"]),
                    str(row["target_library"]),
                    str(row["target_device"]),
                    str(row["mechanism"]),
                ),
                int(row["count"]),
                int(row["nbytes"]),
                float(row["seconds"]),
            )
        return counters

    def format_table(self) -> str:
        """The totals as aligned text, one line per kind of conversion."""

        lines = [
            f"{'source':<28} {'target':<28} {'mechanism':<12} "
            f"{'count':>7} {'MB':>10} {'seconds':>9}"
        ]
        for key, total in sorted(self.totals.items(), key=lambda item: (-item[1].nbytes, item[0])):
            lines.append(
                f"{key[0] + ' ' + key[1]:<28} {key[2] + ' ' + key[3]:<28} {key[4]:<12} "
                f"{total.count:>7} {total.nbytes / 1e6:>10.1f} {total.seconds:>9.3f}"
            )
        return "\n".join(lines)
