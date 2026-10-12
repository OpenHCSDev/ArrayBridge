"""One class per supported array library.

``ArrayLibrary`` is the family. A leaf states the library's identity and owns
everything that library does differently: devices, moving data, DLPack,
out-of-memory recognition, child-process environment and native array
operations. What several libraries share is a parent they inherit from
(``HostDevices``, ``AcceleratorDevices``, ``CudaLibrary``, ``DLPackImport``,
``DLPackProtocolExport``, ``NvidiaWheelLibraries``, ``InPlaceAssignment``,
``ArrayMath``, ``NumpyArrayApi``, ``MutableArrays``). The base class holds no
default that assumes a particular library's API: a library inherits an
implementation only from a parent it declares.

Adding a library is writing one class here or anywhere else: registration,
lookup, the ``MemoryType`` member, the decorator and conversion follow from it.

Typing. ``Array`` is the nominal type of "an array of any supported library";
each library's ``array_class`` is "an array of that library". Third-party
array classes are admitted with ``ABC.register`` when the library first meets
one, so no optional library is imported to define them. Inside a leaf the
native class is the type parameter, named through a ``TYPE_CHECKING`` import
because the optional package cannot be imported at runtime here.

The checker reads each optional library from the declarations in the
repository's ``stubs`` directory, which list what this module calls and take
precedence over an installed copy, so the result of checking does not depend
on what is installed. A leaf states that its ``module`` is that library, so
its calls are checked against those declarations. A role shared by libraries
with the same function names (``NumpyArrayApi``) names its module by a
``Protocol`` of the functions it calls, and each leaf's module is checked
against that Protocol.

Importing this module imports no array library, NumPy included.
"""

from __future__ import annotations

import functools
import gc
import importlib
import importlib.util
import logging
import operator
import os
import sys
import threading
from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, NotImplementedType
from typing import TYPE_CHECKING, ClassVar, Generic, ParamSpec, Protocol, TypeVar

from metaclass_registry import AutoRegisterMeta

from arraybridge.array_payload import ArrayPayload
from arraybridge.arrays import Array
from arraybridge.conversion import ConversionRoute, StaysInPlace
from arraybridge.devices import HOST, AcceleratorDevice, Device, HostDevice, NamedDevice

if TYPE_CHECKING:
    # Optional packages, as declared in ``stubs``: never imported at runtime here.
    import cupy
    import jax
    import numpy as np
    import pyclesperanto
    import tensorflow
    import torch
    from typing_extensions import Self

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

# A dtype as callers spell it: a NumPy dtype, a Python type, a name, or a library's own dtype.
DType = object
Axis = int | tuple[int, ...]

_SCALING_RANGES: dict[str, float | tuple[float, float]] = {
    "uint8": 255.0,
    "uint16": 65535.0,
    "uint32": 4294967295.0,
    "int16": (65535.0, 32768.0),
    "int32": (4294967295.0, 2147483648.0),
}


NativeT = TypeVar("NativeT")
TargetT = TypeVar("TargetT")
DTypeT = TypeVar("DTypeT")


class Intensities(Protocol):
    """The arithmetic that rescaling an array's intensities uses."""

    def __sub__(self, other: Self | float, /) -> Self: ...

    def __mul__(self, other: float, /) -> Self: ...

    def __truediv__(self, other: Self | float, /) -> Self: ...


IntensitiesT = TypeVar("IntensitiesT", bound=Intensities)


class NumpyArray(Protocol):
    """An array with NumPy's own methods, as ``NumpyArrayApi`` calls them."""

    def astype(self, dtype: DType, /, *, copy: bool = ...) -> Self: ...

    def reshape(self, shape: tuple[int, ...], /) -> Self: ...


class MutableNumpyArray(NumpyArray, Intensities, Protocol):
    """A NumPy-style array that ``MutableArrays`` rescales and divides in place."""

    @property
    def dtype(self) -> object: ...

    def __getitem__(self, index: int, /) -> Self: ...

    def __setitem__(self, index: int, value: Self, /) -> None: ...

    def __itruediv__(self, other: Self | float, /) -> Self: ...

    def __le__(self, other: Self, /) -> object: ...


NumpyNativeT = TypeVar("NumpyNativeT", bound=NumpyArray)
MutableNativeT = TypeVar("MutableNativeT", bound=MutableNumpyArray)


class NumpyFunctions(Protocol[NumpyNativeT]):
    """A module with NumPy's function names, as ``NumpyArrayApi`` calls them."""

    @property
    def ndarray(self) -> type: ...

    def max(self, a: NumpyNativeT, /, *, axis: Axis | None = ...) -> NumpyNativeT: ...

    def min(self, a: NumpyNativeT, /, *, axis: Axis | None = ...) -> NumpyNativeT: ...

    def mean(self, a: NumpyNativeT, /, *, axis: Axis | None = ...) -> NumpyNativeT: ...

    def sum(self, a: NumpyNativeT, /, *, axis: Axis | None = ...) -> NumpyNativeT: ...

    def linspace(
        self, start: float, stop: float, num: int, /, *, endpoint: bool = ...
    ) -> NumpyNativeT: ...

    def ones(self, shape: int | tuple[int, ...], /, *, dtype: DType = ...) -> NumpyNativeT: ...

    def outer(self, a: NumpyNativeT, b: NumpyNativeT, /) -> NumpyNativeT: ...

    def floor(self, x: float, /) -> NumpyNativeT: ...

    def stack(self, arrays: Sequence[NumpyNativeT], /, *, axis: int = ...) -> NumpyNativeT: ...

    def logical_and(self, x1: NumpyNativeT, x2: NumpyNativeT, /) -> NumpyNativeT: ...

    def broadcast_to(self, array: NumpyNativeT, shape: tuple[int, ...], /) -> NumpyNativeT: ...


class MutableNumpyFunctions(NumpyFunctions[MutableNativeT], Protocol[MutableNativeT]):
    """The further NumPy function names that ``MutableArrays`` calls."""

    @property
    def floating(self) -> object: ...

    def issubdtype(self, arg1: object, arg2: object, /) -> bool: ...

    def array(
        self, a: MutableNativeT, /, *, dtype: DType = ..., copy: bool = ...
    ) -> MutableNativeT: ...

    def clip(self, a: MutableNativeT, a_min: float, a_max: float, /) -> MutableNativeT: ...


@dataclass(frozen=True, slots=True)
class DLPackPayload:
    """One exported capsule plus the array it was exported from."""

    source: object
    capsule: object


def _dtype_name(dtype: DType) -> str:
    declared_name = getattr(dtype, "name", None)
    if declared_name is not None:
        return str(declared_name)
    return str(getattr(dtype, "__name__", str(dtype).rsplit(".", maxsplit=1)[-1]))


def _scaled_values(
    result: IntensitiesT,
    result_min: IntensitiesT,
    result_max: IntensitiesT,
    target_dtype: DType,
) -> IntensitiesT:
    normalized = (result - result_min) / (result_max - result_min)
    range_info = _SCALING_RANGES.get(_dtype_name(target_dtype))
    if range_info is None:
        return normalized
    if isinstance(range_info, tuple):
        scale, offset = range_info
        return normalized * scale - offset
    return normalized * range_info


def _clamp_bounds(target_dtype: DType) -> tuple[float, float] | None:
    range_info = _SCALING_RANGES.get(_dtype_name(target_dtype))
    if range_info is None:
        return None
    if isinstance(range_info, tuple):
        scale, offset = range_info
        return -offset, scale - offset - 128
    return 0, range_info


def _mapped_dtype(target_dtype: DType, module: object, dtype_class: type[DTypeT]) -> DTypeT:
    """The ``dtype_class`` object that ``module`` keeps under the NumPy name of ``target_dtype``."""

    import numpy as np

    try:
        dtype_name = np.dtype(target_dtype).name
    except TypeError as error:
        raise TypeError(f"Unsupported target dtype {target_dtype!r}") from error
    module_name = getattr(module, "__name__", module)
    mapped: object = getattr(module, dtype_name, None)
    if mapped is None:
        raise TypeError(f"{module_name} does not expose dtype {dtype_name}")
    if not isinstance(mapped, dtype_class):
        raise TypeError(f"{module_name}.{dtype_name} is not a {dtype_class.__name__}")
    return mapped


def _has_dlpack_protocol(data: object) -> bool:
    return callable(getattr(data, "__dlpack__", None)) and callable(
        getattr(data, "__dlpack_device__", None)
    )


class ArrayLibrary(Generic[NativeT], metaclass=AutoRegisterMeta):
    """One array library: its identity and everything it does its own way."""

    __registry_key__ = "name"
    __registry__: ClassVar[Mapping[str, type[ArrayLibrary[Array]]]] = {}
    _instances: ClassVar[dict[str, ArrayLibrary[Array]]] = {}

    # ----- identity, stated by each leaf -----
    name: ClassVar[str]
    display_name: ClassVar[str]
    module_aliases: ClassVar[tuple[str, ...]] = ()
    import_environment: ClassVar[tuple[tuple[str, str], ...]] = ()
    oom_messages: ClassVar[tuple[str, ...]] = ()
    oom_exception_paths: ClassVar[tuple[tuple[str, ...], ...]] = ()
    array_class: ClassVar[ABCMeta]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        declared = cls.__dict__.get("name")
        if declared is None:
            return
        if "import_name" not in cls.__dict__:
            cls.import_name = declared
        if "display_name" not in cls.__dict__:
            cls.display_name = declared
        cls.array_class = ABCMeta(f"{cls.__name__}.Array", (Array,), {})

    import_name: ClassVar[str]

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"

    def __str__(self) -> str:
        return str(self.name)

    def __reduce__(self) -> tuple[Callable[[str], ArrayLibrary[Array]], tuple[str]]:
        return ArrayLibrary.named, (str(self.name),)

    # ----- lookup -----

    @classmethod
    def all(cls) -> tuple[ArrayLibrary[Array], ...]:
        """Every registered library, in declaration order."""

        return tuple(cls.named(name) for name in ArrayLibrary.__registry__)

    @classmethod
    def named(cls, name: str) -> ArrayLibrary[Array]:
        """The library registered under ``name`` (``"numpy"``, ``"torch"``, ...)."""

        try:
            return ArrayLibrary._instances[name]
        except KeyError:
            pass
        try:
            leaf = ArrayLibrary.__registry__[name]
        except KeyError:
            raise ValueError(
                f"'{name}' is not a registered array library. "
                f"Available: {sorted(ArrayLibrary.__registry__)}"
            ) from None
        return ArrayLibrary._instances.setdefault(name, leaf())

    @classmethod
    def for_module_name(cls, module_name: str) -> ArrayLibrary[Array]:
        """The library owning a top-level module name or one of its aliases."""

        top_level = module_name.split(".")[0]
        for library in cls.all():
            if top_level in library.recognized_module_names:
                return library
        raise ValueError(f"No array library owns module '{module_name}'")

    @classmethod
    def of(cls, value: object) -> ArrayLibrary[Array]:
        """Which library made ``value``. Imports nothing."""

        libraries = cls.all()
        for library in libraries:
            if isinstance(value, library.array_class):
                return library
        # First sight of this class: the library is the owner of the nearest
        # ancestor's module, so a subclass defined elsewhere is still recognised.
        for ancestor in type(value).__mro__:
            top_level = ancestor.__module__.split(".")[0]
            for library in libraries:
                if top_level in library.recognized_module_names:
                    library.array_class.register(type(value))
                    return library
        raise ValueError(
            f"Unknown memory type for {type(value)} (module: {type(value).__module__})"
        )

    # ----- the module -----

    @property
    def recognized_module_names(self) -> frozenset[str]:
        """Top-level module names whose objects belong to this library."""

        return frozenset((self.import_name, *self.module_aliases))

    def is_installed(self) -> bool:
        """Whether the package is present, without importing it."""

        if self.loaded_module() is not None:
            return True
        try:
            return importlib.util.find_spec(self.import_name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    def prepare_import(self) -> None:
        """Set this library's import-time environment defaults before it is imported."""

        missing_names = tuple(
            name for name, _value in self.import_environment if name not in os.environ
        )
        if missing_names and self.loaded_module() is not None:
            logger.warning(
                "%s was loaded before its ArrayBridge import defaults were set; "
                "the current process cannot guarantee those import-time settings: %s",
                self.display_name,
                ", ".join(missing_names),
            )
        for name, value in self.import_environment:
            os.environ.setdefault(name, value)

    def loaded_module(self) -> ModuleType | None:
        """The library's module if something already imported it."""

        return sys.modules.get(self.import_name)

    def import_module(self) -> ModuleType:
        """Import the library's module."""

        self.prepare_import()
        return importlib.import_module(self.import_name)

    def import_if_installed(self) -> ModuleType | None:
        """Import the library only when its package is present."""

        loaded = self.loaded_module()
        if loaded is not None:
            self.prepare_import()
            return loaded
        if not self.is_installed():
            return None
        return self.import_module()

    @property
    def module(self) -> ModuleType:
        """The library's module, imported on first use."""

        return self.loaded_module() or self.import_module()

    @property
    def array_module(self) -> object:
        """The module holding the library's array functions."""

        return self.module

    def extend_child_environment(self, environment: dict[str, str]) -> dict[str, str]:
        """Add what a child process needs to import this library."""

        for name, value in self.import_environment:
            environment.setdefault(name, value)
        return environment

    @classmethod
    def child_environment(cls, environment: Mapping[str, str] | None = None) -> dict[str, str]:
        """An environment in which a child process can import every library."""

        prepared = dict(os.environ if environment is None else environment)
        for library in cls.all():
            prepared = library.extend_child_environment(prepared)
        return prepared

    # ----- devices -----

    def host_devices(self) -> tuple[HostDevice, ...]:
        """The host device, if this library keeps arrays in main memory."""

        return ()

    def accelerator_devices(self) -> tuple[AcceleratorDevice, ...]:
        """The accelerator devices this library can use on this machine now."""

        return ()

    def available_devices(self) -> tuple[Device, ...]:
        return (*self.host_devices(), *self.accelerator_devices())

    @abstractmethod
    def device_of(self, data: NativeT) -> Device:
        """The device holding ``data``."""

    def require_device(self, device: Device) -> None:
        """Raise unless this library has ``device`` on this machine."""

        available = self.available_devices()
        if device not in available:
            raise ValueError(
                f"{self.display_name} device {device.label} is unavailable; "
                f"available devices are {tuple(each.label for each in available)}"
            )

    def device_scope(self, device: Device) -> AbstractContextManager[object]:
        """A block in which new arrays of this library are created on ``device``."""

        self.require_device(device)
        return self._scope(device)

    def activate_device(self, device: Device) -> None:
        """Select ``device`` for the whole process where the library has such a setting."""

        self.require_device(device)
        self._activate(device)

    def move_to_device(self, data: NativeT, device: Device) -> NativeT:
        """Return ``data`` on ``device``; the same object when it is already there."""

        self.require_device(device)
        if StaysInPlace.applies(self, self.device_of(data), self, device):
            return data
        with self._scope(device):
            return self._move(data, device)

    def _array_scope(self, reference: NativeT) -> AbstractContextManager[object]:
        """The scope of the accelerator holding ``reference``; host data needs none."""

        device = self.device_of(reference)
        if isinstance(device, AcceleratorDevice):
            return self.device_scope(device)
        return nullcontext()

    # Each device parent answers for its own kind of device and passes the rest on.

    def _scope(self, device: Device) -> AbstractContextManager[object]:
        raise ValueError(f"{self.display_name} has no {device.label} device")

    def _activate(self, device: Device) -> None:
        raise ValueError(f"{self.display_name} has no {device.label} device")

    def _move(self, data: NativeT, device: Device) -> NativeT:
        raise ValueError(f"{self.display_name} has no {device.label} device")

    def _place(self, data: np.ndarray, device: Device) -> NativeT:
        raise ValueError(f"{self.display_name} has no {device.label} device")

    # ----- running a function of this library -----

    def execution_wrapper(self, func: Callable[P, R], oom_recovery: bool) -> Callable[P, R]:
        """Wrap a function written for this library with what its devices need."""

        del oom_recovery
        return func

    def is_oom_error(self, error: BaseException) -> bool:
        """Whether ``error`` is this library running out of memory. Imports nothing."""

        module = self.loaded_module()
        if module is not None:
            for path in self.oom_exception_paths:
                candidate: object = module
                for attribute in path:
                    candidate = getattr(candidate, attribute, None)
                if (
                    isinstance(candidate, type)
                    and issubclass(candidate, BaseException)
                    and isinstance(error, candidate)
                ):
                    return True
        message = str(error).lower()
        return any(pattern in message for pattern in self.oom_messages)

    def cleanup_loaded(self, device: AcceleratorDevice | None = None) -> None:
        """Release cached device memory if the library is loaded; every device by default."""

        del device

    @classmethod
    def cleanup_all_loaded(cls, device: AcceleratorDevice | None = None) -> None:
        """Release cached device memory of every loaded library; failures are logged."""

        for library in cls.all():
            try:
                library.cleanup_loaded(device)
            except Exception as error:
                logger.warning("Failed to cleanup %s GPU memory: %s", library.display_name, error)

    def run_with_oom_recovery(
        self,
        func: Callable[[], R],
        max_retries: int = 2,
        device: AcceleratorDevice | None = None,
    ) -> R:
        """Call ``func``; on this library's out-of-memory error release caches and retry."""

        attempt = 0
        while True:
            try:
                return func()
            except Exception as error:
                if not self.is_oom_error(error) or attempt == max_retries:
                    raise
            attempt += 1
            try:
                self.cleanup_loaded(device)
            except Exception as error:
                logger.warning("Failed to clear cache for %s: %s", self.name, error)
            gc.collect()

    # ----- DLPack -----

    @property
    def imports_dlpack(self) -> bool:
        """Whether this library can wrap another library's buffer through DLPack."""

        return isinstance(self, DLPackImport)

    def imports_dlpack_from(self, device: Device) -> bool:
        """Whether this library can wrap a buffer that lives on ``device``."""

        del device
        return False

    def exports_dlpack_from(self, device: Device) -> bool:
        """Whether arrays of this library on ``device`` are offered through DLPack."""

        del device
        return False

    def exports_dlpack_data(self, data: NativeT) -> bool:
        """Whether this particular array can be exported through DLPack."""

        del data
        return False

    def export_dlpack(self, data: NativeT) -> DLPackPayload | None:
        """Export one array through DLPack, or ``None`` when this array cannot be."""

        del data
        return None

    # ----- conversion -----

    def unrecorded_to_numpy(self, data: NativeT) -> np.ndarray:
        """Download to NumPy without writing a conversion record (for routes)."""

        return self._to_host(data)

    def unrecorded_from_numpy(self, data: np.ndarray, device: Device) -> NativeT:
        """Upload from NumPy without writing a conversion record (for routes)."""

        with self.device_scope(device):
            return self._place(data, device)

    def shares_numpy_buffer_on(self, device: Device) -> bool:
        """Whether an array of this library on ``device`` and a NumPy array can be one buffer."""

        del device
        return False

    def host_address_of(self, data: NativeT) -> int | None:
        """Where ``data`` begins in main memory; ``None`` when no bytes of it are there."""

        del data
        return None

    def to_numpy(self, data: NativeT) -> np.ndarray:
        """``data`` as a NumPy array in main memory: the conversion to NumPy on the host."""

        return ConversionRoute.run(data, self, NumpyLibrary.registered(), HOST)

    def from_numpy(self, data: np.ndarray, device: Device) -> NativeT:
        """An array of this library on ``device``: the conversion from NumPy on the host."""

        return ConversionRoute.run(data, NumpyLibrary.registered(), self, device)

    def convert_to(self, data: NativeT, target: ArrayLibrary[TargetT], device: Device) -> TargetT:
        """Convert ``data`` to ``target`` on ``device`` by the first route that carries it."""

        return ConversionRoute.run(data, self, target, device)

    def nbytes_of(self, data: NativeT) -> int:
        """Size of the array's elements in bytes."""

        import numpy as np

        return int(np.prod(tuple(getattr(data, "shape")), dtype=np.int64)) * int(
            np.dtype(getattr(data, "dtype")).itemsize
        )

    # ----- array operations, each on the device of its input -----

    @abstractmethod
    def accepts(self, value: object) -> bool:
        """Whether ``value`` is a native array of this library."""

    def stack_arrays(self, arrays: Sequence[NativeT], device: Device) -> NativeT:
        """Stack arrays along a new leading axis on ``device``."""

        with self.device_scope(device):
            return self._stack(arrays)

    def normalize_planes(
        self, data: NativeT, dtype: DType, scales: Sequence[float | None]
    ) -> NativeT:
        """Cast every plane to ``dtype`` and divide it by its scale."""

        if len(scales) != int(getattr(data, "shape")[0]):
            raise ValueError("Normalization scales must match the leading plane axis.")
        with self._array_scope(data):
            return self._normalize_planes(data, dtype, scales)

    def scale_dtype(self, data: NativeT, target_dtype: DType) -> NativeT:
        """Convert to ``target_dtype``, rescaling float intensities into an integer range."""

        if self.import_if_installed() is None or not hasattr(data, "dtype"):
            return data
        with self._array_scope(data):
            return self._scale_dtype(data, target_dtype)

    def dtype_name(self, dtype: DType) -> str:
        """The dtype's name as NumPy spells it (``"uint16"``, ``"float32"``)."""

        import numpy as np

        return str(np.dtype(dtype).name)

    def astype(self, data: NativeT, dtype: DType) -> NativeT:
        with self._array_scope(data):
            return self._cast(data, dtype)

    def reshape(self, data: NativeT, shape: Sequence[int]) -> NativeT:
        with self._array_scope(data):
            return self._reshape(data, tuple(shape))

    def broadcast_to(self, data: NativeT, shape: Sequence[int]) -> NativeT:
        with self._array_scope(data):
            return self._broadcast_to(data, tuple(shape))

    def ones_like(
        self,
        reference: NativeT,
        *,
        shape: Sequence[int] | None = None,
        dtype: DType = bool,
    ) -> NativeT:
        """Allocate ones on the device of ``reference``."""

        with self._array_scope(reference):
            return self._ones_like(
                reference,
                tuple(getattr(reference, "shape") if shape is None else shape),
                dtype,
            )

    def logical_and(self, left: NativeT, right: NativeT) -> NativeT:
        with self._array_scope(left):
            return self._logical_and(left, right)

    @abstractmethod
    def max(self, data: NativeT, axis: Axis) -> NativeT:
        """The largest value along ``axis``."""

    @abstractmethod
    def min(self, data: NativeT, axis: Axis) -> NativeT:
        """The smallest value along ``axis``."""

    @abstractmethod
    def mean(self, data: NativeT, axis: Axis) -> NativeT:
        """The mean along ``axis``."""

    @abstractmethod
    def sum(self, data: NativeT, axis: Axis) -> NativeT:
        """The sum along ``axis``."""

    def prepend_axis(self, data: NativeT) -> NativeT:
        """Add a leading axis of length one."""

        return self._reshape(data, (1, *getattr(data, "shape")))

    @abstractmethod
    def assign(self, data: NativeT, index: slice, values: NativeT) -> NativeT:
        """Return ``data`` with ``data[index]`` replaced by ``values``."""

    # ----- what a leaf supplies; these run without choosing a device -----

    @abstractmethod
    def _to_host(self, data: NativeT) -> np.ndarray:
        """Download to NumPy."""

    @abstractmethod
    def _scale_dtype(self, result: NativeT, target_dtype: DType) -> NativeT:
        """The library's intensity-preserving dtype conversion."""

    @abstractmethod
    def _stack(self, values: Sequence[NativeT]) -> NativeT:
        """Stack along a new leading axis."""

    @abstractmethod
    def _cast(self, data: NativeT, dtype: DType) -> NativeT:
        """The values as ``dtype``, without rescaling."""

    @abstractmethod
    def _logical_and(self, left: NativeT, right: NativeT) -> NativeT:
        """Element-wise ``left and right``."""

    @abstractmethod
    def _reshape(self, data: NativeT, shape: tuple[int, ...]) -> NativeT:
        """The same elements in ``shape``."""

    @abstractmethod
    def _broadcast_to(self, data: NativeT, shape: tuple[int, ...]) -> NativeT:
        """``data`` repeated out to ``shape``."""

    @abstractmethod
    def _ones_like(self, reference: NativeT, shape: tuple[int, ...], dtype: DType) -> NativeT:
        """Ones of ``shape`` and ``dtype`` on the device of ``reference``."""

    def _normalize_planes(
        self, data: NativeT, dtype: DType, scales: Sequence[float | None]
    ) -> NativeT:
        return self._stack(
            tuple(
                (
                    self._cast(plane, dtype)
                    if scale is None
                    else operator.truediv(self._cast(plane, dtype), float(scale))
                )
                for plane, scale in (
                    (getattr(data, "__getitem__")(index), scale)
                    for index, scale in enumerate(scales)
                )
            )
        )


# ---------------------------------------------------------------------------
# Roles several libraries share. A library inherits each role it plays.
# ---------------------------------------------------------------------------


class HostDevices(ArrayLibrary[NativeT]):
    """Libraries that keep arrays in main memory (NumPy, torch, TensorFlow, JAX)."""

    def host_devices(self) -> tuple[HostDevice, ...]:
        return (HOST,)

    def _scope(self, device: Device) -> AbstractContextManager[object]:
        if isinstance(device, HostDevice):
            return self._host_scope()
        return super()._scope(device)

    def _activate(self, device: Device) -> None:
        if not isinstance(device, HostDevice):
            super()._activate(device)

    def _move(self, data: NativeT, device: Device) -> NativeT:
        if isinstance(device, HostDevice):
            return self._move_to_host(data)
        return super()._move(data, device)

    def _place(self, data: np.ndarray, device: Device) -> NativeT:
        if isinstance(device, HostDevice):
            return self._wrap_host_array(data)
        return super()._place(data, device)

    def _host_scope(self) -> AbstractContextManager[object]:
        return nullcontext()

    @abstractmethod
    def _wrap_host_array(self, data: np.ndarray) -> NativeT:
        """A native array in main memory holding the NumPy array's values."""

    @abstractmethod
    def _move_to_host(self, data: NativeT) -> NativeT:
        """The same native array in main memory."""


class NumpyBufferSharing(HostDevices[NativeT]):
    """Libraries whose main-memory arrays can be the very buffer of a NumPy array.

    Declared after measuring it: the library's exchange with NumPy on the host
    returns an array that begins at the address of the one it was given, for
    some arrays at least. ``host_address_of`` is what decides it for one array.
    """

    def shares_numpy_buffer_on(self, device: Device) -> bool:
        return isinstance(device, HostDevice)

    def host_address_of(self, data: NativeT) -> int | None:
        if not isinstance(self.device_of(data), HostDevice) or self.nbytes_of(data) == 0:
            return None
        return self._first_element_address(data)

    @abstractmethod
    def _first_element_address(self, data: NativeT) -> int:
        """Address of the first element of an array held in main memory."""


class AcceleratorDevices(ArrayLibrary[NativeT]):
    """Libraries that can keep arrays on an accelerator (every one but NumPy)."""

    def accelerator_devices(self) -> tuple[AcceleratorDevice, ...]:
        if self.import_if_installed() is None:
            return ()
        return tuple(AcceleratorDevice(index) for index in range(self._accelerator_count()))

    def _scope(self, device: Device) -> AbstractContextManager[object]:
        if isinstance(device, AcceleratorDevice):
            return self._accelerator_scope(device)
        return super()._scope(device)

    def _activate(self, device: Device) -> None:
        if isinstance(device, AcceleratorDevice):
            self._activate_accelerator(device)
        else:
            super()._activate(device)

    def _move(self, data: NativeT, device: Device) -> NativeT:
        if isinstance(device, AcceleratorDevice):
            return self._move_to_accelerator(data, device)
        return super()._move(data, device)

    def _place(self, data: np.ndarray, device: Device) -> NativeT:
        if isinstance(device, AcceleratorDevice):
            return self._upload(data, device)
        return super()._place(data, device)

    @abstractmethod
    def _accelerator_count(self) -> int:
        """How many accelerators the imported library sees."""

    @abstractmethod
    def _accelerator_scope(self, device: AcceleratorDevice) -> AbstractContextManager[object]:
        """A block in which new arrays are created on ``device``."""

    def _activate_accelerator(self, device: AcceleratorDevice) -> None:
        """Select ``device`` for the process; libraries without such a setting do nothing."""

        del device

    @abstractmethod
    def _move_to_accelerator(self, data: NativeT, device: AcceleratorDevice) -> NativeT:
        """Copy ``data`` to ``device``; called inside that device's scope."""

    @abstractmethod
    def _upload(self, data: np.ndarray, device: AcceleratorDevice) -> NativeT:
        """Create a native array on ``device``; called inside that device's scope."""

    def _argument_device(self, arguments: Sequence[object]) -> Device | None:
        """The device of the first array of this library among a call's arguments."""

        if self.loaded_module() is None:
            return None
        for value in arguments:
            device = self._device_if_accepted(value)
            if device is not None:
                return device
        return None

    @abstractmethod
    def _device_if_accepted(self, value: object) -> Device | None:
        """The device holding ``value``; ``None`` when it is not an array this library takes."""

    def _selected_accelerator(self) -> AcceleratorDevice | None:
        """The accelerator the library has selected for this thread, where it says."""

        return None

    def _accelerator_call_scope(
        self, device: AcceleratorDevice | None
    ) -> AbstractContextManager[object]:
        """What a call running on ``device`` is wrapped in."""

        del device
        return nullcontext()

    def execution_wrapper(self, func: Callable[P, R], oom_recovery: bool) -> Callable[P, R]:
        """Run ``func`` on the accelerator holding its arguments; a host call is untouched.

        The device is read from the arguments. A call whose arrays are in main
        memory, or that passes no array of a library that has main memory,
        runs as written: nothing of the accelerator is asked or initialised.
        """

        @functools.wraps(func)
        def on_accelerator(*args: P.args, **kwargs: P.kwargs) -> R:
            held_on = self._argument_device((*args, *kwargs.values()))
            if isinstance(held_on, AcceleratorDevice):
                device: AcceleratorDevice | None = held_on
            elif held_on is not None or self.host_devices():
                return func(*args, **kwargs)
            elif self.import_if_installed() is None or not self.accelerator_devices():
                return func(*args, **kwargs)
            else:
                # Every array of this library lives on an accelerator.
                device = self._selected_accelerator()
            scope = self._accelerator_call_scope(device)

            def execute() -> R:
                with scope:
                    return func(*args, **kwargs)

            if oom_recovery:
                return self.run_with_oom_recovery(execute, device=device)
            return execute()

        return on_accelerator


class CudaLibrary(AcceleratorDevices[NativeT]):
    """CUDA libraries with their own streams and a memory cache to release (CuPy, torch)."""

    _thread_streams: ClassVar[threading.local] = threading.local()

    @abstractmethod
    def create_stream(self) -> object:
        """A new CUDA stream on the current device."""

    @abstractmethod
    def stream_scope(self, stream: object) -> AbstractContextManager[object]:
        """A block whose work is queued on ``stream``."""

    @abstractmethod
    def current_device(self) -> AcceleratorDevice | None:
        """The accelerator the process currently has selected, if any."""

    @abstractmethod
    def _free_cached_memory(self) -> None:
        """Release the library's cached memory on the device in scope."""

    def thread_stream(self, device: AcceleratorDevice) -> object:
        """This thread's stream for ``device``, created on first use."""

        streams: dict[tuple[str, int], object] | None = getattr(
            CudaLibrary._thread_streams, "streams", None
        )
        if streams is None:
            streams = {}
            CudaLibrary._thread_streams.streams = streams
        key = (str(self.name), device.index)
        if key not in streams:
            with self.device_scope(device):
                streams[key] = self.create_stream()
            logger.debug(
                "Created %s stream for device %d in thread %s",
                self.display_name,
                device.index,
                threading.current_thread().name,
            )
        return streams[key]

    def _selected_accelerator(self) -> AcceleratorDevice | None:
        return self.current_device()

    def _accelerator_call_scope(
        self, device: AcceleratorDevice | None
    ) -> AbstractContextManager[object]:
        if device is None:
            return nullcontext()
        return self.stream_scope(self.thread_stream(device))

    def cleanup_loaded(self, device: AcceleratorDevice | None = None) -> None:
        if self.loaded_module() is None:
            return
        available = self.accelerator_devices()
        if not available:
            return
        for target in available if device is None else (device,):
            with self.device_scope(target):
                self._free_cached_memory()


class DLPackImport(ArrayLibrary[NativeT]):
    """Libraries that can wrap another library's buffer handed over through DLPack."""

    def imports_dlpack_from(self, device: Device) -> bool:
        # A library without main-memory arrays cannot wrap a main-memory buffer.
        return isinstance(self, HostDevices) or not isinstance(device, HostDevice)

    def from_dlpack(self, data: DLPackPayload | object) -> NativeT | NotImplementedType:
        """Wrap an exported buffer; ``NotImplemented`` when this capsule cannot be read."""

        payload = data if isinstance(data, DLPackPayload) else DLPackPayload(data, data)
        return self._import_dlpack(payload) if self._reads_dlpack(payload) else NotImplemented

    def _reads_dlpack(self, payload: DLPackPayload) -> bool:
        """Whether the library's importer can read this payload."""

        del payload
        return True

    @abstractmethod
    def _import_dlpack(self, payload: DLPackPayload) -> NativeT:
        """Read one payload with the library's importer."""


class DLPackProtocolExport(ArrayLibrary[NativeT]):
    """Libraries whose arrays carry the standard DLPack export methods on every device."""

    _export_methods: ClassVar[tuple[str, ...]] = ("__dlpack__", "to_dlpack", "toDlpack")

    def exports_dlpack_from(self, device: Device) -> bool:
        del device
        return True

    def exports_dlpack_data(self, data: NativeT) -> bool:
        return any(callable(getattr(data, attribute, None)) for attribute in self._export_methods)

    def export_dlpack(self, data: NativeT) -> DLPackPayload | None:
        for attribute in self._export_methods:
            exporter = getattr(data, attribute, None)
            if callable(exporter):
                return DLPackPayload(source=data, capsule=exporter())
        return None


class NvidiaWheelLibraries(ArrayLibrary[NativeT]):
    """Libraries that load CUDA from the ``nvidia-*`` wheels' library directories."""

    @staticmethod
    def nvidia_wheel_library_paths() -> tuple[str, ...]:
        """Native-library directories of the installed NVIDIA wheels."""

        try:
            spec = importlib.util.find_spec("nvidia")
        except (ImportError, ModuleNotFoundError, ValueError):
            return ()
        if spec is None or spec.submodule_search_locations is None:
            return ()
        paths: set[str] = set()
        for package_root in map(Path, spec.submodule_search_locations):
            try:
                components = tuple(package_root.iterdir())
            except OSError:
                continue
            for component in components:
                for library_directory_name in ("lib", "bin"):
                    candidate = component / library_directory_name
                    if candidate.is_dir():
                        paths.add(str(candidate))
        return tuple(sorted(paths))

    def extend_child_environment(self, environment: dict[str, str]) -> dict[str, str]:
        prepared = super().extend_child_environment(environment)
        library_paths = self.nvidia_wheel_library_paths()
        if not library_paths:
            return prepared
        search_variable = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        existing_paths = tuple(
            path for path in prepared.get(search_variable, "").split(os.pathsep) if path
        )
        prepared[search_variable] = os.pathsep.join(
            dict.fromkeys((*library_paths, *existing_paths))
        )
        return prepared


class InPlaceAssignment(ArrayLibrary[NativeT]):
    """Libraries whose arrays take ``array[index] = values``."""

    def assign(self, data: NativeT, index: slice, values: NativeT) -> NativeT:
        getattr(data, "__setitem__")(index, values)
        return data


class ArrayMath(ArrayLibrary[NativeT]):
    """Libraries that make arrays without a reference array, on their default device.

    A library whose module has no such functions (pyclesperanto) does not
    inherit this, so asking it for one fails at the attribute, not in the call.
    """

    @abstractmethod
    def linspace(self, start: float, stop: float, num: int, *, endpoint: bool) -> NativeT:
        """``num`` evenly spaced values from ``start`` to ``stop``."""

    @abstractmethod
    def ones(self, shape: int | tuple[int, ...], dtype: DType) -> NativeT:
        """Ones of ``shape`` and ``dtype``."""

    @abstractmethod
    def outer(self, left: NativeT, right: NativeT) -> NativeT:
        """The outer product of two vectors."""

    @abstractmethod
    def floor(self, value: float) -> NativeT:
        """Floor a host scalar in the library's default precision."""


class NumpyArrayApi(ArrayMath[NumpyNativeT]):
    """Libraries whose array module and arrays follow the NumPy API (NumPy, CuPy, JAX)."""

    @property
    @abstractmethod
    def array_module(self) -> NumpyFunctions[NumpyNativeT]:
        """The library's module of NumPy-named functions."""

    def accepts(self, value: object) -> bool:
        return isinstance(value, self.array_module.ndarray)

    def max(self, data: NumpyNativeT, axis: Axis) -> NumpyNativeT:
        return self.array_module.max(data, axis=axis)

    def min(self, data: NumpyNativeT, axis: Axis) -> NumpyNativeT:
        return self.array_module.min(data, axis=axis)

    def mean(self, data: NumpyNativeT, axis: Axis) -> NumpyNativeT:
        return self.array_module.mean(data, axis=axis)

    def sum(self, data: NumpyNativeT, axis: Axis) -> NumpyNativeT:
        return self.array_module.sum(data, axis=axis)

    def linspace(self, start: float, stop: float, num: int, *, endpoint: bool) -> NumpyNativeT:
        return self.array_module.linspace(start, stop, num, endpoint=endpoint)

    def ones(self, shape: int | tuple[int, ...], dtype: DType) -> NumpyNativeT:
        return self.array_module.ones(shape, dtype=_mapped_dtype(dtype, self.array_module, object))

    def outer(self, left: NumpyNativeT, right: NumpyNativeT) -> NumpyNativeT:
        return self.array_module.outer(left, right)

    def floor(self, value: float) -> NumpyNativeT:
        return self.array_module.floor(value)

    def _stack(self, values: Sequence[NumpyNativeT]) -> NumpyNativeT:
        return self.array_module.stack(tuple(values), axis=0)

    def _cast(self, data: NumpyNativeT, dtype: DType) -> NumpyNativeT:
        return data.astype(dtype, copy=False)

    def _logical_and(self, left: NumpyNativeT, right: NumpyNativeT) -> NumpyNativeT:
        return self.array_module.logical_and(left, right)

    def _reshape(self, data: NumpyNativeT, shape: tuple[int, ...]) -> NumpyNativeT:
        return data.reshape(shape)

    def _broadcast_to(self, data: NumpyNativeT, shape: tuple[int, ...]) -> NumpyNativeT:
        return self.array_module.broadcast_to(data, shape)

    def _ones_like(
        self, reference: NumpyNativeT, shape: tuple[int, ...], dtype: DType
    ) -> NumpyNativeT:
        del reference
        return self.array_module.ones(shape, dtype=dtype)


class MutableArrays(NumpyArrayApi[MutableNativeT], InPlaceAssignment[MutableNativeT]):
    """Libraries with the NumPy array API whose arrays can be changed in place."""

    @property
    @abstractmethod
    def array_module(self) -> MutableNumpyFunctions[MutableNativeT]:
        """The library's module of NumPy-named functions."""

    def _normalize_planes(
        self, data: MutableNativeT, dtype: DType, scales: Sequence[float | None]
    ) -> MutableNativeT:
        import numpy as np

        # Integer division promotes the output dtype, so only floats divide in place.
        if not np.issubdtype(np.dtype(dtype), np.inexact) or len(scales) == 0:
            return super()._normalize_planes(data, dtype, scales)
        normalized = self.array_module.array(data, dtype=dtype, copy=True)
        for index, scale in enumerate(scales):
            if scale is not None:
                normalized[index] /= float(scale)
        return normalized

    def _scale_dtype(self, result: MutableNativeT, target_dtype: DType) -> MutableNativeT:
        module = self.array_module
        if not (
            module.issubdtype(result.dtype, module.floating) and self._rescales_into(target_dtype)
        ):
            return result.astype(target_dtype)
        result_min = module.min(result)
        result_max = module.max(result)
        if result_max <= result_min:
            return result.astype(target_dtype)
        scaled = _scaled_values(result, result_min, result_max, target_dtype)
        bounds = _clamp_bounds(target_dtype)
        if bounds is not None:
            scaled = module.clip(scaled, *bounds)
        return scaled.astype(target_dtype)

    @abstractmethod
    def _rescales_into(self, target_dtype: DType) -> bool:
        """Whether a float array converted to ``target_dtype`` is rescaled first."""


# ---------------------------------------------------------------------------
# The libraries.
# ---------------------------------------------------------------------------


class NumpyLibrary(NumpyBufferSharing["np.ndarray"], MutableArrays["np.ndarray"]):
    name = "numpy"
    display_name = "NumPy"
    oom_messages = ("cannot allocate memory", "memory exhausted")

    if TYPE_CHECKING:
        module = np

    @property
    def array_module(self) -> MutableNumpyFunctions[np.ndarray]:
        return self.module

    @classmethod
    def registered(cls) -> NumpyLibrary:
        """The NumPy library every conversion to or from NumPy names as its other end."""

        library = ArrayLibrary.named(cls.name)
        if not isinstance(library, NumpyLibrary):
            raise TypeError(f"{library!r} is registered as '{cls.name}' and is not NumPy")
        return library

    def device_of(self, data: np.ndarray) -> Device:
        del data
        return HOST

    def _first_element_address(self, data: np.ndarray) -> int:
        return data.ctypes.data

    def accepts(self, value: object) -> bool:
        return isinstance(value, (self.module.ndarray, ArrayPayload))

    def _to_host(self, data: np.ndarray) -> np.ndarray:
        return data

    def _wrap_host_array(self, data: np.ndarray) -> np.ndarray:
        return data

    def _move_to_host(self, data: np.ndarray) -> np.ndarray:
        return data

    def _rescales_into(self, target_dtype: DType) -> bool:
        return bool(self.module.issubdtype(target_dtype, self.module.integer))


class CupyLibrary(
    CudaLibrary["cupy.ndarray"],
    DLPackImport["cupy.ndarray"],
    DLPackProtocolExport["cupy.ndarray"],
    NvidiaWheelLibraries["cupy.ndarray"],
    MutableArrays["cupy.ndarray"],
):
    name = "cupy"
    display_name = "CuPy"
    oom_messages = ("out of memory", "cuda_error_out_of_memory")
    oom_exception_paths = (
        ("cuda", "memory", "OutOfMemoryError"),
        ("cuda", "runtime", "CUDARuntimeError"),
    )

    if TYPE_CHECKING:
        module = cupy

    @property
    def array_module(self) -> MutableNumpyFunctions[cupy.ndarray]:
        return self.module

    def device_of(self, data: cupy.ndarray) -> Device:
        return AcceleratorDevice(int(data.device.id))

    def _device_if_accepted(self, value: object) -> Device | None:
        return self.device_of(value) if isinstance(value, self.module.ndarray) else None

    def _accelerator_count(self) -> int:
        return int(self.module.cuda.runtime.getDeviceCount())

    def _accelerator_scope(self, device: AcceleratorDevice) -> AbstractContextManager[object]:
        return self.module.cuda.Device(device.index)

    def _activate_accelerator(self, device: AcceleratorDevice) -> None:
        self.module.cuda.Device(device.index).use()

    def _move_to_accelerator(self, data: cupy.ndarray, device: AcceleratorDevice) -> cupy.ndarray:
        del device
        return data.copy()

    def _upload(self, data: np.ndarray, device: AcceleratorDevice) -> cupy.ndarray:
        del device
        return self.module.array(data)

    def current_device(self) -> AcceleratorDevice | None:
        return AcceleratorDevice(int(self.module.cuda.runtime.getDevice()))

    def create_stream(self) -> object:
        return self.module.cuda.Stream()

    def stream_scope(self, stream: object) -> AbstractContextManager[object]:
        if not isinstance(stream, AbstractContextManager):
            raise TypeError(f"{stream!r} is not a CuPy stream")
        # A CuPy stream is its own context manager.
        return stream

    def _free_cached_memory(self) -> None:
        self.module.get_default_memory_pool().free_all_blocks()
        self.module.get_default_pinned_memory_pool().free_all_blocks()
        self.module.cuda.runtime.deviceSynchronize()

    def _reads_dlpack(self, payload: DLPackPayload) -> bool:
        # CuPy releases before ``from_dlpack`` took capsules read them with ``fromDlpack``.
        return _has_dlpack_protocol(payload.source) or hasattr(self.module, "fromDlpack")

    def _import_dlpack(self, payload: DLPackPayload) -> cupy.ndarray:
        if _has_dlpack_protocol(payload.source):
            return self.module.from_dlpack(payload.source)
        return self.module.fromDlpack(payload.capsule)

    def _to_host(self, data: cupy.ndarray) -> np.ndarray:
        return data.get()

    def _rescales_into(self, target_dtype: DType) -> bool:
        return not self.module.issubdtype(target_dtype, self.module.floating)


class TorchLibrary(
    NumpyBufferSharing["torch.Tensor"],
    CudaLibrary["torch.Tensor"],
    DLPackImport["torch.Tensor"],
    DLPackProtocolExport["torch.Tensor"],
    ArrayMath["torch.Tensor"],
    InPlaceAssignment["torch.Tensor"],
):
    name = "torch"
    display_name = "PyTorch"
    oom_messages = ("out of memory", "cuda_error_out_of_memory")
    oom_exception_paths = (("cuda", "OutOfMemoryError"),)

    if TYPE_CHECKING:
        module = torch

    def device_of(self, data: torch.Tensor) -> Device:
        return AcceleratorDevice(int(data.device.index)) if data.is_cuda else HOST

    def _device_if_accepted(self, value: object) -> Device | None:
        return self.device_of(value) if isinstance(value, self.module.Tensor) else None

    def _accelerator_count(self) -> int:
        return int(self.module.cuda.device_count()) if self.module.cuda.is_available() else 0

    def _accelerator_scope(self, device: AcceleratorDevice) -> AbstractContextManager[object]:
        return self.module.cuda.device(device.index)

    def _activate_accelerator(self, device: AcceleratorDevice) -> None:
        self.module.cuda.set_device(device.index)

    def _move_to_accelerator(self, data: torch.Tensor, device: AcceleratorDevice) -> torch.Tensor:
        return data.to(f"cuda:{device.index}")

    def _upload(self, data: np.ndarray, device: AcceleratorDevice) -> torch.Tensor:
        return self._wrap_host_array(data).to(f"cuda:{device.index}")

    def _wrap_host_array(self, data: np.ndarray) -> torch.Tensor:
        import numpy as np

        host_data = (
            np.ascontiguousarray(data)
            if any(stride < 0 for stride in getattr(data, "strides", ()))
            else data
        )
        return self.module.from_numpy(host_data)

    def _move_to_host(self, data: torch.Tensor) -> torch.Tensor:
        return data.cpu()

    def _first_element_address(self, data: torch.Tensor) -> int:
        return data.data_ptr()

    def current_device(self) -> AcceleratorDevice | None:
        return (
            AcceleratorDevice(int(self.module.cuda.current_device()))
            if self.module.cuda.is_available()
            else None
        )

    def create_stream(self) -> object:
        return self.module.cuda.Stream()

    def stream_scope(self, stream: object) -> AbstractContextManager[object]:
        if stream is not None and not isinstance(stream, self.module.cuda.Stream):
            raise TypeError(f"{stream!r} is not a PyTorch stream")
        return self.module.cuda.stream(stream)

    def _free_cached_memory(self) -> None:
        self.module.cuda.empty_cache()
        self.module.cuda.synchronize()

    def _import_dlpack(self, payload: DLPackPayload) -> torch.Tensor:
        return self.module.from_dlpack(payload.capsule)

    def nbytes_of(self, data: torch.Tensor) -> int:
        return int(data.element_size()) * int(data.nelement())

    def accepts(self, value: object) -> bool:
        return isinstance(value, self.module.Tensor)

    def dtype_name(self, dtype: DType) -> str:
        return str(dtype).rsplit(".", maxsplit=1)[-1]

    def max(self, data: torch.Tensor, axis: Axis) -> torch.Tensor:
        # CUDA has no uint16 reduction; float32 holds every uint16 exactly.
        values = data.float() if data.dtype == self.module.uint16 else data
        return self.module.amax(values, dim=axis).to(data.dtype)

    def min(self, data: torch.Tensor, axis: Axis) -> torch.Tensor:
        values = data.float() if data.dtype == self.module.uint16 else data
        return self.module.amin(values, dim=axis).to(data.dtype)

    def mean(self, data: torch.Tensor, axis: Axis) -> torch.Tensor:
        return self.module.mean(data.float(), dim=axis)

    def sum(self, data: torch.Tensor, axis: Axis) -> torch.Tensor:
        return self.module.sum(data, dim=axis)

    def linspace(self, start: float, stop: float, num: int, *, endpoint: bool) -> torch.Tensor:
        if endpoint:
            return self.module.linspace(start, stop, num, dtype=self.module.float32)
        return self.module.linspace(start, stop, num + 1, dtype=self.module.float32)[:-1]

    def ones(self, shape: int | tuple[int, ...], dtype: DType) -> torch.Tensor:
        return self.module.ones(shape, dtype=_mapped_dtype(dtype, self.module, self.module.dtype))

    def outer(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.module.outer(left, right)

    def floor(self, value: float) -> torch.Tensor:
        return self.module.floor(self.module.tensor(value))

    def _to_host(self, data: torch.Tensor) -> np.ndarray:
        return data.cpu().numpy()

    def _scale_dtype(self, result: torch.Tensor, target_dtype: DType) -> torch.Tensor:
        import numpy as np

        mapped = _mapped_dtype(target_dtype, self.module, self.module.dtype)
        floats = (self.module.float16, self.module.float32, self.module.float64)
        if not (result.dtype in floats and np.issubdtype(np.dtype(target_dtype), np.integer)):
            return result.to(mapped)
        result_min = result.min()
        result_max = result.max()
        if result_max <= result_min:
            return result.to(mapped)
        scaled = _scaled_values(result, result_min, result_max, target_dtype)
        bounds = _clamp_bounds(target_dtype)
        if bounds is not None:
            scaled = self.module.clamp(scaled, min=bounds[0], max=bounds[1])
        return scaled.to(mapped)

    def _stack(self, values: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.module.stack(tuple(values), dim=0)

    def _cast(self, data: torch.Tensor, dtype: DType) -> torch.Tensor:
        native = (
            dtype
            if isinstance(dtype, self.module.dtype)
            else _mapped_dtype(dtype, self.module, self.module.dtype)
        )
        return data.to(dtype=native)

    def _logical_and(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.module.logical_and(left, right)

    def _reshape(self, data: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        return data.reshape(shape)

    def _broadcast_to(self, data: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        return data.expand(shape)

    def _ones_like(
        self, reference: torch.Tensor, shape: tuple[int, ...], dtype: DType
    ) -> torch.Tensor:
        return self.module.ones(
            shape,
            dtype=_mapped_dtype(dtype, self.module, self.module.dtype),
            device=reference.device,
        )


class TensorflowLibrary(
    HostDevices["tensorflow.Tensor"],
    AcceleratorDevices["tensorflow.Tensor"],
    DLPackImport["tensorflow.Tensor"],
    ArrayMath["tensorflow.Tensor"],
):
    name = "tensorflow"
    display_name = "TensorFlow"
    import_environment = (("TF_FORCE_GPU_ALLOW_GROWTH", "true"),)
    oom_messages = ("out of memory", "resource_exhausted")
    oom_exception_paths = (("errors", "ResourceExhaustedError"),)

    if TYPE_CHECKING:
        module = tensorflow

    def device_of(self, data: tensorflow.Tensor) -> Device:
        device = data.device.lower()
        if "gpu" not in device:
            return HOST
        return AcceleratorDevice(int(device.rsplit(":", maxsplit=1)[-1]))

    def _device_if_accepted(self, value: object) -> Device | None:
        return self.device_of(value) if isinstance(value, self.module.Tensor) else None

    def _accelerator_count(self) -> int:
        return len(self.module.config.list_logical_devices("GPU"))

    def _accelerator_scope(self, device: AcceleratorDevice) -> AbstractContextManager[object]:
        return self.module.device(f"/device:GPU:{device.index}")

    def _host_scope(self) -> AbstractContextManager[object]:
        return self.module.device("/device:CPU:0")

    def _move_to_accelerator(
        self, data: tensorflow.Tensor, device: AcceleratorDevice
    ) -> tensorflow.Tensor:
        del device
        return self.module.identity(data)

    def _move_to_host(self, data: tensorflow.Tensor) -> tensorflow.Tensor:
        return self.module.identity(data)

    def _upload(self, data: np.ndarray, device: AcceleratorDevice) -> tensorflow.Tensor:
        del device
        return self.module.convert_to_tensor(data)

    def _wrap_host_array(self, data: np.ndarray) -> tensorflow.Tensor:
        return self.module.convert_to_tensor(data)

    def exports_dlpack_from(self, device: Device) -> bool:
        # TensorFlow hands over accelerator tensors only.
        return isinstance(device, AcceleratorDevice)

    def exports_dlpack_data(self, data: tensorflow.Tensor) -> bool:
        try:
            major, minor = map(int, self.module.__version__.split(".")[:2])
        except (AttributeError, TypeError, ValueError):
            return False
        if (major, minor) < (2, 12):
            return False
        if "gpu" not in str(getattr(data, "device", "")).lower():
            return False
        dlpack = getattr(getattr(self.module, "experimental", None), "dlpack", None)
        return callable(getattr(dlpack, "to_dlpack", None))

    def export_dlpack(self, data: tensorflow.Tensor) -> DLPackPayload | None:
        if not self.exports_dlpack_data(data):
            return None
        return DLPackPayload(source=data, capsule=self.module.experimental.dlpack.to_dlpack(data))

    def _import_dlpack(self, payload: DLPackPayload) -> tensorflow.Tensor:
        return self.module.experimental.dlpack.from_dlpack(payload.capsule)

    def nbytes_of(self, data: tensorflow.Tensor) -> int:
        return int(data.shape.num_elements()) * int(data.dtype.size)

    def accepts(self, value: object) -> bool:
        return isinstance(value, self.module.Tensor)

    def dtype_name(self, dtype: DType) -> str:
        import numpy as np

        return str(np.dtype(getattr(dtype, "as_numpy_dtype", dtype)).name)

    def max(self, data: tensorflow.Tensor, axis: Axis) -> tensorflow.Tensor:
        return self.module.reduce_max(data, axis=axis)

    def min(self, data: tensorflow.Tensor, axis: Axis) -> tensorflow.Tensor:
        return self.module.reduce_min(data, axis=axis)

    def mean(self, data: tensorflow.Tensor, axis: Axis) -> tensorflow.Tensor:
        return self.module.reduce_mean(self.module.cast(data, self.module.float32), axis=axis)

    def sum(self, data: tensorflow.Tensor, axis: Axis) -> tensorflow.Tensor:
        return self.module.reduce_sum(data, axis=axis)

    def linspace(self, start: float, stop: float, num: int, *, endpoint: bool) -> tensorflow.Tensor:
        if endpoint:
            return self.module.linspace(float(start), float(stop), num)
        return self.module.linspace(float(start), float(stop), num + 1)[:-1]

    def ones(self, shape: int | tuple[int, ...], dtype: DType) -> tensorflow.Tensor:
        return self.module.ones(
            shape, dtype=_mapped_dtype(dtype, self.module, self.module.dtypes.DType)
        )

    def assign(
        self, data: tensorflow.Tensor, index: slice, values: tensorflow.Tensor
    ) -> tensorflow.Tensor:
        rows = self.module.range(*index.indices(int(data.shape[0])))
        return self.module.tensor_scatter_nd_update(
            data, self.module.reshape(rows, [-1, 1]), values
        )

    def outer(self, left: tensorflow.Tensor, right: tensorflow.Tensor) -> tensorflow.Tensor:
        return self.module.tensordot(left, right, axes=0)

    def floor(self, value: float) -> tensorflow.Tensor:
        return self.module.math.floor(value)

    def _to_host(self, data: tensorflow.Tensor) -> np.ndarray:
        return data.numpy()

    def _scale_dtype(self, result: tensorflow.Tensor, target_dtype: DType) -> tensorflow.Tensor:
        import numpy as np

        mapped = _mapped_dtype(target_dtype, self.module, self.module.dtypes.DType)
        floats = (self.module.float16, self.module.float32, self.module.float64)
        if not (result.dtype in floats and np.issubdtype(np.dtype(target_dtype), np.integer)):
            return self.module.cast(result, mapped)
        result_min = self.module.reduce_min(result)
        result_max = self.module.reduce_max(result)
        if result_max <= result_min:
            return self.module.cast(result, mapped)
        scaled = _scaled_values(result, result_min, result_max, target_dtype)
        bounds = _clamp_bounds(target_dtype)
        if bounds is not None:
            scaled = self.module.clip_by_value(scaled, *bounds)
        return self.module.cast(scaled, mapped)

    def _cast(self, data: tensorflow.Tensor, dtype: DType) -> tensorflow.Tensor:
        native = (
            dtype
            if isinstance(dtype, self.module.dtypes.DType)
            else _mapped_dtype(dtype, self.module, self.module.dtypes.DType)
        )
        return self.module.cast(data, native)

    def _stack(self, values: Sequence[tensorflow.Tensor]) -> tensorflow.Tensor:
        return self.module.stack(tuple(values), axis=0)

    def _logical_and(self, left: tensorflow.Tensor, right: tensorflow.Tensor) -> tensorflow.Tensor:
        return self.module.logical_and(left, right)

    def _reshape(self, data: tensorflow.Tensor, shape: tuple[int, ...]) -> tensorflow.Tensor:
        return self.module.reshape(data, shape)

    def _broadcast_to(self, data: tensorflow.Tensor, shape: tuple[int, ...]) -> tensorflow.Tensor:
        return self.module.broadcast_to(data, shape)

    def _ones_like(
        self, reference: tensorflow.Tensor, shape: tuple[int, ...], dtype: DType
    ) -> tensorflow.Tensor:
        with self.module.device(reference.device):
            return self.module.ones(
                shape, dtype=_mapped_dtype(dtype, self.module, self.module.dtypes.DType)
            )


class JaxLibrary(
    NumpyBufferSharing["jax.Array"],
    AcceleratorDevices["jax.Array"],
    DLPackImport["jax.Array"],
    DLPackProtocolExport["jax.Array"],
    NumpyArrayApi["jax.Array"],
):
    name = "jax"
    display_name = "JAX"
    module_aliases = ("jaxlib",)
    import_environment = (("XLA_PYTHON_CLIENT_PREALLOCATE", "false"),)
    oom_messages = ("out of memory", "oom when allocating", "allocation failure")

    if TYPE_CHECKING:
        module = jax

    @property
    def array_module(self) -> NumpyFunctions[jax.Array]:
        return self.module.numpy

    def _platform_devices(self, platform: str) -> tuple[jax.Device, ...]:
        return tuple(device for device in self.module.devices() if device.platform == platform)

    def device_of(self, data: jax.Array) -> Device:
        device = data.device
        device = device() if callable(device) else device
        if getattr(device, "platform", None) != "gpu":
            return HOST
        gpu_devices = self._platform_devices("gpu")
        for index, candidate in enumerate(gpu_devices):
            if candidate is device:
                return AcceleratorDevice(index)
        for index, candidate in enumerate(gpu_devices):
            if str(candidate) == str(device):
                return AcceleratorDevice(index)
        return NamedDevice(str(device))

    def _device_if_accepted(self, value: object) -> Device | None:
        return self.device_of(value) if isinstance(value, self.module.Array) else None

    def _accelerator_count(self) -> int:
        return len(self._platform_devices("gpu"))

    def _accelerator_scope(self, device: AcceleratorDevice) -> AbstractContextManager[object]:
        return self.module.default_device(self._platform_devices("gpu")[device.index])

    def _host_scope(self) -> AbstractContextManager[object]:
        return self.module.default_device(self.module.devices("cpu")[0])

    def _move_to_accelerator(self, data: jax.Array, device: AcceleratorDevice) -> jax.Array:
        return self.module.device_put(data, self._platform_devices("gpu")[device.index])

    def _move_to_host(self, data: jax.Array) -> jax.Array:
        return self.module.device_put(data, self.module.devices("cpu")[0])

    def _upload(self, data: np.ndarray, device: AcceleratorDevice) -> jax.Array:
        return self.module.device_put(data, self._platform_devices("gpu")[device.index])

    def _wrap_host_array(self, data: np.ndarray) -> jax.Array:
        return self.module.device_put(data, self.module.devices("cpu")[0])

    def _first_element_address(self, data: jax.Array) -> int:
        return data.unsafe_buffer_pointer()

    def _reads_dlpack(self, payload: DLPackPayload) -> bool:
        return _has_dlpack_protocol(payload.source)

    def _import_dlpack(self, payload: DLPackPayload) -> jax.Array:
        return self.module.dlpack.from_dlpack(self._aligned(payload.source))

    @staticmethod
    def _aligned(data: object) -> object:
        """Copy a DLPack source only when JAX's 16-byte alignment is unmet."""

        pointer: int | None = None
        data_ptr = getattr(data, "data_ptr", None)
        cuda_interface = getattr(data, "__cuda_array_interface__", None)
        if callable(data_ptr):
            pointer = int(data_ptr())
        elif isinstance(cuda_interface, dict):
            declared, _read_only = cuda_interface.get("data", (None, False))
            pointer = None if declared is None else int(declared)
        if pointer is None or pointer % 16 == 0:
            return data
        for attribute in ("clone", "copy"):
            copier = getattr(data, attribute, None)
            if callable(copier):
                return copier()
        return data

    def mean(self, data: jax.Array, axis: Axis) -> jax.Array:
        return self.module.numpy.mean(data.astype(self.module.numpy.float32), axis=axis)

    def assign(self, data: jax.Array, index: slice, values: jax.Array) -> jax.Array:
        return data.at[index].set(values)

    def _to_host(self, data: jax.Array) -> np.ndarray:
        import numpy as np

        return np.asarray(data)

    def _scale_dtype(self, result: jax.Array, target_dtype: DType) -> jax.Array:
        import numpy as np

        if np.dtype(target_dtype) == np.dtype(np.float64):
            x64_enabled = getattr(self.module.config, "x64_enabled", None)
            if x64_enabled is None:
                x64_enabled = self.module.config.read("jax_enable_x64")
            if not x64_enabled:
                raise ValueError(
                    "JAX float64 output requires x64 mode; set JAX_ENABLE_X64=true before import"
                )
        mapped = _mapped_dtype(target_dtype, self.module.numpy, object)
        floats = (self.module.numpy.float16, self.module.numpy.float32, self.module.numpy.float64)
        if not (result.dtype in floats and np.issubdtype(np.dtype(target_dtype), np.integer)):
            return result.astype(mapped)
        result_min = self.module.numpy.min(result)
        result_max = self.module.numpy.max(result)
        if result_max <= result_min:
            return result.astype(mapped)
        scaled = _scaled_values(result, result_min, result_max, target_dtype)
        bounds = _clamp_bounds(target_dtype)
        if bounds is not None:
            scaled = self.module.numpy.clip(scaled, *bounds)
        return scaled.astype(mapped)

    def _cast(self, data: jax.Array, dtype: DType) -> jax.Array:
        return data.astype(_mapped_dtype(dtype, self.module.numpy, object))

    def _ones_like(self, reference: jax.Array, shape: tuple[int, ...], dtype: DType) -> jax.Array:
        return self.module.numpy.ones(shape, dtype=dtype, device=reference.device)


class PyclesperantoLibrary(
    AcceleratorDevices["pyclesperanto.Array"], InPlaceAssignment["pyclesperanto.Array"]
):
    name = "pyclesperanto"
    oom_messages = ("cl_mem_object_allocation_failure", "cl_out_of_resources", "out of memory")

    if TYPE_CHECKING:
        module = pyclesperanto

    _native_dtype_names: ClassVar[frozenset[str]] = frozenset(
        {"float32", "int8", "int16", "int32", "uint8", "uint16", "uint32"}
    )

    def _gpu_devices(self) -> tuple[object, ...]:
        try:
            return tuple(self.module.list_available_devices("gpu"))
        except TypeError:
            return tuple(self.module.list_available_devices())

    @staticmethod
    def _device_name(device: object) -> str:
        return str(getattr(device, "name", None) or device)

    def _select(self, selector: str | int, device_type: str | None = None) -> None:
        if device_type is None:
            self.module.select_device(selector)
            return
        try:
            self.module.select_device(selector, device_type)
        except TypeError:
            self.module.select_device(selector)

    def _device_named(self, name: str) -> Device:
        for index, device in enumerate(self._gpu_devices()):
            if self._device_name(device) == name:
                return AcceleratorDevice(index)
        # An OpenCL device that is not a listed GPU, such as a CPU OpenCL device.
        return NamedDevice(name)

    def device_of(self, data: object) -> Device:
        # A host array has no device of its own: a kernel pushes it to the selected one.
        declared_device: object = getattr(data, "device", None)
        if declared_device is None:
            declared_device = self.module.get_device()
        return self._device_named(self._device_name(declared_device))

    def _device_if_accepted(self, value: object) -> Device | None:
        return self.device_of(value) if self.accepts(value) else None

    def _accelerator_count(self) -> int:
        return len(self._gpu_devices())

    @contextmanager
    def _accelerator_scope(self, device: AcceleratorDevice) -> Iterator[None]:
        previous_name = self._device_name(self.module.get_device())
        previous = self._device_named(previous_name)
        self._select(device.index, "gpu")
        try:
            yield
        finally:
            if previous != device:
                self._select(previous_name)

    def _activate_accelerator(self, device: AcceleratorDevice) -> None:
        self._select(device.index, "gpu")

    def _move_to_accelerator(
        self, data: pyclesperanto.Array, device: AcceleratorDevice
    ) -> pyclesperanto.Array:
        del device
        result = self.module.create_like(data)
        self.module.copy(data, result)
        return result

    def _upload(self, data: np.ndarray, device: AcceleratorDevice) -> pyclesperanto.Array:
        del device
        return self.module.push(data)

    def accepts(self, value: object) -> bool:
        # pyclesperanto kernels push any host array they receive.
        return hasattr(value, "shape")

    @staticmethod
    def _require_z_axis(axis: Axis) -> None:
        if axis != 0:
            raise NotImplementedError(
                f"pyclesperanto reduces only along the leading (z) axis, not axis {axis!r}"
            )

    def max(self, data: pyclesperanto.Array, axis: Axis) -> pyclesperanto.Array:
        self._require_z_axis(axis)
        return self.module.maximum_z_projection(data)

    def min(self, data: pyclesperanto.Array, axis: Axis) -> pyclesperanto.Array:
        self._require_z_axis(axis)
        return self.module.minimum_z_projection(data)

    def mean(self, data: pyclesperanto.Array, axis: Axis) -> pyclesperanto.Array:
        self._require_z_axis(axis)
        return self.module.mean_z_projection(data)

    def sum(self, data: pyclesperanto.Array, axis: Axis) -> pyclesperanto.Array:
        self._require_z_axis(axis)
        return self.module.sum_z_projection(data)

    def prepend_axis(self, data: pyclesperanto.Array) -> pyclesperanto.Array:
        result = self.module.create((1, *data.shape), dtype=data.dtype)
        result[0] = data
        return result

    def _to_host(self, data: pyclesperanto.Array) -> np.ndarray:
        return self.module.pull(data)

    def _scale_dtype(self, result: pyclesperanto.Array, target_dtype: DType) -> pyclesperanto.Array:
        import numpy as np

        target_is_int = np.issubdtype(np.dtype(target_dtype), np.integer)
        if not (np.issubdtype(result.dtype, np.floating) and target_is_int):
            return self.module.push(self.module.pull(result).astype(target_dtype))
        result_min = float(self.module.minimum_of_all_pixels(result))
        result_max = float(self.module.maximum_of_all_pixels(result))
        if result_max <= result_min:
            return self.module.push(self.module.pull(result).astype(target_dtype))
        normalized = self.module.subtract_image_from_scalar(result, scalar=result_min)
        normalized = self.module.multiply_image_and_scalar(
            normalized,
            scalar=1.0 / (result_max - result_min),
        )
        range_info = _SCALING_RANGES.get(_dtype_name(target_dtype))
        if isinstance(range_info, tuple):
            scale, offset = range_info
            scaled = self.module.multiply_image_and_scalar(normalized, scalar=scale)
            scaled = self.module.subtract_image_from_scalar(scaled, scalar=offset)
        elif range_info is not None:
            scaled = self.module.multiply_image_and_scalar(normalized, scalar=range_info)
        else:
            scaled = normalized
        host_values = self.module.pull(scaled)
        bounds = _clamp_bounds(target_dtype)
        if bounds is not None:
            host_values = np.clip(host_values, *bounds)
        return self.module.push(host_values.astype(target_dtype))

    def _stack(self, values: Sequence[pyclesperanto.Array]) -> pyclesperanto.Array:
        if not values:
            raise ValueError("Cannot stack an empty pyclesperanto sequence")
        if len(values) == 1:
            source = values[0]
            result = self.module.create((1, *source.shape), dtype=source.dtype)
            return self.module.copy_slice(source, result, 0)
        result = values[0]
        for value in values[1:]:
            result = self.module.concatenate_along_z(result, value)
        return result

    def _cast(self, data: pyclesperanto.Array, dtype: DType) -> pyclesperanto.Array:
        return self.module.push(self.module.pull(data).astype(dtype, copy=False))

    def _logical_and(
        self, left: pyclesperanto.Array, right: pyclesperanto.Array
    ) -> pyclesperanto.Array:
        import numpy as np

        return self.module.push(np.logical_and(self.module.pull(left), self.module.pull(right)))

    def _reshape(self, data: pyclesperanto.Array, shape: tuple[int, ...]) -> pyclesperanto.Array:
        if tuple(data.shape) == shape:
            return data
        raise NotImplementedError(
            "pyclesperanto does not provide native array reshape; its reshape downloads pixels"
        )

    def _broadcast_to(
        self, data: pyclesperanto.Array, shape: tuple[int, ...]
    ) -> pyclesperanto.Array:
        if tuple(data.shape) == shape:
            return data
        raise NotImplementedError("pyclesperanto does not provide native array broadcasting")

    def _ones_like(
        self, reference: pyclesperanto.Array, shape: tuple[int, ...], dtype: DType
    ) -> pyclesperanto.Array:
        import numpy as np

        if np.dtype(dtype).name not in self._native_dtype_names:
            raise NotImplementedError(f"pyclesperanto cannot allocate native dtype {dtype!r}")
        if not 1 <= len(shape) <= 3:
            raise NotImplementedError(
                "pyclesperanto native allocation requires one to three dimensions"
            )
        result = self.module.create(shape, dtype=dtype, device=reference.device)
        self.module.set(result, scalar=1, device=reference.device)
        return result
