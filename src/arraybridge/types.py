"""
Memory type definitions for arraybridge.

This module defines the MemoryType enum and related constants for managing
different array/tensor frameworks.
"""

import importlib
import importlib.util
import logging
import os
import sys
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, cast

from arraybridge.array_operations import (
    CUPY_OPERATIONS,
    JAX_OPERATIONS,
    NUMPY_OPERATIONS,
    PYCLESPERANTO_OPERATIONS,
    TENSORFLOW_OPERATIONS,
    TORCH_OPERATIONS,
    ArrayOperations,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)
ConversionFunc = Callable[[Any], Any]
DeviceIdsResolver = Callable[[Any], tuple[int, ...]]
DeviceIdResolver = Callable[[Any, Any], int | None]
DeviceScopeFactory = Callable[[Any, int], AbstractContextManager[None]]
DeviceActivator = Callable[[Any, int], None]
FrameworkCleanup = Callable[[Any], None]
ActiveDeviceMover = Callable[[Any, Any, int], Any]
CurrentDeviceResolver = Callable[[Any], int | None]
StreamFactory = Callable[[Any], Any]
StreamScopeFactory = Callable[[Any, Any], AbstractContextManager[None]]
DLPackExporter = Callable[[Any, Any], Any | None]
DLPackValidator = Callable[[Any, Any], bool]
OOMMatcher = Callable[[BaseException, Any | None], bool]
SubprocessEnvironmentResolver = Callable[[Mapping[str, str]], dict[str, str]]


def _identity_subprocess_environment(
    environment: Mapping[str, str],
) -> dict[str, str]:
    return dict(environment)


def _nvidia_wheel_library_paths() -> tuple[str, ...]:
    """Return native-library directories declared by installed NVIDIA wheels."""

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


def _nvidia_wheel_subprocess_environment(
    environment: Mapping[str, str],
) -> dict[str, str]:
    """Prepend installed NVIDIA wheel libraries to a child environment."""

    prepared = dict(environment)
    library_paths = _nvidia_wheel_library_paths()
    if not library_paths:
        return prepared

    search_variable = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    existing_paths = tuple(
        path for path in prepared.get(search_variable, "").split(os.pathsep) if path
    )
    prepared[search_variable] = os.pathsep.join(dict.fromkeys((*library_paths, *existing_paths)))
    return prepared


class MemoryContractAttribute(str, Enum):
    """Callable metadata keys owned by ArrayBridge's decorator contract."""

    INPUT = "input_memory_type"
    OUTPUT = "output_memory_type"
    EXECUTION = "execution_memory_type"

    def read(self, namespace: Any, default: Any = None) -> Any:
        """Read this declaration from an attribute or mapping namespace."""

        if isinstance(namespace, Mapping):
            return namespace.get(self.value, default)
        return getattr(namespace, self.value, default)

    def write(self, namespace: Any, value: Any) -> None:
        """Write this declaration to an attribute or mutable mapping namespace."""

        if isinstance(namespace, MutableMapping):
            namespace[self.value] = value
            return
        setattr(namespace, self.value, value)


@dataclass(frozen=True, slots=True)
class DLPackPayload:
    """One exported capsule plus its original protocol-bearing array."""

    source: Any
    capsule: Any


DLPackImporter = Callable[[DLPackPayload, Any], Any]


def _no_device_ids(module: Any) -> tuple[int, ...]:
    del module
    return ()


def _no_device_id(data: Any, module: Any) -> None:
    del data, module
    return None


def _cupy_device_id(data: Any, module: Any) -> int:
    del module
    return int(data.device.id)


def _torch_device_id(data: Any, module: Any) -> int | None:
    del module
    return int(data.device.index) if data.is_cuda else None


def _tensorflow_device_id(data: Any, module: Any) -> int | None:
    del module
    device = data.device.lower()
    return int(device.rsplit(":", maxsplit=1)[-1]) if "gpu" in device else None


def _jax_device_id(data: Any, module: Any) -> int | None:
    device = data.device
    device = device() if callable(device) else device
    if getattr(device, "platform", None) != "gpu":
        return None
    gpu_devices = tuple(candidate for candidate in module.devices() if candidate.platform == "gpu")
    identity_match = next(
        (index for index, candidate in enumerate(gpu_devices) if candidate is device),
        None,
    )
    if identity_match is not None:
        return identity_match
    return next(
        (index for index, candidate in enumerate(gpu_devices) if str(candidate) == str(device)),
        None,
    )


def _pyclesperanto_device_id(data: Any, module: Any) -> int | None:
    declared_device = getattr(data, "device", None)
    if declared_device is None:
        declared_device = module.get_device()
    declared_selector = getattr(declared_device, "name", None)
    if declared_selector is None:
        declared_selector = str(declared_device)
    devices = _pyclesperanto_devices(module)
    return next(
        (
            index
            for index, device in enumerate(devices)
            if getattr(device, "name", str(device)) == declared_selector
        ),
        None,
    )


def _cupy_device_ids(module: Any) -> tuple[int, ...]:
    return tuple(range(int(module.cuda.runtime.getDeviceCount())))


def _torch_device_ids(module: Any) -> tuple[int, ...]:
    if not module.cuda.is_available():
        return ()
    return tuple(range(int(module.cuda.device_count())))


def _tensorflow_device_ids(module: Any) -> tuple[int, ...]:
    return tuple(range(len(module.config.list_logical_devices("GPU"))))


def _jax_device_ids(module: Any) -> tuple[int, ...]:
    return tuple(
        range(len(tuple(device for device in module.devices() if device.platform == "gpu")))
    )


def _pyclesperanto_devices(module: Any) -> tuple[Any, ...]:
    try:
        return tuple(module.list_available_devices("gpu"))
    except TypeError:
        return tuple(module.list_available_devices())


def _pyclesperanto_select_device(
    module: Any,
    selector: str | int,
    device_type: str | None = None,
) -> None:
    if device_type is None:
        module.select_device(selector)
        return
    try:
        module.select_device(selector, device_type)
    except TypeError:
        module.select_device(selector)


def _pyclesperanto_device_ids(module: Any) -> tuple[int, ...]:
    return tuple(range(len(_pyclesperanto_devices(module))))


def _null_device_scope(module: Any, device_id: int) -> AbstractContextManager[None]:
    del module, device_id
    return nullcontext()


def _cupy_device_scope(module: Any, device_id: int) -> AbstractContextManager[None]:
    return cast(AbstractContextManager[None], module.cuda.Device(device_id))


def _torch_device_scope(module: Any, device_id: int) -> AbstractContextManager[None]:
    return cast(AbstractContextManager[None], module.cuda.device(device_id))


def _tensorflow_device_scope(
    module: Any,
    device_id: int,
) -> AbstractContextManager[None]:
    return cast(AbstractContextManager[None], module.device(f"/device:GPU:{device_id}"))


def _jax_device_scope(module: Any, device_id: int) -> AbstractContextManager[None]:
    gpu_devices = tuple(device for device in module.devices() if device.platform == "gpu")
    return cast(AbstractContextManager[None], module.default_device(gpu_devices[device_id]))


@contextmanager
def _pyclesperanto_device_scope(module: Any, device_id: int) -> Iterator[None]:
    current_device = module.get_device()
    current_id = _pyclesperanto_device_id(None, module)
    current_selector = getattr(current_device, "name", None)
    if current_selector is None:
        current_selector = str(current_device)
    _pyclesperanto_select_device(module, device_id, "gpu")
    try:
        yield
    finally:
        if current_id != device_id:
            _pyclesperanto_select_device(module, current_selector)


def _no_device_activation(module: Any, device_id: int) -> None:
    del module, device_id


def _cupy_device_activation(module: Any, device_id: int) -> None:
    module.cuda.Device(device_id).use()


def _torch_device_activation(module: Any, device_id: int) -> None:
    module.cuda.set_device(device_id)


def _pyclesperanto_device_activation(module: Any, device_id: int) -> None:
    _pyclesperanto_select_device(module, device_id, "gpu")


def _cupy_cleanup(module: Any) -> None:
    module.get_default_memory_pool().free_all_blocks()
    module.get_default_pinned_memory_pool().free_all_blocks()
    module.cuda.runtime.deviceSynchronize()


def _torch_cleanup(module: Any) -> None:
    module.cuda.empty_cache()
    module.cuda.synchronize()


def _identity_device_move(data: Any, module: Any, device_id: int) -> Any:
    del module, device_id
    return data


def _cupy_device_move(data: Any, module: Any, device_id: int) -> Any:
    del module, device_id
    return data.copy()


def _torch_device_move(data: Any, module: Any, device_id: int) -> Any:
    del module
    return data.to(f"cuda:{device_id}")


def _tensorflow_device_move(data: Any, module: Any, device_id: int) -> Any:
    del device_id
    return module.identity(data)


def _jax_device_move(data: Any, module: Any, device_id: int) -> Any:
    gpu_devices = tuple(device for device in module.devices() if device.platform == "gpu")
    return module.device_put(data, gpu_devices[device_id])


def _pyclesperanto_device_move(data: Any, module: Any, device_id: int) -> Any:
    del device_id
    result = module.create_like(data)
    module.copy(data, result)
    return result


def _no_current_device(module: Any) -> None:
    del module
    return None


def _cupy_current_device(module: Any) -> int:
    return int(module.cuda.runtime.getDevice())


def _torch_current_device(module: Any) -> int | None:
    return int(module.cuda.current_device()) if module.cuda.is_available() else None


def _cupy_stream(module: Any) -> Any:
    return module.cuda.Stream()


def _torch_stream(module: Any) -> Any:
    return module.cuda.Stream()


def _cupy_stream_scope(module: Any, stream: Any) -> AbstractContextManager[None]:
    del module
    return cast(AbstractContextManager[None], stream)


def _torch_stream_scope(module: Any, stream: Any) -> AbstractContextManager[None]:
    return cast(AbstractContextManager[None], module.cuda.stream(stream))


def _null_stream_scope(module: Any, stream: Any) -> AbstractContextManager[None]:
    del module, stream
    return nullcontext()


def _has_modern_dlpack_protocol(data: Any) -> bool:
    return callable(getattr(data, "__dlpack__", None)) and callable(
        getattr(data, "__dlpack_device__", None)
    )


def _cupy_from_dlpack(payload: DLPackPayload, module: Any) -> Any:
    if _has_modern_dlpack_protocol(payload.source):
        return module.from_dlpack(payload.source)
    legacy_importer = getattr(module, "fromDlpack", None)
    return NotImplemented if legacy_importer is None else legacy_importer(payload.capsule)


def _torch_from_dlpack(payload: DLPackPayload, module: Any) -> Any:
    return module.from_dlpack(payload.capsule)


def _export_dlpack(data: Any) -> Any:
    for attribute in ("__dlpack__", "to_dlpack", "toDlpack"):
        exporter = getattr(data, attribute, None)
        if callable(exporter):
            return exporter()
    raise TypeError(f"{type(data).__name__} does not expose a DLPack exporter")


def _tensorflow_from_dlpack(payload: DLPackPayload, module: Any) -> Any:
    return module.experimental.dlpack.from_dlpack(payload.capsule)


def _dlpack_data_pointer(data: Any) -> int | None:
    """Return a protocol source pointer when its framework exposes one."""

    data_ptr = getattr(data, "data_ptr", None)
    if callable(data_ptr):
        return int(data_ptr())
    cuda_interface = getattr(data, "__cuda_array_interface__", None)
    if isinstance(cuda_interface, dict):
        pointer, _read_only = cuda_interface.get("data", (None, False))
        return None if pointer is None else int(pointer)
    return None


def _jax_aligned_dlpack_source(data: Any) -> Any:
    """Copy a DLPack source only when JAX's 16-byte alignment is unmet."""

    pointer = _dlpack_data_pointer(data)
    if pointer is None or pointer % 16 == 0:
        return data
    for attribute in ("clone", "copy"):
        copier = getattr(data, attribute, None)
        if callable(copier):
            return copier()
    return data


def _jax_from_dlpack(payload: DLPackPayload, module: Any) -> Any:
    if not _has_modern_dlpack_protocol(payload.source):
        return NotImplemented
    return module.dlpack.from_dlpack(_jax_aligned_dlpack_source(payload.source))


def _protocol_dlpack(data: Any, module: Any) -> bool:
    del module
    return any(
        callable(getattr(data, attribute, None))
        for attribute in ("__dlpack__", "toDlpack", "to_dlpack")
    )


def _protocol_dlpack_export(data: Any, module: Any) -> Any | None:
    del module
    try:
        return _export_dlpack(data)
    except TypeError:
        return None


def _tensorflow_dlpack(data: Any, module: Any) -> bool:
    try:
        major, minor = map(int, module.__version__.split(".")[:2])
    except (AttributeError, TypeError, ValueError):
        return False
    if (major, minor) < (2, 12):
        return False
    if "gpu" not in str(getattr(data, "device", "")).lower():
        return False
    dlpack = getattr(getattr(module, "experimental", None), "dlpack", None)
    if not callable(getattr(dlpack, "to_dlpack", None)):
        return False
    return True


def _tensorflow_dlpack_export(data: Any, module: Any) -> Any | None:
    if not _tensorflow_dlpack(data, module):
        return None
    return module.experimental.dlpack.to_dlpack(data)


def _message_matches(error: BaseException, *patterns: str) -> bool:
    message = str(error).lower()
    return any(pattern in message for pattern in patterns)


def _exception_type(module: Any, *path: str) -> type[BaseException] | None:
    candidate = module
    for attribute in path:
        candidate = getattr(candidate, attribute, None)
        if candidate is None:
            return None
    return (
        candidate if isinstance(candidate, type) and issubclass(candidate, BaseException) else None
    )


def _matches_declared_exception(error: BaseException, module: Any | None, *path: str) -> bool:
    if module is None:
        return False
    exception_type = _exception_type(module, *path)
    return exception_type is not None and isinstance(error, exception_type)


def _numpy_oom(error: BaseException, module: Any | None) -> bool:
    del module
    return _message_matches(error, "cannot allocate memory", "memory exhausted")


def _never_oom(error: BaseException, module: Any | None) -> bool:
    del error, module
    return False


def _cupy_oom(error: BaseException, module: Any | None) -> bool:
    return (
        _matches_declared_exception(error, module, "cuda", "memory", "OutOfMemoryError")
        or _matches_declared_exception(error, module, "cuda", "runtime", "CUDARuntimeError")
        or _message_matches(error, "out of memory", "cuda_error_out_of_memory")
    )


def _torch_oom(error: BaseException, module: Any | None) -> bool:
    return _matches_declared_exception(
        error, module, "cuda", "OutOfMemoryError"
    ) or _message_matches(error, "out of memory", "cuda_error_out_of_memory")


def _tensorflow_oom(error: BaseException, module: Any | None) -> bool:
    return _matches_declared_exception(
        error, module, "errors", "ResourceExhaustedError"
    ) or _message_matches(error, "out of memory", "resource_exhausted")


def _jax_oom(error: BaseException, module: Any | None) -> bool:
    del module
    return _message_matches(error, "out of memory", "oom when allocating", "allocation failure")


def _pyclesperanto_oom(error: BaseException, module: Any | None) -> bool:
    del module
    return _message_matches(
        error,
        "cl_mem_object_allocation_failure",
        "cl_out_of_resources",
        "out of memory",
    )


@dataclass(frozen=True, slots=True)
class FrameworkRuntime:
    """Typed execution leaves carried by one ``MemoryType`` declaration."""

    device_ids: DeviceIdsResolver = _no_device_ids
    device_id: DeviceIdResolver = _no_device_id
    device_scope: DeviceScopeFactory = _null_device_scope
    activate_device: DeviceActivator = _no_device_activation
    cleanup: FrameworkCleanup | None = None
    move_to_active_device: ActiveDeviceMover = _identity_device_move
    current_device: CurrentDeviceResolver = _no_current_device
    stream_factory: StreamFactory | None = None
    stream_scope: StreamScopeFactory = _null_stream_scope
    dlpack_importer: DLPackImporter | None = None
    dlpack_exporter: DLPackExporter | None = None
    dlpack_validator: DLPackValidator = _protocol_dlpack
    oom_matcher: OOMMatcher = _never_oom
    subprocess_environment: SubprocessEnvironmentResolver = _identity_subprocess_environment


class _MemoryTypeFields:
    import_name: str
    display_name: str
    is_gpu: bool
    module_aliases: tuple[str, ...]
    import_environment: tuple[tuple[str, str], ...]
    _runtime: FrameworkRuntime
    _operations: ArrayOperations


class MemoryType(_MemoryTypeFields, Enum):
    """Array-framework declarations with member-owned runtime capability leaves."""

    def __new__(
        cls,
        value: str,
        *declaration: Any,
    ) -> "MemoryType":
        (
            import_name,
            display_name,
            is_gpu,
            module_aliases,
            import_environment,
            runtime,
            operations,
        ) = declaration
        member = object.__new__(cls)
        member._value_ = value
        member.import_name = cast(str, import_name)
        member.display_name = cast(str, display_name)
        member.is_gpu = cast(bool, is_gpu)
        member.module_aliases = cast(tuple[str, ...], module_aliases)
        member.import_environment = cast(tuple[tuple[str, str], ...], import_environment)
        member._runtime = cast(FrameworkRuntime, runtime)
        member._operations = cast(ArrayOperations, operations)
        return member

    NUMPY = (
        "numpy",
        "numpy",
        "NumPy",
        False,
        (),
        (),
        FrameworkRuntime(oom_matcher=_numpy_oom),
        NUMPY_OPERATIONS,
    )
    CUPY = (
        "cupy",
        "cupy",
        "CuPy",
        True,
        (),
        (),
        FrameworkRuntime(
            device_ids=_cupy_device_ids,
            device_id=_cupy_device_id,
            device_scope=_cupy_device_scope,
            activate_device=_cupy_device_activation,
            cleanup=_cupy_cleanup,
            move_to_active_device=_cupy_device_move,
            current_device=_cupy_current_device,
            stream_factory=_cupy_stream,
            stream_scope=_cupy_stream_scope,
            dlpack_importer=_cupy_from_dlpack,
            dlpack_exporter=_protocol_dlpack_export,
            oom_matcher=_cupy_oom,
            subprocess_environment=_nvidia_wheel_subprocess_environment,
        ),
        CUPY_OPERATIONS,
    )
    TORCH = (
        "torch",
        "torch",
        "PyTorch",
        True,
        (),
        (),
        FrameworkRuntime(
            device_ids=_torch_device_ids,
            device_id=_torch_device_id,
            device_scope=_torch_device_scope,
            activate_device=_torch_device_activation,
            cleanup=_torch_cleanup,
            move_to_active_device=_torch_device_move,
            current_device=_torch_current_device,
            stream_factory=_torch_stream,
            stream_scope=_torch_stream_scope,
            dlpack_importer=_torch_from_dlpack,
            dlpack_exporter=_protocol_dlpack_export,
            oom_matcher=_torch_oom,
        ),
        TORCH_OPERATIONS,
    )
    TENSORFLOW = (
        "tensorflow",
        "tensorflow",
        "TensorFlow",
        True,
        (),
        (("TF_FORCE_GPU_ALLOW_GROWTH", "true"),),
        FrameworkRuntime(
            device_ids=_tensorflow_device_ids,
            device_id=_tensorflow_device_id,
            device_scope=_tensorflow_device_scope,
            move_to_active_device=_tensorflow_device_move,
            dlpack_importer=_tensorflow_from_dlpack,
            dlpack_exporter=_tensorflow_dlpack_export,
            dlpack_validator=_tensorflow_dlpack,
            oom_matcher=_tensorflow_oom,
        ),
        TENSORFLOW_OPERATIONS,
    )
    JAX = (
        "jax",
        "jax",
        "JAX",
        True,
        ("jaxlib",),
        (("XLA_PYTHON_CLIENT_PREALLOCATE", "false"),),
        FrameworkRuntime(
            device_ids=_jax_device_ids,
            device_id=_jax_device_id,
            device_scope=_jax_device_scope,
            move_to_active_device=_jax_device_move,
            dlpack_importer=_jax_from_dlpack,
            dlpack_exporter=_protocol_dlpack_export,
            oom_matcher=_jax_oom,
        ),
        JAX_OPERATIONS,
    )
    PYCLESPERANTO = (
        "pyclesperanto",
        "pyclesperanto",
        "pyclesperanto",
        True,
        (),
        (),
        FrameworkRuntime(
            device_ids=_pyclesperanto_device_ids,
            device_id=_pyclesperanto_device_id,
            device_scope=_pyclesperanto_device_scope,
            activate_device=_pyclesperanto_device_activation,
            move_to_active_device=_pyclesperanto_device_move,
            oom_matcher=_pyclesperanto_oom,
        ),
        PYCLESPERANTO_OPERATIONS,
    )

    @property
    def recognized_module_names(self) -> frozenset[str]:
        """Return top-level module names owned by this framework declaration."""

        return frozenset((self.import_name, *self.module_aliases))

    def is_installed(self) -> bool:
        """Check package presence without importing the optional framework."""

        try:
            return importlib.util.find_spec(self.import_name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    def prepare_import(self) -> None:
        """Apply declaration-owned coexistence defaults before framework import."""

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

    @classmethod
    def subprocess_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Project framework import requirements into a child environment."""

        prepared = dict(os.environ if environment is None else environment)
        for memory_type in cls:
            for name, value in memory_type.import_environment:
                prepared.setdefault(name, value)
            prepared = memory_type._runtime.subprocess_environment(prepared)
        return prepared

    def loaded_module(self) -> Any | None:
        """Return an already-loaded framework without causing an import."""

        return sys.modules.get(self.import_name)

    def import_module(self) -> Any:
        """Import this declaration's framework module."""

        self.prepare_import()
        return importlib.import_module(self.import_name)

    def import_if_installed(self) -> Any | None:
        """Import this framework only when its package is present."""

        loaded = self.loaded_module()
        if loaded is not None:
            self.prepare_import()
            return loaded
        if not self.is_installed():
            return None
        return self.import_module()

    def to_numpy(self, data: Any, module: Any | None = None) -> Any:
        """Project one array to NumPy through this declaration's leaf."""

        framework = module if module is not None else self.import_module()
        return self._operations.to_numpy(data, framework)

    def from_numpy(
        self,
        data: Any,
        device_id: int,
        module: Any | None = None,
    ) -> Any:
        """Create one array from NumPy on a declared framework-local device."""

        framework = module if module is not None else self.import_module()
        with self.device_scope(device_id, framework):
            return self._operations.from_numpy(data, framework, device_id)

    def stack_arrays(
        self,
        arrays: list[Any],
        device_id: int,
        module: Any | None = None,
    ) -> Any:
        """Stack prepared arrays on a declared framework-local device."""

        framework = module if module is not None else self.import_module()
        with self.device_scope(device_id, framework):
            return self._operations.stack(arrays, framework)

    def scale_dtype(
        self,
        data: Any,
        target_dtype: Any,
        module: Any | None = None,
    ) -> Any:
        """Scale an array through this declaration's typed operation leaf."""

        framework = module if module is not None else self.import_if_installed()
        if framework is None:
            return data
        device_id = self.device_id_of(data, framework) if hasattr(data, "dtype") else None
        scope = nullcontext() if device_id is None else self.device_scope(device_id, framework)
        with scope:
            return self._operations.scale_dtype(data, target_dtype, framework)

    def canonical_dtype_name(self, dtype: Any) -> str:
        """Return this framework member's portable dtype identity."""

        return self._operations.dtype_name(dtype)

    def astype(
        self,
        data: Any,
        dtype: Any,
        module: Any | None = None,
    ) -> Any:
        """Cast one array through this framework member's operation leaf."""

        framework = module if module is not None else self.import_module()
        device_id = self.device_id_of(data, framework)
        scope = nullcontext() if device_id is None else self.device_scope(device_id, framework)
        with scope:
            return self._operations.cast(data, dtype, framework)

    def logical_and(
        self,
        left: Any,
        right: Any,
        module: Any | None = None,
    ) -> Any:
        """Intersect two arrays through this framework member's operation leaf."""

        framework = module if module is not None else self.import_module()
        device_id = self.device_id_of(left, framework)
        scope = nullcontext() if device_id is None else self.device_scope(device_id, framework)
        with scope:
            return self._operations.logical_and(left, right, framework)

    def available_device_ids(self, module: Any | None = None) -> tuple[int, ...]:
        """Return every framework-local GPU device identifier."""

        if not self.is_gpu:
            return ()
        framework = module if module is not None else self.import_if_installed()
        return () if framework is None else self._runtime.device_ids(framework)

    def require_device(self, device_id: int, module: Any | None = None) -> Any:
        """Return the framework after proving that its local device exists."""

        if not self.is_gpu:
            return module
        framework = module if module is not None else self.import_module()
        available = self.available_device_ids(framework)
        if device_id not in available:
            raise ValueError(
                f"{self.display_name} device {device_id} is unavailable; "
                f"available device IDs are {available}"
            )
        return framework

    def device_id_of(self, data: Any, module: Any | None = None) -> int | None:
        """Return this framework's local device identifier for an array."""

        if not self.is_gpu:
            return None
        framework = module if module is not None else self.import_module()
        return self._runtime.device_id(data, framework)

    def device_scope(
        self,
        device_id: int,
        module: Any | None = None,
    ) -> AbstractContextManager[None]:
        """Return this framework member's scoped device activation leaf."""

        if not self.is_gpu:
            return nullcontext()
        framework = self.require_device(device_id, module)
        return self._runtime.device_scope(framework, device_id)

    def activate_device(self, device_id: int, module: Any | None = None) -> None:
        """Activate a process-global device where the framework supports it."""

        if not self.is_gpu:
            return
        framework = self.require_device(device_id, module)
        self._runtime.activate_device(framework, device_id)

    def move_to_device(
        self,
        data: Any,
        device_id: int,
        module: Any | None = None,
    ) -> Any:
        """Move an array to one framework-local device without leaking selection."""

        if not self.is_gpu:
            return data
        framework = self.require_device(device_id, module)
        if self.device_id_of(data, framework) == device_id:
            return data
        with self.device_scope(device_id, framework):
            return self._runtime.move_to_active_device(data, framework, device_id)

    @property
    def supports_dlpack(self) -> bool:
        """Whether this framework declares a DLPack import leaf."""

        return self._runtime.dlpack_importer is not None

    def supports_dlpack_data(self, data: Any, module: Any | None = None) -> bool:
        """Validate DLPack export for an array owned by this framework."""

        if self._runtime.dlpack_exporter is None:
            return False
        framework = module if module is not None else self.import_module()
        return self._runtime.dlpack_validator(data, framework)

    def export_dlpack(self, data: Any, module: Any | None = None) -> DLPackPayload | None:
        """Export one array through this framework's declaration, if supported."""

        exporter = self._runtime.dlpack_exporter
        if exporter is None:
            return None
        framework = module if module is not None else self.import_module()
        capsule = exporter(data, framework)
        return None if capsule is None else DLPackPayload(source=data, capsule=capsule)

    def from_dlpack(
        self,
        data: DLPackPayload | Any,
        module: Any | None = None,
    ) -> Any:
        """Import DLPack data through this framework's declared leaf."""

        importer = self._runtime.dlpack_importer
        if importer is None:
            raise NotImplementedError(f"DLPack not supported for {self.value}")
        framework = module if module is not None else self.import_module()
        payload = data if isinstance(data, DLPackPayload) else DLPackPayload(data, data)
        return importer(payload, framework)

    def convert_to(self, data: Any, target: "MemoryType", device_id: int) -> Any:
        """Convert one array through the source and target declarations."""

        if self is target:
            return target.move_to_device(data, device_id)

        if target.supports_dlpack:
            try:
                payload = self.export_dlpack(data)
                if payload is not None:
                    module = target.import_module()
                    with target.device_scope(device_id, module):
                        result = target.from_dlpack(payload, module)
                    if result is not NotImplemented:
                        return target.move_to_device(result, device_id, module)
            except Exception as error:
                logger.warning(
                    "DLPack conversion from %s to %s failed: %s. Using CPU roundtrip.",
                    self.value,
                    target.value,
                    error,
                )

        return target.from_numpy(self.to_numpy(data), device_id)

    def current_device_id(self, module: Any | None = None) -> int | None:
        """Return the framework-local process device used for new streams."""

        if not self.is_gpu:
            return None
        framework = module if module is not None else self.import_module()
        return self._runtime.current_device(framework)

    def create_stream(self, module: Any | None = None) -> Any | None:
        """Create a stream on the framework's current device when supported."""

        factory = self._runtime.stream_factory
        if factory is None:
            return None
        framework = module if module is not None else self.import_module()
        return factory(framework)

    def stream_scope(
        self,
        stream: Any,
        module: Any | None = None,
    ) -> AbstractContextManager[None]:
        """Return this framework member's execution scope for one stream."""

        if stream is None:
            return nullcontext()
        framework = module if module is not None else self.import_module()
        return self._runtime.stream_scope(framework, stream)

    def is_oom_error(self, error: BaseException) -> bool:
        """Classify an error without importing an optional framework."""

        return self._runtime.oom_matcher(error, self.loaded_module())

    def cleanup_loaded(self, device_id: int | None = None) -> None:
        """Clean an already-loaded GPU framework without importing absent modules."""

        if not self.is_gpu:
            return
        framework = self.loaded_module()
        if framework is None:
            return
        cleanup = self._runtime.cleanup
        if cleanup is None:
            return
        available = self.available_device_ids(framework)
        if not available:
            return
        targets = available if device_id is None else (device_id,)
        for target in targets:
            with self.device_scope(target, framework):
                cleanup(framework)


# Memory type sets
CPU_MEMORY_TYPES: frozenset[MemoryType] = frozenset(
    memory_type for memory_type in MemoryType if not memory_type.is_gpu
)
GPU_MEMORY_TYPES: frozenset[MemoryType] = frozenset(
    memory_type for memory_type in MemoryType if memory_type.is_gpu
)
SUPPORTED_MEMORY_TYPES: frozenset[MemoryType] = CPU_MEMORY_TYPES | GPU_MEMORY_TYPES

# String value sets for validation
VALID_MEMORY_TYPES = frozenset(mt.value for mt in MemoryType)
VALID_GPU_MEMORY_TYPES = frozenset(mt.value for mt in GPU_MEMORY_TYPES)

# Compatibility constants are generated projections of the enum declaration.
for _memory_type in MemoryType:
    globals()[f"MEMORY_TYPE_{_memory_type.name}"] = _memory_type.value
