"""
Memory type declaration decorators.

This module provides decorators for explicitly declaring the memory interface
of pure functions and supporting memory-type-aware dispatching and orchestration.

These decorators annotate functions with input_memory_type and output_memory_type
attributes and provide automatic thread-local CUDA stream management for GPU
frameworks to enable true parallelization across multiple threads.

Framework-specific runtime capabilities are delegated to ``MemoryType`` members.
"""

import functools
import inspect
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, TypeVar, cast

import numpy as np
from metaclass_registry import AutoRegisterMeta, RegistryFamily, RegistryKeyAttribute

from arraybridge.array_payload import ArrayPayload
from arraybridge.oom_recovery import _execute_with_oom_recovery
from arraybridge.slice_processing import process_slices
from arraybridge.types import MemoryContractAttribute, MemoryType

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class DtypeConversion(Enum):
    """Data type conversion modes for all memory type functions."""

    PRESERVE_INPUT = "preserve"  # Keep input dtype (default)
    NATIVE_OUTPUT = "native"  # Use framework's native output
    UINT8 = "uint8"  # Force uint8 (0-255 range)
    UINT16 = "uint16"  # Force uint16 (microscopy standard)
    INT16 = "int16"  # Force int16 (signed microscopy data)
    INT32 = "int32"  # Force int32 (large integer values)
    FLOAT32 = "float32"  # Force float32 (GPU performance)
    FLOAT64 = "float64"  # Force float64 (maximum precision)

    @property
    def numpy_dtype(self):
        """Get the corresponding numpy dtype."""
        dtype_map = {
            self.UINT8: np.uint8,
            self.UINT16: np.uint16,
            self.INT16: np.int16,
            self.INT32: np.int32,
            self.FLOAT32: np.float32,
            self.FLOAT64: np.float64,
        }
        return dtype_map.get(self, None)


class DtypeConversionConfig(ABC):
    """Nominal dtype conversion config surface consumed by decorators."""

    @property
    @abstractmethod
    def default_dtype_conversion(self) -> DtypeConversion:
        """Return the dtype conversion mode for decorated function output."""

    @classmethod
    def require_parameter_name(cls) -> str:
        return "dtype_config"

    @classmethod
    def default_value(cls):
        return PRESERVE_INPUT_DTYPE_CONFIG

    @classmethod
    def annotation_type(cls):
        return DtypeConversionConfig

    @classmethod
    def parameter(cls) -> inspect.Parameter:
        return inspect.Parameter(
            cls.require_parameter_name(),
            inspect.Parameter.KEYWORD_ONLY,
            default=cls.default_value(),
            annotation=cls.annotation_type(),
        )


class SliceBySliceRuntimeParameter:
    """Nominal slice-by-slice execution parameter consumed by decorators."""

    preserve_for_execution = True
    is_semantic_control = True

    @classmethod
    def require_parameter_name(cls) -> str:
        return "slice_by_slice"

    @classmethod
    def default_value(cls) -> bool:
        return False

    @classmethod
    def annotation_type(cls) -> type[bool]:
        return bool

    @classmethod
    def parameter(cls, *, default_value: bool | None = None) -> inspect.Parameter:
        return inspect.Parameter(
            cls.require_parameter_name(),
            inspect.Parameter.KEYWORD_ONLY,
            default=(cls.default_value() if default_value is None else default_value),
            annotation=cls.annotation_type(),
        )


@dataclass(frozen=True, slots=True)
class PreserveInputDtypeConfig(DtypeConversionConfig):
    """Direct-call dtype config for wrappers executed outside a pipeline runtime."""

    default_dtype_conversion: DtypeConversion = DtypeConversion.PRESERVE_INPUT


PRESERVE_INPUT_DTYPE_CONFIG = PreserveInputDtypeConfig()


class EnumValueRegistryKeyMixin:
    """Derive AutoRegisterMeta strategy labels from enum-valued class members."""

    strategy_label: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        member = cls.registry_enum_member()
        if isinstance(member, Enum) and cls.__dict__.get("strategy_label") is None:
            cls.strategy_label = member.value

    @classmethod
    @abstractmethod
    def registry_enum_member(cls) -> Enum | None:
        """Return the enum member that should key this concrete strategy."""


@dataclass(frozen=True, slots=True)
class DtypeConversionRequest:
    """Runtime data needed to convert one decorated function output."""

    array: Any
    original_dtype_name: str | None
    array_dtype_name: str | None
    scale_func: Callable[[Any, Any], Any]


class DtypeConversionRunner(
    EnumValueRegistryKeyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered dtype conversion behavior selected by DtypeConversion."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_LABEL)

    dtype_conversion: ClassVar[DtypeConversion | None] = None

    @classmethod
    def registry_enum_member(cls) -> Enum | None:
        return cls.dtype_conversion

    @classmethod
    def for_dtype_conversion(
        cls,
        dtype_conversion: DtypeConversion,
    ) -> "DtypeConversionRunner":
        return cast(DtypeConversionRunner, cls.__registry__[dtype_conversion.value]())

    @abstractmethod
    def apply(self, request: DtypeConversionRequest) -> Any:
        """Return output converted according to the configured dtype policy."""


class PreserveInputDtypeConversionRunner(DtypeConversionRunner):
    """Scale output back to the input dtype when the wrapped function changed it."""

    dtype_conversion = DtypeConversion.PRESERVE_INPUT

    def apply(self, request: DtypeConversionRequest) -> Any:
        if (
            request.original_dtype_name is not None
            and request.array_dtype_name != request.original_dtype_name
        ):
            return request.scale_func(request.array, request.original_dtype_name)
        return request.array


class NativeOutputDtypeConversionRunner(DtypeConversionRunner):
    """Keep the wrapped framework function's native output dtype."""

    dtype_conversion = DtypeConversion.NATIVE_OUTPUT

    def apply(self, request: DtypeConversionRequest) -> Any:
        return request.array


class FixedDtypeConversionRunner(DtypeConversionRunner):
    """Scale output to the dtype declared by a fixed DtypeConversion member."""

    def apply(self, request: DtypeConversionRequest) -> Any:
        if self.dtype_conversion is None:
            raise TypeError("FixedDtypeConversionRunner requires dtype_conversion.")
        target_dtype = self.dtype_conversion.numpy_dtype
        if target_dtype is None:
            return request.array
        return request.scale_func(request.array, target_dtype)


class Uint8DtypeConversionRunner(FixedDtypeConversionRunner):
    dtype_conversion = DtypeConversion.UINT8


class Uint16DtypeConversionRunner(FixedDtypeConversionRunner):
    dtype_conversion = DtypeConversion.UINT16


class Int16DtypeConversionRunner(FixedDtypeConversionRunner):
    dtype_conversion = DtypeConversion.INT16


class Int32DtypeConversionRunner(FixedDtypeConversionRunner):
    dtype_conversion = DtypeConversion.INT32


class Float32DtypeConversionRunner(FixedDtypeConversionRunner):
    dtype_conversion = DtypeConversion.FLOAT32


class Float64DtypeConversionRunner(FixedDtypeConversionRunner):
    dtype_conversion = DtypeConversion.FLOAT64


class KeywordOnlySignatureExtension:
    """Insert decorator-owned keyword-only parameters in valid signature order."""

    def __init__(self, signature: inspect.Signature):
        self.signature = signature

    def with_parameter(self, parameter: inspect.Parameter) -> inspect.Signature:
        parameters = list(self.signature.parameters.values())
        if parameter.name in self.signature.parameters:
            return self.signature
        insertion_index = self._insertion_index(parameters)
        parameters.insert(insertion_index, parameter)
        return self.signature.replace(parameters=parameters)

    @staticmethod
    def _insertion_index(parameters: list[inspect.Parameter]) -> int:
        for index, candidate in enumerate(parameters):
            if candidate.kind is inspect.Parameter.VAR_KEYWORD:
                return index
        return len(parameters)


# Thread-local storage for GPU streams and contexts
_thread_gpu_contexts = threading.local()


class ThreadGPUContext:
    """Thread-local streams keyed by framework-local device identity."""

    def __init__(self):
        self._streams: dict[tuple[MemoryType, int], Any] = {}

    def stream_for(
        self,
        memory_type: MemoryType,
        module: Any,
    ) -> tuple[int | None, Any | None]:
        """Return the current device and its stable thread-local stream."""

        device_id = memory_type.current_device_id(module)
        if device_id is None:
            return None, None
        memory_type.require_device(device_id, module)
        key = (memory_type, device_id)
        if key not in self._streams:
            with memory_type.device_scope(device_id, module):
                stream = memory_type.create_stream(module)
            if stream is None:
                return device_id, None
            self._streams[key] = stream
            logger.debug(
                "Created %s stream for device %d in thread %s",
                memory_type.display_name,
                device_id,
                threading.current_thread().name,
            )
        return device_id, self._streams[key]


def _get_thread_gpu_context():
    """Get or create thread-local GPU context."""
    if not hasattr(_thread_gpu_contexts, "context"):
        _thread_gpu_contexts.context = ThreadGPUContext()
    return _thread_gpu_contexts.context


def memory_types(
    input_type: str | MemoryType,
    output_type: str | MemoryType,
    contract: Any | None = None,
) -> Callable[[F], F]:
    """
    Base decorator for declaring memory types of a function.

    This is the foundation decorator that all memory-type-specific decorators build upon.
    """

    input_member = input_type if isinstance(input_type, MemoryType) else MemoryType(input_type)
    output_member = output_type if isinstance(output_type, MemoryType) else MemoryType(output_type)
    input_memory_type = input_member.value
    output_memory_type = output_member.value

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            # Apply output validation only when a callable contract was provided.
            # Non-callable contracts are declarative metadata consumed by runtimes.
            if callable(contract) and not contract(result):
                raise ValueError(f"Function {func.__name__} violated its output contract")

            return result

        # Attach memory type metadata
        MemoryContractAttribute.INPUT.write(wrapper, input_memory_type)
        MemoryContractAttribute.OUTPUT.write(wrapper, output_memory_type)
        if contract is not None and not callable(contract):
            setattr(wrapper, "__processing_contract__", contract)

        return cast(F, wrapper)

    return decorator


def wrap_dtype_preserving_callable(
    func,
    mem_type: MemoryType,
    *,
    slice_by_slice_default: bool = False,
):
    """
    Return a callable with ArrayBridge dtype and slice controls.

    Host registries can use this public boundary without depending on the
    complete framework decorator or ArrayBridge internals.
    """
    func_name = func.__name__
    input_memory_type = MemoryType(MemoryContractAttribute.INPUT.read(func, mem_type.value))
    output_memory_type = MemoryType(MemoryContractAttribute.OUTPUT.read(func, mem_type.value))
    scale_func = output_memory_type.scale_dtype

    @functools.wraps(func)
    def dtype_wrapper(image, *args, **kwargs):
        # Pipeline runtimes may inject dtype_config; direct calls use the same
        # preserve-input default explicitly.
        slice_by_slice = kwargs.pop(
            SliceBySliceRuntimeParameter.require_parameter_name(),
            slice_by_slice_default,
        )
        dtype_config: DtypeConversionConfig = kwargs.pop(
            DtypeConversionConfig.require_parameter_name(),
            DtypeConversionConfig.default_value(),
        )
        dtype_conversion = dtype_config.default_dtype_conversion

        # Store original dtype
        original_dtype = getattr(image, "dtype", None)
        original_dtype_name = (
            None
            if original_dtype is None
            else input_memory_type.canonical_dtype_name(original_dtype)
        )

        # Handle slice_by_slice processing for 3D arrays
        if slice_by_slice and hasattr(image, "ndim") and image.ndim == 3:
            result = process_slices(image, func, args, kwargs)
        else:
            # Call the original function normally
            result = func(image, *args, **kwargs)

        def _apply_dtype_conversion(array):
            if isinstance(array, ArrayPayload):
                return array.map_array_payload(_apply_dtype_conversion)
            if not hasattr(array, "dtype"):
                return array
            return DtypeConversionRunner.for_dtype_conversion(dtype_conversion).apply(
                DtypeConversionRequest(
                    array=array,
                    original_dtype_name=original_dtype_name,
                    array_dtype_name=output_memory_type.canonical_dtype_name(array.dtype),
                    scale_func=scale_func,
                )
            )

        # Apply dtype conversion to the main output. Conversion errors are
        # contract violations and must remain visible to the caller.
        if isinstance(result, tuple):
            if not result:
                return result
            converted_main = _apply_dtype_conversion(result[0])
            return (converted_main, *result[1:])
        return _apply_dtype_conversion(result)

    # Update function signature to include new parameters
    try:
        dtype_signature = KeywordOnlySignatureExtension(inspect.signature(func)).with_parameter(
            SliceBySliceRuntimeParameter.parameter(
                default_value=slice_by_slice_default,
            )
        )
        dtype_signature = KeywordOnlySignatureExtension(dtype_signature).with_parameter(
            DtypeConversionConfig.parameter()
        )
        setattr(dtype_wrapper, "__signature__", dtype_signature)

        # Update docstring
        if dtype_wrapper.__doc__:
            dtype_wrapper.__doc__ += "\n\n    Additional Parameters\n    ---------------------\n"
            dtype_wrapper.__doc__ += (
                "        slice_by_slice : bool, optional\n"
                f"            Added by the {mem_type.value} memory decorator. "
                "Process 3D arrays slice-by-slice.\n"
            )
            dtype_wrapper.__doc__ += (
                f"            Defaults to {slice_by_slice_default}. "
                "Prevents cross-slice contamination when enabled.\n"
            )

    except Exception as e:
        logger.warning(f"Could not update signature for {func_name}: {e}")

    return dtype_wrapper


def _create_gpu_wrapper(func, mem_type: MemoryType, oom_recovery: bool):
    """
    Auto-generate GPU stream/device wrapper for any GPU memory type.

    This function creates the GPU-specific wrapper with stream management and OOM recovery.
    """

    @functools.wraps(func)
    def gpu_wrapper(*args, **kwargs):
        framework = mem_type.import_if_installed()

        # Check if GPU is available for this framework
        if framework is not None and mem_type.available_device_ids(framework):
            # Get thread-local context
            ctx = _get_thread_gpu_context()

            device_id, stream = ctx.stream_for(mem_type, framework)

            # Define execution function that captures args/kwargs
            def execute_with_stream():
                with mem_type.stream_scope(stream, framework):
                    return func(*args, **kwargs)

            # Execute with OOM recovery if enabled
            if oom_recovery:
                return _execute_with_oom_recovery(
                    execute_with_stream,
                    mem_type.value,
                    device_id=device_id,
                )
            return execute_with_stream()

        # CPU fallback or framework not available
        return func(*args, **kwargs)

    # Preserve memory type attributes
    MemoryContractAttribute.INPUT.write(
        gpu_wrapper,
        MemoryContractAttribute.INPUT.read(func),
    )
    MemoryContractAttribute.OUTPUT.write(
        gpu_wrapper,
        MemoryContractAttribute.OUTPUT.read(func),
    )

    return gpu_wrapper


def _create_memory_decorator(mem_type: MemoryType):
    """
    Factory function that creates a decorator for a specific memory type.

    This single factory replaces 6 nearly-identical decorator functions.
    """

    def decorator(
        func=None,
        *,
        input_type=mem_type.value,
        output_type=mem_type.value,
        oom_recovery=True,
        contract=None,
        slice_by_slice_default=False,
    ):
        """
        Decorator for {mem_type} memory type functions.

        Args:
            func: Function to decorate (when used as @decorator)
            input_type: Expected input memory type (default: {mem_type})
            output_type: Expected output memory type (default: {mem_type})
            oom_recovery: Enable automatic OOM recovery (default: True)
            contract: Optional validation function for outputs
            slice_by_slice_default: Default for the decorator-owned slice control

        Returns:
            Decorated function with memory type metadata and dtype preservation
        """

        def inner_decorator(func):
            # Apply base memory_types decorator
            memory_decorator = memory_types(
                input_type=input_type, output_type=output_type, contract=contract
            )
            func = memory_decorator(func)

            # Apply dtype preservation wrapper
            func = wrap_dtype_preserving_callable(
                func,
                mem_type,
                slice_by_slice_default=slice_by_slice_default,
            )

            # Apply GPU wrapper if this is a GPU memory type
            if mem_type.is_gpu:
                func = _create_gpu_wrapper(func, mem_type, oom_recovery)

            MemoryContractAttribute.EXECUTION.write(func, mem_type.value)

            return func

        # Handle both @decorator and @decorator() forms
        if func is None:
            return inner_decorator
        return inner_decorator(func)

    # Set proper function name and docstring
    decorator.__name__ = mem_type.value
    decorator.__doc__ = (decorator.__doc__ or "").format(mem_type=mem_type.display_name)

    return decorator


# Auto-generate all 6 memory type decorators
for mem_type in MemoryType:
    decorator_func = _create_memory_decorator(mem_type)
    globals()[mem_type.value] = decorator_func


# Export the fixed decorator infrastructure plus declaration-derived helpers.
__all__ = [
    "memory_types",
    "DtypeConversion",
    "DtypeConversionConfig",
    "PreserveInputDtypeConfig",
    "PRESERVE_INPUT_DTYPE_CONFIG",
    "SliceBySliceRuntimeParameter",
    "wrap_dtype_preserving_callable",
] + [memory_type.value for memory_type in MemoryType]
