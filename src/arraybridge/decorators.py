"""
Memory type declaration decorators.

This module provides decorators for explicitly declaring the memory interface
of pure functions and supporting memory-type-aware dispatching and orchestration.

These decorators annotate functions with input_memory_type and output_memory_type
attributes and provide automatic thread-local CUDA stream management for GPU
frameworks to enable true parallelization across multiple threads.

REFACTORED: Uses enum-driven metaprogramming to eliminate 79% of code duplication.
"""

import functools
import inspect
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, ClassVar, Optional, TypeVar

import numpy as np
from metaclass_registry import AutoRegisterMeta, RegistryFamily, RegistryKeyAttribute

from arraybridge.dtype_scaling import SCALING_FUNCTIONS
from arraybridge.framework_ops import _FRAMEWORK_OPS
from arraybridge.oom_recovery import _execute_with_oom_recovery
from arraybridge.slice_processing import process_slices
from arraybridge.types import MemoryType
from arraybridge.utils import optional_import

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
    def parameter(cls) -> inspect.Parameter:
        return inspect.Parameter(
            cls.require_parameter_name(),
            inspect.Parameter.KEYWORD_ONLY,
            default=cls.default_value(),
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
    original_dtype: Any
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
        return cls.__registry__[dtype_conversion.value]()

    @abstractmethod
    def apply(self, request: DtypeConversionRequest) -> Any:
        """Return output converted according to the configured dtype policy."""


class PreserveInputDtypeConversionRunner(DtypeConversionRunner):
    """Scale output back to the input dtype when the wrapped function changed it."""

    dtype_conversion = DtypeConversion.PRESERVE_INPUT

    def apply(self, request: DtypeConversionRequest) -> Any:
        if (
            request.original_dtype is not None
            and request.array.dtype != request.original_dtype
        ):
            return request.scale_func(request.array, request.original_dtype)
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


@dataclass(frozen=True, slots=True)
class GPUStreamRequest:
    """Runtime context for framework stream selection."""

    thread_context: "ThreadGPUContext"


class GPUStreamStrategy(
    EnumValueRegistryKeyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered stream selector for GPU memory frameworks."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_LABEL)

    memory_type: ClassVar[MemoryType | None] = None

    @classmethod
    def registry_enum_member(cls) -> Enum | None:
        return cls.memory_type

    @classmethod
    def for_memory_type(cls, memory_type: MemoryType) -> "GPUStreamStrategy":
        strategy_type = cls.__registry__.get(memory_type.value, NoGPUStreamStrategy)
        return strategy_type()

    @abstractmethod
    def stream(self, request: GPUStreamRequest) -> Any:
        """Return a context-manager stream for the framework, or None."""


class NoGPUStreamStrategy(GPUStreamStrategy):
    """Frameworks without explicit stream support execute directly."""

    def stream(self, request: GPUStreamRequest) -> Any:
        return None


class CupyGPUStreamStrategy(GPUStreamStrategy):
    memory_type = MemoryType.CUPY

    def stream(self, request: GPUStreamRequest) -> Any:
        return request.thread_context.get_cupy_stream()


class TorchGPUStreamStrategy(GPUStreamStrategy):
    memory_type = MemoryType.TORCH

    def stream(self, request: GPUStreamRequest) -> Any:
        return request.thread_context.get_torch_stream()


# Thread-local cache for lazy-loaded GPU frameworks
_gpu_frameworks_cache = {}


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


def _create_lazy_getter(framework_name: str):
    """Factory function that creates a lazy import getter for a framework."""

    def getter():
        if framework_name not in _gpu_frameworks_cache:
            _gpu_frameworks_cache[framework_name] = optional_import(framework_name)
            if _gpu_frameworks_cache[framework_name] is not None:
                logger.debug(
                    f"🔧 Lazy imported {framework_name} in thread "
                    f"{threading.current_thread().name}"
                )
        return _gpu_frameworks_cache[framework_name]

    return getter


# Auto-generate lazy getters for all GPU frameworks
for mem_type in MemoryType:
    ops = _FRAMEWORK_OPS[mem_type]
    if ops["lazy_getter"] is not None:
        getter_func = _create_lazy_getter(ops["import_name"])
        globals()[f"_get_{ops['import_name']}"] = getter_func


# Thread-local storage for GPU streams and contexts
_thread_gpu_contexts = threading.local()


class ThreadGPUContext:
    """Thread-local GPU context manager for CUDA streams."""

    def __init__(self):
        self.cupy_stream = None
        self.torch_stream = None
        self.tensorflow_device = None
        self.jax_device = None

    def get_cupy_stream(self):
        """Get or create thread-local CuPy stream."""
        if self.cupy_stream is None:
            cupy = globals().get("_get_cupy", lambda: None)()  # noqa: F821
            if cupy is not None and hasattr(cupy, "cuda"):
                self.cupy_stream = cupy.cuda.Stream()
                logger.debug(f"🔧 Created CuPy stream for thread {threading.current_thread().name}")
        return self.cupy_stream

    def get_torch_stream(self):
        """Get or create thread-local PyTorch stream."""
        if self.torch_stream is None:
            torch = globals().get("_get_torch", lambda: None)()  # noqa: F821
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                self.torch_stream = torch.cuda.Stream()
                logger.debug(
                    f"🔧 Created PyTorch stream for thread " f"{threading.current_thread().name}"
                )
        return self.torch_stream


def _get_thread_gpu_context():
    """Get or create thread-local GPU context."""
    if not hasattr(_thread_gpu_contexts, "context"):
        _thread_gpu_contexts.context = ThreadGPUContext()
    return _thread_gpu_contexts.context


def memory_types(
    input_type: str,
    output_type: str,
    contract: Optional[Any] = None,
) -> Callable[[F], F]:
    """
    Base decorator for declaring memory types of a function.

    This is the foundation decorator that all memory-type-specific decorators build upon.
    """

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
        wrapper.input_memory_type = input_type
        wrapper.output_memory_type = output_type
        if contract is not None and not callable(contract):
            wrapper.__processing_contract__ = contract

        return wrapper

    return decorator


def _create_dtype_wrapper(func, mem_type: MemoryType, func_name: str):
    """
    Auto-generate dtype preservation wrapper for any memory type.

    This single function replaces 6 nearly-identical dtype wrapper functions.
    """
    _FRAMEWORK_OPS[mem_type]
    scale_func = SCALING_FUNCTIONS[mem_type.value]

    @functools.wraps(func)
    def dtype_wrapper(image, *args, **kwargs):

        # Pipeline runtimes may inject dtype_config; direct calls use the same
        # preserve-input default explicitly.
        slice_by_slice = kwargs.pop(
            SliceBySliceRuntimeParameter.require_parameter_name(),
            SliceBySliceRuntimeParameter.default_value(),
        )
        dtype_config: DtypeConversionConfig = kwargs.pop(
            DtypeConversionConfig.require_parameter_name(),
            DtypeConversionConfig.default_value(),
        )
        dtype_conversion = dtype_config.default_dtype_conversion

        # Store original dtype
        original_dtype = getattr(image, "dtype", None)

        # Handle slice_by_slice processing for 3D arrays
        if slice_by_slice and hasattr(image, "ndim") and image.ndim == 3:
            result = process_slices(image, func, args, kwargs)
        else:
            # Call the original function normally
            result = func(image, *args, **kwargs)

        def _apply_dtype_conversion(array):
            if not hasattr(array, "dtype"):
                return array
            return DtypeConversionRunner.for_dtype_conversion(dtype_conversion).apply(
                DtypeConversionRequest(
                    array=array,
                    original_dtype=original_dtype,
                    scale_func=scale_func,
                )
            )

        try:
            # Apply dtype conversion to the main output
            if isinstance(result, tuple):
                if not result:
                    return result
                converted_main = _apply_dtype_conversion(result[0])
                return (converted_main, *result[1:])
            return _apply_dtype_conversion(result)
        except Exception as e:
            logger.error(
                f"Error in {mem_type.value} dtype/slice preserving wrapper " f"for {func_name}: {e}"
            )
            # Return unmodified result on conversion errors
            return result

    # Update function signature to include new parameters
    try:
        dtype_signature = KeywordOnlySignatureExtension(
            inspect.signature(func)
        ).with_parameter(SliceBySliceRuntimeParameter.parameter())
        dtype_signature = KeywordOnlySignatureExtension(
            dtype_signature
        ).with_parameter(DtypeConversionConfig.parameter())
        dtype_wrapper.__signature__ = dtype_signature

        # Update docstring
        if dtype_wrapper.__doc__:
            dtype_wrapper.__doc__ += (
                f"\n\n    Additional Parameters " f"(added by {mem_type.value} decorator):\n"
            )
            dtype_wrapper.__doc__ += (
                "        slice_by_slice (bool, optional): " "Process 3D arrays slice-by-slice.\n"
            )
            dtype_wrapper.__doc__ += (
                "            Defaults to False. " "Prevents cross-slice contamination.\n"
            )

    except Exception as e:
        logger.warning(f"Could not update signature for {func_name}: {e}")

    return dtype_wrapper


def _create_gpu_wrapper(func, mem_type: MemoryType, oom_recovery: bool):
    """
    Auto-generate GPU stream/device wrapper for any GPU memory type.

    This function creates the GPU-specific wrapper with stream management and OOM recovery.
    """
    ops = _FRAMEWORK_OPS[mem_type]
    framework_name = ops["import_name"]
    lazy_getter = globals().get(ops["lazy_getter"])

    @functools.wraps(func)
    def gpu_wrapper(*args, **kwargs):
        framework = lazy_getter()

        # Check if GPU is available for this framework
        if framework is not None:
            gpu_check_expr = ops["gpu_check"].format(mod=framework_name)
            try:
                gpu_available = eval(gpu_check_expr, {framework_name: framework})
            except Exception:
                gpu_available = False

            if gpu_available:
                # Get thread-local context
                ctx = _get_thread_gpu_context()

                stream = GPUStreamStrategy.for_memory_type(mem_type).stream(
                    GPUStreamRequest(ctx)
                )

                # Define execution function that captures args/kwargs
                def execute_with_stream():
                    if stream is not None:
                        with stream:
                            return func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)

                # Execute with OOM recovery if enabled
                if oom_recovery and ops["has_oom_recovery"]:
                    return _execute_with_oom_recovery(execute_with_stream, mem_type.value)
                else:
                    return execute_with_stream()

        # CPU fallback or framework not available
        return func(*args, **kwargs)

    # Preserve memory type attributes
    gpu_wrapper.input_memory_type = func.input_memory_type
    gpu_wrapper.output_memory_type = func.output_memory_type

    return gpu_wrapper


def _create_memory_decorator(mem_type: MemoryType):
    """
    Factory function that creates a decorator for a specific memory type.

    This single factory replaces 6 nearly-identical decorator functions.
    """
    ops = _FRAMEWORK_OPS[mem_type]

    def decorator(
        func=None,
        *,
        input_type=mem_type.value,
        output_type=mem_type.value,
        oom_recovery=True,
        contract=None,
    ):
        """
        Decorator for {mem_type} memory type functions.

        Args:
            func: Function to decorate (when used as @decorator)
            input_type: Expected input memory type (default: {mem_type})
            output_type: Expected output memory type (default: {mem_type})
            oom_recovery: Enable automatic OOM recovery (default: True)
            contract: Optional validation function for outputs

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
            func = _create_dtype_wrapper(func, mem_type, func.__name__)

            # Apply GPU wrapper if this is a GPU memory type
            if ops["gpu_check"] is not None:
                func = _create_gpu_wrapper(func, mem_type, oom_recovery)

            return func

        # Handle both @decorator and @decorator() forms
        if func is None:
            return inner_decorator
        return inner_decorator(func)

    # Set proper function name and docstring
    decorator.__name__ = mem_type.value
    decorator.__doc__ = decorator.__doc__.format(mem_type=ops["display_name"])

    return decorator


# Auto-generate all 6 memory type decorators
for mem_type in MemoryType:
    decorator_func = _create_memory_decorator(mem_type)
    globals()[mem_type.value] = decorator_func


# Export all decorators
__all__ = [
    "memory_types",
    "DtypeConversion",
    "DtypeConversionConfig",
    "PreserveInputDtypeConfig",
    "PRESERVE_INPUT_DTYPE_CONFIG",
    "SliceBySliceRuntimeParameter",
    "numpy",  # noqa: F822
    "cupy",  # noqa: F822
    "torch",  # noqa: F822
    "tensorflow",  # noqa: F822
    "jax",  # noqa: F822
    "pyclesperanto",  # noqa: F822
]
