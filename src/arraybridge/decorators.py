"""
Memory type declaration decorators.

This module provides decorators for explicitly declaring the memory interface
of pure functions and supporting memory-type-aware dispatching and orchestration.

These decorators annotate functions with input_memory_type and output_memory_type
attributes and provide automatic thread-local CUDA stream management for GPU
frameworks to enable true parallelization across multiple threads.

What a library needs around a call (streams, out-of-memory retries) comes from
its ``ArrayLibrary`` class. The decorator named after a library (``numpy``,
``torch``, ...) is made on first access for any registered library.
"""

import functools
import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, ParamSpec, TypeVar

import numpy as np
from metaclass_registry import AutoRegisterMeta, RegistryFamily, RegistryKeyAttribute
from python_introspect import RuntimeParameterDeclarationABC

from arraybridge.array_payload import ArrayPayload
from arraybridge.libraries import Array, ArrayLibrary, DType
from arraybridge.slice_processing import process_slices
from arraybridge.types import MemoryContractAttribute, MemoryType

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


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


class DtypeConversionConfig(RuntimeParameterDeclarationABC):
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
    def parameter(
        cls, *, default_value: "DtypeConversionConfig | None" = None
    ) -> inspect.Parameter:
        return inspect.Parameter(
            cls.require_parameter_name(),
            inspect.Parameter.KEYWORD_ONLY,
            default=cls.default_value() if default_value is None else default_value,
            annotation=cls.annotation_type(),
        )


class SliceBySliceRuntimeParameter(RuntimeParameterDeclarationABC):
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

    def __init_subclass__(cls, **kwargs: object) -> None:
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

    array: Array
    original_dtype_name: str | None
    array_dtype_name: str | None
    scale_func: Callable[[Array, DType], Array]


class DtypeConversionRunner(
    EnumValueRegistryKeyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered dtype conversion behavior selected by DtypeConversion."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.STRATEGY_LABEL)
    # Installed by the metaclass: each runner class under its conversion's value.
    __registry__: ClassVar["Mapping[str, type[DtypeConversionRunner]]"]

    dtype_conversion: ClassVar[DtypeConversion | None] = None

    @classmethod
    def registry_enum_member(cls) -> Enum | None:
        return cls.dtype_conversion

    @classmethod
    def for_dtype_conversion(
        cls,
        dtype_conversion: DtypeConversion,
    ) -> "DtypeConversionRunner":
        runner: DtypeConversionRunner = cls.__registry__[dtype_conversion.value]()
        return runner

    @abstractmethod
    def apply(self, request: DtypeConversionRequest) -> Array:
        """Return output converted according to the configured dtype policy."""


class PreserveInputDtypeConversionRunner(DtypeConversionRunner):
    """Scale output back to the input dtype when the wrapped function changed it."""

    dtype_conversion = DtypeConversion.PRESERVE_INPUT

    def apply(self, request: DtypeConversionRequest) -> Array:
        if (
            request.original_dtype_name is not None
            and request.array_dtype_name != request.original_dtype_name
        ):
            return request.scale_func(request.array, request.original_dtype_name)
        return request.array


class NativeOutputDtypeConversionRunner(DtypeConversionRunner):
    """Keep the wrapped framework function's native output dtype."""

    dtype_conversion = DtypeConversion.NATIVE_OUTPUT

    def apply(self, request: DtypeConversionRequest) -> Array:
        return request.array


class FixedDtypeConversionRunner(DtypeConversionRunner):
    """Scale output to the dtype declared by a fixed DtypeConversion member."""

    def apply(self, request: DtypeConversionRequest) -> Array:
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


def memory_types(
    input_type: str | MemoryType,
    output_type: str | MemoryType,
    contract: object | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Base decorator for declaring memory types of a function.

    This is the foundation decorator that all memory-type-specific decorators build upon.
    """

    input_member = input_type if isinstance(input_type, MemoryType) else MemoryType(input_type)
    output_member = output_type if isinstance(output_type, MemoryType) else MemoryType(output_type)
    input_memory_type = input_member.value
    output_memory_type = output_member.value

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
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

        return wrapper

    return decorator


def wrap_dtype_preserving_callable(
    func,
    mem_type: MemoryType,
    *,
    slice_by_slice_default: bool = False,
    dtype_config_default: DtypeConversionConfig | None = None,
):
    """
    Return a callable with ArrayBridge dtype and slice controls.

    Host registries can use this public boundary without depending on the
    complete framework decorator or ArrayBridge internals.
    """
    func_name = func.__name__
    input_library = MemoryType(MemoryContractAttribute.INPUT.read(func, mem_type.value)).library
    output_library = MemoryType(MemoryContractAttribute.OUTPUT.read(func, mem_type.value)).library
    scale_func = output_library.scale_dtype
    default_dtype_config = (
        DtypeConversionConfig.default_value()
        if dtype_config_default is None
        else dtype_config_default
    )
    if not isinstance(default_dtype_config, DtypeConversionConfig):
        raise TypeError("Callable dtype default must be a DtypeConversionConfig.")

    @functools.wraps(func)
    def dtype_wrapper(image, *args, **kwargs):
        # Pipeline runtimes may inject dtype_config; direct calls use the same
        # callable-owned default explicitly (preserve-input when undeclared).
        slice_by_slice = kwargs.pop(
            SliceBySliceRuntimeParameter.require_parameter_name(),
            slice_by_slice_default,
        )
        dtype_config: DtypeConversionConfig = kwargs.pop(
            DtypeConversionConfig.require_parameter_name(),
            default_dtype_config,
        )
        dtype_conversion = dtype_config.default_dtype_conversion

        # Store original dtype
        original_dtype = getattr(image, "dtype", None)
        original_dtype_name = (
            None if original_dtype is None else input_library.dtype_name(original_dtype)
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
                    array_dtype_name=output_library.dtype_name(array.dtype),
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
            DtypeConversionConfig.parameter(default_value=default_dtype_config)
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


def _create_memory_decorator(mem_type: MemoryType):
    """
    Factory function that creates a decorator for a specific memory type.

    """

    def decorator(
        func=None,
        *,
        input_type=mem_type.value,
        output_type=mem_type.value,
        oom_recovery=True,
        contract=None,
        slice_by_slice_default=False,
        dtype_config_default: DtypeConversionConfig | None = None,
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
            dtype_config_default: Callable-owned dtype policy for direct calls;
                explicit runtime dtype_config still overrides this default.

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
                dtype_config_default=dtype_config_default,
            )

            # What the library's devices need around the call (streams, OOM retries).
            func = mem_type.library.execution_wrapper(func, oom_recovery)

            MemoryContractAttribute.EXECUTION.write(func, mem_type.value)

            return func

        # Handle both @decorator and @decorator() forms
        if func is None:
            return inner_decorator
        return inner_decorator(func)

    # Set proper function name and docstring
    decorator.__name__ = mem_type.value
    decorator.__doc__ = (decorator.__doc__ or "").format(mem_type=mem_type.library.display_name)

    return decorator


def __getattr__(name: str):
    """The decorator named after a registered library, made on first access."""

    if name.startswith("_") or name not in ArrayLibrary.__registry__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    decorator = _create_memory_decorator(MemoryType(name))
    globals()[name] = decorator
    return decorator


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(ArrayLibrary.__registry__))


__all__ = [
    "memory_types",
    "DtypeConversion",
    "DtypeConversionConfig",
    "PreserveInputDtypeConfig",
    "PRESERVE_INPUT_DTYPE_CONFIG",
    "SliceBySliceRuntimeParameter",
    "wrap_dtype_preserving_callable",
]
