"""
arraybridge: Unified API for NumPy, CuPy, PyTorch, TensorFlow, JAX, and pyclesperanto.

This package provides automatic memory type conversion, declarative decorators,
and unified utilities for working with multiple array/tensor frameworks.
"""

__version__ = "0.3.0"

from . import decorators as _decorators
from .array_payload import ArrayPayload
from .converters import convert_memory, detect_memory_type
from .dtype_scaling import SCALING_FUNCTIONS
from .exceptions import MemoryConversionError
from .framework_config import _FRAMEWORK_CONFIG
from .framework_ops import _FRAMEWORK_OPS
from .gpu_cleanup import cleanup_all_gpu_frameworks
from .oom_recovery import _execute_with_oom_recovery
from .slice_processing import process_slices
from .stack_utils import stack_slices, unstack_slices
from .types import (
    CPU_MEMORY_TYPES,
    GPU_MEMORY_TYPES,
    SUPPORTED_MEMORY_TYPES,
    MemoryContractAttribute,
    MemoryType,
)
from .utils import _ensure_module, _get_device_id, _supports_dlpack

DtypeConversion = _decorators.DtypeConversion
SliceBySliceRuntimeParameter = _decorators.SliceBySliceRuntimeParameter
memory_types = _decorators.memory_types
wrap_dtype_preserving_callable = _decorators.wrap_dtype_preserving_callable
for _memory_type in MemoryType:
    globals()[_memory_type.value] = getattr(_decorators, _memory_type.value)

__all__ = [
    # Types
    "MemoryType",
    "MemoryContractAttribute",
    "ArrayPayload",
    "CPU_MEMORY_TYPES",
    "GPU_MEMORY_TYPES",
    "SUPPORTED_MEMORY_TYPES",
    # Converters
    "convert_memory",
    "detect_memory_type",
    # Decorators
    "memory_types",
    "DtypeConversion",
    "SliceBySliceRuntimeParameter",
    "wrap_dtype_preserving_callable",
    # Stack utilities
    "stack_slices",
    "unstack_slices",
    # Slice processing
    "process_slices",
    # GPU cleanup
    "cleanup_all_gpu_frameworks",
    # Exceptions
    "MemoryConversionError",
    # Scaling
    "SCALING_FUNCTIONS",
    # Framework config (internal but needed by some consumers)
    "_FRAMEWORK_CONFIG",
    "_FRAMEWORK_OPS",
    # OOM recovery
    "_execute_with_oom_recovery",
    # Utils
    "_ensure_module",
    "_supports_dlpack",
    "_get_device_id",
] + [memory_type.value for memory_type in MemoryType]
