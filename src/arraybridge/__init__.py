"""
arraybridge: Unified API for NumPy, CuPy, PyTorch, TensorFlow, JAX, and pyclesperanto.

This package provides automatic memory type conversion, declarative decorators,
and unified utilities for working with multiple array/tensor frameworks.
"""

__version__ = "0.2.1"

from .converters import convert_memory, detect_memory_type
from .decorators import (
    DtypeConversion,
    cupy,
    jax,
    memory_types,
    numpy,
    pyclesperanto,
    tensorflow,
    torch,
)
from .dtype_scaling import SCALING_FUNCTIONS
from .exceptions import MemoryConversionError
from .framework_config import _FRAMEWORK_CONFIG
from .framework_ops import _FRAMEWORK_OPS
from .gpu_cleanup import cleanup_all_gpu_frameworks
from .oom_recovery import _execute_with_oom_recovery
from .slice_processing import process_slices
from .stack_utils import stack_slices, unstack_slices
from .types import CPU_MEMORY_TYPES, GPU_MEMORY_TYPES, SUPPORTED_MEMORY_TYPES, MemoryType
from .utils import _ensure_module, _get_device_id, _supports_dlpack

__all__ = [
    # Types
    "MemoryType",
    "CPU_MEMORY_TYPES",
    "GPU_MEMORY_TYPES",
    "SUPPORTED_MEMORY_TYPES",
    # Converters
    "convert_memory",
    "detect_memory_type",
    # Decorators
    "memory_types",
    "numpy",
    "cupy",
    "torch",
    "tensorflow",
    "jax",
    "pyclesperanto",
    "DtypeConversion",
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
]
