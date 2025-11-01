"""
arraybridge: Unified API for NumPy, CuPy, PyTorch, TensorFlow, JAX, and pyclesperanto.

This package provides automatic memory type conversion, declarative decorators,
and unified utilities for working with multiple array/tensor frameworks.
"""

__version__ = "0.1.0"

from .converters import convert_memory, detect_memory_type
from .decorators import cupy, jax, memory_types, numpy, tensorflow, torch
from .exceptions import MemoryConversionError
from .stack_utils import stack_slices, unstack_slices
from .types import CPU_MEMORY_TYPES, GPU_MEMORY_TYPES, SUPPORTED_MEMORY_TYPES, MemoryType

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
    # Stack utilities
    "stack_slices",
    "unstack_slices",
    # Exceptions
    "MemoryConversionError",
]
