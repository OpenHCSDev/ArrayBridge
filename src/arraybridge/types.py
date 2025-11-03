"""
Memory type definitions for arraybridge.

This module defines the MemoryType enum and related constants for managing
different array/tensor frameworks.
"""

from enum import Enum
from typing import Any, Callable, TypeVar

T = TypeVar("T")
ConversionFunc = Callable[[Any], Any]


class MemoryType(Enum):
    """Enum representing different array/tensor framework types."""

    NUMPY = "numpy"
    CUPY = "cupy"
    TORCH = "torch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    PYCLESPERANTO = "pyclesperanto"

    @property
    def converter(self):
        """Get the converter instance for this memory type."""
        from arraybridge.converters_registry import get_converter

        return get_converter(self.value)


# Auto-generate to_X() methods on enum
def _add_conversion_methods():
    """Add to_X() conversion methods to MemoryType enum."""
    for target_type in MemoryType:
        method_name = f"to_{target_type.value}"

        def make_method(target):
            def method(self, data, gpu_id):
                return getattr(self.converter, f"to_{target.value}")(data, gpu_id)

            return method

        setattr(MemoryType, method_name, make_method(target_type))


_add_conversion_methods()


# Memory type sets
CPU_MEMORY_TYPES: set[MemoryType] = {MemoryType.NUMPY}
GPU_MEMORY_TYPES: set[MemoryType] = {
    MemoryType.CUPY,
    MemoryType.TORCH,
    MemoryType.TENSORFLOW,
    MemoryType.JAX,
    MemoryType.PYCLESPERANTO,
}
SUPPORTED_MEMORY_TYPES: set[MemoryType] = CPU_MEMORY_TYPES | GPU_MEMORY_TYPES

# String value sets for validation
VALID_MEMORY_TYPES = {mt.value for mt in MemoryType}
VALID_GPU_MEMORY_TYPES = {mt.value for mt in GPU_MEMORY_TYPES}

# Memory type constants for direct access
MEMORY_TYPE_NUMPY = MemoryType.NUMPY.value
MEMORY_TYPE_CUPY = MemoryType.CUPY.value
MEMORY_TYPE_TORCH = MemoryType.TORCH.value
MEMORY_TYPE_TENSORFLOW = MemoryType.TENSORFLOW.value
MEMORY_TYPE_JAX = MemoryType.JAX.value
MEMORY_TYPE_PYCLESPERANTO = MemoryType.PYCLESPERANTO.value
