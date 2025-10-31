"""Tests for arraybridge.types module."""

import pytest
from arraybridge.types import (
    MemoryType,
    CPU_MEMORY_TYPES,
    GPU_MEMORY_TYPES,
    SUPPORTED_MEMORY_TYPES,
    VALID_MEMORY_TYPES,
)


class TestMemoryType:
    """Tests for MemoryType enum."""

    def test_memory_type_values(self):
        """Test that MemoryType enum has expected values."""
        assert MemoryType.NUMPY.value == "numpy"
        assert MemoryType.CUPY.value == "cupy"
        assert MemoryType.TORCH.value == "torch"
        assert MemoryType.TENSORFLOW.value == "tensorflow"
        assert MemoryType.JAX.value == "jax"
        assert MemoryType.PYCLESPERANTO.value == "pyclesperanto"

    def test_memory_type_from_string(self):
        """Test creating MemoryType from string."""
        assert MemoryType("numpy") == MemoryType.NUMPY
        assert MemoryType("torch") == MemoryType.TORCH
        assert MemoryType("cupy") == MemoryType.CUPY

    def test_invalid_memory_type_raises_error(self):
        """Test that invalid memory type raises ValueError."""
        with pytest.raises(ValueError):
            MemoryType("invalid_type")

    def test_cpu_memory_types(self):
        """Test that CPU_MEMORY_TYPES contains only NumPy."""
        assert CPU_MEMORY_TYPES == {MemoryType.NUMPY}
        assert len(CPU_MEMORY_TYPES) == 1

    def test_gpu_memory_types(self):
        """Test that GPU_MEMORY_TYPES contains all GPU frameworks."""
        expected_gpu_types = {
            MemoryType.CUPY,
            MemoryType.TORCH,
            MemoryType.TENSORFLOW,
            MemoryType.JAX,
            MemoryType.PYCLESPERANTO,
        }
        assert GPU_MEMORY_TYPES == expected_gpu_types
        assert len(GPU_MEMORY_TYPES) == 5

    def test_supported_memory_types(self):
        """Test that SUPPORTED_MEMORY_TYPES contains all types."""
        assert SUPPORTED_MEMORY_TYPES == CPU_MEMORY_TYPES | GPU_MEMORY_TYPES
        assert len(SUPPORTED_MEMORY_TYPES) == 6

    def test_valid_memory_types_strings(self):
        """Test that VALID_MEMORY_TYPES contains string values."""
        expected_strings = {"numpy", "cupy", "torch", "tensorflow", "jax", "pyclesperanto"}
        assert VALID_MEMORY_TYPES == expected_strings


class TestMemoryTypeConverter:
    """Tests for MemoryType converter property."""

    def test_converter_property_exists(self):
        """Test that converter property exists on MemoryType."""
        # The converter property should exist
        assert hasattr(MemoryType.NUMPY, "converter")

    def test_conversion_methods_exist(self):
        """Test that to_X() methods are added to MemoryType."""
        # Check that conversion methods exist for all memory types
        assert hasattr(MemoryType.NUMPY, "to_numpy")
        assert hasattr(MemoryType.NUMPY, "to_torch")
        assert hasattr(MemoryType.NUMPY, "to_cupy")
        assert hasattr(MemoryType.NUMPY, "to_tensorflow")
        assert hasattr(MemoryType.NUMPY, "to_jax")
        assert hasattr(MemoryType.NUMPY, "to_pyclesperanto")
