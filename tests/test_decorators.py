"""Tests for arraybridge.decorators module."""

import numpy as np
import pytest

from arraybridge.decorators import DtypeConversion, memory_types


class TestDtypeConversion:
    """Tests for DtypeConversion enum."""

    def test_dtype_conversion_enum_values(self):
        """Test all DtypeConversion enum values exist."""
        assert DtypeConversion.PRESERVE_INPUT.value == "preserve"
        assert DtypeConversion.NATIVE_OUTPUT.value == "native"
        assert DtypeConversion.UINT8.value == "uint8"
        assert DtypeConversion.UINT16.value == "uint16"
        assert DtypeConversion.INT16.value == "int16"
        assert DtypeConversion.INT32.value == "int32"
        assert DtypeConversion.FLOAT32.value == "float32"
        assert DtypeConversion.FLOAT64.value == "float64"

    def test_numpy_dtype_property(self):
        """Test numpy_dtype property returns correct dtypes."""
        assert DtypeConversion.UINT8.numpy_dtype == np.uint8
        assert DtypeConversion.UINT16.numpy_dtype == np.uint16
        assert DtypeConversion.INT16.numpy_dtype == np.int16
        assert DtypeConversion.INT32.numpy_dtype == np.int32
        assert DtypeConversion.FLOAT32.numpy_dtype == np.float32
        assert DtypeConversion.FLOAT64.numpy_dtype == np.float64
        assert DtypeConversion.PRESERVE_INPUT.numpy_dtype is None
        assert DtypeConversion.NATIVE_OUTPUT.numpy_dtype is None


class TestMemoryTypesDecorator:
    """Tests for memory_types decorator."""

    def test_memory_types_basic_decoration(self):
        """Test basic memory_types decorator functionality."""

        @memory_types("numpy", "numpy")
        def test_func(x):
            return x * 2

        # Check metadata is attached
        assert hasattr(test_func, "input_memory_type")
        assert hasattr(test_func, "output_memory_type")
        assert test_func.input_memory_type == "numpy"
        assert test_func.output_memory_type == "numpy"

        # Test function still works
        result = test_func(5)
        assert result == 10

    def test_memory_types_with_contract(self):
        """Test memory_types decorator with contract validation."""

        def positive_contract(x):
            return x > 0

        @memory_types("numpy", "numpy", contract=positive_contract)
        def test_func(x):
            return x * 2

        # Valid result
        result = test_func(5)
        assert result == 10

        # Invalid result should raise ValueError
        with pytest.raises(ValueError, match="violated its output contract"):
            test_func(-1)

    def test_memory_types_preserves_function_metadata(self):
        """Test that memory_types preserves function name, docstring, etc."""

        @memory_types("numpy", "numpy")
        def test_func(x, y=10):
            """Test function docstring."""
            return x + y

        assert test_func.__name__ == "test_func"
        assert test_func.__doc__ == "Test function docstring."
        assert test_func(5) == 15
        assert test_func(5, y=20) == 25


class TestFrameworkDecorators:
    """Tests for auto-generated framework-specific decorators."""

    def test_numpy_decorator_exists(self):
        """Test that numpy decorator is available."""
        from arraybridge.decorators import numpy

        assert callable(numpy)

    def test_numpy_decorator_basic(self):
        """Test basic numpy decorator functionality."""
        from arraybridge.decorators import numpy

        @numpy
        def add_one(arr):
            return arr + 1

        # Check metadata
        assert add_one.input_memory_type == "numpy"
        assert add_one.output_memory_type == "numpy"

        # Test with numpy array
        arr = np.array([1, 2, 3])
        result = add_one(arr)
        np.testing.assert_array_equal(result, [2, 3, 4])

    def test_numpy_decorator_dtype_preservation(self):
        """Test numpy decorator preserves input dtype."""
        from arraybridge.decorators import numpy

        @numpy
        def to_float(arr):
            return arr.astype(np.float32)

        # Test with uint8 input
        arr = np.array([0, 127, 255], dtype=np.uint8)
        result = to_float(arr)

        # Should preserve uint8 dtype
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, [0, 127, 255])

    def test_numpy_decorator_dtype_conversion(self):
        """Test numpy decorator with explicit dtype conversion."""
        from arraybridge.decorators import numpy

        @numpy
        def identity(arr):
            return arr

        arr = np.array([0.5, 1.0], dtype=np.float64)
        result = identity(arr, dtype_conversion=DtypeConversion.UINT8)

        # Should convert to uint8
        assert result.dtype == np.uint8
        assert result.shape == arr.shape

    def test_cupy_decorator_exists(self):
        """Test that cupy decorator is available."""
        from arraybridge.decorators import cupy

        assert callable(cupy)

    def test_torch_decorator_exists(self):
        """Test that torch decorator is available."""
        from arraybridge.decorators import torch

        assert callable(torch)

    def test_tensorflow_decorator_exists(self):
        """Test that tensorflow decorator is available."""
        from arraybridge.decorators import tensorflow

        assert callable(tensorflow)

    def test_jax_decorator_exists(self):
        """Test that jax decorator is available."""
        from arraybridge.decorators import jax

        assert callable(jax)

    def test_pyclesperanto_decorator_exists(self):
        """Test that pyclesperanto decorator is available."""
        from arraybridge.decorators import pyclesperanto

        assert callable(pyclesperanto)


class TestDecoratorParameters:
    """Tests for decorator parameter handling."""

    def test_decorator_with_custom_memory_types(self):
        """Test decorator with custom input/output memory types."""
        from arraybridge.decorators import numpy

        @numpy(input_type="torch", output_type="cupy")
        def test_func(x):
            return x

        assert test_func.input_memory_type == "torch"
        assert test_func.output_memory_type == "cupy"

    def test_decorator_with_oom_recovery_disabled(self):
        """Test decorator with OOM recovery disabled."""
        from arraybridge.decorators import numpy

        @numpy(oom_recovery=False)
        def test_func(x):
            return x

        # Function should still work normally
        assert test_func(5) == 5

    def test_slice_by_slice_parameter(self):
        """Test slice_by_slice parameter in function signature."""
        from arraybridge.decorators import numpy

        @numpy
        def process_3d(arr):
            return arr

        # Check that slice_by_slice parameter was added to signature
        import inspect

        sig = inspect.signature(process_3d)
        assert "slice_by_slice" in sig.parameters
        assert "dtype_conversion" in sig.parameters

        # Test with slice_by_slice=False (default)
        arr_3d = np.random.rand(3, 10, 10)
        result = process_3d(arr_3d, slice_by_slice=False)
        assert result.shape == arr_3d.shape
