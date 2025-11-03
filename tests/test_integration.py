"""Integration tests for arraybridge."""

import numpy as np
import pytest

from arraybridge import (
    CPU_MEMORY_TYPES,
    GPU_MEMORY_TYPES,
    MemoryType,
    convert_memory,
    detect_memory_type,
)


class TestBasicWorkflow:
    """Test basic arraybridge workflows."""

    def test_import_all_exports(self):
        """Test that all main exports are importable."""
        from arraybridge import (
            CPU_MEMORY_TYPES,
            GPU_MEMORY_TYPES,
            SUPPORTED_MEMORY_TYPES,
            MemoryConversionError,
            MemoryType,
            convert_memory,
            detect_memory_type,
            memory_types,
        )

        # Verify types exist
        assert MemoryType is not None
        assert CPU_MEMORY_TYPES is not None
        assert GPU_MEMORY_TYPES is not None
        assert SUPPORTED_MEMORY_TYPES is not None

        # Verify functions exist
        assert callable(convert_memory)
        assert callable(detect_memory_type)
        assert callable(memory_types)

        # Verify exception exists
        assert issubclass(MemoryConversionError, Exception)

    def test_simple_numpy_workflow(self):
        """Test a simple workflow with NumPy arrays."""
        # Create array
        data = np.array([[1, 2], [3, 4]])

        # Detect type
        mem_type = detect_memory_type(data)
        assert mem_type == "numpy"

        # Convert (no-op for numpy to numpy)
        result = convert_memory(data, source_type="numpy", target_type="numpy", gpu_id=0)
        np.testing.assert_array_equal(result, data)

    def test_readme_quick_start_example(self):
        """Test the Quick Start example from README."""
        # Create NumPy array
        data = np.array([[1, 2], [3, 4]])

        # Detect memory type
        mem_type = detect_memory_type(data)
        assert mem_type == "numpy"

    def test_multiple_conversions(self):
        """Test multiple sequential conversions."""
        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Multiple numpy->numpy conversions
        result = original
        for _ in range(5):
            result = convert_memory(result, source_type="numpy", target_type="numpy", gpu_id=0)

        np.testing.assert_array_almost_equal(result, original)

    def test_different_array_shapes(self):
        """Test conversion with different array shapes."""
        shapes = [
            (10,),  # 1D
            (10, 10),  # 2D
            (5, 10, 10),  # 3D
            (2, 3, 4, 5),  # 4D
        ]

        for shape in shapes:
            arr = np.random.rand(*shape).astype(np.float32)
            detected = detect_memory_type(arr)
            assert detected == "numpy"

            result = convert_memory(arr, source_type="numpy", target_type="numpy", gpu_id=0)
            assert result.shape == shape


class TestFrameworkAvailability:
    """Test framework availability detection."""

    def test_numpy_always_available(self):
        """Test that NumPy is always available."""
        import numpy

        assert numpy is not None

    def test_optional_framework_import(self):
        """Test optional framework imports."""
        from arraybridge.utils import optional_import

        # NumPy should always work
        np = optional_import("numpy")
        assert np
        assert hasattr(np, "array")

        # Non-existent module should return falsy placeholder
        fake = optional_import("nonexistent_module_12345")
        assert not fake


class TestMemoryTypeConstants:
    """Test memory type constants and sets."""

    def test_cpu_memory_types_contain_only_numpy(self):
        """Test CPU memory types."""
        assert len(CPU_MEMORY_TYPES) == 1
        assert MemoryType.NUMPY in CPU_MEMORY_TYPES

    def test_gpu_memory_types_contain_all_gpu_frameworks(self):
        """Test GPU memory types."""
        expected = {
            MemoryType.CUPY,
            MemoryType.TORCH,
            MemoryType.TENSORFLOW,
            MemoryType.JAX,
            MemoryType.PYCLESPERANTO,
        }
        assert GPU_MEMORY_TYPES == expected

    def test_no_overlap_between_cpu_and_gpu(self):
        """Test that CPU and GPU memory types don't overlap."""
        overlap = CPU_MEMORY_TYPES & GPU_MEMORY_TYPES
        assert len(overlap) == 0


class TestErrorHandling:
    """Test error handling in arraybridge."""

    def test_invalid_source_type_raises_error(self):
        """Test that invalid source type raises error."""
        arr = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            convert_memory(arr, source_type="invalid", target_type="numpy", gpu_id=0)

    def test_invalid_target_type_raises_error(self):
        """Test that invalid target type raises error."""
        arr = np.array([1, 2, 3])
        with pytest.raises((ValueError, AttributeError)):
            convert_memory(arr, source_type="numpy", target_type="invalid", gpu_id=0)

    def test_detect_invalid_type_raises_error(self):
        """Test that detecting invalid type raises error."""
        with pytest.raises(ValueError):
            detect_memory_type("not an array")

    def test_detect_none_raises_error(self):
        """Test that detecting None raises error."""
        with pytest.raises(ValueError):
            detect_memory_type(None)


class TestDtypePreservation:
    """Test that dtypes are preserved during conversions."""

    def test_uint8_dtype_preserved(self):
        """Test uint8 dtype preservation."""
        arr = np.array([0, 128, 255], dtype=np.uint8)
        result = convert_memory(arr, source_type="numpy", target_type="numpy", gpu_id=0)
        assert result.dtype == np.uint8

    def test_uint16_dtype_preserved(self):
        """Test uint16 dtype preservation."""
        arr = np.array([0, 1000, 65535], dtype=np.uint16)
        result = convert_memory(arr, source_type="numpy", target_type="numpy", gpu_id=0)
        assert result.dtype == np.uint16

    def test_float32_dtype_preserved(self):
        """Test float32 dtype preservation."""
        arr = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        result = convert_memory(arr, source_type="numpy", target_type="numpy", gpu_id=0)
        assert result.dtype == np.float32

    def test_float64_dtype_preserved(self):
        """Test float64 dtype preservation."""
        arr = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        result = convert_memory(arr, source_type="numpy", target_type="numpy", gpu_id=0)
        assert result.dtype == np.float64
