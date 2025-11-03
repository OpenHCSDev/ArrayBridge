"""Tests for arraybridge.converters module."""

import numpy as np
import pytest

from arraybridge.converters import convert_memory, detect_memory_type
from arraybridge.types import MemoryType


class TestDetectMemoryType:
    """Tests for detect_memory_type function."""

    def test_detect_numpy_array(self):
        """Test detecting NumPy array."""
        arr = np.array([1, 2, 3])
        detected = detect_memory_type(arr)
        assert detected == "numpy"
        assert detected == MemoryType.NUMPY.value

    def test_detect_numpy_2d_array(self):
        """Test detecting 2D NumPy array."""
        arr = np.array([[1, 2], [3, 4]])
        detected = detect_memory_type(arr)
        assert detected == "numpy"

    def test_detect_numpy_3d_array(self):
        """Test detecting 3D NumPy array."""
        arr = np.zeros((5, 10, 10))
        detected = detect_memory_type(arr)
        assert detected == "numpy"

    @pytest.mark.torch
    def test_detect_torch_tensor(self, torch_available):
        """Test detecting PyTorch tensor."""
        if not torch_available:
            pytest.skip("PyTorch not available")

        import torch

        tensor = torch.tensor([1, 2, 3])
        detected = detect_memory_type(tensor)
        assert detected == "torch"
        assert detected == MemoryType.TORCH.value

    def test_detect_unknown_type_raises_error(self):
        """Test that unknown types raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            detect_memory_type([1, 2, 3])  # Plain list

        assert "Unknown memory type" in str(exc_info.value)

    def test_detect_none_raises_error(self):
        """Test that None raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            detect_memory_type(None)

        assert "Unknown memory type" in str(exc_info.value)


class TestConvertMemory:
    """Tests for convert_memory function."""

    def test_convert_numpy_to_numpy(self):
        """Test converting NumPy to NumPy (no-op)."""
        arr = np.array([1, 2, 3, 4, 5])
        result = convert_memory(arr, source_type="numpy", target_type="numpy", gpu_id=0)

        # Should return the same or equivalent array
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_convert_preserves_data(self):
        """Test that conversion preserves data values."""
        arr = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        result = convert_memory(arr, source_type="numpy", target_type="numpy", gpu_id=0)

        np.testing.assert_array_almost_equal(result, arr)

    def test_convert_invalid_source_type(self):
        """Test that invalid source type raises ValueError."""
        arr = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            convert_memory(arr, source_type="invalid_type", target_type="numpy", gpu_id=0)

    def test_convert_invalid_target_type(self):
        """Test that invalid target type raises error."""
        arr = np.array([1, 2, 3])
        # This might raise ValueError or AttributeError depending on implementation
        with pytest.raises((ValueError, AttributeError)):
            convert_memory(arr, source_type="numpy", target_type="invalid_type", gpu_id=0)

    @pytest.mark.torch
    def test_convert_numpy_to_torch(self, torch_available):
        """Test converting NumPy to PyTorch."""
        if not torch_available:
            pytest.skip("PyTorch not available")

        import torch

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = convert_memory(arr, source_type="numpy", target_type="torch", gpu_id=0)

        assert isinstance(result, torch.Tensor)
        np.testing.assert_array_almost_equal(result.cpu().numpy(), arr)

    @pytest.mark.torch
    def test_convert_torch_to_numpy(self, torch_available):
        """Test converting PyTorch to NumPy."""
        if not torch_available:
            pytest.skip("PyTorch not available")

        import torch

        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        result = convert_memory(tensor, source_type="torch", target_type="numpy", gpu_id=0)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, tensor.cpu().numpy())

    def test_convert_2d_arrays(self):
        """Test converting 2D arrays."""
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = convert_memory(arr, source_type="numpy", target_type="numpy", gpu_id=0)

        assert result.shape == arr.shape
        np.testing.assert_array_equal(result, arr)

    def test_convert_3d_arrays(self):
        """Test converting 3D arrays."""
        arr = np.random.rand(5, 10, 10).astype(np.float32)
        result = convert_memory(arr, source_type="numpy", target_type="numpy", gpu_id=0)

        assert result.shape == arr.shape
        np.testing.assert_array_almost_equal(result, arr)

    def test_convert_different_dtypes(self):
        """Test converting arrays with different dtypes."""
        for dtype in [np.uint8, np.uint16, np.int32, np.float32, np.float64]:
            arr = np.array([1, 2, 3], dtype=dtype)
            result = convert_memory(arr, source_type="numpy", target_type="numpy", gpu_id=0)
            assert result.dtype == dtype


class TestConversionIntegration:
    """Integration tests for memory conversion."""

    def test_detect_and_convert_workflow(self):
        """Test detecting type and then converting."""
        arr = np.array([1, 2, 3, 4, 5])

        # Detect the type
        detected_type = detect_memory_type(arr)
        assert detected_type == "numpy"

        # Convert using detected type
        result = convert_memory(arr, source_type=detected_type, target_type="numpy", gpu_id=0)
        np.testing.assert_array_equal(result, arr)

    @pytest.mark.torch
    def test_round_trip_conversion_numpy_torch(self, torch_available):
        """Test round-trip conversion: NumPy -> PyTorch -> NumPy."""
        if not torch_available:
            pytest.skip("PyTorch not available")

        original = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        # NumPy -> PyTorch
        torch_tensor = convert_memory(original, source_type="numpy", target_type="torch", gpu_id=0)

        # PyTorch -> NumPy
        result = convert_memory(torch_tensor, source_type="torch", target_type="numpy", gpu_id=0)

        np.testing.assert_array_almost_equal(result, original)
