"""Tests for arraybridge.stack_utils module."""

import numpy as np
import pytest


class TestStackUtils:
    """Tests for stack utilities functions."""

    def test_is_2d_numpy(self):
        """Test _is_2d with numpy arrays."""
        from arraybridge.stack_utils import _is_2d

        # 2D array
        arr_2d = np.array([[1, 2], [3, 4]])
        assert _is_2d(arr_2d) is True

        # 1D array
        arr_1d = np.array([1, 2, 3])
        assert _is_2d(arr_1d) is False

        # 3D array
        arr_3d = np.array([[[1, 2]], [[3, 4]]])
        assert _is_2d(arr_3d) is False

    def test_is_3d_numpy(self):
        """Test _is_3d with numpy arrays."""
        from arraybridge.stack_utils import _is_3d

        # 3D array
        arr_3d = np.array([[[1, 2]], [[3, 4]]])
        assert _is_3d(arr_3d) is True

        # 2D array
        arr_2d = np.array([[1, 2], [3, 4]])
        assert _is_3d(arr_2d) is False

        # 1D array
        arr_1d = np.array([1, 2, 3])
        assert _is_3d(arr_1d) is False

    def test_enforce_gpu_device_requirements_valid(self):
        """Test _enforce_gpu_device_requirements with valid inputs."""
        from arraybridge.stack_utils import _enforce_gpu_device_requirements

        # CPU memory type should not raise
        _enforce_gpu_device_requirements("numpy", 0)

        # GPU memory type with valid device ID
        _enforce_gpu_device_requirements("torch", 0)
        _enforce_gpu_device_requirements("cupy", 1)

    def test_enforce_gpu_device_requirements_invalid_gpu_id(self):
        """Test _enforce_gpu_device_requirements with invalid GPU device ID."""
        from arraybridge.stack_utils import _enforce_gpu_device_requirements

        with pytest.raises(ValueError) as exc_info:
            _enforce_gpu_device_requirements("torch", -1)
        assert "Invalid GPU device ID" in str(exc_info.value)

    def test_stack_slices_empty_list(self):
        """Test stack_slices with empty list raises error."""
        from arraybridge.stack_utils import stack_slices

        with pytest.raises(ValueError) as exc_info:
            stack_slices([], "numpy", 0)
        assert "Cannot stack empty list" in str(exc_info.value)

    def test_stack_slices_not_2d(self):
        """Test stack_slices with non-2D slices raises error."""
        from arraybridge.stack_utils import stack_slices

        slices = [np.array([1, 2, 3]), np.array([4, 5, 6])]  # 1D arrays

        with pytest.raises(ValueError) as exc_info:
            stack_slices(slices, "numpy", 0)
        assert "not a 2D array" in str(exc_info.value)

    def test_stack_slices_numpy(self):
        """Test stack_slices with numpy arrays."""
        from arraybridge.stack_utils import stack_slices

        slice1 = np.array([[1, 2], [3, 4]])
        slice2 = np.array([[5, 6], [7, 8]])
        slices = [slice1, slice2]

        result = stack_slices(slices, "numpy", 0)

        assert result.shape == (2, 2, 2)
        expected = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        np.testing.assert_array_equal(result, expected)

    def test_stack_slices_single_slice(self):
        """Test stack_slices with single slice."""
        from arraybridge.stack_utils import stack_slices

        slice1 = np.array([[1, 2, 3], [4, 5, 6]])
        result = stack_slices([slice1], "numpy", 0)

        assert result.shape == (1, 2, 3)
        expected = np.array([[[1, 2, 3], [4, 5, 6]]])
        np.testing.assert_array_equal(result, expected)

    def test_unstack_slices_not_3d(self):
        """Test unstack_slices with non-3D array raises error."""
        from arraybridge.stack_utils import unstack_slices

        arr_2d = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError) as exc_info:
            unstack_slices(arr_2d, "numpy", 0)
        assert "Array must be 3D" in str(exc_info.value)

    def test_unstack_slices_numpy(self):
        """Test unstack_slices with numpy array."""
        from arraybridge.stack_utils import unstack_slices

        arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = unstack_slices(arr_3d, "numpy", 0)

        assert len(result) == 2
        assert result[0].shape == (2, 2)
        assert result[1].shape == (2, 2)

        np.testing.assert_array_equal(result[0], np.array([[1, 2], [3, 4]]))
        np.testing.assert_array_equal(result[1], np.array([[5, 6], [7, 8]]))

    def test_unstack_slices_single_slice(self):
        """Test unstack_slices with single slice."""
        from arraybridge.stack_utils import unstack_slices

        arr_3d = np.array([[[1, 2, 3], [4, 5, 6]]])
        result = unstack_slices(arr_3d, "numpy", 0)

        assert len(result) == 1
        assert result[0].shape == (2, 3)
        np.testing.assert_array_equal(result[0], np.array([[1, 2, 3], [4, 5, 6]]))

    def test_unstack_slices_validate_slices_false(self):
        """Test unstack_slices with validate_slices=False."""
        from arraybridge.stack_utils import unstack_slices

        arr_3d = np.array([[[1, 2]], [[3, 4]]])  # Shape: (2, 1, 2)
        result = unstack_slices(arr_3d, "numpy", 0, validate_slices=False)

        assert len(result) == 2
        assert result[0].shape == (1, 2)  # Each slice has shape (1, 2)
        assert result[1].shape == (1, 2)

    @pytest.mark.parametrize("memory_type", ["numpy", "torch", "cupy", "tensorflow", "jax"])
    def test_stack_unstack_roundtrip(self, memory_type):
        """Test roundtrip: stack_slices -> unstack_slices."""
        from arraybridge.stack_utils import stack_slices, unstack_slices

        # Create test slices
        slice1 = np.array([[1, 2], [3, 4]])
        slice2 = np.array([[5, 6], [7, 8]])
        original_slices = [slice1, slice2]

        # Stack them
        stacked = stack_slices(original_slices, "numpy", 0)

        # Unstack them
        unstaked = unstack_slices(stacked, "numpy", 0)

        # Verify roundtrip
        assert len(unstaked) == len(original_slices)
        for original, result in zip(original_slices, unstaked):
            np.testing.assert_array_equal(original, result)
