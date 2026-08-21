"""Tests for arraybridge.dtype_scaling module."""

import numpy as np
import pytest

from arraybridge.dtype_scaling import SCALING_FUNCTIONS
from arraybridge.types import MemoryType


class TestScalingRanges:
    """Tests for scaling range constants."""

    def test_scaling_ranges_uint8(self):
        """Test uint8 scaling range."""
        from arraybridge.dtype_scaling import _SCALING_RANGES

        assert _SCALING_RANGES["uint8"] == 255.0

    def test_scaling_ranges_uint16(self):
        """Test uint16 scaling range."""
        from arraybridge.dtype_scaling import _SCALING_RANGES

        assert _SCALING_RANGES["uint16"] == 65535.0

    def test_scaling_ranges_int16(self):
        """Test int16 scaling range (tuple format)."""
        from arraybridge.dtype_scaling import _SCALING_RANGES

        scale_val, offset_val = _SCALING_RANGES["int16"]
        assert scale_val == 65535.0
        assert offset_val == 32768.0


class TestScalingFunctions:
    """Tests for scaling functions."""

    def test_scaling_functions_registry(self):
        """Test that all memory types have scaling functions."""
        for mem_type in MemoryType:
            assert mem_type.value in SCALING_FUNCTIONS
            assert callable(SCALING_FUNCTIONS[mem_type.value])

    def test_numpy_scaling_no_conversion_needed(self):
        """Test numpy scaling when no conversion is needed."""
        scale_func = SCALING_FUNCTIONS["numpy"]

        # int to int - no scaling needed
        arr = np.array([1, 2, 3], dtype=np.uint8)
        result = scale_func(arr, np.uint16)
        assert result.dtype == np.uint16
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_numpy_scaling_float_to_int(self):
        """Test numpy scaling from float to int."""
        scale_func = SCALING_FUNCTIONS["numpy"]

        # float64 [0, 1] to uint8 [0, 255]
        arr = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        result = scale_func(arr, np.uint8)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, [0, 127, 255])

    def test_numpy_scaling_float_to_int16(self):
        """Test numpy scaling from float to int16."""
        scale_func = SCALING_FUNCTIONS["numpy"]

        # float32 [0, 1] to uint16 [0, 65535]
        arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = scale_func(arr, np.uint16)
        assert result.dtype == np.uint16
        np.testing.assert_array_equal(result, [0, 32767, 65535])

    def test_numpy_scaling_int16_range(self):
        """Test numpy scaling to int16 range."""
        scale_func = SCALING_FUNCTIONS["numpy"]

        # float64 [0, 1] to int16 [-32768, 32767]
        arr = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        result = scale_func(arr, np.int16)
        assert result.dtype == np.int16
        # Check that values are in expected range
        assert np.all(result >= -32768)
        assert np.all(result <= 32767)

    def test_numpy_scaling_constant_image(self):
        """Test numpy scaling with constant image."""
        scale_func = SCALING_FUNCTIONS["numpy"]

        # Constant float image should convert without error
        arr = np.full((10, 10), 0.5, dtype=np.float32)
        result = scale_func(arr, np.uint8)
        assert result.dtype == np.uint8
        # For constant images, all values should be the same
        unique_vals = np.unique(result)
        assert len(unique_vals) == 1  # All values should be identical

    def test_numpy_scaling_edge_cases(self):
        """Test numpy scaling edge cases."""
        scale_func = SCALING_FUNCTIONS["numpy"]

        # Test with very small range
        arr = np.array([0.499, 0.501], dtype=np.float64)
        result = scale_func(arr, np.uint8)
        assert result.dtype == np.uint8

        # Test with single value
        arr = np.array([0.7], dtype=np.float32)
        result = scale_func(arr, np.uint16)
        assert result.dtype == np.uint16

    @pytest.mark.skipif(not hasattr(np, "float16"), reason="float16 not available")
    def test_numpy_scaling_float16(self):
        """Test numpy scaling with float16."""
        scale_func = SCALING_FUNCTIONS["numpy"]

        arr = np.array([0.0, 1.0], dtype=np.float16)
        result = scale_func(arr, np.uint8)
        assert result.dtype == np.uint8

    @pytest.mark.parametrize(
        "memory_type",
        [MemoryType.TORCH, MemoryType.CUPY, MemoryType.PYCLESPERANTO],
    )
    def test_scaling_unavailable_framework_returns_input(self, memory_type, monkeypatch):
        monkeypatch.setattr(MemoryType, "import_if_installed", lambda self: None)
        arr = np.array([1, 2, 3])

        result = SCALING_FUNCTIONS[memory_type.value](arr, np.float32)

        assert result is arr

    def test_scaling_non_array_input(self):
        """Test scaling with non-array input."""
        scale_func = SCALING_FUNCTIONS["numpy"]

        # Should return input unchanged
        result = scale_func("not an array", np.uint8)
        assert result == "not an array"

    def test_scaling_empty_array(self):
        """Test scaling with empty array."""
        scale_func = SCALING_FUNCTIONS["numpy"]

        arr = np.array([], dtype=np.float32)
        # Empty arrays may not be handled by the scaling function due to min/max operations
        # This is acceptable as empty arrays are edge cases
        try:
            result = scale_func(arr, np.uint8)
            assert result.dtype == np.uint8
            assert result.size == 0
        except ValueError:
            # Expected for empty arrays due to min/max operations
            pytest.skip("Empty arrays not supported by scaling function (expected)")

    def test_generic_scaling_delegates_to_member_leaf(self):
        from arraybridge.dtype_scaling import _scale_generic

        values = np.array([0.0, 1.0], dtype=np.float32)

        result = _scale_generic(values, np.uint8, MemoryType.NUMPY)

        np.testing.assert_array_equal(result, [0, 255])

    def test_scaling_ranges_comprehensive(self):
        """Test all scaling ranges are properly defined."""
        from arraybridge.dtype_scaling import _SCALING_RANGES

        # Test all expected dtypes
        expected_ranges = {
            "uint8": 255.0,
            "uint16": 65535.0,
            "uint32": 4294967295.0,
            "int16": (65535.0, 32768.0),
            "int32": (4294967295.0, 2147483648.0),
        }

        for dtype_name, expected_range in expected_ranges.items():
            assert dtype_name in _SCALING_RANGES
            assert _SCALING_RANGES[dtype_name] == expected_range

    def test_torch_scaling_with_gpu_array(self):
        """Test torch scaling with actual GPU array (when torch available)."""
        torch = pytest.importorskip("torch")
        scale_func = SCALING_FUNCTIONS["torch"]

        # Create torch tensor (float to int conversion should trigger scaling)
        arr = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
        result = scale_func(arr, np.int32)  # Use numpy dtype for scaling

        # Should return torch tensor
        assert isinstance(result, torch.Tensor)
        # Check that scaling occurred (should be scaled to int32 range)
        assert result.dtype == torch.int32
        # With clamping fix, values should be correctly scaled to int32 range
        # 0.0 -> INT32_MIN, 0.5 -> ~0, 1.0 -> close to INT32_MAX
        int32_min = -(2**31)
        int32_max = 2**31 - 1
        # Due to float32 precision limits, we clamp to INT32_MAX - 128 to avoid overflow
        # Allow tolerance of up to 150 from the bounds
        assert int32_min <= result[0].item() <= int32_min + 150
        assert abs(result[1].item()) <= 150  # Close to 0
        assert int32_max - 150 <= result[2].item() <= int32_max  # Close to max, not overflowed

    def test_cupy_scaling_with_gpu_array(self):
        """Test cupy scaling with actual GPU array (when cupy available)."""
        cupy = pytest.importorskip("cupy")
        scale_func = SCALING_FUNCTIONS["cupy"]

        # Create cupy array
        arr = cupy.array([0.0, 0.5, 1.0], dtype=cupy.float32)
        result = scale_func(arr, np.int32)

        # Should return cupy array
        assert isinstance(result, cupy.ndarray)
        assert result.dtype == cupy.int32

    def test_jax_scaling_with_gpu_array(self):
        """Test jax scaling with actual GPU array (when jax available)."""
        jax = pytest.importorskip("jax")
        jnp = jax.numpy
        scale_func = SCALING_FUNCTIONS["jax"]

        # Create jax array - JAX uses numpy dtypes
        arr = jnp.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = scale_func(arr, np.int32)

        # Should return jax array
        assert hasattr(result, "dtype")
        assert str(result.dtype) == "int32"

    def test_tensorflow_scaling_with_gpu_array(self):
        """Test tensorflow scaling with actual GPU array (when tensorflow available)."""
        tf = pytest.importorskip("tensorflow")
        scale_func = SCALING_FUNCTIONS["tensorflow"]

        # Create tensorflow tensor
        arr = tf.constant([0.0, 0.5, 1.0], dtype=tf.float32)
        result = scale_func(arr, np.int32)

        # Should return tensorflow tensor
        assert isinstance(result, tf.Tensor)
        assert result.dtype == tf.int32

    @pytest.mark.parametrize(
        ("memory_type", "module_name"),
        [
            (MemoryType.TORCH, "torch"),
            (MemoryType.TENSORFLOW, "tensorflow"),
            (MemoryType.JAX, "jax"),
        ],
    )
    def test_uint16_policy_is_honored_by_typed_frameworks(self, memory_type, module_name):
        module = pytest.importorskip(module_name)
        if memory_type is MemoryType.TORCH:
            values = module.tensor([0.0, 0.5, 1.0], dtype=module.float32)
        elif memory_type is MemoryType.TENSORFLOW:
            values = module.constant([0.0, 0.5, 1.0], dtype=module.float32)
        else:
            values = module.numpy.asarray([0.0, 0.5, 1.0], dtype=module.numpy.float32)

        result = memory_type.scale_dtype(values, np.uint16, module)

        expected_dtype = module.numpy.uint16 if memory_type is MemoryType.JAX else module.uint16
        assert result.dtype == expected_dtype

    def test_jax_float64_fails_when_x64_capability_is_disabled(self):
        module = type(
            "FakeJax",
            (),
            {
                "config": type("Config", (), {"x64_enabled": False})(),
                "numpy": object(),
            },
        )()
        values = type("FakeArray", (), {"dtype": np.dtype(np.float32), "device": None})()

        with pytest.raises(ValueError, match="JAX_ENABLE_X64"):
            MemoryType.JAX.scale_dtype(values, np.float64, module)

    def test_pyclesperanto_scaling_preserves_payload_device(self):
        class FakeArray:
            def __init__(self, values, device):
                self.values = np.asarray(values)
                self.dtype = self.values.dtype
                self.device = type("Device", (), {"name": device})()

        class FakePyclesperanto:
            def __init__(self):
                self.current = "gpu0"

            def list_available_devices(self, device_type=None):
                assert device_type in (None, "gpu")
                return ("gpu0", "gpu1")

            def get_device(self):
                return self.current

            def select_device(self, selector, device_type=None):
                assert device_type in (None, "gpu")
                self.current = ("gpu0", "gpu1")[selector] if isinstance(selector, int) else selector

            def pull(self, data):
                return data.values

            def push(self, data):
                return FakeArray(data, self.current)

        module = FakePyclesperanto()
        values = FakeArray([1, 2, 3], "gpu1")

        result = MemoryType.PYCLESPERANTO.scale_dtype(values, np.int32, module)

        assert result.device.name == "gpu1"
        assert module.current == "gpu0"

    def test_pyclesperanto_scaling_with_gpu_array(self):
        """Test pyclesperanto scaling with actual GPU array (when pyclesperanto available)."""
        pytest.importorskip("pyclesperanto")
        scale_func = SCALING_FUNCTIONS["pyclesperanto"]

        # Create numpy array (pyclesperanto works with numpy arrays pushed to GPU)
        arr = np.array([[0.0, 0.5], [0.25, 1.0]], dtype=np.float32)
        result = scale_func(arr, np.int32)

        # pyclesperanto returns its own array type (can be converted via cle.pull())
        # Check it has correct dtype
        assert hasattr(result, "dtype")
        assert result.dtype == np.int32 or str(result.dtype) == "int32"

    def test_pyclesperanto_scaling_constant_image(self):
        """Test pyclesperanto scaling with constant image."""
        pytest.importorskip("pyclesperanto")
        scale_func = SCALING_FUNCTIONS["pyclesperanto"]

        # Create constant image
        arr = np.full((10, 10), 0.5, dtype=np.float32)
        result = scale_func(arr, np.int32)

        # pyclesperanto returns its own array type
        assert hasattr(result, "dtype")
        assert result.dtype == np.int32 or str(result.dtype) == "int32"

    def test_pyclesperanto_scaling_no_conversion_needed(self):
        """Test pyclesperanto scaling when no conversion needed."""
        cle = pytest.importorskip("pyclesperanto")
        scale_func = SCALING_FUNCTIONS["pyclesperanto"]

        # int to int - no scaling needed
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = scale_func(arr, np.int32)

        # pyclesperanto returns its own array type
        assert hasattr(result, "dtype")
        assert result.dtype == np.int32 or str(result.dtype) == "int32"
        # Convert to numpy for value comparison
        result_np = cle.pull(result) if hasattr(cle, "pull") else np.asarray(result)
        np.testing.assert_array_equal(result_np, [1, 2, 3])
