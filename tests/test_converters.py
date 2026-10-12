"""``convert_memory`` and ``detect_memory_type``: the name-based conversion functions."""

import sys

import numpy as np
import pytest

from arraybridge.converters import convert_memory, detect_memory_type
from arraybridge.devices import HOST, AcceleratorDevice
from arraybridge.exceptions import MemoryConversionError
from arraybridge.types import MemoryType
from tests.fakes import FakeTorch, FakeTorchTensor, fake_jax


class TestDetectMemoryType:
    @pytest.mark.parametrize("shape", [(3,), (2, 2), (5, 10, 10), (2, 3, 4, 5)])
    def test_numpy_array_of_any_rank(self, shape):
        assert detect_memory_type(np.zeros(shape)) == "numpy" == MemoryType.NUMPY.value

    def test_array_of_a_loaded_library(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", FakeTorch())

        assert detect_memory_type(FakeTorchTensor([1, 2, 3])) == "torch"

    @pytest.mark.torch
    def test_installed_torch_tensor(self):
        torch = pytest.importorskip("torch")

        assert detect_memory_type(torch.tensor([1, 2, 3])) == "torch"

    @pytest.mark.parametrize("value", [[1, 2, 3], None, "not an array"])
    def test_value_of_no_library_is_rejected(self, value):
        with pytest.raises(ValueError, match="Unknown memory type"):
            detect_memory_type(value)


class TestConvertMemory:
    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.float32, np.float64])
    @pytest.mark.parametrize("shape", [(10,), (10, 10), (5, 10, 10), (2, 3, 4, 5)])
    def test_numpy_to_numpy_returns_the_array(self, dtype, shape):
        values = np.random.rand(*shape).astype(dtype)

        assert convert_memory(values, "numpy", "numpy", HOST) is values
        assert convert_memory(values, MemoryType.NUMPY, MemoryType.NUMPY, HOST) is values

    @pytest.mark.parametrize(
        ("source_type", "target_type", "role"),
        [("invalid_type", "numpy", "source_type"), ("numpy", "invalid_type", "target_type")],
    )
    def test_unregistered_name_is_rejected(self, source_type, target_type, role):
        with pytest.raises(ValueError, match=f"Invalid {role} 'invalid_type'"):
            convert_memory(np.array([1, 2, 3]), source_type, target_type, HOST)

    def test_device_the_target_lacks_is_a_conversion_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "jax", fake_jax("cpu"))

        with pytest.raises(
            MemoryConversionError, match="JAX device accelerator:0 is unavailable"
        ) as raised:
            convert_memory(
                np.array([1, 2, 3]), MemoryType.NUMPY, MemoryType.JAX, AcceleratorDevice(0)
            )

        assert (raised.value.source_type, raised.value.target_type) == ("numpy", "jax")
        assert isinstance(raised.value.__cause__, ValueError)

    def test_numpy_has_no_accelerator(self):
        with pytest.raises(MemoryConversionError, match="NumPy device accelerator:0"):
            convert_memory(np.array([1, 2, 3]), "numpy", "numpy", AcceleratorDevice(0))

    def test_round_trip_through_a_loaded_library(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", FakeTorch())
        original = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        tensor = convert_memory(original, "numpy", "torch", HOST)
        result = convert_memory(tensor, detect_memory_type(tensor), "numpy", HOST)

        assert isinstance(tensor, FakeTorchTensor)
        assert result.dtype == original.dtype
        np.testing.assert_array_equal(result, original)

    @pytest.mark.torch
    def test_round_trip_through_installed_torch_in_main_memory(self):
        torch = pytest.importorskip("torch")
        original = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        tensor = convert_memory(original, "numpy", "torch", HOST)
        result = convert_memory(tensor, "torch", "numpy", HOST)

        assert isinstance(tensor, torch.Tensor) and not tensor.is_cuda
        np.testing.assert_array_equal(result, original)
        if not torch.cuda.is_available():
            with pytest.raises(MemoryConversionError, match="accelerator:0 is unavailable"):
                convert_memory(original, "numpy", "torch", AcceleratorDevice(0))

    @pytest.mark.torch
    @pytest.mark.gpu
    def test_round_trip_through_installed_torch_on_a_gpu(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("PyTorch CUDA device not available")
        original = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        tensor = convert_memory(original, "numpy", "torch", AcceleratorDevice(0))
        result = convert_memory(tensor, "torch", "numpy", HOST)

        assert tensor.is_cuda
        np.testing.assert_array_equal(result, original)

    @pytest.mark.jax
    def test_installed_jax_to_numpy(self):
        jnp = pytest.importorskip("jax.numpy")
        array = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)

        result = convert_memory(array, MemoryType.JAX, MemoryType.NUMPY, HOST)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.asarray(array))
