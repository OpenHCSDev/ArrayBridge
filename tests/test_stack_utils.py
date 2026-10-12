"""Stacking 2D slices into a 3D array of a named library on a device, and back."""

import sys

import numpy as np
import pytest

from arraybridge.devices import HOST, AcceleratorDevice
from arraybridge.libraries import ArrayLibrary
from arraybridge.stack_utils import _is_2d, _is_3d, stack_slices, unstack_slices
from tests.fakes import (
    FakeCupy,
    FakeCupyArray,
    FakePyclesperanto,
    FakePyclesperantoArray,
    FakeTensorflow,
    FakeTensorflowTensor,
    FakeTorch,
    FakeTorchTensor,
)

PLANES = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_rank_checks(rank):
    values = np.zeros((2,) * rank)

    assert _is_2d(values) is (rank == 2)
    assert _is_3d(values) is (rank == 3)
    assert not _is_2d("no shape") and not _is_3d("no shape")


class TestNumpy:
    @pytest.mark.parametrize("planes", [PLANES, PLANES[:1]], ids=["two", "one"])
    def test_stack_and_unstack_round_trip(self, planes):
        stacked = stack_slices(planes, "numpy", HOST)
        unstacked = unstack_slices(stacked, "numpy", HOST)

        assert stacked.shape == (len(planes), 2, 2)
        np.testing.assert_array_equal(stacked, np.stack(planes))
        assert len(unstacked) == len(planes)
        for original, result in zip(planes, unstacked, strict=True):
            assert result.shape == (2, 2)
            np.testing.assert_array_equal(result, original)

    def test_unstack_can_skip_the_slice_check(self):
        slices = unstack_slices(np.zeros((2, 1, 2)), "numpy", HOST, validate_slices=False)

        assert [each.shape for each in slices] == [(1, 2), (1, 2)]

    def test_malformed_input_is_rejected(self):
        with pytest.raises(ValueError, match="Cannot stack empty list"):
            stack_slices([], "numpy", HOST)
        with pytest.raises(ValueError, match="Slice at index 0 is not a 2D array"):
            stack_slices([np.array([1, 2, 3])], "numpy", HOST)
        with pytest.raises(ValueError, match="Array must be 3D"):
            unstack_slices(PLANES[0], "numpy", HOST)

    def test_unregistered_library_and_missing_device_are_rejected(self):
        with pytest.raises(ValueError, match="not a registered array library"):
            stack_slices(PLANES, "invalid_type", HOST)
        with pytest.raises(ValueError, match="NumPy device accelerator:0 is unavailable"):
            stack_slices(PLANES, "numpy", AcceleratorDevice(0))
        with pytest.raises(ValueError, match="NumPy device accelerator:0 is unavailable"):
            unstack_slices(np.stack(PLANES), "numpy", AcceleratorDevice(0))


class TestOtherLibraries:
    def test_stack_and_unstack_move_to_the_requested_device(self, monkeypatch):
        module = FakeCupy()
        monkeypatch.setitem(sys.modules, "cupy", module)
        inputs = [FakeCupyArray(plane, 0, module) for plane in PLANES]

        stacked = stack_slices(inputs, "cupy", AcceleratorDevice(1))
        slices = unstack_slices(stacked, "cupy", AcceleratorDevice(1))

        assert stacked.device.id == 1
        assert [each.device.id for each in slices] == [1, 1]
        np.testing.assert_array_equal(stacked.values, np.stack(PLANES))
        assert module.current == 0

    def test_tensorflow_tensors_are_copied_to_the_device_not_changed(self, monkeypatch):
        module = FakeTensorflow()
        monkeypatch.setitem(sys.modules, "tensorflow", module)
        inputs = [FakeTensorflowTensor(plane, 0) for plane in PLANES]

        stacked = stack_slices(inputs, "tensorflow", AcceleratorDevice(1))
        slices = unstack_slices(stacked, "tensorflow", AcceleratorDevice(1))

        assert stacked.device == "/device:GPU:1"
        assert [each.device for each in slices] == ["/device:GPU:1", "/device:GPU:1"]
        assert [each.device for each in inputs] == ["/device:GPU:0", "/device:GPU:0"]
        np.testing.assert_array_equal(stacked.values, np.stack(PLANES))
        assert module.current == 0

    def test_slices_of_another_library_are_converted_first(self, monkeypatch):
        module = FakeTorch()
        module.stack = lambda values, dim: FakeTorchTensor(
            np.stack([each.values for each in values], axis=dim)
        )
        monkeypatch.setitem(sys.modules, "torch", module)

        stacked = stack_slices([PLANES[0], FakeTorchTensor(PLANES[1])], "torch", HOST)
        as_numpy = unstack_slices(stacked, "numpy", HOST)

        assert isinstance(stacked, FakeTorchTensor)
        np.testing.assert_array_equal(stacked.values, np.stack(PLANES))
        np.testing.assert_array_equal(as_numpy, PLANES)

    def test_pyclesperanto_stacks_pairwise_and_keeps_a_single_plane_3d(self, monkeypatch):
        module = FakePyclesperanto()
        monkeypatch.setitem(sys.modules, "pyclesperanto", module)
        library = ArrayLibrary.named("pyclesperanto")
        first, second, third = (
            FakePyclesperantoArray(plane, "gpu0") for plane in (*PLANES, PLANES[0] + 8)
        )

        single = library.stack_arrays([first], AcceleratorDevice(0))
        stack = library.stack_arrays([first, second, third], AcceleratorDevice(0))

        assert single.shape == (1, 2, 2)
        assert stack.shape == (3, 2, 2)
        np.testing.assert_array_equal(stack.values[2], third.values)
        assert module.events == [
            ("create", (1, 2, 2)),
            ("copy_slice", 0),
            ("concatenate", (2, 2), (2, 2)),
            ("concatenate", (2, 2, 2), (2, 2)),
        ]
        with pytest.raises(ValueError, match="empty pyclesperanto sequence"):
            library.stack_arrays([], AcceleratorDevice(0))
