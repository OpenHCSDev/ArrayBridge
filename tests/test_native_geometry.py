"""Native geometry must preserve pixels, sharing and the reference device."""

import sys

import numpy as np
import pytest

from arraybridge.libraries import ArrayLibrary
from tests.fakes import FakePyclesperanto

NUMPY = ArrayLibrary.named("numpy")
CUPY = ArrayLibrary.named("cupy")
PYCLESPERANTO = ArrayLibrary.named("pyclesperanto")


def test_numpy_views_and_independent_reference_allocation():
    source = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::-1]
    shaped = NUMPY.reshape(source, (3, 1, 4))
    broadcast = NUMPY.broadcast_to(shaped, (3, 2, 4))
    assert np.shares_memory(shaped, source)
    assert np.shares_memory(broadcast, source)
    np.testing.assert_array_equal(broadcast, np.broadcast_to(source[:, None], (3, 2, 4)))
    ones = NUMPY.ones_like(source, shape=(2, 4), dtype=bool)
    assert ones.dtype == np.bool_
    assert not np.shares_memory(ones, source)
    np.testing.assert_array_equal(ones, np.ones((2, 4), dtype=bool))


def test_numpy_geometry_rejects_invalid_shapes():
    source = np.arange(6).reshape(2, 3)
    with pytest.raises(ValueError):
        NUMPY.reshape(source, (7,))
    with pytest.raises(ValueError):
        NUMPY.broadcast_to(source, (2, 4))


def test_cupy_geometry_and_allocation_stay_native():
    cp = pytest.importorskip("cupy")
    if not CUPY.accelerator_devices():
        pytest.skip("No CUDA device")
    source = cp.asarray(np.arange(12, dtype=np.float32).reshape(3, 4))[:, ::-1]
    shaped = CUPY.reshape(source, (3, 1, 4))
    broadcast = CUPY.broadcast_to(shaped, (3, 2, 4))
    assert shaped.data.ptr == source.data.ptr
    assert broadcast.data.ptr == source.data.ptr
    assert broadcast.device.id == source.device.id
    np.testing.assert_array_equal(
        broadcast.get(), np.broadcast_to(source.get()[:, None], (3, 2, 4))
    )
    ones = CUPY.ones_like(source, shape=(2, 4), dtype=bool)
    assert ones.device.id == source.device.id
    assert ones.dtype == cp.bool_
    np.testing.assert_array_equal(ones.get(), np.ones((2, 4), dtype=bool))


def test_pycles_geometry_rejects_host_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyclesperanto", FakePyclesperanto(current="cpu0"))

    class DeviceOnlyArray:
        shape = (2, 3)

        def __array__(self):
            pytest.fail("Native geometry must never download pixels")

    source = DeviceOnlyArray()
    assert PYCLESPERANTO.reshape(source, (2, 3)) is source
    assert PYCLESPERANTO.broadcast_to(source, (2, 3)) is source
    with pytest.raises(NotImplementedError, match="reshape"):
        PYCLESPERANTO.reshape(source, (1, 2, 3))
    with pytest.raises(NotImplementedError, match="broadcasting"):
        PYCLESPERANTO.broadcast_to(source, (4, 2, 3))
    with pytest.raises(NotImplementedError, match="dtype"):
        PYCLESPERANTO.ones_like(source, shape=(2, 3), dtype=bool)


@pytest.mark.parametrize("dtype", (np.float32, np.complex64, np.int32))
def test_normalize_planes_preserves_source_and_division_dtype(dtype):
    source = np.arange(24, dtype=dtype).reshape(2, 3, 4)[:, :, ::-1]
    original = source.copy()
    expected = np.stack((source[0] / 3.0, source[1]))
    actual = NUMPY.normalize_planes(source, dtype, (3.0, None))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, original)
    assert actual.dtype == expected.dtype
    assert not np.shares_memory(actual, source)
    for scales in ((3.0,), (3.0, None, 2.0)):
        with pytest.raises(ValueError, match="leading plane axis"):
            NUMPY.normalize_planes(source, dtype, scales)
