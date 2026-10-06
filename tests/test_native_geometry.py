"""Native geometry must preserve pixels, sharing and the reference device."""

import numpy as np
import pytest

from arraybridge.array_operations import PYCLESPERANTO_OPERATIONS
from arraybridge.types import MemoryType


def test_numpy_views_and_independent_reference_allocation():
    source = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::-1]
    shaped = MemoryType.NUMPY.reshape(source, (3, 1, 4))
    broadcast = MemoryType.NUMPY.broadcast_to(shaped, (3, 2, 4))
    assert np.shares_memory(shaped, source)
    assert np.shares_memory(broadcast, source)
    np.testing.assert_array_equal(broadcast, np.broadcast_to(source[:, None], (3, 2, 4)))
    ones = MemoryType.NUMPY.ones_like(source, shape=(2, 4), dtype=bool)
    assert ones.dtype == np.bool_
    assert not np.shares_memory(ones, source)
    np.testing.assert_array_equal(ones, np.ones((2, 4), dtype=bool))


def test_numpy_geometry_rejects_invalid_shapes():
    source = np.arange(6).reshape(2, 3)
    with pytest.raises(ValueError):
        MemoryType.NUMPY.reshape(source, (7,))
    with pytest.raises(ValueError):
        MemoryType.NUMPY.broadcast_to(source, (2, 4))


def test_cupy_geometry_and_allocation_stay_native():
    cp = pytest.importorskip("cupy")
    if not MemoryType.CUPY.available_device_ids(cp):
        pytest.skip("No CUDA device")
    source = cp.asarray(np.arange(12, dtype=np.float32).reshape(3, 4))[:, ::-1]
    shaped = MemoryType.CUPY.reshape(source, (3, 1, 4))
    broadcast = MemoryType.CUPY.broadcast_to(shaped, (3, 2, 4))
    assert shaped.data.ptr == source.data.ptr
    assert broadcast.data.ptr == source.data.ptr
    assert broadcast.device.id == source.device.id
    np.testing.assert_array_equal(
        broadcast.get(), np.broadcast_to(source.get()[:, None], (3, 2, 4))
    )
    ones = MemoryType.CUPY.ones_like(source, shape=(2, 4), dtype=bool)
    assert ones.device.id == source.device.id
    assert ones.dtype == cp.bool_
    np.testing.assert_array_equal(ones.get(), np.ones((2, 4), dtype=bool))


def test_pycles_geometry_rejects_host_fallback():
    class DeviceOnlyArray:
        shape = (2, 3)

        def __array__(self):
            pytest.fail("Native geometry must never download pixels")

    source = DeviceOnlyArray()
    assert PYCLESPERANTO_OPERATIONS.reshape(source, (2, 3), None) is source
    assert PYCLESPERANTO_OPERATIONS.broadcast_to(source, (2, 3), None) is source
    with pytest.raises(NotImplementedError, match="reshape"):
        PYCLESPERANTO_OPERATIONS.reshape(source, (1, 2, 3), None)
    with pytest.raises(NotImplementedError, match="broadcasting"):
        PYCLESPERANTO_OPERATIONS.broadcast_to(source, (4, 2, 3), None)
    with pytest.raises(NotImplementedError, match="dtype"):
        PYCLESPERANTO_OPERATIONS.ones_like(source, (2, 3), bool, None)
