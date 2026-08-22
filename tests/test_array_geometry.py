from __future__ import annotations

import numpy as np
import pytest

from arraybridge import ArrayGeometry, ArrayPayload


class DeviceArray:
    """Array-shaped value that rejects implicit host conversion."""

    shape = (2, 3, 4)

    def __array__(self, dtype=None):
        del dtype
        raise TypeError("implicit host conversion is forbidden")


class DevicePayload(ArrayPayload):
    def __init__(self, data):
        self.data = data

    def array_payload_data(self):
        return self.data

    def with_data(self, data):
        return type(self)(data)


def test_array_geometry_reads_declared_device_shape_without_host_conversion() -> None:
    assert ArrayGeometry.from_value(DeviceArray()) == ArrayGeometry((2, 3, 4))


def test_array_geometry_unwraps_nominal_array_payload() -> None:
    assert ArrayGeometry.from_value(DevicePayload(DeviceArray())) == ArrayGeometry((2, 3, 4))


def test_array_geometry_falls_back_for_python_array_inputs() -> None:
    assert ArrayGeometry.from_value([[1, 2], [3, 4]]) == ArrayGeometry((2, 2))
    assert ArrayGeometry.from_value(np.zeros((4, 5))) == ArrayGeometry((4, 5))


def test_array_geometry_requires_concrete_shape() -> None:
    class UnshapedValue:
        def __array__(self, dtype=None):
            del dtype
            raise TypeError("not an array")

    with pytest.raises(TypeError, match="Runtime output requires concrete array geometry"):
        ArrayGeometry.require_from_value(
            UnshapedValue(),
            value_name="Runtime output",
        )
