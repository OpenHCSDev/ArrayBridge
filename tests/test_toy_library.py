"""A library is one class: writing it is all that adding a library takes."""

from contextlib import contextmanager

import numpy as np
import pytest
from metaclass_registry.testing import restored_registry

import arraybridge
from arraybridge import decorators
from arraybridge.conversion import ConversionRoute
from arraybridge.converters import convert_memory, detect_memory_type
from arraybridge.devices import HOST
from arraybridge.libraries import ArrayLibrary, HostDevices
from arraybridge.types import MemoryType


class ToyArray:
    """The smallest array: NumPy values behind a class of this module."""

    def __init__(self, values):
        self.values = np.asarray(values)

    @property
    def shape(self):
        return self.values.shape

    @property
    def dtype(self):
        return self.values.dtype


@contextmanager
def toy_library_defined():
    """Define ``ToyLibrary`` for a block, then remove every trace of it.

    A registered class stays registered for the life of the process, and each
    thing derived from it is kept where it was first asked for, so all of
    those are undone here.
    """
    with restored_registry(ArrayLibrary):

        class ToyLibrary(HostDevices[ToyArray]):
            name = "toy"
            import_name = __name__.split(".")[0]

            def device_of(self, data):
                return HOST

            def _to_host(self, data):
                return data.values

            def _wrap_host_array(self, data):
                return ToyArray(data)

            def _move_to_host(self, data):
                return data

            def _scale_dtype(self, result, target_dtype):
                return ToyArray(result.values.astype(target_dtype))

            # Array operations: nothing is inherited for them, because the
            # base class assumes no array API. This array wraps NumPy values.
            def accepts(self, value):
                return isinstance(value, ToyArray)

            def max(self, data, axis):
                return ToyArray(data.values.max(axis=axis))

            def min(self, data, axis):
                return ToyArray(data.values.min(axis=axis))

            def mean(self, data, axis):
                return ToyArray(data.values.mean(axis=axis))

            def sum(self, data, axis):
                return ToyArray(data.values.sum(axis=axis))

            def assign(self, data, index, values):
                updated = data.values.copy()
                updated[index] = values.values
                return ToyArray(updated)

            def _stack(self, values):
                return ToyArray(np.stack([value.values for value in values]))

            def _cast(self, data, dtype):
                return ToyArray(data.values.astype(dtype))

            def _logical_and(self, left, right):
                return ToyArray(np.logical_and(left.values, right.values))

            def _reshape(self, data, shape):
                return ToyArray(data.values.reshape(shape))

            def _broadcast_to(self, data, shape):
                return ToyArray(np.broadcast_to(data.values, shape))

            def _ones_like(self, reference, shape, dtype):
                return ToyArray(np.ones(shape, dtype=dtype))

        try:
            yield ToyLibrary
        finally:
            ArrayLibrary._instances.pop("toy", None)
            MemoryType._value2member_map_.pop("toy", None)
            if MemoryType._member_map_.pop("TOY", None) is not None:
                MemoryType._member_names_.remove("TOY")
            vars(decorators).pop("toy", None)
            vars(arraybridge).pop("toy", None)


@pytest.fixture
def toy_library_class():
    with toy_library_defined() as library_class:
        yield library_class


def test_defining_the_class_is_the_whole_registration(toy_library_class):
    from arraybridge.decorators import toy

    library = ArrayLibrary.named("toy")
    host_values = np.arange(6, dtype=np.uint16).reshape(2, 3)

    assert type(library) is toy_library_class
    assert ArrayLibrary.all()[-1] is library
    assert ArrayLibrary.for_module_name(__name__) is library
    assert library.display_name == "toy"
    assert library.available_devices() == (HOST,)

    assert MemoryType("toy") is MemoryType.TOY
    assert MemoryType.TOY in list(MemoryType)
    assert (MemoryType.TOY.name, MemoryType.TOY.value) == ("TOY", "toy")
    assert MemoryType.TOY.library is library

    toy_values = convert_memory(host_values, "numpy", "toy", HOST)
    assert isinstance(toy_values, ToyArray)
    assert ArrayLibrary.of(toy_values) is library
    assert isinstance(toy_values, library.array_class)
    assert detect_memory_type(toy_values) == "toy"
    np.testing.assert_array_equal(toy_values.values, host_values)
    np.testing.assert_array_equal(
        convert_memory(toy_values, MemoryType.TOY, MemoryType.NUMPY, HOST), host_values
    )

    numpy_library = ArrayLibrary.named("numpy")
    assert ConversionRoute.planned(numpy_library, HOST, library, HOST).mechanism == "host copy"
    assert ConversionRoute.planned(library, HOST, numpy_library, HOST).mechanism == "host copy"
    assert ConversionRoute.planned(library, HOST, library, HOST).mechanism == "device move"

    @toy
    def halve(image):
        return ToyArray(image.values / 2)

    assert toy is arraybridge.toy
    assert toy.__name__ == "toy"
    assert halve.input_memory_type == "toy"
    assert halve.output_memory_type == "toy"
    assert halve.execution_memory_type == "toy"
    halved = halve(ToyArray(np.array([0, 2, 4], dtype=np.uint16)))
    assert isinstance(halved, ToyArray)
    assert halved.dtype == np.uint16
    np.testing.assert_array_equal(halved.values, [0, 1, 2])


def test_name_lookup_sees_a_library_defined_after_import(toy_library_class):
    # The first look at the enum after the class statement is by member name.
    assert MemoryType["TOY"].value == "toy"


def test_nothing_of_the_class_is_left_afterwards():
    registered = list(ArrayLibrary.__registry__)
    members = list(MemoryType)

    with toy_library_defined():
        assert MemoryType.TOY.library is ArrayLibrary.named("toy")
        assert callable(arraybridge.toy) and callable(decorators.toy)

    assert list(ArrayLibrary.__registry__) == registered
    assert [library.name for library in ArrayLibrary.all()] == registered
    assert list(MemoryType) == members
    assert list(MemoryType.__members__) == [member.name for member in members]
    for lookup in (lambda: MemoryType("toy"), lambda: ArrayLibrary.named("toy")):
        with pytest.raises(ValueError):
            lookup()
    for owner in (MemoryType, decorators, arraybridge):
        assert not hasattr(owner, "TOY") and not hasattr(owner, "toy")
    with pytest.raises(ValueError, match="Unknown memory type"):
        ArrayLibrary.of(ToyArray([1]))
