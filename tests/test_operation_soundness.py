"""No library inherits an array operation it cannot perform.

Two checks. Where a library can be imported here, every public array
operation it has is called and compared with NumPy. For every library,
importable or not, the class structure is checked: the base class supplies no
implementation that assumes an array API, and the operations a library cannot
perform are absent from its class.
"""

import inspect

import numpy as np
import pytest

from arraybridge.libraries import (
    ArrayLibrary,
    ArrayMath,
    InPlaceAssignment,
    NumpyArrayApi,
    PyclesperantoLibrary,
)

OPERATION_OWNERS = (ArrayLibrary, ArrayMath, InPlaceAssignment)


def _returns_native_array(function) -> bool:
    return inspect.signature(function).return_annotation == "NativeT"


# Every public method of the family that returns an array of the library.
ARRAY_OPERATIONS = frozenset(
    name
    for owner in OPERATION_OWNERS
    for name, function in vars(owner).items()
    if not name.startswith("_") and inspect.isfunction(function)
    if _returns_native_array(function)
)
# The private steps a leaf or a role supplies for them.
ARRAY_STEPS = frozenset(
    name
    for name, function in vars(ArrayLibrary).items()
    if name.startswith("_") and not name.startswith("__") and inspect.isfunction(function)
    if _returns_native_array(function)
)

STACK = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
VECTOR = np.array([1.0, 2.0, 3.0], dtype=np.float32)
RAMP = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float32)

# name -> (call(library, device, upload), the NumPy answer)
OPERATION_CALLS = {
    "move_to_device": (lambda lib, dev, up: lib.move_to_device(up(STACK), dev), STACK),
    "unrecorded_from_numpy": (lambda lib, dev, up: lib.unrecorded_from_numpy(STACK, dev), STACK),
    "from_numpy": (lambda lib, dev, up: lib.from_numpy(STACK, dev), STACK),
    "stack_arrays": (
        lambda lib, dev, up: lib.stack_arrays([up(STACK[0]), up(STACK[1])], dev),
        STACK,
    ),
    "normalize_planes": (
        lambda lib, dev, up: lib.normalize_planes(up(STACK), np.float32, [2.0, None]),
        np.stack([STACK[0] / 2.0, STACK[1]]),
    ),
    "scale_dtype": (lambda lib, dev, up: lib.scale_dtype(up(STACK), np.float32), STACK),
    "astype": (lambda lib, dev, up: lib.astype(up(STACK), np.int32), STACK.astype(np.int32)),
    "reshape": (lambda lib, dev, up: lib.reshape(up(STACK), (6, 4)), STACK.reshape(6, 4)),
    "broadcast_to": (
        lambda lib, dev, up: lib.broadcast_to(up(STACK[:1]), (2, 3, 4)),
        np.broadcast_to(STACK[:1], (2, 3, 4)),
    ),
    "ones_like": (
        lambda lib, dev, up: lib.ones_like(up(STACK), dtype=np.float32),
        np.ones_like(STACK),
    ),
    "logical_and": (
        lambda lib, dev, up: lib.logical_and(up(STACK > 3), up(STACK < 10)),
        np.logical_and(STACK > 3, STACK < 10),
    ),
    "max": (lambda lib, dev, up: lib.max(up(STACK), 0), STACK.max(axis=0)),
    "min": (lambda lib, dev, up: lib.min(up(STACK), 0), STACK.min(axis=0)),
    "mean": (lambda lib, dev, up: lib.mean(up(STACK), 0), STACK.mean(axis=0)),
    "sum": (lambda lib, dev, up: lib.sum(up(STACK), 0), STACK.sum(axis=0)),
    "prepend_axis": (lambda lib, dev, up: lib.prepend_axis(up(STACK[0])), STACK[:1]),
    "assign": (
        lambda lib, dev, up: lib.assign(
            up(np.ones(5, dtype=np.float32)), slice(-2, None), up(RAMP[:2])
        ),
        np.array([1, 1, 1, 0, 0.25], dtype=np.float32),
    ),
    "linspace": (lambda lib, dev, up: lib.linspace(0, 1, 4, endpoint=False), RAMP),
    "ones": (lambda lib, dev, up: lib.ones(5, np.float32), np.ones(5, dtype=np.float32)),
    "outer": (
        lambda lib, dev, up: lib.outer(up(VECTOR), up(VECTOR)),
        np.outer(VECTOR, VECTOR),
    ),
    "floor": (lambda lib, dev, up: lib.floor(2.7), np.float32(2.0)),
}

LIBRARY_CLASSES = tuple(ArrayLibrary.__registry__.values())


def test_every_array_operation_has_a_call() -> None:
    assert set(OPERATION_CALLS) == ARRAY_OPERATIONS


@pytest.mark.parametrize("library", ArrayLibrary.all(), ids=str)
@pytest.mark.parametrize("operation", sorted(OPERATION_CALLS))
def test_importable_library_performs_every_operation_it_has(library, operation) -> None:
    if not hasattr(library, operation):
        pytest.skip(f"{library.display_name} does not have {operation}")
    if library.import_if_installed() is None:
        pytest.skip(f"{library.display_name} is not installed")
    devices = library.available_devices()
    if not devices:
        pytest.skip(f"{library.display_name} has no device on this machine")
    device = devices[0]
    call, expected = OPERATION_CALLS[operation]

    result = call(library, device, lambda values: library.from_numpy(values, device))

    # ``floor`` of a host scalar is a scalar of the library (NumPy: ``np.float64``).
    assert library.accepts(result) or np.ndim(expected) == 0
    np.testing.assert_allclose(library.to_numpy(result), expected, rtol=1e-6)


def test_the_base_class_assumes_no_array_api() -> None:
    """An implementation in ``ArrayLibrary`` never calls the library's module."""

    for name in ARRAY_OPERATIONS | ARRAY_STEPS | {"accepts"}:
        function = vars(ArrayLibrary).get(name)
        if function is None or getattr(function, "__isabstractmethod__", False):
            continue
        assert not {"module", "array_module"} & set(function.__code__.co_names), name


@pytest.mark.parametrize("library_class", LIBRARY_CLASSES, ids=lambda cls: cls.name)
def test_library_class_leaves_nothing_abstract(library_class) -> None:
    assert not inspect.isabstract(library_class)


@pytest.mark.parametrize("library_class", LIBRARY_CLASSES, ids=lambda cls: cls.name)
def test_reference_free_operations_exist_only_on_array_math(library_class) -> None:
    reference_free = {
        name for name in vars(ArrayMath) if not name.startswith("_") and name in ARRAY_OPERATIONS
    }
    assert reference_free == {"linspace", "ones", "outer", "floor"}
    for name in reference_free:
        assert hasattr(library_class, name) == issubclass(library_class, ArrayMath), name


@pytest.mark.parametrize("library_class", LIBRARY_CLASSES, ids=lambda cls: cls.name)
def test_numpy_style_implementations_reach_only_numpy_api_libraries(library_class) -> None:
    """Outside ``NumpyArrayApi`` every operation comes from the leaf or a non-NumPy role."""

    if issubclass(library_class, NumpyArrayApi):
        return
    for name in ARRAY_OPERATIONS | ARRAY_STEPS | {"accepts"}:
        if not hasattr(library_class, name):
            continue
        owner = next(base for base in library_class.__mro__ if name in vars(base))
        assert owner is not NumpyArrayApi, name


def test_which_libraries_follow_the_numpy_api() -> None:
    assert {cls.name for cls in LIBRARY_CLASSES if issubclass(cls, NumpyArrayApi)} == {
        "numpy",
        "cupy",
        "jax",
    }
    assert not issubclass(PyclesperantoLibrary, ArrayMath)
    with pytest.raises(AttributeError):
        ArrayLibrary.named("pyclesperanto").linspace
