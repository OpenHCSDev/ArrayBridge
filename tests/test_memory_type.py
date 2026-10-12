"""``MemoryType`` is the registered library names as an enum, and nothing more."""

import pickle
from types import SimpleNamespace

import pytest

import arraybridge
from arraybridge.libraries import ArrayLibrary
from arraybridge.types import MemoryContractAttribute, MemoryType

LIBRARY_NAMES = {"numpy", "cupy", "torch", "tensorflow", "jax", "pyclesperanto"}


def test_members_are_the_registered_libraries_in_order():
    assert [member.value for member in MemoryType] == list(ArrayLibrary.__registry__)
    assert {member.value for member in MemoryType} == LIBRARY_NAMES
    assert len(MemoryType) == len(ArrayLibrary.all())
    for member in MemoryType:
        assert member.name == member.value.upper()
        assert MemoryType(member.value) is member
        assert getattr(MemoryType, member.name) is member
        assert MemoryType.__members__[member.name] is member
        assert member in MemoryType
        assert member.library is ArrayLibrary.named(member.value)
        assert pickle.loads(pickle.dumps(member)) is member


def test_unregistered_name_is_not_a_member():
    with pytest.raises(ValueError):
        MemoryType("invalid_type")
    with pytest.raises(AttributeError):
        MemoryType.INVALID_TYPE


def test_members_carry_no_library_behaviour():
    removed = ("is_gpu", "to_numpy", "from_numpy", "stack_arrays", "scale_dtype", "import_name")
    for member in MemoryType:
        assert not any(hasattr(member, name) for name in removed)
    for constant in ("CPU_MEMORY_TYPES", "GPU_MEMORY_TYPES", "SUPPORTED_MEMORY_TYPES"):
        assert not hasattr(arraybridge, constant)


def test_importing_the_enum_imports_no_array_library(loaded_libraries_after):
    loaded = loaded_libraries_after(
        "from arraybridge.types import MemoryType\n"
        "names = [member.value for member in MemoryType]\n"
        "assert len(names) == 6 and 'numpy' in names, names\n"
        "assert MemoryType.TORCH.value == 'torch'\n"
    )
    assert loaded == "[]"


def test_package_root_exports_resolve():
    for name in arraybridge.__all__:
        assert getattr(arraybridge, name) is not None
    assert arraybridge.MemoryType is MemoryType
    assert LIBRARY_NAMES <= set(arraybridge.__all__)
    with pytest.raises(AttributeError):
        arraybridge.not_a_library


class TestMemoryContractAttribute:
    def test_reads_and_writes_object_namespaces(self):
        namespace = SimpleNamespace()

        MemoryContractAttribute.EXECUTION.write(namespace, "torch")

        assert MemoryContractAttribute.EXECUTION.read(namespace) == "torch"

    def test_reads_and_writes_mapping_namespaces(self):
        namespace = {}

        MemoryContractAttribute.INPUT.write(namespace, "numpy")

        assert MemoryContractAttribute.INPUT.read(namespace) == "numpy"
        assert MemoryContractAttribute.OUTPUT.read(namespace) is None
