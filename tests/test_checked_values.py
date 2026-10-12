"""The places where a value of unknown type is checked before it is given a type."""

import sys
from enum import Enum
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from arraybridge.conversion import (
    DeviceMove,
    DLPackExchange,
    RouteDeclinedError,
    _imports_dlpack,
    _is_array_of_both,
)
from arraybridge.devices import HOST, AcceleratorDevice
from arraybridge.libraries import ArrayLibrary, _mapped_dtype
from arraybridge.types import (
    MemoryContractAttribute,
    MemoryType,
    _add_missing_members,
    _holds_members_of,
)
from tests.fakes import FakeCupy, FakeCupyArray, FakePyclesperanto, FakeTorch, FakeTorchTensor

NUMPY = ArrayLibrary.named("numpy")
TORCH = ArrayLibrary.named("torch")
CUPY = ArrayLibrary.named("cupy")
PYCLESPERANTO = ArrayLibrary.named("pyclesperanto")


class TestMappedDtype:
    def test_returns_the_object_of_the_asked_class(self):
        assert _mapped_dtype("float32", np, type) is np.float32
        assert _mapped_dtype(np.uint16, np, object) is np.uint16

    def test_rejects_an_object_of_another_class(self):
        with pytest.raises(TypeError, match="numpy.float32 is not a int"):
            _mapped_dtype("float32", np, int)

    def test_rejects_a_name_the_module_does_not_have(self):
        with pytest.raises(TypeError, match="does not expose dtype float32"):
            _mapped_dtype("float32", SimpleNamespace(), object)


class TestMemoryContractAttributeRead:
    @pytest.mark.parametrize("namespace", [{}, SimpleNamespace()], ids=["mapping", "object"])
    def test_returns_a_written_name_and_the_default(self, namespace):
        assert MemoryContractAttribute.INPUT.read(namespace) is None
        assert MemoryContractAttribute.INPUT.read(namespace, "numpy") == "numpy"
        MemoryContractAttribute.INPUT.write(namespace, "torch")
        assert MemoryContractAttribute.INPUT.read(namespace) == "torch"

    @pytest.mark.parametrize(
        "namespace",
        [{"input_memory_type": 3}, SimpleNamespace(input_memory_type=3)],
        ids=["mapping", "object"],
    )
    def test_rejects_a_value_that_is_not_a_name(self, namespace):
        with pytest.raises(TypeError, match="not a library name"):
            MemoryContractAttribute.INPUT.read(namespace)


class TestMemoryTypeMembers:
    def test_members_of_the_enum_are_recognised(self):
        members = MappingProxyType({"NUMPY": MemoryType.NUMPY})
        assert _holds_members_of(members, MemoryType)
        assert MemoryType.__members__["NUMPY"] is MemoryType.NUMPY

    def test_a_member_of_another_enum_is_not(self):
        class Other(Enum):
            NUMPY = "numpy"

        assert not _holds_members_of(MappingProxyType({"NUMPY": Other.NUMPY}), MemoryType)

    def test_only_an_enum_class_takes_members(self):
        with pytest.raises(TypeError, match="not an enum class"):
            _add_missing_members(int)


class TestDeviceMove:
    def test_an_array_of_one_library_is_an_array_of_the_same_library(self):
        values = np.arange(3)
        assert _is_array_of_both(values, NUMPY, NUMPY)
        assert DeviceMove.convert(values, NUMPY, NUMPY, HOST) is values

    def test_two_libraries_are_rejected(self):
        values = np.arange(3)
        assert not _is_array_of_both(values, NUMPY, TORCH)
        with pytest.raises(TypeError, match="different libraries"):
            DeviceMove.convert(values, NUMPY, TORCH, HOST)


class TestDLPackTarget:
    def test_only_a_dlpack_importer_is_one(self):
        assert _imports_dlpack(TORCH)
        assert not _imports_dlpack(NUMPY)

    def test_a_target_that_does_not_import_declines_the_route(self):
        with pytest.raises(RouteDeclinedError, match="does not import DLPack"):
            DLPackExchange.convert(np.arange(3), NUMPY, NUMPY, HOST)


class TestTorchStreamScope:
    def test_takes_a_torch_stream_and_no_stream(self, monkeypatch):
        module = FakeTorch(device_count=1)
        monkeypatch.setitem(sys.modules, "torch", module)
        stream = TORCH.create_stream()

        assert TORCH.stream_scope(stream) is stream
        assert TORCH.stream_scope(None) is None

    def test_rejects_what_is_not_a_stream(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", FakeTorch(device_count=1))

        with pytest.raises(TypeError, match="is not a PyTorch stream"):
            TORCH.stream_scope(object())


class TestDeviceIfAccepted:
    def test_an_array_of_the_library_has_its_device(self, monkeypatch):
        cupy = FakeCupy(device_count=2)
        monkeypatch.setitem(sys.modules, "cupy", cupy)
        monkeypatch.setitem(sys.modules, "torch", FakeTorch(device_count=2))

        assert CUPY._device_if_accepted(FakeCupyArray([1], 1, cupy)) == AcceleratorDevice(1)
        assert TORCH._device_if_accepted(FakeTorchTensor([1], cuda_index=1)) == AcceleratorDevice(1)
        assert TORCH._device_if_accepted(FakeTorchTensor([1])) == HOST

    def test_another_object_has_none(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cupy", FakeCupy(device_count=2))
        monkeypatch.setitem(sys.modules, "torch", FakeTorch(device_count=2))

        for value in (np.arange(3), "text", None):
            assert CUPY._device_if_accepted(value) is None
            assert TORCH._device_if_accepted(value) is None

    def test_pyclesperanto_takes_a_host_array_on_its_selected_device(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyclesperanto", FakePyclesperanto())

        assert PYCLESPERANTO._device_if_accepted(np.arange(3)) == PYCLESPERANTO.device_of(
            np.arange(3)
        )
        assert PYCLESPERANTO._device_if_accepted("text") is None
