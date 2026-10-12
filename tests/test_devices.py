"""Devices are values, and each library selects, reports and moves between its own."""

import importlib
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from arraybridge.devices import HOST, AcceleratorDevice, Device, HostDevice, NamedDevice
from arraybridge.libraries import ArrayLibrary, HostDevices
from tests.fakes import (
    FakeCupy,
    FakePyclesperanto,
    FakePyclesperantoArray,
    FakeTorch,
    FakeTorchTensor,
    fake_jax,
)

LIBRARIES = ArrayLibrary.all()
NUMPY = ArrayLibrary.named("numpy")
TORCH = ArrayLibrary.named("torch")
JAX = ArrayLibrary.named("jax")
PYCLESPERANTO = ArrayLibrary.named("pyclesperanto")


class TestDeviceValues:
    @pytest.mark.parametrize(
        ("device", "equal", "label"),
        [
            (HOST, HostDevice(), "host"),
            (AcceleratorDevice(1), AcceleratorDevice(1), "accelerator:1"),
            (NamedDevice("cpu0"), NamedDevice("cpu0"), "device:cpu0"),
        ],
    )
    def test_leaves_are_equal_by_value_hashable_and_picklable(self, device, equal, label):
        assert isinstance(device, Device)
        assert device == equal
        assert device is not equal
        assert hash(device) == hash(equal)
        assert {device: "found"}[equal] == "found"
        assert pickle.loads(pickle.dumps(device)) == device
        assert device.label == label

    def test_different_devices_are_not_equal(self):
        devices = [HOST, AcceleratorDevice(0), AcceleratorDevice(1), NamedDevice("0")]

        assert len(set(devices)) == len(devices)

    def test_a_device_cannot_be_changed(self):
        with pytest.raises(AttributeError):
            AcceleratorDevice(0).index = 1


class TestEveryLibrary:
    @pytest.mark.parametrize("library", LIBRARIES, ids=str)
    def test_without_its_package_only_main_memory_is_available(self, library, absent_libraries):
        host = (HOST,) if isinstance(library, HostDevices) else ()

        assert library.host_devices() == host
        assert library.accelerator_devices() == ()
        assert library.available_devices() == host
        for device in host:
            assert library.require_device(device) is None
        for missing in (AcceleratorDevice(0), NamedDevice("cpu0"), *({HOST} - set(host))):
            with pytest.raises(ValueError, match=f"device {missing.label} is unavailable"):
                library.require_device(missing)
            with pytest.raises(ValueError, match="is unavailable"):
                library.device_scope(missing)


class TestNumpy:
    def test_arrays_live_in_main_memory(self):
        values = np.array([1, 2, 3])

        assert NUMPY.device_of(values) == HOST
        assert NUMPY.move_to_device(values, HOST) is values
        assert NUMPY.activate_device(HOST) is None
        with pytest.raises(ValueError, match="NumPy device accelerator:0 is unavailable"):
            NUMPY.move_to_device(values, AcceleratorDevice(0))

    def test_host_scope_does_not_import_numpy(self, monkeypatch):
        monkeypatch.setattr(
            importlib, "import_module", lambda name: pytest.fail(f"imported {name}")
        )

        with NUMPY.device_scope(HOST):
            pass


class TestTorch:
    def test_device_of_reads_the_tensor(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", FakeTorch())

        assert TORCH.device_of(FakeTorchTensor([1, 2])) == HOST
        assert TORCH.device_of(FakeTorchTensor([1, 2], cuda_index=1)) == AcceleratorDevice(1)
        with pytest.raises(AttributeError):
            TORCH.device_of(SimpleNamespace())

    def test_cuda_devices_are_numbered_as_torch_numbers_them(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", FakeTorch(device_count=2))

        assert TORCH.accelerator_devices() == (AcceleratorDevice(0), AcceleratorDevice(1))
        assert TORCH.available_devices() == (HOST, AcceleratorDevice(0), AcceleratorDevice(1))
        for index in (-1, 2):
            with pytest.raises(ValueError, match=f"device accelerator:{index} is unavailable"):
                TORCH.require_device(AcceleratorDevice(index))

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_activate_selects_the_device_for_the_process(self, index, monkeypatch):
        module = FakeTorch(device_count=3)
        monkeypatch.setitem(sys.modules, "torch", module)

        TORCH.activate_device(AcceleratorDevice(index))

        assert module.events == [("set_device", index)]

    def test_move_copies_only_when_the_device_differs(self, monkeypatch):
        module = FakeTorch(device_count=2)
        monkeypatch.setitem(sys.modules, "torch", module)
        on_first = FakeTorchTensor([1, 2], cuda_index=0)

        on_second = TORCH.move_to_device(on_first, AcceleratorDevice(1))
        on_host = TORCH.move_to_device(on_second, HOST)

        assert TORCH.move_to_device(on_first, AcceleratorDevice(0)) is on_first
        assert TORCH.device_of(on_second) == AcceleratorDevice(1)
        assert TORCH.device_of(on_host) == HOST
        assert module.events == [("enter", 1), ("exit", 1)]

    @pytest.mark.torch
    def test_installed_torch_works_in_main_memory(self):
        torch = pytest.importorskip("torch")
        values = np.arange(6, dtype=np.float32).reshape(2, 3)

        tensor = TORCH.from_numpy(values, HOST)

        assert isinstance(tensor, torch.Tensor) and not tensor.is_cuda
        assert TORCH.device_of(tensor) == HOST
        assert TORCH.move_to_device(tensor, HOST) is tensor
        np.testing.assert_array_equal(TORCH.to_numpy(tensor), values)
        assert TORCH.available_devices()[0] == HOST
        if not torch.cuda.is_available():
            assert TORCH.accelerator_devices() == ()
            assert TORCH.current_device() is None
            with pytest.raises(ValueError, match="accelerator:0 is unavailable"):
                TORCH.require_device(AcceleratorDevice(0))


class TestJax:
    def test_cpu_install_has_no_accelerator(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "jax", fake_jax("cpu"))

        assert JAX.accelerator_devices() == ()
        assert JAX.available_devices() == (HOST,)
        with pytest.raises(ValueError, match=r"available devices are \('host',\)"):
            JAX.device_scope(AcceleratorDevice(0))

    @pytest.mark.parametrize("index", [-1, 2])
    def test_scope_rejects_an_accelerator_it_does_not_list(self, index, monkeypatch):
        monkeypatch.setitem(sys.modules, "jax", fake_jax("gpu", "gpu"))

        with pytest.raises(ValueError, match=f"device accelerator:{index} is unavailable"):
            JAX.device_scope(AcceleratorDevice(index))

    def test_accelerators_are_numbered_among_the_gpu_devices(self, monkeypatch):
        module = fake_jax("cpu", "gpu", "gpu")
        monkeypatch.setitem(sys.modules, "jax", module)
        cpu, _first_gpu, second_gpu = module.devices()

        assert JAX.accelerator_devices() == (AcceleratorDevice(0), AcceleratorDevice(1))
        assert JAX.device_of(SimpleNamespace(device=second_gpu)) == AcceleratorDevice(1)
        assert JAX.device_of(SimpleNamespace(device=cpu)) == HOST
        assert JAX.device_of(SimpleNamespace(device=lambda: second_gpu)) == AcceleratorDevice(1)


class TestPyclesperanto:
    def test_move_restores_the_selected_gpu(self, monkeypatch):
        module = FakePyclesperanto(current="gpu0")
        monkeypatch.setitem(sys.modules, "pyclesperanto", module)
        source = object()

        result = PYCLESPERANTO.move_to_device(source, AcceleratorDevice(1))

        assert result == ("copy", source)
        assert module.current == "gpu0"
        assert module.events == [("copy", source, result)]

    def test_scope_restores_a_device_that_is_not_a_listed_gpu(self, monkeypatch):
        module = FakePyclesperanto(current="cpu0")
        monkeypatch.setitem(sys.modules, "pyclesperanto", module)

        with PYCLESPERANTO.device_scope(AcceleratorDevice(1)):
            assert module.current == "gpu1"

        assert module.current == "cpu0"

    def test_array_on_an_unlisted_device_is_not_taken_for_gpu_zero(self, monkeypatch):
        module = FakePyclesperanto(current="cpu0")
        monkeypatch.setitem(sys.modules, "pyclesperanto", module)
        source = object()

        result = PYCLESPERANTO.move_to_device(source, AcceleratorDevice(0))

        assert result == ("copy", source)
        assert module.current == "cpu0"
        assert module.events == [("copy", source, result)]

    @pytest.mark.parametrize(
        ("current", "array_device", "expected"),
        [
            (SimpleNamespace(id=None, name="gpu1"), None, AcceleratorDevice(1)),
            ("gpu0", "gpu1", AcceleratorDevice(1)),
            (SimpleNamespace(id=0, name="cpu0"), None, NamedDevice("cpu0")),
            ("gpu0", "cpu0", NamedDevice("cpu0")),
        ],
    )
    def test_device_of_goes_by_the_device_name(self, current, array_device, expected, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyclesperanto", FakePyclesperanto(current=current))
        data = object() if array_device is None else FakePyclesperantoArray([1], array_device)

        assert PYCLESPERANTO.device_of(data) == expected


class TestCudaStreams:
    def test_only_cuda_libraries_have_streams(self):
        assert {library.name for library in LIBRARIES if hasattr(library, "thread_stream")} == {
            "cupy",
            "torch",
        }

    def test_streams_are_kept_per_thread_and_device(self, monkeypatch):
        module = FakeCupy()
        monkeypatch.setitem(sys.modules, "cupy", module)
        cupy = ArrayLibrary.named("cupy")

        first = cupy.thread_stream(AcceleratorDevice(0))
        second = cupy.thread_stream(AcceleratorDevice(1))
        with ThreadPoolExecutor(max_workers=1) as executor:
            other_thread, repeated = executor.submit(
                lambda: (
                    cupy.thread_stream(AcceleratorDevice(0)),
                    cupy.thread_stream(AcceleratorDevice(0)),
                )
            ).result()

        assert cupy.thread_stream(AcceleratorDevice(0)) is first
        assert cupy.thread_stream(AcceleratorDevice(1)) is second
        assert (first.device_index, second.device_index) == (0, 1)
        assert other_thread is repeated
        assert other_thread is not first
        assert module.current == 0

    def test_current_device_is_the_one_the_library_has_selected(self, monkeypatch):
        cupy_module = FakeCupy()
        cupy_module.current = 1
        monkeypatch.setitem(sys.modules, "cupy", cupy_module)
        monkeypatch.setitem(sys.modules, "torch", FakeTorch())

        assert ArrayLibrary.named("cupy").current_device() == AcceleratorDevice(1)
        assert TORCH.current_device() is None
