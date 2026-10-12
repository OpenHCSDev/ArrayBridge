"""Conversion routes: which mechanism is planned, which one ran, and what was recorded."""

import asyncio
import contextvars
import json
import pickle
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from arraybridge import conversion, libraries
from arraybridge.conversion import (
    ConversionCounters,
    ConversionObserver,
    ConversionRecord,
    ConversionRoute,
    DeviceMove,
    DLPackExchange,
    HostCopy,
    SharedHostBuffer,
    StaysInPlace,
)
from arraybridge.devices import HOST, AcceleratorDevice
from arraybridge.libraries import (
    AcceleratorDevices,
    ArrayLibrary,
    DLPackImport,
    DLPackPayload,
    HostDevices,
)
from tests.fakes import FakeCupy, FakeTorch, FakeTorchTensor

LIBRARIES = ArrayLibrary.all()
NUMPY, CUPY, TORCH, TENSORFLOW, JAX, PYCLESPERANTO = (
    ArrayLibrary.named(name)
    for name in ("numpy", "cupy", "torch", "tensorflow", "jax", "pyclesperanto")
)
GPU = AcceleratorDevice(0)


def placements():
    """Every (library, device) a conversion can start from or end at, by declaration."""

    return [
        (library, device)
        for library in LIBRARIES
        for role, device in ((HostDevices, HOST), (AcceleratorDevices, GPU))
        if isinstance(library, role)
    ]


def planned(source, source_device, target, target_device):
    return ConversionRoute.planned(source, source_device, target, target_device).mechanism


class TestPlanning:
    def test_routes_are_tried_in_declaration_order(self):
        assert list(ConversionRoute.__registry__.items()) == [
            ("stays in place", StaysInPlace),
            ("device move", DeviceMove),
            ("dlpack", DLPackExchange),
            ("shared host buffer", SharedHostBuffer),
            ("host copy", HostCopy),
        ]

    @pytest.mark.parametrize(("library", "device"), placements(), ids=str)
    def test_same_library_and_device_stays_in_place(self, library, device):
        assert ConversionRoute.planned(library, device, library, device) is StaysInPlace
        assert not DeviceMove.applies(library, device, library, device)
        assert StaysInPlace.writes_a_record is False
        assert all(
            route.writes_a_record
            for route in ConversionRoute.__registry__.values()
            if route is not StaysInPlace
        )

    @pytest.mark.parametrize(("source", "source_device"), placements(), ids=str)
    def test_every_pair_has_a_route_ending_in_a_host_copy(self, source, source_device):
        for target, target_device in placements():
            candidates = ConversionRoute.candidates(source, source_device, target, target_device)
            mechanism = planned(source, source_device, target, target_device)

            assert candidates[0].mechanism == mechanism
            assert candidates[-1] is HostCopy
            if source is target:
                same_device = source_device == target_device
                assert mechanism == ("stays in place" if same_device else "device move")
            elif PYCLESPERANTO in (source, target) or GPU in (source_device, target_device):
                assert mechanism in ("dlpack", "host copy")
            elif NUMPY in (source, target):
                assert mechanism in ("shared host buffer", "host copy")
            else:
                assert mechanism in ("dlpack", "shared host buffer", "host copy")

    @pytest.mark.parametrize(
        ("source", "source_device", "target", "target_device", "mechanism"),
        [
            (CUPY, GPU, TORCH, GPU, "dlpack"),
            (CUPY, GPU, TORCH, HOST, "dlpack"),
            (TORCH, GPU, CUPY, GPU, "dlpack"),
            (TORCH, HOST, CUPY, GPU, "host copy"),
            (TORCH, HOST, JAX, HOST, "dlpack"),
            (TORCH, GPU, JAX, GPU, "dlpack"),
            (JAX, HOST, TORCH, HOST, "dlpack"),
            (JAX, GPU, TORCH, GPU, "dlpack"),
            (JAX, HOST, CUPY, GPU, "host copy"),
            (TENSORFLOW, GPU, TORCH, GPU, "dlpack"),
            (TENSORFLOW, HOST, TORCH, HOST, "host copy"),
            (TORCH, HOST, TENSORFLOW, HOST, "dlpack"),
            (PYCLESPERANTO, GPU, TORCH, GPU, "host copy"),
            (TORCH, GPU, PYCLESPERANTO, GPU, "host copy"),
            (TORCH, HOST, TORCH, GPU, "device move"),
            (TORCH, GPU, TORCH, HOST, "device move"),
            (TORCH, GPU, TORCH, GPU, "stays in place"),
            (NUMPY, HOST, NUMPY, HOST, "stays in place"),
            (NUMPY, HOST, TORCH, HOST, "shared host buffer"),
            (TORCH, HOST, NUMPY, HOST, "shared host buffer"),
            (NUMPY, HOST, JAX, HOST, "shared host buffer"),
            (JAX, HOST, NUMPY, HOST, "shared host buffer"),
            (NUMPY, HOST, TENSORFLOW, HOST, "host copy"),
            (TENSORFLOW, HOST, NUMPY, HOST, "host copy"),
            (TENSORFLOW, HOST, JAX, HOST, "host copy"),
            (NUMPY, HOST, TORCH, GPU, "host copy"),
            (TORCH, GPU, NUMPY, HOST, "host copy"),
            (NUMPY, HOST, CUPY, GPU, "host copy"),
            (NUMPY, HOST, PYCLESPERANTO, GPU, "host copy"),
        ],
        ids=str,
    )
    def test_expected_mechanism(self, source, source_device, target, target_device, mechanism):
        assert planned(source, source_device, target, target_device) == mechanism

    def test_planning_every_pair_imports_no_array_library(self, loaded_libraries_after):
        loaded = loaded_libraries_after(
            "from arraybridge.conversion import ConversionRoute\n"
            "from arraybridge.devices import HOST, AcceleratorDevice\n"
            "from arraybridge.libraries import AcceleratorDevices, ArrayLibrary, HostDevices\n"
            "placements = [\n"
            "    (library, device)\n"
            "    for library in ArrayLibrary.all()\n"
            "    for role, device in (\n"
            "        (HostDevices, HOST), (AcceleratorDevices, AcceleratorDevice(0))\n"
            "    )\n"
            "    if isinstance(library, role)\n"
            "]\n"
            "assert len(placements) == 9, placements\n"
            "for source in placements:\n"
            "    for target in placements:\n"
            "        route = ConversionRoute.planned(*source, *target)\n"
            "        assert issubclass(route, ConversionRoute) and route.mechanism\n"
        )
        assert loaded == "[]"


class TestDLPack:
    def test_libraries_that_wrap_foreign_buffers(self):
        importers = {library.name for library in LIBRARIES if library.imports_dlpack}

        assert importers == {"cupy", "torch", "tensorflow", "jax"}
        assert importers == {
            library.name for library in LIBRARIES if isinstance(library, DLPackImport)
        }
        assert not CUPY.imports_dlpack_from(HOST) and CUPY.imports_dlpack_from(GPU)
        assert TORCH.imports_dlpack_from(HOST) and TORCH.imports_dlpack_from(GPU)

    @pytest.mark.parametrize(
        ("version", "device", "exportable"),
        [
            ("2.15.0", "GPU:0", True),
            ("2.10.0", "GPU:0", False),
            ("2.15.0", "CPU:0", False),
            ("unknown", "GPU:0", False),
        ],
    )
    def test_tensorflow_exports_accelerator_tensors_of_recent_versions(
        self, version, device, exportable, monkeypatch
    ):
        capsule = object()
        module = SimpleNamespace(
            __version__=version,
            experimental=SimpleNamespace(dlpack=SimpleNamespace(to_dlpack=lambda data: capsule)),
        )
        monkeypatch.setitem(sys.modules, "tensorflow", module)
        data = SimpleNamespace(device=device)

        payload = TENSORFLOW.export_dlpack(data)

        assert TENSORFLOW.exports_dlpack_data(data) is exportable
        if exportable:
            assert (payload.source, payload.capsule) == (data, capsule)
        else:
            assert payload is None

    def test_tensorflow_without_an_exporter_exports_nothing(self, monkeypatch):
        module = SimpleNamespace(
            __version__="2.15.0", experimental=SimpleNamespace(dlpack=object())
        )
        monkeypatch.setitem(sys.modules, "tensorflow", module)

        assert not TENSORFLOW.exports_dlpack_data(SimpleNamespace(device="GPU:0"))

    @pytest.mark.parametrize(
        ("library", "importer_path", "received"),
        [
            (TENSORFLOW, ("experimental", "dlpack", "from_dlpack"), "capsule"),
            (TORCH, ("from_dlpack",), "capsule"),
            (CUPY, ("from_dlpack",), "source"),
        ],
        ids=str,
    )
    def test_importer_receives_what_the_library_reads(
        self, library, importer_path, received, monkeypatch
    ):
        capsule = object()
        source = SimpleNamespace(__dlpack__=lambda: capsule, __dlpack_device__=lambda: (2, 0))
        imported = object()
        calls = []
        module = lambda value: calls.append(value) or imported  # noqa: E731
        for attribute in reversed(importer_path):
            module = SimpleNamespace(**{attribute: module})
        monkeypatch.setitem(sys.modules, library.import_name, module)

        assert library.from_dlpack(DLPackPayload(source, capsule)) is imported
        assert calls == [{"capsule": capsule, "source": source}[received]]

    def test_bare_capsule_is_its_own_source(self, monkeypatch):
        capsule = object()
        calls = []
        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(from_dlpack=calls.append))

        TORCH.from_dlpack(capsule)

        assert calls == [capsule]

    def test_cupy_reads_a_capsule_only_with_its_capsule_importer(self, monkeypatch):
        payload = DLPackPayload(source=object(), capsule=object())
        imported = object()
        monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
        assert CUPY.from_dlpack(payload) is NotImplemented

        monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(fromDlpack=lambda _: imported))
        assert CUPY.from_dlpack(payload) is imported

    def test_jax_declines_a_capsule_without_its_source_array(self):
        payload = DLPackPayload(source=object(), capsule=object())

        assert JAX.from_dlpack(payload) is NotImplemented

    @pytest.mark.parametrize("exporter_name", ["__dlpack__", "to_dlpack", "toDlpack"])
    def test_protocol_export_accepts_each_exporter_name(self, exporter_name):
        capsule = object()
        data = SimpleNamespace(**{exporter_name: lambda: capsule})

        payload = CUPY.export_dlpack(data)

        assert CUPY.exports_dlpack_data(data)
        assert (payload.source, payload.capsule) == (data, capsule)
        assert CUPY.export_dlpack(object()) is None
        assert not CUPY.exports_dlpack_data(object())

    @pytest.mark.torch
    @pytest.mark.jax
    def test_jax_import_copies_a_misaligned_torch_view(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("jax")
        source = torch.arange(9, dtype=torch.int32).to(torch.uint16)[1:]
        assert source.data_ptr() % 16 != 0

        result = TORCH.convert_to(source, JAX, HOST)

        assert np.array_equal(np.asarray(result), np.arange(1, 9, dtype=np.uint16))


@pytest.fixture
def cupy_module(monkeypatch):
    module = FakeCupy()
    monkeypatch.setitem(sys.modules, "cupy", module)
    return module


@pytest.fixture
def torch_module(monkeypatch):
    module = FakeTorch(device_count=1)
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


class TestExecution:
    """The route that ran is the planned one, or the next candidate when it declined."""

    def test_dlpack_shares_the_buffer(self, cupy_module, torch_module, recorded):
        source = cupy_module.array(np.arange(6, dtype=np.float32))
        assert planned(CUPY, GPU, TORCH, GPU) == "dlpack"

        result = CUPY.convert_to(source, TORCH, GPU)

        assert result.values is source.values
        assert TORCH.device_of(result) == GPU
        (record,) = recorded
        assert record == ConversionRecord(
            "cupy", "accelerator:0", "torch", "accelerator:0", "dlpack", 24, record.seconds
        )
        assert record.declined_mechanisms == ()

    def test_failed_dlpack_import_falls_back_to_a_host_copy(
        self, cupy_module, torch_module, recorded, monkeypatch, caplog
    ):
        source = cupy_module.array(np.arange(6, dtype=np.float32))

        def cannot_import(capsule):
            raise RuntimeError("capsule rejected")

        monkeypatch.setattr(torch_module, "from_dlpack", cannot_import)
        assert planned(CUPY, GPU, TORCH, HOST) == "dlpack"

        result = CUPY.convert_to(source, TORCH, HOST)

        assert isinstance(result, FakeTorchTensor)
        assert TORCH.device_of(result) == HOST
        assert result.values is not source.values
        np.testing.assert_array_equal(result.values, source.values)
        (record,) = recorded
        assert record.mechanism == "host copy"
        assert record.declined_mechanisms == ("dlpack",)
        assert (record.source_library, record.source_device) == ("cupy", "accelerator:0")
        assert (record.target_library, record.target_device) == ("torch", "host")
        assert "capsule rejected" in caplog.text

    def test_array_without_an_exporter_falls_back_to_a_host_copy(
        self, cupy_module, torch_module, recorded
    ):
        source = SimpleNamespace(
            device=SimpleNamespace(id=0), get=lambda: np.arange(3, dtype=np.uint8)
        )

        result = CUPY.convert_to(source, TORCH, HOST)

        np.testing.assert_array_equal(result.values, [0, 1, 2])
        assert [(each.mechanism, each.declined_mechanisms) for each in recorded] == [
            ("host copy", ("dlpack",))
        ]

    def test_no_route_left_is_an_error(self, cupy_module, torch_module, monkeypatch):
        monkeypatch.setattr(ConversionRoute, "candidates", lambda *pair: ())

        with pytest.raises(LookupError, match="No conversion route from cupy to torch"):
            ConversionRoute.run(cupy_module.array([1]), CUPY, TORCH, HOST)


class TestRecords:
    values = np.arange(12, dtype=np.uint16).reshape(3, 4)

    def test_default_observer_asks_for_no_records(self, torch_module, monkeypatch):
        def no_record(**fields):
            pytest.fail(f"a record was built: {fields}")

        monkeypatch.setattr(conversion, "ConversionRecord", no_record)
        assert not hasattr(libraries, "ConversionRecord")

        assert type(ConversionObserver.current) is conversion.IgnoreConversions
        assert ConversionObserver.current.wants_records is False
        tensor = NUMPY.convert_to(self.values, TORCH, HOST)
        np.testing.assert_array_equal(TORCH.to_numpy(tensor), self.values)
        TORCH.from_numpy(self.values, HOST)

    def test_install_replaces_the_observer_until_it_is_put_back(self):
        default = ConversionObserver.current
        counters = ConversionCounters()

        assert counters.install() is default
        assert ConversionObserver.current is counters
        assert default.install() is counters
        with counters.installed() as installed:
            assert installed is counters is ConversionObserver.current
        assert ConversionObserver.current is default

    @pytest.mark.parametrize(
        ("convert", "source", "target"),
        [
            (lambda: NUMPY.convert_to(TestRecords.values, TORCH, HOST), "numpy", "torch"),
            (lambda: TORCH.from_numpy(TestRecords.values, HOST), "numpy", "torch"),
            (lambda: TORCH.to_numpy(FakeTorchTensor(TestRecords.values)), "torch", "numpy"),
            (
                lambda: TORCH.convert_to(FakeTorchTensor(TestRecords.values), NUMPY, HOST),
                "torch",
                "numpy",
            ),
        ],
        ids=["convert_to", "from_numpy", "to_numpy", "convert_to numpy"],
    )
    def test_each_conversion_writes_one_record(
        self, convert, source, target, torch_module, recorded
    ):
        convert()

        (record,) = recorded
        assert record == ConversionRecord(
            source_library=source,
            source_device="host",
            target_library=target,
            target_device="host",
            mechanism="shared host buffer",
            nbytes=self.values.nbytes,
            seconds=record.seconds,
        )
        assert record.seconds >= 0

    def test_upload_to_an_accelerator_names_the_device(self, torch_module, recorded):
        TORCH.from_numpy(self.values, GPU)

        assert [(each.target_library, each.target_device) for each in recorded] == [
            ("torch", "accelerator:0")
        ]

    def test_returning_the_input_writes_no_record(self, torch_module, recorded):
        tensor = FakeTorchTensor(self.values)

        assert NUMPY.to_numpy(self.values) is self.values
        assert NUMPY.from_numpy(self.values, HOST) is self.values
        assert NUMPY.convert_to(self.values, NUMPY, HOST) is self.values
        assert TORCH.convert_to(tensor, TORCH, HOST) is tensor
        assert TORCH.move_to_device(tensor, HOST) is tensor

        assert recorded == []
        assert planned(NUMPY, HOST, NUMPY, HOST) == planned(TORCH, HOST, TORCH, HOST)
        assert planned(TORCH, HOST, TORCH, HOST) == "stays in place"

    def test_a_copy_where_sharing_was_planned_is_recorded_as_a_copy(
        self, torch_module, recorded, monkeypatch
    ):
        monkeypatch.setattr(
            torch_module, "from_numpy", lambda data: FakeTorchTensor(np.array(data))
        )
        assert planned(NUMPY, HOST, TORCH, HOST) == "shared host buffer"

        tensor = TORCH.from_numpy(self.values, HOST)

        assert not np.shares_memory(tensor.values, self.values)
        assert [(each.mechanism, each.declined_mechanisms) for each in recorded] == [
            ("host copy", ("shared host buffer",))
        ]

    def test_an_empty_array_shares_nothing(self, torch_module, recorded):
        TORCH.from_numpy(np.zeros((0, 3), dtype=np.uint8), HOST)

        assert [(each.mechanism, each.nbytes) for each in recorded] == [("host copy", 0)]

    def test_device_move_is_recorded(self, torch_module, recorded):
        moved = TORCH.convert_to(FakeTorchTensor(self.values), TORCH, GPU)

        assert TORCH.device_of(moved) == GPU
        assert [(each.mechanism, each.source_device, each.target_device) for each in recorded] == [
            ("device move", "host", "accelerator:0")
        ]

    def test_record_is_plain_data(self):
        record = ConversionRecord(
            "cupy", "accelerator:0", "torch", "host", "host copy", 24, 0.5, ("dlpack",)
        )

        assert pickle.loads(pickle.dumps(record)) == record
        assert ConversionRecord.from_json(json.loads(json.dumps(record.to_json()))) == record
        assert hash(record) == hash(pickle.loads(pickle.dumps(record)))


def address(array):
    return array.ctypes.data


def buffer_starting_at(alignment, offset, count=256):
    """A writable float32 array whose first byte is ``offset`` past a multiple of ``alignment``."""

    raw = np.zeros(count * 4 + 2 * alignment, dtype=np.uint8)
    start = (-address(raw)) % alignment + offset
    array = raw[start : start + count * 4].view(np.float32)
    array[:] = np.arange(count, dtype=np.float32)
    assert address(array) % alignment == offset
    return array


class TestHostBufferIsSharedOrCopied:
    """What the record says is what the memory shows, measured on the installed library."""

    plain = staticmethod(lambda: np.arange(24, dtype=np.float32).reshape(4, 6))

    @staticmethod
    def one(recorded):
        (record,) = recorded
        recorded.clear()
        return record.mechanism, record.declined_mechanisms

    @pytest.mark.torch
    @pytest.mark.parametrize(
        "make",
        [
            lambda: np.arange(24, dtype=np.float32).reshape(4, 6),
            lambda: np.arange(24, dtype=np.uint16).reshape(4, 6),
            lambda: np.arange(24, dtype=np.int64).reshape(4, 6) > 3,
            lambda: np.asfortranarray(np.arange(24, dtype=np.float64).reshape(4, 6)),
            lambda: np.arange(24, dtype=np.float32).reshape(4, 6)[1:, ::2],
        ],
        ids=["float32", "uint16", "bool", "fortran order", "offset stepped view"],
    )
    def test_torch_and_numpy_share_the_buffer_both_ways(self, make, recorded):
        pytest.importorskip("torch")
        values = make()

        tensor = TORCH.from_numpy(values, HOST)
        assert tensor.data_ptr() == address(values)
        assert self.one(recorded) == ("shared host buffer", ())

        back = TORCH.to_numpy(tensor)
        assert address(back) == tensor.data_ptr()
        assert np.shares_memory(back, values)
        assert self.one(recorded) == ("shared host buffer", ())

        values[0, 0] = not values[0, 0]
        assert back[0, 0] == values[0, 0] == tensor[0, 0].item()

    @pytest.mark.torch
    def test_torch_copies_a_reversed_view_and_the_record_says_copy(self, recorded):
        pytest.importorskip("torch")
        values = self.plain()[::-1]

        tensor = TORCH.from_numpy(values, HOST)

        assert not np.shares_memory(tensor.numpy(), values)
        assert self.one(recorded) == ("host copy", ("shared host buffer",))
        np.testing.assert_array_equal(tensor.numpy(), values)

    @pytest.mark.torch
    def test_torch_shares_a_read_only_buffer(self, recorded):
        pytest.importorskip("torch")
        values = self.plain()
        values.setflags(write=False)

        with pytest.warns(UserWarning, match="not writable"):
            tensor = TORCH.from_numpy(values, HOST)

        assert tensor.data_ptr() == address(values)
        assert self.one(recorded) == ("shared host buffer", ())

    @pytest.mark.jax
    def test_jax_hands_numpy_its_own_buffer(self, recorded):
        pytest.importorskip("jax")
        array = JAX.from_numpy(self.plain(), HOST)
        recorded.clear()

        first, second = JAX.to_numpy(array), JAX.to_numpy(array)

        assert address(first) == address(second) == array.unsafe_buffer_pointer()
        assert np.shares_memory(first, second)
        assert not first.flags.writeable
        assert [each.mechanism for each in recorded] == ["shared host buffer"] * 2

    @pytest.mark.jax
    @pytest.mark.parametrize(
        ("offset", "expected"),
        [(0, ("shared host buffer", ())), (4, ("host copy", ("shared host buffer",)))],
        ids=["64-byte aligned", "not aligned"],
    )
    def test_jax_wraps_a_numpy_buffer_only_when_it_is_aligned(self, offset, expected, recorded):
        pytest.importorskip("jax")
        values = buffer_starting_at(64, offset)

        array = JAX.from_numpy(values, HOST)

        shares = array.unsafe_buffer_pointer() == address(values)
        assert shares == np.shares_memory(np.asarray(array), values)
        assert shares == (expected[0] == "shared host buffer")
        assert self.one(recorded) == expected

    @pytest.mark.tensorflow
    @pytest.mark.parametrize("offset", [0, 4], ids=["64-byte aligned", "not aligned"])
    def test_tensorflow_copies_both_ways(self, offset, recorded):
        pytest.importorskip("tensorflow")
        values = buffer_starting_at(64, offset)
        assert planned(NUMPY, HOST, TENSORFLOW, HOST) == "host copy"
        assert planned(TENSORFLOW, HOST, NUMPY, HOST) == "host copy"

        tensor = TENSORFLOW.from_numpy(values, HOST)
        assert self.one(recorded) == ("host copy", ())
        first, second = TENSORFLOW.to_numpy(tensor), TENSORFLOW.to_numpy(tensor)

        assert not np.shares_memory(first, values)
        assert not np.shares_memory(first, second)
        values[0] += 1
        assert first[0] == second[0] == values[0] - 1
        assert [each.mechanism for each in recorded] == ["host copy"] * 2

    @pytest.mark.torch
    @pytest.mark.tensorflow
    def test_tensorflow_to_torch_on_the_host_is_a_copy(self, recorded):
        pytest.importorskip("torch")
        pytest.importorskip("tensorflow")
        tensor = TENSORFLOW.from_numpy(self.plain(), HOST)
        recorded.clear()

        result = TENSORFLOW.convert_to(tensor, TORCH, HOST)

        assert not np.shares_memory(result.numpy(), TENSORFLOW.unrecorded_to_numpy(tensor))
        assert self.one(recorded) == ("host copy", ())


def counted(counters):
    return [(row["mechanism"], row["nbytes"]) for row in counters.rows()]


class TestObserverBelongsToItsContext:
    """Each thread and each asyncio task has its own ``ConversionObserver.current``."""

    def test_threads_that_leave_in_entry_order_keep_their_own_observers(self, torch_module):
        default = ConversionObserver.current
        first_entered, second_entered, first_left = (threading.Event() for _ in range(3))
        observers = {name: ConversionCounters() for name in "AB"}
        current_inside, current_after, failures = {}, {}, []

        def work(name, nbytes):
            try:
                # Forced order: A enters, B enters, A leaves, B leaves.
                assert name == "A" or first_entered.wait(10)
                with observers[name].installed():
                    (first_entered if name == "A" else second_entered).set()
                    assert (second_entered if name == "A" else first_left).wait(10)
                    current_inside[name] = ConversionObserver.current
                    TORCH.from_numpy(np.zeros(nbytes, dtype=np.uint8), HOST)
                current_after[name] = ConversionObserver.current
                if name == "A":
                    first_left.set()
            except BaseException as error:  # reported by the main thread
                failures.append(error)

        threads = [threading.Thread(target=work, args=pair) for pair in (("A", 8), ("B", 16))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(20)

        assert failures == []
        assert current_inside == observers
        assert counted(observers["A"]) == [("shared host buffer", 8)]
        assert counted(observers["B"]) == [("shared host buffer", 16)]
        assert type(current_after["A"]) is type(current_after["B"]) is type(default)
        assert current_after["A"] is current_after["B"] is default is ConversionObserver.current

    def test_a_new_thread_starts_with_the_default_observer(self, torch_module):
        default = ConversionObserver.current
        seen = []

        def work():
            seen.append(ConversionObserver.current)
            TORCH.from_numpy(np.zeros(8, dtype=np.uint8), HOST)

        with ConversionCounters().installed() as counters:
            thread = threading.Thread(target=work)
            thread.start()
            thread.join(20)

        assert seen == [default]
        assert counters.rows() == []

    def test_a_thread_run_in_a_copied_context_reports_to_the_spawner(self, torch_module):
        seen = []

        def work():
            seen.append(ConversionObserver.current)
            TORCH.from_numpy(np.zeros(8, dtype=np.uint8), HOST)

        with ConversionCounters().installed() as counters:
            thread = threading.Thread(target=contextvars.copy_context().run, args=(work,))
            thread.start()
            thread.join(20)

        assert seen == [counters]
        assert counted(counters) == [("shared host buffer", 8)]

    def test_assigning_current_changes_this_context_only(self):
        default = ConversionObserver.current
        counters = ConversionCounters()
        seen = []

        def assign():
            ConversionObserver.current = counters
            seen.append(ConversionObserver.current)

        contextvars.copy_context().run(assign)
        thread = threading.Thread(target=assign)
        thread.start()
        thread.join(20)

        assert seen == [counters, counters]
        assert ConversionObserver.current is default
        assert isinstance(vars(type(ConversionObserver))["current"], property)

    def test_asyncio_tasks_keep_their_own_observers(self, torch_module):
        default = ConversionObserver.current

        async def convert(nbytes, started, may_leave):
            with ConversionCounters().installed() as counters:
                started.set()
                await may_leave.wait()
                TORCH.from_numpy(np.zeros(nbytes, dtype=np.uint8), HOST)
                assert ConversionObserver.current is counters
            return counters, ConversionObserver.current

        async def inherited():
            TORCH.from_numpy(np.zeros(32, dtype=np.uint8), HOST)
            return ConversionObserver.current

        async def main():
            a_started, b_started, a_leaves, b_leaves = (asyncio.Event() for _ in range(4))
            a = asyncio.create_task(convert(8, a_started, a_leaves))
            await a_started.wait()
            b = asyncio.create_task(convert(16, b_started, b_leaves))
            await b_started.wait()
            a_leaves.set()
            a_result = await a
            b_leaves.set()
            b_result = await b
            with ConversionCounters().installed() as spawner:
                child_saw = await asyncio.create_task(inherited())
            return a_result, b_result, spawner, child_saw, ConversionObserver.current

        (a, a_after), (b, b_after), spawner, child_saw, after = asyncio.run(main())

        assert counted(a) == [("shared host buffer", 8)]
        assert counted(b) == [("shared host buffer", 16)]
        assert a_after is b_after is after is default
        assert child_saw is spawner
        assert counted(spawner) == [("shared host buffer", 32)]
        assert ConversionObserver.current is default


class TestCounters:
    @staticmethod
    def record(source="numpy", target="torch", nbytes=10, seconds=0.25, mechanism="host copy"):
        return ConversionRecord(source, "host", target, "accelerator:0", mechanism, nbytes, seconds)

    def counters(self, *records):
        counters = ConversionCounters()
        for record in records:
            counters.record(record)
        return counters

    def test_totals_are_kept_per_kind_of_conversion(self):
        counters = self.counters(
            self.record(), self.record(nbytes=30), self.record("cupy", mechanism="dlpack")
        )

        assert counters.rows() == [
            {
                "source_library": "numpy",
                "source_device": "host",
                "target_library": "torch",
                "target_device": "accelerator:0",
                "mechanism": "host copy",
                "count": 2,
                "nbytes": 40,
                "seconds": 0.5,
            },
            {
                "source_library": "cupy",
                "source_device": "host",
                "target_library": "torch",
                "target_device": "accelerator:0",
                "mechanism": "dlpack",
                "count": 1,
                "nbytes": 10,
                "seconds": 0.25,
            },
        ]

    def test_installed_counters_count_conversions(self, torch_module):
        values = np.zeros(8, dtype=np.uint8)

        with ConversionCounters().installed() as counters:
            for _ in range(3):
                TORCH.from_numpy(values, HOST)

        (row,) = counters.rows()
        assert (row["count"], row["nbytes"], row["mechanism"]) == (3, 24, "shared host buffer")

    def test_merge_sums_counts_bytes_and_seconds(self):
        parent = self.counters(self.record(), self.record("cupy", nbytes=5))
        worker = self.counters(self.record(nbytes=30, seconds=1.0), self.record("jax", nbytes=7))

        assert parent.merge(worker) is parent

        totals = {
            row["source_library"]: (row["count"], row["nbytes"], row["seconds"])
            for row in parent.rows()
        }
        assert totals == {"numpy": (2, 40, 1.25), "cupy": (1, 5, 0.25), "jax": (1, 7, 0.25)}
        assert len(worker.rows()) == 2

    def test_counters_are_plain_data(self):
        counters = self.counters(self.record(), self.record("cupy", mechanism="dlpack"))

        assert pickle.loads(pickle.dumps(counters)) == counters
        assert ConversionCounters.from_json(json.loads(json.dumps(counters.to_json()))) == counters

    def test_table_has_one_line_per_row(self):
        counters = self.counters(
            self.record(nbytes=2_500_000), self.record("cupy", mechanism="dlpack")
        )

        header, *lines = counters.format_table().splitlines()

        assert header.split() == ["source", "target", "mechanism", "count", "MB", "seconds"]
        assert len(lines) == len(counters.rows())
        for line, row in zip(lines, counters.rows(), strict=True):
            for field in ("source_library", "source_device", "target_library", "target_device"):
                assert row[field] in line
            assert row["mechanism"] in line
            assert f"{row['count']:>7}" in line
            assert f"{row['nbytes'] / 1e6:.1f}" in line
            assert f"{row['seconds']:.3f}" in line
