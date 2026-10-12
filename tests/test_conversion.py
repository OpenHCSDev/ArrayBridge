"""Conversion routes: which mechanism is planned, which one ran, and what was recorded."""

import json
import pickle
import sys
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
            ("device move", DeviceMove),
            ("dlpack", DLPackExchange),
            ("host copy", HostCopy),
        ]

    @pytest.mark.parametrize(("source", "source_device"), placements(), ids=str)
    def test_every_pair_has_a_route_ending_in_a_host_copy(self, source, source_device):
        for target, target_device in placements():
            candidates = ConversionRoute.candidates(source, source_device, target, target_device)
            mechanism = planned(source, source_device, target, target_device)

            assert candidates[0].mechanism == mechanism
            assert candidates[-1] is HostCopy
            if source is target:
                assert mechanism == "device move"
            elif source is NUMPY or target in (NUMPY, PYCLESPERANTO):
                assert mechanism == "host copy"
            else:
                assert mechanism in ("dlpack", "host copy")

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
        monkeypatch.setattr(libraries, "ConversionRecord", no_record)

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
            mechanism="host copy",
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

        assert recorded == []

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
        assert (row["count"], row["nbytes"], row["mechanism"]) == (3, 24, "host copy")

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
