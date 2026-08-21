"""Tests for arraybridge.types module."""

import os
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from arraybridge.types import (
    CPU_MEMORY_TYPES,
    GPU_MEMORY_TYPES,
    SUPPORTED_MEMORY_TYPES,
    VALID_MEMORY_TYPES,
    DLPackPayload,
    MemoryContractAttribute,
    MemoryType,
)


class TestMemoryContractAttribute:
    """Tests for the declaration-owned callable metadata protocol."""

    def test_reads_and_writes_object_namespaces(self):
        namespace = SimpleNamespace()

        MemoryContractAttribute.EXECUTION.write(namespace, "torch")

        assert MemoryContractAttribute.EXECUTION.read(namespace) == "torch"

    def test_reads_and_writes_mapping_namespaces(self):
        namespace = {}

        MemoryContractAttribute.INPUT.write(namespace, "numpy")

        assert MemoryContractAttribute.INPUT.read(namespace) == "numpy"
        assert MemoryContractAttribute.OUTPUT.read(namespace) is None


class TestMemoryType:
    """Tests for MemoryType enum."""

    def test_memory_type_values(self):
        """Test that MemoryType enum has expected values."""
        assert MemoryType.NUMPY.value == "numpy"
        assert MemoryType.CUPY.value == "cupy"
        assert MemoryType.TORCH.value == "torch"
        assert MemoryType.TENSORFLOW.value == "tensorflow"
        assert MemoryType.JAX.value == "jax"
        assert MemoryType.PYCLESPERANTO.value == "pyclesperanto"

    def test_memory_type_from_string(self):
        """Test creating MemoryType from string."""
        assert MemoryType("numpy") == MemoryType.NUMPY
        assert MemoryType("torch") == MemoryType.TORCH
        assert MemoryType("cupy") == MemoryType.CUPY

    def test_invalid_memory_type_raises_error(self):
        """Test that invalid memory type raises ValueError."""
        with pytest.raises(ValueError):
            MemoryType("invalid_type")

    def test_cpu_memory_types(self):
        """Test that CPU_MEMORY_TYPES contains only NumPy."""
        assert CPU_MEMORY_TYPES == {MemoryType.NUMPY}
        assert isinstance(CPU_MEMORY_TYPES, frozenset)
        assert len(CPU_MEMORY_TYPES) == 1

    def test_gpu_memory_types(self):
        """Test that GPU_MEMORY_TYPES contains all GPU frameworks."""
        expected_gpu_types = {
            MemoryType.CUPY,
            MemoryType.TORCH,
            MemoryType.TENSORFLOW,
            MemoryType.JAX,
            MemoryType.PYCLESPERANTO,
        }
        assert GPU_MEMORY_TYPES == expected_gpu_types
        assert isinstance(GPU_MEMORY_TYPES, frozenset)
        assert len(GPU_MEMORY_TYPES) == 5

    def test_supported_memory_types(self):
        """Test that SUPPORTED_MEMORY_TYPES contains all types."""
        assert SUPPORTED_MEMORY_TYPES == CPU_MEMORY_TYPES | GPU_MEMORY_TYPES
        assert len(SUPPORTED_MEMORY_TYPES) == 6

    def test_valid_memory_types_strings(self):
        """Test that VALID_MEMORY_TYPES contains string values."""
        expected_strings = {"numpy", "cupy", "torch", "tensorflow", "jax", "pyclesperanto"}
        assert VALID_MEMORY_TYPES == expected_strings
        assert isinstance(VALID_MEMORY_TYPES, frozenset)


class TestMemoryTypeOwnership:
    """Tests for declaration-owned framework and device capabilities."""

    def test_array_operations_are_owned_by_enum_members(self):
        assert not hasattr(MemoryType.NUMPY, "converter")
        assert callable(MemoryType.NUMPY.to_numpy)
        assert callable(MemoryType.NUMPY.from_numpy)
        assert callable(MemoryType.NUMPY.stack_arrays)
        assert callable(MemoryType.NUMPY.scale_dtype)

    def test_framework_identity_is_owned_by_members(self):
        assert MemoryType.NUMPY.import_name == "numpy"
        assert MemoryType.NUMPY.display_name == "NumPy"
        assert MemoryType.NUMPY.is_gpu is False
        assert MemoryType.JAX.recognized_module_names == frozenset({"jax", "jaxlib"})
        assert all(memory_type.is_gpu for memory_type in GPU_MEMORY_TYPES)

    def test_absent_optional_framework_is_not_imported(self, monkeypatch):
        from arraybridge import types

        monkeypatch.delitem(types.sys.modules, "cupy", raising=False)
        monkeypatch.setattr(types.importlib.util, "find_spec", lambda name: None)

        def fail_import(name):
            raise AssertionError(f"unexpected import of {name}")

        monkeypatch.setattr(types.importlib, "import_module", fail_import)

        assert MemoryType.CUPY.import_if_installed() is None
        assert MemoryType.CUPY.available_device_ids() == ()
        MemoryType.CUPY.cleanup_loaded()

    def test_absent_framework_does_not_apply_import_environment(self, monkeypatch):
        from arraybridge import types

        monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
        monkeypatch.delitem(types.sys.modules, "jax", raising=False)
        monkeypatch.setattr(types.importlib.util, "find_spec", lambda name: None)

        assert MemoryType.JAX.import_if_installed() is None
        assert "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ

    def test_import_preparation_sets_only_declaration_owned_defaults(self, monkeypatch):
        monkeypatch.delenv("TF_FORCE_GPU_ALLOW_GROWTH", raising=False)
        monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "custom")

        MemoryType.TENSORFLOW.prepare_import()
        MemoryType.JAX.prepare_import()

        assert MemoryType.TENSORFLOW.import_environment == (("TF_FORCE_GPU_ALLOW_GROWTH", "true"),)
        assert MemoryType.JAX.import_environment == (("XLA_PYTHON_CLIENT_PREALLOCATE", "false"),)
        assert os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"
        assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "custom"

    def test_dlpack_capability_is_owned_by_members(self):
        assert {memory_type for memory_type in MemoryType if memory_type.supports_dlpack} == {
            MemoryType.CUPY,
            MemoryType.TORCH,
            MemoryType.TENSORFLOW,
            MemoryType.JAX,
        }

    @pytest.mark.parametrize(
        ("version", "device"),
        [
            ("2.10.0", "GPU:0"),
            ("2.15.0", "CPU:0"),
            ("unknown", "GPU:0"),
        ],
    )
    def test_tensorflow_dlpack_validation_is_declaration_owned(
        self,
        version,
        device,
    ):
        module = SimpleNamespace(
            __version__=version,
            experimental=SimpleNamespace(dlpack=object()),
        )
        data = SimpleNamespace(device=device, __dlpack__=lambda: object())

        assert not MemoryType.TENSORFLOW.supports_dlpack_data(data, module)

    def test_tensorflow_dlpack_validation_accepts_supported_gpu_tensor(self):
        def exporter(data):
            return object()

        module = SimpleNamespace(
            __version__="2.15.0",
            experimental=SimpleNamespace(dlpack=SimpleNamespace(to_dlpack=exporter)),
        )
        data = SimpleNamespace(device="GPU:0")

        assert MemoryType.TENSORFLOW.supports_dlpack_data(data, module)

    def test_tensorflow_dlpack_export_uses_module_owned_leaf(self):
        capsule = object()
        received = []
        module = SimpleNamespace(
            __version__="2.15.0",
            experimental=SimpleNamespace(
                dlpack=SimpleNamespace(to_dlpack=lambda data: received.append(data) or capsule)
            ),
        )
        data = SimpleNamespace(device="GPU:0")

        payload = MemoryType.TENSORFLOW.export_dlpack(data, module)

        assert payload is not None
        assert payload.source is data
        assert payload.capsule is capsule
        assert received == [data]

    def test_tensorflow_dlpack_import_uses_exported_capsule(self):
        capsule = object()
        imported = object()
        received = []
        module = SimpleNamespace(
            experimental=SimpleNamespace(
                dlpack=SimpleNamespace(
                    from_dlpack=lambda value: received.append(value) or imported,
                )
            )
        )
        assert MemoryType.TENSORFLOW.from_dlpack(capsule, module) is imported
        assert received == [capsule]

    def test_modern_cupy_dlpack_import_uses_protocol_source(self):
        capsule = object()
        source = SimpleNamespace(__dlpack__=lambda: capsule, __dlpack_device__=lambda: (2, 0))
        imported = object()
        received = []
        module = SimpleNamespace(from_dlpack=lambda value: received.append(value) or imported)

        result = MemoryType.CUPY.from_dlpack(DLPackPayload(source, capsule), module)

        assert result is imported
        assert received == [source]

    def test_torch_dlpack_import_uses_capsule(self):
        capsule = object()
        source = SimpleNamespace(__dlpack__=lambda: capsule, __dlpack_device__=lambda: (2, 0))
        imported = object()
        received = []
        module = SimpleNamespace(from_dlpack=lambda value: received.append(value) or imported)

        result = MemoryType.TORCH.from_dlpack(DLPackPayload(source, capsule), module)

        assert result is imported
        assert received == [capsule]

    def test_jax_dlpack_import_declines_capsule_only_source(self):
        payload = DLPackPayload(source=object(), capsule=object())

        assert MemoryType.JAX.from_dlpack(payload, SimpleNamespace()) is NotImplemented

    @pytest.mark.parametrize("exporter_name", ["to_dlpack", "toDlpack"])
    def test_protocol_dlpack_export_accepts_legacy_exporters(self, exporter_name):
        capsule = object()
        data = SimpleNamespace(**{exporter_name: lambda: capsule})

        payload = MemoryType.CUPY.export_dlpack(data, object())

        assert payload is not None
        assert payload.source is data
        assert payload.capsule is capsule

    def test_oom_classification_uses_loaded_module_without_import(self, monkeypatch):
        from arraybridge import types

        class TorchOOMError(Exception):
            pass

        module = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=TorchOOMError))
        monkeypatch.setitem(types.sys.modules, "torch", module)

        def fail_import(name):
            raise AssertionError(f"unexpected import of {name}")

        monkeypatch.setattr(types.importlib, "import_module", fail_import)

        assert MemoryType.TORCH.is_oom_error(TorchOOMError("allocation failed"))
        assert not MemoryType.TORCH.is_oom_error(RuntimeError("unrelated"))

    def test_cpu_device_scope_does_not_import_numpy(self, monkeypatch):
        from arraybridge import types

        def fail_import(name):
            raise AssertionError(f"unexpected import of {name}")

        monkeypatch.setattr(types.importlib, "import_module", fail_import)
        with MemoryType.NUMPY.device_scope(0):
            pass

    def test_jax_cpu_install_does_not_declare_a_gpu(self):
        cpu_device = SimpleNamespace(platform="cpu")
        module = SimpleNamespace(devices=lambda: [cpu_device])

        assert MemoryType.JAX.available_device_ids(module) == ()
        with pytest.raises(ValueError, match="available device IDs are \\(\\)"):
            MemoryType.JAX.device_scope(0, module)

    @pytest.mark.parametrize("device_id", [-1, 2])
    def test_device_scope_rejects_undeclared_ids(self, device_id):
        devices = [SimpleNamespace(platform="gpu") for _ in range(2)]

        @contextmanager
        def default_device(device):
            yield device

        module = SimpleNamespace(devices=lambda: devices, default_device=default_device)

        with pytest.raises(ValueError, match=f"device {device_id} is unavailable"):
            MemoryType.JAX.device_scope(device_id, module)

    def test_jax_device_identity_uses_gpu_local_index(self):
        cpu = SimpleNamespace(platform="cpu")
        first_gpu = SimpleNamespace(platform="gpu")
        second_gpu = SimpleNamespace(platform="gpu")
        module = SimpleNamespace(devices=lambda: [cpu, first_gpu, second_gpu])
        data = SimpleNamespace(device=second_gpu)

        assert MemoryType.JAX.available_device_ids(module) == (0, 1)
        assert MemoryType.JAX.device_id_of(data, module) == 1

    def test_pyclesperanto_move_restores_recognized_origin(self):
        module = _FakePyclesperanto(current="gpu0")
        source = object()

        result = MemoryType.PYCLESPERANTO.move_to_device(source, 1, module)

        assert result == ("copy", source)
        assert module.current == "gpu0"
        assert module.copies == [(source, result)]

    def test_pyclesperanto_scope_restores_origin_outside_gpu_inventory(self):
        module = _FakePyclesperanto(current="cpu0")

        with MemoryType.PYCLESPERANTO.device_scope(1, module):
            assert module.current == "gpu1"

        assert module.current == "cpu0"

    def test_pyclesperanto_unknown_origin_is_not_guessed_as_gpu_zero(self):
        module = _FakePyclesperanto(current="cpu0")
        source = object()

        result = MemoryType.PYCLESPERANTO.move_to_device(source, 0, module)

        assert result == ("copy", source)
        assert module.current == "cpu0"
        assert module.copies == [(source, result)]

    def test_pyclesperanto_device_identity_uses_device_name(self):
        module = _FakePyclesperanto(current=SimpleNamespace(id=None, name="gpu1"))

        assert MemoryType.PYCLESPERANTO.device_id_of(object(), module) == 1

    def test_pyclesperanto_device_identity_prefers_payload_evidence(self):
        module = _FakePyclesperanto(current="gpu0")
        data = SimpleNamespace(device=SimpleNamespace(id=None, name="gpu1"))

        assert MemoryType.PYCLESPERANTO.device_id_of(data, module) == 1

    def test_pyclesperanto_non_gpu_numeric_id_is_not_accepted(self):
        module = _FakePyclesperanto(current=SimpleNamespace(id=0, name="cpu0"))

        assert MemoryType.PYCLESPERANTO.device_id_of(object(), module) is None

    def test_torch_from_numpy_materializes_negative_strides(self):
        received = []

        class FakeTensor:
            def to(self, device):
                assert device == "cuda:0"
                return self

        @contextmanager
        def device_scope(device_id):
            assert device_id == 0
            yield

        def from_numpy(data):
            received.append(data)
            assert all(stride >= 0 for stride in data.strides)
            return FakeTensor()

        module = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
                device=device_scope,
            ),
            from_numpy=from_numpy,
        )
        values = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::-1]

        MemoryType.TORCH.from_numpy(values, 0, module)

        np.testing.assert_array_equal(received[0], values)


class _FakePyclesperanto:
    def __init__(self, current: str) -> None:
        self.current = current
        self.gpu_devices = ("gpu0", "gpu1")
        self.copies = []

    def list_available_devices(self, device_type=None):
        assert device_type in (None, "gpu")
        return self.gpu_devices

    def get_device(self):
        return self.current

    def select_device(self, selector, device_type=None):
        if isinstance(selector, int):
            assert device_type in (None, "gpu")
            self.current = self.gpu_devices[selector]
        else:
            self.current = selector

    def create_like(self, data):
        return ("copy", data)

    def copy(self, source, target):
        self.copies.append((source, target))
