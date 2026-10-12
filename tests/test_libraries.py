"""The ``ArrayLibrary`` family: lookup, identity, import environment, memory and operations."""

import importlib
import importlib.util
import os
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from metaclass_registry.exceptions import DuplicateRegistryKeyError

from arraybridge.devices import HOST, AcceleratorDevice
from arraybridge.libraries import (
    AcceleratorDevices,
    ArrayLibrary,
    CudaLibrary,
    HostDevices,
    NumpyLibrary,
)
from tests.fakes import FakeCupy, FakePyclesperanto, FakePyclesperantoArray, FakeTorch, fake_jax

LIBRARIES = ArrayLibrary.all()
NUMPY = ArrayLibrary.named("numpy")


def names(role):
    return {library.name for library in LIBRARIES if isinstance(library, role)}


class TestLookup:
    def test_one_object_per_registered_class(self):
        assert [library.name for library in LIBRARIES] == list(ArrayLibrary.__registry__)
        for library in LIBRARIES:
            assert type(library) is ArrayLibrary.__registry__[library.name]
            assert ArrayLibrary.named(library.name) is library
            assert ArrayLibrary.for_module_name(f"{library.import_name}.submodule") is library
            assert pickle.loads(pickle.dumps(library)) is library
            assert str(library) == library.name

    def test_roles_are_inherited(self):
        assert names(HostDevices) == {"numpy", "torch", "tensorflow", "jax"}
        assert names(AcceleratorDevices) == {
            "cupy",
            "torch",
            "tensorflow",
            "jax",
            "pyclesperanto",
        }
        assert names(CudaLibrary) == {"cupy", "torch"}

    def test_identity_is_stated_by_each_class(self):
        assert NUMPY.import_name == "numpy"
        assert NUMPY.display_name == "NumPy"
        assert ArrayLibrary.named("pyclesperanto").display_name == "pyclesperanto"
        assert ArrayLibrary.named("jax").recognized_module_names == frozenset({"jax", "jaxlib"})
        assert ArrayLibrary.for_module_name("jaxlib") is ArrayLibrary.named("jax")

    def test_unknown_names_are_rejected(self):
        with pytest.raises(ValueError, match="not a registered array library"):
            ArrayLibrary.named("unknown_type")
        with pytest.raises(ValueError, match="No array library owns module"):
            ArrayLibrary.for_module_name("scipy.ndimage")

    def test_of_names_the_library_that_made_a_value(self):
        assert ArrayLibrary.of(np.zeros((2, 2))) is NUMPY
        for value in ([1, 2, 3], None, "not an array"):
            with pytest.raises(ValueError, match="Unknown memory type"):
                ArrayLibrary.of(value)

    def test_a_name_is_registered_to_one_class(self):
        with pytest.raises(DuplicateRegistryKeyError, match="key 'numpy' is already registered"):

            class SecondNumpy(NumpyLibrary):
                name = "numpy"

        assert ArrayLibrary.__registry__["numpy"] is NumpyLibrary
        assert ArrayLibrary.named("numpy") is NUMPY


class TestImporting:
    @pytest.mark.parametrize("library", LIBRARIES[1:], ids=str)
    def test_absent_library_is_never_imported(self, library, absent_libraries, monkeypatch):
        monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
        monkeypatch.delenv("TF_FORCE_GPU_ALLOW_GROWTH", raising=False)
        values = np.array([1, 2, 3])

        assert library.import_if_installed() is None
        assert library.accelerator_devices() == ()
        assert library.scale_dtype(values, np.float32) is values
        library.cleanup_loaded()
        assert not library.is_oom_error(RuntimeError("unrelated"))
        assert not {"XLA_PYTHON_CLIENT_PREALLOCATE", "TF_FORCE_GPU_ALLOW_GROWTH"} & set(os.environ)

    def test_cleanup_of_every_library_imports_nothing(self, absent_libraries):
        ArrayLibrary.cleanup_all_loaded()
        ArrayLibrary.cleanup_all_loaded(AcceleratorDevice(0))

    def test_import_preparation_sets_only_missing_defaults(self, monkeypatch):
        tensorflow, jax = ArrayLibrary.named("tensorflow"), ArrayLibrary.named("jax")
        monkeypatch.delenv("TF_FORCE_GPU_ALLOW_GROWTH", raising=False)
        monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "custom")

        tensorflow.prepare_import()
        jax.prepare_import()

        assert tensorflow.import_environment == (("TF_FORCE_GPU_ALLOW_GROWTH", "true"),)
        assert jax.import_environment == (("XLA_PYTHON_CLIENT_PREALLOCATE", "false"),)
        assert os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"
        assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "custom"

    def test_library_loaded_before_its_defaults_is_reported(self, monkeypatch, caplog):
        module = fake_jax("cpu")
        monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
        monkeypatch.setitem(sys.modules, "jax", module)
        jax = ArrayLibrary.named("jax")

        assert jax.import_if_installed() is module
        assert jax.module is module
        assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
        assert "loaded before its ArrayBridge import defaults" in caplog.text

    def test_child_environment_adds_nvidia_wheel_directories(self, monkeypatch, tmp_path):
        nvidia_root = tmp_path / "nvidia"
        cublas_lib = nvidia_root / "cublas" / "lib"
        runtime_bin = nvidia_root / "cuda_runtime" / "bin"
        cublas_lib.mkdir(parents=True)
        runtime_bin.mkdir(parents=True)
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: (
                SimpleNamespace(submodule_search_locations=(str(nvidia_root),))
                if name == "nvidia"
                else None
            ),
        )
        search_variable = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        original = {search_variable: "/existing", "CUSTOM": "value"}

        prepared = ArrayLibrary.child_environment(original)

        assert original == {search_variable: "/existing", "CUSTOM": "value"}
        assert prepared[search_variable].split(os.pathsep) == [
            *sorted((str(cublas_lib), str(runtime_bin))),
            "/existing",
        ]
        assert prepared["CUSTOM"] == "value"
        assert prepared["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"
        assert prepared["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"

    def test_child_environment_keeps_the_caller_values(self, monkeypatch):
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        original = {
            "TF_FORCE_GPU_ALLOW_GROWTH": "host",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "host",
        }

        prepared = ArrayLibrary.child_environment(original)

        assert prepared == original
        assert prepared is not original


class TestOutOfMemory:
    @pytest.mark.parametrize(
        ("name", "messages"),
        [
            ("numpy", ("cannot allocate memory", "Memory exhausted")),
            ("cupy", ("Out of memory", "CUDA_ERROR_OUT_OF_MEMORY")),
            ("torch", ("CUDA out of memory", "cuda_error_out_of_memory")),
            ("tensorflow", ("out of memory", "RESOURCE_EXHAUSTED: failed to allocate")),
            ("jax", ("out of memory", "OOM when allocating", "allocation failure")),
            ("pyclesperanto", ("CL_MEM_OBJECT_ALLOCATION_FAILURE", "CL_OUT_OF_RESOURCES")),
        ],
    )
    def test_messages_are_recognised_per_library(self, name, messages, absent_libraries):
        library = ArrayLibrary.named(name)
        for message in messages:
            assert library.is_oom_error(RuntimeError(message)), message
        assert not library.is_oom_error(RuntimeError("some random error"))

    @pytest.mark.parametrize(
        ("name", "path"),
        [
            ("torch", ("cuda", "OutOfMemoryError")),
            ("cupy", ("cuda", "memory", "OutOfMemoryError")),
            ("cupy", ("cuda", "runtime", "CUDARuntimeError")),
            ("tensorflow", ("errors", "ResourceExhaustedError")),
        ],
    )
    def test_loaded_module_exception_classes_are_recognised(self, name, path, monkeypatch):
        class LibraryOutOfMemoryError(Exception):
            pass

        module = LibraryOutOfMemoryError
        for attribute in reversed(path):
            module = SimpleNamespace(**{attribute: module})
        monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(
            importlib, "import_module", lambda name: pytest.fail(f"imported {name}")
        )
        library = ArrayLibrary.named(name)

        assert library.is_oom_error(LibraryOutOfMemoryError("allocation failed"))
        assert not library.is_oom_error(RuntimeError("allocation failed"))

    @pytest.mark.parametrize("library", LIBRARIES, ids=str)
    def test_retries_after_an_out_of_memory_error(self, library, absent_libraries):
        calls = []

        def fails_once():
            calls.append(len(calls))
            if len(calls) == 1:
                raise MemoryError(library.oom_messages[0])
            return "success"

        assert library.run_with_oom_recovery(fails_once, max_retries=1) == "success"
        assert calls == [0, 1]

    @pytest.mark.parametrize(("max_retries", "expected_calls"), [(0, 1), (2, 3)])
    def test_gives_up_after_the_allowed_retries(self, max_retries, expected_calls):
        calls = []

        def always_fails():
            calls.append(None)
            raise MemoryError("out of memory")

        with pytest.raises(MemoryError, match="out of memory"):
            ArrayLibrary.named("torch").run_with_oom_recovery(
                always_fails, max_retries=max_retries
            )
        assert len(calls) == expected_calls

    def test_other_errors_are_not_retried(self):
        calls = []

        def raises_value_error():
            calls.append(None)
            raise ValueError("not an OOM error")

        with pytest.raises(ValueError, match="not an OOM error"):
            ArrayLibrary.named("torch").run_with_oom_recovery(raises_value_error)
        assert len(calls) == 1

    def test_retry_releases_the_cache_of_the_given_device(self, monkeypatch):
        module = FakeCupy()
        monkeypatch.setitem(sys.modules, "cupy", module)
        outcomes = [MemoryError("out of memory"), "success"]

        def fails_once():
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        result = ArrayLibrary.named("cupy").run_with_oom_recovery(
            fails_once, device=AcceleratorDevice(1)
        )

        assert result == "success"
        assert module.events == [
            ("enter", 1),
            ("free", "regular"),
            ("free", "pinned"),
            ("synchronize", None),
            ("exit", 1),
        ]


class TestCleanup:
    def test_cupy_pools_are_freed_on_every_device(self, monkeypatch):
        module = FakeCupy()
        monkeypatch.setitem(sys.modules, "cupy", module)

        ArrayLibrary.named("cupy").cleanup_loaded()

        freed = [("free", "regular"), ("free", "pinned"), ("synchronize", None)]
        assert module.events == [
            ("enter", 0),
            *freed,
            ("exit", 0),
            ("enter", 1),
            *freed,
            ("exit", 1),
        ]

    def test_torch_cache_is_emptied_on_the_given_device(self, monkeypatch):
        module = FakeTorch(device_count=2)
        monkeypatch.setitem(sys.modules, "torch", module)

        ArrayLibrary.cleanup_all_loaded(AcceleratorDevice(1))

        assert module.events == [
            ("enter", 1),
            ("empty_cache", 1),
            ("synchronize", 1),
            ("exit", 1),
        ]

    def test_torch_without_cuda_has_nothing_to_release(self, monkeypatch):
        module = FakeTorch()
        monkeypatch.setitem(sys.modules, "torch", module)

        ArrayLibrary.named("torch").cleanup_loaded()

        assert module.events == []

    def test_unavailable_device_is_rejected_by_the_library(self, monkeypatch):
        module = FakeCupy(device_count=1)
        monkeypatch.setitem(sys.modules, "cupy", module)

        with pytest.raises(ValueError, match="device accelerator:3 is unavailable"):
            ArrayLibrary.named("cupy").cleanup_loaded(AcceleratorDevice(3))

        assert module.events == []

    def test_cleanup_of_every_library_logs_a_failing_one(self, monkeypatch, caplog):
        module = FakeCupy(device_count=1)
        monkeypatch.setitem(sys.modules, "cupy", module)

        ArrayLibrary.cleanup_all_loaded(AcceleratorDevice(3))

        assert module.events == []
        assert "Failed to cleanup CuPy GPU memory" in caplog.text
        assert "device accelerator:3 is unavailable" in caplog.text

    def test_jax_compilation_cache_is_not_cleared_as_device_memory(self, monkeypatch):
        events = []
        module = fake_jax("gpu")
        module.clear_caches = lambda: events.append("clear")
        monkeypatch.setitem(sys.modules, "jax", module)

        ArrayLibrary.named("jax").cleanup_loaded(AcceleratorDevice(0))

        assert events == []


class TestScaleDtype:
    @pytest.mark.parametrize(
        ("values", "dtype", "expected"),
        [
            (np.array([1, 2, 3], dtype=np.uint8), np.uint16, [1, 2, 3]),
            (np.array([0.0, 0.5, 1.0], dtype=np.float64), np.uint8, [0, 127, 255]),
            (np.array([0.0, 0.5, 1.0], dtype=np.float32), np.uint16, [0, 32767, 65535]),
            (np.array([0.0, 1.0], dtype=np.float16), np.uint8, [0, 255]),
            (np.array([0.0, 1.0], dtype=np.float64), np.int16, [-32768, 32639]),
            (np.array([0.0, 1.0], dtype=np.float64), np.uint32, [0, 4294967295]),
            (np.array([0.25, 0.75], dtype=np.float64), np.float32, [0.25, 0.75]),
        ],
    )
    def test_numpy_values(self, values, dtype, expected):
        result = NUMPY.scale_dtype(values, dtype)

        assert result.dtype == dtype
        np.testing.assert_array_equal(result, expected)

    def test_numpy_int32_stays_inside_the_range(self):
        result = NUMPY.scale_dtype(np.array([0.0, 0.5, 1.0], dtype=np.float64), np.int32)

        assert result.dtype == np.int32
        assert result[0] == np.iinfo(np.int32).min
        assert abs(int(result[1])) <= 150
        assert np.iinfo(np.int32).max - 150 <= result[2] <= np.iinfo(np.int32).max

    @pytest.mark.parametrize(
        "values",
        [
            np.full((10, 10), 0.5, dtype=np.float32),
            np.array([0.7], dtype=np.float32),
        ],
    )
    def test_numpy_constant_image_is_cast_without_rescaling(self, values):
        result = NUMPY.scale_dtype(values, np.uint8)

        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, np.zeros(values.shape, dtype=np.uint8))

    def test_value_without_a_dtype_is_returned_unchanged(self):
        assert NUMPY.scale_dtype("not an array", np.uint8) == "not an array"

    def test_numpy_empty_array_has_no_range_to_scale(self):
        with pytest.raises(ValueError):
            NUMPY.scale_dtype(np.array([], dtype=np.float32), np.uint8)

    @pytest.mark.parametrize("name", ["torch", "cupy", "jax", "tensorflow"])
    def test_installed_library_scales_into_the_integer_range(self, name):
        module = pytest.importorskip(name)
        library = ArrayLibrary.named(name)
        host_values = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        values = library.from_numpy(host_values, _first_device(library))

        as_uint16 = library.scale_dtype(values, np.uint16)
        as_int32 = library.to_numpy(library.scale_dtype(values, np.int32))

        assert library.accepts(as_uint16)
        assert library.dtype_name(as_uint16.dtype) == "uint16"
        np.testing.assert_array_equal(library.to_numpy(as_uint16), [0, 32767, 65535])
        assert as_int32.dtype == np.int32
        assert as_int32[0] <= np.iinfo(np.int32).min + 150
        assert abs(int(as_int32[1])) <= 150
        assert np.iinfo(np.int32).max - 150 <= as_int32[2]
        assert module is library.module

    def test_jax_float64_fails_when_x64_is_disabled(self, monkeypatch):
        module = fake_jax("cpu")
        module.config = SimpleNamespace(x64_enabled=False)
        module.numpy = object()
        monkeypatch.setitem(sys.modules, "jax", module)
        values = SimpleNamespace(dtype=np.dtype(np.float32), device=None)

        with pytest.raises(ValueError, match="JAX_ENABLE_X64"):
            ArrayLibrary.named("jax").scale_dtype(values, np.float64)

    def test_pyclesperanto_scales_on_the_device_of_its_input(self, monkeypatch):
        module = FakePyclesperanto(current="gpu0")
        monkeypatch.setitem(sys.modules, "pyclesperanto", module)
        values = FakePyclesperantoArray([1, 2, 3], "gpu1")

        result = ArrayLibrary.named("pyclesperanto").scale_dtype(values, np.int32)

        assert result.device.name == "gpu1"
        assert result.dtype == np.int32
        np.testing.assert_array_equal(result.values, [1, 2, 3])
        assert module.current == "gpu0"

    @pytest.mark.pyclesperanto
    @pytest.mark.parametrize(
        "values",
        [
            np.array([[0.0, 0.5], [0.25, 1.0]], dtype=np.float32),
            np.full((10, 10), 0.5, dtype=np.float32),
            np.array([[1, 2, 3]], dtype=np.int32),
        ],
    )
    def test_installed_pyclesperanto_scales_to_int32(self, values):
        pytest.importorskip("pyclesperanto")
        library = ArrayLibrary.named("pyclesperanto")
        if not library.accelerator_devices():
            pytest.skip("No OpenCL GPU device")

        result = library.scale_dtype(library.from_numpy(values, AcceleratorDevice(0)), np.int32)

        assert str(result.dtype) == "int32"
        if values.dtype == np.int32:
            np.testing.assert_array_equal(library.to_numpy(result), values)


def _first_device(library):
    """Main memory where the library has it, otherwise its first accelerator."""

    devices = library.available_devices()
    if not devices:
        pytest.skip(f"{library.display_name} has no device on this machine")
    return devices[0]


class TestOperations:
    def test_numpy_operations(self):
        stack = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)

        assert NUMPY.module is np
        assert NUMPY.accepts(stack) and not NUMPY.accepts([1, 2])
        np.testing.assert_array_equal(NUMPY.max(stack, 0), stack[1])
        np.testing.assert_array_equal(NUMPY.min(stack, 0), stack[0])
        np.testing.assert_array_equal(NUMPY.mean(stack, 0), stack.mean(axis=0))
        np.testing.assert_array_equal(NUMPY.sum(stack, (1, 2)), stack.sum(axis=(1, 2)))
        assert NUMPY.prepend_axis(stack[0]).shape == (1, 3, 4)
        assert NUMPY.astype(stack, np.float32).dtype == np.float32
        ramp = NUMPY.linspace(0, 1, 4, endpoint=False)
        np.testing.assert_array_equal(ramp, [0, 0.25, 0.5, 0.75])
        weights = NUMPY.assign(NUMPY.ones(5, np.float32), slice(-2, None), ramp[:2])
        np.testing.assert_array_equal(weights, [1, 1, 1, 0, 0.25])
        np.testing.assert_array_equal(NUMPY.outer(ramp, ramp), np.outer(ramp, ramp))
        assert NUMPY.floor(2.7) == 2
        assert NUMPY.nbytes_of(stack) == stack.nbytes

    @pytest.mark.jax
    def test_jax_assignment_returns_a_new_array(self):
        jnp = pytest.importorskip("jax.numpy")
        jax = ArrayLibrary.named("jax")
        values = jnp.ones(4, dtype=jnp.float32)

        updated = jax.assign(values, slice(None, 2), jnp.zeros(2, dtype=jnp.float32))

        np.testing.assert_array_equal(np.asarray(updated), [0, 0, 1, 1])
        np.testing.assert_array_equal(np.asarray(values), [1, 1, 1, 1])
        stack = jnp.arange(24, dtype=jnp.uint16).reshape(2, 3, 4)
        assert jax.mean(stack, 0).dtype == jnp.float32

    def test_dtype_names_are_spelled_as_numpy_spells_them(self):
        class TensorflowDtype:
            as_numpy_dtype = np.uint16

        class TorchDtype:
            def __str__(self):
                return "torch.uint16"

        assert NUMPY.dtype_name(np.uint16) == "uint16"
        assert ArrayLibrary.named("tensorflow").dtype_name(TensorflowDtype()) == "uint16"
        assert ArrayLibrary.named("torch").dtype_name(TorchDtype()) == "uint16"

    def test_torch_wraps_a_reversed_view_as_a_contiguous_copy(self, monkeypatch):
        module = FakeTorch(device_count=1)
        monkeypatch.setitem(sys.modules, "torch", module)
        values = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::-1]
        torch = ArrayLibrary.named("torch")

        on_host = torch.from_numpy(values, HOST)
        on_accelerator = torch.from_numpy(values, AcceleratorDevice(0))

        wrapped = [data for event, data in module.events if event == "from_numpy"]
        assert len(wrapped) == 2
        for host_array in wrapped:
            assert all(stride >= 0 for stride in host_array.strides)
            np.testing.assert_array_equal(host_array, values)
        assert torch.device_of(on_host) == HOST
        assert torch.device_of(on_accelerator) == AcceleratorDevice(0)
