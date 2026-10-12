"""Tests for arraybridge.decorators module."""

import sys
import threading
from dataclasses import dataclass
from typing import Any

import dill
import numpy as np
import pytest
from python_introspect import RuntimeParameterDeclarationABC

import arraybridge
from arraybridge import ArrayPayload, decorators
from arraybridge.decorators import (
    DtypeConversion,
    DtypeConversionConfig,
    PreserveInputDtypeConfig,
    SliceBySliceRuntimeParameter,
    memory_types,
    wrap_dtype_preserving_callable,
)
from arraybridge.devices import AcceleratorDevice
from arraybridge.libraries import ArrayLibrary
from arraybridge.types import MemoryType
from tests.fakes import FakeCupy, FakeTorch


class TestDtypeConversion:
    """Tests for DtypeConversion enum."""

    def test_dtype_conversion_enum_values(self):
        """Test all DtypeConversion enum values exist."""
        assert DtypeConversion.PRESERVE_INPUT.value == "preserve"
        assert DtypeConversion.NATIVE_OUTPUT.value == "native"
        assert DtypeConversion.UINT8.value == "uint8"
        assert DtypeConversion.UINT16.value == "uint16"
        assert DtypeConversion.INT16.value == "int16"

    def test_runtime_parameters_are_nominal_declarations(self):
        assert issubclass(DtypeConversionConfig, RuntimeParameterDeclarationABC)
        assert issubclass(
            SliceBySliceRuntimeParameter,
            RuntimeParameterDeclarationABC,
        )
        assert DtypeConversion.INT32.value == "int32"
        assert DtypeConversion.FLOAT32.value == "float32"
        assert DtypeConversion.FLOAT64.value == "float64"

    def test_numpy_dtype_property(self):
        """Test numpy_dtype property returns correct dtypes."""
        assert DtypeConversion.UINT8.numpy_dtype == np.uint8
        assert DtypeConversion.UINT16.numpy_dtype == np.uint16
        assert DtypeConversion.INT16.numpy_dtype == np.int16
        assert DtypeConversion.INT32.numpy_dtype == np.int32
        assert DtypeConversion.FLOAT32.numpy_dtype == np.float32
        assert DtypeConversion.FLOAT64.numpy_dtype == np.float64
        assert DtypeConversion.PRESERVE_INPUT.numpy_dtype is None
        assert DtypeConversion.NATIVE_OUTPUT.numpy_dtype is None


class TestExecutionOnAccelerators:
    """What a library's devices need around a decorated call."""

    @pytest.mark.parametrize(
        ("name", "fake"), [("torch", FakeTorch), ("cupy", FakeCupy)], ids=["torch", "cupy"]
    )
    def test_cuda_call_runs_on_this_threads_stream(self, name, fake, monkeypatch):
        module = fake(device_count=1)
        monkeypatch.setitem(sys.modules, name, module)

        @getattr(decorators, name)
        def identity(value):
            module.events.append(("call", value))
            return value

        assert identity("payload") == "payload"
        assert identity("again") == "again"
        stream = ArrayLibrary.named(name).thread_stream(AcceleratorDevice(0))
        assert module.events == [
            ("enter", 0),
            ("exit", 0),
            ("stream enter", stream),
            ("call", "payload"),
            ("stream exit", stream),
            ("stream enter", stream),
            ("call", "again"),
            ("stream exit", stream),
        ]

    def test_torch_without_cuda_calls_the_function_directly(self, monkeypatch):
        module = FakeTorch()
        monkeypatch.setitem(sys.modules, "torch", module)

        @decorators.torch
        def identity(value):
            return value

        assert identity("payload") == "payload"
        assert module.events == []

    @pytest.mark.parametrize("name", [library.name for library in ArrayLibrary.all()])
    def test_library_that_is_not_installed_calls_the_function_directly(
        self, name, absent_libraries
    ):
        @getattr(decorators, name)
        def add_one(values):
            return values + 1

        np.testing.assert_array_equal(add_one(np.array([1, 2])), [2, 3])
        assert add_one.execution_memory_type == name

    def test_out_of_memory_releases_the_cache_and_retries(self, monkeypatch):
        module = FakeCupy(device_count=1)
        monkeypatch.setitem(sys.modules, "cupy", module)
        calls = []

        def fails_once(value):
            calls.append(value)
            if len(calls) == 1:
                raise MemoryError("out of memory")
            return value

        assert decorators.cupy(fails_once)("payload") == "payload"
        assert calls == ["payload", "payload"]
        assert [event for event in module.events if event[0] == "free"] == [
            ("free", "regular"),
            ("free", "pinned"),
        ]

        calls.clear()
        with pytest.raises(MemoryError, match="out of memory"):
            decorators.cupy(fails_once, oom_recovery=False)("payload")
        assert calls == ["payload"]

    def test_dtype_is_restored_by_name_across_libraries(self, monkeypatch):
        class TorchDtype:
            def __str__(self):
                return "torch.uint16"

        class Array:
            def __init__(self, dtype):
                self.dtype = dtype

        converted = []

        def scale_dtype(library, array, target_dtype):
            converted.append((library.name, target_dtype))
            return Array(np.dtype(target_dtype))

        monkeypatch.setattr(ArrayLibrary, "scale_dtype", scale_dtype)

        @memory_types("torch", "jax")
        def convert(_image):
            return Array(np.dtype("float32"))

        wrapped = wrap_dtype_preserving_callable(convert, MemoryType.JAX)
        result = wrapped(Array(TorchDtype()))

        assert converted == [("jax", "uint16")]
        assert result.dtype == np.dtype("uint16")

    @pytest.mark.parametrize("name", ["numpy", "cupy"])
    def test_decorated_function_pickles_without_this_threads_streams(self, name, monkeypatch):
        # A function with no importable name, as kept by an undo snapshot: dill
        # stores its body, and the stream of the thread that made it stays behind.
        module = FakeCupy(device_count=1)
        monkeypatch.setitem(sys.modules, "cupy", module)
        stream = ArrayLibrary.named("cupy").thread_stream(AcceleratorDevice(0))
        stream.unpicklable = threading.Lock()

        @getattr(decorators, name)
        def declared(image, scale=3):
            return image * scale

        restored = dill.loads(dill.dumps(declared))

        assert restored is not declared
        assert restored.__name__ == declared.__name__
        assert restored.__wrapped__.__name__ == declared.__wrapped__.__name__
        assert restored.execution_memory_type == name
        np.testing.assert_array_equal(
            restored(np.array([1, 2], dtype=np.uint16), scale=4), [4, 8]
        )
        assert ArrayLibrary.named("cupy").thread_stream(AcceleratorDevice(0)) is stream


class TestMemoryTypesDecorator:
    """Tests for memory_types decorator."""

    def test_memory_types_basic_decoration(self):
        """Test basic memory_types decorator functionality."""

        @memory_types("numpy", "numpy")
        def test_func(x):
            return x * 2

        # Check metadata is attached
        assert hasattr(test_func, "input_memory_type")
        assert hasattr(test_func, "output_memory_type")
        assert test_func.input_memory_type == "numpy"
        assert test_func.output_memory_type == "numpy"

        # Test function still works
        result = test_func(5)
        assert result == 10

    def test_memory_types_with_contract(self):
        """Test memory_types decorator with contract validation."""

        def positive_contract(x):
            return x > 0

        @memory_types("numpy", "numpy", contract=positive_contract)
        def test_func(x):
            return x * 2

        # Valid result
        result = test_func(5)
        assert result == 10

        # Invalid result should raise ValueError
        with pytest.raises(ValueError, match="violated its output contract"):
            test_func(-1)

    def test_memory_types_normalizes_owner_enum_members(self):
        @memory_types(MemoryType.NUMPY, MemoryType.TORCH)
        def test_func(x):
            return x

        assert test_func.input_memory_type == "numpy"
        assert test_func.output_memory_type == "torch"

    def test_framework_helper_declares_execution_owner_independently(self):
        from arraybridge.decorators import torch

        @torch(input_type=MemoryType.CUPY, output_type=MemoryType.NUMPY)
        def test_func(value):
            return value

        assert test_func.input_memory_type == "cupy"
        assert test_func.output_memory_type == "numpy"
        assert test_func.execution_memory_type == "torch"

    def test_dtype_preservation_uses_declared_output_framework(self):
        torch = pytest.importorskip("torch")
        from arraybridge.decorators import numpy

        @numpy(input_type="numpy", output_type="torch")
        def to_torch(value):
            return torch.as_tensor(value, dtype=torch.float32)

        source = np.arange(12, dtype=np.uint16).reshape(3, 4)
        result = to_torch(source)

        assert isinstance(result, torch.Tensor)
        assert result.dtype is torch.uint16
        assert result.shape == source.shape
        assert result[0, 0].item() == 0
        assert result[-1, -1].item() == np.iinfo(np.uint16).max

    def test_slice_stacking_uses_declared_output_framework(self):
        torch = pytest.importorskip("torch")
        from arraybridge.decorators import numpy

        @numpy(output_type="torch", slice_by_slice_default=True)
        def planes_to_torch(value):
            return torch.as_tensor(value)

        source = np.arange(24, dtype=np.uint16).reshape(3, 2, 4)
        result = planes_to_torch(source)

        assert isinstance(result, torch.Tensor)
        assert result.dtype is torch.uint16
        assert tuple(result.shape) == source.shape

    def test_every_registered_library_has_a_decorator(self):
        names = {library.name for library in ArrayLibrary.all()}

        assert names <= set(dir(decorators))
        assert names <= set(arraybridge.__all__)
        for name in names:
            decorator = getattr(decorators, name)
            assert decorator is getattr(arraybridge, name)
            assert decorator.__name__ == name
            assert ArrayLibrary.named(name).display_name in decorator.__doc__
        with pytest.raises(AttributeError):
            decorators.not_a_library

    @pytest.mark.torch
    def test_torch_function_runs_in_main_memory_without_cuda(self):
        torch = pytest.importorskip("torch")

        @arraybridge.torch
        def double(tensor):
            return tensor * 2

        source = torch.arange(6, dtype=torch.int32).reshape(2, 3)
        result = double(source)

        assert isinstance(result, torch.Tensor) and not result.is_cuda
        assert result.dtype is torch.int32
        assert double.execution_memory_type == "torch"
        np.testing.assert_array_equal(result.numpy(), np.arange(6).reshape(2, 3) * 2)

    def test_memory_types_preserves_function_metadata(self):
        """Test that memory_types preserves function name, docstring, etc."""

        @memory_types("numpy", "numpy")
        def test_func(x, y=10):
            """Test function docstring."""
            return x + y

        assert test_func.__name__ == "test_func"
        assert test_func.__doc__ == "Test function docstring."
        assert test_func(5) == 15
        assert test_func(5, y=20) == 25


class TestFrameworkDecorators:
    """Tests for auto-generated framework-specific decorators."""

    def test_numpy_decorator_basic(self):
        """Test basic numpy decorator functionality."""
        from arraybridge.decorators import numpy

        @numpy
        def add_one(arr):
            return arr + 1

        # Check metadata
        assert add_one.input_memory_type == "numpy"
        assert add_one.output_memory_type == "numpy"

        # Test with numpy array
        arr = np.array([1, 2, 3])
        result = add_one(arr)
        np.testing.assert_array_equal(result, [2, 3, 4])

    def test_numpy_decorator_dtype_preservation(self):
        """Test numpy decorator preserves input dtype."""
        from arraybridge.decorators import numpy

        @numpy
        def to_float(arr):
            return arr.astype(np.float32)

        # Test with uint8 input
        arr = np.array([0, 127, 255], dtype=np.uint8)
        result = to_float(arr)

        # Should preserve uint8 dtype
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, [0, 127, 255])

    def test_numpy_decorator_dtype_conversion(self):
        """Test numpy decorator with explicit dtype conversion."""
        from arraybridge.decorators import numpy

        @numpy
        def identity(arr):
            return arr

        arr = np.array([0.5, 1.0], dtype=np.float64)
        result = identity(
            arr,
            dtype_config=PreserveInputDtypeConfig(DtypeConversion.UINT8),
        )

        # Should convert to uint8
        assert result.dtype == np.uint8
        assert result.shape == arr.shape

    def test_numpy_decorator_tuple_dtype_conversion(self):
        """Test tuple output dtype conversion applies to main output only."""
        from arraybridge.decorators import (
            numpy,
        )

        @numpy
        def to_float_with_meta(arr):
            return arr.astype(np.float32), {"meta": "ok"}

        arr = np.array([0, 127, 255], dtype=np.uint8)
        result = to_float_with_meta(arr)

        assert isinstance(result, tuple)
        assert result[0].dtype == np.uint8
        assert result[0].shape == arr.shape
        assert result[1]["meta"] == "ok"

    def test_numpy_decorator_preserves_nominal_array_payload(self):
        from arraybridge.decorators import numpy

        @dataclass(frozen=True)
        class ContextPayload(ArrayPayload):
            data: Any
            context: str

            def array_payload_data(self):
                return self.data

            def with_data(self, data):
                return type(self)(data=data, context=self.context)

        @numpy
        def label_image(_image):
            return ContextPayload(
                data=np.array([0, 1, 2], dtype=np.int32),
                context="source-plane-2",
            )

        result = label_image(np.zeros(3, dtype=np.float32))

        assert isinstance(result, ContextPayload)
        assert result.context == "source-plane-2"
        assert result.data.dtype == np.float32
        np.testing.assert_array_equal(result.data, [0, 1, 2])

        unchanged_payload = ContextPayload(
            data=np.zeros(3, dtype=np.float32),
            context="source-plane-3",
        )

        @numpy
        def unchanged(_image):
            return unchanged_payload

        assert unchanged(np.zeros(3, dtype=np.float32)) is unchanged_payload


class TestDecoratorParameters:
    """Tests for decorator parameter handling."""

    def test_decorator_with_custom_memory_types(self):
        """Test decorator with custom input/output memory types."""
        from arraybridge.decorators import numpy

        @numpy(input_type="torch", output_type="cupy")
        def test_func(x):
            return x

        assert test_func.input_memory_type == "torch"
        assert test_func.output_memory_type == "cupy"

    def test_decorator_with_oom_recovery_disabled(self):
        """Test decorator with OOM recovery disabled."""
        from arraybridge.decorators import numpy

        @numpy(oom_recovery=False)
        def test_func(x):
            return x

        # Function should still work normally
        assert test_func(5) == 5

    def test_slice_by_slice_parameter(self):
        """Test slice_by_slice parameter in function signature."""
        from arraybridge.decorators import numpy

        @numpy
        def process_3d(arr):
            return arr

        # Check that slice_by_slice parameter was added to signature
        import inspect

        sig = inspect.signature(process_3d)
        assert SliceBySliceRuntimeParameter.require_parameter_name() in sig.parameters
        assert DtypeConversionConfig.require_parameter_name() in sig.parameters

        # Test with slice_by_slice=False (default)
        arr_3d = np.random.rand(3, 10, 10)
        result = process_3d(
            arr_3d,
            **{SliceBySliceRuntimeParameter.require_parameter_name(): False},
        )
        assert result.shape == arr_3d.shape

    def test_slice_by_slice_default_is_owned_by_decorator_declaration(self):
        """A callable can declare a non-default slice execution policy once."""
        import inspect

        from arraybridge.decorators import numpy

        observed_shapes = []

        @numpy(slice_by_slice_default=True)
        def process_plane(arr):
            """Process one plane."""
            observed_shapes.append(arr.shape)
            return arr

        signature = inspect.signature(process_plane)
        slice_parameter = signature.parameters[
            SliceBySliceRuntimeParameter.require_parameter_name()
        ]
        assert slice_parameter.default is True

        arr_3d = np.random.rand(3, 10, 10)
        result = process_plane(arr_3d)

        assert result.shape == arr_3d.shape
        assert observed_shapes == [(10, 10)] * 3
        assert "Defaults to True" in (process_plane.__doc__ or "")
