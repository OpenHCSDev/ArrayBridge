"""Tests for arraybridge.decorators module."""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from arraybridge import ArrayPayload
from arraybridge.decorators import (
    DtypeConversion,
    DtypeConversionConfig,
    PreserveInputDtypeConfig,
    SliceBySliceRuntimeParameter,
    ThreadGPUContext,
    memory_types,
    wrap_dtype_preserving_callable,
)
from arraybridge.decorators import (
    torch as torch_memory,
)
from arraybridge.types import MemoryType


class TestDtypeConversion:
    """Tests for DtypeConversion enum."""

    def test_dtype_conversion_enum_values(self):
        """Test all DtypeConversion enum values exist."""
        assert DtypeConversion.PRESERVE_INPUT.value == "preserve"
        assert DtypeConversion.NATIVE_OUTPUT.value == "native"
        assert DtypeConversion.UINT8.value == "uint8"
        assert DtypeConversion.UINT16.value == "uint16"
        assert DtypeConversion.INT16.value == "int16"
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


class TestThreadGPUContext:
    def test_streams_are_cached_by_framework_local_device(self):
        state = {"device": 0}
        created = []

        class DeviceScope:
            def __init__(self, device_id):
                self.device_id = device_id
                self.previous = None

            def __enter__(self):
                self.previous = state["device"]
                state["device"] = self.device_id

            def __exit__(self, exc_type, exc_value, traceback):
                state["device"] = self.previous
                return False

        class Stream:
            def __init__(self):
                self.device_id = state["device"]
                created.append(self)

        module = type(
            "FakeCupy",
            (),
            {
                "cuda": type(
                    "Cuda",
                    (),
                    {
                        "runtime": type(
                            "Runtime",
                            (),
                            {
                                "getDeviceCount": staticmethod(lambda: 2),
                                "getDevice": staticmethod(lambda: state["device"]),
                            },
                        )(),
                        "Device": staticmethod(DeviceScope),
                        "Stream": Stream,
                    },
                )()
            },
        )()
        context = ThreadGPUContext()

        _, first = context.stream_for(MemoryType.CUPY, module)
        _, repeated = context.stream_for(MemoryType.CUPY, module)
        state["device"] = 1
        _, second = context.stream_for(MemoryType.CUPY, module)
        state["device"] = 0
        _, first_again = context.stream_for(MemoryType.CUPY, module)

        assert first is repeated is first_again
        assert second is not first
        assert [stream.device_id for stream in created] == [0, 1]

    def test_framework_without_stream_leaf_does_not_invent_one(self):
        module = type(
            "FakeTensorFlow",
            (),
            {"config": type("Config", (), {"list_logical_devices": lambda self, _: []})()},
        )()

        assert ThreadGPUContext().stream_for(MemoryType.TENSORFLOW, module) == (
            None,
            None,
        )

    def test_torch_stream_execution_uses_framework_owned_scope(self, monkeypatch):
        events = []

        class Stream:
            pass

        class StreamScope:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                events.append(("enter", self.stream))

            def __exit__(self, exc_type, exc_value, traceback):
                events.append(("exit", self.stream))
                return False

        stream = Stream()
        module = type(
            "FakeTorch",
            (),
            {
                "cuda": type(
                    "Cuda",
                    (),
                    {
                        "is_available": staticmethod(lambda: True),
                        "device_count": staticmethod(lambda: 1),
                        "current_device": staticmethod(lambda: 0),
                        "device": staticmethod(lambda _device_id: nullcontext()),
                        "Stream": staticmethod(lambda: stream),
                        "stream": staticmethod(StreamScope),
                    },
                )()
            },
        )()
        monkeypatch.setattr(MemoryType, "import_if_installed", lambda self: module)

        @torch_memory
        def identity(value):
            events.append(("call", value))
            return value

        assert identity("payload") == "payload"
        assert events == [
            ("enter", stream),
            ("call", "payload"),
            ("exit", stream),
        ]

    def test_cross_framework_dtype_preservation_uses_portable_identity(
        self,
        monkeypatch,
    ):
        class TorchDtype:
            def __str__(self):
                return "torch.uint16"

        class Array:
            def __init__(self, dtype):
                self.dtype = dtype

        converted = []

        def scale_dtype(memory_type, array, target_dtype, module=None):
            del memory_type, module
            converted.append(target_dtype)
            return Array(np.dtype(target_dtype))

        monkeypatch.setattr(MemoryType, "scale_dtype", scale_dtype)

        @memory_types("torch", "jax")
        def convert(_image):
            return Array(np.dtype("float32"))

        wrapped = wrap_dtype_preserving_callable(convert, MemoryType.JAX)
        result = wrapped(Array(TorchDtype()))

        assert converted == ["uint16"]
        assert result.dtype == np.dtype("uint16")


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

        if not MemoryType.TORCH.available_device_ids(torch):
            pytest.skip("PyTorch GPU stacking requires an available device")

        @numpy(output_type="torch", slice_by_slice_default=True)
        def planes_to_torch(value):
            return torch.as_tensor(value)

        source = np.arange(24, dtype=np.uint16).reshape(3, 2, 4)
        result = planes_to_torch(source)

        assert isinstance(result, torch.Tensor)
        assert result.dtype is torch.uint16
        assert tuple(result.shape) == source.shape

    def test_decorator_exports_are_declaration_derived(self):
        import arraybridge
        from arraybridge import decorators

        expected = {memory_type.value for memory_type in MemoryType}

        assert expected <= set(decorators.__all__)
        assert expected <= set(arraybridge.__all__)
        assert all(callable(getattr(arraybridge, name)) for name in expected)

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

    def test_numpy_decorator_exists(self):
        """Test that numpy decorator is available."""
        from arraybridge.decorators import numpy

        assert callable(numpy)

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

    def test_cupy_decorator_exists(self):
        """Test that cupy decorator is available."""
        from arraybridge.decorators import cupy

        assert callable(cupy)

    def test_torch_decorator_exists(self):
        """Test that torch decorator is available."""
        from arraybridge.decorators import torch

        assert callable(torch)

    def test_tensorflow_decorator_exists(self):
        """Test that tensorflow decorator is available."""
        from arraybridge.decorators import tensorflow

        assert callable(tensorflow)

    def test_jax_decorator_exists(self):
        """Test that jax decorator is available."""
        from arraybridge.decorators import jax

        assert callable(jax)

    def test_pyclesperanto_decorator_exists(self):
        """Test that pyclesperanto decorator is available."""
        from arraybridge.decorators import pyclesperanto

        assert callable(pyclesperanto)


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
