"""Callable defaults use the existing typed dtype policy and real conversion."""

import inspect
from dataclasses import dataclass

import numpy as np
import pytest

from arraybridge.decorators import (
    DtypeConversion,
    DtypeConversionConfig,
    PreserveInputDtypeConfig,
)
from arraybridge.decorators import (
    numpy as numpy_func,
)


@dataclass(frozen=True)
class NativeConfig(DtypeConversionConfig):
    default_dtype_conversion: DtypeConversion = DtypeConversion.NATIVE_OUTPUT


def test_declared_native_output_preserves_fraction_negative_and_large_values():
    native = NativeConfig()

    @numpy_func(dtype_config_default=native)
    def correct(image):
        return np.array([-1.25, 12.5, 70000.5], dtype=np.float32)

    output = correct(np.array([0, 1, 65535], dtype=np.uint16))
    np.testing.assert_array_equal(output, [-1.25, 12.5, 70000.5])
    assert output.dtype == np.float32
    assert inspect.signature(correct).parameters["dtype_config"].default is native


def test_explicit_runtime_dtype_policy_overrides_callable_default():
    @numpy_func(dtype_config_default=NativeConfig())
    def correct(image):
        return image.astype(np.float32) + 0.25, np.array([2.5], dtype=np.float32)

    output, sidecar = correct(
        np.array([1, 2], dtype=np.uint16),
        dtype_config=PreserveInputDtypeConfig(),
    )
    assert output.dtype == np.uint16
    assert sidecar.dtype == np.float32
    np.testing.assert_array_equal(sidecar, [2.5])


def test_undeclared_defaults_still_preserve_input_dtype():
    @numpy_func
    def correct(image):
        return image.astype(np.float32) + 0.25

    assert correct(np.array([1, 2], dtype=np.uint16)).dtype == np.uint16


def test_untyped_default_is_rejected_at_decoration():
    with pytest.raises(TypeError, match="DtypeConversionConfig"):
        numpy_func(dtype_config_default={"default_dtype_conversion": "native"})(
            lambda image: image
        )
