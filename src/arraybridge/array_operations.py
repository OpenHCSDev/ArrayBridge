"""Typed array-operation leaves carried by ``MemoryType`` declarations."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

ToNumpy = Callable[[Any, Any], Any]
FromNumpy = Callable[[Any, Any, int], Any]
StackArrays = Callable[[Sequence[Any], Any], Any]
ScaleDtype = Callable[[Any, Any, Any], Any]

_SCALING_RANGES: dict[str, float | tuple[float, float]] = {
    "uint8": 255.0,
    "uint16": 65535.0,
    "uint32": 4294967295.0,
    "int16": (65535.0, 32768.0),
    "int32": (4294967295.0, 2147483648.0),
}


def _dtype_name(dtype: Any) -> str:
    return getattr(dtype, "__name__", str(dtype).rsplit(".", maxsplit=1)[-1])


def _scaled_values(result: Any, result_min: Any, result_max: Any, target_dtype: Any) -> Any:
    normalized = (result - result_min) / (result_max - result_min)
    range_info = _SCALING_RANGES.get(_dtype_name(target_dtype))
    if range_info is None:
        return normalized
    if isinstance(range_info, tuple):
        scale, offset = range_info
        return normalized * scale - offset
    return normalized * range_info


def _clamp_bounds(target_dtype: Any) -> tuple[float, float] | None:
    range_info = _SCALING_RANGES.get(_dtype_name(target_dtype))
    if range_info is None:
        return None
    if isinstance(range_info, tuple):
        scale, offset = range_info
        return -offset, scale - offset - 128
    return 0, range_info


def _identity_to_numpy(data: Any, module: Any) -> Any:
    del module
    return data


def _identity_from_numpy(data: Any, module: Any, device_id: int) -> Any:
    del module, device_id
    return data


def _numpy_stack(values: Sequence[Any], module: Any) -> Any:
    return module.stack(values, axis=0)


def _numpy_scale(result: Any, target_dtype: Any, module: Any) -> Any:
    if not hasattr(result, "dtype"):
        return result
    if not (
        module.issubdtype(result.dtype, module.floating)
        and module.issubdtype(target_dtype, module.integer)
    ):
        return result.astype(target_dtype)
    result_min = result.min()
    result_max = result.max()
    if result_max <= result_min:
        return result.astype(target_dtype)
    scaled = _scaled_values(result, result_min, result_max, target_dtype)
    bounds = _clamp_bounds(target_dtype)
    if bounds is not None:
        scaled = module.clip(scaled, *bounds)
    return scaled.astype(target_dtype)


def _cupy_to_numpy(data: Any, module: Any) -> Any:
    del module
    return data.get()


def _cupy_from_numpy(data: Any, module: Any, device_id: int) -> Any:
    del device_id
    return module.array(data)


def _cupy_stack(values: Sequence[Any], module: Any) -> Any:
    return module.stack(values, axis=0)


def _cupy_scale(result: Any, target_dtype: Any, module: Any) -> Any:
    if not hasattr(result, "dtype"):
        return result
    if not (
        module.issubdtype(result.dtype, module.floating)
        and not module.issubdtype(target_dtype, module.floating)
    ):
        return result.astype(target_dtype)
    result_min = module.min(result)
    result_max = module.max(result)
    if result_max <= result_min:
        return result.astype(target_dtype)
    scaled = _scaled_values(result, result_min, result_max, target_dtype)
    bounds = _clamp_bounds(target_dtype)
    if bounds is not None:
        scaled = module.clip(scaled, *bounds)
    return scaled.astype(target_dtype)


def _torch_to_numpy(data: Any, module: Any) -> Any:
    del module
    return data.cpu().numpy()


def _torch_from_numpy(data: Any, module: Any, device_id: int) -> Any:
    host_data = (
        np.ascontiguousarray(data)
        if any(stride < 0 for stride in getattr(data, "strides", ()))
        else data
    )
    return module.from_numpy(host_data).to(f"cuda:{device_id}")


def _torch_stack(values: Sequence[Any], module: Any) -> Any:
    return module.stack(tuple(values), dim=0)


def _mapped_dtype(target_dtype: Any, module: Any) -> Any:
    try:
        dtype_name = np.dtype(target_dtype).name
    except TypeError as error:
        raise TypeError(f"Unsupported target dtype {target_dtype!r}") from error
    mapped = getattr(module, dtype_name, None)
    if mapped is None:
        module_name = getattr(module, "__name__", type(module).__name__)
        raise TypeError(f"{module_name} does not expose dtype {dtype_name}")
    return mapped


def _torch_scale(result: Any, target_dtype: Any, module: Any) -> Any:
    if not hasattr(result, "dtype"):
        return result
    mapped = _mapped_dtype(target_dtype, module)
    floats = (module.float16, module.float32, module.float64)
    if not (result.dtype in floats and np.issubdtype(np.dtype(target_dtype), np.integer)):
        return result.to(mapped)
    result_min = result.min()
    result_max = result.max()
    if result_max <= result_min:
        return result.to(mapped)
    scaled = _scaled_values(result, result_min, result_max, target_dtype)
    bounds = _clamp_bounds(target_dtype)
    if bounds is not None:
        scaled = module.clamp(scaled, min=bounds[0], max=bounds[1])
    return scaled.to(mapped)


def _tensorflow_to_numpy(data: Any, module: Any) -> Any:
    del module
    return data.numpy()


def _tensorflow_from_numpy(data: Any, module: Any, device_id: int) -> Any:
    del device_id
    return module.convert_to_tensor(data)


def _tensorflow_stack(values: Sequence[Any], module: Any) -> Any:
    return module.stack(tuple(values), axis=0)


def _tensorflow_scale(result: Any, target_dtype: Any, module: Any) -> Any:
    if not hasattr(result, "dtype"):
        return result
    mapped = _mapped_dtype(target_dtype, module)
    floats = (module.float16, module.float32, module.float64)
    if not (result.dtype in floats and np.issubdtype(np.dtype(target_dtype), np.integer)):
        return module.cast(result, mapped)
    result_min = module.reduce_min(result)
    result_max = module.reduce_max(result)
    if result_max <= result_min:
        return module.cast(result, mapped)
    scaled = _scaled_values(result, result_min, result_max, target_dtype)
    bounds = _clamp_bounds(target_dtype)
    if bounds is not None:
        scaled = module.clip_by_value(scaled, *bounds)
    return module.cast(scaled, mapped)


def _jax_to_numpy(data: Any, module: Any) -> Any:
    del module
    return np.asarray(data)


def _jax_from_numpy(data: Any, module: Any, device_id: int) -> Any:
    devices = tuple(device for device in module.devices() if device.platform == "gpu")
    return module.device_put(data, devices[device_id])


def _jax_stack(values: Sequence[Any], module: Any) -> Any:
    return module.numpy.stack(tuple(values), axis=0)


def _jax_scale(result: Any, target_dtype: Any, module: Any) -> Any:
    if not hasattr(result, "dtype"):
        return result
    if np.dtype(target_dtype) == np.dtype(np.float64):
        x64_enabled = getattr(module.config, "x64_enabled", None)
        if x64_enabled is None:
            x64_enabled = module.config.read("jax_enable_x64")
        if not x64_enabled:
            raise ValueError(
                "JAX float64 output requires x64 mode; set JAX_ENABLE_X64=true before import"
            )
    jnp = module.numpy
    mapped = _mapped_dtype(target_dtype, jnp)
    floats = (jnp.float16, jnp.float32, jnp.float64)
    if not (result.dtype in floats and np.issubdtype(np.dtype(target_dtype), np.integer)):
        return result.astype(mapped)
    result_min = jnp.min(result)
    result_max = jnp.max(result)
    if result_max <= result_min:
        return result.astype(mapped)
    scaled = _scaled_values(result, result_min, result_max, target_dtype)
    bounds = _clamp_bounds(target_dtype)
    if bounds is not None:
        scaled = jnp.clip(scaled, *bounds)
    return scaled.astype(mapped)


def _pyclesperanto_to_numpy(data: Any, module: Any) -> Any:
    return module.pull(data)


def _pyclesperanto_from_numpy(data: Any, module: Any, device_id: int) -> Any:
    del device_id
    return module.push(data)


def _pyclesperanto_stack(values: Sequence[Any], module: Any) -> Any:
    if not values:
        raise ValueError("Cannot stack an empty pyclesperanto sequence")
    if len(values) == 1:
        source = values[0]
        result = module.create((1, *source.shape), dtype=source.dtype)
        return module.copy_slice(source, result, 0)
    result = values[0]
    for value in values[1:]:
        result = module.concatenate_along_z(result, value)
    return result


def _pyclesperanto_scale(result: Any, target_dtype: Any, module: Any) -> Any:
    if not hasattr(result, "dtype"):
        return result
    target_is_int = target_dtype in {
        np.uint8,
        np.uint16,
        np.uint32,
        np.int8,
        np.int16,
        np.int32,
    }
    if not (np.issubdtype(result.dtype, np.floating) and target_is_int):
        return module.push(module.pull(result).astype(target_dtype))
    result_min = float(module.minimum_of_all_pixels(result))
    result_max = float(module.maximum_of_all_pixels(result))
    if result_max <= result_min:
        return module.push(module.pull(result).astype(target_dtype))
    normalized = module.subtract_image_from_scalar(result, scalar=result_min)
    normalized = module.multiply_image_and_scalar(
        normalized,
        scalar=1.0 / (result_max - result_min),
    )
    range_info = _SCALING_RANGES.get(_dtype_name(target_dtype))
    if isinstance(range_info, tuple):
        scale, offset = range_info
        scaled = module.multiply_image_and_scalar(normalized, scalar=scale)
        scaled = module.subtract_image_from_scalar(scaled, scalar=offset)
    elif range_info is not None:
        scaled = module.multiply_image_and_scalar(normalized, scalar=range_info)
    else:
        scaled = normalized
    host_values = module.pull(scaled)
    bounds = _clamp_bounds(target_dtype)
    if bounds is not None:
        host_values = np.clip(host_values, *bounds)
    return module.push(host_values.astype(target_dtype))


@dataclass(frozen=True, slots=True)
class ArrayOperations:
    """Framework-specific array leaves referenced by one declaration."""

    to_numpy: ToNumpy
    from_numpy: FromNumpy
    stack: StackArrays
    scale_dtype: ScaleDtype


NUMPY_OPERATIONS = ArrayOperations(
    to_numpy=_identity_to_numpy,
    from_numpy=_identity_from_numpy,
    stack=_numpy_stack,
    scale_dtype=_numpy_scale,
)
CUPY_OPERATIONS = ArrayOperations(
    to_numpy=_cupy_to_numpy,
    from_numpy=_cupy_from_numpy,
    stack=_cupy_stack,
    scale_dtype=_cupy_scale,
)
TORCH_OPERATIONS = ArrayOperations(
    to_numpy=_torch_to_numpy,
    from_numpy=_torch_from_numpy,
    stack=_torch_stack,
    scale_dtype=_torch_scale,
)
TENSORFLOW_OPERATIONS = ArrayOperations(
    to_numpy=_tensorflow_to_numpy,
    from_numpy=_tensorflow_from_numpy,
    stack=_tensorflow_stack,
    scale_dtype=_tensorflow_scale,
)
JAX_OPERATIONS = ArrayOperations(
    to_numpy=_jax_to_numpy,
    from_numpy=_jax_from_numpy,
    stack=_jax_stack,
    scale_dtype=_jax_scale,
)
PYCLESPERANTO_OPERATIONS = ArrayOperations(
    to_numpy=_pyclesperanto_to_numpy,
    from_numpy=_pyclesperanto_from_numpy,
    stack=_pyclesperanto_stack,
    scale_dtype=_pyclesperanto_scale,
)
