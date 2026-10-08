"""Native array operations carried by MemoryType declarations."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

_SCALING_RANGES: dict[str, float | tuple[float, float]] = {
    "uint8": 255.0,
    "uint16": 65535.0,
    "uint32": 4294967295.0,
    "int16": (65535.0, 32768.0),
    "int32": (4294967295.0, 2147483648.0),
}


def _dtype_name(dtype: Any) -> str:
    declared_name = getattr(dtype, "name", None)
    if declared_name is not None:
        return str(declared_name)
    return getattr(dtype, "__name__", str(dtype).rsplit(".", maxsplit=1)[-1])


def _numpy_dtype_name(dtype: Any) -> str:
    return str(np.dtype(dtype).name)


def _torch_dtype_name(dtype: Any) -> str:
    return str(dtype).rsplit(".", maxsplit=1)[-1]


def _tensorflow_dtype_name(dtype: Any) -> str:
    numpy_dtype = getattr(dtype, "as_numpy_dtype", dtype)
    return str(np.dtype(numpy_dtype).name)


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


class ArrayOperations(ABC):
    """Native array semantics selected by the existing MemoryType declaration."""

    @staticmethod
    @abstractmethod
    def to_numpy(data: Any, module: Any) -> Any:
        """Project pixels to the explicit host boundary."""

    @staticmethod
    @abstractmethod
    def from_numpy(data: Any, module: Any, device_id: int) -> Any:
        """Admit host pixels on the selected framework device."""

    @staticmethod
    @abstractmethod
    def scale_dtype(result: Any, target_dtype: Any, module: Any) -> Any:
        """Apply the framework's existing intensity conversion law."""

    @staticmethod
    def dtype_name(dtype: Any) -> str:
        return _numpy_dtype_name(dtype)

    @staticmethod
    def stack(values: Sequence[Any], module: Any) -> Any:
        return module.stack(tuple(values), axis=0)

    @staticmethod
    def cast(data: Any, dtype: Any, module: Any) -> Any:
        return data.astype(dtype, copy=False)

    @staticmethod
    def logical_and(left: Any, right: Any, module: Any) -> Any:
        return module.logical_and(left, right)

    @staticmethod
    def reshape(data: Any, shape: tuple[int, ...], module: Any) -> Any:
        return data.reshape(shape)

    @staticmethod
    def broadcast_to(data: Any, shape: tuple[int, ...], module: Any) -> Any:
        return module.broadcast_to(data, shape)

    @staticmethod
    def ones_like(reference: Any, shape: tuple[int, ...], dtype: Any, module: Any) -> Any:
        return module.ones(shape, dtype=dtype)

    @classmethod
    def normalize_planes(
        cls,
        data: Any,
        dtype: Any,
        scales: Sequence[float | None],
        module: Any,
    ) -> Any:
        """Cast and scale each plane before assembling its owned output."""
        return cls.stack(
            tuple(
                (
                    cls.cast(plane, dtype, module)
                    if scale is None
                    else cls.cast(plane, dtype, module) / float(scale)
                )
                for plane, scale in zip(data, scales, strict=True)
            ),
            module,
        )


class MutableArrayOperations(ArrayOperations):
    """Arrays whose allocated output supports native in-place arithmetic."""

    @classmethod
    def normalize_planes(
        cls,
        data: Any,
        dtype: Any,
        scales: Sequence[float | None],
        module: Any,
    ) -> Any:
        # Integer division promotes the output dtype in the original recipe.
        if not np.issubdtype(np.dtype(dtype), np.inexact) or len(data) == 0:
            return super().normalize_planes(data, dtype, scales, module)
        normalized = module.array(data, dtype=dtype, copy=True)
        for index, scale in enumerate(scales):
            if scale is not None:
                normalized[index] /= float(scale)
        return normalized


class NumpyArrayOperations(MutableArrayOperations):
    """Numpy native operation leaves."""

    @staticmethod
    def to_numpy(data: Any, module: Any) -> Any:
        del module
        return data

    @staticmethod
    def from_numpy(data: Any, module: Any, device_id: int) -> Any:
        del module, device_id
        return data

    @staticmethod
    def scale_dtype(result: Any, target_dtype: Any, module: Any) -> Any:
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


class CupyArrayOperations(MutableArrayOperations):
    """Cupy native operation leaves."""

    @staticmethod
    def to_numpy(data: Any, module: Any) -> Any:
        del module
        return data.get()

    @staticmethod
    def from_numpy(data: Any, module: Any, device_id: int) -> Any:
        del device_id
        return module.array(data)

    @staticmethod
    def scale_dtype(result: Any, target_dtype: Any, module: Any) -> Any:
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


class TorchArrayOperations(ArrayOperations):
    """Torch native operation leaves."""

    @staticmethod
    def to_numpy(data: Any, module: Any) -> Any:
        del module
        return data.cpu().numpy()

    @staticmethod
    def from_numpy(data: Any, module: Any, device_id: int) -> Any:
        host_data = (
            np.ascontiguousarray(data)
            if any(stride < 0 for stride in getattr(data, "strides", ()))
            else data
        )
        return module.from_numpy(host_data).to(f"cuda:{device_id}")

    @staticmethod
    def scale_dtype(result: Any, target_dtype: Any, module: Any) -> Any:
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

    @staticmethod
    def stack(values: Sequence[Any], module: Any) -> Any:
        return module.stack(tuple(values), dim=0)

    @staticmethod
    def cast(data: Any, dtype: Any, module: Any) -> Any:
        return data.to(dtype=_mapped_dtype(dtype, module))

    @staticmethod
    def dtype_name(dtype: Any) -> str:
        return str(dtype).rsplit(".", maxsplit=1)[-1]

    @staticmethod
    def broadcast_to(data: Any, shape: tuple[int, ...], module: Any) -> Any:
        return data.expand(shape)

    @staticmethod
    def ones_like(reference: Any, shape: tuple[int, ...], dtype: Any, module: Any) -> Any:
        return module.ones(shape, dtype=_mapped_dtype(dtype, module), device=reference.device)


class TensorflowArrayOperations(ArrayOperations):
    """Tensorflow native operation leaves."""

    @staticmethod
    def to_numpy(data: Any, module: Any) -> Any:
        del module
        return data.numpy()

    @staticmethod
    def from_numpy(data: Any, module: Any, device_id: int) -> Any:
        del device_id
        return module.convert_to_tensor(data)

    @staticmethod
    def scale_dtype(result: Any, target_dtype: Any, module: Any) -> Any:
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

    @staticmethod
    def cast(data: Any, dtype: Any, module: Any) -> Any:
        return module.cast(data, _mapped_dtype(dtype, module))

    @staticmethod
    def reshape(data: Any, shape: tuple[int, ...], module: Any) -> Any:
        return module.reshape(data, shape)

    @staticmethod
    def ones_like(reference: Any, shape: tuple[int, ...], dtype: Any, module: Any) -> Any:
        with module.device(reference.device):
            return module.ones(shape, dtype=_mapped_dtype(dtype, module))


class JaxArrayOperations(ArrayOperations):
    """Jax native operation leaves."""

    @staticmethod
    def to_numpy(data: Any, module: Any) -> Any:
        del module
        return np.asarray(data)

    @staticmethod
    def from_numpy(data: Any, module: Any, device_id: int) -> Any:
        devices = tuple(device for device in module.devices() if device.platform == "gpu")
        return module.device_put(data, devices[device_id])

    @staticmethod
    def scale_dtype(result: Any, target_dtype: Any, module: Any) -> Any:
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

    @staticmethod
    def stack(values: Sequence[Any], module: Any) -> Any:
        return module.numpy.stack(tuple(values), axis=0)

    @staticmethod
    def cast(data: Any, dtype: Any, module: Any) -> Any:
        return data.astype(_mapped_dtype(dtype, module.numpy))

    @staticmethod
    def logical_and(left: Any, right: Any, module: Any) -> Any:
        return module.numpy.logical_and(left, right)

    @staticmethod
    def broadcast_to(data: Any, shape: tuple[int, ...], module: Any) -> Any:
        return module.numpy.broadcast_to(data, shape)

    @staticmethod
    def ones_like(reference: Any, shape: tuple[int, ...], dtype: Any, module: Any) -> Any:
        return module.numpy.ones(shape, dtype=dtype, device=reference.device)


class PyclesperantoArrayOperations(ArrayOperations):
    """Pyclesperanto native operation leaves."""

    @staticmethod
    def to_numpy(data: Any, module: Any) -> Any:
        return module.pull(data)

    @staticmethod
    def from_numpy(data: Any, module: Any, device_id: int) -> Any:
        del device_id
        return module.push(data)

    @staticmethod
    def scale_dtype(result: Any, target_dtype: Any, module: Any) -> Any:
        if not hasattr(result, "dtype"):
            return result
        target_is_int = np.issubdtype(np.dtype(target_dtype), np.integer)
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

    @staticmethod
    def stack(values: Sequence[Any], module: Any) -> Any:
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

    @staticmethod
    def cast(data: Any, dtype: Any, module: Any) -> Any:
        return module.push(module.pull(data).astype(dtype, copy=False))

    @staticmethod
    def logical_and(left: Any, right: Any, module: Any) -> Any:
        return module.push(np.logical_and(module.pull(left), module.pull(right)))

    @staticmethod
    def reshape(data: Any, shape: tuple[int, ...], module: Any) -> Any:
        if tuple(data.shape) == shape:
            return data
        raise NotImplementedError(
            "pyclesperanto does not provide native array reshape; its reshape downloads pixels"
        )

    @staticmethod
    def broadcast_to(data: Any, shape: tuple[int, ...], module: Any) -> Any:
        if tuple(data.shape) == shape:
            return data
        raise NotImplementedError("pyclesperanto does not provide native array broadcasting")

    @staticmethod
    def ones_like(reference: Any, shape: tuple[int, ...], dtype: Any, module: Any) -> Any:
        if np.dtype(dtype).name not in {
            "float32",
            "int8",
            "int16",
            "int32",
            "uint8",
            "uint16",
            "uint32",
        }:
            raise NotImplementedError(f"pyclesperanto cannot allocate native dtype {dtype!r}")
        if not 1 <= len(shape) <= 3:
            raise NotImplementedError(
                "pyclesperanto native allocation requires one to three dimensions"
            )
        result = module.create(shape, dtype=dtype, device=reference.device)
        module.set(result, scalar=1, device=reference.device)
        return result


NUMPY_OPERATIONS = NumpyArrayOperations()
CUPY_OPERATIONS = CupyArrayOperations()
TORCH_OPERATIONS = TorchArrayOperations()
TENSORFLOW_OPERATIONS = TensorflowArrayOperations()
JAX_OPERATIONS = JaxArrayOperations()
PYCLESPERANTO_OPERATIONS = PyclesperantoArrayOperations()
