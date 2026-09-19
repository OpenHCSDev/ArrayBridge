"""
arraybridge: Unified API for NumPy, CuPy, PyTorch, TensorFlow, JAX, and pyclesperanto.

This package provides automatic memory type conversion, declarative decorators,
and unified utilities for working with multiple array/tensor frameworks.

Package-root imports stay lightweight: submodule exports load on first
attribute access instead of at import time. Declaration-only consumers
(memory-type tables, configuration modules) therefore do not pay the
NumPy/numcodecs import cost.
"""

__version__ = "0.3.4"

_LAZY_EXPORTS: dict[str, str] = {
    "MemoryType": ".types",
    "MemoryContractAttribute": ".types",
    "ArrayPayload": ".array_payload",
    "ArrayGeometry": ".array_geometry",
    "CPU_MEMORY_TYPES": ".types",
    "GPU_MEMORY_TYPES": ".types",
    "SUPPORTED_MEMORY_TYPES": ".types",
    "convert_memory": ".converters",
    "detect_memory_type": ".converters",
    "memory_types": ".decorators",
    "DtypeConversion": ".decorators",
    "SliceBySliceRuntimeParameter": ".decorators",
    "wrap_dtype_preserving_callable": ".decorators",
    "stack_slices": ".stack_utils",
    "unstack_slices": ".stack_utils",
    "process_slices": ".slice_processing",
    "cleanup_all_gpu_frameworks": ".gpu_cleanup",
    "MemoryConversionError": ".exceptions",
    "SCALING_FUNCTIONS": ".dtype_scaling",
    "_FRAMEWORK_CONFIG": ".framework_config",
    "_FRAMEWORK_OPS": ".framework_ops",
    "_execute_with_oom_recovery": ".oom_recovery",
    "_ensure_module": ".utils",
    "_supports_dlpack": ".utils",
    "_get_device_id": ".utils",
    # Decorator exports named after each memory type (numpy, cupy, torch, ...).
    "numpy": ".decorators",
    "cupy": ".decorators",
    "torch": ".decorators",
    "tensorflow": ".decorators",
    "jax": ".decorators",
    "pyclesperanto": ".decorators",
}

__all__ = [
    # Types
    "MemoryType",
    "MemoryContractAttribute",
    "ArrayPayload",
    "ArrayGeometry",
    "CPU_MEMORY_TYPES",
    "GPU_MEMORY_TYPES",
    "SUPPORTED_MEMORY_TYPES",
    # Converters
    "convert_memory",
    "detect_memory_type",
    # Decorators
    "memory_types",
    "DtypeConversion",
    "SliceBySliceRuntimeParameter",
    "wrap_dtype_preserving_callable",
    # Stack utilities
    "stack_slices",
    "unstack_slices",
    # Slice processing
    "process_slices",
    # GPU cleanup
    "cleanup_all_gpu_frameworks",
    # Exceptions
    "MemoryConversionError",
    # Scaling
    "SCALING_FUNCTIONS",
    # Framework config (internal but needed by some consumers)
    "_FRAMEWORK_CONFIG",
    "_FRAMEWORK_OPS",
    # OOM recovery
    "_execute_with_oom_recovery",
    # Utils
    "_ensure_module",
    "_supports_dlpack",
    "_get_device_id",
    "numpy",
    "cupy",
    "torch",
    "tensorflow",
    "jax",
    "pyclesperanto",
]


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
