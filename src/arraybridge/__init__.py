"""
arraybridge: Unified API for NumPy, CuPy, PyTorch, TensorFlow, JAX, and pyclesperanto.

This package provides automatic memory type conversion, declarative decorators,
and unified utilities for working with multiple array/tensor frameworks.

Package-root imports stay lightweight: submodule exports load on first
attribute access instead of at import time. Declaration-only consumers
(memory-type tables, configuration modules) therefore do not pay the
NumPy/numcodecs import cost.
"""

__version__ = "0.5.0"

_LAZY_EXPORTS: dict[str, str] = {
    "MemoryType": ".types",
    "ArrayLibrary": ".libraries",
    "Array": ".libraries",
    "Device": ".devices",
    "HostDevice": ".devices",
    "AcceleratorDevice": ".devices",
    "NamedDevice": ".devices",
    "HOST": ".devices",
    "ConversionRoute": ".conversion",
    "ConversionRecord": ".conversion",
    "ConversionObserver": ".conversion",
    "ConversionCounters": ".conversion",
    "MemoryContractAttribute": ".types",
    "ArrayPayload": ".array_payload",
    "ArrayGeometry": ".array_geometry",
    "convert_memory": ".converters",
    "detect_memory_type": ".converters",
    "memory_types": ".decorators",
    "DtypeConversion": ".decorators",
    "SliceBySliceRuntimeParameter": ".decorators",
    "wrap_dtype_preserving_callable": ".decorators",
    "stack_slices": ".stack_utils",
    "unstack_slices": ".stack_utils",
    "process_slices": ".slice_processing",
    "MemoryConversionError": ".exceptions",
    "_ensure_module": ".utils",
    "_supports_dlpack": ".utils",
}

def _library_names() -> list[str]:
    """The registered library names; each is also the name of its decorator."""

    from arraybridge.libraries import ArrayLibrary

    return list(ArrayLibrary.__registry__)


def __getattr__(name: str):
    import importlib

    if name == "__all__":
        # Read on demand so that a library registered later is exported too.
        return [*_LAZY_EXPORTS, *_library_names()]
    # A name that is not a fixed export is the decorator of a registered library.
    module_name = _LAZY_EXPORTS.get(name, ".decorators")
    if name.startswith("_") and name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS) | set(_library_names()))
