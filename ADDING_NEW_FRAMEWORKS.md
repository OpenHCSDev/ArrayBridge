# Adding a library

A library is one class: a subclass of `arraybridge.libraries.ArrayLibrary`
that inherits the role parents it plays, states `name = "..."`, and defines
the abstract methods those parents ask for. Nothing else is edited. The class
can live in `src/arraybridge/libraries.py` or in any other module; it
registers when its module is imported.

## What follows from the class

- Lookup: `ArrayLibrary.named(name)`, `ArrayLibrary.of(array)`,
  `ArrayLibrary.for_module_name(module_name)`, `ArrayLibrary.all()`.
- A `MemoryType` member: `MemoryType.<NAME>` with value `name` and
  `.library` returning the library object.
- The decorator `arraybridge.<name>`.
- Conversion to and from every other library through `convert_memory`,
  `stack_slices` and `unstack_slices`. `ConversionRoute.planned` picks the
  route from the role parents: same library is a device move, a DLPack
  importer paired with a DLPack exporter is a DLPack exchange, anything else
  is a copy through NumPy.
- Device answers (`host_devices`, `accelerator_devices`, `available_devices`,
  `require_device`) and the array operations (`max`, `reshape`,
  `stack_arrays`, ...) with NumPy-style defaults that call the library's
  module.

## A host-only example

This file runs as it is with only NumPy installed.

```python
import numpy as np

from arraybridge import HOST, ConversionRoute, MemoryType
from arraybridge import convert_memory, detect_memory_type, stack_slices
from arraybridge.libraries import ArrayLibrary, HostDevices


class FrozenArray:
    """Values in main memory that cannot be changed after creation."""

    def __init__(self, values):
        self.values = np.array(values)
        self.values.flags.writeable = False
        self.shape = self.values.shape
        self.dtype = self.values.dtype

    def __getitem__(self, index):
        return FrozenArray(self.values[index])


class FrozenLibrary(HostDevices[FrozenArray]):
    name = "frozen"
    # The module whose classes are this library's arrays. It defaults to
    # ``name``; here the array class lives in this file.
    import_name = __name__

    # asked for by ArrayLibrary
    def device_of(self, data):
        return HOST

    def _to_host(self, data):
        return np.array(data.values)

    def _scale_dtype(self, result, target_dtype):
        return FrozenArray(result.values.astype(target_dtype))

    # asked for by HostDevices
    def _wrap_host_array(self, data):
        return FrozenArray(data)

    def _move_to_host(self, data):
        return data

    # Array operations: nothing is inherited for them, because the
    # base class assumes no array API. This array wraps NumPy values.
    def accepts(self, value):
        return isinstance(value, FrozenArray)
    
    def max(self, data, axis):
        return FrozenArray(data.values.max(axis=axis))
    
    def min(self, data, axis):
        return FrozenArray(data.values.min(axis=axis))
    
    def mean(self, data, axis):
        return FrozenArray(data.values.mean(axis=axis))
    
    def sum(self, data, axis):
        return FrozenArray(data.values.sum(axis=axis))
    
    def assign(self, data, index, values):
        updated = data.values.copy()
        updated[index] = values.values
        return FrozenArray(updated)
    
    def _stack(self, values):
        return FrozenArray(np.stack([value.values for value in values]))
    
    def _cast(self, data, dtype):
        return FrozenArray(data.values.astype(dtype))
    
    def _logical_and(self, left, right):
        return FrozenArray(np.logical_and(left.values, right.values))
    
    def _reshape(self, data, shape):
        return FrozenArray(data.values.reshape(shape))
    
    def _broadcast_to(self, data, shape):
        return FrozenArray(np.broadcast_to(data.values, shape))
    
    def _ones_like(self, reference, shape, dtype):
        return FrozenArray(np.ones(shape, dtype=dtype))


# Lookup, by name and by array.
library = ArrayLibrary.named("frozen")
assert library in ArrayLibrary.all()
assert ArrayLibrary.of(FrozenArray([1, 2])) is library

# The MemoryType member.
assert MemoryType.FROZEN.value == "frozen"
assert MemoryType.FROZEN.library is library

# Devices.
assert library.available_devices() == (HOST,)

# Conversion in both directions, by host copy.
numpy_library = ArrayLibrary.named("numpy")
assert ConversionRoute.planned(numpy_library, HOST, library, HOST).mechanism == "host copy"
frozen = convert_memory(np.arange(6).reshape(2, 3), "numpy", "frozen", HOST)
assert detect_memory_type(frozen) == "frozen"
assert convert_memory(frozen, "frozen", "numpy", HOST).tolist() == [[0, 1, 2], [3, 4, 5]]

# Stacking.
stack = stack_slices([frozen, np.ones((2, 3))], "frozen", HOST)
assert stack.shape == (2, 2, 3)

# The decorator named after the library.
from arraybridge import frozen as frozen_function


@frozen_function
def first_row(image):
    return image[0]


assert first_row.execution_memory_type == "frozen"
assert first_row(frozen).shape == (3,)
```

## What the class states

| Attribute | Meaning | Default |
|---|---|---|
| `name` | The registered name, the `MemoryType` value and the decorator name. | required |
| `import_name` | Top-level module to import; objects from it belong to the library. | `name` |
| `display_name` | Name used in messages. | `name` |
| `module_aliases` | Other top-level module names whose objects belong to the library (JAX: `jaxlib`). | `()` |
| `import_environment` | Environment variables set, if unset, before the first import and in child processes. | `()` |
| `oom_messages` | Lower-case text that marks an out-of-memory error message. | `()` |
| `oom_exception_paths` | Attribute paths from the module to its out-of-memory exception classes. | `()` |

Every library defines three methods asked for by `ArrayLibrary` itself, and
the array operations listed under "Array operations" below:

- `device_of(data)`: the device holding the array. It never returns `None`.
- `_to_host(data)`: the array as a NumPy array.
- `_scale_dtype(result, target_dtype)`: convert the dtype, rescaling float
  intensities into the integer range where the library does that.

## Role parents

Inherit each role the library plays. The abstract methods are the ones the
class must define.

| Parent | The library... | Abstract methods |
|---|---|---|
| `HostDevices` | keeps arrays in main memory. | `_wrap_host_array(data)`, `_move_to_host(data)` |
| `AcceleratorDevices` | can keep arrays on numbered accelerators. | `_accelerator_count()`, `_accelerator_scope(device)`, `_move_to_accelerator(data, device)`, `_upload(data, device)` |
| `CudaLibrary` | is an `AcceleratorDevices` library with its own CUDA streams and a memory cache to release. | those of `AcceleratorDevices`, plus `stream_scope(stream)`, `current_device()`, `_free_cached_memory()` |
| `DLPackImport` | can wrap another library's buffer through DLPack. | `_import_dlpack(payload)` |
| `DLPackProtocolExport` | has arrays with the standard DLPack export methods on every device. | none |
| `NvidiaWheelLibraries` | loads CUDA from the `nvidia-*` wheels; child processes get their library directories on the search path. | none |

Methods with a default worth knowing:

- `HostDevices._host_scope()`: the block in which new arrays are created in
  main memory. The default does nothing; TensorFlow and JAX override it.
- `AcceleratorDevices._activate_accelerator(device)`: select a device for the
  whole process. The default does nothing.
- `cleanup_loaded(device=None)`: release cached device memory. `CudaLibrary`
  implements it; elsewhere the default does nothing.
- `exports_dlpack_from`, `exports_dlpack_data`, `export_dlpack`: override
  these directly when export depends on the device or the array, as
  TensorFlow does.
- `nbytes_of`, `dtype_name`: the defaults read `shape` and `dtype` from the
  array and name the dtype as NumPy does. Override where the library differs.

## Array operations

`ArrayLibrary` asks every library for `accepts`, `max`, `min`, `mean`, `sum`,
`assign`, `_stack`, `_cast`, `_reshape`, `_broadcast_to`, `_ones_like` and
`_logical_and`, and gives no default for any of them: the base class assumes
no array API. A library defines them or inherits them from a parent it
declares.

| Parent | The library... | Supplies | Abstract methods |
|---|---|---|---|
| `InPlaceAssignment` | has arrays that take `array[index] = values`. | `assign` | none |
| `ArrayMath` | can make arrays without a reference array, on its default device. A library that cannot (pyclesperanto) does not inherit it and has none of these four methods. | nothing | `linspace`, `ones`, `outer`, `floor` |
| `NumpyArrayApi` | has an array module and arrays that follow the NumPy API (`array_module.max(data, axis=...)`, `data.astype(...)`, ...). It is an `ArrayMath` library. | every operation above except `assign`, and the four of `ArrayMath` | none |
| `MutableArrays` | is a `NumpyArrayApi` and `InPlaceAssignment` library; supplies `_scale_dtype` and in-place plane normalisation. | all of the above | `_rescales_into(target_dtype)` |

Override an inherited operation only where the library's numeric rule
differs (JAX's `mean` casts to float32 first).

The six built-in classes at the end of `src/arraybridge/libraries.py` are the
reference for each combination of roles.

## Tests to add

- detection by array and by module name, and behaviour when the package is
  not installed;
- NumPy round trips, and the planned route against every other library;
- each device the library accepts, and `ValueError` for one it does not have;
- moves between two devices of the library;
- stacking and unstacking one, two and several planes;
- cleanup and out-of-memory recognition without importing an absent library.

Run:

```bash
ruff check src tests
black --check src tests
mypy src --ignore-missing-imports
pytest
python -m sphinx -E -W --keep-going -b html docs/source /tmp/arraybridge-docs
```

When accelerator hardware is available, also convert between every installed
pair of libraries in one process. Import order, allocators sharing a device
and DLPack differences only show there.
