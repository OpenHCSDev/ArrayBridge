# Changelog

## 0.5.0

Each supported array library is now one class of one family,
`arraybridge.libraries.ArrayLibrary`. The class states the library's identity
and owns everything that library does its own way: devices, moving data,
streams, DLPack, out-of-memory recognition, child-process environment and
native array operations. Adding a library is writing one class; lookup,
conversion, the `MemoryType` member and the decorator follow from it.

This release removes names without aliases. The table says what to call
instead.

### The family

| Class | Parents (roles) |
|---|---|
| `NumpyLibrary` | `HostDevices`, `MutableArrays` |
| `CupyLibrary` | `CudaLibrary`, `DLPackImport`, `DLPackProtocolExport`, `NvidiaWheelLibraries`, `MutableArrays` |
| `TorchLibrary` | `HostDevices`, `CudaLibrary`, `DLPackImport`, `DLPackProtocolExport` |
| `TensorflowLibrary` | `HostDevices`, `AcceleratorDevices`, `DLPackImport` |
| `JaxLibrary` | `HostDevices`, `AcceleratorDevices`, `DLPackImport`, `DLPackProtocolExport` |
| `PyclesperantoLibrary` | `AcceleratorDevices` |

Lookup: `ArrayLibrary.named("torch")`, `ArrayLibrary.of(array)` (which library
made this), `ArrayLibrary.for_module_name("jaxlib")`, `ArrayLibrary.all()`.

### `MemoryType`

`MemoryType` is kept as a view: an `Enum` whose members are generated from the
registered library classes (`MemoryType.TORCH.value == "torch"`). It carries no
behaviour. `MemoryType.X.library` is the library object. A library class
defined later appears as a member.

### Devices

`arraybridge.devices`: `Device` with leaves `HostDevice()` (the value `HOST`),
`AcceleratorDevice(index)` and `NamedDevice(name)` (a device the library
reports that is not one of its numbered accelerators, such as pyclesperanto on
a CPU OpenCL device). Each has a `label`.

`is_gpu` is removed. It answered "is this NumPy" and was read as "this needs a
GPU", which is false for torch, JAX and TensorFlow. Ask instead:

| Question | Call |
|---|---|
| Does the library keep arrays in main memory? | `isinstance(library, HostDevices)`; `library.host_devices()` |
| Can the library use accelerators at all? | `isinstance(library, AcceleratorDevices)` |
| Which accelerators exist on this machine now? | `library.accelerator_devices()` |
| Every device it can use now | `library.available_devices()` |
| Where is this array? | `library.device_of(array)` (never `None`) |

Every `gpu_id` / `device_id` integer parameter is now a `Device`. A device the
library does not have raises `ValueError` (`convert_memory` wraps it in
`MemoryConversionError`, as it wraps every failure); in particular a NumPy target with an
accelerator device raises (0.4.1 ignored the number), and torch, JAX and
TensorFlow accept `HOST` (0.4.1 always moved torch data to CUDA).
`process_slices` keeps a torch CPU tensor on the host (0.4.1 sent it to GPU 0).

### Conversion

`arraybridge.conversion`:

- `ConversionRoute`: the mechanisms, tried in declaration order: `DeviceMove`
  (`"device move"`), `DLPackExchange` (`"dlpack"`), `HostCopy` (`"host copy"`).
  `ConversionRoute.planned(source_library, source_device, target_library,
  target_device)` returns the route that will be tried first. It reads only
  the library declarations and the device values: no device, no import. The
  conversion itself selects its route with the same test.
- `ConversionRecord`: source and target library and device label, mechanism,
  `nbytes`, `seconds`, and `declined_mechanisms` (routes that applied by
  declaration but could not carry that array). Plain fields; `to_json()`,
  `from_json()`.
- `ConversionObserver`: one hook, `ConversionObserver.current`. The default
  ignores conversions and nothing is timed. `observer.install()` or
  `with observer.installed():`. `convert_to`, `to_numpy` and `from_numpy`
  report; a call that returns its input unchanged does not.
- `ConversionCounters`: an observer keeping totals per (source, target,
  mechanism); `merge`, `rows`, `to_json`, `from_json`, `format_table`.

DLPack is no longer attempted from a main-memory buffer into CuPy, which
cannot wrap one; the result (a host copy) is unchanged and the warning is gone.

### Removed, and what to call instead

| Removed | Use |
|---|---|
| `MemoryType.X.<method>(...)` (any method) | `MemoryType.X.library.<method>(...)` |
| the `module=` argument of those methods | gone; the library uses its imported module |
| `MemoryType.is_gpu` | see Devices |
| `CPU_MEMORY_TYPES`, `GPU_MEMORY_TYPES` | filter `MemoryType` by `isinstance(m.library, HostDevices / AcceleratorDevices)` |
| `SUPPORTED_MEMORY_TYPES`, `VALID_MEMORY_TYPES`, `VALID_GPU_MEMORY_TYPES`, `MEMORY_TYPE_*` | `MemoryType` itself; `m.value` |
| `device_id_of(data)` | `library.device_of(data)` |
| `available_device_ids()` | `library.accelerator_devices()` |
| `require_device(id)` returning the module | `library.require_device(device)` returns `None`; `library.module` |
| `canonical_dtype_name(dtype)` | `library.dtype_name(dtype)` |
| `supports_dlpack` | `library.imports_dlpack`, `library.imports_dlpack_from(device)` |
| `supports_dlpack_data(data)` | `library.exports_dlpack_data(data)` |
| `current_device_id()`, `create_stream()` | on `CudaLibrary` only: `current_device()`, `create_stream()`, `thread_stream(device)` |
| `MemoryType.subprocess_environment(env)` | `ArrayLibrary.child_environment(env)` |
| `arraybridge.types.FrameworkRuntime` and the `_<library>_<capability>` functions | methods of the library classes |
| `arraybridge.types.DLPackPayload` | `arraybridge.libraries.DLPackPayload` |
| `arraybridge.array_operations.ArrayOperations`, `ArrayOperations.for_memory(m)` | `ArrayLibrary`, `m.library` (same primitive names) |
| `ArrayOperations.to_numpy(data, module)` and the other `module`-taking leaves | the library method without `module`: `to_numpy(data)`, `from_numpy(data, device)`, `scale_dtype`, `stack_arrays(values, device)`, `astype`, `logical_and`, `reshape`, `broadcast_to`, `ones_like`, `normalize_planes` |
| `arraybridge.framework_config`, `framework_ops`, `_FRAMEWORK_CONFIG`, `_FRAMEWORK_OPS` | nothing used them |
| `arraybridge.converters_registry` (`ConverterBase`, `get_converter`) | `library.convert_to(data, target_library, device)` |
| `arraybridge.dtype_scaling.SCALING_FUNCTIONS` | `library.scale_dtype(data, dtype)` |
| `arraybridge.gpu_cleanup.cleanup_all_gpu_frameworks(device_id)` | `ArrayLibrary.cleanup_all_loaded(device)` |
| `cleanup_<library>_gpu`, `MEMORY_TYPE_CLEANUP_REGISTRY` | `library.cleanup_loaded(device)` |
| `arraybridge.oom_recovery._execute_with_oom_recovery(f, name, retries, device_id)` | `library.run_with_oom_recovery(f, retries, device)` |
| `_is_oom_error(e, name)` | `library.is_oom_error(e)` |
| `arraybridge.utils._get_device_id`, `_set_device`, `_move_to_device` | `library.device_of`, `activate_device`, `move_to_device` |
| `arraybridge.decorators.ThreadGPUContext` | `CudaLibrary.thread_stream(device)`; per-call wrapping is `library.execution_wrapper(func, oom_recovery)` |
| `convert_memory(..., gpu_id=)`, `stack_slices(..., gpu_id=)`, `unstack_slices(..., gpu_id=)`, `process_slices(..., gpu_id=)` | the same functions with `device` |

### Typing

`typing.Any` is not used in the package. `arraybridge.arrays.Array` is the
nominal type of an array of any supported library; `library.array_class` is an
array of that library. Native array classes are registered with them when a
library first meets one.

`mypy` runs with `warn_return_any` and without `ignore_missing_imports`. Every
library the package imports is declared in the repository's `stubs/`
directory, which lists what arraybridge calls and comes before any installed
copy, so the result is the same on every machine. A library class states that
its `module` is that library; `NumpyArrayApi` names its module by the
`NumpyFunctions` Protocol. For a class written outside the package this means:
`NumpyArrayApi` and `MutableArrays` leave `array_module` abstract,
`AcceleratorDevices` asks for `_device_if_accepted(value)`, `CudaLibrary` for
`create_stream()`, and a `DLPackImport` library that cannot read a payload says
so in `_reads_dlpack(payload)` instead of returning `NotImplemented` from
`_import_dlpack`.

Annotations that now say what was already returned: a device or stream scope
is an `AbstractContextManager[object]` (CuPy's device and stream are their own
context managers), `from_dlpack` returns the array or `NotImplementedType`,
and `ArrayLibrary.array_module` is an `object` in the base class.

A value of the wrong kind is now refused with `TypeError` in four places:
`MemoryContractAttribute.read` when the stored value is neither a string nor
`None`, `TorchLibrary.stream_scope` for an object that is neither a
`torch.cuda.Stream` nor `None`, a dtype lookup in torch or TensorFlow whose
attribute is not a dtype of that library, and `DeviceMove.convert` called with
two different libraries.
