# Changelog

## 0.5.1

### No library inherits an operation it cannot perform

`ArrayLibrary` no longer supplies NumPy-style implementations. In 0.5.0 every
library inherited `self.array_module.linspace(...)`, `.ones`, `.outer`,
`.floor`, `.max(axis=...)`, `data.astype(...)` and so on, whether or not its
module had those functions. They now live in parents a library declares:

| Parent | Supplies | Inherited by |
|---|---|---|
| `NumpyArrayApi` | every array operation, by NumPy's names on `array_module` | NumPy, CuPy, JAX |
| `ArrayMath` | declares `linspace`, `ones`, `outer`, `floor` (abstract) | `NumpyArrayApi`, torch, TensorFlow |
| `InPlaceAssignment` | `assign` by `array[index] = values` | `MutableArrays` (NumPy, CuPy), torch, pyclesperanto |

`MutableArrays` is now a `NumpyArrayApi` and `InPlaceAssignment` library.
`accepts`, `max`, `min`, `mean`, `sum`, `assign`, `_stack`, `_cast`,
`_reshape`, `_broadcast_to`, `_ones_like` and `_logical_and` are abstract in
`ArrayLibrary`. Torch and TensorFlow state the few they used to inherit; JAX
drops four overrides that only repeated the NumPy spelling.

What changes for callers: `PyclesperantoLibrary` has no `linspace`, `ones`,
`outer` or `floor`. In 0.5.0 `ArrayLibrary.named("pyclesperanto").linspace(...)`
existed and failed inside the call; it is now an `AttributeError` at the
attribute. Ask `isinstance(library, ArrayMath)`. Nothing else is renamed or
removed, and every other built-in library answers every operation as before.

What changes for a library class written outside the package: it defines the
operations above or inherits `NumpyArrayApi` (`ADDING_NEW_FRAMEWORKS.md` shows
both).

### A decorated call in main memory does not touch the accelerator

The wrapper a library decorator puts around a function (`execution_wrapper`)
reads the device from the call's arguments. In 0.5.0 it asked the library for
its selected device whenever the machine had an accelerator, which for torch
calls `torch.cuda.current_device()`: CUDA was initialised and a stream opened
for a function working on host tensors.

- Arguments held on accelerator N: the call runs on this thread's stream for
  device N, with out-of-memory recovery for that device. (0.5.0 used the
  stream of the selected device even when the data was on another one.)
- Arguments in main memory, or no array of a library that has main memory
  (torch, TensorFlow, JAX): the function is called as written. Nothing of the
  accelerator is asked, and there is no out-of-memory retry.
- A library without main memory (CuPy, pyclesperanto) called without one of
  its arrays runs on its selected device, as before.

### Typing

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

`MemoryType`'s metaclass overrides are annotated with `EnumMeta`'s signatures;
its `__call__` override is gone (`Enum` already asks `_missing_`).

### Conversion

The mechanism names a consumer can now see, in the order the routes are tried:

| Mechanism | Route class | Meaning |
|---|---|---|
| `"stays in place"` | `StaysInPlace` (new) | Same library, same device. Nothing is converted; the input object is returned. No record. |
| `"device move"` | `DeviceMove` | Same library, another device. |
| `"dlpack"` | `DLPackExchange` | The target wraps the source's buffer through DLPack. |
| `"shared host buffer"` | `SharedHostBuffer` (new) | Both arrays are in main memory and the result begins at the same address as the source: no bytes were copied. |
| `"host copy"` | `HostCopy` | The values were copied through a NumPy array in main memory. |

**The plan and the run agree that nothing is converted.**
`ConversionRoute.planned(library, device, library, device)` returns
`StaysInPlace`. In 0.5.0 it returned `DeviceMove`, while running that
conversion returned the input and recorded nothing, so a caller had to restate
the rule. `DeviceMove` now applies only when the device differs. The rule is
stated once, in `StaysInPlace.applies`; `convert_to`, `to_numpy`, `from_numpy`
and `move_to_device` all ask it. Every route has `writes_a_record`; it is
`False` for `StaysInPlace` only. A record means data was moved, copied or
wrapped.

**A host exchange that shares the buffer is no longer called a copy.** On the
host, torch and NumPy share one buffer in both directions, and JAX hands NumPy
its own buffer; 0.5.0 recorded these as `"host copy"`. They are now recorded
as `"shared host buffer"`. A library declares that it can share by inheriting
`NumpyBufferSharing` (NumPy, torch and JAX do; TensorFlow copies in both
directions and does not). Whether one array was shared is decided after the
exchange by comparing where the two arrays begin (`host_address_of`).

What was measured (Python 3.12, NumPy 2.5.4, torch 2.14.1 CPU, JAX 0.11.2,
TensorFlow 2.21.0 CPU; pointers, `np.shares_memory` and writing to one array
and reading the other):

| Exchange on the host | Result |
|---|---|
| NumPy to torch | shared, for every dtype tried, Fortran order, stepped and offset views, read-only arrays. A view with a negative step is copied (arraybridge makes it contiguous first). |
| torch to NumPy | shared |
| JAX to NumPy | shared (the NumPy array is read-only) |
| NumPy to JAX | copied, unless the buffer is C-contiguous, starts on a 64-byte boundary and keeps its dtype; then shared |
| NumPy to TensorFlow, TensorFlow to NumPy | copied |
| an array with no elements | nothing to share: `"host copy"` of 0 bytes |

**What `planned` returns when sharing depends on the array.** A plan sees
only declarations and devices, never an array. It returns the route that is
tried first: `SharedHostBuffer` for NumPy, torch and JAX with NumPy on the
host. The record names what was done. Where the exchange copied (NumPy to JAX
for an ordinary NumPy allocation, a reversed view to torch) the record's
mechanism is `"host copy"` and `declined_mechanisms` is
`("shared host buffer",)`, the way a declined DLPack exchange is already
reported. The exchange is done once either way.

What a consumer that keyed on the old answers sees:

- `planned(...) is DeviceMove` is no longer true for the same library and
  device; test `is StaysInPlace` (or `route.writes_a_record`).
- Rows that were `"host copy"` for NumPy and torch on the host, and for JAX
  to NumPy on the host, are now `"shared host buffer"`. `"host copy"` rows
  that follow a declined `SharedHostBuffer` carry it in `declined_mechanisms`,
  which `ConversionCounters` does not key on.
- `planned` for NumPy with torch or JAX on the host (either direction) is
  `SharedHostBuffer`. Between torch and JAX on the host it is still `"dlpack"`.
  Anything to or from an accelerator, pyclesperanto or TensorFlow is planned
  as before.
- The `nbytes` of a `from_numpy` record is the size of the array that was
  made (it was the size of the NumPy input; they differ when JAX narrows
  float64 to float32).

`ConversionRoute.taken(data, result, source, target)` is new: the route that
names what `convert` did, the route itself unless it says otherwise.
`ArrayLibrary.shares_numpy_buffer_on(device)` and
`ArrayLibrary.host_address_of(data)` are new, as is
`NumpyLibrary.registered()`. `HostCopy` and `SharedHostBuffer` share the new
parent `HostExchange`, which has no mechanism of its own.
`arraybridge.libraries` no longer imports `ConversionObserver`,
`ConversionRecord` and `HostCopy` (they were never exported from it; they are
in `arraybridge.conversion`).

**`ConversionObserver.current` belongs to the context that reads it.** It was
one attribute for the whole process, and `installed()` put back whatever it
had seen on entry: two threads that entered and left in the same order left
the second thread's observer installed for the rest of the process. The hook
is now a `contextvars.ContextVar`.

- `ConversionObserver.current` reads the observer of the running thread or
  asyncio task. Assigning to it sets the observer of that context only.
- `observer.install()` makes it the observer of this context and returns the
  one it replaced. `with observer.installed():` does so for a block.
- A new thread starts with the default observer, which ignores conversions.
  **An observer installed in the main thread no longer sees conversions made
  in worker threads.** To observe a thread you start, run it in a copy of your
  context: `Thread(target=contextvars.copy_context().run, args=(work,))`, or
  `executor.submit(contextvars.copy_context().run, work)`.
- An asyncio task starts with the observer of the code that created it, and
  what it installs stays in that task.
- `current` is read on the class (`ConversionObserver.current`). It is no
  longer an attribute of an observer instance.

### Not verified on a GPU

0.5.1 was tested with NumPy on Python 3.10, 3.12 and 3.13 and with the torch
CPU wheel on 3.12. The host exchanges in the Conversion section were also run
against the JAX and TensorFlow CPU builds named there. CuPy and pyclesperanto,
every accelerator path of torch, TensorFlow and JAX, and everything else
TensorFlow and JAX do, were exercised only through the fakes in
`tests/fakes.py`. `TESTING.md` has the commands for a GPU machine.

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

