# arraybridge

ArrayBridge converts arrays between NumPy, CuPy, PyTorch, TensorFlow, JAX and
pyclesperanto, and gives code one way to ask each library about devices,
memory and array operations.

Core dependencies are NumPy and metaclass-registry. The other libraries are
optional and are imported only when used.

Version 0.5.0 removed names without aliases. [CHANGELOG.md](CHANGELOG.md)
lists each removed name and what to call instead.

## Quick start

```python
import numpy as np

from arraybridge import HOST, convert_memory, detect_memory_type

value = np.arange(6).reshape(2, 3)
assert detect_memory_type(value) == "numpy"

result = convert_memory(value, source_type="numpy", target_type="numpy", device=HOST)
assert result is value
```

`convert_memory(data, source_type, target_type, device)` takes all four
arguments. `device` is a device of the target library: `HOST` for main memory,
`AcceleratorDevice(0)` for the library's first accelerator. A conversion to a
device the target library does not have fails; nothing is placed somewhere
else instead.

## Devices

A device is a value: `HOST` (main memory), `AcceleratorDevice(index)` (an
accelerator as its library numbers it) or `NamedDevice(name)` (a device a
library reports that is not one of its numbered accelerators, such as
pyclesperanto on a CPU OpenCL device). Indices belong to one library; the same
index in two libraries need not be the same hardware.

| Target library | Devices it accepts |
|---|---|
| NumPy | `HOST` |
| CuPy, pyclesperanto | `AcceleratorDevice(i)` |
| PyTorch, JAX, TensorFlow | `HOST` or `AcceleratorDevice(i)` |

Each library answers what it has on this machine:

```python
import numpy as np

from arraybridge import HOST, AcceleratorDevice, ArrayLibrary

numpy_library = ArrayLibrary.named("numpy")
assert numpy_library.host_devices() == (HOST,)
assert numpy_library.accelerator_devices() == ()
assert numpy_library.available_devices() == (HOST,)
assert numpy_library.device_of(np.zeros(3)) == HOST

assert AcceleratorDevice(0).label == "accelerator:0"
```

## Decorators

```python
from arraybridge import numpy

@numpy
def normalize(image):
    return image / max(float(image.max()), 1.0)

assert normalize.input_memory_type == "numpy"
```

The decorator named after a library sets `input_memory_type`,
`output_memory_type` and `execution_memory_type` on the function, adds the
`slice_by_slice` and `dtype_config` keyword arguments, and wraps the call with
what that library's devices need (per-thread CUDA streams for CuPy and
PyTorch, out-of-memory retries for libraries with accelerators). It does not
convert inputs or outputs between libraries and does not choose a device. The
caller converts with `convert_memory`.

## Stack utilities

```python
import numpy as np

from arraybridge import HOST, stack_slices, unstack_slices

slices = [np.zeros((8, 8)), np.ones((8, 8))]
stack = stack_slices(slices, memory_type="numpy", device=HOST)
restored = unstack_slices(stack, memory_type="numpy", device=HOST)
assert stack.shape == (2, 8, 8) and len(restored) == 2
```

`stack_slices` takes a non-empty list of 2D arrays. `unstack_slices` takes a
3D array. Both raise `ValueError` for a wrong shape or a device the target
library does not have.

## Which route a conversion takes

A conversion tries five mechanisms in order: `"stays in place"` (same library
and device: nothing is converted), `"device move"` (same library, other
device), `"dlpack"` (the target wraps the source's buffer),
`"shared host buffer"` (both arrays are in main memory and the result is the
source's own buffer) and `"host copy"` (the values are copied through NumPy).
`ConversionRoute.planned` tells which one will be tried first; the record of
a conversion names the one that was used. It reads only the library classes and the
device values, so it works without the libraries or the hardware:

```python
from arraybridge import HOST, AcceleratorDevice, ArrayLibrary, ConversionRoute

torch = ArrayLibrary.named("torch")
cupy = ArrayLibrary.named("cupy")
gpu = AcceleratorDevice(0)

assert ConversionRoute.planned(torch, gpu, cupy, gpu).mechanism == "dlpack"
assert ConversionRoute.planned(torch, HOST, cupy, gpu).mechanism == "host copy"
assert ConversionRoute.planned(torch, HOST, torch, gpu).mechanism == "device move"
assert ConversionRoute.planned(torch, gpu, torch, gpu).mechanism == "stays in place"
```

To see what conversions actually ran, install a `ConversionCounters` observer.
It keeps a count, bytes and seconds per source, target and mechanism. This
example needs TensorFlow:

```python
import numpy as np

from arraybridge import HOST, ConversionCounters, convert_memory

counters = ConversionCounters()
with counters.installed():
    convert_memory(np.zeros((4, 4), dtype=np.float32), "numpy", "tensorflow", HOST)

row = counters.rows()[0]
assert (row["source_library"], row["target_library"]) == ("numpy", "tensorflow")
assert (row["mechanism"], row["count"], row["nbytes"]) == ("host copy", 1, 64)
print(counters.format_table())
```

A conversion that stays in place is not counted. With no observer installed
nothing is timed. An observer belongs to the thread or asyncio task that
installed it; a thread you start is observed when you run it in a copy of your
context (`contextvars.copy_context().run`).

## Array operations in the array's own library

`ArrayLibrary.named("torch")`, `ArrayLibrary.of(array)` and
`MemoryType.TORCH.library` all return the library object. Its methods run in
that library, on the device of their input: `max`, `min`, `mean`, `sum`,
`astype`, `reshape`, `broadcast_to`, `ones_like`, `ones`, `prepend_axis`,
`linspace`, `assign`, `outer`, `floor`, `logical_and`, `stack_arrays`,
`normalize_planes`, `scale_dtype`, `to_numpy`, `from_numpy`.
`linspace`, `ones`, `outer` and `floor` exist only on libraries that can make
an array without a reference array (`ArrayMath`: every built-in library but
pyclesperanto).

```python
import numpy as np

from arraybridge import ArrayLibrary


def max_projection(stack):
    library = ArrayLibrary.of(stack)
    return library.prepend_axis(library.max(stack, 0))


assert max_projection(np.ones((4, 8, 8))).shape == (1, 8, 8)
```

`library.module` is the library's module, imported on first use. Each library
keeps its own numeric rules: PyTorch, JAX and TensorFlow take means in
float32, and pyclesperanto reduces only along the leading axis.

`reshape`, `broadcast_to` and `ones_like` do not download pixels to NumPy.
pyclesperanto cannot change shape or broadcast natively, so those calls raise
`NotImplementedError` unless the shape is unchanged; its `ones_like` supports
one to three dimensions and its float32 and integer dtypes, not bool.

## Adding a library

A library is one class. See
[ADDING_NEW_FRAMEWORKS.md](ADDING_NEW_FRAMEWORKS.md).

## Installation

```bash
pip install arraybridge
pip install "arraybridge[torch]"
pip install "arraybridge[cupy]"
```

Documentation: <https://arraybridge.readthedocs.io>
