---
title: 'arraybridge: Zero-Copy GPU Array Conversion with OOM Recovery Across Six Frameworks'
tags:
  - Python
  - GPU computing
  - array programming
  - microscopy
  - scientific computing
authors:
  - name: Tristan Simas
    orcid: 0000-0002-6526-3149
    affiliation: 1
affiliations:
  - name: McGill University
    index: 1
date: 13 January 2026
bibliography: paper.bib
---

# Summary

`arraybridge` converts arrays between NumPy [@numpy], CuPy [@cupy], PyTorch [@pytorch], TensorFlow [@tensorflow], JAX [@jax], and pyclesperanto [@pyclesperanto], and wraps functions written for one of them with what that library's devices need. A microscopy pipeline might use CuPy for filtering, PyTorch for segmentation, and pyclesperanto for morphology. Each has its own array type, device API, and memory management, and its own way back to NumPy:

```python
cupy_to_numpy = data.get()
torch_to_numpy = data.cpu().numpy()
jax_to_numpy = np.asarray(data)
tensorflow_to_numpy = data.numpy()
pyclesperanto_to_numpy = cle.pull(data)
```

With arraybridge one call converts between any two registered libraries, and the caller can ask beforehand which mechanism will be tried and afterwards which one was used:

```python
from arraybridge import AcceleratorDevice, ArrayLibrary, ConversionCounters, ConversionRoute

cupy_library, torch_library = ArrayLibrary.named("cupy"), ArrayLibrary.named("torch")
gpu = AcceleratorDevice(0)

ConversionRoute.planned(cupy_library, gpu, torch_library, gpu).mechanism  # "dlpack"

counters = ConversionCounters()
with counters.installed():
    tensor = cupy_library.convert_to(cupy_array, torch_library, gpu)
print(counters.format_table())
```

A function is declared as belonging to a library with the decorator of that library's name:

```python
import cupy as cp
from arraybridge import cupy

@cupy
def horizontal_edges(image):
    return cp.abs(cp.gradient(image, axis=-1))
```

The decorator records the function's input and output library for a pipeline runtime to read, runs a call on a CUDA device inside a per-thread stream, retries after a GPU out-of-memory error, and converts the output back to the input dtype. It does not convert the arguments; conversion is the explicit call above.

# Statement of Need

GPU-accelerated scientific computing increasingly mixes frameworks: PyTorch for deep learning, CuPy for NumPy-compatible GPU arrays, JAX for automatic differentiation, pyclesperanto for GPU image processing. A single pipeline may use three or four of these.

The problem is not only syntax. Each framework has different:

- **Device management**: CuPy uses `cuda.Device(id)`, PyTorch uses `tensor.cuda(id)`, JAX uses `device_put(data, devices()[id])`
- **OOM exceptions**: CuPy raises `cuda.memory.OutOfMemoryError`, PyTorch raises `torch.cuda.OutOfMemoryError`, TensorFlow raises `errors.ResourceExhaustedError`
- **Cache clearing**: CuPy requires `get_default_memory_pool().free_all_blocks()`, PyTorch requires `cuda.empty_cache()`
- **Stream management**: CuPy and PyTorch expose CUDA streams; TensorFlow and JAX manage streams internally

Code that mixes frameworks restates these differences at each call site and cannot tell afterwards whether a conversion wrapped a buffer or copied it. `arraybridge` states each library's behaviour once, in one class per library, and derives the rest from those classes: the lookup by name or by array, the decorators, the `MemoryType` enumeration, and the conversion between every pair of libraries and devices.

| Feature | DLPack | Framework utils | arraybridge |
|---------|:------:|:---------------:|:-----------:|
| Zero-copy transfer | ✓ | — | ✓ |
| One call for every library pair | — | — | ✓ |
| Record of the mechanism used | — | — | ✓ |
| OOM detection and retry | — | — | ✓ |
| Per-thread CUDA streams | — | — | ✓ |
| Output dtype restored | — | — | ✓ |

# State of the Field

**DLPack** [@dlpack] provides a zero-copy tensor sharing protocol adopted by all major frameworks. It handles only the data transfer. Users must still detect framework types, handle a capsule the importer cannot read, manage device placement, and deal with framework-specific exceptions.

**The Python Array API Standard** [@arrayapi] defines common operations (`mean()`, `sum()`, `reshape()`) across frameworks but does not address conversion between frameworks or GPU memory management.

**Framework-specific utilities** (`torch.from_numpy()`, `cupy.asarray()`) handle only their own framework pairs. They provide no OOM retry, no stream management, and no report of whether data was shared or copied.

# Software Design

**One class per library.** `ArrayLibrary` is an abstract family registered by `AutoRegisterMeta` from `metaclass-registry` [@metaclassregistry]. A library is a subclass that states its name and inherits the roles it plays (abridged from the source):

```python
class CupyLibrary(CudaLibrary, DLPackImport, DLPackProtocolExport,
                  NvidiaWheelLibraries, MutableArrays):
    name = "cupy"
    oom_exception_paths = (
        ("cuda", "memory", "OutOfMemoryError"),
        ("cuda", "runtime", "CUDARuntimeError"),
    )

    def _to_host(self, data):
        return data.get()

    def _free_cached_memory(self):
        self.module.get_default_memory_pool().free_all_blocks()
```

The role parents carry the behaviour: `CudaLibrary` adds streams and cache release, `DLPackImport` and `DLPackProtocolExport` say which side of a DLPack exchange a library can take, and `NumpyBufferSharing` says that a host array can be the same buffer as a NumPy array. Adding a seventh library is one subclass; no other file is edited. Devices are values (`HOST`, `AcceleratorDevice(0)`), not strings or integers.

**Conversion routes.** How an array gets from one library and device to another is an ordered family of `ConversionRoute` subclasses: `"stays in place"`, `"device move"`, `"dlpack"`, `"shared host buffer"`, and `"host copy"`. Each route has one predicate, `applies`, that reads only the two library classes and the two devices. `ConversionRoute.planned` returns the first route whose predicate holds, and `ConversionRoute.run` tries the routes selected by the same predicate in the same order, so the plan and the conversion cannot disagree, and a plan can be made on a machine without the device. A plan cannot see the array. When a route cannot carry a particular array (a capsule the importer cannot read, a NumPy buffer the target has to copy), the next route runs, and the `ConversionRecord` names the mechanism that was used and lists the declined ones. `ConversionCounters` totals count, bytes, and seconds per source, target, and mechanism.

**Per-thread GPU streams.** For CuPy and PyTorch, a decorated call whose arguments are on CUDA device N runs inside that thread's stream for device N, created on first use and kept in `threading.local()`. A call on host arrays runs as written and does not initialise CUDA.

**OOM retry** (enabled by default for calls that run on an accelerator). Each library declares its out-of-memory exception types and message fragments (for example "out of memory", "resource_exhausted"). On a match the wrapper frees the library's cached memory on that device (CuPy and PyTorch; the other libraries declare no cache to free), collects garbage, and calls the function again. The retry runs on the same device. After two failed retries the error is raised to the caller; there is no fallback to the CPU.

**Output dtype.** Microscopy data is typically uint16, and many GPU operations return floating point. By default a decorated function's output is converted back to the dtype of its first argument. When a floating-point result goes to an integer dtype, each library's `scale_dtype` maps the result's minimum and maximum onto the full range of that dtype, clamps, and casts, so a uint16 input yields a uint16 output without overflow, with intensities stretched to 0 to 65535 rather than rounded in place.

# Research Impact Statement

`arraybridge` is used by OpenHCS, an open-source platform for high-content screening microscopy. In OpenHCS pipelines:

- **GPU-accelerated stitching** uses CuPy for phase correlation
- **Flatfield correction** uses CuPy with OOM retry
- **Edge detection** uses CuPy with the output returned in the input's uint16
- **Deep learning denoising** declares PyTorch functions with the `torch` decorator

# AI Usage Disclosure

Generative AI (Claude) assisted with code generation and documentation. All content was reviewed and tested. Core architectural decisions (one class per library, metaclass registration, DLPack-first conversion with a recorded fallback, per-thread streams) were human-designed based on multi-framework interoperability requirements.

# Acknowledgements

This work was supported in part by the Fournier lab at the Montreal Neurological Institute, McGill University.

# References
