---
title: 'arraybridge: Unified Array Conversion Across Five GPU Frameworks and NumPy with OOM Recovery and Thread-Local Streams'
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

`arraybridge` provides unified array conversion between NumPy, CuPy, PyTorch, TensorFlow, JAX, and pyclesperanto. A microscopy image analysis pipeline might use CuPy for Gaussian filtering, PyTorch for neural network segmentation, and pyclesperanto for morphological operations—each with different array types, device APIs, and memory management. Without arraybridge, this requires learning six different conversion syntaxes:

```python
# Manual conversion nightmare
cupy_to_numpy = data.get()
torch_to_numpy = data.cpu().numpy()
jax_to_numpy = np.asarray(data)
tensorflow_to_numpy = data.numpy()
pyclesperanto_to_numpy = cle.pull(data)
```

With arraybridge:

```python
result = convert_memory(data, source_type='cupy', target_type='torch', gpu_id=0)
```

The library handles DLPack zero-copy transfers when available, falls back to NumPy bridging otherwise, manages per-thread CUDA streams for safe parallelization, detects and recovers from GPU out-of-memory errors across all frameworks, and preserves dtypes through conversions preventing silent precision loss.

# Statement of Need

GPU-accelerated scientific computing increasingly requires mixing frameworks. PyTorch dominates deep learning, CuPy provides NumPy-compatible GPU arrays, JAX offers automatic differentiation with XLA compilation, pyclesperanto provides GPU image processing primitives, and TensorFlow remains common in production deployments. A single pipeline may use three or four of these.

The problem is not just syntax—each framework has different:

- **Device management**: CuPy uses `cuda.Device(id)`, PyTorch uses `tensor.cuda(id)`, JAX uses `device_put(data, devices()[id])`
- **OOM exceptions**: CuPy raises `cuda.memory.OutOfMemoryError`, PyTorch raises `torch.cuda.OutOfMemoryError`, TensorFlow raises `errors.ResourceExhaustedError`
- **Cache clearing**: CuPy requires `get_default_memory_pool().free_all_blocks()`, PyTorch requires `cuda.empty_cache()`, TensorFlow has no explicit cache API
- **Stream management**: CuPy and PyTorch support CUDA streams; TensorFlow and JAX manage streams internally

Writing correct multi-framework code requires understanding all these differences. `arraybridge` consolidates this knowledge into a single declarative configuration, generating 36 conversion methods (6 frameworks × 6 target types) from ~450 lines of configuration rather than hand-written code for each path.

# State of the Field

**DLPack** [@dlpack] provides a zero-copy tensor sharing protocol adopted by all major frameworks. However, DLPack handles only the data transfer—users must still detect framework types, handle fallbacks when DLPack fails, manage device placement, and deal with framework-specific exceptions.

**The Python Array API Standard** [@arrayapi] defines common operations (`mean()`, `sum()`, `reshape()`) across frameworks but does not address conversion between frameworks or GPU memory management.

**Framework-specific utilities** (`torch.from_numpy()`, `cupy.asarray()`) handle only their own framework pairs. They provide no OOM recovery, no stream management, and no dtype preservation guarantees.

`arraybridge` differs in four key areas:

1. **Unified conversion API**: Single function for all 36 source/target combinations
2. **Automatic OOM recovery**: Detects framework-specific exception types and string patterns, clears caches, retries
3. **Thread-local CUDA streams**: Each thread gets its own stream, enabling true parallel GPU execution
4. **Dtype preservation**: Scales floating-point results to integer ranges when converting back to integer dtypes, preventing silent precision loss

# Software Design

The architecture is data-driven. All framework-specific behavior is defined in `_FRAMEWORK_CONFIG`:

```python
_FRAMEWORK_CONFIG = {
    MemoryType.CUPY: {
        "conversion_ops": {
            "to_numpy": "data.get()",
            "from_numpy": "({mod}.cuda.Device(gpu_id), {mod}.array(data))[1]",
            "from_dlpack": "{mod}.from_dlpack(data)",
        },
        "oom_exception_types": ["{mod}.cuda.memory.OutOfMemoryError"],
        "oom_clear_cache": "{mod}.get_default_memory_pool().free_all_blocks()",
        "stream_context": "{mod}.cuda.Stream()",
    },
    MemoryType.TORCH: {
        "conversion_ops": {
            "to_numpy": "data.cpu().numpy()",
            "from_numpy": "{mod}.from_numpy(data).cuda(gpu_id)",
            "from_dlpack": "{mod}.from_dlpack(data)",
        },
        "oom_exception_types": ["{mod}.cuda.OutOfMemoryError"],
        "oom_clear_cache": "{mod}.cuda.empty_cache()",
        "stream_context": "{mod}.cuda.Stream()",
    },
    # Similar entries for TENSORFLOW, JAX, PYCLESPERANTO, NUMPY
}
```

At import time, converter classes are generated dynamically via `AutoRegisterMeta` from the `metaclass-registry` library. Each converter implements `to_numpy()`, `from_numpy()`, `from_dlpack()`, and `to_X()` methods for all target frameworks. The metaclass auto-registers each converter by its `memory_type` attribute, eliminating manual registration.

**Thread-local GPU streams** are managed via `threading.local()`. The `@cupy` and `@torch` decorators automatically create per-thread CUDA streams:

```python
@torch(oom_recovery=True)
def segment_image(image):
    return model(image)  # Runs on thread-local stream
```

This enables true parallelization—multiple threads can execute GPU operations simultaneously without stream conflicts.

**OOM recovery** unifies detection across frameworks. The library checks both exception types and error string patterns (e.g., "out of memory", "resource_exhausted"), clears framework-specific caches, and retries:

```python
def _execute_with_oom_recovery(func, memory_type, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            if not _is_oom_error(e, memory_type) or attempt == max_retries:
                raise
            _clear_cache_for_memory_type(memory_type)
```

**Dtype preservation** handles the common case where GPU operations produce float32 output from integer input. The `SCALING_FUNCTIONS` registry applies framework-specific scaling and clamping to convert back to the original dtype without overflow.

# Research Impact Statement

`arraybridge` is a core component of OpenHCS, an open-source platform for high-content screening microscopy. In OpenHCS pipelines:

- **GPU-accelerated stitching** (`ashlar_compute_tile_positions_gpu`) uses CuPy for phase correlation
- **Flatfield correction** (`basic_flatfield_correction_cupy`) uses CuPy with OOM recovery and automatic fallback to CPU
- **Edge detection** (`sobel_2d_vectorized`) uses CuPy with dtype preservation to maintain uint16 microscopy data
- **Deep learning segmentation** integrates PyTorch models via the `@torch` decorator

The stack utilities (`stack_slices`, `unstack_slices`) enable efficient 3D volume processing where 2D slices are stacked to GPU, processed in parallel, and unstacked back to CPU. This pattern is used throughout OpenHCS for processing microscopy Z-stacks.

The thread-local stream management is critical for high-throughput screening where thousands of images must be processed per experiment. Multiple worker threads can process different images on the same GPU without coordination overhead.

# AI Usage Disclosure

Generative AI (Claude) assisted with code generation and documentation. All content was reviewed and tested. Core architectural decisions—data-driven configuration, metaclass registration, DLPack-first conversion with fallback, thread-local streams—were human-designed based on multi-framework interoperability requirements.

# Acknowledgements

This work was supported by [TODO: Add funding sources].

# References
