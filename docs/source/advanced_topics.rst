Advanced Topics
===============

This document covers advanced features and optimization techniques in arraybridge,
including OOM recovery, performance tuning, and advanced patterns.

Out-of-Memory (OOM) Recovery
-----------------------------

arraybridge provides automatic OOM recovery for GPU operations, helping prevent crashes
from memory exhaustion.

Automatic OOM Recovery
~~~~~~~~~~~~~~~~~~~~~~

Enable OOM recovery with decorators:

.. code-block:: python

   from arraybridge import torch

   @torch(gpu_id=0, oom_recovery=True, clear_cuda_cache=True)
   def memory_intensive_operation(data):
       """Automatically handles OOM errors."""
       # Will retry with cache clearing if OOM occurs
       return data @ data.T @ data

   # Won't crash on OOM - will retry after clearing cache
   result = memory_intensive_operation(large_array)

How OOM Recovery Works
~~~~~~~~~~~~~~~~~~~~~~~

When enabled, arraybridge:

1. **Detects OOM**: Catches framework-specific OOM errors
2. **Clears Cache**: Runs garbage collection and clears GPU caches
3. **Retries**: Attempts the operation again
4. **Falls Back**: If retry fails, raises the original error

.. code-block:: python

   # Pseudo-code of OOM recovery process
   def with_oom_recovery(func):
       try:
           return func()
       except OutOfMemoryError:
           # Clear memory
           gc.collect()
           torch.cuda.empty_cache()
           # Retry once
           return func()

Manual OOM Handling
~~~~~~~~~~~~~~~~~~~

Implement custom OOM recovery:

.. code-block:: python

   import torch
   import gc

   def process_with_fallback(data):
       """Process with manual OOM handling."""
       try:
           # Try full batch
           return process_on_gpu(data)
       except RuntimeError as e:
           if "out of memory" in str(e):
               # Clear memory
               gc.collect()
               torch.cuda.empty_cache()

               # Try with smaller batch
               half_size = len(data) // 2
               result1 = process_on_gpu(data[:half_size])
               result2 = process_on_gpu(data[half_size:])
               return torch.cat([result1, result2])
           raise

Batch Size Reduction
~~~~~~~~~~~~~~~~~~~~

Automatically reduce batch size on OOM:

.. code-block:: python

   def adaptive_batch_process(data, initial_batch_size=32):
       """Adaptively reduce batch size on OOM."""
       batch_size = initial_batch_size
       results = []

       for i in range(0, len(data), batch_size):
           batch = data[i:i+batch_size]

           while True:
               try:
                   result = process_batch(batch)
                   results.append(result)
                   break
               except RuntimeError as e:
                   if "out of memory" in str(e):
                       # Reduce batch size
                       batch_size = max(1, batch_size // 2)
                       torch.cuda.empty_cache()
                       # Retry with smaller batch
                       batch = batch[:batch_size]
                   else:
                       raise

       return results

Memory Management Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Strategy 1: Gradient Checkpointing**

.. code-block:: python

   import torch.utils.checkpoint as checkpoint

   def memory_efficient_forward(x):
       """Use checkpointing to save memory."""
       # Trade compute for memory
       return checkpoint.checkpoint(expensive_layer, x)

**Strategy 2: Mixed Precision**

.. code-block:: python

   from torch.cuda.amp import autocast

   @torch(gpu_id=0)
   def mixed_precision_op(x):
       """Use FP16 for memory savings."""
       with autocast():
           return x @ x.T

**Strategy 3: CPU Offloading**

.. code-block:: python

   def cpu_offload_process(data):
       """Offload intermediate results to CPU."""
       # Process in stages
       stage1 = process_gpu_stage1(data).cpu()
       stage2 = process_gpu_stage2(stage1.cuda())
       stage3 = process_gpu_stage3(stage2)
       return stage3

Performance Optimization
------------------------

Conversion Performance
~~~~~~~~~~~~~~~~~~~~~~

**Use Zero-Copy When Possible:**

.. code-block:: python

   import cupy as cp
   from arraybridge import convert_memory

   # Zero-copy via DLPack (fast!)
   cupy_data = cp.random.rand(1000, 1000)
   torch_data = convert_memory(cupy_data, 'cupy', 'torch', gpu_id=0)

   # Verify zero-copy
   cupy_data[0, 0] = 999
   print(torch_data[0, 0])  # Also 999 - same memory!

**Batch Conversions:**

.. code-block:: python

   # Bad: Convert in loop
   for item in items:
       gpu_item = convert_memory(item, 'numpy', 'torch', gpu_id=0)
       process(gpu_item)

   # Good: Batch convert
   batch = np.stack(items)
   gpu_batch = convert_memory(batch, 'numpy', 'torch', gpu_id=0)
   for i in range(len(gpu_batch)):
       process(gpu_batch[i])

Memory Layout Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Contiguous Arrays:**

.. code-block:: python

   import numpy as np

   # Ensure contiguous for fast conversion
   non_contiguous = data[::2, ::2]  # Strided view
   contiguous = np.ascontiguousarray(non_contiguous)

   # Faster conversion
   gpu_data = convert_memory(contiguous, 'numpy', 'torch', gpu_id=0)

**Optimal Data Types:**

.. code-block:: python

   # Use float32 instead of float64 when possible
   data_f32 = np.array(data, dtype=np.float32)  # Half the memory
   gpu_data = convert_memory(data_f32, 'numpy', 'torch', gpu_id=0)

Caching and Memoization
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from functools import lru_cache
   from arraybridge import convert_memory

   class GPUCache:
       """Cache GPU conversions."""

       def __init__(self):
           self.cache = {}

       def get_or_convert(self, data_id, data, target_type, gpu_id=0):
           """Get cached GPU data or convert."""
           key = (data_id, target_type, gpu_id)
           if key not in self.cache:
               self.cache[key] = convert_memory(
                   data, 'numpy', target_type, gpu_id=gpu_id
               )
           return self.cache[key]

       def clear(self):
           """Clear cache."""
           self.cache.clear()

Profiling and Debugging
-----------------------

Timing Conversions
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   from arraybridge import convert_memory

   def time_conversion(data, source, target, gpu_id=0):
       """Time a conversion."""
       start = time.time()
       result = convert_memory(data, source, target, gpu_id=gpu_id)

       # Synchronize GPU if needed
       if target in ['torch', 'cupy']:
           if target == 'torch':
               import torch
               torch.cuda.synchronize()
           else:
               import cupy as cp
               cp.cuda.Stream.null.synchronize()

       elapsed = time.time() - start
       print(f"{source} → {target}: {elapsed*1000:.2f} ms")
       return result

Memory Profiling
~~~~~~~~~~~~~~~~

**PyTorch Memory Profiling:**

.. code-block:: python

   import torch

   def profile_memory(func):
       """Profile GPU memory usage."""
       torch.cuda.reset_peak_memory_stats()
       torch.cuda.empty_cache()

       result = func()

       peak_memory = torch.cuda.max_memory_allocated() / 1e9
       print(f"Peak GPU memory: {peak_memory:.2f} GB")

       return result

**CuPy Memory Profiling:**

.. code-block:: python

   import cupy as cp

   def profile_cupy_memory(func):
       """Profile CuPy memory usage."""
       mempool = cp.get_default_memory_pool()
       mempool.free_all_blocks()

       result = func()

       used = mempool.used_bytes() / 1e9
       total = mempool.total_bytes() / 1e9
       print(f"CuPy memory: {used:.2f} GB used, {total:.2f} GB total")

       return result

Debugging Conversions
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import detect_memory_type, convert_memory
   import logging

   logging.basicConfig(level=logging.DEBUG)

   def debug_convert(data, target_type, gpu_id=0):
       """Convert with debug logging."""
       source_type = detect_memory_type(data)
       logging.debug(f"Source type: {source_type}")
       logging.debug(f"Source shape: {data.shape}")
       logging.debug(f"Source dtype: {data.dtype}")

       result = convert_memory(data, source_type, target_type, gpu_id=gpu_id)

       logging.debug(f"Target type: {target_type}")
       logging.debug(f"Target shape: {result.shape}")
       logging.debug(f"Target dtype: {result.dtype}")

       return result

Advanced Patterns
-----------------

Pattern: Lazy Conversion
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class LazyArray:
       """Lazy conversion wrapper."""

       def __init__(self, data):
           self.data = data
           self.cached_conversions = {}

       def as_type(self, target_type, gpu_id=0):
           """Convert only when needed."""
           key = (target_type, gpu_id)
           if key not in self.cached_conversions:
               source_type = detect_memory_type(self.data)
               self.cached_conversions[key] = convert_memory(
                   self.data, source_type, target_type, gpu_id=gpu_id
               )
           return self.cached_conversions[key]

   # Usage
   lazy = LazyArray(np_data)
   torch_data = lazy.as_type('torch', gpu_id=0)  # Converts
   torch_data2 = lazy.as_type('torch', gpu_id=0)  # Cached

Pattern: Conversion Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import convert_memory

   class ConversionPipeline:
       """Chain multiple conversions and operations."""

       def __init__(self, data):
           self.data = data
           self.operations = []

       def convert_to(self, target_type, gpu_id=0):
           """Add conversion step."""
           def op(data):
               source_type = detect_memory_type(data)
               return convert_memory(data, source_type, target_type, gpu_id)
           self.operations.append(op)
           return self

       def apply(self, func):
           """Add processing step."""
           self.operations.append(func)
           return self

       def execute(self):
           """Execute pipeline."""
           result = self.data
           for op in self.operations:
               result = op(result)
           return result

   # Usage
   result = (ConversionPipeline(np_data)
             .convert_to('torch', gpu_id=0)
             .apply(lambda x: x * 2)
             .apply(lambda x: x + 1)
             .convert_to('numpy')
             .execute())

Pattern: Framework Fallback Chain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def try_frameworks(data, operation, frameworks=['cupy', 'torch', 'numpy']):
       """Try operation with different frameworks."""
       errors = []

       for framework in frameworks:
           try:
               # Convert to framework
               source_type = detect_memory_type(data)
               converted = convert_memory(
                   data, source_type, framework, gpu_id=0
               )

               # Try operation
               result = operation(converted)

               # Convert back
               return convert_memory(result, framework, source_type)

           except Exception as e:
               errors.append((framework, str(e)))
               continue

       # All frameworks failed
       raise RuntimeError(f"All frameworks failed: {errors}")

   # Usage
   result = try_frameworks(data, lambda x: x @ x.T)

Pattern: Multi-Backend Abstraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class MultiBackendArray:
       """Array that works with any backend."""

       def __init__(self, data, preferred_backend='torch'):
           self.data = data
           self.backend = preferred_backend
           self._cached = None

       def _ensure_backend(self):
           """Ensure data is in preferred backend."""
           if self._cached is None:
               source_type = detect_memory_type(self.data)
               self._cached = convert_memory(
                   self.data, source_type, self.backend, gpu_id=0
               )
           return self._cached

       def __matmul__(self, other):
           """Matrix multiplication."""
           self_backend = self._ensure_backend()
           other_backend = other._ensure_backend()
           result = self_backend @ other_backend
           return MultiBackendArray(result, self.backend)

       def to_numpy(self):
           """Convert to NumPy."""
           backend_data = self._ensure_backend()
           return convert_memory(backend_data, self.backend, 'numpy')

Thread and Process Safety
--------------------------

Thread-Local GPU Contexts
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import threading
   from arraybridge import convert_memory

   class ThreadLocalGPU:
       """Thread-local GPU device management."""

       def __init__(self):
           self.local = threading.local()

       def get_gpu_id(self):
           """Get GPU ID for current thread."""
           if not hasattr(self.local, 'gpu_id'):
               # Assign GPU based on thread ID
               self.local.gpu_id = threading.get_ident() % 4
           return self.local.gpu_id

       def convert(self, data, target_type):
           """Convert using thread-local GPU."""
           gpu_id = self.get_gpu_id()
           source_type = detect_memory_type(data)
           return convert_memory(data, source_type, target_type, gpu_id=gpu_id)

   # Usage
   gpu_manager = ThreadLocalGPU()

   def worker(data):
       # Each thread uses its own GPU
       gpu_data = gpu_manager.convert(data, 'torch')
       return process(gpu_data)

Multiprocessing with GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from multiprocessing import Pool
   from arraybridge import convert_memory

   def init_worker(gpu_id):
       """Initialize worker with specific GPU."""
       global worker_gpu_id
       worker_gpu_id = gpu_id

   def process_with_gpu(data):
       """Process using worker's GPU."""
       global worker_gpu_id
       gpu_data = convert_memory(
           data, 'numpy', 'torch', gpu_id=worker_gpu_id
       )
       result = process(gpu_data)
       return convert_memory(result, 'torch', 'numpy')

   # Create pool with 4 workers, each on different GPU
   with Pool(4, initializer=init_worker, initargs=(range(4),)) as pool:
       results = pool.map(process_with_gpu, batches)

Custom Memory Types
-------------------

While arraybridge doesn't support custom memory types out-of-the-box, you can wrap
conversions:

.. code-block:: python

   class CustomArrayWrapper:
       """Wrapper for custom array types."""

       def __init__(self, custom_array):
           self.custom_array = custom_array

       def to_numpy(self):
           """Convert custom array to NumPy."""
           # Implement custom conversion logic
           return np.array(self.custom_array.data)

       def convert_to(self, target_type):
           """Convert to arraybridge-supported type."""
           np_data = self.to_numpy()
           return convert_memory(np_data, 'numpy', target_type)

Integration with Other Libraries
---------------------------------

scikit-learn Integration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.decomposition import PCA
   from arraybridge import convert_memory

   def gpu_accelerated_pca(data, n_components=10):
       """PCA with GPU-accelerated computation."""
       # Convert to GPU for covariance computation
       gpu_data = convert_memory(data, 'numpy', 'torch', gpu_id=0)

       # Compute covariance on GPU
       centered = gpu_data - gpu_data.mean(dim=0)
       cov = (centered.T @ centered) / len(gpu_data)

       # Back to NumPy for sklearn
       np_cov = convert_memory(cov, 'torch', 'numpy')

       # Use sklearn for eigendecomposition
       pca = PCA(n_components=n_components)
       pca.fit_transform(np_data)

       return pca

Dask Integration
~~~~~~~~~~~~~~~~

.. code-block:: python

   import dask.array as da
   from arraybridge import convert_memory

   def dask_gpu_process(dask_array):
       """Process Dask array with GPU."""
       def process_chunk(chunk):
           # Convert chunk to GPU
           gpu_chunk = convert_memory(chunk, 'numpy', 'torch', gpu_id=0)
           # Process
           result = gpu_process(gpu_chunk)
           # Back to NumPy
           return convert_memory(result, 'torch', 'numpy')

       return dask_array.map_blocks(process_chunk)

Troubleshooting
---------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue: Slow Conversions**

Solution: Use zero-copy when possible, ensure contiguous arrays

**Issue: OOM Errors**

Solution: Enable OOM recovery, reduce batch sizes, use mixed precision

**Issue: Incorrect Results**

Solution: Check dtype preservation, verify device placement

**Issue: Memory Leaks**

Solution: Clear caches regularly, delete unused variables

See Also
--------

- :doc:`gpu_features` for GPU-specific features
- :doc:`converters` for conversion details
- :doc:`decorators` for decorator usage
- :doc:`api_reference` for complete API
