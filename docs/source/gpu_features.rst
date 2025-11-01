GPU Features
============

arraybridge provides comprehensive GPU support including device management, multi-GPU
processing, and automatic memory optimization.

Overview
--------

GPU support in arraybridge includes:

- Multi-GPU device selection
- Automatic GPU memory management
- Cross-GPU data transfer
- GPU memory cleanup utilities
- OOM (Out-of-Memory) recovery
- CUDA stream management (framework-specific)

Supported GPU Frameworks
------------------------

arraybridge supports GPU operations with:

- **CuPy**: NVIDIA CUDA arrays
- **PyTorch**: CUDA tensors
- **JAX**: GPU arrays (CUDA and TPU)
- **TensorFlow**: GPU tensors
- **pyclesperanto**: OpenCL GPU arrays

Device Selection
----------------

Basic GPU Selection
~~~~~~~~~~~~~~~~~~~

Specify GPU device with the ``gpu_id`` parameter:

.. code-block:: python

   from arraybridge import convert_memory
   import numpy as np

   # Move to GPU 0
   gpu0_data = convert_memory(
       np_data,
       source_type='numpy',
       target_type='torch',
       gpu_id=0
   )

   # Move to GPU 1
   gpu1_data = convert_memory(
       np_data,
       source_type='numpy',
       target_type='torch',
       gpu_id=1
   )

Device Selection with Decorators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import torch

   @torch(gpu_id=0)
   def process_gpu0(x):
       """Process on GPU 0."""
       return x @ x.T

   @torch(gpu_id=1)
   def process_gpu1(x):
       """Process on GPU 1."""
       return x @ x.T

Querying Available GPUs
~~~~~~~~~~~~~~~~~~~~~~~~

Check available GPUs for each framework:

.. code-block:: python

   import torch
   import cupy as cp

   # PyTorch
   if torch.cuda.is_available():
       num_gpus = torch.cuda.device_count()
       print(f"PyTorch: {num_gpus} GPUs available")
       for i in range(num_gpus):
           print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

   # CuPy
   num_gpus = cp.cuda.runtime.getDeviceCount()
   print(f"CuPy: {num_gpus} GPUs available")

Multi-GPU Processing
--------------------

Data Parallel Processing
~~~~~~~~~~~~~~~~~~~~~~~~

Distribute batches across GPUs:

.. code-block:: python

   from arraybridge import convert_memory
   import numpy as np

   def multi_gpu_process(batches, num_gpus=4):
       """Process batches across multiple GPUs."""
       results = []

       for i, batch in enumerate(batches):
           # Round-robin GPU assignment
           gpu_id = i % num_gpus

           # Move to GPU
           gpu_batch = convert_memory(
               batch,
               source_type='numpy',
               target_type='torch',
               gpu_id=gpu_id
           )

           # Process on this GPU
           result = process_batch(gpu_batch)

           # Move back to CPU
           cpu_result = convert_memory(
               result,
               source_type='torch',
               target_type='numpy'
           )
           results.append(cpu_result)

       return results

Model Parallel Processing
~~~~~~~~~~~~~~~~~~~~~~~~~

Distribute model layers across GPUs:

.. code-block:: python

   from arraybridge import torch

   class MultiGPUModel:
       def __init__(self):
           # Layer 1 on GPU 0
           self.layer1 = self._create_layer(0)
           # Layer 2 on GPU 1
           self.layer2 = self._create_layer(1)

       @torch(gpu_id=0, output_type='torch')
       def _layer1_forward(self, x):
           return self.layer1(x)

       @torch(gpu_id=1, output_type='torch')
       def _layer2_forward(self, x):
           return self.layer2(x)

       def forward(self, x):
           # GPU 0
           x = self._layer1_forward(x)
           # Transfer GPU 0 → GPU 1
           x = convert_memory(x, 'torch', 'torch', gpu_id=1)
           # GPU 1
           x = self._layer2_forward(x)
           return x

Concurrent GPU Processing
~~~~~~~~~~~~~~~~~~~~~~~~~

Use Python threading or multiprocessing:

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor
   from arraybridge import convert_memory

   def process_on_gpu(data, gpu_id):
       """Process data on specific GPU."""
       gpu_data = convert_memory(data, 'numpy', 'torch', gpu_id=gpu_id)
       result = gpu_data @ gpu_data.T
       return convert_memory(result, 'torch', 'numpy')

   # Process on 4 GPUs concurrently
   with ThreadPoolExecutor(max_workers=4) as executor:
       futures = [
           executor.submit(process_on_gpu, batch, i % 4)
           for i, batch in enumerate(batches)
       ]
       results = [f.result() for f in futures]

Memory Management
-----------------

GPU Memory Monitoring
~~~~~~~~~~~~~~~~~~~~~

Monitor GPU memory usage:

.. code-block:: python

   import torch
   import cupy as cp

   def print_gpu_memory():
       """Print current GPU memory usage."""
       # PyTorch
       if torch.cuda.is_available():
           for i in range(torch.cuda.device_count()):
               allocated = torch.cuda.memory_allocated(i) / 1e9
               reserved = torch.cuda.memory_reserved(i) / 1e9
               print(f"GPU {i} (PyTorch): {allocated:.2f}GB allocated, "
                     f"{reserved:.2f}GB reserved")

       # CuPy
       mempool = cp.get_default_memory_pool()
       used = mempool.used_bytes() / 1e9
       total = mempool.total_bytes() / 1e9
       print(f"CuPy memory: {used:.2f}GB used, {total:.2f}GB total")

Manual Memory Cleanup
~~~~~~~~~~~~~~~~~~~~~

Clear GPU caches when needed:

.. code-block:: python

   import torch
   import cupy as cp
   import gc

   def clear_gpu_memory():
       """Clear GPU memory caches."""
       # Python garbage collection
       gc.collect()

       # PyTorch CUDA cache
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
           for i in range(torch.cuda.device_count()):
               with torch.cuda.device(i):
                   torch.cuda.empty_cache()

       # CuPy memory pool
       mempool = cp.get_default_memory_pool()
       mempool.free_all_blocks()
       pinned_mempool = cp.get_default_pinned_memory_pool()
       pinned_mempool.free_all_blocks()

Automatic Memory Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use context managers for automatic cleanup:

.. code-block:: python

   from contextlib import contextmanager
   from arraybridge import convert_memory

   @contextmanager
   def gpu_context(gpu_id=0):
       """Context manager for GPU operations with cleanup."""
       try:
           yield gpu_id
       finally:
           clear_gpu_memory()

   # Usage
   with gpu_context(gpu_id=0) as gpu_id:
       gpu_data = convert_memory(data, 'numpy', 'torch', gpu_id=gpu_id)
       result = process(gpu_data)
   # GPU memory automatically cleared

CPU-GPU Data Transfer
---------------------

Optimizing Transfers
~~~~~~~~~~~~~~~~~~~~

Minimize CPU-GPU transfers:

.. code-block:: python

   from arraybridge import convert_memory

   # Bad: Multiple small transfers
   for item in data_list:
       gpu_item = convert_memory(item, 'numpy', 'torch', gpu_id=0)
       results.append(process(gpu_item))

   # Good: Batch transfer
   batch = np.stack(data_list)
   gpu_batch = convert_memory(batch, 'numpy', 'torch', gpu_id=0)
   results = [process(gpu_batch[i]) for i in range(len(gpu_batch))]

Pinned Memory
~~~~~~~~~~~~~

Use pinned memory for faster transfers (PyTorch):

.. code-block:: python

   import torch
   import numpy as np

   # Create pinned NumPy array
   np_data = np.random.rand(1000, 1000).astype(np.float32)
   torch_pinned = torch.from_numpy(np_data).pin_memory()

   # Faster transfer to GPU
   gpu_data = torch_pinned.cuda(non_blocking=True)

Asynchronous Transfers
~~~~~~~~~~~~~~~~~~~~~~~

Use streams for async transfers (PyTorch):

.. code-block:: python

   import torch

   # Create stream for async operations
   stream = torch.cuda.Stream()

   with torch.cuda.stream(stream):
       # Async transfer
       gpu_data = cpu_data.cuda(non_blocking=True)
       # Async processing
       result = process(gpu_data)

   # Synchronize when needed
   stream.synchronize()

Cross-GPU Transfers
~~~~~~~~~~~~~~~~~~~

Move data between GPUs:

.. code-block:: python

   from arraybridge import convert_memory

   # Method 1: Direct GPU-GPU transfer
   gpu0_data = convert_memory(data, 'numpy', 'torch', gpu_id=0)
   gpu1_data = convert_memory(gpu0_data, 'torch', 'torch', gpu_id=1)

   # Method 2: Via CPU (slower but more compatible)
   cpu_data = convert_memory(gpu0_data, 'torch', 'numpy')
   gpu1_data = convert_memory(cpu_data, 'numpy', 'torch', gpu_id=1)

Framework-Specific GPU Features
-------------------------------

PyTorch GPU Features
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from arraybridge import torch as torch_decorator

   @torch_decorator(gpu_id=0)
   def pytorch_gpu_ops(x):
       """PyTorch-specific GPU operations."""
       # Check device
       print(f"Device: {x.device}")

       # CUDA events for timing
       start = torch.cuda.Event(enable_timing=True)
       end = torch.cuda.Event(enable_timing=True)

       start.record()
       result = x @ x.T
       end.record()

       torch.cuda.synchronize()
       print(f"Time: {start.elapsed_time(end)} ms")

       return result

CuPy GPU Features
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cupy as cp
   from arraybridge import cupy as cupy_decorator

   @cupy_decorator(gpu_id=0)
   def cupy_gpu_ops(x):
       """CuPy-specific GPU operations."""
       # Get device
       device = cp.cuda.Device()
       print(f"Device: {device.id}")

       # CUDA events for profiling
       start = cp.cuda.Event()
       end = cp.cuda.Event()

       start.record()
       result = x @ x.T
       end.record()
       end.synchronize()

       elapsed = cp.cuda.get_elapsed_time(start, end)
       print(f"Time: {elapsed} ms")

       return result

JAX GPU Features
~~~~~~~~~~~~~~~~

.. code-block:: python

   import jax
   from arraybridge import jax as jax_decorator

   @jax_decorator(gpu_id=0)
   def jax_gpu_ops(x):
       """JAX GPU operations with JIT."""
       # Get device
       print(f"Device: {x.device()}")

       # JIT compilation
       @jax.jit
       def compiled_op(x):
           return x @ x.T

       return compiled_op(x)

pyclesperanto GPU Features
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pyclesperanto_prototype as cle
   from arraybridge import convert_memory

   def cle_gpu_ops(image):
       """pyclesperanto GPU image processing."""
       # Convert to pyclesperanto
       gpu_image = convert_memory(
           image,
           source_type='numpy',
           target_type='pyclesperanto',
           gpu_id=0
       )

       # GPU-accelerated image operations
       filtered = cle.gaussian_blur(gpu_image, sigma_x=2, sigma_y=2)
       result = cle.pull(filtered)  # Back to NumPy

       return result

Performance Optimization
------------------------

GPU Memory Efficiency
~~~~~~~~~~~~~~~~~~~~~

Tips for efficient GPU memory usage:

1. **Use appropriate dtypes:**

   .. code-block:: python

      # float32 uses half the memory of float64
      data = np.array([1, 2, 3], dtype=np.float32)
      gpu_data = convert_memory(data, 'numpy', 'torch', gpu_id=0)

2. **Delete intermediate results:**

   .. code-block:: python

      intermediate = gpu_operation1(data)
      result = gpu_operation2(intermediate)
      del intermediate  # Free memory
      torch.cuda.empty_cache()

3. **Process in batches:**

   .. code-block:: python

      for batch in data_loader:
          gpu_batch = convert_memory(batch, 'numpy', 'torch', gpu_id=0)
          process(gpu_batch)
          del gpu_batch  # Free after each batch

Kernel Fusion
~~~~~~~~~~~~~

Combine operations to reduce kernel launches:

.. code-block:: python

   # Bad: Multiple kernel launches
   result = data + 1
   result = result * 2
   result = result ** 2

   # Good: Fused operations
   result = ((data + 1) * 2) ** 2

GPU Utilization
~~~~~~~~~~~~~~~

Monitor GPU utilization:

.. code-block:: bash

   # Monitor GPUs in terminal
   watch -n 1 nvidia-smi

Check utilization in code:

.. code-block:: python

   import pynvml

   pynvml.nvmlInit()
   handle = pynvml.nvmlDeviceGetHandleByIndex(0)
   util = pynvml.nvmlDeviceGetUtilizationRates(handle)
   print(f"GPU utilization: {util.gpu}%")
   print(f"Memory utilization: {util.memory}%")

Common Patterns
---------------

Pattern: GPU Auto-Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch

   def get_least_used_gpu():
       """Select GPU with most free memory."""
       if not torch.cuda.is_available():
           return None

       max_free = 0
       best_gpu = 0

       for i in range(torch.cuda.device_count()):
           free = torch.cuda.get_device_properties(i).total_memory - \
                  torch.cuda.memory_allocated(i)
           if free > max_free:
               max_free = free
               best_gpu = i

       return best_gpu

   # Use least loaded GPU
   gpu_id = get_least_used_gpu()
   gpu_data = convert_memory(data, 'numpy', 'torch', gpu_id=gpu_id)

Pattern: GPU Memory Pool
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class GPUMemoryPool:
       """Manage reusable GPU memory."""

       def __init__(self, gpu_id=0):
           self.gpu_id = gpu_id
           self.pool = {}

       def allocate(self, shape, dtype='float32'):
           """Get or create GPU array."""
           key = (shape, dtype)
           if key not in self.pool:
               data = torch.zeros(shape, dtype=dtype).cuda(self.gpu_id)
               self.pool[key] = data
           return self.pool[key]

       def clear(self):
           """Clear all pooled memory."""
           self.pool.clear()
           torch.cuda.empty_cache()

Troubleshooting
---------------

Common GPU Issues
~~~~~~~~~~~~~~~~~

**CUDA out of memory:**

.. code-block:: python

   try:
       result = process_on_gpu(large_data)
   except RuntimeError as e:
       if "out of memory" in str(e):
           # Clear cache and retry
           torch.cuda.empty_cache()
           result = process_on_gpu(smaller_batch)

**GPU not available:**

.. code-block:: python

   import torch

   if not torch.cuda.is_available():
       print("CUDA not available, using CPU")
       gpu_id = None
   else:
       gpu_id = 0

See :doc:`advanced_topics` for OOM recovery strategies.

Next Steps
----------

- Learn about :doc:`advanced_topics` for OOM recovery
- Check :doc:`converters` for conversion details
- Review :doc:`decorators` for GPU decorator usage
- See :doc:`api_reference` for complete API
