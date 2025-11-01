User Guide
==========

This comprehensive guide covers all aspects of using arraybridge for unified array/tensor
operations across multiple frameworks.

Overview
--------

arraybridge provides a unified API for working with arrays and tensors from six different
frameworks: NumPy, CuPy, PyTorch, TensorFlow, JAX, and pyclesperanto. It handles automatic
memory type conversion, device management, and provides both imperative and declarative APIs.

Key Concepts
------------

Memory Types
~~~~~~~~~~~~

arraybridge recognizes six memory types:

- ``numpy``: NumPy arrays (CPU)
- ``cupy``: CuPy arrays (GPU)
- ``torch``: PyTorch tensors (CPU or GPU)
- ``tensorflow``: TensorFlow tensors (CPU or GPU)
- ``jax``: JAX arrays (CPU or GPU)
- ``pyclesperanto``: pyclesperanto GPU arrays

Each memory type is represented by the ``MemoryType`` enum for type safety.

Conversion Strategies
~~~~~~~~~~~~~~~~~~~~~

arraybridge uses different conversion strategies based on compatibility:

1. **DLPack (Zero-Copy)**: Used between compatible frameworks on GPU

   - CuPy ↔ PyTorch (GPU)
   - CuPy ↔ JAX (GPU)
   - PyTorch ↔ JAX (GPU)

2. **NumPy Bridge**: Used when DLPack isn't available

   - Converts through NumPy as an intermediate format
   - Involves memory copy but ensures compatibility

3. **Framework-Specific**: Special handling for pyclesperanto and TensorFlow

Core Functionality
------------------

Memory Type Detection
~~~~~~~~~~~~~~~~~~~~~

Automatically detect the framework and type of an array:

.. code-block:: python

   from arraybridge import detect_memory_type
   import numpy as np
   import torch

   # NumPy array
   np_data = np.array([1, 2, 3])
   print(detect_memory_type(np_data))  # 'numpy'

   # PyTorch tensor
   torch_data = torch.tensor([1, 2, 3])
   print(detect_memory_type(torch_data))  # 'torch'

   # Unknown types return None
   print(detect_memory_type([1, 2, 3]))  # None

Memory Conversion
~~~~~~~~~~~~~~~~~

Convert arrays between different frameworks:

.. code-block:: python

   from arraybridge import convert_memory
   import numpy as np

   # Create source data
   data = np.array([[1, 2], [3, 4]], dtype=np.float32)

   # Convert to PyTorch on GPU 0
   torch_data = convert_memory(
       data,
       source_type='numpy',
       target_type='torch',
       gpu_id=0
   )

   # Convert back to NumPy
   np_data = convert_memory(
       torch_data,
       source_type='torch',
       target_type='numpy'
   )

Automatic Type Detection
~~~~~~~~~~~~~~~~~~~~~~~~

You can omit the source type for automatic detection:

.. code-block:: python

   from arraybridge import convert_memory, detect_memory_type

   # Auto-detect source type
   def convert_to_torch(data):
       source_type = detect_memory_type(data)
       return convert_memory(data, source_type, 'torch', gpu_id=0)

Working with Decorators
-----------------------

arraybridge provides declarative decorators for automatic conversion at function boundaries.

Basic Decorator Usage
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import torch
   import numpy as np

   @torch(input_type='numpy', output_type='torch')
   def process_data(x):
       """Processes data as PyTorch tensor."""
       return x * 2 + 1

   # Pass NumPy array, get PyTorch tensor back
   result = process_data(np.array([1, 2, 3]))
   print(type(result))  # <class 'torch.Tensor'>

Memory Type Decorators
~~~~~~~~~~~~~~~~~~~~~~

Use framework-specific decorators:

.. code-block:: python

   from arraybridge import numpy, cupy, torch, tensorflow, jax

   @numpy(gpu_id=None)
   def numpy_operation(x):
       """Ensures input and output are NumPy arrays."""
       return x + 1

   @cupy(gpu_id=0)
   def cupy_operation(x):
       """Ensures input and output are CuPy arrays on GPU 0."""
       return x * 2

   @torch(gpu_id=0, output_type='numpy')
   def torch_to_numpy(x):
       """Processes as PyTorch, returns NumPy."""
       return x.pow(2)

Multi-Argument Functions
~~~~~~~~~~~~~~~~~~~~~~~~

Handle multiple array arguments:

.. code-block:: python

   from arraybridge import torch

   @torch(input_type='numpy', output_type='torch', gpu_id=0)
   def dot_product(a, b):
       """Compute dot product with auto-conversion."""
       return a @ b

   # Both arguments converted automatically
   result = dot_product(np.array([[1, 2]]), np.array([[3], [4]]))

Generic Memory Type Decorator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the generic decorator for more control:

.. code-block:: python

   from arraybridge.decorators import memory_types

   @memory_types(
       input_type='numpy',
       output_type='cupy',
       gpu_id=0,
       oom_recovery=True
   )
   def flexible_operation(data):
       """Convert from NumPy to CuPy with OOM recovery."""
       return data.sum(axis=0)

GPU Management
--------------

Device Selection
~~~~~~~~~~~~~~~~

Specify GPU devices for operations:

.. code-block:: python

   from arraybridge import convert_memory

   # Move to specific GPU
   gpu0_data = convert_memory(data, 'numpy', 'torch', gpu_id=0)
   gpu1_data = convert_memory(data, 'numpy', 'torch', gpu_id=1)

Multi-GPU Processing
~~~~~~~~~~~~~~~~~~~~

Distribute work across multiple GPUs:

.. code-block:: python

   from arraybridge import convert_memory
   import numpy as np

   def process_on_multi_gpu(data_list, num_gpus=4):
       results = []
       for i, data in enumerate(data_list):
           gpu_id = i % num_gpus
           # Convert to GPU
           gpu_data = convert_memory(
               data, 'numpy', 'torch', gpu_id=gpu_id
           )
           # Process
           result = gpu_data.sum()
           results.append(result)
       return results

   # Process batches across 4 GPUs
   batches = [np.random.rand(100, 100) for _ in range(16)]
   results = process_on_multi_gpu(batches, num_gpus=4)

CPU-GPU Data Movement
~~~~~~~~~~~~~~~~~~~~~

Efficiently move data between CPU and GPU:

.. code-block:: python

   from arraybridge import convert_memory

   # CPU to GPU
   cpu_data = np.random.rand(1000, 1000)
   gpu_data = convert_memory(cpu_data, 'numpy', 'cupy', gpu_id=0)

   # Process on GPU
   result_gpu = gpu_data @ gpu_data.T

   # GPU to CPU
   result_cpu = convert_memory(result_gpu, 'cupy', 'numpy')

Stack Utilities
---------------

Processing 2D Slices
~~~~~~~~~~~~~~~~~~~~

Stack multiple 2D slices into a 3D volume:

.. code-block:: python

   from arraybridge import stack_slices
   import numpy as np

   # Create 2D slices
   slices = [np.random.rand(512, 512) for _ in range(100)]

   # Stack into 3D volume
   volume = stack_slices(slices, target_type='numpy')
   print(volume.shape)  # (100, 512, 512)

Unstacking Volumes
~~~~~~~~~~~~~~~~~~

Split a 3D volume into 2D slices:

.. code-block:: python

   from arraybridge import unstack_slices

   # Create 3D volume
   volume = np.random.rand(100, 512, 512)

   # Unstack into list of slices
   slices = unstack_slices(volume, target_type='numpy')
   print(len(slices))  # 100
   print(slices[0].shape)  # (512, 512)

GPU Stack Processing
~~~~~~~~~~~~~~~~~~~~

Process stacks on GPU:

.. code-block:: python

   from arraybridge import stack_slices

   # Stack directly to GPU
   gpu_volume = stack_slices(
       slices,
       target_type='cupy',
       gpu_id=0
   )

   # Process entire volume on GPU
   result = gpu_volume.sum(axis=0)

Data Type Handling
------------------

Dtype Preservation
~~~~~~~~~~~~~~~~~~

arraybridge preserves data types during conversion:

.. code-block:: python

   import numpy as np
   from arraybridge import convert_memory

   # Float32 data
   data_f32 = np.array([1, 2, 3], dtype=np.float32)
   torch_data = convert_memory(data_f32, 'numpy', 'torch')
   print(torch_data.dtype)  # torch.float32

   # Int64 data
   data_i64 = np.array([1, 2, 3], dtype=np.int64)
   torch_data = convert_memory(data_i64, 'numpy', 'torch')
   print(torch_data.dtype)  # torch.int64

Dtype Conversion
~~~~~~~~~~~~~~~~

Some frameworks have different dtype systems:

.. code-block:: python

   # arraybridge handles dtype mapping
   # NumPy float64 → PyTorch float64
   # NumPy int32 → PyTorch int32
   # Handles framework-specific quirks automatically

Framework-Specific Features
---------------------------

PyTorch Integration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import torch
   import torch.nn as nn

   @torch(gpu_id=0)
   def neural_network_forward(x):
       """Process with PyTorch on GPU."""
       model = nn.Linear(10, 5).cuda()
       return model(x)

   # Pass NumPy data, get PyTorch result
   result = neural_network_forward(np.random.rand(32, 10))

CuPy Integration
~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import cupy
   import cupyx.scipy.ndimage as ndi

   @cupy(gpu_id=0)
   def image_filter(image):
       """Apply Gaussian filter on GPU."""
       return ndi.gaussian_filter(image, sigma=2)

JAX Integration
~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import jax
   import jax.numpy as jnp

   @jax(gpu_id=0)
   def jax_computation(x):
       """JIT-compiled JAX operation."""
       return jnp.fft.fft2(x)

Error Handling
--------------

Basic Error Handling
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import convert_memory, MemoryConversionError

   try:
       result = convert_memory(data, 'numpy', 'torch', gpu_id=0)
   except MemoryConversionError as e:
       print(f"Conversion failed: {e}")
       # Handle error or fallback

Framework Not Available
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import convert_memory

   try:
       # This will fail if PyTorch is not installed
       result = convert_memory(data, 'numpy', 'torch')
   except MemoryConversionError as e:
       if "not available" in str(e):
           print("PyTorch is not installed")
           # Fallback to NumPy processing

Out of Memory Errors
~~~~~~~~~~~~~~~~~~~~

See :doc:`advanced_topics` for OOM recovery strategies.

Best Practices
--------------

1. **Minimize Conversions**

   Convert once at boundaries, not in tight loops:

   .. code-block:: python

      # Good
      gpu_data = convert_memory(data, 'numpy', 'torch', gpu_id=0)
      for i in range(100):
          gpu_data = process(gpu_data)

      # Bad
      for i in range(100):
          gpu_data = convert_memory(data, 'numpy', 'torch', gpu_id=0)
          gpu_data = process(gpu_data)

2. **Use Decorators for Clean APIs**

   Decorators make function interfaces clear:

   .. code-block:: python

      @torch(input_type='numpy', output_type='numpy', gpu_id=0)
      def public_api(data):
          """Users can pass NumPy, get NumPy back."""
          # Process as PyTorch internally
          return data * 2

3. **Leverage Zero-Copy**

   Use compatible frameworks for zero-copy:

   .. code-block:: python

      # Zero-copy between CuPy and PyTorch on GPU
      cupy_data = get_cupy_array()
      torch_data = convert_memory(cupy_data, 'cupy', 'torch')  # Fast!

4. **Monitor Memory Usage**

   Use context managers for memory management:

   .. code-block:: python

      from arraybridge import convert_memory

      def process_large_data(data):
          gpu_data = convert_memory(data, 'numpy', 'cupy', gpu_id=0)
          result = gpu_data.sum()
          # GPU data will be garbage collected
          return float(result)

5. **Handle Optional Dependencies**

   Check for framework availability:

   .. code-block:: python

      from arraybridge import detect_memory_type, convert_memory

      def smart_convert(data, prefer_gpu=True):
          source = detect_memory_type(data)
          if prefer_gpu:
              try:
                  return convert_memory(data, source, 'cupy', gpu_id=0)
              except MemoryConversionError:
                  # CuPy not available, fallback to NumPy
                  return convert_memory(data, source, 'numpy')
          return data

Common Patterns
---------------

Pattern: Framework-Agnostic Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import detect_memory_type, convert_memory

   def universal_sum(data):
       """Sum operation that works with any framework."""
       mem_type = detect_memory_type(data)
       np_data = convert_memory(data, mem_type, 'numpy')
       result = np_data.sum()
       return convert_memory(result, 'numpy', mem_type)

Pattern: Batched GPU Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def process_batches_on_gpu(batches, batch_size=32):
       """Process batches efficiently on GPU."""
       results = []
       for i in range(0, len(batches), batch_size):
           batch = batches[i:i+batch_size]
           gpu_batch = convert_memory(batch, 'numpy', 'torch', gpu_id=0)
           result = model(gpu_batch)
           cpu_result = convert_memory(result, 'torch', 'numpy')
           results.append(cpu_result)
       return results

Pattern: Cross-Framework Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def pipeline(data):
       """Multi-stage pipeline using different frameworks."""
       # Stage 1: NumPy preprocessing
       preprocessed = np_preprocess(data)

       # Stage 2: PyTorch model inference
       torch_data = convert_memory(preprocessed, 'numpy', 'torch', gpu_id=0)
       features = pytorch_model(torch_data)

       # Stage 3: CuPy post-processing
       cupy_features = convert_memory(features, 'torch', 'cupy')
       result = cupy_postprocess(cupy_features)

       # Return as NumPy
       return convert_memory(result, 'cupy', 'numpy')

Next Steps
----------

- Learn about :doc:`converters` for detailed conversion mechanics
- Explore :doc:`decorators` for advanced decorator usage
- Read :doc:`gpu_features` for GPU-specific features
- Check :doc:`stack_utils` for volume processing
- Review :doc:`advanced_topics` for OOM recovery and optimization
- See :doc:`api_reference` for complete API documentation
