Quick Start Guide
=================

This guide will help you get started with arraybridge quickly.

Installation
------------

Basic Installation
~~~~~~~~~~~~~~~~~~

Install arraybridge with just NumPy support:

.. code-block:: bash

   pip install arraybridge

With Framework Support
~~~~~~~~~~~~~~~~~~~~~~

Install with specific framework support:

.. code-block:: bash

   # PyTorch support
   pip install arraybridge[torch]

   # CuPy support (requires CUDA)
   pip install arraybridge[cupy]

   # TensorFlow support
   pip install arraybridge[tensorflow]

   # JAX support
   pip install arraybridge[jax]

   # All frameworks
   pip install arraybridge[all]

Basic Usage
-----------

Memory Type Detection
~~~~~~~~~~~~~~~~~~~~~

Automatically detect the memory type of arrays:

.. code-block:: python

   from arraybridge import detect_memory_type
   import numpy as np

   data = np.array([1, 2, 3])
   mem_type = detect_memory_type(data)
   print(mem_type)  # 'numpy'

Memory Conversion
~~~~~~~~~~~~~~~~~

Convert between different array/tensor types:

.. code-block:: python

   from arraybridge import convert_memory
   import numpy as np

   # Create NumPy array
   np_data = np.array([[1, 2], [3, 4]])

   # Convert to PyTorch (if installed)
   torch_data = convert_memory(
       np_data,
       source_type='numpy',
       target_type='torch',
       gpu_id=0
   )

Using Decorators
~~~~~~~~~~~~~~~~

Use declarative decorators for automatic conversion:

.. code-block:: python

   from arraybridge import torch, numpy
   import numpy as np

   @torch(input_type='numpy', output_type='torch')
   def process_on_gpu(data):
       """Automatically converts NumPy input to PyTorch."""
       return data * 2

   # Use with NumPy input - automatically converted
   result = process_on_gpu(np.array([1, 2, 3]))

Common Patterns
---------------

Pattern 1: Detect and Convert
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import detect_memory_type, convert_memory

   def process_data(data, target_type='torch'):
       # Detect source type
       source_type = detect_memory_type(data)

       # Convert if needed
       if source_type != target_type:
           data = convert_memory(data, source_type, target_type, gpu_id=0)

       # Process the data
       return data * 2

Pattern 2: Framework-Agnostic Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import detect_memory_type, convert_memory

   def universal_operation(data):
       """Works with any array type."""
       # Save original type
       original_type = detect_memory_type(data)

       # Convert to NumPy for processing
       np_data = convert_memory(data, original_type, 'numpy', gpu_id=0)

       # Process
       result = np_data + 1

       # Convert back to original type
       return convert_memory(result, 'numpy', original_type, gpu_id=0)

Pattern 3: OOM Recovery
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import cupy

   @cupy(oom_recovery=True)
   def memory_intensive_op(data):
       """Automatically handles out-of-memory errors."""
       return data @ data.T

Pattern 4: Stack Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Process 2D slices and stack them:

.. code-block:: python

   from arraybridge import stack_slices
   import numpy as np

   # Create list of 2D slices
   slices = [np.random.rand(10, 10) for _ in range(5)]

   # Stack into 3D array
   volume = stack_slices(slices, target_type='numpy')
   print(volume.shape)  # (5, 10, 10)

GPU Processing
--------------

Moving Data to GPU
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import convert_memory
   import numpy as np

   # CPU data
   cpu_data = np.random.rand(1000, 1000)

   # Move to GPU using CuPy
   gpu_data = convert_memory(
       cpu_data,
       source_type='numpy',
       target_type='cupy',
       gpu_id=0
   )

   # Process on GPU
   result = gpu_data @ gpu_data.T

   # Move back to CPU
   cpu_result = convert_memory(
       result,
       source_type='cupy',
       target_type='numpy'
   )

Multi-GPU Processing
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import convert_memory

   # Distribute work across GPUs
   for gpu_id in range(4):
       # Convert to specific GPU
       gpu_data = convert_memory(
           data,
           source_type='numpy',
           target_type='torch',
           gpu_id=gpu_id
       )
       # Process on this GPU
       results[gpu_id] = process(gpu_data)

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
       # Fallback to CPU processing
       result = process_on_cpu(data)

Automatic OOM Recovery
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge.decorators import torch

   @torch(oom_recovery=True, clear_cuda_cache=True)
   def gpu_operation(data):
       """Automatically recovers from OOM errors."""
       # Will retry with cache clearing if OOM occurs
       return data.pow(2).sum()

Performance Tips
----------------

1. **Use Zero-Copy When Possible**: arraybridge uses DLPack for zero-copy conversion
   between compatible frameworks (PyTorch, CuPy, JAX on GPU)

2. **Minimize Data Movement**: Keep data on GPU when doing multiple operations

3. **Use Decorators**: The decorator API handles conversion overhead efficiently

4. **Batch Processing**: Process data in batches to manage memory usage

.. code-block:: python

   from arraybridge import convert_memory

   # Good: Convert once, process multiple times
   gpu_data = convert_memory(data, 'numpy', 'torch', gpu_id=0)
   result1 = operation1(gpu_data)
   result2 = operation2(gpu_data)

   # Avoid: Converting repeatedly
   # result1 = operation1(convert_memory(data, ...))
   # result2 = operation2(convert_memory(data, ...))

Next Steps
----------

* Read the :doc:`installation` guide for setup details
* Explore the :doc:`user_guide` for comprehensive usage patterns
* Check the :doc:`api_reference` for detailed API documentation
* Review :doc:`examples/index` for more complex use cases
* Learn about :doc:`gpu_features` for GPU-specific features
* Understand :doc:`advanced_topics` for OOM recovery and optimization
