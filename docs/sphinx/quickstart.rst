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

Next Steps
----------

* Read the :doc:`api/index` for detailed API documentation
* Check out :doc:`examples/index` for more complex use cases
* Learn about :doc:`contributing` to the project
