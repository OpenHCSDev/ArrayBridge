Decorator Examples
==================

Basic Decorator Usage
---------------------

.. code-block:: python

   from arraybridge import torch, numpy
   import numpy as np

   @torch(input_type='numpy', output_type='torch')
   def gpu_process(data):
       """Automatically converts NumPy input to PyTorch."""
       return data * 2

   # Use with NumPy array - automatic conversion
   np_data = np.array([1, 2, 3, 4, 5])
   result = gpu_process(np_data)
   # result is a PyTorch tensor

OOM Recovery
------------

.. code-block:: python

   from arraybridge import cupy
   import numpy as np

   @cupy(oom_recovery=True)
   def large_matrix_multiply(matrix):
       """Automatically handles GPU out-of-memory errors."""
       return matrix @ matrix.T

   # Will automatically clear GPU cache and retry if OOM occurs
   large_matrix = np.random.rand(10000, 10000)
   result = large_matrix_multiply(large_matrix)

Dtype Preservation
------------------

.. code-block:: python

   from arraybridge import torch
   from arraybridge.decorators import DtypeConversion
   import numpy as np

   @torch()
   def process_image(image, dtype_conversion=DtypeConversion.PRESERVE_INPUT):
       """Process image while preserving dtype."""
       # Normalize to 0-1 range
       return image / image.max()

   # uint8 input -> uint8 output (scaled appropriately)
   uint8_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
   result = process_image(uint8_image)

Slice-by-Slice Processing
--------------------------

.. code-block:: python

   from arraybridge import cupy
   import numpy as np

   @cupy()
   def denoise_3d(volume, slice_by_slice=True):
       """Process 3D volume slice-by-slice to avoid cross-slice artifacts."""
       # Apply denoising filter
       return volume  # Your denoising logic here

   # Process each 2D slice independently
   volume = np.random.rand(50, 512, 512)
   denoised = denoise_3d(volume, slice_by_slice=True)

Multiple Framework Support
---------------------------

.. code-block:: python

   from arraybridge import numpy, torch, cupy

   @numpy(output_type='numpy')
   def cpu_process(data):
       return data + 1

   @torch(output_type='torch')
   def pytorch_process(data):
       return data * 2

   @cupy(output_type='cupy')
   def cupy_process(data):
       return data ** 2

   # Each function enforces its output type
   import numpy as np
   data = np.array([1, 2, 3])

   np_result = cpu_process(data)
   torch_result = pytorch_process(data)
   cupy_result = cupy_process(data)

Custom Contracts
----------------

.. code-block:: python

   from arraybridge import numpy

   def positive_values(result):
       """Contract: all values must be positive."""
       return (result >= 0).all()

   @numpy(contract=positive_values)
   def sqrt_operation(data):
       """Take square root - output must be positive."""
       return data ** 0.5

   # Raises error if contract violated
   data = np.array([1, 4, 9, 16])
   result = sqrt_operation(data)  # OK

   negative_data = np.array([-1, 4, 9])
   # result = sqrt_operation(negative_data)  # Would raise ValueError
