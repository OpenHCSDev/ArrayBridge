Basic Conversion Examples
=========================

Simple NumPy to PyTorch
-----------------------

.. code-block:: python

   from arraybridge import convert_memory, detect_memory_type
   import numpy as np

   # Create NumPy array
   np_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

   # Convert to PyTorch
   torch_data = convert_memory(
       data=np_data,
       source_type='numpy',
       target_type='torch',
       gpu_id=0
   )

   print(f"Original type: {detect_memory_type(np_data)}")
   print(f"Converted type: {detect_memory_type(torch_data)}")

Preserving Data Types
----------------------

.. code-block:: python

   import numpy as np
   from arraybridge import convert_memory

   # Create uint8 array (common for images)
   image = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)

   # Convert to PyTorch - dtype is preserved
   torch_image = convert_memory(image, 'numpy', 'torch', gpu_id=0)

   print(f"NumPy dtype: {image.dtype}")
   print(f"PyTorch dtype: {torch_image.dtype}")  # Still uint8

Round-Trip Conversion
----------------------

.. code-block:: python

   import numpy as np
   from arraybridge import convert_memory

   # Original data
   original = np.array([1.5, 2.5, 3.5], dtype=np.float32)

   # NumPy -> PyTorch -> CuPy -> NumPy
   step1 = convert_memory(original, 'numpy', 'torch', gpu_id=0)
   step2 = convert_memory(step1, 'torch', 'cupy', gpu_id=0)
   final = convert_memory(step2, 'cupy', 'numpy', gpu_id=0)

   # Verify data integrity
   np.testing.assert_array_almost_equal(original, final)

3D Array Conversion
-------------------

.. code-block:: python

   import numpy as np
   from arraybridge import convert_memory

   # Create 3D volume (e.g., medical imaging)
   volume = np.random.rand(50, 512, 512).astype(np.float32)

   # Convert to GPU framework for processing
   gpu_volume = convert_memory(volume, 'numpy', 'cupy', gpu_id=0)

   # Process on GPU...

   # Convert back
   result = convert_memory(gpu_volume, 'cupy', 'numpy', gpu_id=0)

Automatic Type Detection
-------------------------

.. code-block:: python

   from arraybridge import detect_memory_type, convert_memory
   import numpy as np

   def convert_to_numpy(data):
       """Convert any supported type to NumPy."""
       source_type = detect_memory_type(data)
       if source_type == 'numpy':
           return data
       return convert_memory(data, source_type, 'numpy', gpu_id=0)

   # Works with any framework
   np_data = np.array([1, 2, 3])
   result = convert_to_numpy(np_data)
