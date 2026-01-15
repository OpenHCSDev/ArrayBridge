Stack Utilities
===============

arraybridge provides utilities for stacking and unstacking 2D slices into 3D volumes,
with support for all frameworks and automatic memory type conversion.

Overview
--------

Stack utilities are useful for:

- Processing image stacks (e.g., microscopy, medical imaging)
- Batch processing of 2D data
- Creating 3D volumes from 2D slices
- Efficient slice-by-slice processing
- Memory-efficient volume operations

Core Functions
--------------

stack_slices
~~~~~~~~~~~~

Stack a list of 2D arrays into a 3D volume.

.. code-block:: python

   from arraybridge import stack_slices
   import numpy as np

   # Create list of 2D slices
   slices = [np.random.rand(512, 512) for _ in range(100)]

   # Stack into 3D volume (100, 512, 512)
   volume = stack_slices(slices, memory_type='numpy', gpu_id=0)

**Parameters:**

- ``slices``: List of 2D arrays
- ``memory_type``: Target memory type ('numpy', 'cupy', 'torch', 'tensorflow', 'jax', 'pyclesperanto')
- ``gpu_id``: GPU device ID (required, validated for GPU memory types)

**Returns:**

- 3D array/tensor with shape (num_slices, height, width)

**Raises:**

- ``ValueError``: If slices have inconsistent shapes or are not 2D

unstack_slices
~~~~~~~~~~~~~~

Unstack a 3D volume into a list of 2D slices.

.. code-block:: python

   from arraybridge import unstack_slices

   # Create 3D volume
   volume = np.random.rand(100, 512, 512)

   # Unstack into list of 2D slices
   slices = unstack_slices(volume, memory_type='numpy', gpu_id=0)

   print(len(slices))  # 100
   print(slices[0].shape)  # (512, 512)

**Parameters:**

- ``array``: 3D array/tensor with shape (depth, height, width)
- ``memory_type``: Target memory type for output slices
- ``gpu_id``: GPU device ID (required, validated for GPU memory types)
- ``validate_slices``: If True, validates that each extracted slice is 2D (default: True)

**Returns:**

- List of 2D arrays/tensors

**Raises:**

- ``ValueError``: If input is not 3D

Basic Usage
-----------

NumPy Stacking
~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import stack_slices, unstack_slices
   import numpy as np

   # Create slices
   slices = [np.random.rand(256, 256) for _ in range(50)]

   # Stack
   volume = stack_slices(slices, memory_type='numpy', gpu_id=0)
   print(volume.shape)  # (50, 256, 256)

   # Unstack
   recovered_slices = unstack_slices(volume, memory_type='numpy', gpu_id=0)
   print(len(recovered_slices))  # 50

GPU Stacking
~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import stack_slices
   import numpy as np

   # Create CPU slices
   cpu_slices = [np.random.rand(512, 512) for _ in range(100)]

   # Stack directly to GPU
   gpu_volume = stack_slices(
       cpu_slices,
       memory_type='cupy',
       gpu_id=0
   )

   # Process entire volume on GPU
   result = gpu_volume.sum(axis=0)

Cross-Framework Stacking
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import stack_slices
   import torch

   # PyTorch slices
   torch_slices = [torch.rand(128, 128) for _ in range(20)]

   # Stack to NumPy
   np_volume = stack_slices(torch_slices, memory_type='numpy', gpu_id=0)

   # Or stack to CuPy
   cupy_volume = stack_slices(torch_slices, memory_type='cupy', gpu_id=0)

Image Processing
----------------

Slice-by-Slice Processing
~~~~~~~~~~~~~~~~~~~~~~~~~

Process each slice individually:

.. code-block:: python

   from arraybridge import unstack_slices, stack_slices
   import numpy as np

   def process_slice(slice_2d):
       """Apply processing to single slice."""
       return slice_2d * 2 + 1

   # Load volume
   volume = np.random.rand(100, 512, 512)

   # Unstack, process, restack
   slices = unstack_slices(volume, memory_type='numpy', gpu_id=0)
   processed_slices = [process_slice(s) for s in slices]
   processed_volume = stack_slices(processed_slices, memory_type='numpy', gpu_id=0)

GPU-Accelerated Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import unstack_slices, stack_slices, convert_memory
   import cupyx.scipy.ndimage as ndi

   def gpu_filter_volume(volume):
       """Apply GPU filtering to each slice."""
       # Unstack to list
       slices = unstack_slices(volume, memory_type='numpy', gpu_id=0)

       # Process each slice on GPU
       filtered_slices = []
       for slice_2d in slices:
           # Move to GPU
           gpu_slice = convert_memory(
               slice_2d,
               source_type='numpy',
               target_type='cupy',
               gpu_id=0
           )
           # Apply filter
           filtered = ndi.gaussian_filter(gpu_slice, sigma=2.0)
           # Move back to CPU
           cpu_filtered = convert_memory(
               filtered,
               source_type='cupy',
               target_type='numpy',
               gpu_id=0
           )
           filtered_slices.append(cpu_filtered)

       # Restack
       return stack_slices(filtered_slices, memory_type='numpy', gpu_id=0)

Batch Processing
~~~~~~~~~~~~~~~~

Process slices in batches for efficiency:

.. code-block:: python

   from arraybridge import unstack_slices, stack_slices

   def batch_process_volume(volume, batch_size=10):
       """Process volume in batches."""
       slices = unstack_slices(volume, memory_type='numpy', gpu_id=0)
       processed_slices = []

       for i in range(0, len(slices), batch_size):
           batch = slices[i:i+batch_size]

           # Stack batch
           batch_volume = stack_slices(batch, memory_type='torch', gpu_id=0)

           # Process batch on GPU
           processed_batch = process_on_gpu(batch_volume)

           # Unstack batch
           batch_slices = unstack_slices(processed_batch, memory_type='numpy', gpu_id=0)
           processed_slices.extend(batch_slices)

       return stack_slices(processed_slices, memory_type='numpy', gpu_id=0)

Medical Imaging Applications
-----------------------------

CT/MRI Volume Processing
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import stack_slices, unstack_slices
   import numpy as np

   def load_dicom_slices(dicom_dir):
       """Load DICOM slices (simplified example)."""
       # Load DICOM files and extract pixel arrays
       slices = []
       for filename in sorted(os.listdir(dicom_dir)):
           # ... load DICOM slice ...
           slices.append(pixel_array)
       return slices

   def process_medical_volume(dicom_dir):
       """Process medical imaging volume."""
       # Load slices
       slices = load_dicom_slices(dicom_dir)

       # Stack into volume
       volume = stack_slices(slices, memory_type='numpy', gpu_id=0)

       # Apply processing (e.g., segmentation, registration)
       processed = medical_processing(volume)

       # Unstack for saving
       output_slices = unstack_slices(processed, memory_type='numpy', gpu_id=0)

       return output_slices

Microscopy Image Stacks
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import stack_slices
   from scipy import ndimage

   def process_microscopy_stack(image_files):
       """Process microscopy Z-stack."""
       # Load images
       slices = [load_image(f) for f in image_files]

       # Stack
       stack = stack_slices(slices, memory_type='numpy', gpu_id=0)

       # Maximum intensity projection
       mip = stack.max(axis=0)

       # Average intensity projection
       avg = stack.mean(axis=0)

       return mip, avg

Performance Optimization
------------------------

Memory-Efficient Stacking
~~~~~~~~~~~~~~~~~~~~~~~~~

For large volumes, consider processing in chunks:

.. code-block:: python

   def memory_efficient_stack(slices, chunk_size=10):
       """Stack large volumes in chunks."""
       chunks = []

       for i in range(0, len(slices), chunk_size):
           chunk_slices = slices[i:i+chunk_size]
           chunk = stack_slices(chunk_slices, memory_type='numpy', gpu_id=0)
           chunks.append(chunk)

       # Concatenate chunks
       return np.concatenate(chunks, axis=0)

Lazy Loading
~~~~~~~~~~~~

Load and process slices on-demand:

.. code-block:: python

   class LazyVolumeProcessor:
       """Lazy slice loading and processing."""

       def __init__(self, slice_files):
           self.slice_files = slice_files

       def __len__(self):
           return len(self.slice_files)

       def __getitem__(self, idx):
           """Load slice on-demand."""
           slice_data = load_image(self.slice_files[idx])
           return process_slice(slice_data)

       def to_volume(self):
           """Stack all slices."""
           slices = [self[i] for i in range(len(self))]
           return stack_slices(slices, memory_type='numpy', gpu_id=0)

Parallel Processing
~~~~~~~~~~~~~~~~~~~

Process slices in parallel:

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor
   from arraybridge import stack_slices

   def parallel_process_slices(slices, num_workers=4):
       """Process slices in parallel."""
       def process_one(slice_2d):
           return expensive_operation(slice_2d)

       with ThreadPoolExecutor(max_workers=num_workers) as executor:
           processed_slices = list(executor.map(process_one, slices))

       return stack_slices(processed_slices, memory_type='numpy', gpu_id=0)

Multi-GPU Stacking
~~~~~~~~~~~~~~~~~~

Distribute slices across GPUs:

.. code-block:: python

   from arraybridge import convert_memory, stack_slices

   def multi_gpu_stack_process(slices, num_gpus=4):
       """Process slices across multiple GPUs."""
       results = []

       for i, slice_2d in enumerate(slices):
           gpu_id = i % num_gpus

           # Move to GPU
           gpu_slice = convert_memory(
               slice_2d,
               source_type='numpy',
               target_type='torch',
               gpu_id=gpu_id
           )

           # Process
           processed = process_on_gpu(gpu_slice)

           # Move back
           cpu_slice = convert_memory(
               processed,
               source_type='torch',
               target_type='numpy',
               gpu_id=gpu_id
           )
           results.append(cpu_slice)

       return stack_slices(results, memory_type='numpy', gpu_id=0)

Advanced Usage
--------------

Custom Stack Dimensions
~~~~~~~~~~~~~~~~~~~~~~~

Stack along different axes:

.. code-block:: python

   from arraybridge import stack_slices
   import numpy as np

   slices = [np.random.rand(10, 20) for _ in range(5)]

   # Default: stack along axis 0 → (5, 10, 20)
   volume1 = stack_slices(slices, memory_type='numpy', gpu_id=0)

   # For other axes, use NumPy directly after stacking
   volume2 = np.moveaxis(volume1, 0, 2)  # (10, 20, 5)

Mixed Precision Stacking
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import stack_slices
   import numpy as np

   # Create float32 slices
   slices_f32 = [np.random.rand(256, 256).astype(np.float32)
                 for _ in range(50)]

   # Stack maintains dtype
   volume = stack_slices(slices_f32, memory_type='numpy', gpu_id=0)
   print(volume.dtype)  # float32

Weighted Stacking
~~~~~~~~~~~~~~~~~

Apply weights during stacking:

.. code-block:: python

   from arraybridge import stack_slices
   import numpy as np

   def weighted_stack(slices, weights):
       """Stack with per-slice weights."""
       # Stack normally
       volume = stack_slices(slices, memory_type='numpy', gpu_id=0)

       # Apply weights
       weights = np.array(weights).reshape(-1, 1, 1)
       weighted_volume = volume * weights

       return weighted_volume

   slices = [np.random.rand(100, 100) for _ in range(10)]
   weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
   result = weighted_stack(slices, weights)

Common Patterns
---------------

Pattern: Volume Iterator
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class VolumeIterator:
       """Iterate over volume slices."""

       def __init__(self, volume):
           self.volume = volume
           self.slices = unstack_slices(volume, memory_type='numpy', gpu_id=0)

       def __iter__(self):
           return iter(self.slices)

       def __len__(self):
           return len(self.slices)

       def __getitem__(self, idx):
           return self.slices[idx]

   # Usage
   volume = np.random.rand(100, 512, 512)
   for slice_2d in VolumeIterator(volume):
       process(slice_2d)

Pattern: Slice Cache
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from functools import lru_cache

   class SliceCache:
       """Cache processed slices."""

       def __init__(self, volume):
           self.volume = volume
           self.slices = unstack_slices(volume, memory_type='numpy', gpu_id=0)

       @lru_cache(maxsize=32)
       def get_processed_slice(self, idx):
           """Get processed slice with caching."""
           return process_expensive(self.slices[idx])

Pattern: Progressive Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def progressive_volume_processor(slice_generator):
       """Process volume progressively as slices arrive."""
       processed_slices = []

       for slice_2d in slice_generator:
           # Process each slice as it arrives
           processed = process_slice(slice_2d)
           processed_slices.append(processed)

           # Optionally save intermediate results
           if len(processed_slices) % 10 == 0:
               save_checkpoint(processed_slices)

       return stack_slices(processed_slices, memory_type='numpy', gpu_id=0)

Error Handling
--------------

Shape Validation
~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import stack_slices

   def safe_stack(slices):
       """Stack with shape validation."""
       if not slices:
           raise ValueError("Empty slice list")

       # Check all slices have same shape
       first_shape = slices[0].shape
       for i, s in enumerate(slices[1:], 1):
           if s.shape != first_shape:
               raise ValueError(
                   f"Slice {i} shape {s.shape} != first shape {first_shape}"
               )

       return stack_slices(slices, memory_type='numpy', gpu_id=0)

Memory Error Handling
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def robust_stack(slices, memory_type='numpy', gpu_id=0):
       """Stack with memory error handling."""
       try:
           return stack_slices(slices, memory_type=memory_type, gpu_id=gpu_id)
       except MemoryError:
           # Try processing in smaller chunks
           print("Memory error, processing in chunks...")
           return memory_efficient_stack(slices, chunk_size=5)

Testing
-------

Unit Tests
~~~~~~~~~~

.. code-block:: python

   import pytest
   from arraybridge import stack_slices, unstack_slices
   import numpy as np

   def test_stack_unstack_roundtrip():
       """Test stack/unstack preserves data."""
       slices = [np.random.rand(10, 10) for _ in range(5)]
       volume = stack_slices(slices, memory_type='numpy', gpu_id=0)
       recovered = unstack_slices(volume, memory_type='numpy', gpu_id=0)

       assert len(recovered) == len(slices)
       for orig, rec in zip(slices, recovered):
           np.testing.assert_array_almost_equal(orig, rec)

API Reference
-------------

See :doc:`api_reference` for complete function signatures.

Next Steps
----------

- Learn about :doc:`converters` for type conversion details
- Explore :doc:`gpu_features` for GPU-accelerated processing
- Check :doc:`advanced_topics` for optimization strategies
- Review :doc:`user_guide` for more examples
