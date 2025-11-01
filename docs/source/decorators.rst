Decorator System
================

arraybridge provides a powerful decorator system for automatic memory type conversion at
function boundaries. This enables clean, declarative APIs and reduces boilerplate code.

Overview
--------

The decorator system allows you to:

- Declare input and output memory types
- Automatically convert function arguments
- Handle multiple array arguments
- Specify GPU devices
- Enable OOM recovery
- Create framework-specific functions

Available Decorators
--------------------

Framework-Specific Decorators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

arraybridge provides decorators for each supported framework:

- ``@numpy()``: Convert to/from NumPy
- ``@cupy()``: Convert to/from CuPy
- ``@torch()``: Convert to/from PyTorch
- ``@tensorflow()``: Convert to/from TensorFlow
- ``@jax()``: Convert to/from JAX

Generic Decorator
~~~~~~~~~~~~~~~~~

- ``@memory_types()``: Generic decorator with full control

Basic Usage
-----------

Simple Decorator
~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import torch
   import numpy as np

   @torch()
   def process_with_pytorch(x):
       """Function processes data as PyTorch tensor."""
       return x * 2 + 1

   # Pass NumPy, get PyTorch back
   result = process_with_pytorch(np.array([1, 2, 3]))
   print(type(result))  # <class 'torch.Tensor'>

Specifying Input/Output Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import torch

   @torch(input_type='numpy', output_type='numpy')
   def public_api(x):
       """Public API: NumPy in, NumPy out (PyTorch internally)."""
       # x is automatically converted to PyTorch
       result = x.pow(2).sum()
       # result is automatically converted back to NumPy
       return result

   # Users only see NumPy
   np_result = public_api(np.array([1, 2, 3]))

GPU Device Selection
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import cupy

   @cupy(gpu_id=0)
   def process_on_gpu0(x):
       """Process on GPU 0 using CuPy."""
       return x @ x.T

   @cupy(gpu_id=1)
   def process_on_gpu1(x):
       """Process on GPU 1 using CuPy."""
       return x @ x.T

Decorator Parameters
--------------------

Common Parameters
~~~~~~~~~~~~~~~~~

All decorators support these parameters:

- ``input_type``: Source memory type for conversion (default: auto-detect)
- ``output_type``: Target memory type for output (default: decorator's framework)
- ``gpu_id``: GPU device ID (default: 0 for GPU frameworks, None for CPU)
- ``oom_recovery``: Enable OOM recovery (default: False)
- ``clear_cuda_cache``: Clear CUDA cache on OOM (default: False)

Example with All Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import torch

   @torch(
       input_type='numpy',
       output_type='cupy',
       gpu_id=0,
       oom_recovery=True,
       clear_cuda_cache=True
   )
   def complex_operation(x):
       """
       - Input: NumPy → converted to PyTorch
       - Process: PyTorch operations
       - Output: PyTorch → converted to CuPy
       - OOM recovery enabled
       """
       return x.pow(2).sum()

Multi-Argument Functions
-------------------------

Automatic Conversion of All Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import torch

   @torch(input_type='numpy', gpu_id=0)
   def matrix_multiply(a, b):
       """Both arguments converted to PyTorch."""
       return a @ b

   # Both NumPy arrays converted automatically
   result = matrix_multiply(
       np.array([[1, 2]]),
       np.array([[3], [4]])
   )

Mixed Argument Types
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import torch

   @torch(input_type='numpy')
   def weighted_sum(x, weight=2.0):
       """x is converted, weight is left as-is."""
       return x * weight

   result = weighted_sum(np.array([1, 2, 3]), weight=3.0)

Keyword Arguments
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import cupy

   @cupy(gpu_id=0)
   def process_with_kwargs(data, scale=1.0, offset=0.0):
       """Converts data, preserves scalar kwargs."""
       return (data * scale) + offset

   result = process_with_kwargs(
       np.random.rand(100),
       scale=2.0,
       offset=1.0
   )

Framework-Specific Decorators
------------------------------

@numpy
~~~~~~

Convert to NumPy arrays (CPU-only).

.. code-block:: python

   from arraybridge import numpy

   @numpy(input_type='torch', output_type='numpy')
   def process_as_numpy(x):
       """Process PyTorch input as NumPy."""
       return x + 1

   # PyTorch → NumPy → NumPy
   result = process_as_numpy(torch.tensor([1, 2, 3]))

@cupy
~~~~~

Convert to CuPy arrays (GPU-only).

.. code-block:: python

   from arraybridge import cupy
   import cupyx.scipy.ndimage as ndi

   @cupy(gpu_id=0, input_type='numpy')
   def gpu_filter(image):
       """GPU-accelerated image filtering."""
       return ndi.gaussian_filter(image, sigma=2.0)

   # Automatically uses GPU
   result = gpu_filter(np.random.rand(512, 512))

@torch
~~~~~~

Convert to PyTorch tensors (CPU or GPU).

.. code-block:: python

   from arraybridge import torch
   import torch.nn.functional as F

   @torch(gpu_id=0, input_type='numpy')
   def neural_network(x):
       """PyTorch neural network operations."""
       return F.relu(x) @ x.T

   result = neural_network(np.random.rand(100, 100))

@tensorflow
~~~~~~~~~~~

Convert to TensorFlow tensors (CPU or GPU).

.. code-block:: python

   from arraybridge import tensorflow
   import tensorflow as tf

   @tensorflow(gpu_id=0, input_type='numpy')
   def tf_operation(x):
       """TensorFlow operations."""
       return tf.matmul(x, x, transpose_b=True)

   result = tf_operation(np.random.rand(100, 100))

@jax
~~~~

Convert to JAX arrays (CPU or GPU).

.. code-block:: python

   from arraybridge import jax
   import jax.numpy as jnp

   @jax(gpu_id=0, input_type='numpy')
   def jax_fft(x):
       """JAX FFT operation."""
       return jnp.fft.fft2(x)

   result = jax_fft(np.random.rand(256, 256))

@memory_types (Generic)
~~~~~~~~~~~~~~~~~~~~~~~~

The generic decorator provides maximum flexibility.

.. code-block:: python

   from arraybridge.decorators import memory_types

   @memory_types(
       input_type='numpy',
       output_type='cupy',
       gpu_id=0
   )
   def flexible_function(x):
       """Can specify any input/output combination."""
       return x * 2

Advanced Features
-----------------

OOM Recovery
~~~~~~~~~~~~

Enable automatic out-of-memory recovery:

.. code-block:: python

   from arraybridge import torch

   @torch(gpu_id=0, oom_recovery=True, clear_cuda_cache=True)
   def memory_intensive(x):
       """Automatically handles OOM errors."""
       # Will retry with cache clearing if OOM occurs
       return x @ x.T @ x

   # Won't crash on OOM
   result = memory_intensive(torch.rand(10000, 10000))

See :doc:`advanced_topics` for more on OOM recovery.

Chaining Decorators
~~~~~~~~~~~~~~~~~~~~

Decorators can be chained for complex pipelines:

.. code-block:: python

   from arraybridge import torch, numpy
   import functools

   @numpy(output_type='numpy')
   def final_output(x):
       """Ensure final output is NumPy."""
       return x

   @torch(gpu_id=0, input_type='numpy')
   def stage2(x):
       """Second stage on GPU."""
       return x.pow(2)

   @numpy(input_type='torch')
   def stage1(x):
       """First stage as NumPy."""
       return x + 1

   # Use functools.compose or manual chaining
   def pipeline(x):
       return final_output(stage2(stage1(x)))

Return Value Handling
~~~~~~~~~~~~~~~~~~~~~

Decorators handle different return types:

.. code-block:: python

   from arraybridge import torch

   @torch(output_type='numpy')
   def returns_tuple(x):
       """Returns tuple of arrays."""
       return x * 2, x + 1

   # Both arrays in tuple are converted
   result1, result2 = returns_tuple(np.array([1, 2, 3]))

   @torch(output_type='numpy')
   def returns_dict(x):
       """Returns dict of arrays."""
       return {'doubled': x * 2, 'incremented': x + 1}

   # All arrays in dict are converted
   results = returns_dict(np.array([1, 2, 3]))

Class Methods
~~~~~~~~~~~~~

Decorators work with class methods:

.. code-block:: python

   from arraybridge import torch

   class ImageProcessor:
       @torch(gpu_id=0, input_type='numpy', output_type='numpy')
       def process(self, image):
           """Process image on GPU."""
           return image * 2

       @staticmethod
       @torch(gpu_id=0)
       def static_process(image):
           """Static method with decorator."""
           return image + 1

       @classmethod
       @torch(gpu_id=0)
       def class_process(cls, image):
           """Class method with decorator."""
           return image.mean()

   processor = ImageProcessor()
   result = processor.process(np.random.rand(512, 512))

Performance Considerations
--------------------------

Decorator Overhead
~~~~~~~~~~~~~~~~~~

Decorators add minimal overhead:

- Type detection: ~0.001 ms
- Conversion: depends on strategy (see :doc:`converters`)
- Function call: normal Python overhead

Optimization Tips
~~~~~~~~~~~~~~~~~

1. **Avoid Nested Decorated Calls:**

   .. code-block:: python

      # Bad: Repeated conversions
      @torch()
      def outer(x):
          return inner(x)

      @torch()
      def inner(x):
          return x * 2

      # Good: Single conversion
      @torch()
      def combined(x):
          return _inner(x)

      def _inner(x):
          """No decorator - already correct type."""
          return x * 2

2. **Use Specific Input Types:**

   .. code-block:: python

      # Slower: Auto-detection
      @torch()
      def process(x):
          return x * 2

      # Faster: Explicit input type
      @torch(input_type='numpy')
      def process(x):
          return x * 2

3. **Reuse GPU Data:**

   .. code-block:: python

      # Convert once, use multiple decorated functions
      gpu_data = convert_memory(data, 'numpy', 'torch', gpu_id=0)

      @torch()
      def process1(x):
          return x * 2

      @torch()
      def process2(x):
          return x + 1

      # Both use already-converted data
      result1 = process1(gpu_data)
      result2 = process2(gpu_data)

Common Patterns
---------------

Pattern: Clean Public API
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import torch

   class ModelWrapper:
       """Public API accepts NumPy, uses PyTorch internally."""

       @torch(input_type='numpy', output_type='numpy', gpu_id=0)
       def predict(self, x):
           """Prediction with automatic conversion."""
           # x is PyTorch tensor here
           return self.model(x)

       @torch(input_type='numpy', gpu_id=0)
       def train(self, x, y):
           """Training with automatic conversion."""
           loss = self.loss_fn(self.model(x), y)
           loss.backward()
           return float(loss)

Pattern: Multi-GPU Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import torch

   class MultiGPUProcessor:
       def __init__(self, num_gpus=4):
           self.num_gpus = num_gpus

       def process_batch(self, batch):
           """Distribute batch across GPUs."""
           results = []
           for i, data in enumerate(batch):
               gpu_id = i % self.num_gpus
               result = self._process_on_gpu(data, gpu_id)
               results.append(result)
           return results

       def _process_on_gpu(self, data, gpu_id):
           """Dynamic GPU selection."""
           @torch(gpu_id=gpu_id, input_type='numpy', output_type='numpy')
           def process(x):
               return x.sum()
           return process(data)

Pattern: Framework Abstraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import torch, cupy

   def create_processor(framework='torch'):
       """Factory function for framework-specific processors."""
       if framework == 'torch':
           @torch(gpu_id=0)
           def process(x):
               return x @ x.T
       elif framework == 'cupy':
           @cupy(gpu_id=0)
           def process(x):
               return x @ x.T
       return process

   # User chooses framework
   processor = create_processor('torch')
   result = processor(np.random.rand(100, 100))

Error Handling
--------------

Decorator Error Handling
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import torch, MemoryConversionError

   @torch(input_type='numpy', gpu_id=0)
   def safe_process(x):
       """Decorator handles conversion errors."""
       try:
           return x.pow(2)
       except Exception as e:
           print(f"Processing error: {e}")
           return x

   # Conversion errors raised before function execution
   try:
       result = safe_process(invalid_data)
   except MemoryConversionError as e:
       print(f"Conversion failed: {e}")

Fallback Logic
~~~~~~~~~~~~~~

.. code-block:: python

   def robust_process(data):
       """Try GPU, fallback to CPU."""
       try:
           @torch(gpu_id=0, input_type='numpy')
           def gpu_process(x):
               return x @ x.T
           return gpu_process(data)
       except Exception:
           @torch(gpu_id=None, input_type='numpy')
           def cpu_process(x):
               return x @ x.T
           return cpu_process(data)

Testing with Decorators
-----------------------

Unit Testing
~~~~~~~~~~~~

.. code-block:: python

   import pytest
   from arraybridge import torch

   @torch(input_type='numpy', output_type='numpy')
   def add_one(x):
       return x + 1

   def test_add_one():
       """Test decorated function."""
       input_data = np.array([1, 2, 3])
       result = add_one(input_data)
       expected = np.array([2, 3, 4])
       np.testing.assert_array_equal(result, expected)

Mocking
~~~~~~~

.. code-block:: python

   from unittest.mock import patch
   from arraybridge import torch

   @torch()
   def process(x):
       return x * 2

   def test_with_mock():
       """Test with mocked conversion."""
       with patch('arraybridge.convert_memory') as mock_convert:
           mock_convert.return_value = torch.tensor([2, 4, 6])
           result = process(np.array([1, 2, 3]))
           assert mock_convert.called

API Reference
-------------

See :doc:`api_reference` for complete decorator signatures.

Next Steps
----------

- Learn about :doc:`converters` for conversion details
- Explore :doc:`gpu_features` for device management
- Check :doc:`advanced_topics` for OOM recovery
- Review :doc:`examples/index` for more examples
