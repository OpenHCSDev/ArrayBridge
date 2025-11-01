API Reference
=============

Complete API documentation for arraybridge.

Core Functions
--------------

Memory Type Detection
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: arraybridge.detect_memory_type

Memory Conversion
~~~~~~~~~~~~~~~~~

.. autofunction:: arraybridge.convert_memory

Stack Utilities
~~~~~~~~~~~~~~~

.. autofunction:: arraybridge.stack_slices

.. autofunction:: arraybridge.unstack_slices

Decorators
----------

Generic Decorator
~~~~~~~~~~~~~~~~~

.. autofunction:: arraybridge.decorators.memory_types

Framework-Specific Decorators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: arraybridge.numpy

.. autofunction:: arraybridge.cupy

.. autofunction:: arraybridge.torch

.. autofunction:: arraybridge.tensorflow

.. autofunction:: arraybridge.jax

Types and Constants
-------------------

Memory Types
~~~~~~~~~~~~

.. autoclass:: arraybridge.MemoryType
   :members:
   :undoc-members:

Constants
~~~~~~~~~

.. autodata:: arraybridge.CPU_MEMORY_TYPES
   :annotation: = Tuple of CPU memory type strings

.. autodata:: arraybridge.GPU_MEMORY_TYPES
   :annotation: = Tuple of GPU memory type strings

.. autodata:: arraybridge.SUPPORTED_MEMORY_TYPES
   :annotation: = Tuple of all supported memory type strings

Exceptions
----------

.. autoexception:: arraybridge.MemoryConversionError
   :members:
   :show-inheritance:

Detailed API Documentation
--------------------------

arraybridge.detect_memory_type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detect the memory type of an array or tensor.

**Signature:**

.. code-block:: python

   def detect_memory_type(data: Any) -> Optional[str]:
       ...

**Parameters:**

- ``data`` (*Any*): Array or tensor to detect

**Returns:**

- *Optional[str]*: Memory type string ('numpy', 'cupy', 'torch', 'tensorflow', 'jax',
  'pyclesperanto') or None if type is unknown

**Examples:**

.. code-block:: python

   import numpy as np
   from arraybridge import detect_memory_type

   data = np.array([1, 2, 3])
   mem_type = detect_memory_type(data)
   print(mem_type)  # 'numpy'

   # Unknown types return None
   print(detect_memory_type([1, 2, 3]))  # None

**Notes:**

- Thread-safe
- Fast O(1) operation using isinstance checks
- Returns None for unsupported types rather than raising an exception

arraybridge.convert_memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert arrays between different memory types and devices.

**Signature:**

.. code-block:: python

   def convert_memory(
       data: Any,
       source_type: Union[str, MemoryType],
       target_type: Union[str, MemoryType],
       gpu_id: Optional[int] = 0
   ) -> Any:
       ...

**Parameters:**

- ``data`` (*Any*): Input array or tensor to convert
- ``source_type`` (*Union[str, MemoryType]*): Source memory type
- ``target_type`` (*Union[str, MemoryType]*): Target memory type
- ``gpu_id`` (*Optional[int]*): GPU device ID for GPU operations. Use 0 (default) for first GPU,
  or None for CPU-only operations

**Returns:**

- *Any*: Converted array/tensor in the target format

**Raises:**

- ``MemoryConversionError``: If conversion fails due to framework not available,
  incompatible types, or GPU errors

**Examples:**

.. code-block:: python

   from arraybridge import convert_memory
   import numpy as np

   # NumPy to PyTorch (GPU)
   np_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
   torch_data = convert_memory(np_data, 'numpy', 'torch', gpu_id=0)

   # PyTorch to CuPy (zero-copy on GPU)
   cupy_data = convert_memory(torch_data, 'torch', 'cupy', gpu_id=0)

   # Back to NumPy (CPU)
   result = convert_memory(cupy_data, 'cupy', 'numpy', gpu_id=None)

**Notes:**

- Uses DLPack for zero-copy when possible (GPU-GPU conversions)
- Falls back to NumPy bridge for CPU operations
- Preserves dtypes across conversions
- Thread-safe with thread-local GPU contexts

**Conversion Strategies:**

1. **DLPack (Zero-Copy)**: Used for compatible GPU frameworks
2. **NumPy Bridge**: Used when DLPack isn't available
3. **Framework-Specific**: Special handling for TensorFlow and pyclesperanto

arraybridge.stack_slices
~~~~~~~~~~~~~~~~~~~~~~~~~

Stack a list of 2D arrays into a 3D volume.

**Signature:**

.. code-block:: python

   def stack_slices(
       slices: List[Any],
       target_type: Union[str, MemoryType] = 'numpy',
       gpu_id: Optional[int] = 0
   ) -> Any:
       ...

**Parameters:**

- ``slices`` (*List[Any]*): List of 2D arrays or tensors
- ``target_type`` (*Union[str, MemoryType]*): Target memory type for output (default: 'numpy')
- ``gpu_id`` (*Optional[int]*): GPU device ID (default: 0 for GPU types, None for CPU)

**Returns:**

- *Any*: 3D array/tensor with shape (num_slices, height, width)

**Raises:**

- ``ValueError``: If slices have inconsistent shapes or are not 2D
- ``MemoryConversionError``: If conversion to target type fails

**Examples:**

.. code-block:: python

   from arraybridge import stack_slices
   import numpy as np

   # Create 2D slices
   slices = [np.random.rand(512, 512) for _ in range(100)]

   # Stack to NumPy volume
   volume = stack_slices(slices, target_type='numpy')
   print(volume.shape)  # (100, 512, 512)

   # Stack directly to GPU
   gpu_volume = stack_slices(slices, target_type='cupy', gpu_id=0)

**Notes:**

- All slices must have the same shape
- Slices can be from different frameworks (automatically converted)
- Efficient for large volumes

arraybridge.unstack_slices
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unstack a 3D volume into a list of 2D slices.

**Signature:**

.. code-block:: python

   def unstack_slices(
       volume: Any,
       target_type: Union[str, MemoryType] = 'numpy',
       gpu_id: Optional[int] = 0
   ) -> List[Any]:
       ...

**Parameters:**

- ``volume`` (*Any*): 3D array or tensor with shape (depth, height, width)
- ``target_type`` (*Union[str, MemoryType]*): Target memory type for output slices (default: 'numpy')
- ``gpu_id`` (*Optional[int]*): GPU device ID (default: 0 for GPU types, None for CPU)

**Returns:**

- *List[Any]*: List of 2D arrays/tensors

**Raises:**

- ``ValueError``: If input is not 3D
- ``MemoryConversionError``: If conversion to target type fails

**Examples:**

.. code-block:: python

   from arraybridge import unstack_slices
   import numpy as np

   # Create 3D volume
   volume = np.random.rand(100, 512, 512)

   # Unstack to list
   slices = unstack_slices(volume, target_type='numpy')
   print(len(slices))  # 100
   print(slices[0].shape)  # (512, 512)

**Notes:**

- Returns a Python list, not a stacked array
- Each slice is an independent array
- Useful for slice-by-slice processing

Decorator API
~~~~~~~~~~~~~

arraybridge.decorators.memory_types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generic decorator for memory type conversion.

**Signature:**

.. code-block:: python

   def memory_types(
       input_type: Optional[Union[str, MemoryType]] = None,
       output_type: Optional[Union[str, MemoryType]] = None,
       gpu_id: Optional[int] = 0,
       oom_recovery: bool = False,
       clear_cuda_cache: bool = False
   ) -> Callable:
       ...

**Parameters:**

- ``input_type`` (*Optional[Union[str, MemoryType]]*): Source memory type for inputs.
  If None, auto-detects input type
- ``output_type`` (*Optional[Union[str, MemoryType]]*): Target memory type for outputs.
  If None, keeps function's natural output type
- ``gpu_id`` (*Optional[int]*): GPU device ID (default: 0)
- ``oom_recovery`` (*bool*): Enable automatic OOM recovery (default: False)
- ``clear_cuda_cache`` (*bool*): Clear CUDA cache on OOM (default: False)

**Returns:**

- *Callable*: Decorated function

**Examples:**

.. code-block:: python

   from arraybridge.decorators import memory_types

   @memory_types(input_type='numpy', output_type='torch', gpu_id=0)
   def process(x):
       return x * 2

**Notes:**

- Converts all array arguments automatically
- Preserves non-array arguments
- Works with multiple arguments and keyword arguments

arraybridge.numpy
^^^^^^^^^^^^^^^^^

Decorator for NumPy conversion.

**Signature:**

.. code-block:: python

   def numpy(
       input_type: Optional[Union[str, MemoryType]] = None,
       output_type: Union[str, MemoryType] = 'numpy',
       gpu_id: Optional[int] = None,
       oom_recovery: bool = False,
       clear_cuda_cache: bool = False
   ) -> Callable:
       ...

**Parameters:**

Same as ``memory_types`` but defaults to NumPy output and CPU (``gpu_id=None``)

**Examples:**

.. code-block:: python

   from arraybridge import numpy

   @numpy()
   def process_as_numpy(x):
       return x + 1

arraybridge.cupy
^^^^^^^^^^^^^^^^

Decorator for CuPy conversion.

**Signature:**

.. code-block:: python

   def cupy(
       input_type: Optional[Union[str, MemoryType]] = None,
       output_type: Union[str, MemoryType] = 'cupy',
       gpu_id: int = 0,
       oom_recovery: bool = False,
       clear_cuda_cache: bool = False
   ) -> Callable:
       ...

**Parameters:**

Same as ``memory_types`` but defaults to CuPy output and GPU 0

**Examples:**

.. code-block:: python

   from arraybridge import cupy

   @cupy(gpu_id=0)
   def process_on_gpu(x):
       return x @ x.T

arraybridge.torch
^^^^^^^^^^^^^^^^^

Decorator for PyTorch conversion.

**Signature:**

.. code-block:: python

   def torch(
       input_type: Optional[Union[str, MemoryType]] = None,
       output_type: Union[str, MemoryType] = 'torch',
       gpu_id: int = 0,
       oom_recovery: bool = False,
       clear_cuda_cache: bool = False
   ) -> Callable:
       ...

**Parameters:**

Same as ``memory_types`` but defaults to PyTorch output and GPU 0

**Examples:**

.. code-block:: python

   from arraybridge import torch

   @torch(gpu_id=0, oom_recovery=True)
   def neural_network(x):
       return model(x)

arraybridge.tensorflow
^^^^^^^^^^^^^^^^^^^^^^

Decorator for TensorFlow conversion.

**Signature:**

.. code-block:: python

   def tensorflow(
       input_type: Optional[Union[str, MemoryType]] = None,
       output_type: Union[str, MemoryType] = 'tensorflow',
       gpu_id: int = 0,
       oom_recovery: bool = False,
       clear_cuda_cache: bool = False
   ) -> Callable:
       ...

**Parameters:**

Same as ``memory_types`` but defaults to TensorFlow output and GPU 0

**Examples:**

.. code-block:: python

   from arraybridge import tensorflow

   @tensorflow(gpu_id=0)
   def tf_operation(x):
       return tf.matmul(x, x, transpose_b=True)

arraybridge.jax
^^^^^^^^^^^^^^^

Decorator for JAX conversion.

**Signature:**

.. code-block:: python

   def jax(
       input_type: Optional[Union[str, MemoryType]] = None,
       output_type: Union[str, MemoryType] = 'jax',
       gpu_id: int = 0,
       oom_recovery: bool = False,
       clear_cuda_cache: bool = False
   ) -> Callable:
       ...

**Parameters:**

Same as ``memory_types`` but defaults to JAX output and GPU 0

**Examples:**

.. code-block:: python

   from arraybridge import jax

   @jax(gpu_id=0)
   def jax_fft(x):
       return jnp.fft.fft2(x)

Type Definitions
~~~~~~~~~~~~~~~~

MemoryType
^^^^^^^^^^

Enumeration of supported memory types.

**Values:**

- ``MemoryType.NUMPY``: NumPy arrays (CPU)
- ``MemoryType.CUPY``: CuPy arrays (GPU)
- ``MemoryType.TORCH``: PyTorch tensors (CPU or GPU)
- ``MemoryType.TENSORFLOW``: TensorFlow tensors (CPU or GPU)
- ``MemoryType.JAX``: JAX arrays (CPU or GPU)
- ``MemoryType.PYCLESPERANTO``: pyclesperanto GPU arrays

**Examples:**

.. code-block:: python

   from arraybridge import MemoryType, convert_memory

   # Using enum
   result = convert_memory(data, MemoryType.NUMPY, MemoryType.TORCH)

   # Using strings (equivalent)
   result = convert_memory(data, 'numpy', 'torch')

Exception Classes
~~~~~~~~~~~~~~~~~

MemoryConversionError
^^^^^^^^^^^^^^^^^^^^^

Exception raised when memory conversion fails.

**Inheritance:**

``Exception`` → ``MemoryConversionError``

**Attributes:**

- ``message`` (*str*): Error message describing the failure

**Common Causes:**

1. Framework not installed
2. GPU not available
3. Incompatible array types
4. Out of memory
5. Invalid memory type specification

**Examples:**

.. code-block:: python

   from arraybridge import convert_memory, MemoryConversionError

   try:
       result = convert_memory(data, 'numpy', 'torch', gpu_id=0)
   except MemoryConversionError as e:
       print(f"Conversion failed: {e}")

Usage Examples
--------------

Basic Conversion
~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import convert_memory, detect_memory_type
   import numpy as np

   # Create NumPy array
   data = np.array([[1, 2], [3, 4]], dtype=np.float32)

   # Detect type
   source_type = detect_memory_type(data)
   print(source_type)  # 'numpy'

   # Convert to PyTorch
   torch_data = convert_memory(data, source_type, 'torch', gpu_id=0)

   # Convert back
   np_data = convert_memory(torch_data, 'torch', 'numpy')

Decorator Usage
~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import torch

   @torch(input_type='numpy', output_type='numpy', gpu_id=0)
   def process_on_gpu(x):
       """Process with PyTorch internally, NumPy interface."""
       return x @ x.T

   # Users pass NumPy, get NumPy
   result = process_on_gpu(np.random.rand(100, 100))

Stack Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import stack_slices, unstack_slices

   # Create slices
   slices = [np.random.rand(512, 512) for _ in range(100)]

   # Stack
   volume = stack_slices(slices, target_type='numpy')

   # Process
   processed_volume = volume * 2

   # Unstack
   processed_slices = unstack_slices(processed_volume, target_type='numpy')

See Also
--------

- :doc:`user_guide` for comprehensive usage examples
- :doc:`converters` for conversion system details
- :doc:`decorators` for decorator patterns
- :doc:`gpu_features` for GPU-specific features
