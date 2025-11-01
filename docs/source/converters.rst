Conversion System
=================

This document provides detailed information about arraybridge's conversion system, including
the algorithms, optimization strategies, and implementation details.

Overview
--------

arraybridge provides automatic conversion between six array/tensor frameworks using a combination
of zero-copy operations (via DLPack) and NumPy-based fallback conversions. The conversion
system is designed to be:

- **Fast**: Uses zero-copy when possible via DLPack
- **Reliable**: Falls back to NumPy bridge when needed
- **Type-preserving**: Maintains dtypes across conversions
- **Device-aware**: Handles CPU/GPU device management

Core Functions
--------------

detect_memory_type
~~~~~~~~~~~~~~~~~~

Detect the framework and memory type of an array or tensor.

.. code-block:: python

   from arraybridge import detect_memory_type

   # Returns: 'numpy', 'cupy', 'torch', 'tensorflow', 'jax', 'pyclesperanto', or None
   mem_type = detect_memory_type(array)

**Implementation Details:**

- Uses ``isinstance()`` checks for known types
- Returns ``None`` for unsupported types
- Handles both CPU and GPU arrays
- Thread-safe and performant

**Supported Types:**

- NumPy: ``numpy.ndarray``
- CuPy: ``cupy.ndarray``
- PyTorch: ``torch.Tensor``
- TensorFlow: ``tensorflow.Tensor``, ``tensorflow.EagerTensor``
- JAX: ``jax.Array``, ``jaxlib.xla_extension.DeviceArray``
- pyclesperanto: ``pyclesperanto_prototype._tier0._pycl.OCLArray``

convert_memory
~~~~~~~~~~~~~~

Convert arrays between different memory types and devices.

.. code-block:: python

   from arraybridge import convert_memory

   result = convert_memory(
       data,
       source_type='numpy',
       target_type='torch',
       gpu_id=0
   )

**Parameters:**

- ``data``: Input array/tensor
- ``source_type``: Source memory type (MemoryType or str)
- ``target_type``: Target memory type (MemoryType or str)
- ``gpu_id``: GPU device ID (default: 0), None for CPU

**Returns:**

- Converted array/tensor in the target format

**Raises:**

- ``MemoryConversionError``: If conversion fails

Conversion Strategies
---------------------

DLPack (Zero-Copy)
~~~~~~~~~~~~~~~~~~

DLPack enables zero-copy sharing of GPU memory between frameworks.

**Supported Paths:**

- CuPy ↔ PyTorch (GPU only)
- CuPy ↔ JAX (GPU only)
- PyTorch ↔ JAX (GPU only)

**Example:**

.. code-block:: python

   import cupy as cp
   from arraybridge import convert_memory

   # Create CuPy array on GPU
   cupy_data = cp.random.rand(1000, 1000)

   # Zero-copy conversion to PyTorch
   torch_data = convert_memory(cupy_data, 'cupy', 'torch', gpu_id=0)

   # Same memory location - zero copy!
   assert torch_data.data_ptr() == cupy_data.data.ptr

**When DLPack is Used:**

1. Source and target are both GPU-based
2. Both frameworks support DLPack
3. Arrays are contiguous in memory
4. No dtype conversion is needed

NumPy Bridge
~~~~~~~~~~~~

When DLPack is not available, arraybridge uses NumPy as an intermediate format.

**Conversion Path:**

1. Source → NumPy (using ``__array__()`` or ``.cpu().numpy()``)
2. NumPy → Target (using framework-specific constructors)

**Example:**

.. code-block:: python

   # PyTorch (GPU) → NumPy → TensorFlow (GPU)
   torch_data = torch.rand(100, 100).cuda()
   tf_data = convert_memory(torch_data, 'torch', 'tensorflow', gpu_id=0)

**When NumPy Bridge is Used:**

- Source or target is CPU-based
- DLPack not supported for the framework pair
- Dtype conversion is required
- Non-contiguous arrays

Framework-Specific Conversions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some frameworks require special handling:

**TensorFlow:**

.. code-block:: python

   # Uses tf.constant() or tf.identity()
   # Handles eager vs graph mode
   tf_data = convert_memory(np_data, 'numpy', 'tensorflow')

**pyclesperanto:**

.. code-block:: python

   # Uses push() and pull() operations
   # Always GPU-based
   cle_data = convert_memory(np_data, 'numpy', 'pyclesperanto', gpu_id=0)

**JAX:**

.. code-block:: python

   # Uses jax.device_put() for device placement
   # Respects JAX's device management
   jax_data = convert_memory(np_data, 'numpy', 'jax', gpu_id=0)

Device Management
-----------------

GPU Device Selection
~~~~~~~~~~~~~~~~~~~~

arraybridge handles device selection for GPU operations:

.. code-block:: python

   # Move to GPU 0
   gpu0_data = convert_memory(data, 'numpy', 'cupy', gpu_id=0)

   # Move to GPU 1
   gpu1_data = convert_memory(data, 'numpy', 'cupy', gpu_id=1)

**Framework-Specific Behavior:**

- **CuPy**: Uses ``cp.cuda.Device(gpu_id)``
- **PyTorch**: Uses ``torch.cuda.device(gpu_id)``
- **JAX**: Uses ``jax.devices('gpu')[gpu_id]``
- **TensorFlow**: Uses ``tf.device(f'/GPU:{gpu_id}')``

CPU Operations
~~~~~~~~~~~~~~

When ``gpu_id=None``, operations are CPU-only:

.. code-block:: python

   # CPU-only conversion
   torch_cpu = convert_memory(np_data, 'numpy', 'torch', gpu_id=None)
   print(torch_cpu.device)  # cpu

Cross-Device Transfers
~~~~~~~~~~~~~~~~~~~~~~

Moving data between devices:

.. code-block:: python

   # GPU 0 to GPU 1
   gpu0_data = convert_memory(data, 'numpy', 'torch', gpu_id=0)
   gpu1_data = convert_memory(gpu0_data, 'torch', 'torch', gpu_id=1)

   # GPU to CPU
   cpu_data = convert_memory(gpu0_data, 'torch', 'numpy', gpu_id=None)

Dtype Handling
--------------

Dtype Preservation
~~~~~~~~~~~~~~~~~~

arraybridge preserves dtypes during conversion:

.. code-block:: python

   import numpy as np
   from arraybridge import convert_memory

   # Float32 preservation
   f32_data = np.array([1, 2, 3], dtype=np.float32)
   torch_data = convert_memory(f32_data, 'numpy', 'torch')
   assert torch_data.dtype == torch.float32

   # Int64 preservation
   i64_data = np.array([1, 2, 3], dtype=np.int64)
   torch_data = convert_memory(i64_data, 'numpy', 'torch')
   assert torch_data.dtype == torch.int64

Dtype Mapping
~~~~~~~~~~~~~

Framework-specific dtype mappings:

**NumPy → PyTorch:**

- ``float32`` → ``torch.float32``
- ``float64`` → ``torch.float64``
- ``int32`` → ``torch.int32``
- ``int64`` → ``torch.int64``

**NumPy → TensorFlow:**

- ``float32`` → ``tf.float32``
- ``float64`` → ``tf.float64``
- ``int32`` → ``tf.int32``
- ``int64`` → ``tf.int64``

**NumPy → CuPy:**

- Exact dtype mapping (CuPy follows NumPy)

Conversion Performance
----------------------

Benchmarks
~~~~~~~~~~

Relative performance of different conversion strategies:

1. **DLPack (Zero-Copy)**: ~0.001 ms (pointer sharing)
2. **NumPy Bridge (Small)**: ~0.1-1 ms (< 1 MB)
3. **NumPy Bridge (Large)**: ~10-100 ms (> 100 MB)
4. **CPU-GPU Transfer**: 10-1000 ms (depends on size and PCIe)

Optimization Tips
~~~~~~~~~~~~~~~~~

1. **Use Zero-Copy When Possible:**

   .. code-block:: python

      # Fast: DLPack zero-copy
      cupy_data = cp.random.rand(1000, 1000)
      torch_data = convert_memory(cupy_data, 'cupy', 'torch')

2. **Avoid Unnecessary Conversions:**

   .. code-block:: python

      # Bad: Convert in loop
      for i in range(100):
          torch_data = convert_memory(np_data, 'numpy', 'torch')
          result = process(torch_data)

      # Good: Convert once
      torch_data = convert_memory(np_data, 'numpy', 'torch')
      for i in range(100):
          result = process(torch_data)

3. **Batch CPU-GPU Transfers:**

   .. code-block:: python

      # Transfer multiple arrays together
      batch = np.stack([arr1, arr2, arr3])
      gpu_batch = convert_memory(batch, 'numpy', 'torch', gpu_id=0)

4. **Use Pinned Memory for Large Transfers:**

   .. code-block:: python

      # PyTorch pinned memory for faster CPU-GPU transfer
      import torch
      pinned_data = torch.from_numpy(np_data).pin_memory()
      gpu_data = pinned_data.cuda()

Error Handling
--------------

Common Errors
~~~~~~~~~~~~~

**Framework Not Available:**

.. code-block:: python

   try:
       result = convert_memory(data, 'numpy', 'torch')
   except MemoryConversionError as e:
       if "not available" in str(e):
           print("PyTorch is not installed")

**Invalid Memory Type:**

.. code-block:: python

   try:
       result = convert_memory(data, 'invalid', 'numpy')
   except MemoryConversionError as e:
       print(f"Invalid memory type: {e}")

**GPU Not Available:**

.. code-block:: python

   try:
       result = convert_memory(data, 'numpy', 'cupy', gpu_id=0)
   except MemoryConversionError as e:
       if "CUDA" in str(e):
           print("GPU not available, fallback to CPU")
           result = convert_memory(data, 'numpy', 'numpy')

Recovery Strategies
~~~~~~~~~~~~~~~~~~~

Implement fallback logic:

.. code-block:: python

   def safe_convert(data, target_type, gpu_id=0):
       """Convert with automatic fallback."""
       source_type = detect_memory_type(data)

       try:
           # Try GPU conversion
           return convert_memory(data, source_type, target_type, gpu_id=gpu_id)
       except MemoryConversionError:
           # Fallback to CPU
           print("GPU conversion failed, using CPU")
           return convert_memory(data, source_type, target_type, gpu_id=None)

Advanced Topics
---------------

Custom Memory Types
~~~~~~~~~~~~~~~~~~~

arraybridge supports a fixed set of memory types. To add custom types,
you would need to extend the ``MemoryType`` enum and implement conversion
logic in the converters module.

Thread Safety
~~~~~~~~~~~~~

The conversion functions are thread-safe, but:

- GPU contexts are thread-local
- Framework-specific thread safety applies
- Use locks when sharing GPU devices across threads

Memory Management
~~~~~~~~~~~~~~~~~

- Converted arrays are independent copies (except DLPack)
- Source arrays are not modified
- Garbage collection works normally
- GPU memory is released when arrays are deleted

Conversion Matrix
-----------------

Full conversion support matrix:

.. list-table:: Conversion Support
   :header-rows: 1
   :stub-columns: 1

   * - Source\\Target
     - NumPy
     - CuPy
     - PyTorch
     - TensorFlow
     - JAX
     - pyclesperanto
   * - NumPy
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - CuPy
     - ✓
     - ✓
     - ✓ (DLPack)
     - ✓
     - ✓ (DLPack)
     - ✓
   * - PyTorch
     - ✓
     - ✓ (DLPack)
     - ✓
     - ✓
     - ✓ (DLPack)
     - ✓
   * - TensorFlow
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - JAX
     - ✓
     - ✓ (DLPack)
     - ✓ (DLPack)
     - ✓
     - ✓
     - ✓
   * - pyclesperanto
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓

✓ = Supported, DLPack = Zero-copy via DLPack

API Reference
-------------

See :doc:`api_reference` for complete function signatures and parameters.

Next Steps
----------

- Learn about :doc:`decorators` for automatic conversion
- Explore :doc:`gpu_features` for device management
- Check :doc:`advanced_topics` for optimization strategies
