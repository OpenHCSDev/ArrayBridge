.. arraybridge documentation master file

arraybridge Documentation
=========================

**Unified API for NumPy, CuPy, PyTorch, TensorFlow, JAX, and pyclesperanto**

arraybridge provides a unified interface for working with multiple array/tensor frameworks,
featuring automatic memory type conversion, declarative decorators, and zero-copy operations
when possible.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   api/index
   examples/index
   contributing
   ci-cd

Features
--------

* **Unified API**: Single interface for 6 array/tensor frameworks
* **Automatic Conversion**: DLPack + NumPy fallback with automatic path selection
* **Declarative Decorators**: ``@numpy``, ``@torch``, ``@cupy`` for memory type declarations
* **Device Management**: Thread-local GPU contexts and automatic stream management
* **OOM Recovery**: Automatic out-of-memory detection and cache clearing
* **Dtype Preservation**: Automatic dtype preservation across conversions
* **Zero Dependencies**: Only requires NumPy (framework dependencies are optional)

Quick Example
-------------

.. code-block:: python

   from arraybridge import convert_memory, detect_memory_type
   import numpy as np

   # Create NumPy array
   data = np.array([[1, 2], [3, 4]])

   # Convert to PyTorch (if installed)
   torch_data = convert_memory(data, source_type='numpy', target_type='torch', gpu_id=0)

   # Detect memory type
   mem_type = detect_memory_type(torch_data)  # 'torch'

Installation
------------

.. code-block:: bash

   # Base installation (NumPy only)
   pip install arraybridge

   # With specific frameworks
   pip install arraybridge[torch]
   pip install arraybridge[cupy]
   pip install arraybridge[all]

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
