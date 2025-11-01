Installation
============

This guide covers the installation of arraybridge and its optional dependencies.

Requirements
------------

**Minimum Requirements:**

- Python 3.9 or later
- NumPy 1.20 or later

**Optional Framework Support:**

- CuPy 10.0+ (for GPU arrays via CuPy)
- PyTorch 1.10+ (for PyTorch tensors)
- TensorFlow 2.8+ (for TensorFlow tensors)
- JAX 0.3+ with jaxlib 0.3+ (for JAX arrays)
- pyclesperanto 0.10+ (for GPU image processing)

Basic Installation
------------------

Install arraybridge with pip:

.. code-block:: bash

   pip install arraybridge

This installs arraybridge with NumPy as the only dependency. All framework integrations
are optional and can be installed separately.

Framework-Specific Installation
--------------------------------

Install with specific framework support using extras:

PyTorch Support
~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install arraybridge[torch]

This adds PyTorch as a dependency, enabling conversion to/from PyTorch tensors.

CuPy Support
~~~~~~~~~~~~

.. code-block:: bash

   pip install arraybridge[cupy]

This adds CuPy for GPU array support. Note that CuPy requires CUDA to be installed
on your system.

TensorFlow Support
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install arraybridge[tensorflow]

This adds TensorFlow, enabling conversion to/from TensorFlow tensors.

JAX Support
~~~~~~~~~~~

.. code-block:: bash

   pip install arraybridge[jax]

This adds JAX and jaxlib for JAX array support.

pyclesperanto Support
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install arraybridge[pyclesperanto]

This adds pyclesperanto for GPU-accelerated image processing.

All Frameworks
~~~~~~~~~~~~~~

Install all optional framework dependencies at once:

.. code-block:: bash

   pip install arraybridge[all]

This is useful for development and testing across multiple frameworks.

Development Installation
------------------------

For development work, install arraybridge in editable mode with development dependencies:

.. code-block:: bash

   git clone https://github.com/trissim/arraybridge.git
   cd arraybridge
   pip install -e .[dev]

This includes testing tools (pytest, pytest-cov), linters (black, ruff), and type
checking (mypy).

Documentation Building
~~~~~~~~~~~~~~~~~~~~~~

To build the documentation locally:

.. code-block:: bash

   pip install -e .[docs]
   cd docs
   make html

The built documentation will be available in ``docs/build/html/``.

Verifying Installation
----------------------

After installation, verify that arraybridge is working:

.. code-block:: python

   import arraybridge
   print(arraybridge.__version__)

   # Check which frameworks are available
   from arraybridge import detect_memory_type
   import numpy as np

   data = np.array([1, 2, 3])
   print(detect_memory_type(data))  # Should print 'numpy'

If you installed framework support, verify it works:

.. code-block:: python

   from arraybridge import convert_memory
   import numpy as np

   # Example with PyTorch (if installed)
   try:
       data = np.array([1, 2, 3])
       torch_data = convert_memory(data, source_type='numpy', target_type='torch')
       print("PyTorch support is working!")
   except Exception as e:
       print(f"PyTorch not available: {e}")

GPU Setup
---------

If you want to use GPU features (CuPy, GPU-enabled PyTorch, etc.), ensure you have:

1. **NVIDIA GPU**: With compute capability 3.5 or higher
2. **CUDA Toolkit**: Compatible version for your framework
3. **cuDNN**: Required for TensorFlow and some PyTorch operations

CUDA Installation
~~~~~~~~~~~~~~~~~

For CuPy and GPU support:

- Install CUDA Toolkit from NVIDIA's website
- For CuPy, the CUDA version should match your CuPy installation
- Check CuPy's installation guide for version compatibility

PyTorch CUDA
~~~~~~~~~~~~

PyTorch can be installed with CUDA support:

.. code-block:: bash

   # Example for CUDA 12.1
   pip install torch --index-url https://download.pytorch.org/whl/cu121

Refer to PyTorch's official website for the latest installation instructions.

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Import errors for optional frameworks**

If you see import errors like ``ModuleNotFoundError: No module named 'torch'``,
this means the optional framework is not installed. arraybridge will work with
NumPy-only operations, but framework-specific conversions will fail.

**CUDA/GPU errors**

If you encounter CUDA-related errors:

- Verify CUDA is installed: ``nvidia-smi``
- Check that your CUDA version matches your framework requirements
- Ensure GPU drivers are up to date

**Out of memory errors**

arraybridge includes automatic OOM recovery, but you may need to:

- Reduce batch sizes or array dimensions
- Use the memory cleanup utilities (see :doc:`advanced_topics`)
- Monitor GPU memory usage with ``nvidia-smi``

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the :doc:`user_guide` for usage patterns
2. Review the :doc:`api_reference` for function signatures
3. Open an issue on `GitHub <https://github.com/trissim/arraybridge/issues>`_

Next Steps
----------

After installation:

- Follow the :doc:`quickstart` guide for basic usage
- Read the :doc:`user_guide` for comprehensive examples
- Explore the :doc:`api_reference` for detailed API documentation
