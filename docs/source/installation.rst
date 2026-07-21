Installation
============

ArrayBridge requires Python 3.9 or newer. NumPy and metaclass-registry are its
only required runtime dependencies.

Install the base package with:

.. code-block:: bash

   python -m pip install arraybridge

Optional frameworks
-------------------

Install only the framework integrations the application uses:

.. code-block:: bash

   python -m pip install "arraybridge[torch]"
   python -m pip install "arraybridge[cupy]"
   python -m pip install "arraybridge[tensorflow]"
   python -m pip install "arraybridge[jax]"
   python -m pip install "arraybridge[pyclesperanto]"

``arraybridge[all]`` installs every optional framework. ``arraybridge[gpu]`` is
the project's CUDA-oriented tested set and is intentionally more constrained.
Framework hardware and driver compatibility remains the responsibility of the
framework installation.

Development and documentation
-----------------------------

From a repository checkout:

.. code-block:: bash

   python -m pip install -e ".[dev,docs]"
   python -m pytest
   python -m sphinx -E -W --keep-going -b html docs/source docs/source/_build/html

Verify the required NumPy path without optional frameworks:

.. code-block:: python

   import numpy as np

   from arraybridge import convert_memory, detect_memory_type

   value = np.arange(3)
   assert detect_memory_type(value) == "numpy"
   converted = convert_memory(value, "numpy", "numpy", gpu_id=0)

An optional-framework conversion fails when that framework is not installed.
ArrayBridge does not fall back to another target.

See the `repository <https://github.com/OpenHCSDev/arraybridge>`_ for supported
version constraints and current workflows.
