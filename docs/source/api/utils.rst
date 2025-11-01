Utils Module
============

.. automodule:: arraybridge.utils
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

Module Import
~~~~~~~~~~~~~

.. autofunction:: arraybridge.utils.optional_import

Support Checks
~~~~~~~~~~~~~~

.. autofunction:: arraybridge.utils._supports_cuda_array_interface
.. autofunction:: arraybridge.utils._supports_dlpack

Module Management
~~~~~~~~~~~~~~~~~

.. autofunction:: arraybridge.utils._ensure_module

Helper Classes
--------------

ModulePlaceholder
~~~~~~~~~~~~~~~~~

.. autoclass:: arraybridge.utils._ModulePlaceholder
   :members:
   :undoc-members:

Examples
--------

Optional Imports
~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge.utils import optional_import

   # Import with graceful fallback
   torch = optional_import('torch')

   if torch:
       # PyTorch is available
       tensor = torch.tensor([1, 2, 3])
   else:
       # PyTorch not available
       print("PyTorch not installed")

Checking Framework Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge.utils import _supports_dlpack
   import numpy as np

   data = np.array([1, 2, 3])
   if _supports_dlpack(data):
       print("DLPack conversion available")
   else:
       print("Using fallback conversion")
