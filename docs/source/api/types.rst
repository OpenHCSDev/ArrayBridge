Types Module
============

.. automodule:: arraybridge.types
   :members:
   :undoc-members:
   :show-inheritance:

Memory Types
------------

MemoryType Enum
~~~~~~~~~~~~~~~

.. autoclass:: arraybridge.types.MemoryType
   :members:
   :undoc-members:

Memory Type Sets
~~~~~~~~~~~~~~~~

.. autodata:: arraybridge.types.CPU_MEMORY_TYPES
.. autodata:: arraybridge.types.GPU_MEMORY_TYPES
.. autodata:: arraybridge.types.SUPPORTED_MEMORY_TYPES
.. autodata:: arraybridge.types.VALID_MEMORY_TYPES

Examples
--------

Using Memory Types
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge.types import MemoryType, GPU_MEMORY_TYPES

   # Access enum values
   print(MemoryType.NUMPY.value)  # 'numpy'
   print(MemoryType.TORCH.value)  # 'torch'

   # Check if a type is GPU-based
   is_gpu = MemoryType.CUPY in GPU_MEMORY_TYPES  # True

String Values
~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge.types import VALID_MEMORY_TYPES

   # Validate memory type strings
   user_input = "torch"
   if user_input in VALID_MEMORY_TYPES:
       print("Valid memory type")
