Exceptions Module
=================

.. automodule:: arraybridge.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Exception Classes
-----------------

MemoryConversionError
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: arraybridge.exceptions.MemoryConversionError
   :members:
   :undoc-members:

Examples
--------

Catching Conversion Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import convert_memory
   from arraybridge.exceptions import MemoryConversionError
   import numpy as np

   try:
       data = np.array([1, 2, 3])
       result = convert_memory(data, 'numpy', 'invalid_type', gpu_id=0)
   except MemoryConversionError as e:
       print(f"Conversion failed: {e}")
       print(f"Source: {e.source_type}")
       print(f"Target: {e.target_type}")
       print(f"Method: {e.method}")
       print(f"Reason: {e.reason}")
