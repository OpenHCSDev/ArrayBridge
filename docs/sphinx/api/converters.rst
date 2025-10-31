Converters Module
=================

.. automodule:: arraybridge.converters
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
--------------

convert_memory
~~~~~~~~~~~~~~

.. autofunction:: arraybridge.converters.convert_memory

detect_memory_type
~~~~~~~~~~~~~~~~~~

.. autofunction:: arraybridge.converters.detect_memory_type

Examples
--------

Basic Conversion
~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import convert_memory
   import numpy as np

   data = np.array([1, 2, 3])
   result = convert_memory(data, 'numpy', 'torch', gpu_id=0)

Type Detection
~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import detect_memory_type
   import numpy as np

   data = np.array([1, 2, 3])
   mem_type = detect_memory_type(data)
   print(mem_type)  # 'numpy'
