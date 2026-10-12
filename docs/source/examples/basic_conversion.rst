Basic conversion
================

.. code-block:: python

   import numpy as np

   from arraybridge import HOST, convert_memory, detect_memory_type

   source = np.arange(9).reshape(3, 3)
   source_type = detect_memory_type(source)
   result = convert_memory(source, source_type, "numpy", HOST)

For an optional target, install that library, replace the target string and
pass a device the target has: ``convert_memory(source, source_type, "torch",
HOST)`` or ``convert_memory(source, source_type, "cupy",
AcceleratorDevice(0))``.
