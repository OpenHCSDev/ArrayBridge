Basic conversion
================

.. code-block:: python

   import numpy as np

   from arraybridge import convert_memory, detect_memory_type

   source = np.arange(9).reshape(3, 3)
   source_type = detect_memory_type(source)
   result = convert_memory(source, source_type, "numpy", 0)

For an optional target such as ``torch`` or ``cupy``, install that framework and
replace the target string. The returned value is allocated according to the
registered converter and requested device.
