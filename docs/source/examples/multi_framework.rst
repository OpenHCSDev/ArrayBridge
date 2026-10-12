Converting between libraries
============================

Convert at the place that knows the target library and device:

.. code-block:: python

   import numpy as np

   from arraybridge import HOST, convert_memory, detect_memory_type

   def as_library(value, target, device):
       return convert_memory(value, detect_memory_type(value), target, device)

   value = np.arange(4)
   assert as_library(value, "numpy", HOST) is value

A conversion to the library and device the value is already on returns the
same object, so no check is needed first.

For a sequence of functions written for different libraries, read their
declared input and output memory types and convert between steps. The
functions should not each guess the library of what they receive.
