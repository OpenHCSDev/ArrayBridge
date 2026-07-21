Decorator declarations
======================

.. code-block:: python

   import numpy as np

   from arraybridge import numpy

   @numpy
   def threshold(image, *, cutoff=0.5):
       return image > cutoff

   value = threshold(np.array([0.25, 0.75]))
   assert threshold.input_memory_type == "numpy"

The caller supplies the NumPy value. A compiler or orchestration layer may use
the metadata to plan conversion before calling ``threshold``.
