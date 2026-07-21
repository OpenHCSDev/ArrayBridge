Quick start
===========

Detection and conversion
------------------------

.. code-block:: python

   import numpy as np

   from arraybridge import convert_memory, detect_memory_type

   value = np.arange(6).reshape(2, 3)
   source = detect_memory_type(value)
   converted = convert_memory(
       value,
       source_type=source,
       target_type="numpy",
       gpu_id=0,
   )

``detect_memory_type`` returns a string such as ``"numpy"`` or ``"torch"``.
Unknown array classes fail loudly. ``convert_memory`` requires all four
arguments; use a real GPU id when the target framework is GPU-backed.

Callable declarations
---------------------

.. code-block:: python

   from arraybridge import numpy

   @numpy
   def double(image):
       return image * 2

   assert double.input_memory_type == "numpy"
   assert double.output_memory_type == "numpy"

The decorator expects the caller to supply a compatible array. A host runtime
may read the metadata and call ``convert_memory`` before invocation.

Stacks
------

.. code-block:: python

   from arraybridge import stack_slices, unstack_slices

   slices = [np.zeros((4, 4)), np.ones((4, 4))]
   volume = stack_slices(slices, memory_type="numpy", gpu_id=0)
   restored = unstack_slices(volume, memory_type="numpy", gpu_id=0)

See :doc:`decorators`, :doc:`converters`, and :doc:`stack_utils`.
