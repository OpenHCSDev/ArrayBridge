Quick start
===========

Detection and conversion
------------------------

.. code-block:: python

   import numpy as np

   from arraybridge import HOST, convert_memory, detect_memory_type

   value = np.arange(6).reshape(2, 3)
   source = detect_memory_type(value)
   converted = convert_memory(
       value,
       source_type=source,
       target_type="numpy",
       device=HOST,
   )

``detect_memory_type`` returns a string such as ``"numpy"`` or ``"torch"``.
An array of an unknown class raises ``ValueError``. ``convert_memory``
requires all four arguments. ``device`` is a device of the target library:
``HOST`` for NumPy, ``AcceleratorDevice(i)`` for CuPy and pyclesperanto, either
for PyTorch, JAX and TensorFlow.

Callable declarations
---------------------

.. code-block:: python

   from arraybridge import numpy

   @numpy
   def double(image):
       return image * 2

   assert double.input_memory_type == "numpy"
   assert double.output_memory_type == "numpy"

The decorator expects the caller to pass an array of the declared library. An
application may read the attributes and call ``convert_memory`` first.

Stacks
------

.. code-block:: python

   import numpy as np

   from arraybridge import HOST, stack_slices, unstack_slices

   slices = [np.zeros((4, 4)), np.ones((4, 4))]
   volume = stack_slices(slices, memory_type="numpy", device=HOST)
   restored = unstack_slices(volume, memory_type="numpy", device=HOST)

See :doc:`decorators`, :doc:`converters`, and :doc:`stack_utils`.
