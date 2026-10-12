Stack utilities
===============

``stack_slices`` converts a non-empty list of 2D planes to the declared memory
type and returns one 3D ``[Z, Y, X]`` array.

``unstack_slices`` requires a 3D array, converts it to the declared memory type,
and returns its 2D planes.

.. code-block:: python

   import numpy as np

   from arraybridge import HOST, stack_slices, unstack_slices

   planes = [np.zeros((3, 5)), np.ones((3, 5))]
   stack = stack_slices(planes, memory_type="numpy", device=HOST)
   planes_again = unstack_slices(stack, memory_type="numpy", device=HOST)

Both functions require ``memory_type`` and ``device``, a device of the target
library (see :doc:`converters`). They raise ``ValueError`` for an empty list,
a wrong number of dimensions or a device the target library does not have.
What the stack axis means is the application's knowledge.

``process_slices(image, func, args, kwargs, device=None)`` is the helper
decorated functions use when ``slice_by_slice=True``. It unstacks on the
device holding ``image`` unless ``device`` is given, calls ``func`` on each
plane, and stacks the results in the library and on the device of the first
result.
