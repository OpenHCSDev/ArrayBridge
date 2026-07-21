Stack utilities
===============

``stack_slices`` converts a non-empty list of 2D planes to the declared memory
type and returns one 3D ``[Z, Y, X]`` array.

``unstack_slices`` requires a 3D array, converts it to the declared memory type,
and returns its 2D planes.

.. code-block:: python

   import numpy as np

   from arraybridge import stack_slices, unstack_slices

   planes = [np.zeros((3, 5)), np.ones((3, 5))]
   stack = stack_slices(planes, memory_type="numpy", gpu_id=0)
   planes_again = unstack_slices(stack, memory_type="numpy", gpu_id=0)

Both functions require ``memory_type`` and ``gpu_id``. They validate dimensions
and fail on empty/malformed inputs. A host should declare what the stack axis
means; ArrayBridge owns only array shape, conversion, and device mechanics.

``process_slices`` is the lower-level helper used by decorated functions when
``slice_by_slice=True``.
