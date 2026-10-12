Conversion
==========

``convert_memory(data, source_type, target_type, device)`` converts ``data``
from the source library to the target library on ``device``. It looks up the
two library objects and calls ``source.convert_to(data, target, device)``.

.. code-block:: python

   import numpy as np

   from arraybridge import HOST, MemoryType, convert_memory

   value = np.arange(4)
   result = convert_memory(value, MemoryType.NUMPY, MemoryType.NUMPY, HOST)
   assert result is value

``source_type`` and ``target_type`` accept a ``MemoryType`` member or its
string value. Call ``detect_memory_type`` when the source library is not
already known. Passing the wrong source type is a caller error.

Devices
-------

``device`` is required and is a device of the target library: ``HOST`` for
NumPy; ``AcceleratorDevice(i)`` for CuPy and pyclesperanto; either for
PyTorch, JAX and TensorFlow. Moving an array between two devices of one
library is a conversion with the same source and target type. See
:doc:`api_reference` for the device values and the questions a library
answers about them.

.. code-block:: python

   import numpy as np

   from arraybridge import AcceleratorDevice, MemoryConversionError, convert_memory

   try:
       convert_memory(np.arange(4), "numpy", "numpy", AcceleratorDevice(0))
   except MemoryConversionError as error:
       assert "accelerator:0 is unavailable" in str(error)
   else:
       raise AssertionError("NumPy has no accelerator device")

Routes
------

A conversion tries, in order: a device move when source and target are the
same library; a DLPack exchange when the target can wrap the source's buffer
on its device; a copy through a NumPy array in main memory.
``ConversionRoute.planned`` tells which is tried first for a pair, and a
``ConversionObserver`` receives what actually ran; both are described in
:doc:`api_reference`. No pair is promised to be zero-copy.

Failures
--------

An unregistered library name raises ``ValueError``. Any failure during the
conversion, including a missing optional package or a device the target does
not have, raises ``MemoryConversionError`` with the source and target names.
An optional library is imported only when a conversion uses it.
