Conversion system
=================

``convert_memory(data, source_type, target_type, gpu_id)`` resolves the source
and target ``MemoryType`` declarations and lets those declarations perform the
conversion.

.. code-block:: python

   import numpy as np

   from arraybridge import MemoryType, convert_memory

   value = np.arange(4)
   result = convert_memory(
       value,
       MemoryType.NUMPY,
       MemoryType.NUMPY,
       0,
   )

Both ``source_type`` and ``target_type`` accept either the canonical
``MemoryType`` member or its string value. Decorators normalize the same
members to string metadata at declaration time, so callable contracts and
conversion plans share one taxonomy without carrying duplicate enum classes.

Call ``detect_memory_type`` when the source type is not already known. Passing a
wrong source declaration is a caller error; conversion planning should keep the
declared type aligned with the actual value.

Device semantics
----------------

``gpu_id`` is required. GPU converters use it for allocation or device
selection. CPU conversion accepts the same argument for a uniform API. Moving a
value between two devices is expressed as conversion with the target device id.

Declarations prefer framework-native or DLPack paths when both endpoints
support them and otherwise use an explicit NumPy round trip. No zero-copy
guarantee applies to every pair.

Failures
--------

Invalid framework names raise ``ValueError``. Conversion failures raise
``MemoryConversionError`` with source/target context. Optional frameworks are
loaded only when their declaration is used.
