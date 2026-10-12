User guide
==========

Choose the operation by what it does:

``detect_memory_type``
  Name the library that made an array.

``convert_memory``
  Convert an array from a stated source library to a stated target library on
  a stated device of the target.

Library decorator (``numpy``, ``torch``, ...)
  Declare a function's input and output memory types and apply the dtype,
  slice, stream and out-of-memory wrappers. It does not move the value between
  libraries.

``stack_slices`` / ``unstack_slices``
  Check, convert and stack 2D planes, or split a 3D array into planes.

``ArrayLibrary.named(name)`` / ``ArrayLibrary.of(array)``
  Get the library object: its devices, its array operations, its cleanup.

``ArrayLibrary.cleanup_all_loaded(device=None)``
  Ask every imported library to release its cached device memory.

``ConversionRoute.planned`` / ``ConversionCounters``
  Tell which mechanism a conversion will use; count the conversions that ran.

Library names
-------------

The conversion and stack functions take the string values ``numpy``, ``cupy``,
``torch``, ``tensorflow``, ``jax`` and ``pyclesperanto``. ``convert_memory``
also accepts the ``MemoryType`` member for source and target. What is usable
depends on the installed packages and the hardware. ArrayBridge does not
substitute another library for a target that is not usable.

``MemoryType`` is the set of registered names as an ``Enum``. Use it where a
name is stored or offered as a choice, and ``MemoryType.X.library`` for
anything the library does.

Devices
-------

Every ``device`` argument is ``HOST``, ``AcceleratorDevice(index)`` or
``NamedDevice(name)`` from ``arraybridge``. See :doc:`api_reference`.
