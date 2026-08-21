User guide
==========

Choose the operation by ownership:

``detect_memory_type``
  Identify the framework for an existing supported value.

``convert_memory``
  Convert one value from an explicit source type to an explicit target type on
  the requested device.

Framework decorator
  Declare a callable's input/output memory contract and apply dtype, slice, and
  supported stream/OOM wrappers. It does not move the value between frameworks.

``stack_slices`` / ``unstack_slices``
  Validate and convert plane collections at an explicit stack boundary.

``cleanup_all_gpu_frameworks``
  Ask every loaded supported GPU framework to release caches after an
  application-owned lifecycle boundary.

Framework names
---------------

The public conversion and stack APIs use string values: ``numpy``, ``cupy``,
``torch``, ``tensorflow``, ``jax``, and ``pyclesperanto``. ``convert_memory``
accepts the canonical ``MemoryType`` member at both its source and target
boundaries. Availability depends on optional dependencies and hardware.
ArrayBridge never silently substitutes a different framework for an invalid
target.

Applications should carry these values through typed configuration or plans,
not infer them from function names.
