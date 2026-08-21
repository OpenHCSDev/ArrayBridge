API reference
=============

Public imports
--------------

``MemoryType``, ``CPU_MEMORY_TYPES``, ``GPU_MEMORY_TYPES``, ``SUPPORTED_MEMORY_TYPES``
  Framework declarations. Each member owns import identity, conversion,
  stacking, dtype scaling, device discovery, scoped selection, movement,
  DLPack import/export, cleanup, and pre-import coexistence defaults.

``detect_memory_type`` and ``convert_memory``
  Detection and explicit conversion.

``memory_types``, ``numpy``, ``cupy``, ``torch``, ``tensorflow``, ``jax``, ``pyclesperanto``
  Callable memory declarations and wrappers.

``DtypeConversion``
  Output dtype policy values.

``stack_slices``, ``unstack_slices``, ``process_slices``
  Validated plane/stack operations.

``cleanup_all_gpu_frameworks``
  Installed-framework cache cleanup.

``MemoryConversionError``
  Conversion failure boundary.

The canonical export list is ``arraybridge.__all__``. Names beginning with an
underscore are implementation surfaces even when temporarily re-exported for a
host migration.

Documented public surface
-------------------------

.. automodule:: arraybridge
   :members:
   :exclude-members: _FRAMEWORK_CONFIG, _FRAMEWORK_OPS, _ensure_module, _execute_with_oom_recovery, _get_device_id, _supports_dlpack
   :member-order: bysource
