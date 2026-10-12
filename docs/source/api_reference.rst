API reference
=============

Version 0.5.0 removed names without aliases. The repository's `CHANGELOG.md
<https://github.com/OpenHCSDev/arraybridge/blob/main/CHANGELOG.md>`_ lists
each removed name and what to call instead.

Public imports
--------------

``ArrayLibrary``
  The family of supported libraries. ``ArrayLibrary.named("torch")``,
  ``ArrayLibrary.of(array)``, ``ArrayLibrary.for_module_name("jaxlib")`` and
  ``ArrayLibrary.all()`` return library objects. A library object answers for
  its devices, moves and converts arrays, and runs array operations in that
  library.

``MemoryType``
  The registered library names as an ``Enum``
  (``MemoryType.TORCH.value == "torch"``). It carries no behaviour;
  ``MemoryType.TORCH.library`` is the library object. A library class defined
  later appears as a member.

``Device``, ``HostDevice``, ``AcceleratorDevice``, ``NamedDevice``, ``HOST``
  Where an array lives. See `Devices`_.

``detect_memory_type`` and ``convert_memory``
  Name the library that made an array; convert an array to another library on
  a device.

``ConversionRoute``, ``ConversionRecord``, ``ConversionObserver``, ``ConversionCounters``
  Which mechanism a conversion uses, and a hook that receives each conversion.
  See `Conversion reporting`_.

``Array``
  The nominal type of an array of any supported library.
  ``library.array_class`` is an array of that library.

``ArrayGeometry``
  Shape inspection for arrays and ``ArrayPayload`` values. It reads the
  declared shape and does not move device data to main memory.

``ArrayPayload``
  Abstract container that keeps its context around array data.

``memory_types``, ``numpy``, ``cupy``, ``torch``, ``tensorflow``, ``jax``, ``pyclesperanto``
  Decorators that declare a function's memory types. There is one for every
  registered library, made on first access.

``MemoryContractAttribute``
  Names of the attributes the decorators set on a function.

``DtypeConversion``
  Output dtype policy values.

``SliceBySliceRuntimeParameter``, ``wrap_dtype_preserving_callable``
  The slice control and the dtype and slice wrapper used by the decorators.

``stack_slices``, ``unstack_slices``, ``process_slices``
  Plane and stack operations.

``MemoryConversionError``
  Raised by ``convert_memory`` when a conversion fails.

The export list is ``arraybridge.__all__``. Names beginning with an underscore
are not part of the public interface.

Devices
-------

A device is a value of ``arraybridge.devices``. Equal fields mean equal
devices, and each has a ``label``.

``HOST``
  Main memory, the one ``HostDevice()``. Label ``"host"``.

``AcceleratorDevice(index)``
  One accelerator as its library numbers it (a CUDA or an OpenCL device).
  Label ``"accelerator:<index>"``. The index belongs to one library; the same
  index in two libraries need not be the same hardware.

``NamedDevice(name)``
  A device a library reports that is not one of its numbered accelerators,
  such as pyclesperanto running on a CPU OpenCL device. Label
  ``"device:<name>"``.

Every ``device`` parameter takes one of these. NumPy accepts ``HOST``; CuPy
and pyclesperanto accept ``AcceleratorDevice(i)``; PyTorch, JAX and TensorFlow
accept either. A device the library does not have on this machine raises
``ValueError`` from the library methods and from ``stack_slices`` and
``unstack_slices``; ``convert_memory`` raises ``MemoryConversionError``
carrying that message.

A library object answers these questions:

``library.host_devices()``
  ``(HOST,)`` if the library keeps arrays in main memory, else ``()``.

``library.accelerator_devices()``
  The accelerators the library can use on this machine now. Empty when the
  package is not installed or sees no accelerator.

``library.available_devices()``
  Both of the above, host first.

``library.device_of(array)``
  The device holding the array. Never ``None``.

``library.require_device(device)``
  Raises ``ValueError`` unless the library has the device.

``isinstance(library, HostDevices)`` and ``isinstance(library, AcceleratorDevices)``
(both in ``arraybridge.libraries``) answer whether the library can use main
memory or accelerators at all, without importing it.

.. code-block:: python

   import numpy as np

   from arraybridge import HOST, AcceleratorDevice, ArrayLibrary, MemoryType
   from arraybridge.libraries import AcceleratorDevices, HostDevices

   numpy_library = ArrayLibrary.named("numpy")
   assert numpy_library.host_devices() == (HOST,)
   assert numpy_library.accelerator_devices() == ()
   assert numpy_library.available_devices() == (HOST,)
   assert numpy_library.device_of(np.zeros(3)) == HOST

   assert (HOST.label, AcceleratorDevice(1).label) == ("host", "accelerator:1")

   on_host = [m.value for m in MemoryType if isinstance(m.library, HostDevices)]
   on_accelerators = [m.value for m in MemoryType if isinstance(m.library, AcceleratorDevices)]
   assert on_host == ["numpy", "torch", "tensorflow", "jax"]
   assert on_accelerators == ["cupy", "torch", "tensorflow", "jax", "pyclesperanto"]

Conversion reporting
--------------------

``ConversionRoute`` has one subclass per mechanism, tried in this order:

``DeviceMove`` (``"device move"``)
  Source and target are the same library; the array changes device only.

``DLPackExchange`` (``"dlpack"``)
  The target wraps the source's buffer. Applies when the target imports
  DLPack from the source's device and the source exports from it.

``HostCopy`` (``"host copy"``)
  Download to a NumPy array, then upload to the target. Always applies.

``ConversionRoute.planned(source_library, source_device, target_library,
target_device)`` returns the route class that a conversion of that pair tries
first. It reads only the library classes and the device values: no library is
imported and no device is touched. The conversion selects its route with the
same test. A route that applies but cannot carry a particular array is
skipped and the next one is tried.

.. code-block:: python

   from arraybridge import HOST, AcceleratorDevice, ArrayLibrary, ConversionRoute

   numpy_library = ArrayLibrary.named("numpy")
   torch = ArrayLibrary.named("torch")
   cupy = ArrayLibrary.named("cupy")
   gpu = AcceleratorDevice(0)

   assert ConversionRoute.planned(torch, gpu, cupy, gpu).mechanism == "dlpack"
   assert ConversionRoute.planned(torch, HOST, cupy, gpu).mechanism == "host copy"
   assert ConversionRoute.planned(numpy_library, HOST, torch, HOST).mechanism == "host copy"
   assert ConversionRoute.planned(torch, HOST, torch, gpu).mechanism == "device move"

``ConversionObserver``
  Receives one ``ConversionRecord`` per conversion while it is installed.
  Subclass it and define ``record(record)``. ``observer.install()`` makes it
  the process's observer and returns the one it replaced;
  ``with observer.installed():`` does so for a block.
  ``ConversionObserver.current`` is the installed observer. The default
  ignores conversions, and then nothing is timed. ``library.convert_to``
  (which ``convert_memory`` calls), ``library.to_numpy`` and
  ``library.from_numpy`` report. A call that returns its input unchanged does
  not.

``ConversionRecord``
  Fields: ``source_library``, ``source_device``, ``target_library``,
  ``target_device`` (device labels), ``mechanism``, ``nbytes``, ``seconds``
  and ``declined_mechanisms`` (routes that applied but could not carry that
  array). ``to_json()`` and ``from_json()`` convert to and from a plain
  mapping.

``ConversionCounters``
  An observer that keeps count, bytes and seconds per source library and
  device, target library and device, and mechanism. ``rows()`` (also named
  ``to_json()``) returns one row per kind, largest byte total first;
  ``from_json(rows)`` rebuilds counters; ``merge(other)`` adds another
  process's totals; ``format_table()`` returns aligned text.

.. code-block:: python

   from arraybridge import ConversionCounters, ConversionObserver, ConversionRecord

   record = ConversionRecord(
       source_library="numpy",
       source_device="host",
       target_library="torch",
       target_device="accelerator:0",
       mechanism="host copy",
       nbytes=64,
       seconds=0.25,
   )
   assert ConversionRecord.from_json(record.to_json()) == record

   counters = ConversionCounters()
   with counters.installed():
       # Conversions made in this block are reported to ``counters``.
       ConversionObserver.current.record(record)

   worker = ConversionCounters.from_json(counters.to_json())
   counters.merge(worker)
   row = counters.rows()[0]
   assert (row["mechanism"], row["count"], row["nbytes"]) == ("host copy", 2, 128)
   print(counters.format_table())

With JAX installed, a real conversion is counted the same way:

.. code-block:: python

   import numpy as np

   from arraybridge import HOST, ConversionCounters, convert_memory

   counters = ConversionCounters()
   with counters.installed():
       convert_memory(np.zeros((4, 4), dtype=np.float32), "numpy", "jax", HOST)

   row = counters.rows()[0]
   assert (row["source_library"], row["target_library"]) == ("numpy", "jax")
   assert (row["mechanism"], row["count"], row["nbytes"]) == ("host copy", 1, 64)

Modules
-------

The root package exports the names listed above. The role parents and the
library classes are in ``arraybridge.libraries``, and the route classes in
``arraybridge.conversion``.

.. automodule:: arraybridge
   :members:
   :exclude-members: _ensure_module, _supports_dlpack
   :member-order: bysource

.. automodule:: arraybridge.libraries
   :members:
   :member-order: bysource

.. automodule:: arraybridge.devices
   :members:
   :member-order: bysource

.. automodule:: arraybridge.conversion
   :members:
   :member-order: bysource
