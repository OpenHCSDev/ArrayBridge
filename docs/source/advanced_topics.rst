Adding a library
================

Each supported library is one class of the family
``arraybridge.libraries.ArrayLibrary``. The class states the library's name
and owns what that library does its own way: devices, moving data, streams,
DLPack, out-of-memory recognition, child-process environment and array
operations. What several libraries share is a role parent they inherit:
``HostDevices``, ``AcceleratorDevices``, ``CudaLibrary``, ``DLPackImport``,
``DLPackProtocolExport``, ``NvidiaWheelLibraries``, ``MutableArrays``.

To add a library, write one subclass with ``name = "..."``, the role parents
it plays and their abstract methods. Lookup, the ``MemoryType`` member, the
decorator and the conversion routes follow from the class. The repository's
`ADDING_NEW_FRAMEWORKS.md
<https://github.com/OpenHCSDev/arraybridge/blob/main/ADDING_NEW_FRAMEWORKS.md>`_
has a runnable example and the list of methods each role parent asks for.

Do not keep a second table of library names in a consumer. Iterate
``MemoryType`` or ``ArrayLibrary.all()``.

Caching and batching
--------------------

ArrayBridge does not cache or batch conversions. How long a converted value
stays valid is the caller's knowledge. The state ArrayBridge keeps is the
per-thread CUDA stream of each CuPy and PyTorch device and the installed
``ConversionObserver``.
