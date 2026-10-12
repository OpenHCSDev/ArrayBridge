Accelerators and memory
=======================

Conversion and stacking use the ``device`` passed to ``convert_memory``,
``stack_slices`` or ``unstack_slices``. Each library class owns its device
discovery, checks, scopes, moves and cache cleanup. An accelerator index
belongs to one library: equal indices in two libraries need not be the same
hardware.

Import defaults
---------------

Before importing TensorFlow or JAX, ArrayBridge sets defaults with
``setdefault`` so the libraries can share a GPU: TensorFlow memory growth is
enabled (``TF_FORCE_GPU_ALLOW_GROWTH=true``) and JAX preallocation is disabled
(``XLA_PYTHON_CLIENT_PREALLOCATE=false``). A value already in the environment
is not overwritten. An application that imports an optional library itself
should call ``library.prepare_import()`` first, for example
``ArrayLibrary.named("tensorflow").prepare_import()``. ArrayBridge logs a
warning when it meets an already imported library whose defaults were not set.

``ArrayLibrary.child_environment(environment=None)`` returns an environment
with the same defaults for a child process. For CuPy it also adds the library
directories of installed NVIDIA wheels to the search path, so a new
interpreter does not depend on another library having loaded CUDA first.

.. code-block:: python

   from arraybridge import ArrayLibrary

   environment = ArrayLibrary.child_environment({})
   assert environment["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"
   assert environment["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"

JAX float64 output requires x64 mode to be enabled before import. Scaling a
JAX array to float64 raises ``ValueError`` when it is not.

Decorated functions
-------------------

A function decorated for a library with accelerators is wrapped only when
that library is installed and sees an accelerator. For CuPy and PyTorch the
call runs on a per-thread CUDA stream of the device the library currently has
selected. With ``oom_recovery=True`` (the default) a call that fails with the
library's out-of-memory error is retried up to two times; before each retry
CuPy and PyTorch release their cached memory on that device and Python's
garbage collector runs. Decorators do not choose a device and do not convert
an input of another library.

The same pieces are methods of the library object:
``library.run_with_oom_recovery(func, max_retries=2, device=None)``,
``library.is_oom_error(error)`` and, on CuPy and PyTorch,
``library.thread_stream(device)``.

Cleanup
-------

``ArrayLibrary.cleanup_all_loaded(device=None)`` asks every library that is
already imported to release its cached device memory, on every accelerator or
on the given ``AcceleratorDevice``. ``library.cleanup_loaded(device=None)``
does it for one library. CuPy and PyTorch release caches; the other libraries
have nothing to release. Cleanup never imports a library. The application
decides when values are no longer needed.

.. code-block:: python

   from arraybridge import ArrayLibrary

   ArrayLibrary.cleanup_all_loaded()

ArrayBridge provides these mechanisms and no scheduling. Worker counts, device
assignment and concurrency limits belong to the application.
