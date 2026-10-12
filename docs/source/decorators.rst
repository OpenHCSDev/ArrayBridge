Memory declaration decorators
=============================

``memory_types`` and the decorator named after each library (``numpy``,
``cupy``, ``torch``, ``tensorflow``, ``jax``, ``pyclesperanto``) declare which
library a function's input and output arrays belong to. A decorator exists for
every registered library and is made on first access.

.. code-block:: python

   from arraybridge import memory_types, numpy

   @memory_types(input_type="numpy", output_type="numpy")
   def identity(image):
       return image

   @numpy(oom_recovery=False)
   def offset(image, value=1):
       return image + value

The library decorators accept ``input_type``, ``output_type``,
``oom_recovery``, ``contract``, ``slice_by_slice_default`` and
``dtype_config_default``. They may be used as ``@numpy`` or ``@numpy()``.

Direct-call behavior
--------------------

Decorators do not convert the input or output between libraries. They set
the memory type attributes, add the keyword-only ``slice_by_slice`` and
``dtype_config`` parameters, apply the selected dtype policy, and wrap the
call with ``library.execution_wrapper(func, oom_recovery)``: per-thread CUDA
streams for CuPy and PyTorch and out-of-memory retries for libraries with
accelerators (see :doc:`gpu_features`). For NumPy the wrapper adds nothing.

The library decorators also set ``execution_memory_type``: the library the
function body is written for, even when ``input_type`` or ``output_type``
names another one.

Attribute names
---------------

``MemoryContractAttribute`` names the attributes: ``INPUT`` is
``input_memory_type``, ``OUTPUT`` is ``output_memory_type`` and ``EXECUTION``
is ``execution_memory_type``. Each member has ``read(namespace, default=None)``
and ``write(namespace, value)`` for objects and mappings.

The decorators take no device argument. The device is chosen by the
conversion that produces the input or by the application. A non-callable
``contract`` is stored as declarative metadata; a callable contract validates
the returned value.

Dtype policy
------------

Direct calls default to preserving the input dtype. Hosts can pass an object
implementing ``DtypeConversionConfig`` to select native or explicit output dtype
behavior.

``wrap_dtype_preserving_callable(func, mem_type)`` applies only the dtype and
slice wrapper, for applications that declare memory types themselves.
``mem_type`` is a ``MemoryType`` member.
