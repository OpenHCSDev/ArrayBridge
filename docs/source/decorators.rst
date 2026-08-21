Memory declaration decorators
=============================

``memory_types`` and the six framework helpers declare callable memory
contracts.

.. code-block:: python

   from arraybridge import memory_types, numpy

   @memory_types(input_type="numpy", output_type="numpy")
   def identity(image):
       return image

   @numpy(oom_recovery=False)
   def offset(image, value=1):
       return image + value

Framework helpers accept ``input_type``, ``output_type``, ``oom_recovery``, and
``contract``. They may be used as ``@numpy`` or ``@numpy()``.

Direct-call behavior
--------------------

Decorators do not convert the input or output between frameworks. They attach
metadata, add keyword-only ``slice_by_slice`` and ``dtype_config`` runtime
parameters, apply the selected dtype policy, and wrap supported GPU frameworks
with thread-local stream/OOM handling.

Framework helpers also attach ``execution_memory_type``. It identifies the
framework that owns the function body even when ``input_type`` or
``output_type`` is overridden. Host runtimes use that declaration when scoping
callable execution; boundary types alone do not prove execution ownership.

Metadata key API
----------------

``MemoryContractAttribute`` is the public owner of the callable metadata keys:
``INPUT`` maps to ``input_memory_type``, ``OUTPUT`` maps to
``output_memory_type``, and ``EXECUTION`` maps to ``execution_memory_type``.
Each member provides ``read(namespace, default=None)`` and
``write(namespace, value)`` for object and mapping namespaces. Host libraries
can consume these members without maintaining a duplicate key registry.

The decorator does not accept ``gpu_id`` or ``clear_cuda_cache``. Device
selection belongs to explicit conversion or the host runtime. A non-callable
``contract`` is stored as declarative metadata; a callable contract validates
the returned value.

Dtype policy
------------

Direct calls default to preserving the input dtype. Hosts can pass an object
implementing ``DtypeConversionConfig`` to select native or explicit output dtype
behavior.
