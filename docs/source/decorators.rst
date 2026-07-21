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

The decorator does not accept ``gpu_id`` or ``clear_cuda_cache``. Device
selection belongs to explicit conversion or the host runtime. A non-callable
``contract`` is stored as declarative metadata; a callable contract validates
the returned value.

Dtype policy
------------

Direct calls default to preserving the input dtype. Hosts can pass an object
implementing ``DtypeConversionConfig`` to select native or explicit output dtype
behavior.
