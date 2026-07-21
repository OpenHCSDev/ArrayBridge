Multi-framework boundary
========================

Keep conversion visible at the point that owns the boundary:

.. code-block:: python

   from arraybridge import convert_memory, detect_memory_type

   def as_framework(value, target, gpu_id):
       source = detect_memory_type(value)
       if source == target:
           return value
       return convert_memory(value, source, target, gpu_id)

For a sequence of framework-specific functions, a host compiler should read
their declared input/output memory types and create one conversion plan. The
functions themselves should not each guess the incoming framework.
