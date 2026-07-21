arraybridge
===========

ArrayBridge owns explicit array-framework conversion, memory-type declarations,
stack utilities, dtype policy, GPU stream/OOM helpers, and cleanup for scientific
applications.

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   user_guide
   converters
   decorators
   stack_utils
   gpu_features
   advanced_topics
   api_reference
   examples/index
   contributing
   ci-cd

Boundary rule
-------------

Conversion is explicit. ``convert_memory`` moves values between frameworks.
Decorators declare callable input/output memory types and runtime policies; they
do not perform cross-framework conversion on direct calls.

ArrayBridge does not own a host application's compiler, function catalog, or
resource scheduler. Hosts consume the declarations and plan conversions/devices
at their own boundary.
