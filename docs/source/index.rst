arraybridge
===========

ArrayBridge converts arrays between NumPy, CuPy, PyTorch, TensorFlow, JAX and
pyclesperanto. Each library is one class that answers for its devices, moves
and converts its arrays, and runs array operations natively. The package also
provides memory-type decorators, stack utilities, a dtype policy, per-thread
CUDA streams, out-of-memory retries and cache cleanup.

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
   release_035_validation
   release_036_validation

What converts and what does not
-------------------------------

``convert_memory`` moves values between libraries. Decorators declare a
function's input and output memory types; calling a decorated function does
not convert anything.

ArrayBridge does not plan pipelines or schedule devices. An application reads
the declared types and decides where to convert and on which device.
