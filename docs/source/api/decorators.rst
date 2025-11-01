Decorators Module
=================

.. automodule:: arraybridge.decorators
   :members:
   :undoc-members:
   :show-inheritance:

Available Decorators
--------------------

Base Decorator
~~~~~~~~~~~~~~

.. autofunction:: arraybridge.decorators.memory_types

Framework-Specific Decorators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: arraybridge.numpy
.. autofunction:: arraybridge.cupy
.. autofunction:: arraybridge.torch
.. autofunction:: arraybridge.tensorflow
.. autofunction:: arraybridge.jax

Dtype Conversion
~~~~~~~~~~~~~~~~

.. autoclass:: arraybridge.decorators.DtypeConversion
   :members:
   :undoc-members:

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from arraybridge import torch
   import numpy as np

   @torch(input_type='numpy', output_type='torch')
   def process_data(data):
       return data * 2

   result = process_data(np.array([1, 2, 3]))

With OOM Recovery
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import cupy

   @cupy(oom_recovery=True)
   def gpu_operation(data):
       return data @ data.T

Dtype Preservation
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arraybridge import torch
   from arraybridge.decorators import DtypeConversion

   @torch()
   def process_image(image, dtype_conversion=DtypeConversion.PRESERVE_INPUT):
       # Automatically preserves input dtype
       return image * 2
