Multi-Framework Examples
========================

Framework-Agnostic Pipeline
----------------------------

.. code-block:: python

   from arraybridge import convert_memory, detect_memory_type

   class DataPipeline:
       """Framework-agnostic data processing pipeline."""

       def __init__(self, target_framework='torch'):
           self.target_framework = target_framework

       def process(self, data):
           # Detect input type
           source_type = detect_memory_type(data)

           # Convert to target framework
           data = convert_memory(
               data, source_type, self.target_framework, gpu_id=0
           )

           # Process (framework-specific code here)
           result = data * 2

           return result

   # Can work with any input framework
   import numpy as np
   pipeline = DataPipeline(target_framework='torch')

   np_data = np.array([1, 2, 3])
   result = pipeline.process(np_data)

Mixed Framework Processing
---------------------------

.. code-block:: python

   from arraybridge import convert_memory
   import numpy as np

   def hybrid_pipeline(data):
       """Use different frameworks for different steps."""

       # Step 1: Preprocess on CPU with NumPy
       preprocessed = convert_memory(data, 'numpy', 'numpy', gpu_id=0)
       preprocessed = preprocessed - preprocessed.mean()

       # Step 2: Heavy computation on GPU with PyTorch
       gpu_data = convert_memory(preprocessed, 'numpy', 'torch', gpu_id=0)
       # ... PyTorch operations ...

       # Step 3: Post-process with CuPy for specialized GPU ops
       cupy_data = convert_memory(gpu_data, 'torch', 'cupy', gpu_id=0)
       # ... CuPy operations ...

       # Step 4: Return as NumPy for compatibility
       return convert_memory(cupy_data, 'cupy', 'numpy', gpu_id=0)

Multi-GPU Processing
---------------------

.. code-block:: python

   from arraybridge import convert_memory
   import numpy as np

   def multi_gpu_process(data_list, num_gpus=2):
       """Distribute processing across multiple GPUs."""
       results = []

       for i, data in enumerate(data_list):
           # Distribute across GPUs
           gpu_id = i % num_gpus

           # Convert to GPU
           gpu_data = convert_memory(data, 'numpy', 'torch', gpu_id=gpu_id)

           # Process on assigned GPU
           result = gpu_data * 2  # Your processing here

           # Convert back
           cpu_result = convert_memory(result, 'torch', 'numpy', gpu_id=gpu_id)
           results.append(cpu_result)

       return results

   # Process multiple arrays across 2 GPUs
   arrays = [np.random.rand(100, 100) for _ in range(10)]
   results = multi_gpu_process(arrays, num_gpus=2)

Framework Compatibility Layer
------------------------------

.. code-block:: python

   from arraybridge import detect_memory_type, convert_memory

   class FrameworkAdapter:
       """Adapter to make functions work with any framework."""

       def __init__(self, func, output_type='numpy'):
           self.func = func
           self.output_type = output_type

       def __call__(self, data, *args, **kwargs):
           # Detect input type
           input_type = detect_memory_type(data)

           # Convert to NumPy for processing
           np_data = convert_memory(data, input_type, 'numpy', gpu_id=0)

           # Process
           result = self.func(np_data, *args, **kwargs)

           # Convert to desired output type
           return convert_memory(result, 'numpy', self.output_type, gpu_id=0)

   # Make NumPy function work with any framework
   def numpy_function(data):
       return data + 1

   # Create adapted version
   universal_func = FrameworkAdapter(numpy_function, output_type='torch')

   # Works with any input type
   import numpy as np
   np_result = universal_func(np.array([1, 2, 3]))
