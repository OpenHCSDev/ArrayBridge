# arraybridge

ArrayBridge provides explicit conversion and shared lifecycle utilities for
NumPy, CuPy, PyTorch, TensorFlow, JAX, and pyclesperanto arrays.

Core dependencies are NumPy and metaclass-registry. Other frameworks are
optional.

## Quick start

```python
import numpy as np

from arraybridge import convert_memory, detect_memory_type

value = np.arange(6).reshape(2, 3)
assert detect_memory_type(value) == "numpy"

copy = convert_memory(
    value,
    source_type="numpy",
    target_type="numpy",
    gpu_id=0,
)
```

`convert_memory` requires the declared source type, target type, and device id.
The source and target `MemoryType` declarations perform conversion directly.
The device id is required even for CPU conversions so call sites have one
stable signature. A GPU target must declare that identifier as available;
ArrayBridge does not silently place the value on the CPU.

## Declarative decorators

```python
from arraybridge import numpy

@numpy
def normalize(image):
    return image / max(float(image.max()), 1.0)
```

The framework decorators attach `input_memory_type`, `output_memory_type`, and
`execution_memory_type` metadata, provide dtype/slice runtime parameters, and
add framework-specific stream/OOM handling where supported. They do **not**
convert inputs or outputs between frameworks and do not accept a `gpu_id`
argument. A host runtime must call `convert_memory` at the boundary it plans and
scope execution using the execution declaration.

## Stack utilities

```python
import numpy as np

from arraybridge import stack_slices, unstack_slices

slices = [np.zeros((8, 8)), np.ones((8, 8))]
stack = stack_slices(slices, memory_type="numpy", gpu_id=0)
restored = unstack_slices(stack, memory_type="numpy", gpu_id=0)
```

`stack_slices` requires non-empty 2D inputs. `unstack_slices` requires a 3D
array. Both validate shape and use explicit target memory/device declarations.

## Installation

```bash
pip install arraybridge
pip install "arraybridge[torch]"
pip install "arraybridge[cupy]"
```

Documentation: <https://arraybridge.readthedocs.io>
