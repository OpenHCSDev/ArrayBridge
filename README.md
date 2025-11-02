# arraybridge

**Unified API for NumPy, CuPy, PyTorch, TensorFlow, JAX, and pyclesperanto**

[![PyPI version](https://badge.fury.io/py/arraybridge.svg)](https://badge.fury.io/py/arraybridge)
[![Documentation Status](https://readthedocs.org/projects/arraybridge/badge/?version=latest)](https://arraybridge.readthedocs.io/en/latest/?badge=latest)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage](https://raw.githubusercontent.com/trissim/arraybridge/main/.github/badges/coverage.svg)](https://trissim.github.io/arraybridge/coverage/)

## Features

- **Unified API**: Single interface for 6 array/tensor frameworks
- **Automatic Conversion**: DLPack + NumPy fallback with automatic path selection
- **Declarative Decorators**: `@numpy`, `@torch`, `@cupy` for memory type declarations
- **Device Management**: Thread-local GPU contexts and automatic stream management
- **OOM Recovery**: Automatic out-of-memory detection and cache clearing
- **Dtype Preservation**: Automatic dtype preservation across conversions
- **Zero Dependencies**: Only requires NumPy (framework dependencies are optional)

## Quick Start

```python
from arraybridge import convert_memory, detect_memory_type
import numpy as np

# Create NumPy array
data = np.array([[1, 2], [3, 4]])

# Convert to PyTorch (if installed)
torch_data = convert_memory(data, source_type='numpy', target_type='torch', gpu_id=0)

# Detect memory type
mem_type = detect_memory_type(torch_data)  # 'torch'
```

## Declarative Decorators

```python
from arraybridge import numpy, torch, cupy

@torch(input_type='numpy', output_type='torch', oom_recovery=True)
def my_gpu_function(data):
    """Automatically converts input from NumPy to PyTorch."""
    return data * 2

# Use with NumPy input
result = my_gpu_function(np.array([1, 2, 3]))  # Returns PyTorch tensor
```

## Installation

```bash
# Base installation (NumPy only)
pip install arraybridge

# With specific frameworks
pip install arraybridge[torch]
pip install arraybridge[cupy]
pip install arraybridge[tensorflow]
pip install arraybridge[jax]
pip install arraybridge[pyclesperanto]

# With all frameworks
pip install arraybridge[all]
```

## Supported Frameworks

| Framework | CPU | GPU | DLPack | Notes |
|-----------|-----|-----|--------|-------|
| NumPy | ✅ | ❌ | ❌ | Base framework |
| CuPy | ❌ | ✅ | ✅ | CUDA arrays |
| PyTorch | ✅ | ✅ | ✅ | Tensors |
| TensorFlow | ✅ | ✅ | ✅ | Tensors |
| JAX | ✅ | ✅ | ✅ | Arrays |
| pyclesperanto | ❌ | ✅ | ❌ | OpenCL arrays |

## Why arraybridge?

**Before** (Manual conversion hell):
```python
import numpy as np
import torch
import cupy as cp

def process_data(data, target='torch'):
    if target == 'torch':
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).cuda()
        elif isinstance(data, cp.ndarray):
            return torch.as_tensor(data, device='cuda')
    elif target == 'cupy':
        if isinstance(data, np.ndarray):
            return cp.asarray(data)
        elif hasattr(data, '__cuda_array_interface__'):
            return cp.asarray(data)
    # ... 30 more lines of if/elif ...
```

**After** (arraybridge):
```python
from arraybridge import convert_memory, detect_memory_type

def process_data(data, target='torch'):
    source = detect_memory_type(data)
    return convert_memory(data, source_type=source, target_type=target, gpu_id=0)
```

## Advanced Features

### Thread-Local GPU Streams

```python
from arraybridge import torch

@torch(oom_recovery=True)
def parallel_processing(data):
    # Automatically uses thread-local CUDA stream
    # Enables true parallelization across threads
    return data * 2
```

### OOM Recovery

```python
from arraybridge import cupy

@cupy(oom_recovery=True)
def memory_intensive_operation(data):
    # Automatically catches OOM errors
    # Clears GPU cache and retries
    return data @ data.T
```

### Stack Utilities

```python
from arraybridge import stack_slices, unstack_slices

# Stack 2D slices into 3D array
slices_2d = [np.random.rand(100, 100) for _ in range(50)]
volume_3d = stack_slices(slices_2d, memory_type='torch', gpu_id=0)

# Unstack 3D array into 2D slices
slices_back = unstack_slices(volume_3d, memory_type='torch')
```

## Documentation

Full documentation available at [arraybridge.readthedocs.io](https://arraybridge.readthedocs.io)

## Performance

arraybridge uses DLPack for zero-copy conversions when possible:

| Conversion | Method | Speed |
|------------|--------|-------|
| NumPy → PyTorch | `torch.from_numpy()` | Zero-copy |
| PyTorch → CuPy | DLPack | Zero-copy |
| CuPy → JAX | DLPack | Zero-copy |
| NumPy → CuPy | Copy | Fast |
| PyTorch → NumPy | `.numpy()` | Zero-copy (CPU) |

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## Credits

Developed by Tristan Simas as part of the OpenHCS project.
