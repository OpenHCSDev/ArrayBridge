# Adding New Frameworks to arraybridge

With the metaclass-registry integration, adding a new framework is now simpler and requires no manual wiring.

## Quick Overview

To add a new framework (e.g., MXNet), you need to:

1. Add the new memory type to the `MemoryType` enum in `types.py`
2. Add framework configuration to `_FRAMEWORK_CONFIG` in `framework_config.py`
3. The converter automatically registers itself - no manual registration needed!

## Step-by-Step Guide

### Step 1: Add to MemoryType Enum

Edit `src/arraybridge/types.py`:

```python
class MemoryType(Enum):
    """Enum representing different array/tensor framework types."""
    
    NUMPY = "numpy"
    CUPY = "cupy"
    TORCH = "torch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    PYCLESPERANTO = "pyclesperanto"
    MXNET = "mxnet"  # <-- Add your new framework
```

### Step 2: Add Framework Configuration

Edit `src/arraybridge/framework_config.py` and add a new entry to `_FRAMEWORK_CONFIG`:

```python
_FRAMEWORK_CONFIG = {
    # ... existing configurations ...
    
    MemoryType.MXNET: {
        # Metadata
        'import_name': 'mxnet',
        'display_name': 'MXNet',
        'is_gpu': True,
        
        # Conversion operations - these define the converter methods
        'conversion_ops': {
            'to_numpy': 'data.asnumpy()',  # How to convert to numpy
            'from_numpy': '{mod}.nd.array(data, ctx={mod}.gpu(gpu_id))',  # How to create from numpy
            'from_dlpack': '{mod}.nd.from_dlpack(data)',  # DLPack support (if available)
            'move_to_device': 'data.as_in_context({mod}.gpu(device_id))',  # Move between devices
        },
        
        # Device operations (optional)
        'get_device_id': 'data.context.device_id',
        'set_device': None,
        
        # Other configuration...
        'supports_dlpack': True,
        'validate_dlpack': None,
        
        # ... add other required fields based on existing frameworks
    }
}
```

### Step 3: That's It!

The converter class is automatically created and registered when the module loads. You can verify it works:

```python
from arraybridge.converters_registry import ConverterBase, get_converter
from arraybridge.types import MemoryType

# Check that it's registered
print(sorted(ConverterBase.__registry__.keys()))
# Output: ['cupy', 'jax', 'mxnet', 'numpy', 'pyclesperanto', 'tensorflow', 'torch']

# Get the converter
mxnet_converter = get_converter("mxnet")
print(mxnet_converter.memory_type)  # Output: 'mxnet'

# Use via MemoryType enum
converter = MemoryType.MXNET.converter
```

## What Happens Behind the Scenes

1. **Auto-generation**: A `MxnetConverter` class is created dynamically with methods from `conversion_ops`
2. **Auto-registration**: The metaclass `AutoRegisterMeta` automatically registers it in `ConverterBase.__registry__`
3. **Auto-validation**: Module load validates that all `MemoryType` values have registered converters
4. **Auto-methods**: The converter automatically gets `to_X()` methods for all other frameworks

## Benefits of This Approach

### Before (Manual Wiring - Old System)
```python
# Had to manually create converter class with all methods
class MxnetConverter(MemoryTypeConverter):
    def to_numpy(self, data, gpu_id):
        return data.asnumpy()

    def from_numpy(self, data, gpu_id):
        return mxnet.nd.array(data)

    def to_torch(self, data, gpu_id):
        raise NotImplementedError

    def to_cupy(self, data, gpu_id):
        raise NotImplementedError

# Had to manually register
_CONVERTERS[MemoryType.MXNET] = MxnetConverter()
```

### After (Auto-registration - New System)
```python
# Add one enum member and one config entry; everything else is automatic.
class MemoryType(Enum):
    # Existing members omitted.
    MXNET = "mxnet"

_FRAMEWORK_CONFIG[MemoryType.MXNET] = {
    'conversion_ops': {
        'to_numpy': 'data.asnumpy()',
        # ...
    }
}
```

## Framework Configuration Reference

Required fields in `conversion_ops`:
- `to_numpy`: Expression to convert framework data to numpy
- `from_numpy`: Expression to create framework data from numpy
- `from_dlpack`: Expression for DLPack conversion (or `None`)
- `move_to_device`: Expression to move data between devices

Available template variables in expressions:
- `{mod}`: The imported module (e.g., `mxnet`)
- `data`: The input data
- `gpu_id` / `device_id`: Target device ID

## Testing Your New Framework

Add tests in `tests/test_converters.py`:

```python
@pytest.mark.mxnet
def test_convert_numpy_to_mxnet(self, mxnet_available):
    """Test converting NumPy to MXNet."""
    if not mxnet_available:
        pytest.skip("MXNet not available")
    
    import mxnet as mx
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = convert_memory(arr, source_type="numpy", target_type="mxnet", gpu_id=0)
    
    assert isinstance(result, mx.nd.NDArray)
    np.testing.assert_array_almost_equal(result.asnumpy(), arr)
```

## Common Patterns

### GPU Framework with DLPack
```python
conversion_ops = {
    'to_numpy': 'data.cpu().numpy()',
    'from_numpy': '{mod}.from_numpy(data).to(device=gpu_id)',
    'from_dlpack': '{mod}.from_dlpack(data)',
    'move_to_device': 'data.to(device=device_id)',
}
```

### CPU-only Framework
```python
conversion_ops = {
    'to_numpy': 'np.array(data)',
    'from_numpy': '{mod}.array(data)',
    'from_dlpack': None,  # Not supported
    'move_to_device': 'data',  # No-op for CPU
}
```

### Complex Operations with Helpers
If you need complex logic, define a helper function in `framework_config.py`:

```python
def _mxnet_special_conversion(data, gpu_id, mod):
    # Complex logic here
    return result

_FRAMEWORK_CONFIG[MemoryType.MXNET] = {
    'conversion_ops': {
        'from_numpy': _mxnet_special_conversion,  # Use callable instead of string
        # ...
    }
}
```

## Troubleshooting

### Converter not registered
Make sure:
1. You added the framework to `MemoryType` enum
2. You added configuration to `_FRAMEWORK_CONFIG`
3. The key in `_FRAMEWORK_CONFIG` matches the `MemoryType` enum value

### Import errors
If you get import errors, check:
1. The `import_name` matches the actual package name
2. The `conversion_ops` expressions use correct module syntax

### Validation errors
Run this to check registration:
```python
from arraybridge.converters_registry import _validate_registry
_validate_registry()  # Raises RuntimeError if validation fails
```
