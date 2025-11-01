# Testing Strategy for ArrayBridge

## Overview

ArrayBridge uses a multi-tier testing strategy to balance comprehensive coverage with CI efficiency.

## Test Categories

### 1. CPU Tests (Fast, Always Run)
- **NumPy tests**: Core functionality, always available
- **PyTorch CPU tests**: Conversion logic using PyTorch CPU-only version
- **Framework detection**: Tests that check framework availability
- **Type system tests**: MemoryType enum and constants

These tests run on every PR and push to main using standard GitHub runners.

### 2. GPU Tests (Slow, Optional)
- **CuPy tests**: Require CUDA and GPU hardware
- **PyTorch GPU tests**: GPU-specific operations
- **Performance tests**: GPU memory management and optimization

These tests run:
- Manually via workflow dispatch
- Optionally on a schedule (weekly)
- On self-hosted GPU runners or GitHub's beta GPU runners

## Running Tests Locally

### Basic Tests (CPU only)
```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all CPU-compatible tests
pytest

# Run with coverage
pytest --cov=arraybridge --cov-report=html
```

### With PyTorch
```bash
# Install PyTorch CPU version
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Run all tests including PyTorch
pytest

# Run only PyTorch tests
pytest -m torch
```

### With GPU Frameworks (requires CUDA)
```bash
# Install CuPy (requires CUDA)
pip install cupy-cuda12x  # Adjust for your CUDA version

# Run only GPU tests
pytest -m gpu

# Run only CuPy tests
pytest -m cupy
```

## Test Markers

Tests are marked with pytest markers for selective execution:

- `@pytest.mark.gpu` - Requires actual GPU hardware
- `@pytest.mark.cupy` - Requires CuPy (GPU-only framework)
- `@pytest.mark.torch` - Requires PyTorch (CPU or GPU)
- `@pytest.mark.tensorflow` - Requires TensorFlow
- `@pytest.mark.jax` - Requires JAX
- `@pytest.mark.pyclesperanto` - Requires pyclesperanto
- `@pytest.mark.slow` - Long-running tests

### Examples

```bash
# Skip GPU tests
pytest -m "not gpu"

# Run only framework-specific tests
pytest -m "torch or cupy"

# Skip slow tests
pytest -m "not slow"
```

## CI Workflows

### Main CI (`.github/workflows/ci.yml`)
- Runs on: Every PR and push to main
- Runners: Standard GitHub runners (ubuntu, windows, macos)
- Frameworks: NumPy (always) + PyTorch CPU (ubuntu only)
- Python versions: 3.10, 3.11, 3.12
- Purpose: Fast feedback, ensure basic functionality works

### GPU Tests (`.github/workflows/gpu-tests.yml`)
- Runs on: Manual trigger or weekly schedule
- Runners: Self-hosted GPU runners or GitHub beta GPU runners
- Frameworks: CuPy, PyTorch GPU
- Purpose: Validate GPU-specific functionality
- Non-blocking: Failures don't block PR merges

## Why This Strategy?

### Problem
- CuPy requires CUDA drivers and can't even be imported without GPU
- GPU runners are expensive and slow (long queue times)
- Most conversion logic doesn't actually need GPU to test
- PyTorch has excellent CPU-only support

### Solution
1. **Separate CPU and GPU tests**: Most tests work fine on CPU
2. **Use PyTorch CPU**: Tests conversion API without needing GPU
3. **Skip CuPy in standard CI**: Can't install without CUDA anyway
4. **Optional GPU workflow**: Run when needed, not on every commit
5. **Pytest markers**: Clear organization and selective execution

### Benefits
- ✅ Fast CI feedback (no GPU queue waits)
- ✅ Free standard runners for most tests
- ✅ Comprehensive coverage of conversion logic
- ✅ GPU tests available when needed
- ✅ Clear test organization with markers

## Adding New Tests

### For CPU-compatible tests
```python
def test_my_feature():
    """Test that works with NumPy or PyTorch CPU."""
    # No special markers needed
    pass
```

### For framework-specific tests
```python
@pytest.mark.torch
def test_torch_feature(torch_available):
    """Test that requires PyTorch."""
    if not torch_available:
        pytest.skip("PyTorch not available")
    
    import torch
    # Your test code
```

### For GPU-only tests
```python
@pytest.mark.gpu
@pytest.mark.cupy
def test_cupy_gpu_feature(cupy_available, gpu_available):
    """Test that requires CuPy and GPU."""
    if not cupy_available or not gpu_available:
        pytest.skip("CuPy or GPU not available")
    
    import cupy as cp
    # Your GPU test code
```

## Coverage Goals

- **Core modules**: 80%+ coverage
- **CPU-compatible code**: 90%+ coverage
- **GPU-specific code**: Best effort (tested manually or in GPU workflow)

## Future Improvements

1. Add mocking for GPU frameworks to test API without hardware
2. Expand GPU test suite as GPU runners become more accessible
3. Add integration tests with real scientific workflows
4. Performance benchmarks for conversion operations

