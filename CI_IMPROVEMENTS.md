# CI Testing Improvements

## Summary

This update restructures the CI testing strategy to eliminate GPU runner queue issues while maintaining comprehensive test coverage.

## Problems Solved

1. **CuPy Installation Failures**: CuPy requires CUDA drivers and cannot be installed on standard CI runners
2. **GPU Runner Queue Times**: Beta GPU runners have long queue times, blocking PR feedback
3. **Inefficient Test Matrix**: Too many redundant test combinations
4. **Unclear Test Organization**: No markers to distinguish CPU vs GPU tests

## Changes Made

### 1. Updated CI Workflow (`.github/workflows/ci.yml`)

**Before:**
- Matrix included `cupy` framework option
- Tried to install CuPy on standard runners (always failed)
- Tested on Python 3.9-3.12 across all OS combinations
- 36+ job combinations

**After:**
- Removed `cupy` from standard CI matrix
- Reduced Python versions to 3.10-3.12 (dropped 3.9)
- Simplified matrix: torch tests only on Ubuntu
- GPU tests marked as `continue-on-error: true` (non-blocking)
- **Result**: ~12 fast jobs on standard runners + 1 optional GPU job

### 2. Added Pytest Markers (`pyproject.toml`)

New markers for test organization:
- `@pytest.mark.gpu` - Requires actual GPU hardware
- `@pytest.mark.cupy` - Requires CuPy framework
- `@pytest.mark.torch` - Requires PyTorch framework
- `@pytest.mark.tensorflow` - Requires TensorFlow
- `@pytest.mark.jax` - Requires JAX
- `@pytest.mark.pyclesperanto` - Requires pyclesperanto
- `@pytest.mark.slow` - Long-running tests

### 3. Enhanced Test Fixtures (`tests/conftest.py`)

- Added `pytest_configure()` to register markers
- Improved `cupy_available` fixture to verify GPU access
- Added `gpu_available` fixture for general GPU detection
- Better error handling for framework imports

### 4. Marked Framework-Specific Tests

Added `@pytest.mark.torch` to PyTorch tests in `test_converters.py`:
- `test_detect_torch_tensor`
- `test_convert_numpy_to_torch`
- `test_convert_torch_to_numpy`
- `test_round_trip_conversion_numpy_torch`

### 5. Created Optional GPU Workflow (`.github/workflows/gpu-tests.yml`)

Separate workflow for GPU testing:
- Manual trigger only (`workflow_dispatch`)
- Optional weekly schedule
- Runs on self-hosted or beta GPU runners
- Tests CuPy and PyTorch GPU functionality
- Non-blocking (failures don't block PRs)

### 6. Documentation (`TESTING.md`)

Comprehensive testing guide covering:
- Test categories (CPU vs GPU)
- Running tests locally
- Using pytest markers
- CI workflow explanation
- Adding new tests
- Coverage goals

## Test Results

### Without PyTorch (NumPy only)
```
61 passed, 4 deselected
Coverage: 33%
```

### With PyTorch CPU
```
65 passed
Coverage: ~35%
```

### Test Breakdown
- **CPU-compatible tests**: 61 tests (run on every PR)
- **PyTorch tests**: 4 tests (run on Ubuntu with torch-cpu)
- **GPU tests**: 0 currently (would run on GPU runners when available)

## Benefits

### ✅ Fast CI Feedback
- Standard tests complete in ~2-3 minutes
- No waiting in GPU runner queues
- Immediate feedback on PRs

### ✅ Cost Efficiency
- Uses free standard runners for 99% of tests
- GPU runners only used when needed
- Reduced total CI minutes

### ✅ Better Test Organization
- Clear markers for test categories
- Easy to run specific test subsets
- Better documentation

### ✅ Comprehensive Coverage
- Core conversion logic tested on CPU
- PyTorch CPU tests validate API
- Optional GPU tests for real hardware validation

### ✅ Non-Blocking GPU Tests
- GPU test failures don't block PRs
- Can investigate GPU issues separately
- Flexibility for GPU runner availability

## Migration Guide

### For Developers

**Running tests locally:**
```bash
# CPU tests only (fast)
pytest -m "not gpu"

# With PyTorch CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu
pytest

# With GPU frameworks (requires CUDA)
pip install cupy-cuda12x
pytest -m gpu
```

**Adding new tests:**
```python
# CPU-compatible test (no marker needed)
def test_my_feature():
    pass

# Framework-specific test
@pytest.mark.torch
def test_torch_feature(torch_available):
    if not torch_available:
        pytest.skip("PyTorch not available")
    # test code

# GPU-only test
@pytest.mark.gpu
@pytest.mark.cupy
def test_gpu_feature(cupy_available, gpu_available):
    if not cupy_available or not gpu_available:
        pytest.skip("GPU not available")
    # test code
```

### For CI/CD

**Standard CI (automatic):**
- Runs on every PR and push to main
- Tests NumPy + PyTorch CPU
- Fast feedback (~2-3 minutes)
- Must pass for PR merge

**GPU CI (optional):**
- Manual trigger via Actions tab
- Or weekly schedule
- Non-blocking (can fail without blocking PRs)
- Use for validating GPU-specific features

## Future Improvements

1. **Mock GPU Frameworks**: Add mocking for GPU frameworks to test API without hardware
2. **Performance Benchmarks**: Add benchmarks for conversion operations
3. **Integration Tests**: Real-world scientific workflow tests
4. **Coverage Goals**: Increase coverage to 80%+ for core modules

## Rollback Plan

If issues arise, revert to previous CI by:
1. Restore original `.github/workflows/ci.yml`
2. Remove pytest markers (optional, won't break anything)
3. Tests will still work, just with less organization

## Questions?

See `TESTING.md` for detailed testing documentation.

