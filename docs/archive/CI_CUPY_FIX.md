# CI CuPy Import Fix

## Problem
The CI pipeline was failing because test code was using `@pytest.mark.skipif` decorators with `__import__()` calls that would fail at module import time if CuPy (and other GPU frameworks) were not installed. This caused CI failures on standard CPU-only runners.

### Error Pattern
```python
# ❌ BROKEN - Fails at module load time if cupy is not installed
@pytest.mark.skipif(not hasattr(__import__('cupy', fromlist=['']), 'cuda'), 
                   reason="CuPy CUDA not available")
def test_cupy_cleanup_with_gpu(self):
    import cupy as cp
    ...
```

When `tests/test_gpu_cleanup.py` was imported:
- The decorator would execute `__import__('cupy', ...)` immediately
- If CuPy wasn't installed, this would raise `ModuleNotFoundError`
- The test module would fail to load entirely
- CI would fail on ALL tests in that file

## Solution
Replace `@pytest.mark.skipif` with dynamic import checks using `pytest.importorskip()` inside the test function:

```python
# ✅ FIXED - Gracefully skips if cupy is not installed
def test_cupy_cleanup_with_gpu(self):
    """Test cupy cleanup when cupy and GPU are available."""
    cp = pytest.importorskip("cupy")  # Skip test if not available
    from arraybridge.gpu_cleanup import cleanup_cupy_gpu
    import unittest.mock
    ...
```

### Why This Works
- `pytest.importorskip()` is called at test execution time (not module load time)
- If the module is not available, it cleanly skips the test
- The test module can still be imported and other tests can run
- Graceful degradation instead of hard failures

## Changes Made

### 1. `tests/conftest.py`
Added helper functions (for future use, though not strictly needed now):
- `_module_available()` - Check if module is importable
- `_module_has_attribute()` - Check if module has attribute
- `_can_import_and_has_cuda()` - Check GPU framework availability

### 2. `tests/test_gpu_cleanup.py`
Replaced all problematic `@pytest.mark.skipif` decorators:

| Framework | Old Pattern | New Pattern |
|-----------|-----------|-----------|
| CuPy | `__import__('cupy', ...'cuda')` | `pytest.importorskip("cupy")` |
| PyTorch | `__import__('torch', ...'cuda')` | `pytest.importorskip("torch")` |
| TensorFlow | `__import__('tensorflow', ...'config')` | `pytest.importorskip("tensorflow")` |
| JAX | `__import__('jax', ...'numpy')` | `pytest.importorskip("jax")` |
| pyclesperanto | `__import__('pyclesperanto', ...)` | `pytest.importorskip("pyclesperanto")` |

## CI Impact

### CPU-Only Tests (GitHub Actions standard runners)
- ✅ All tests now pass
- ✅ GPU framework tests gracefully skipped
- ✅ Test module imports successfully
- ✅ Coverage reporting works

### GPU Tests (Kaggle free GPU runner)
- ✅ All GPU frameworks can be installed with `pip install -e ".[dev,gpu]"`
- ✅ Tests run normally when frameworks are available
- ✅ Automatic retry logic handles OOM errors

## Verification

Run tests locally:
```bash
# CPU-only (should skip GPU tests gracefully)
pytest tests/test_gpu_cleanup.py -v

# With GPU frameworks installed (should run GPU tests)
pip install cupy-cuda12x torch tensorflow jax
pytest tests/test_gpu_cleanup.py -v
```

Expected output for CPU-only:
```
test_cupy_cleanup_unavailable PASSED
test_cupy_cleanup_with_gpu SKIPPED (cupy not available)
test_torch_cleanup_unavailable PASSED
test_torch_cleanup_with_gpu SKIPPED (torch not available)
...
```

## Related Configuration

The CI workflows use:
- **CPU Tests**: `.github/workflows/ci.yml` - Standard GitHub Actions runners
  - Installs `.[dev]` dependencies only
  - GPU framework tests gracefully skipped
  
- **GPU Tests**: `.github/workflows/ci.yml` (gpu-test job) - Kaggle free GPU runner
  - Installs `.[dev,gpu]` dependencies (includes all GPU frameworks)
  - All GPU tests run with CUDA support
  - Non-blocking (marked `continue-on-error: true`)
