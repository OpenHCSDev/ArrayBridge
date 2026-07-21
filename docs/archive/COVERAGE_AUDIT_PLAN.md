# Coverage Audit & Test Enhancement Plan

**Date:** November 2, 2025  
**Branch:** `dev/increase-test-coverage`  
**Current Coverage:** 34% (835 statements, 550 missed)  
**Target Coverage:** 60–80% (realistic after implementing high/medium-priority tests)

---

## Executive Summary

The codebase currently has 34% test coverage. This audit identifies the lowest-coverage modules and proposes a phased testing strategy to reach 60–80% coverage. Quick-win modules (e.g., `converters_registry.py`, `utils.py`) can be tackled first for fast coverage gains; harder modules (decorators, optional framework integrations) follow as needed.

---

## Current Coverage Snapshot

### Excellent coverage (≥97%)
- `src/arraybridge/__init__.py` — 100%
- `src/arraybridge/converters.py` — 100%
- `src/arraybridge/exceptions.py` — 100%
- `src/arraybridge/framework_ops.py` — 100%
- `src/arraybridge/types.py` — 97%

### Good coverage (80–96%)
- `src/arraybridge/converters_registry.py` — 80% (18 missed; lines: 33, 38, 43, 48, 77, 157–163, 175–182)

### Moderate coverage (20–79%)
- `src/arraybridge/utils.py` — 36% (72 missed)
- `src/arraybridge/decorators.py` — 31% (123 missed)
- `src/arraybridge/framework_config.py` — 20% (51 missed)

### Low coverage (14–19%)
- `src/arraybridge/oom_recovery.py` — 15% (58 missed)
- `src/arraybridge/dtype_scaling.py` — 14% (59 missed)
- `src/arraybridge/slice_processing.py` — 13% (20 missed)
- `src/arraybridge/stack_utils.py` — 13% (92 missed)

### Uncovered (0%)
- `src/arraybridge/gpu_cleanup.py` — 0% (56 missed)

---

## Prioritized Testing Strategy

### Phase 1: Quick Wins (Est. +6–12% coverage, 2–4 hours effort)

#### 1.1 `converters_registry.py` (80% → ~95%)
**Missing lines:** 33, 38, 43, 48, 77, 157–163, 175–182  
**Effort:** Low — small, focused module with clear API.

**Test targets:**
- Register a converter and verify it appears in registry
- Attempt to register duplicate/conflicting converters (error handling or overwrite)
- Test `get_converter()` / `find_converter()` success and failure paths
- Test error cases (missing converter, invalid arguments)
- Test any factory methods or initialization logic (lines 157–182 likely involve setup/cleanup)

**Estimated gain:** +10–15% (covers error branches and edge cases)

---

#### 1.2 `utils.py` (36% → ~55%)
**Missing ranges:** 103–108, 159–196, 226–247, 262–289, 313–347  
**Effort:** Low — independent utility functions, mostly pure Python.

**Test targets:**
- Shape/dtype validators: test with valid and invalid numpy arrays
- Numeric dtype checkers: test with int, float, complex, and non-numeric dtypes
- Array concatenation/stacking helpers (if any)
- Argument validation or preprocessing functions
- Test error conditions (mismatched shapes, wrong dtypes, None inputs)

**Estimated gain:** +10–15% (many small functions → many small tests)

---

#### 1.3 `slice_processing.py` (13% → ~60%)
**Missing lines:** 34–72 (mostly the body of slice functions)  
**Effort:** Low — narrow module, focused array slicing logic.

**Test targets:**
- Slice with positive, negative indices, steps
- Edge cases: empty slices, out-of-bounds, single element
- Test with different array shapes (1D, 2D, 3D)
- Test with different dtypes (int, float, complex)

**Estimated gain:** +30–40% (covers most of the module)

---

### Phase 2: Medium Effort (Est. +8–15% coverage, 4–8 hours effort)

#### 2.1 `stack_utils.py` (13% → ~50%)
**Missing ranges:** 37–41, 55–59, 74–76, 99–145, 169–242, 270–317  
**Effort:** Medium — array manipulation, needs fixture setup with numpy.

**Test targets:**
- Stack/concat operations: test axis parameter, multiple arrays, different shapes
- Split/reshape operations: test valid splits, error on mismatched shapes
- Error handling: incompatible shapes, invalid axes, None inputs
- Edge cases: single array, empty array, broadcasting

**Estimated gain:** +25–35% (covers primary logic and error paths)

---

#### 2.2 `oom_recovery.py` (15% → ~50%)
**Missing ranges:** 36–76, 90–122, 140–148  
**Effort:** Medium — requires monkeypatching to simulate memory errors and recovery.

**Test targets:**
- Decorate a function that raises `MemoryError` and assert retry logic kicks in
- Test configurable retry count, backoff, and eventual success
- Test that function succeeds on Nth retry, not earlier
- Test exception re-raised if retries exhausted
- Test with different exception types (should not retry non-OOM exceptions)

**Test approach:**
```python
def test_oom_recovery_retries(monkeypatch):
    from arraybridge import oom_recovery
    call_count = {"n": 0}
    def flaky_func():
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise MemoryError("OOM")
        return "success"
    
    wrapped = oom_recovery.retry_on_oom(flaky_func, max_retries=3)
    result = wrapped()
    assert result == "success"
    assert call_count["n"] == 3
```

**Estimated gain:** +25–35%

---

### Phase 3: Higher Effort (Est. +8–20% coverage, 8–16 hours effort)

#### 3.1 `decorators.py` (31% → ~65%)
**Missing ranges:** 51–59, 69–76, 95–98, 102–107, 111–119, 124–126, 139–156, 165–266, 275–323, 349–370  
**Effort:** High — many decorator variants, registry side-effects, framework-specific logic.

**Test targets:**
- Apply decorator and verify registration in `converters_registry`
- Test different decorator variants (if multiple exist)
- Test metadata/attribute attachment to decorated function
- Test error handling (invalid arguments, duplicate names)
- Test with different framework pairs (numpy↔torch, numpy↔jax, etc.)
- Test that decorator preserves function signature/docstring

**Test approach:**
```python
def test_converter_decorator_registers(monkeypatch):
    from arraybridge import decorators, converters_registry
    
    # clear registry for test isolation
    converters_registry._CONVERTERS = {}
    
    @decorators.converter("numpy→torch")
    def convert_np_to_torch(x):
        return x
    
    assert "numpy→torch" in converters_registry._CONVERTERS
    assert converters_registry._CONVERTERS["numpy→torch"] is convert_np_to_torch
```

**Estimated gain:** +20–30%

---

#### 3.2 `dtype_scaling.py` (14% → ~60%)
**Missing ranges:** 40–102, 107–146  
**Effort:** Medium–High — depends on framework dtypes; use numpy + monkeypatch for torch/jax.

**Test targets:**
- Scale int32 → int64, float32 → float64, etc.
- Test no-op when target equals source dtype
- Test error on non-numeric dtypes
- Test with numpy arrays and (via monkeypatch) torch tensors
- Test edge cases: NaN, inf, overflow/underflow

**Test approach:**
```python
import numpy as np
from arraybridge import dtype_scaling

def test_scale_int32_to_int64():
    a = np.array([1, 2, 3], dtype=np.int32)
    b = dtype_scaling.scale_dtype(a, target_dtype=np.int64)
    assert b.dtype == np.int64

def test_scale_noop_same_dtype():
    a = np.array([1.0], dtype=np.float32)
    b = dtype_scaling.scale_dtype(a, target_dtype=np.float32)
    assert b.dtype == np.float32
    assert np.array_equal(a, b)
```

**Estimated gain:** +30–40%

---

#### 3.3 `framework_config.py` (20% → ~60%)
**Missing ranges:** 28–39, 44–47, 53–62, 67–91, 96, 102–126, 131–132, 137  
**Effort:** Medium–High — config loading, environment variables, optional framework imports.

**Test targets:**
- Load config from env vars with different values
- Test config defaults vs. explicit values
- Test framework availability checks (use monkeypatch to hide/expose frameworks)
- Test error handling (missing required config, bad values)
- Test caching/singleton patterns if applicable

**Test approach:**
```python
import sys, types
from arraybridge import framework_config as fc

def test_load_config_from_env(monkeypatch):
    monkeypatch.setenv("ARRAYBRIDGE_GPU_ENABLED", "false")
    config = fc.load_config()
    assert config.gpu_enabled is False

def test_framework_detection(monkeypatch):
    # Mock torch as unavailable
    monkeypatch.setitem(sys.modules, 'torch', None)
    config = fc.load_config()
    assert config.has_torch is False
```

**Estimated gain:** +30–40%

---

#### 3.4 `gpu_cleanup.py` (0% → ~40%)
**Missing ranges:** 11–139 (entire module)  
**Effort:** High — likely heavy GPU API usage; heavy mocking required.

**Test targets:**
- Mock GPU cleanup APIs (torch.cuda.empty_cache, etc.)
- Test cleanup is called under expected conditions
- Test cleanup errors are handled gracefully
- Test with different frameworks (torch, cupy, etc. — all via monkeypatch)

**Test approach:**
```python
import sys, types
from unittest.mock import MagicMock

def test_gpu_cleanup_calls_torch_cuda(monkeypatch):
    # Create mock torch module
    mock_cuda = MagicMock()
    mock_torch = types.SimpleNamespace(cuda=mock_cuda)
    monkeypatch.setitem(sys.modules, 'torch', mock_torch)
    
    from arraybridge import gpu_cleanup
    gpu_cleanup.cleanup_gpu()
    mock_cuda.empty_cache.assert_called_once()
```

**Estimated gain:** +30–50% (covers entire module, though with mocks)

---

## Implementation Roadmap

### Week 1: Phase 1 (Quick Wins)
- **Monday–Tuesday:** Implement `converters_registry.py` tests (1–2 hours)
- **Wednesday:** Implement `utils.py` tests (2–3 hours)
- **Thursday:** Implement `slice_processing.py` tests (1–2 hours)
- **Coverage check:** Expected 40–46%

### Week 2: Phase 2 (Medium Effort)
- **Monday–Wednesday:** Implement `stack_utils.py` tests (3–4 hours)
- **Thursday–Friday:** Implement `oom_recovery.py` tests (2–3 hours)
- **Coverage check:** Expected 50–65%

### Week 3+: Phase 3 (Higher Effort)
- **Monday–Wednesday:** `decorators.py` tests (4–6 hours)
- **Wednesday–Thursday:** `dtype_scaling.py` tests (2–3 hours)
- **Friday:** `framework_config.py` tests (2–3 hours)
- **Following week:** `gpu_cleanup.py` tests (3–5 hours) — defer if time-constrained
- **Coverage check:** Expected 65–80%+

---

## Testing Infrastructure Notes

### Fixtures & Monkeypatching
- Create a **`conftest.py` fixture** to inject lightweight dummy frameworks (torch, jax, cupy) to avoid heavy imports:
  ```python
  import pytest, types, sys
  
  @pytest.fixture
  def mock_frameworks(monkeypatch):
      """Inject lightweight mock frameworks to avoid heavy optional dependencies."""
      dummy_torch = types.SimpleNamespace(
          cuda=types.SimpleNamespace(empty_cache=lambda: None),
          Tensor=type('Tensor', (), {})
      )
      monkeypatch.setitem(sys.modules, 'torch', dummy_torch)
      # Add other mocks as needed
      yield
  ```

### Test Organization
- Keep tests organized by module: `test_converters_registry.py`, `test_utils.py`, etc.
- Use `@pytest.mark.parametrize` for testing multiple inputs/scenarios in one test.
- Use `pytest.raises()` for error conditions.

### Coverage Check Command
```bash
source ../openhcs/.venv/bin/activate
PYTHONPATH=src python -m pytest --cov=arraybridge --cov-report=term-missing --cov-report=html
```

Generate an HTML report with `--cov-report=html` to visualize covered/uncovered lines in a browser.

---

## Success Criteria

| Phase | Target Coverage | Effort | Status |
|-------|-----------------|--------|--------|
| Current | 34% | — | ✓ Baseline |
| Phase 1 (Quick Wins) | 40–46% | 2–4h | Proposed |
| Phase 2 (Medium) | 50–65% | 4–8h | Proposed |
| Phase 3 (Higher) | 65–80%+ | 8–16h | Proposed |

---

## Decision Points for Review

1. **Priority order:** Should we follow Phase 1 → Phase 2 → Phase 3, or prioritize specific modules?
2. **Phase 3 scope:** Is `gpu_cleanup.py` (0%, heavy mocks) worth the effort, or should we defer it?
3. **Target coverage:** Is 65–80% the final goal, or do we aim higher (85%+)?
4. **Timeline:** Can this work be parallelized across multiple developers, or is it sequential?

---

## Appendix: Test Template Examples

### Example 1: Testing a Registry (converters_registry.py)
```python
import pytest
from arraybridge import converters_registry

def test_register_converter():
    def dummy_conv(x): return x
    converters_registry.register_converter("test_conv", dummy_conv)
    assert converters_registry.get_converter("test_conv") is dummy_conv

def test_get_converter_not_found():
    with pytest.raises(KeyError):
        converters_registry.get_converter("nonexistent")
```

### Example 2: Testing Array Utilities (utils.py)
```python
import numpy as np
import pytest
from arraybridge import utils

@pytest.mark.parametrize("arr,expected", [
    (np.array([1, 2, 3], dtype=np.int32), True),
    (np.array([1.0, 2.0], dtype=np.float32), True),
    (np.array(["a", "b"], dtype=object), False),
])
def test_is_numeric_dtype(arr, expected):
    assert utils.is_numeric_dtype(arr.dtype) == expected
```

### Example 3: Testing with Retry Logic (oom_recovery.py)
```python
import pytest
from arraybridge import oom_recovery

def test_retry_on_oom():
    call_count = {"n": 0}
    
    def flaky():
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise MemoryError("OOM")
        return "success"
    
    wrapped = oom_recovery.retry_on_oom(flaky, max_retries=3)
    result = wrapped()
    assert result == "success"
    assert call_count["n"] == 2
```

---

## References

- Current coverage report: `htmlcov/index.html` (generated after each test run with `--cov-report=html`)
- Pytest docs: https://docs.pytest.org/
- Coverage.py docs: https://coverage.readthedocs.io/

---

**Next Steps:**
- [ ] Review this plan and provide feedback on priority/timeline
- [ ] Approve Phase 1 modules for implementation
- [ ] Assign developers to specific modules (if parallelizing)
- [ ] Schedule weekly coverage check-ins
