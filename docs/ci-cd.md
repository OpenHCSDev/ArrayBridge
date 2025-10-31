# CI/CD Documentation

## Overview

arraybridge uses GitHub Actions for continuous integration and deployment. The CI/CD pipeline ensures code quality, runs comprehensive tests across multiple environments, and automates package publishing.

## Workflows

### 1. CI Workflow (`.github/workflows/ci.yml`)

The CI workflow runs on every push to `main`/`master` branches and on all pull requests.

#### Test Matrix

The test suite runs across:
- **Python versions**: 3.9, 3.10, 3.11, 3.12
- **Operating systems**: Ubuntu, Windows, macOS
- **Framework combinations**:
  - `none`: Base NumPy only
  - `torch`: NumPy + PyTorch (CPU)
  - `cupy`: NumPy + CuPy (limited to Ubuntu)

#### Jobs

**1. Test Job**
- Runs pytest with coverage across the matrix
- Installs framework-specific dependencies
- Generates coverage reports (XML, HTML, terminal)
- Uploads coverage to Codecov (Ubuntu + Python 3.12 + torch only)

**2. Code Quality Job**
- **Ruff**: Fast Python linter for code quality
- **Black**: Code formatting check
- **Mypy**: Static type checking (continues on error for scientific code flexibility)

#### Running Locally

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=arraybridge --cov-report=html --cov-report=term

# Code quality checks
ruff check src/
black --check src/
mypy src/ --ignore-missing-imports
```

### 2. Publish Workflow (`.github/workflows/publish.yml`)

Automatically publishes to PyPI when a version tag is pushed.

#### Trigger

Push a tag starting with `v`:
```bash
git tag v0.1.0
git push origin v0.1.0
```

#### Steps

1. **Build**: Creates source distribution and wheel
2. **GitHub Release**: Creates release with built artifacts
3. **PyPI Upload**: Publishes to PyPI using `PYPI_API_TOKEN` secret

#### Prerequisites

Set up PyPI API token in repository secrets:
1. Generate token at https://pypi.org/manage/account/token/
2. Add as `PYPI_API_TOKEN` in repository settings → Secrets and variables → Actions

## Test Configuration

### pytest Configuration (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "--cov=arraybridge --cov-report=term-missing --cov-report=html"
```

### Coverage

- **Target**: Aim for >80% code coverage
- **Reports**: HTML reports generated in `htmlcov/`
- **CI Integration**: Codecov for tracking coverage over time

## Code Quality Standards

### Ruff Configuration

```toml
[tool.ruff]
line-length = 100
target-version = "py39"
select = ["E", "F", "I", "N", "W", "UP"]
```

Checks:
- **E**: Pycodestyle errors
- **F**: Pyflakes (undefined names, unused imports)
- **I**: Import sorting (isort)
- **N**: PEP 8 naming conventions
- **W**: Pycodestyle warnings
- **UP**: Pyupgrade (modern Python syntax)

### Black Configuration

```toml
[tool.black]
line-length = 100
target-version = ["py39", "py310", "py311", "py312"]
```

### Mypy Configuration

```toml
[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false  # Scientific code flexibility
```

## Local Development Workflow

### 1. Setup

```bash
# Clone repository
git clone https://github.com/trissim/arraybridge.git
cd arraybridge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Optional: Install framework dependencies
pip install -e ".[torch]"  # For PyTorch tests
```

### 2. Development Cycle

```bash
# 1. Make changes
# 2. Format code
black src/ tests/

# 3. Check code quality
ruff check src/ tests/

# 4. Run tests
pytest

# 5. Check coverage
pytest --cov=arraybridge --cov-report=html
# Open htmlcov/index.html in browser

# 6. Type check
mypy src/ --ignore-missing-imports
```

### 3. Pre-commit Checks

Before committing, ensure all checks pass:

```bash
# Format
black src/ tests/

# Lint
ruff check src/ tests/ --fix

# Test
pytest

# Type check
mypy src/
```

## Writing Tests

### Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── test_types.py            # Type system tests
├── test_exceptions.py       # Exception tests
├── test_utils.py            # Utility function tests
├── test_converters.py       # Converter tests
└── test_integration.py      # Integration tests
```

### Test Fixtures

Common fixtures in `conftest.py`:
- `sample_2d_array`: 2D NumPy array
- `sample_3d_array`: 3D NumPy array
- `sample_slices`: List of 2D slices
- `torch_available`: PyTorch availability check
- `cupy_available`: CuPy availability check

### Writing Framework-Specific Tests

Use framework availability fixtures:

```python
def test_pytorch_conversion(torch_available):
    if not torch_available:
        pytest.skip("PyTorch not available")

    import torch
    # Test PyTorch-specific functionality
```

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Coverage Guidelines

- Aim for >80% overall coverage
- All public APIs should have tests
- Edge cases and error conditions must be tested
- Framework-specific code can use conditional skips

## Continuous Integration Details

### Caching

The CI uses pip caching to speed up dependency installation:

```yaml
- name: Setup Python
  uses: actions/setup-python@v5
  with:
    python-version: ${{ matrix.python-version }}
    cache: 'pip'  # Automatically caches pip dependencies
```

### Matrix Strategy

```yaml
strategy:
  fail-fast: false  # Continue running other jobs if one fails
  matrix:
    python-version: ["3.9", "3.10", "3.11", "3.12"]
    os: [ubuntu-latest, windows-latest, macos-latest]
    framework: [none, torch, cupy]
```

### Conditional Steps

Framework installations are conditional:

```yaml
- name: Install PyTorch (CPU)
  if: matrix.framework == 'torch'
  run: pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Troubleshooting

### Common Issues

**1. Tests fail locally but pass in CI**
- Check Python version matches CI
- Ensure clean virtual environment
- Check for environment-specific dependencies

**2. Coverage drops unexpectedly**
- Run `pytest --cov=arraybridge --cov-report=html`
- Open `htmlcov/index.html` to see uncovered lines
- Add tests for uncovered code paths

**3. Black formatting fails**
- Run `black src/ tests/` to auto-format
- Check `.black` configuration in `pyproject.toml`

**4. Ruff errors**
- Run `ruff check src/ --fix` to auto-fix
- Some errors require manual fixes
- Check `pyproject.toml` for configured rules

**5. Import errors in tests**
- Ensure package is installed: `pip install -e .`
- Check `PYTHONPATH` if running tests directly
- Use `pytest` command, not `python -m pytest`

## Performance Optimization

### Test Parallelization

Run tests in parallel for faster execution:

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
pytest -n auto
```

### Selective Testing

Run specific test files or functions:

```bash
# Single file
pytest tests/test_converters.py

# Single test function
pytest tests/test_converters.py::TestConvertMemory::test_convert_numpy_to_numpy

# Tests matching pattern
pytest -k "numpy"
```

## Best Practices

1. **Write tests first** (TDD) when fixing bugs
2. **Keep tests fast** - use mocks for slow operations
3. **Test edge cases** - empty arrays, single elements, large arrays
4. **Use descriptive test names** - test name should describe what's being tested
5. **One assertion per test** when possible
6. **Use fixtures** for common setup
7. **Skip framework-specific tests** when framework not available
8. **Document complex tests** with docstrings

## References

- [pytest documentation](https://docs.pytest.org/)
- [GitHub Actions documentation](https://docs.github.com/en/actions)
- [Codecov documentation](https://docs.codecov.com/)
- [Ruff documentation](https://docs.astral.sh/ruff/)
- [Black documentation](https://black.readthedocs.io/)
