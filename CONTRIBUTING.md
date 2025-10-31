# Contributing to arraybridge

Thank you for your interest in contributing to arraybridge! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards other contributors

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Finding Issues

- Check the [issue tracker](https://github.com/trissim/arraybridge/issues)
- Look for issues labeled `good first issue` or `help wanted`
- Feel free to create new issues for bugs or feature requests

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/arraybridge.git
cd arraybridge

# Add upstream remote
git remote add upstream https://github.com/trissim/arraybridge.git
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install package in editable mode with dev dependencies
pip install -e ".[dev]"

# Optional: Install specific framework for testing
pip install -e ".[torch]"    # PyTorch
pip install -e ".[cupy]"     # CuPy (requires CUDA)
pip install -e ".[tensorflow]"  # TensorFlow
pip install -e ".[jax]"      # JAX
pip install -e ".[all]"      # All frameworks
```

### 4. Verify Setup

```bash
# Run tests to verify setup
pytest

# Check code quality tools
black --version
ruff --version
mypy --version
```

## Making Changes

### 1. Create a Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
# Or for bug fixes
git checkout -b fix/issue-description
```

### 2. Make Your Changes

- Write clear, readable code
- Follow existing code style
- Add docstrings to all public functions/classes
- Update documentation as needed

### 3. Document Your Changes

#### Docstring Format

Use Google-style docstrings:

```python
def convert_memory(data: Any, source_type: str, target_type: str, gpu_id: int) -> Any:
    """
    Convert data between memory types.

    Args:
        data: The data to convert
        source_type: The source memory type (e.g., "numpy", "torch")
        target_type: The target memory type (e.g., "cupy", "jax")
        gpu_id: The target GPU device ID

    Returns:
        The converted data in the target memory type

    Raises:
        ValueError: If source_type or target_type is invalid
        MemoryConversionError: If conversion fails

    Example:
        ```python
        import numpy as np
        from arraybridge import convert_memory

        data = np.array([1, 2, 3])
        result = convert_memory(data, "numpy", "torch", gpu_id=0)
        ```
    """
```

## Testing

### Writing Tests

1. **Create test file**: `tests/test_<module>.py`
2. **Use pytest conventions**:
   - Test classes: `class TestFeatureName:`
   - Test functions: `def test_specific_behavior():`
3. **Use fixtures** from `conftest.py` when appropriate
4. **Test edge cases**: empty arrays, None values, invalid inputs

### Example Test

```python
import pytest
import numpy as np
from arraybridge import convert_memory

class TestConvertMemory:
    """Tests for convert_memory function."""

    def test_convert_numpy_to_numpy(self):
        """Test converting NumPy to NumPy (no-op)."""
        arr = np.array([1, 2, 3])
        result = convert_memory(arr, "numpy", "numpy", gpu_id=0)
        np.testing.assert_array_equal(result, arr)

    def test_invalid_source_type_raises_error(self):
        """Test that invalid source type raises ValueError."""
        arr = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            convert_memory(arr, "invalid", "numpy", gpu_id=0)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=arraybridge --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_converters.py

# Run specific test
pytest tests/test_converters.py::TestConvertMemory::test_convert_numpy_to_numpy

# Run tests matching pattern
pytest -k "numpy"

# Run in parallel (requires pytest-xdist)
pytest -n auto
```

### Framework-Specific Tests

Use availability fixtures for optional frameworks:

```python
def test_pytorch_feature(torch_available):
    if not torch_available:
        pytest.skip("PyTorch not available")

    import torch
    # Test PyTorch-specific code
```

### Coverage Guidelines

- Aim for >80% code coverage
- All new features must have tests
- Bug fixes should include regression tests
- Test both success and failure paths

## Code Quality

### Formatting with Black

```bash
# Format all code
black src/ tests/

# Check formatting without changes
black --check src/ tests/
```

### Linting with Ruff

```bash
# Check for issues
ruff check src/ tests/

# Auto-fix issues where possible
ruff check src/ tests/ --fix
```

### Type Checking with Mypy

```bash
# Run type checker
mypy src/ --ignore-missing-imports

# Check specific file
mypy src/arraybridge/converters.py
```

### Pre-commit Checklist

Before committing, ensure:

```bash
# 1. Format code
black src/ tests/

# 2. Lint code
ruff check src/ tests/ --fix

# 3. Run tests
pytest

# 4. Check coverage
pytest --cov=arraybridge --cov-report=term

# 5. Type check (optional but recommended)
mypy src/ --ignore-missing-imports
```

## Submitting Changes

### 1. Commit Your Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add feature: short description

Detailed explanation of changes:
- What was changed
- Why it was changed
- Any breaking changes

Fixes #123"
```

#### Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- First line: short summary (50 chars or less)
- Blank line after summary
- Detailed explanation if needed
- Reference issues/PRs: `Fixes #123`, `Closes #456`

### 2. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 3. Create Pull Request

1. Go to your fork on GitHub
2. Click "Pull Request"
3. Select your branch
4. Fill in the PR template:
   - **Title**: Clear, concise description
   - **Description**: What, why, and how
   - **Related issues**: Link to relevant issues
   - **Testing**: Describe how you tested
   - **Breaking changes**: List any breaking changes

### 4. PR Review Process

- Automated checks must pass (tests, linting, formatting)
- At least one maintainer approval required
- Address review feedback promptly
- Keep PR focused on single feature/fix
- Squash commits if requested

## Release Process

(For maintainers)

### 1. Update Version

Edit `pyproject.toml`:

```toml
[project]
version = "0.2.0"
```

### 2. Update Changelog

Add release notes to `CHANGELOG.md` (if exists) or create GitHub release notes.

### 3. Create Tag

```bash
# Ensure you're on main branch
git checkout main
git pull upstream main

# Create and push tag
git tag -a v0.2.0 -m "Release version 0.2.0"
git push upstream v0.2.0
```

### 4. Automated Publishing

GitHub Actions will automatically:
1. Build the package
2. Create GitHub release
3. Publish to PyPI

## Development Guidelines

### Code Style

- Follow PEP 8 (enforced by Black and Ruff)
- Use type hints where possible
- Maximum line length: 100 characters
- Use meaningful variable names

### Architecture Principles

1. **No Inferred Capabilities**: Explicitly check for framework features
2. **Fail Loudly**: Raise clear exceptions instead of silent failures
3. **Declarative Memory Types**: Use decorators and type hints
4. **DLPack First**: Prefer zero-copy conversions when available
5. **Framework Independence**: Don't assume any framework is installed

### Adding New Features

1. **Discuss first**: Open an issue to discuss major changes
2. **Start small**: Begin with MVP, iterate based on feedback
3. **Document**: Add docstrings, update README/docs
4. **Test thoroughly**: Include unit, integration, and edge case tests
5. **Maintain compatibility**: Avoid breaking changes when possible

### Adding Framework Support

To add support for a new framework:

1. Update `MemoryType` enum in `types.py`
2. Add framework config in `framework_config.py`
3. Implement converter in `conversion_helpers.py`
4. Add decorator support in `decorators.py`
5. Update documentation
6. Add tests with availability checks

## Getting Help

- **Questions**: Open a GitHub issue with `question` label
- **Bugs**: Open a GitHub issue with `bug` label
- **Features**: Open a GitHub issue with `enhancement` label
- **Chat**: (Add Discord/Slack link if available)

## Recognition

Contributors will be recognized in:
- GitHub contributors list
- Release notes (for significant contributions)
- Project README (for major features)

Thank you for contributing to arraybridge! 🎉
