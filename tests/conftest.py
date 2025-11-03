"""Pytest configuration and fixtures for arraybridge tests."""

import sys
import pytest
import numpy as np


# Helper functions for safe module checking
def _module_available(module_name):
    """Check if a module is available without triggering ImportError."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def _module_has_attribute(module_name, attribute):
    """Check if a module is available and has an attribute."""
    try:
        module = __import__(module_name)
        return hasattr(module, attribute)
    except ImportError:
        return False


def _can_import_and_has_cuda(module_name):
    """Check if module is available and has CUDA."""
    try:
        module = __import__(module_name)
        if module_name == "cupy":
            return hasattr(module, "cuda")
        elif module_name == "torch":
            return hasattr(module, "cuda")
        elif module_name == "tensorflow":
            return hasattr(module, "config")
        elif module_name == "jax":
            return hasattr(module, "numpy")
        elif module_name == "pyclesperanto":
            return hasattr(module, "get_device")
        return False
    except ImportError:
        return False


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "gpu: tests that require actual GPU hardware")
    config.addinivalue_line("markers", "cupy: tests that require CuPy")
    config.addinivalue_line("markers", "torch: tests that require PyTorch")
    config.addinivalue_line("markers", "tensorflow: tests that require TensorFlow")
    config.addinivalue_line("markers", "jax: tests that require JAX")
    config.addinivalue_line("markers", "pyclesperanto: tests that require pyclesperanto")
    config.addinivalue_line("markers", "slow: tests that take a long time to run")


@pytest.fixture
def sample_2d_array():
    """Create a sample 2D NumPy array for testing."""
    return np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)


@pytest.fixture
def sample_3d_array():
    """Create a sample 3D NumPy array for testing."""
    return np.random.rand(5, 10, 10).astype(np.float32)


@pytest.fixture
def sample_slices():
    """Create a list of 2D slices for testing."""
    return [np.random.rand(10, 10).astype(np.float32) for _ in range(5)]


@pytest.fixture
def sample_uint8_array():
    """Create a sample uint8 array for dtype testing."""
    return np.random.randint(0, 256, size=(10, 10), dtype=np.uint8)


@pytest.fixture
def sample_uint16_array():
    """Create a sample uint16 array for dtype testing."""
    return np.random.randint(0, 65536, size=(10, 10), dtype=np.uint16)


# Framework availability fixtures
@pytest.fixture(scope="session")
def torch_available():
    """Check if PyTorch is available."""
    try:
        import torch

        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def cupy_available():
    """Check if CuPy is available and has GPU access."""
    try:
        import cupy as cp

        # Try to create a small array to verify GPU access
        _ = cp.array([1, 2, 3])
        return True
    except (ImportError, Exception):
        # CuPy not installed or no GPU available
        return False


@pytest.fixture(scope="session")
def tensorflow_available():
    """Check if TensorFlow is available."""
    try:
        import tensorflow as tf

        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def jax_available():
    """Check if JAX is available."""
    try:
        import jax

        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def pyclesperanto_available():
    """Check if pyclesperanto is available."""
    try:
        import pyclesperanto_prototype

        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def gpu_available():
    """Check if a GPU is available (CUDA or similar)."""
    try:
        import torch

        if torch.cuda.is_available():
            return True
    except ImportError:
        pass

    try:
        import cupy as cp

        _ = cp.array([1])
        return True
    except (ImportError, Exception):
        pass

    return False
