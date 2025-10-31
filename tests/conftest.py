"""Pytest configuration and fixtures for arraybridge tests."""

import pytest
import numpy as np


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
    """Check if CuPy is available."""
    try:
        import cupy
        return True
    except ImportError:
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
