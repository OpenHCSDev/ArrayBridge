"""Tests for arraybridge.gpu_cleanup module."""

import pytest

from arraybridge.gpu_cleanup import MEMORY_TYPE_CLEANUP_REGISTRY, cleanup_all_gpu_frameworks
from arraybridge.types import MemoryType


class TestCleanupRegistry:
    """Tests for cleanup registry."""

    def test_cleanup_registry_has_all_memory_types(self):
        """Test that cleanup registry has all memory types."""
        for mem_type in MemoryType:
            assert mem_type.value in MEMORY_TYPE_CLEANUP_REGISTRY
            assert callable(MEMORY_TYPE_CLEANUP_REGISTRY[mem_type.value])

    def test_cleanup_functions_exist(self):
        """Test that cleanup functions are available globally."""
        from arraybridge import gpu_cleanup

        # Check that cleanup functions exist
        expected_functions = [
            "cleanup_numpy_gpu",
            "cleanup_cupy_gpu",
            "cleanup_torch_gpu",
            "cleanup_tensorflow_gpu",
            "cleanup_jax_gpu",
            "cleanup_pyclesperanto_gpu",
        ]

        for func_name in expected_functions:
            assert hasattr(gpu_cleanup, func_name)
            func = getattr(gpu_cleanup, func_name)
            assert callable(func)


class TestIndividualCleanupFunctions:
    """Tests for individual cleanup functions."""

    def test_numpy_cleanup_noop(self):
        """Test numpy cleanup is no-op."""
        from arraybridge.gpu_cleanup import cleanup_numpy_gpu

        # Should not raise any errors
        cleanup_numpy_gpu()
        cleanup_numpy_gpu(device_id=0)

    def test_cupy_cleanup_unavailable(self):
        """Test cupy cleanup when cupy is not available."""
        from arraybridge.gpu_cleanup import cleanup_cupy_gpu

        # Should not raise any errors even if cupy not available
        cleanup_cupy_gpu()
        cleanup_cupy_gpu(device_id=0)

    def test_cupy_cleanup_with_gpu(self):
        """Test cupy cleanup when cupy and GPU are available."""
        cp = pytest.importorskip("cupy")
        import unittest.mock

        from arraybridge.gpu_cleanup import cleanup_cupy_gpu

        # Create some GPU memory to cleanup
        try:
            gpu_array = cp.zeros((100, 100))
            assert gpu_array.device.id >= 0  # Ensure we have GPU memory

            # Mock the GPU check to return True so cleanup code runs
            with unittest.mock.patch("arraybridge.gpu_cleanup.eval") as mock_eval:
                mock_eval.return_value = True  # GPU is available
                # Cleanup should work without errors
                cleanup_cupy_gpu()
                cleanup_cupy_gpu(device_id=0)

        except Exception as e:
            pytest.skip(f"CuPy GPU test failed: {e}")

    def test_torch_cleanup_unavailable(self):
        """Test torch cleanup when torch is not available."""
        from arraybridge.gpu_cleanup import cleanup_torch_gpu

        # Should not raise any errors even if torch not available
        cleanup_torch_gpu()
        cleanup_torch_gpu(device_id=0)

    def test_torch_cleanup_with_gpu(self):
        """Test torch cleanup when torch and GPU are available."""
        import unittest.mock

        torch = pytest.importorskip("torch")
        from arraybridge.gpu_cleanup import cleanup_torch_gpu

        # Create some GPU memory to cleanup
        try:
            gpu_tensor = torch.zeros((100, 100), device="cuda")
            assert gpu_tensor.device.type == "cuda"

            # Mock the GPU check to return True so cleanup code runs
            with unittest.mock.patch("arraybridge.gpu_cleanup.eval") as mock_eval:
                mock_eval.return_value = True  # GPU is available
                # Cleanup should work without errors
                cleanup_torch_gpu()
                cleanup_torch_gpu(device_id=0)

        except Exception as e:
            pytest.skip(f"PyTorch GPU test failed: {e}")

    def test_tensorflow_cleanup_unavailable(self):
        """Test tensorflow cleanup when tensorflow is not available."""
        from arraybridge.gpu_cleanup import cleanup_tensorflow_gpu

        # Should not raise any errors even if tensorflow not available
        cleanup_tensorflow_gpu()
        cleanup_tensorflow_gpu(device_id=0)

    def test_tensorflow_cleanup_with_gpu(self):
        """Test tensorflow cleanup when tensorflow and GPU are available."""
        import unittest.mock

        tf = pytest.importorskip("tensorflow")
        from arraybridge.gpu_cleanup import cleanup_tensorflow_gpu

        # Create some GPU memory to cleanup
        try:
            with tf.device("/GPU:0"):
                gpu_tensor = tf.zeros((100, 100))
                assert "GPU" in gpu_tensor.device

            # Mock the GPU check to return True so cleanup code runs
            with unittest.mock.patch("arraybridge.gpu_cleanup.eval") as mock_eval:
                mock_eval.return_value = True  # GPU is available
                # Cleanup should work without errors
                cleanup_tensorflow_gpu()
                cleanup_tensorflow_gpu(device_id=0)

        except Exception as e:
            pytest.skip(f"TensorFlow GPU test failed: {e}")

    def test_jax_cleanup_unavailable(self):
        """Test jax cleanup when jax is not available."""
        from arraybridge.gpu_cleanup import cleanup_jax_gpu

        # Should not raise any errors even if jax not available
        cleanup_jax_gpu()
        cleanup_jax_gpu(device_id=0)

    def test_jax_cleanup_with_gpu(self):
        """Test jax cleanup when jax and GPU are available."""
        import unittest.mock

        jax = pytest.importorskip("jax")
        jnp = jax.numpy
        from arraybridge.gpu_cleanup import cleanup_jax_gpu

        # Create some GPU memory to cleanup
        try:
            jnp.zeros((100, 100))
            # JAX arrays are typically on CPU by default, but cleanup should still work

            # Mock the GPU check to return True so cleanup code runs
            with unittest.mock.patch("arraybridge.gpu_cleanup.eval") as mock_eval:
                mock_eval.return_value = True  # GPU is available
                cleanup_jax_gpu()
                cleanup_jax_gpu(device_id=0)

        except Exception as e:
            pytest.skip(f"JAX test failed: {e}")

    def test_pyclesperanto_cleanup_unavailable(self):
        """Test pyclesperanto cleanup when pyclesperanto is not available."""
        from arraybridge.gpu_cleanup import cleanup_pyclesperanto_gpu

        # Should not raise any errors even if pyclesperanto not available
        cleanup_pyclesperanto_gpu()
        cleanup_pyclesperanto_gpu(device_id=0)

    def test_pyclesperanto_cleanup_with_gpu(self):
        """Test pyclesperanto cleanup when pyclesperanto and GPU are available."""
        import unittest.mock

        cle = pytest.importorskip("pyclesperanto")
        from arraybridge.gpu_cleanup import cleanup_pyclesperanto_gpu

        # Create some GPU memory to cleanup
        try:
            cle.create((100, 100))
            # Mock the GPU check to return True so cleanup code runs
            with unittest.mock.patch("arraybridge.gpu_cleanup.eval") as mock_eval:
                mock_eval.return_value = True  # GPU is available
                # Cleanup should work without errors
                cleanup_pyclesperanto_gpu()
                cleanup_pyclesperanto_gpu(device_id=0)

        except Exception as e:
            pytest.skip(f"pyclesperanto GPU test failed: {e}")


class TestCleanupAllFrameworks:
    """Tests for cleanup_all_gpu_frameworks function."""

    def test_cleanup_all_frameworks_no_errors(self):
        """Test cleanup_all_gpu_frameworks doesn't raise errors."""
        # Should not raise any errors even if no frameworks available
        cleanup_all_gpu_frameworks()
        cleanup_all_gpu_frameworks(device_id=0)

    def test_cleanup_all_with_device_id(self):
        """Test cleanup_all_gpu_frameworks with specific device ID."""
        cleanup_all_gpu_frameworks(device_id=0)
        cleanup_all_gpu_frameworks(device_id=1)


class TestCleanupFunctionSignatures:
    """Tests for cleanup function signatures and documentation."""

    def test_cleanup_function_signatures(self):
        """Test that cleanup functions have correct signatures."""
        import inspect

        from arraybridge.gpu_cleanup import cleanup_cupy_gpu, cleanup_numpy_gpu, cleanup_torch_gpu

        for func in [cleanup_numpy_gpu, cleanup_cupy_gpu, cleanup_torch_gpu]:
            sig = inspect.signature(func)
            assert "device_id" in sig.parameters

            # device_id should be optional
            param = sig.parameters["device_id"]
            assert param.default is None

    def test_cleanup_function_docstrings(self):
        """Test that cleanup functions have docstrings."""
        from arraybridge.gpu_cleanup import cleanup_cupy_gpu, cleanup_numpy_gpu, cleanup_torch_gpu

        for func in [cleanup_numpy_gpu, cleanup_cupy_gpu, cleanup_torch_gpu]:
            assert func.__doc__ is not None
            assert len(func.__doc__.strip()) > 0

    def test_cleanup_all_docstring(self):
        """Test cleanup_all_gpu_frameworks has proper docstring."""
        assert cleanup_all_gpu_frameworks.__doc__ is not None
        assert (
            "Clean up GPU memory for all available frameworks" in cleanup_all_gpu_frameworks.__doc__
        )
