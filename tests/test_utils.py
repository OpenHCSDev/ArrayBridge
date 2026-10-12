"""Tests for arraybridge.utils module."""

import sys

import numpy as np
import pytest

from arraybridge.utils import optional_import


class TestOptionalImport:
    """Tests for optional_import function."""

    def test_import_existing_module(self):
        """Test importing an existing module."""
        np_module = optional_import("numpy")
        assert np_module is not None
        assert hasattr(np_module, "array")
        # Should be the real numpy module
        assert np_module.array is np.array

    def test_import_nonexistent_module_returns_none(self):
        assert optional_import("this_module_does_not_exist_12345") is None

    def test_broken_installed_module_import_is_not_hidden(self, monkeypatch):
        from arraybridge import utils

        def broken_import(name):
            raise ImportError(f"{name} binary initialization failed")

        monkeypatch.setattr(utils.importlib, "import_module", broken_import)

        with pytest.raises(ImportError, match="binary initialization failed"):
            optional_import("installed_but_broken")

    def test_missing_transitive_dependency_is_not_hidden(self, monkeypatch):
        from arraybridge import utils

        def broken_import(name):
            raise ModuleNotFoundError(
                "missing transitive dependency",
                name="dependency_of_installed_module",
            )

        monkeypatch.setattr(utils.importlib, "import_module", broken_import)

        with pytest.raises(ModuleNotFoundError, match="transitive dependency"):
            optional_import("installed_module")


class TestEnsureModule:
    """Tests for _ensure_module function."""

    def test_ensure_existing_module(self):
        """Test ensuring an existing module."""
        from arraybridge.utils import _ensure_module

        np_module = _ensure_module("numpy")
        assert np_module is not None
        assert hasattr(np_module, "array")

    def test_ensure_nonexistent_module_raises(self):
        """Test that ensuring non-existent module raises ImportError."""
        from arraybridge.utils import _ensure_module

        with pytest.raises(ImportError) as exc_info:
            _ensure_module("nonexistent_module_xyz")

        assert "required" in str(exc_info.value).lower()


class TestSupportsChecks:
    """Tests for CUDA and DLPack support check functions."""

    def test_supports_cuda_array_interface_numpy(self):
        """Test that NumPy arrays don't support CUDA array interface."""
        from arraybridge.utils import _supports_cuda_array_interface

        arr = np.array([1, 2, 3])
        assert not _supports_cuda_array_interface(arr)

    def test_supports_dlpack_numpy(self):
        """Test DLPack support for NumPy arrays.

        NumPy 2.0+ supports DLPack via __dlpack__ and __dlpack_device__ methods.
        Older versions do not support DLPack.
        """
        from arraybridge.utils import _supports_dlpack

        arr = np.array([1, 2, 3])
        # NumPy 2.0+ has DLPack support, older versions don't
        has_dlpack = hasattr(arr, "__dlpack__")
        assert _supports_dlpack(arr) == has_dlpack

    def test_supports_cuda_array_interface_object_without_it(self):
        """Test that regular objects don't support CUDA array interface."""
        from arraybridge.utils import _supports_cuda_array_interface

        obj = {"data": [1, 2, 3]}
        assert not _supports_cuda_array_interface(obj)

    def test_supports_dlpack_object_without_it(self):
        """Test that regular objects don't support DLPack."""
        from arraybridge.utils import _supports_dlpack

        obj = {"data": [1, 2, 3]}
        assert not _supports_dlpack(obj)


class TestEnsureModuleTensorFlowVersion:
    def test_ordinary_tensorflow_import_respects_supported_baseline(self, monkeypatch):
        import types

        mock_tf = types.SimpleNamespace(__version__="2.10.0")
        monkeypatch.setitem(sys.modules, "tensorflow", mock_tf)

        from arraybridge.utils import _ensure_module

        assert _ensure_module("tensorflow") is mock_tf
