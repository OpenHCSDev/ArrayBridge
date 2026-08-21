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


class TestDeviceOperations:
    """Tests for device-related utility functions."""

    def test_get_device_id_numpy(self):
        """Test getting device ID for NumPy arrays."""
        import numpy as np

        from arraybridge.utils import _get_device_id

        arr = np.array([1, 2, 3])
        device_id = _get_device_id(arr, "numpy")
        assert device_id is None  # NumPy is CPU-only

    def test_set_device_numpy(self):
        """Test setting device for NumPy (should be no-op)."""
        from arraybridge.utils import _set_device

        # Should not raise
        _set_device("numpy", 0)

    def test_move_to_device_numpy(self):
        """Test moving NumPy array to device (should return same array)."""
        import numpy as np

        from arraybridge.utils import _move_to_device

        arr = np.array([1, 2, 3])
        result = _move_to_device(arr, "numpy", 0)
        assert result is arr  # Should return same object

    @pytest.mark.parametrize("device_id", [0, 1, 2])
    def test_set_device_torch_mock(self, device_id, monkeypatch):
        """Test setting device for torch with mock."""
        import types

        selected = []
        mock_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 3,
                set_device=selected.append,
            )
        )
        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        from arraybridge.utils import _set_device

        _set_device("torch", device_id)
        assert selected == [device_id]

    def test_get_device_id_torch_mock(self, monkeypatch):
        """Test getting device ID for torch tensor with mock."""
        import types

        mock_device = types.SimpleNamespace(index=1)
        mock_tensor = types.SimpleNamespace(is_cuda=True, device=mock_device)
        mock_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(current_device=lambda: 1))
        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        from arraybridge.utils import _get_device_id

        device_id = _get_device_id(mock_tensor, "torch")
        assert device_id == 1

    def test_move_to_device_torch_mock(self, monkeypatch):
        """Test moving torch tensor to device with mock."""
        import types

        mock_tensor = types.SimpleNamespace(
            is_cuda=True, device=types.SimpleNamespace(index=0), to=lambda device: "moved_tensor"
        )

        class DeviceScope:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        mock_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 2,
                device=lambda device_id: DeviceScope(),
            )
        )
        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        from arraybridge.utils import _move_to_device

        assert _move_to_device(mock_tensor, "torch", 1) == "moved_tensor"


class TestEnsureModuleTensorFlowVersion:
    def test_ordinary_tensorflow_import_respects_supported_baseline(self, monkeypatch):
        import types

        mock_tf = types.SimpleNamespace(__version__="2.10.0")
        monkeypatch.setitem(sys.modules, "tensorflow", mock_tf)

        from arraybridge.utils import _ensure_module

        assert _ensure_module("tensorflow") is mock_tf


class TestGetDeviceIdCallableHandler:
    """Tests for _get_device_id with callable handlers."""

    def test_get_device_id_with_callable_handler(self, monkeypatch):
        """Test _get_device_id with a callable handler (pyclesperanto)."""
        import types

        from arraybridge.utils import _get_device_id

        mock_cle = types.SimpleNamespace(
            get_device=lambda: "gpu1",
            list_available_devices=lambda device_type=None: ["gpu0", "gpu1"],
        )
        monkeypatch.setitem(sys.modules, "pyclesperanto", mock_cle)

        # Create mock data
        mock_data = types.SimpleNamespace()

        assert _get_device_id(mock_data, "pyclesperanto") == 1

    def test_get_device_id_fallback_on_error(self, monkeypatch):
        """Invalid GPU objects fail rather than inventing a fallback device."""
        import types

        from arraybridge.exceptions import MemoryConversionError
        from arraybridge.utils import _get_device_id

        # Create a mock torch tensor that will fail device ID extraction
        mock_tensor = types.SimpleNamespace()  # Missing device attribute
        mock_torch = types.SimpleNamespace()
        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        with pytest.raises(MemoryConversionError, match="device_identification"):
            _get_device_id(mock_tensor, "torch")
