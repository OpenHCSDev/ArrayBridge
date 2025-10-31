"""Tests for arraybridge.utils module."""

import pytest
import numpy as np
from arraybridge.utils import optional_import, _ModulePlaceholder


class TestOptionalImport:
    """Tests for optional_import function."""

    def test_import_existing_module(self):
        """Test importing an existing module."""
        np_module = optional_import("numpy")
        assert np_module is not None
        assert hasattr(np_module, "array")
        # Should be the real numpy module
        assert np_module.array is np.array

    def test_import_nonexistent_module(self):
        """Test importing a non-existent module returns placeholder."""
        fake_module = optional_import("this_module_does_not_exist_12345")
        assert fake_module is not None
        # Should be a placeholder
        assert isinstance(fake_module, _ModulePlaceholder)

    def test_placeholder_is_falsy(self):
        """Test that placeholder evaluates to False in boolean context."""
        fake_module = optional_import("nonexistent_module")
        assert not fake_module
        assert bool(fake_module) is False

    def test_placeholder_attribute_access(self):
        """Test that placeholder allows attribute access."""
        fake_module = optional_import("nonexistent_module")
        # Should not raise error
        attr = fake_module.some_attribute
        # Should return another placeholder
        assert isinstance(attr, _ModulePlaceholder)

    def test_placeholder_chained_attribute_access(self):
        """Test that placeholder allows chained attribute access."""
        fake_module = optional_import("nonexistent_module")
        # Should not raise error on chained access
        attr = fake_module.submodule.function.attribute
        assert isinstance(attr, _ModulePlaceholder)

    def test_placeholder_call_raises_error(self):
        """Test that calling a placeholder function raises ImportError."""
        fake_module = optional_import("nonexistent_module")
        with pytest.raises(ImportError) as exc_info:
            fake_module.some_function()

        assert "not available" in str(exc_info.value)
        assert "nonexistent_module" in str(exc_info.value)

    def test_placeholder_repr(self):
        """Test that placeholder has informative repr."""
        fake_module = optional_import("test_module")
        repr_str = repr(fake_module)
        assert "ModulePlaceholder" in repr_str
        assert "test_module" in repr_str


class TestModulePlaceholder:
    """Tests for _ModulePlaceholder class."""

    def test_placeholder_creation(self):
        """Test creating a placeholder."""
        placeholder = _ModulePlaceholder("test_module")
        assert placeholder._module_name == "test_module"

    def test_placeholder_bool(self):
        """Test placeholder boolean conversion."""
        placeholder = _ModulePlaceholder("test")
        assert not placeholder

    def test_placeholder_getattr(self):
        """Test placeholder attribute access."""
        placeholder = _ModulePlaceholder("test")
        attr = placeholder.attribute
        assert isinstance(attr, _ModulePlaceholder)
        assert attr._module_name == "test.attribute"

    def test_placeholder_call_error_message(self):
        """Test placeholder call error message."""
        placeholder = _ModulePlaceholder("my_module")
        with pytest.raises(ImportError) as exc_info:
            placeholder()

        error_msg = str(exc_info.value)
        assert "my_module" in error_msg
        assert "not available" in error_msg


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
        has_dlpack = hasattr(arr, '__dlpack__')
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
