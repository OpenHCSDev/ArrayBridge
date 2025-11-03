"""Tests for arraybridge.exceptions module."""

import pytest

from arraybridge.exceptions import MemoryConversionError


class TestMemoryConversionError:
    """Tests for MemoryConversionError exception."""

    def test_basic_exception_creation(self):
        """Test creating a MemoryConversionError."""
        error = MemoryConversionError(
            source_type="numpy",
            target_type="torch",
            method="dlpack",
            reason="Framework not installed",
        )

        assert error.source_type == "numpy"
        assert error.target_type == "torch"
        assert error.method == "dlpack"
        assert error.reason == "Framework not installed"

    def test_exception_message_format(self):
        """Test that exception message is properly formatted."""
        error = MemoryConversionError(
            source_type="numpy",
            target_type="cupy",
            method="array_interface",
            reason="CUDA not available",
        )

        error_message = str(error)
        assert "numpy" in error_message
        assert "cupy" in error_message
        assert "array_interface" in error_message
        assert "CUDA not available" in error_message

    def test_exception_can_be_raised(self):
        """Test that MemoryConversionError can be raised and caught."""
        with pytest.raises(MemoryConversionError) as exc_info:
            raise MemoryConversionError(
                source_type="torch",
                target_type="tensorflow",
                method="dlpack",
                reason="Incompatible versions",
            )

        assert exc_info.value.source_type == "torch"
        assert exc_info.value.target_type == "tensorflow"

    def test_exception_inheritance(self):
        """Test that MemoryConversionError inherits from Exception."""
        error = MemoryConversionError(
            source_type="jax", target_type="numpy", method="numpy_conversion", reason="Test error"
        )

        assert isinstance(error, Exception)
        assert isinstance(error, MemoryConversionError)
