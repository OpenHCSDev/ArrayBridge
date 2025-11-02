"""Tests for arraybridge.converters_registry module."""

import pytest


class TestConverterRegistry:
    """Tests for converter registry functionality."""

    def test_registry_contains_all_memory_types(self):
        """Test that registry contains converters for all memory types."""
        from arraybridge.converters_registry import ConverterBase
        from arraybridge.types import MemoryType

        expected_types = {mt.value for mt in MemoryType}
        registered_types = set(ConverterBase.__registry__.keys())

        assert expected_types == registered_types, (
            f"Registry mismatch. Expected: {expected_types}, Got: {registered_types}"
        )

    def test_get_converter_returns_valid_converter(self):
        """Test that get_converter returns a valid converter instance."""
        from arraybridge.converters_registry import get_converter

        converter = get_converter("numpy")
        assert converter is not None
        assert hasattr(converter, "to_numpy")
        assert hasattr(converter, "from_numpy")
        assert hasattr(converter, "from_dlpack")
        assert hasattr(converter, "move_to_device")

    def test_get_converter_for_all_types(self):
        """Test that get_converter works for all memory types."""
        from arraybridge.converters_registry import get_converter
        from arraybridge.types import MemoryType

        for mem_type in MemoryType:
            converter = get_converter(mem_type.value)
            assert converter is not None
            assert converter.memory_type == mem_type.value

    def test_get_converter_invalid_type_raises_error(self):
        """Test that get_converter raises ValueError for invalid types."""
        from arraybridge.converters_registry import get_converter

        with pytest.raises(ValueError) as exc_info:
            get_converter("invalid_type")

        assert "No converter registered" in str(exc_info.value)
        assert "invalid_type" in str(exc_info.value)

    def test_converter_has_to_x_methods(self):
        """Test that converters have to_X() methods for all memory types."""
        from arraybridge.converters_registry import get_converter
        from arraybridge.types import MemoryType

        numpy_converter = get_converter("numpy")

        # Check that it has to_X() methods for all memory types
        for target_type in MemoryType:
            method_name = f"to_{target_type.value}"
            assert hasattr(numpy_converter, method_name), (
                f"Converter missing method: {method_name}"
            )

    def test_converter_classes_registered_with_correct_names(self):
        """Test that converter classes are registered with expected names."""
        from arraybridge.converters_registry import ConverterBase

        # Check numpy converter
        numpy_class = ConverterBase.__registry__["numpy"]
        assert numpy_class.__name__ == "NumpyConverter"

        # Check torch converter
        torch_class = ConverterBase.__registry__["torch"]
        assert torch_class.__name__ == "TorchConverter"

    def test_multiple_get_converter_calls_return_new_instances(self):
        """Test that get_converter returns new instances each time."""
        from arraybridge.converters_registry import get_converter

        converter1 = get_converter("numpy")
        converter2 = get_converter("numpy")

        # They should be different instances
        assert converter1 is not converter2
        # But same type
        assert type(converter1) == type(converter2)


class TestMemoryTypeConverterProperty:
    """Tests for MemoryType.converter property using registry."""

    def test_memory_type_converter_property_uses_registry(self):
        """Test that MemoryType.converter uses the registry."""
        from arraybridge.types import MemoryType

        numpy_converter = MemoryType.NUMPY.converter
        assert numpy_converter is not None
        assert numpy_converter.memory_type == "numpy"

    def test_converter_property_for_all_types(self):
        """Test that converter property works for all memory types."""
        from arraybridge.types import MemoryType

        for mem_type in MemoryType:
            converter = mem_type.converter
            assert converter is not None
            assert converter.memory_type == mem_type.value
