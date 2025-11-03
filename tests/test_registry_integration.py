"""Integration tests demonstrating metaclass-registry benefits."""

import numpy as np


class TestRegistryIntegration:
    """Integration tests showing how the registry simplifies converter management."""

    def test_registry_discoverability(self):
        """Test that all converters are discoverable via the registry."""
        from arraybridge.converters_registry import ConverterBase

        # Registry makes it easy to discover all available converters
        available_converters = sorted(ConverterBase.__registry__.keys())

        assert len(available_converters) == 6
        assert available_converters == [
            "cupy",
            "jax",
            "numpy",
            "pyclesperanto",
            "tensorflow",
            "torch",
        ]

    def test_registry_enables_programmatic_access(self):
        """Test that registry enables programmatic access to all converters."""
        from arraybridge.converters_registry import ConverterBase, get_converter

        # Can iterate over all registered converters
        for memory_type, converter_class in ConverterBase.__registry__.items():
            converter = get_converter(memory_type)

            # Verify each converter has the expected interface
            assert hasattr(converter, "to_numpy")
            assert hasattr(converter, "from_numpy")
            assert hasattr(converter, "from_dlpack")
            assert hasattr(converter, "move_to_device")

            # Verify memory_type matches
            assert converter.memory_type == memory_type

    def test_memory_type_enum_integration(self):
        """Test that MemoryType enum integrates seamlessly with registry."""
        from arraybridge.types import MemoryType

        np.array([1, 2, 3, 4, 5])

        # Can use MemoryType enum to get converter
        for mem_type in MemoryType:
            converter = mem_type.converter
            assert converter.memory_type == mem_type.value

    def test_convert_memory_uses_registry(self):
        """Test that convert_memory function uses registry-based converters."""
        from arraybridge.converters import convert_memory

        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        # convert_memory should work with registry
        result = convert_memory(arr, source_type="numpy", target_type="numpy", gpu_id=0)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, arr)

    def test_registry_validation_on_import(self):
        """Test that registry validates all memory types are registered on import."""
        from arraybridge.converters_registry import ConverterBase
        from arraybridge.types import MemoryType

        # Registry should contain exactly the memory types defined in MemoryType enum
        expected = {mt.value for mt in MemoryType}
        actual = set(ConverterBase.__registry__.keys())

        assert (
            expected == actual
        ), f"Registry validation failed. Expected: {expected}, Got: {actual}"

    def test_adding_new_framework_would_be_simple(self):
        """
        Demonstrate how easy it would be to add a new framework.

        This test shows the benefit: to add a new framework, you would just:
        1. Add it to MemoryType enum
        2. Add its config to _FRAMEWORK_CONFIG
        3. The converter auto-registers - no manual wiring needed!
        """
        from arraybridge.converters_registry import ConverterBase
        from arraybridge.types import MemoryType

        # Current count
        current_count = len(ConverterBase.__registry__)

        # To add a new framework, you'd just need to:
        # 1. Add to MemoryType enum (e.g., MXNET = "mxnet")
        # 2. Add to _FRAMEWORK_CONFIG with conversion_ops
        # 3. The converter class would auto-register via metaclass!

        # Verify that all current MemoryType values are registered
        assert current_count == len(MemoryType)

        # This is the key benefit: no manual _CONVERTERS[MemoryType.MXNET] = ...
        # needed anymore!


class TestConverterIsolation:
    """Test that converters are properly isolated and independent."""

    def test_converters_are_independent_instances(self):
        """Test that multiple calls to get_converter return independent instances."""
        from arraybridge.converters_registry import get_converter

        # Each call should return a new instance
        conv1 = get_converter("numpy")
        conv2 = get_converter("numpy")

        assert conv1 is not conv2
        assert isinstance(conv1, type(conv2))
        assert conv1.memory_type == conv2.memory_type

    def test_converter_classes_are_registered_not_instances(self):
        """Test that registry stores classes, not instances."""
        from arraybridge.converters_registry import ConverterBase, get_converter

        # Registry should contain classes
        numpy_class = ConverterBase.__registry__["numpy"]
        assert isinstance(numpy_class, type)

        # get_converter creates instances
        instance = get_converter("numpy")
        assert isinstance(instance, numpy_class)
