"""
Registry-based converter infrastructure using metaclass-registry.

This module provides the ConverterBase class using AutoRegisterMeta,
concrete converter implementations for each framework, and a helper
function for registry lookups.
"""

from abc import abstractmethod
from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from arraybridge.types import MemoryType


class ConverterBase(metaclass=AutoRegisterMeta):
    """Base class for memory type converters using auto-registration.

    Each concrete converter sets memory_type to register itself in the registry.
    The registry key is the memory_type attribute (e.g., "numpy", "torch").
    """

    __registry_key__ = "memory_type"
    # Simple dict: converters are created in this module, without lazy discovery.
    __registry__: ClassVar[Mapping[str, type["ConverterBase"]]] = {}
    memory_type: str | None = None

    @abstractmethod
    def to_numpy(self, data, gpu_id):
        """Extract to NumPy (type-specific implementation)."""
        pass

    @abstractmethod
    def from_numpy(self, data, gpu_id):
        """Create from NumPy (type-specific implementation)."""
        pass

    @abstractmethod
    def from_dlpack(self, data, gpu_id):
        """Create from DLPack capsule (type-specific implementation)."""
        pass

    @abstractmethod
    def move_to_device(self, data, gpu_id):
        """Move data to specified GPU device if needed (type-specific implementation)."""
        pass


def _make_to_numpy(mem_type: MemoryType):
    def to_numpy(self, data, gpu_id):
        del self, gpu_id
        return mem_type.to_numpy(data)

    to_numpy.__qualname__ = f"{mem_type.value.capitalize()}Converter.to_numpy"
    return to_numpy


def _make_from_numpy(mem_type: MemoryType):
    def from_numpy(self, data, gpu_id):
        del self
        return mem_type.from_numpy(data, gpu_id)

    from_numpy.__qualname__ = f"{mem_type.value.capitalize()}Converter.from_numpy"
    return from_numpy


def _make_device_mover(mem_type: MemoryType):
    """Create a converter adapter over declaration-owned device movement."""

    def move_to_device(self, data, gpu_id):
        del self
        return mem_type.move_to_device(data, gpu_id)

    move_to_device.__qualname__ = f"{mem_type.value.capitalize()}Converter.move_to_device"
    return move_to_device


def _make_dlpack_importer(mem_type: MemoryType):
    """Create a converter adapter over declaration-owned DLPack import."""

    def from_dlpack(self, data, gpu_id):
        del self
        module = mem_type.import_module()
        with mem_type.device_scope(gpu_id, module):
            return mem_type.from_dlpack(data, module)

    from_dlpack.__qualname__ = f"{mem_type.value.capitalize()}Converter.from_dlpack"
    return from_dlpack


# Auto-generate converter classes for each memory type
def _create_converter_classes():
    """Create concrete converter classes for each memory type."""
    for mem_type in MemoryType:
        class_attrs = {
            "memory_type": mem_type.value,
            "to_numpy": _make_to_numpy(mem_type),
            "from_numpy": _make_from_numpy(mem_type),
            "move_to_device": _make_device_mover(mem_type),
            "from_dlpack": _make_dlpack_importer(mem_type),
        }
        class_name = f"{mem_type.value.capitalize()}Converter"
        type(class_name, (ConverterBase,), class_attrs)


# Create all converter classes at module load time
_create_converter_classes()


def get_converter(memory_type: str):
    """Get a converter instance for the given memory type.

    Args:
        memory_type: The memory type string (e.g., "numpy", "torch")

    Returns:
        A converter instance for the memory type

    Raises:
        ValueError: If memory type is not registered
    """
    converter_class = ConverterBase.__registry__.get(memory_type)
    if converter_class is None:
        raise ValueError(
            f"No converter registered for memory type '{memory_type}'. "
            f"Available types: {sorted(ConverterBase.__registry__.keys())}"
        )
    return converter_class()


def _add_converter_methods():
    """Add to_X() methods to ConverterBase.

    For each target memory type, generates a method like to_cupy(), to_torch(), etc.
    that tries GPU-to-GPU conversion via DLPack first, then falls back to CPU roundtrip.
    """
    for target_type in MemoryType:
        method_name = f"to_{target_type.value}"

        def make_method(tgt):
            def method(self, data, gpu_id):
                source_type = MemoryType(self.memory_type)
                return source_type.convert_to(data, tgt, gpu_id)

            return method

        setattr(ConverterBase, method_name, make_method(target_type))


def _validate_registry():
    """Validate that all memory types are registered."""
    required_types = {mt.value for mt in MemoryType}
    registered_types = set(ConverterBase.__registry__.keys())

    if required_types != registered_types:
        missing = required_types - registered_types
        extra = registered_types - required_types
        msg_parts = []
        if missing:
            msg_parts.append(f"Missing: {missing}")
        if extra:
            msg_parts.append(f"Extra: {extra}")
        raise RuntimeError(f"Registry validation failed. {', '.join(msg_parts)}")


# Add to_X() conversion methods after converter classes are created
_add_converter_methods()

# Run validation at module load time
_validate_registry()
ConverterBase.__registry__ = MappingProxyType(dict(ConverterBase.__registry__))
