"""
Registry-based converter infrastructure using metaclass-registry.

This module provides the ConverterBase class using AutoRegisterMeta,
concrete converter implementations for each framework, and a helper
function for registry lookups.
"""

import logging
from abc import abstractmethod

from metaclass_registry import AutoRegisterMeta

from arraybridge.framework_config import _FRAMEWORK_CONFIG
from arraybridge.types import MemoryType

logger = logging.getLogger(__name__)


class ConverterBase(metaclass=AutoRegisterMeta):
    """Base class for memory type converters using auto-registration.

    Each concrete converter sets memory_type to register itself in the registry.
    The registry key is the memory_type attribute (e.g., "numpy", "torch").
    """

    __registry_key__ = "memory_type"
    memory_type: str = None

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


def _ensure_module(memory_type: str):
    """Import and return the module for the given memory type."""
    from arraybridge.utils import _ensure_module as _ensure_module_impl

    return _ensure_module_impl(memory_type)


def _make_lambda_with_name(expr_str, mem_type, method_name):
    """Create a lambda from expression string and add proper __name__ for debugging.

    Note: Uses eval() for dynamic code generation from trusted framework_config.py strings.
    This is safe because:
    1. Input strings come from _FRAMEWORK_CONFIG, not user input
    2. Strings are defined at module load time by package maintainers
    3. This pattern enables declarative framework configuration
    """
    module_str = f'_ensure_module("{mem_type.value}")'
    lambda_expr = f"lambda self, data, gpu_id: {expr_str.format(mod=module_str)}"
    lambda_func = eval(lambda_expr)
    lambda_func.__name__ = method_name
    lambda_func.__qualname__ = f"{mem_type.value.capitalize()}Converter.{method_name}"
    return lambda_func


def _make_not_implemented(mem_type_value, method_name):
    """Create a lambda that raises NotImplementedError with the correct signature."""

    def not_impl(self, data, gpu_id):
        raise NotImplementedError(f"DLPack not supported for {mem_type_value}")

    not_impl.__name__ = method_name
    not_impl.__qualname__ = f"{mem_type_value.capitalize()}Converter.{method_name}"
    return not_impl


# Auto-generate converter classes for each memory type
def _create_converter_classes():
    """Create concrete converter classes for each memory type."""
    converters = {}

    for mem_type in MemoryType:
        config = _FRAMEWORK_CONFIG[mem_type]
        conversion_ops = config["conversion_ops"]

        # Build class attributes
        class_attrs = {
            "memory_type": mem_type.value,
        }

        # Add conversion methods
        for method_name, expr in conversion_ops.items():
            if expr is None:
                class_attrs[method_name] = _make_not_implemented(mem_type.value, method_name)
            else:
                class_attrs[method_name] = _make_lambda_with_name(expr, mem_type, method_name)

        # Create the class
        class_name = f"{mem_type.value.capitalize()}Converter"
        converter_class = type(class_name, (ConverterBase,), class_attrs)

        converters[mem_type] = converter_class

    return converters


# Create all converter classes at module load time
_CONVERTER_CLASSES = _create_converter_classes()


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
    from arraybridge.utils import _supports_dlpack

    for target_type in MemoryType:
        method_name = f"to_{target_type.value}"

        def make_method(tgt):
            def method(self, data, gpu_id):
                # Try GPU-to-GPU first (DLPack)
                if _supports_dlpack(data):
                    try:
                        target_converter = get_converter(tgt.value)
                        result = target_converter.from_dlpack(data, gpu_id)
                        return target_converter.move_to_device(result, gpu_id)
                    except Exception as e:
                        logger.warning(f"DLPack conversion failed: {e}. Using CPU roundtrip.")

                # CPU roundtrip using polymorphism
                numpy_data = self.to_numpy(data, gpu_id)
                target_converter = get_converter(tgt.value)
                return target_converter.from_numpy(numpy_data, gpu_id)

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

    logger.debug(f"✅ Validated {len(registered_types)} memory type converters in registry")


# Add to_X() conversion methods after converter classes are created
_add_converter_methods()

# Run validation at module load time
_validate_registry()
