"""
Memory conversion helpers for OpenHCS.

This module provides the ABC and metaprogramming infrastructure for memory type conversions.
Uses enum-driven polymorphism to eliminate 1,567 lines of duplication.
"""

import logging
from abc import ABC, abstractmethod

from arraybridge.framework_config import _FRAMEWORK_CONFIG
from arraybridge.types import MemoryType
from arraybridge.utils import _supports_dlpack

logger = logging.getLogger(__name__)


class MemoryTypeConverter(ABC):
    """Abstract base class for memory type converters.

    Each memory type (numpy, cupy, torch, etc.) has a concrete converter
    that implements these four core operations. All to_X() methods are
    auto-generated using polymorphism.
    """

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


def _add_converter_methods():
    """Add to_X() methods to MemoryTypeConverter ABC.

    NOTE: This must be called AFTER _CONVERTERS is defined (see below).

    For each target memory type, generates a method like to_cupy(), to_torch(), etc.
    that tries GPU-to-GPU conversion via DLPack first, then falls back to CPU roundtrip.
    """
    for target_type in MemoryType:
        method_name = f"to_{target_type.value}"

        def make_method(tgt):
            def method(self, data, gpu_id):
                # Try GPU-to-GPU first (DLPack)
                if _supports_dlpack(data):
                    try:
                        target_converter = _CONVERTERS[tgt]
                        result = target_converter.from_dlpack(data, gpu_id)
                        return target_converter.move_to_device(result, gpu_id)
                    except Exception as e:
                        logger.warning(f"DLPack conversion failed: {e}. Using CPU roundtrip.")

                # CPU roundtrip using polymorphism
                numpy_data = self.to_numpy(data, gpu_id)
                target_converter = _CONVERTERS[tgt]
                return target_converter.from_numpy(numpy_data, gpu_id)
            return method

        setattr(MemoryTypeConverter, method_name, make_method(target_type))


# Import registry-based converters
from arraybridge.converters_registry import get_converter

# Populate _CONVERTERS from the registry for backward compatibility
_CONVERTERS = {
    mem_type: get_converter(mem_type.value)
    for mem_type in MemoryType
}

# NOW call _add_converter_methods() after _CONVERTERS exists
_add_converter_methods()


# Runtime validation: ensure all converters have required methods
def _validate_converters():
    """Validate that all generated converters have the required methods."""
    required_methods = ['to_numpy', 'from_numpy', 'from_dlpack', 'move_to_device']

    for mem_type, converter in _CONVERTERS.items():
        # Check ABC methods
        for method in required_methods:
            if not hasattr(converter, method):
                raise RuntimeError(f"{mem_type.value} converter missing method: {method}")

        # Check to_X() methods for all memory types
        for target_type in MemoryType:
            method_name = f'to_{target_type.value}'
            if not hasattr(converter, method_name):
                raise RuntimeError(f"{mem_type.value} converter missing method: {method_name}")

    logger.debug(f"✅ Validated {len(_CONVERTERS)} memory type converters")

# Run validation at module load time
_validate_converters()

