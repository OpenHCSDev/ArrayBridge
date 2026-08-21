"""
GPU memory cleanup utilities for different frameworks.

This module provides unified GPU memory cleanup functions for PyTorch, CuPy,
TensorFlow, JAX, and pyclesperanto. The cleanup functions are designed to be called
after processing steps to free up GPU memory that's no longer needed.

Framework-specific cleanup behavior belongs to each ``MemoryType`` declaration.
"""

import logging
from types import MappingProxyType

from arraybridge.types import MemoryType

logger = logging.getLogger(__name__)


def _cleanup_declared(mem_type: MemoryType, device_id: int | None = None) -> None:
    try:
        mem_type.cleanup_loaded(device_id)
    except Exception as error:
        logger.warning(
            "Failed to cleanup %s GPU memory: %s",
            mem_type.display_name,
            error,
        )


def _create_cleanup_function(mem_type: MemoryType):
    """Create one compatibility cleanup function from its enum declaration."""

    def cleanup(device_id: int | None = None) -> None:
        """Clean an already-loaded framework without importing it."""
        _cleanup_declared(mem_type, device_id)

    cleanup.__name__ = f"cleanup_{mem_type.import_name}_gpu"
    cleanup.__doc__ = f"Clean already-loaded {mem_type.display_name} GPU resources."

    return cleanup


# Auto-generate all cleanup functions
for mem_type in MemoryType:
    cleanup_func = _create_cleanup_function(mem_type)
    globals()[cleanup_func.__name__] = cleanup_func


# Auto-generate cleanup registry
MEMORY_TYPE_CLEANUP_REGISTRY = MappingProxyType(
    {mem_type.value: globals()[f"cleanup_{mem_type.import_name}_gpu"] for mem_type in MemoryType}
)


def cleanup_all_gpu_frameworks(device_id: int | None = None) -> None:
    """
    Clean up GPU memory for all available frameworks.

    This function calls cleanup for all GPU frameworks that are currently loaded.
    It's safe to call even if some frameworks aren't available.

    Args:
        device_id: Optional GPU device ID. If None, cleans all devices.
    """
    logger.debug(f"🔥 GPU CLEANUP: Starting cleanup for all GPU frameworks (device_id={device_id})")

    for mem_type in MemoryType:
        if mem_type.is_gpu:
            _cleanup_declared(mem_type, device_id)

    logger.debug("🔥 GPU CLEANUP: Completed cleanup for all GPU frameworks")


# Export all cleanup functions and utilities
__all__ = [
    "cleanup_all_gpu_frameworks",
    "MEMORY_TYPE_CLEANUP_REGISTRY",
] + [f"cleanup_{mem_type.import_name}_gpu" for mem_type in MemoryType]
