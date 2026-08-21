"""
Memory conversion utility functions for arraybridge.

This module provides utility functions for memory conversion operations,
supporting Clause 251 (Declarative Memory Conversion Interface) and
Clause 65 (Fail Loudly).
"""

import importlib
import logging
from typing import Any

from arraybridge.types import MemoryType

from .exceptions import MemoryConversionError

logger = logging.getLogger(__name__)


def optional_import(module_name: str) -> Any | None:
    """Import an optional module, returning ``None`` when it is unavailable.

    Args:
        module_name: Name of the module to import

    Returns:
        The imported module if available, otherwise ``None``.

    Example:
        ```python
        # Import torch if available
        torch = optional_import("torch")

        if torch is not None:
            # Use torch
            tensor = torch.tensor([1, 2, 3])
        else:
            # Handle the case where torch is not available
            raise ImportError("PyTorch is required for this function")
        ```
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        missing_name = error.name or ""
        if module_name == missing_name or module_name.startswith(f"{missing_name}."):
            return None
        raise


def _ensure_module(module_name: str) -> Any:
    """
    Ensure a module is imported and meets version requirements.

    Args:
        module_name: The name of the module to import

    Returns:
        The imported module

    Raises:
        ImportError: If the module cannot be imported or does not meet version requirements
        RuntimeError: If the module has known issues with specific versions
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        raise ImportError(
            f"Module {module_name} is required for this operation " f"but is not installed"
        )

    return module


def _supports_cuda_array_interface(obj: Any) -> bool:
    """
    Check if an object supports the CUDA Array Interface.

    Args:
        obj: The object to check

    Returns:
        True if the object supports the CUDA Array Interface, False otherwise
    """
    return hasattr(obj, "__cuda_array_interface__")


def _supports_dlpack(obj: Any) -> bool:
    """Return whether an object exposes a standard DLPack export protocol."""

    return any(
        callable(getattr(obj, attribute, None))
        for attribute in ("__dlpack__", "toDlpack", "to_dlpack")
    )


# Compatibility adapters over declaration-owned device operations.


def _get_device_id(data: Any, memory_type: str) -> int | None:
    """
    Get the declaration-owned GPU device ID from a data object.

    Args:
        data: The data object
        memory_type: The memory type

    Returns:
        The GPU device ID or None if not applicable

    Raises:
        MemoryConversionError: If the device ID cannot be determined for a GPU memory type
    """
    mem_type = MemoryType(memory_type)
    try:
        return mem_type.device_id_of(data)
    except Exception as e:
        raise MemoryConversionError(
            source_type=memory_type,
            target_type=memory_type,
            method="device_identification",
            reason=f"Failed to identify the {mem_type.value} device: {e}",
        ) from e


def _set_device(memory_type: str, device_id: int) -> None:
    """
    Set the current device through its memory-type declaration.

    Args:
        memory_type: The memory type
        device_id: The GPU device ID

    Raises:
        MemoryConversionError: If the device cannot be set
    """
    mem_type = MemoryType(memory_type)
    try:
        mem_type.activate_device(device_id)
    except Exception as e:
        raise MemoryConversionError(
            source_type=memory_type,
            target_type=memory_type,
            method="device_selection",
            reason=f"Failed to set {mem_type.value} device to {device_id}: {e}",
        ) from e


def _move_to_device(data: Any, memory_type: str, device_id: int) -> Any:
    """
    Move data through its memory-type declaration.

    Args:
        data: The data to move
        memory_type: The memory type
        device_id: The target GPU device ID

    Returns:
        The data on the target device

    Raises:
        MemoryConversionError: If the data cannot be moved to the specified device
    """
    mem_type = MemoryType(memory_type)
    try:
        return mem_type.move_to_device(data, device_id)
    except Exception as e:
        raise MemoryConversionError(
            source_type=memory_type,
            target_type=memory_type,
            method="device_movement",
            reason=f"Failed to move {mem_type.value} array to device {device_id}: {e}",
        ) from e
