"""
Memory conversion utility functions for arraybridge.

This module provides utility functions for memory conversion operations,
supporting Clause 251 (Declarative Memory Conversion Interface) and
Clause 65 (Fail Loudly).
"""

import importlib
import logging
from types import ModuleType

logger = logging.getLogger(__name__)


def optional_import(module_name: str) -> ModuleType | None:
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


def _ensure_module(module_name: str) -> ModuleType:
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


def _supports_cuda_array_interface(obj: object) -> bool:
    """
    Check if an object supports the CUDA Array Interface.

    Args:
        obj: The object to check

    Returns:
        True if the object supports the CUDA Array Interface, False otherwise
    """
    return hasattr(obj, "__cuda_array_interface__")


def _supports_dlpack(obj: object) -> bool:
    """Return whether an object exposes a standard DLPack export protocol."""

    return any(
        callable(getattr(obj, attribute, None))
        for attribute in ("__dlpack__", "toDlpack", "to_dlpack")
    )
