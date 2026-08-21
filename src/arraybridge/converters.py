"""Memory conversion public API for OpenHCS."""

from typing import Any, cast

import numpy as np

from arraybridge.exceptions import MemoryConversionError
from arraybridge.types import VALID_MEMORY_TYPES, MemoryType


def convert_memory(
    data: Any,
    source_type: str | MemoryType,
    target_type: str | MemoryType,
    gpu_id: int,
) -> Any:
    """
    Convert data between memory types using the unified converter infrastructure.

    Args:
        data: The data to convert
        source_type: The source memory type (e.g., "numpy", "torch")
        target_type: The target memory type (e.g., "cupy", "jax")
        gpu_id: The target GPU device ID

    Returns:
        The converted data in the target memory type

    Raises:
        ValueError: If source_type or target_type is invalid
        MemoryConversionError: If conversion fails
    """
    source_name = source_type.value if isinstance(source_type, MemoryType) else source_type
    target_name = target_type.value if isinstance(target_type, MemoryType) else target_type
    if source_name not in VALID_MEMORY_TYPES:
        raise ValueError(
            f"Invalid source_type '{source_name}'. Available types: {sorted(VALID_MEMORY_TYPES)}"
        )
    if target_name not in VALID_MEMORY_TYPES:
        raise ValueError(
            f"Invalid target_type '{target_name}'. Available types: {sorted(VALID_MEMORY_TYPES)}"
        )

    source = MemoryType(source_name)
    target = MemoryType(target_name)
    try:
        return source.convert_to(data, target, gpu_id)
    except MemoryConversionError:
        raise
    except Exception as error:
        raise MemoryConversionError(
            source_type=source_name,
            target_type=target_name,
            method="MemoryType.convert_to",
            reason=str(error),
        ) from error


def detect_memory_type(data: Any) -> str:
    """
    Detect the memory type of data using framework config.

    Args:
        data: The data to detect

    Returns:
        The detected memory type string (e.g., "numpy", "torch")

    Raises:
        ValueError: If memory type cannot be detected
    """
    # NumPy special case (most common, check first)
    if isinstance(data, np.ndarray):
        return cast(str, MemoryType.NUMPY.value)

    # Check all frameworks using their module names from config
    module_name = type(data).__module__

    top_level = module_name.split(".")[0]

    for mem_type in MemoryType:
        if top_level in mem_type.recognized_module_names:
            return cast(str, mem_type.value)

    raise ValueError(f"Unknown memory type for {type(data)} (module: {module_name})")
