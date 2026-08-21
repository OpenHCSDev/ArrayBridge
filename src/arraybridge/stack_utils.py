"""
Stack utilities module for OpenHCS.

This module provides functions for stacking 2D slices into a 3D array
and unstacking a 3D array into 2D slices, with explicit memory type handling.

This module enforces Clause 278 — Mandatory 3D Output Enforcement:
All functions must return a 3D array of shape [Z, Y, X], even when operating
on a single 2D slice. No logic may check, coerce, or infer rank at unstack time.
"""

import logging
from typing import Any

from arraybridge.converters import detect_memory_type
from arraybridge.types import MemoryType

logger = logging.getLogger(__name__)

# 🔍 MEMORY CONVERSION LOGGING: Test log to verify logger is working
logger.debug("🔄 STACK_UTILS: Module loaded - memory conversion logging enabled")


def _is_2d(data: Any) -> bool:
    """
    Check if data is a 2D array.

    Args:
        data: Data to check

    Returns:
        True if data is 2D, False otherwise
    """
    # Check if data has a shape attribute
    if not hasattr(data, "shape"):
        return False

    # Check if shape has length 2
    return len(data.shape) == 2


def _is_3d(data: Any) -> bool:
    """
    Check if data is a 3D array.

    Args:
        data: Data to check

    Returns:
        True if data is 3D, False otherwise
    """
    # Check if data has a shape attribute
    if not hasattr(data, "shape"):
        return False

    # Check if shape has length 3
    return len(data.shape) == 3


def _enforce_gpu_device_requirements(memory_type: str, gpu_id: int) -> None:
    """
    Enforce GPU device requirements.

    Args:
        memory_type: The memory type
        gpu_id: The GPU device ID

    Raises:
        ValueError: If gpu_id is negative
    """
    mem_type = MemoryType(memory_type)
    if mem_type.is_gpu:
        mem_type.require_device(gpu_id)


def stack_slices(slices: list[Any], memory_type: str, gpu_id: int) -> Any:
    """
    Stack 2D slices into a 3D array with the specified memory type.

    STRICT VALIDATION: Assumes all slices are 2D arrays.
    No automatic handling of improper inputs.

    Args:
        slices: List of 2D slices (numpy arrays, cupy arrays, torch tensors, etc.)
        memory_type: The memory type to use for the stacked array (REQUIRED)
        gpu_id: The target GPU device ID (REQUIRED)

    Returns:
        A 3D array with the specified memory type of shape [Z, Y, X]

    Raises:
        ValueError: If memory_type is not supported or slices is empty
        ValueError: If gpu_id is negative for GPU memory types
        ValueError: If slices are not 2D arrays
        MemoryConversionError: If conversion fails
    """
    if not slices:
        raise ValueError("Cannot stack empty list of slices")

    # Verify all slices are 2D
    for i, slice_data in enumerate(slices):
        if not _is_2d(slice_data):
            raise ValueError(f"Slice at index {i} is not a 2D array. All slices must be 2D.")

    # Check GPU requirements
    _enforce_gpu_device_requirements(memory_type, gpu_id)

    # Convert each slice and enforce the requested framework-local device.
    conversion_count = 0
    mem_type = MemoryType(memory_type)
    converted_slices = []
    for slice_data in slices:
        source_type = detect_memory_type(slice_data)
        if source_type == memory_type:
            converted_data = mem_type.move_to_device(slice_data, gpu_id)
        else:
            from arraybridge.converters import convert_memory

            converted_data = convert_memory(
                data=slice_data,
                source_type=source_type,
                target_type=memory_type,
                gpu_id=gpu_id,
            )
            conversion_count += 1
        converted_slices.append(converted_data)

    result = mem_type.stack_arrays(converted_slices, gpu_id)

    # 🔍 MEMORY CONVERSION LOGGING: Only log when conversions happen or issues occur
    if conversion_count > 0:
        logger.debug(
            f"🔄 STACK_SLICES: Converted {conversion_count}/{len(slices)} "
            f"slices to {memory_type}"
        )
    # Silent success for no-conversion cases to reduce log pollution

    return result


def unstack_slices(
    array: Any, memory_type: str, gpu_id: int, validate_slices: bool = True
) -> list[Any]:
    """
    Split a 3D array into 2D slices along axis 0 and convert to the specified memory type.

    STRICT VALIDATION: Input must be a 3D array. No automatic handling of improper inputs.

    Args:
        array: 3D array to split - MUST BE 3D
        memory_type: The memory type to use for the output slices (REQUIRED)
        gpu_id: The target GPU device ID (REQUIRED)
        validate_slices: If True, validates that each extracted slice is 2D

    Returns:
        List of 2D slices in the specified memory type

    Raises:
        ValueError: If array is not 3D
        ValueError: If validate_slices is True and any extracted slice is not 2D
        ValueError: If gpu_id is negative for GPU memory types
        ValueError: If memory_type is not supported
        MemoryConversionError: If conversion fails
    """
    # Detect input type and check if conversion is needed
    input_type = detect_memory_type(array)
    getattr(array, "shape", "unknown")

    # Verify the array is 3D - fail loudly if not
    if not _is_3d(array):
        raise ValueError(f"Array must be 3D, got shape {getattr(array, 'shape', 'unknown')}")

    # Check GPU requirements
    _enforce_gpu_device_requirements(memory_type, gpu_id)

    # Convert to target memory type
    source_type = input_type  # Reuse already detected type

    # Direct conversion
    if source_type == memory_type:
        array = MemoryType(memory_type).move_to_device(array, gpu_id)
    else:
        # Convert and log the conversion
        from arraybridge.converters import convert_memory

        logger.debug(f"🔄 UNSTACK_SLICES: Converting array - {source_type} → {memory_type}")
        array = convert_memory(
            data=array, source_type=source_type, target_type=memory_type, gpu_id=gpu_id
        )

    # Extract slices along axis 0 (already in the target memory type)
    slices = [array[i] for i in range(array.shape[0])]

    # Validate that all extracted slices are 2D if requested
    if validate_slices:
        for i, slice_data in enumerate(slices):
            if not _is_2d(slice_data):
                raise ValueError(
                    f"Extracted slice at index {i} is not 2D. "
                    f"This indicates a malformed 3D array."
                )

    # 🔍 MEMORY CONVERSION LOGGING: Only log conversions or issues
    if source_type != memory_type:
        logger.debug(f"🔄 UNSTACK_SLICES: Converted and extracted {len(slices)} slices")
    elif len(slices) == 0:
        logger.warning("🔄 UNSTACK_SLICES: No slices extracted (empty array)")
    # Silent success for no-conversion cases to reduce log pollution

    return slices
