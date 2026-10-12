"""
Stack utilities module for OpenHCS.

This module provides functions for stacking 2D slices into a 3D array
and unstacking a 3D array into 2D slices, with explicit memory type handling.

This module enforces Clause 278 — Mandatory 3D Output Enforcement:
All functions must return a 3D array of shape [Z, Y, X], even when operating
on a single 2D slice. No logic may check, coerce, or infer rank at unstack time.
"""

import logging

from arraybridge.converters import convert_memory
from arraybridge.devices import Device
from arraybridge.libraries import Array, ArrayLibrary

logger = logging.getLogger(__name__)

# 🔍 MEMORY CONVERSION LOGGING: Test log to verify logger is working
logger.debug("🔄 STACK_UTILS: Module loaded - memory conversion logging enabled")


def _is_2d(data: object) -> bool:
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


def _is_3d(data: object) -> bool:
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


def stack_slices(slices: list[Array], memory_type: str, device: Device) -> Array:
    """
    Stack 2D slices into a 3D array with the specified memory type.

    STRICT VALIDATION: Assumes all slices are 2D arrays.
    No automatic handling of improper inputs.

    Args:
        slices: List of 2D slices (numpy arrays, cupy arrays, torch tensors, etc.)
        memory_type: The memory type to use for the stacked array (REQUIRED)
        device: The device of the target library to put the result on (REQUIRED)

    Returns:
        A 3D array with the specified memory type of shape [Z, Y, X]

    Raises:
        ValueError: If memory_type is not supported or slices is empty
        ValueError: If the target library does not have the device
        ValueError: If slices are not 2D arrays
        MemoryConversionError: If conversion fails
    """
    if not slices:
        raise ValueError("Cannot stack empty list of slices")

    # Verify all slices are 2D
    for i, slice_data in enumerate(slices):
        if not _is_2d(slice_data):
            raise ValueError(f"Slice at index {i} is not a 2D array. All slices must be 2D.")

    library = ArrayLibrary.named(memory_type)
    library.require_device(device)

    conversion_count = 0
    converted_slices = []
    for slice_data in slices:
        source = ArrayLibrary.of(slice_data)
        if source is not library:
            conversion_count += 1
            converted_slices.append(
                convert_memory(slice_data, str(source.name), memory_type, device)
            )
        else:
            converted_slices.append(library.move_to_device(slice_data, device))

    result = library.stack_arrays(converted_slices, device)

    # 🔍 MEMORY CONVERSION LOGGING: Only log when conversions happen or issues occur
    if conversion_count > 0:
        logger.debug(
            f"🔄 STACK_SLICES: Converted {conversion_count}/{len(slices)} "
            f"slices to {memory_type}"
        )
    # Silent success for no-conversion cases to reduce log pollution

    return result


def unstack_slices(
    array: Array, memory_type: str, device: Device, validate_slices: bool = True
) -> list[Array]:
    """
    Split a 3D array into 2D slices along axis 0 and convert to the specified memory type.

    STRICT VALIDATION: Input must be a 3D array. No automatic handling of improper inputs.

    Args:
        array: 3D array to split - MUST BE 3D
        memory_type: The memory type to use for the output slices (REQUIRED)
        device: The device of the target library to put the result on (REQUIRED)
        validate_slices: If True, validates that each extracted slice is 2D

    Returns:
        List of 2D slices in the specified memory type

    Raises:
        ValueError: If array is not 3D
        ValueError: If validate_slices is True and any extracted slice is not 2D
        ValueError: If the target library does not have the device
        ValueError: If memory_type is not supported
        MemoryConversionError: If conversion fails
    """
    source = ArrayLibrary.of(array)
    source_type = source.name

    # Verify the array is 3D - fail loudly if not
    if not _is_3d(array):
        raise ValueError(f"Array must be 3D, got shape {getattr(array, 'shape', 'unknown')}")

    library = ArrayLibrary.named(memory_type)
    library.require_device(device)
    if source is not library:
        logger.debug(f"🔄 UNSTACK_SLICES: Converting array - {source_type} → {memory_type}")
        array = convert_memory(array, str(source_type), memory_type, device)
    else:
        array = library.move_to_device(array, device)

    # Extract slices along axis 0 (already in the target memory type)
    slices = [getattr(array, "__getitem__")(index) for index in range(getattr(array, "shape")[0])]

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
