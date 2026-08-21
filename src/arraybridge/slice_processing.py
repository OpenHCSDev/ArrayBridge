"""
Shared slice-by-slice processing logic for all memory types.

This module provides a single implementation of slice-by-slice processing
that works for all memory types, eliminating duplication across dtype wrappers.
"""

from arraybridge.converters import detect_memory_type
from arraybridge.stack_utils import stack_slices, unstack_slices
from arraybridge.utils import _get_device_id


def process_slices(image, func, args, kwargs, gpu_id=None):
    """
    Process a 3D array slice-by-slice using the provided function.

    This function handles:
    - Unstacking 3D arrays into 2D slices
    - Processing each slice independently
    - Handling functions that return tuples (main output + special outputs)
    - Stacking results back into 3D arrays in the returned main-output framework
    - Combining special outputs from all slices

    Args:
        image: 3D array to process
        func: Function to apply to each slice
        args: Positional arguments to pass to func
        kwargs: Keyword arguments to pass to func
        gpu_id: Optional GPU device ID override. If not provided, attempts
            to derive from the input image and falls back to 0.

    Returns:
        Processed 3D array, or tuple of (processed_3d_array, special_outputs...)
        if func returns tuples
    """
    # Detect memory type and use proper OpenHCS utilities
    memory_type = detect_memory_type(image)
    if gpu_id is None:
        detected_gpu_id = _get_device_id(image, memory_type)
        gpu_id = 0 if detected_gpu_id is None else detected_gpu_id

    # Unstack 3D array into 2D slices
    slices_2d = unstack_slices(image, memory_type, gpu_id)

    # Process each slice and handle special outputs
    main_outputs = []
    special_outputs_list = []
    returns_tuple = None
    tuple_arity = None

    for slice_index, slice_2d in enumerate(slices_2d):
        slice_result = func(slice_2d, *args, **kwargs)

        # Check if result is a tuple (indicating special outputs)
        result_is_tuple = isinstance(slice_result, tuple)
        if returns_tuple is None:
            returns_tuple = result_is_tuple
        elif result_is_tuple != returns_tuple:
            raise TypeError(
                "Slice processing cannot mix tuple and non-tuple results; "
                f"slice {slice_index} returned {type(slice_result).__name__}."
            )

        if result_is_tuple:
            if not slice_result:
                raise ValueError("Slice processing result tuples cannot be empty")
            if tuple_arity is None:
                tuple_arity = len(slice_result)
            elif len(slice_result) != tuple_arity:
                raise ValueError(
                    "Slice processing requires every result tuple to have the "
                    f"same arity; slice {slice_index} returned {len(slice_result)}, "
                    f"expected {tuple_arity}."
                )
            main_outputs.append(slice_result[0])  # First element is main output
            special_outputs_list.append(slice_result[1:])  # Rest are special outputs
        else:
            main_outputs.append(slice_result)  # Single output

    # Stack main outputs in the framework returned by the callable. The input
    # framework owns unstacking only; decorators may declare a different output
    # framework for the per-slice function.
    if not main_outputs:
        raise ValueError("Slice processing produced no main outputs to stack")
    output_memory_type = detect_memory_type(main_outputs[0])
    output_gpu_id = _get_device_id(main_outputs[0], output_memory_type)
    result = stack_slices(
        main_outputs,
        output_memory_type,
        0 if output_gpu_id is None else output_gpu_id,
    )

    # If we have special outputs, combine them and return tuple
    if special_outputs_list:
        # Combine special outputs from all slices
        combined_special_outputs = []
        num_special_outputs = len(special_outputs_list[0])

        for i in range(num_special_outputs):
            # Collect the i-th special output from all slices
            special_output_values = [slice_outputs[i] for slice_outputs in special_outputs_list]
            combined_special_outputs.append(special_output_values)

        # Return tuple: (stacked_main_output, combined_special_output1,  # noqa: E501
        # combined_special_output2, ...)
        return (result, *combined_special_outputs)

    return result
