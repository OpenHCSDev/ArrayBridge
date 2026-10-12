"""
Shared slice-by-slice processing logic for all memory types.

This module provides a single implementation of slice-by-slice processing
that works for all memory types, eliminating duplication across dtype wrappers.
"""

from arraybridge.libraries import ArrayLibrary
from arraybridge.stack_utils import stack_slices, unstack_slices


def process_slices(image, func, args, kwargs, device=None):
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
        device: The device to unstack on. By default the device holding the image.

    Returns:
        Processed 3D array, or tuple of (processed_3d_array, special_outputs...)
        if func returns tuples
    """
    library = ArrayLibrary.of(image)
    if device is None:
        device = library.device_of(image)

    slices_2d = unstack_slices(image, library.name, device)

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
    output_library = ArrayLibrary.of(main_outputs[0])
    result = stack_slices(
        main_outputs,
        output_library.name,
        output_library.device_of(main_outputs[0]),
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
