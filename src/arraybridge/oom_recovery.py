"""
GPU Out of Memory (OOM) recovery utilities.

Provides comprehensive OOM detection and cache clearing for all supported
GPU frameworks in OpenHCS.

OOM classification and cache cleanup delegate to each ``MemoryType`` declaration.
"""

import gc
import logging

from arraybridge.types import MemoryType

logger = logging.getLogger(__name__)


def _is_oom_error(e: Exception, memory_type: str) -> bool:
    """
    Detect Out of Memory errors for all GPU frameworks.

    Args:
        e: Exception to check
        memory_type: Memory type string (e.g., 'torch', 'cupy')

    Returns:
        True if exception is an OOM error for the given framework
    """
    try:
        mem_type_enum = MemoryType(memory_type)
    except ValueError:
        return False

    return mem_type_enum.is_oom_error(e)


def _clear_cache_for_memory_type(memory_type: str, device_id: int | None = None):
    """
    Clear GPU cache for specific memory type.

    Args:
        memory_type: Memory type string (e.g., 'torch', 'cupy')
        device_id: Optional framework-local GPU device ID. ``None`` cleans all.
    """
    try:
        mem_type_enum = MemoryType(memory_type)
    except ValueError:
        logger.warning(f"Unknown memory type for cache clearing: {memory_type}")
        gc.collect()
        return

    try:
        mem_type_enum.cleanup_loaded(device_id)
    except Exception as e:
        logger.warning(f"Failed to clear cache for {memory_type}: {e}")

    # Always trigger Python garbage collection
    gc.collect()


def _execute_with_oom_recovery(
    func_callable,
    memory_type: str,
    max_retries: int = 2,
    device_id: int | None = None,
):
    """
    Execute function with automatic OOM recovery.

    Args:
        func_callable: Function to execute
        memory_type: Memory type from MemoryType enum
        max_retries: Maximum number of retry attempts
        device_id: Optional framework-local device whose cache should be cleared

    Returns:
        Function result

    Raises:
        Original exception if not OOM or retries exhausted
    """
    for attempt in range(max_retries + 1):
        try:
            return func_callable()
        except Exception as e:
            if not _is_oom_error(e, memory_type) or attempt == max_retries:
                raise

            # Clear cache and retry
            _clear_cache_for_memory_type(memory_type, device_id)
