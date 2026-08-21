"""Declaration-owned dtype scaling and compatibility projections."""

from functools import partial
from types import MappingProxyType
from typing import Any

from arraybridge.array_operations import _SCALING_RANGES as _SCALING_RANGES
from arraybridge.types import MemoryType


def _scale_generic(result: Any, target_dtype: Any, mem_type: MemoryType) -> Any:
    """Scale through the operation leaf carried by ``mem_type``."""

    return mem_type.scale_dtype(result, target_dtype)


def _scale_pyclesperanto(result: Any, target_dtype: Any) -> Any:
    """Compatibility adapter for the pyclesperanto scaling leaf."""

    return MemoryType.PYCLESPERANTO.scale_dtype(result, target_dtype)


SCALING_FUNCTIONS = MappingProxyType(
    {memory_type.value: partial(_scale_generic, mem_type=memory_type) for memory_type in MemoryType}
)
