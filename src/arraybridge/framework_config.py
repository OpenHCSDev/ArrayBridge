"""Deprecated compatibility view of declaration-owned array operations.

Runtime code consumes :class:`arraybridge.types.MemoryType` directly. This
read-only mapping exists only for callers that still import the historical
private name.
"""

from types import MappingProxyType

from arraybridge.types import MemoryType


def _operation_view(memory_type: MemoryType):
    return MappingProxyType(
        {
            "to_numpy": memory_type.to_numpy,
            "from_numpy": memory_type.from_numpy,
            "stack_arrays": memory_type.stack_arrays,
            "scale_dtype": memory_type.scale_dtype,
        }
    )


_FRAMEWORK_CONFIG = MappingProxyType(
    {memory_type: _operation_view(memory_type) for memory_type in MemoryType}
)
