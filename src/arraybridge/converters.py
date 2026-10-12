"""Convert an array from one library to another, and tell which library made one."""

from __future__ import annotations

from arraybridge.devices import Device
from arraybridge.exceptions import MemoryConversionError
from arraybridge.libraries import Array, ArrayLibrary
from arraybridge.types import MemoryType


def _library(memory_type: str | MemoryType, role: str) -> ArrayLibrary[Array]:
    name = memory_type.value if isinstance(memory_type, MemoryType) else memory_type
    try:
        return ArrayLibrary.named(name)
    except ValueError:
        raise ValueError(
            f"Invalid {role} '{name}'. Available types: {sorted(ArrayLibrary.__registry__)}"
        ) from None


def convert_memory(
    data: Array,
    source_type: str | MemoryType,
    target_type: str | MemoryType,
    device: Device,
) -> Array:
    """Convert ``data`` from the source library to the target library on ``device``.

    Raises:
        ValueError: If source_type or target_type is not a registered library
        MemoryConversionError: If conversion fails
    """
    source = _library(source_type, "source_type")
    target = _library(target_type, "target_type")
    try:
        return source.convert_to(data, target, device)
    except MemoryConversionError:
        raise
    except Exception as error:
        raise MemoryConversionError(
            source_type=str(source.name),
            target_type=str(target.name),
            method="ArrayLibrary.convert_to",
            reason=str(error),
        ) from error


def detect_memory_type(data: object) -> str:
    """The name of the library that made ``data`` (``"numpy"``, ``"torch"``, ...).

    Raises:
        ValueError: If no registered library made it
    """
    return str(ArrayLibrary.of(data).name)
