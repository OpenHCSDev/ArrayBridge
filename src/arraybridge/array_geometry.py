"""Framework-neutral array geometry inspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from arraybridge.array_payload import ArrayPayload


@dataclass(frozen=True, slots=True)
class ArrayGeometry:
    """Concrete shape metadata without moving array data between frameworks."""

    shape: tuple[int, ...]

    @property
    def ndim(self) -> int:
        """Return the rank derived from the canonical shape."""

        return len(self.shape)

    @classmethod
    def from_value(cls, value: Any) -> ArrayGeometry | None:
        """Inspect an array or nominal payload without forcing host conversion."""

        data = value.array_payload_data() if isinstance(value, ArrayPayload) else value
        declared_shape = getattr(data, "shape", None)
        if declared_shape is not None:
            try:
                return cls(tuple(int(axis_size) for axis_size in declared_shape))
            except (TypeError, ValueError):
                return None

        try:
            array = np.asarray(data)
        except (TypeError, ValueError):
            return None
        return cls(tuple(int(axis_size) for axis_size in array.shape))

    @classmethod
    def require_from_value(
        cls,
        value: Any,
        *,
        value_name: str = "Value",
    ) -> ArrayGeometry:
        """Return concrete geometry or reject a value without an array shape."""

        geometry = cls.from_value(value)
        if geometry is None:
            raise TypeError(
                f"{value_name} requires concrete array geometry, got " f"{type(value).__name__}."
            )
        return geometry
