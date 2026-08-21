"""Nominal contract for containers that preserve semantics around array data."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar

ArrayPayloadT = TypeVar("ArrayPayloadT", bound="ArrayPayload")


class ArrayPayload(ABC):
    """Container whose array data can be transformed without losing its context."""

    @abstractmethod
    def array_payload_data(self) -> Any:
        """Return the concrete array governed by this payload."""

    @abstractmethod
    def with_data(self: ArrayPayloadT, data: Any) -> ArrayPayloadT:
        """Return this payload's context attached to replacement array data."""

    def map_array_payload(
        self: ArrayPayloadT,
        transform: Callable[[Any], Any],
    ) -> ArrayPayloadT:
        """Apply one array transform while preserving the payload container."""

        current_data = self.array_payload_data()
        transformed_data = transform(current_data)
        if transformed_data is current_data:
            return self
        return self.with_data(transformed_data)
