"""Exceptions for arraybridge."""


class MemoryConversionError(Exception):
    """
    Exception raised when memory conversion fails.

    Attributes:
        source_type: The source memory type
        target_type: The target memory type
        method: The conversion method that was attempted
        reason: The reason for the failure
    """

    def __init__(self, source_type: str, target_type: str, method: str, reason: str):
        self.source_type = source_type
        self.target_type = target_type
        self.method = method
        self.reason = reason

        message = (
            f"Cannot convert from {source_type} to {target_type} using {method}. "
            f"Reason: {reason}"
        )

        super().__init__(message)
