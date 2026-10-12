"""What arraybridge uses of metaclass-registry."""

from abc import ABCMeta
from enum import Enum

class RegistryKeyAttribute(str, Enum):
    STRATEGY_LABEL = "strategy_label"

class RegistryFamily:
    def __init__(self, key_attribute: str | RegistryKeyAttribute) -> None: ...

class AutoRegisterMeta(ABCMeta): ...
