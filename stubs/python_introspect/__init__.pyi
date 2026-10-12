"""What arraybridge uses of python-introspect."""

import inspect
from abc import ABC, abstractmethod
from typing import ClassVar

class RuntimeParameterDeclarationABC(ABC):
    preserve_for_execution: ClassVar[bool]
    is_semantic_control: ClassVar[bool]
    @classmethod
    @abstractmethod
    def require_parameter_name(cls) -> str: ...
    @classmethod
    @abstractmethod
    def parameter(cls) -> inspect.Parameter: ...
