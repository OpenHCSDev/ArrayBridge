"""What arraybridge calls of ``cupy.cuda``."""

from types import TracebackType

from typing_extensions import Self

from . import runtime as runtime

class Device:
    id: int
    def __init__(self, device: int = ...) -> None: ...
    def use(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

class Stream:
    def __init__(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
