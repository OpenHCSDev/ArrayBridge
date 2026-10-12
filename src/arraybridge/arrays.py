"""The nominal type of an array of any supported library."""

from abc import ABC


class Array(ABC):
    """An array of any supported library.

    Membership is by registration: each library's ``array_class`` is a
    subclass, and a library's native array classes are registered with it when
    the library first meets one (``ArrayLibrary.of``).
    """
