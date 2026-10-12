"""``MemoryType``: the names of the registered array libraries as an enum.

The enum is a view. Its members are generated from the classes registered in
``arraybridge.libraries.ArrayLibrary`` and carry no behaviour; use it where a
name is stored, serialised or offered as a choice, and ask
``MemoryType.X.library`` for anything the library does. A library class
defined after this module was imported appears as a member the next time the
enum is looked at.

Importing this module imports no array library, NumPy included.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from enum import Enum, EnumMeta
from types import MappingProxyType

from arraybridge.libraries import Array, ArrayLibrary


class _RegisteredLibraryNames(EnumMeta):
    """Keeps the members equal to the registered library names.

    The standard library offers no way to add a member to an existing enum, so
    a missing one is entered into the enum's own member tables here. The
    overrides below keep ``EnumMeta``'s signatures and add only that step.
    """

    def _add_missing_members(cls):
        for name in ArrayLibrary.__registry__:
            if name not in cls._value2member_map_:
                member = object.__new__(cls)
                member._name_ = name.upper()
                member._value_ = name
                cls._member_map_[member._name_] = member
                cls._member_names_.append(member._name_)
                cls._value2member_map_[name] = member

    def __call__(cls, *args, **kwargs):
        cls._add_missing_members()
        return super().__call__(*args, **kwargs)

    def __getitem__(cls, name):
        cls._add_missing_members()
        return super().__getitem__(name)

    def __reversed__(cls):
        cls._add_missing_members()
        return super().__reversed__()

    def __dir__(cls):
        cls._add_missing_members()
        return super().__dir__()

    def __iter__(cls):
        cls._add_missing_members()
        return super().__iter__()

    def __len__(cls):
        cls._add_missing_members()
        return super().__len__()

    def __contains__(cls, value):
        cls._add_missing_members()
        return super().__contains__(value)

    @property
    def __members__(cls):
        cls._add_missing_members()
        return MappingProxyType(cls._member_map_)

    def __getattr__(cls, name):
        if not name.startswith("_"):
            cls._add_missing_members()
            member = cls._member_map_.get(name)
            if member is not None:
                return member
        raise AttributeError(name)


class MemoryType(Enum, metaclass=_RegisteredLibraryNames):
    """The name of one registered array library."""

    @classmethod
    def _missing_(cls, value: object) -> MemoryType | None:
        cls._add_missing_members()
        member = cls._value2member_map_.get(value)
        return member if isinstance(member, cls) else None

    @property
    def library(self) -> ArrayLibrary[Array]:
        """The library this name stands for."""

        return ArrayLibrary.named(self.value)


MemoryType._add_missing_members()


class MemoryContractAttribute(str, Enum):
    """Names of the attributes the memory decorators set on a function."""

    INPUT = "input_memory_type"
    OUTPUT = "output_memory_type"
    EXECUTION = "execution_memory_type"

    def read(self, namespace: object, default: str | None = None) -> str | None:
        """Read this attribute from an object or a mapping."""

        if isinstance(namespace, Mapping):
            return namespace.get(self.value, default)
        return getattr(namespace, self.value, default)

    def write(self, namespace: object, value: str) -> None:
        """Write this attribute to an object or a mutable mapping."""

        if isinstance(namespace, MutableMapping):
            namespace[self.value] = value
            return
        setattr(namespace, self.value, value)
