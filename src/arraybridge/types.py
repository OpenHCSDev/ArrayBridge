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

import sys
from collections.abc import Iterator, Mapping, MutableMapping
from enum import Enum, EnumMeta
from types import MappingProxyType
from typing import TypeGuard, TypeVar

from arraybridge.libraries import Array, ArrayLibrary

MemberT = TypeVar("MemberT")


def _add_missing_members(enum_class: type) -> None:
    """Enter a member for every registered library name the enum does not have yet."""

    if not issubclass(enum_class, Enum):
        raise TypeError(f"{enum_class!r} is not an enum class")
    for name in ArrayLibrary.__registry__:
        if name not in enum_class._value2member_map_:
            member = object.__new__(enum_class)
            member._name_ = name.upper()
            member._value_ = name
            enum_class._member_map_[member._name_] = member
            enum_class._member_names_.append(member._name_)
            enum_class._value2member_map_[name] = member


def _holds_members_of(
    members: MappingProxyType[str, Enum], enum_class: type[MemberT]
) -> TypeGuard[MappingProxyType[str, MemberT]]:
    return all(isinstance(member, enum_class) for member in members.values())


class _RegisteredLibraryNames(EnumMeta):
    """Keeps the members equal to the registered library names.

    The standard library offers no way to add a member to an existing enum, so
    a missing one is entered into the enum's own member tables. The overrides
    below keep ``EnumMeta``'s signatures and add only that step. A lookup by
    value needs no override: ``Enum`` asks ``_missing_`` for a value it does
    not know, and ``MemoryType._missing_`` adds the missing members.
    """

    def __getitem__(cls: type[MemberT], name: str) -> MemberT:
        _add_missing_members(cls)
        return EnumMeta.__getitem__(cls, name)

    def __reversed__(cls: type[MemberT]) -> Iterator[MemberT]:
        _add_missing_members(cls)
        return EnumMeta.__reversed__(cls)

    def __dir__(cls) -> list[str]:
        _add_missing_members(cls)
        return super().__dir__()

    def __iter__(cls: type[MemberT]) -> Iterator[MemberT]:
        _add_missing_members(cls)
        return EnumMeta.__iter__(cls)

    def __len__(cls) -> int:
        _add_missing_members(cls)
        return super().__len__()

    # ``EnumMeta.__contains__`` takes any value from Python 3.12 and only a member before.
    if sys.version_info >= (3, 12):

        def __contains__(cls, value: object) -> bool:
            _add_missing_members(cls)
            return super().__contains__(value)

    else:

        def __contains__(cls, member: Enum) -> bool:
            _add_missing_members(cls)
            return super().__contains__(member)

    @property
    def __members__(cls: type[MemberT]) -> MappingProxyType[str, MemberT]:
        _add_missing_members(cls)
        if not issubclass(cls, Enum):
            raise TypeError(f"{cls!r} is not an enum class")
        members = MappingProxyType(cls._member_map_)
        if not _holds_members_of(members, cls):
            raise TypeError(f"{cls!r} holds a member of another class")
        return members

    def __getattr__(cls: type[MemberT], name: str) -> MemberT:
        if not name.startswith("_"):
            _add_missing_members(cls)
            try:
                return EnumMeta.__getitem__(cls, name)
            except KeyError:
                pass
        raise AttributeError(name)


class MemoryType(Enum, metaclass=_RegisteredLibraryNames):
    """The name of one registered array library."""

    @classmethod
    def _missing_(cls, value: object) -> MemoryType | None:
        _add_missing_members(cls)
        member = cls._value2member_map_.get(value)
        return member if isinstance(member, cls) else None

    @property
    def library(self) -> ArrayLibrary[Array]:
        """The library this name stands for."""

        return ArrayLibrary.named(self.value)


_add_missing_members(MemoryType)


class MemoryContractAttribute(str, Enum):
    """Names of the attributes the memory decorators set on a function."""

    INPUT = "input_memory_type"
    OUTPUT = "output_memory_type"
    EXECUTION = "execution_memory_type"

    def read(self, namespace: object, default: str | None = None) -> str | None:
        """Read this attribute from an object or a mapping."""

        value: object = (
            namespace.get(self.value, default)
            if isinstance(namespace, Mapping)
            else getattr(namespace, self.value, default)
        )
        if value is None or isinstance(value, str):
            return value
        raise TypeError(f"{self.value} of {namespace!r} is {value!r}, not a library name")

    def write(self, namespace: object, value: str) -> None:
        """Write this attribute to an object or a mutable mapping."""

        if isinstance(namespace, MutableMapping):
            namespace[self.value] = value
            return
        setattr(namespace, self.value, value)
