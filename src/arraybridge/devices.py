"""Where an array lives: the host processor or one accelerator of a library."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class Device(ABC):
    """One place array data can live. Leaves are values: equal fields, equal device."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Short text naming this device in records and messages."""


@dataclass(frozen=True, slots=True)
class HostDevice(Device):
    """Main memory, computed on by the host processor."""

    @property
    def label(self) -> str:
        return "host"


@dataclass(frozen=True, slots=True)
class AcceleratorDevice(Device):
    """One accelerator as its library numbers it (a CUDA or an OpenCL device)."""

    index: int

    @property
    def label(self) -> str:
        return f"accelerator:{self.index}"


@dataclass(frozen=True, slots=True)
class NamedDevice(Device):
    """A device a library reports that is not one of its numbered accelerators.

    pyclesperanto running on a CPU OpenCL device is the case that exists: the
    array belongs to that OpenCL device, which is neither NumPy's main memory
    nor a listed GPU. It is identified by the name the library gives it.
    """

    name: str

    @property
    def label(self) -> str:
        return f"device:{self.name}"


HOST = HostDevice()
