"""Stand-ins for array libraries that are not installed.

A library class always works on ``sys.modules[import_name]``, so a test installs
one of these with ``monkeypatch.setitem(sys.modules, "cupy", FakeCupy())``. Each
fake implements only what the library class calls and writes what was done to
``events``. Fake arrays set ``__module__`` to the library's module name, which is
how ``ArrayLibrary.of`` recognises them.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np


class DeviceScope:
    """Makes ``index`` the fake module's current device for a block."""

    def __init__(self, module, index):
        self.module = module
        self.index = index
        self.previous = None

    def __enter__(self):
        self.previous = self.module.current
        self.module.current = self.index
        self.module.events.append(("enter", self.index))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.module.current = self.previous
        self.module.events.append(("exit", self.index))
        return False


class FakeStream:
    """A CUDA stream: remembers the device it was created on and is its own scope."""

    def __init__(self, module):
        self.module = module
        self.device_index = module.current

    def __enter__(self):
        self.module.events.append(("stream enter", self))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.module.events.append(("stream exit", self))
        return False


class _Pool:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def free_all_blocks(self):
        self.events.append(("free", self.name))


@dataclass(frozen=True)
class FakeCapsule:
    """What a fake array hands over through DLPack: its buffer and where it lives."""

    values: np.ndarray
    device_index: int | None


class _FakeArray:
    def __init__(self, values):
        self.values = np.asarray(values)

    @property
    def shape(self):
        return self.values.shape

    @property
    def dtype(self):
        return self.values.dtype


class FakeCupyArray(_FakeArray):
    __module__ = "cupy"

    def __init__(self, values, device_index, module):
        super().__init__(values)
        self.device = SimpleNamespace(id=device_index)
        self.module = module

    def copy(self):
        return type(self)(self.values.copy(), self.module.current, self.module)

    def get(self):
        return self.values.copy()

    def __getitem__(self, index):
        return type(self)(self.values[index], self.device.id, self.module)

    def __dlpack__(self):
        return FakeCapsule(self.values, self.device.id)

    def __dlpack_device__(self):
        return (2, self.device.id)


class FakeCupy:
    ndarray = FakeCupyArray

    def __init__(self, device_count=2):
        self.events = []
        self.current = 0
        self.cuda = SimpleNamespace(
            runtime=SimpleNamespace(
                getDeviceCount=lambda: device_count,
                getDevice=lambda: self.current,
                deviceSynchronize=lambda: self.events.append(("synchronize", None)),
            ),
            Device=lambda index: DeviceScope(self, index),
            Stream=lambda: FakeStream(self),
        )
        self._pools = (_Pool("regular", self.events), _Pool("pinned", self.events))

    def get_default_memory_pool(self):
        return self._pools[0]

    def get_default_pinned_memory_pool(self):
        return self._pools[1]

    def array(self, data):
        return FakeCupyArray(data, self.current, self)

    def stack(self, values, axis):
        stacked = np.stack([value.values for value in values], axis=axis)
        return FakeCupyArray(stacked, self.current, self)


class FakeTorchTensor(_FakeArray):
    __module__ = "torch"

    def __init__(self, values, cuda_index=None):
        super().__init__(values)
        self.device = SimpleNamespace(index=cuda_index)

    @property
    def is_cuda(self):
        return self.device.index is not None

    def cpu(self):
        return type(self)(self.values) if self.is_cuda else self

    def to(self, device):
        return type(self)(self.values, int(device.removeprefix("cuda:")))

    def numpy(self):
        return self.values.copy()

    def element_size(self):
        return self.values.dtype.itemsize

    def nelement(self):
        return self.values.size


class FakeTorch:
    """torch with main memory and ``device_count`` CUDA devices (none by default)."""

    Tensor = FakeTorchTensor

    def __init__(self, device_count=0):
        module = self

        class Stream(FakeStream):
            """``torch.cuda.Stream`` is a class: a stream is recognised by being one."""

            def __init__(self):
                super().__init__(module)

        self.events = []
        self.current = 0
        self.cuda = SimpleNamespace(
            is_available=lambda: device_count > 0,
            device_count=lambda: device_count,
            current_device=lambda: self.current,
            set_device=lambda index: self.events.append(("set_device", index)),
            device=lambda index: DeviceScope(self, index),
            Stream=Stream,
            stream=lambda stream: stream,
            empty_cache=lambda: self.events.append(("empty_cache", self.current)),
            synchronize=lambda: self.events.append(("synchronize", self.current)),
        )

    def from_numpy(self, data):
        self.events.append(("from_numpy", data))
        return FakeTorchTensor(data)

    def from_dlpack(self, capsule):
        return FakeTorchTensor(capsule.values, capsule.device_index)


class FakeTensorflowTensor(_FakeArray):
    __module__ = "tensorflow"

    def __init__(self, values, gpu_index):
        super().__init__(values)
        self.device = f"/device:GPU:{gpu_index}"

    def __getitem__(self, index):
        return type(self)(self.values[index], int(self.device.rsplit(":", 1)[-1]))

    def numpy(self):
        return self.values


class FakeTensorflow:
    """TensorFlow with two GPUs; new tensors land on the device in scope."""

    def __init__(self):
        self.events = []
        self.current = 0
        self.config = SimpleNamespace(
            list_logical_devices=lambda kind: [object(), object()] if kind == "GPU" else []
        )

    def device(self, name):
        return DeviceScope(self, int(name.rsplit(":", 1)[-1]))

    def identity(self, data):
        return FakeTensorflowTensor(data.values, self.current)

    def stack(self, values, axis):
        stacked = np.stack([value.values for value in values], axis=axis)
        return FakeTensorflowTensor(stacked, self.current)


class FakePyclesperantoArray(_FakeArray):
    __module__ = "pyclesperanto"

    def __init__(self, values, device_name):
        super().__init__(values)
        self.device = SimpleNamespace(name=device_name)


class FakePyclesperanto:
    """pyclesperanto with GPUs ``gpu0`` and ``gpu1``; ``current`` may be any device."""

    gpu_devices = ("gpu0", "gpu1")

    def __init__(self, current="gpu0"):
        self.current = current
        self.events = []

    def list_available_devices(self, device_type=None):
        assert device_type in (None, "gpu")
        return self.gpu_devices

    def get_device(self):
        return self.current

    def select_device(self, selector, device_type=None):
        if isinstance(selector, int):
            assert device_type in (None, "gpu")
            self.current = self.gpu_devices[selector]
        else:
            self.current = selector

    def create_like(self, data):
        return ("copy", data)

    def copy(self, source, target):
        self.events.append(("copy", source, target))

    def pull(self, data):
        return data.values

    def push(self, data):
        return FakePyclesperantoArray(data, self.current)

    def create(self, shape, dtype):
        self.events.append(("create", shape))
        return FakePyclesperantoArray(np.empty(shape, dtype=dtype), self.current)

    def copy_slice(self, source, target, index):
        self.events.append(("copy_slice", index))
        target.values[index] = source.values
        return target

    def concatenate_along_z(self, left, right):
        self.events.append(("concatenate", left.shape, right.shape))
        planes = [
            each.values if each.values.ndim == 3 else each.values[None] for each in (left, right)
        ]
        return FakePyclesperantoArray(np.concatenate(planes, axis=0), self.current)


def fake_jax(*platforms):
    """A jax module that lists one device per given platform (``"cpu"``, ``"gpu"``)."""

    devices = [SimpleNamespace(platform=platform) for platform in platforms]

    def list_devices(platform=None):
        return [each for each in devices if platform in (None, each.platform)]

    return SimpleNamespace(devices=list_devices)
