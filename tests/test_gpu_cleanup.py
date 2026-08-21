"""Behavioral tests for declaration-owned GPU cleanup."""

import sys
from contextlib import AbstractContextManager
from types import MappingProxyType, SimpleNamespace

import pytest

from arraybridge.gpu_cleanup import MEMORY_TYPE_CLEANUP_REGISTRY, cleanup_all_gpu_frameworks
from arraybridge.types import GPU_MEMORY_TYPES, MemoryType


class _DeviceScope(AbstractContextManager):
    def __init__(self, events, device_id):
        self.events = events
        self.device_id = device_id

    def __enter__(self):
        self.events.append(("enter", self.device_id))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.events.append(("exit", self.device_id))
        return False


class _Pool:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def free_all_blocks(self):
        self.events.append(("free", self.name))


def _fake_cupy(events, device_count=2):
    regular_pool = _Pool("regular", events)
    pinned_pool = _Pool("pinned", events)
    runtime = SimpleNamespace(
        getDeviceCount=lambda: device_count,
        deviceSynchronize=lambda: events.append(("synchronize", None)),
    )
    cuda = SimpleNamespace(
        runtime=runtime,
        Device=lambda device_id: _DeviceScope(events, device_id),
    )
    return SimpleNamespace(
        cuda=cuda,
        get_default_memory_pool=lambda: regular_pool,
        get_default_pinned_memory_pool=lambda: pinned_pool,
    )


class TestCleanupProjection:
    def test_compatibility_registry_is_derived_for_every_declaration(self):
        assert isinstance(MEMORY_TYPE_CLEANUP_REGISTRY, MappingProxyType)
        assert set(MEMORY_TYPE_CLEANUP_REGISTRY) == {
            memory_type.value for memory_type in MemoryType
        }
        for memory_type in MemoryType:
            cleanup = MEMORY_TYPE_CLEANUP_REGISTRY[memory_type.value]
            assert cleanup.__name__ == f"cleanup_{memory_type.import_name}_gpu"

    def test_cleanup_functions_remain_importable(self):
        from arraybridge import gpu_cleanup

        for memory_type in MemoryType:
            assert callable(getattr(gpu_cleanup, f"cleanup_{memory_type.import_name}_gpu"))


class TestCleanupBehavior:
    def test_absent_frameworks_are_never_imported(self, monkeypatch):
        from arraybridge import types

        for memory_type in GPU_MEMORY_TYPES:
            monkeypatch.delitem(sys.modules, memory_type.import_name, raising=False)

        def fail_import(name):
            raise AssertionError(f"cleanup imported absent framework {name}")

        monkeypatch.setattr(types.importlib, "import_module", fail_import)
        cleanup_all_gpu_frameworks()

    def test_loaded_cupy_cleanup_executes_declared_leaf(self, monkeypatch):
        events = []
        monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(events))

        MEMORY_TYPE_CLEANUP_REGISTRY[MemoryType.CUPY.value]()

        assert events == [
            ("enter", 0),
            ("free", "regular"),
            ("free", "pinned"),
            ("synchronize", None),
            ("exit", 0),
            ("enter", 1),
            ("free", "regular"),
            ("free", "pinned"),
            ("synchronize", None),
            ("exit", 1),
        ]

    def test_device_specific_cleanup_is_scoped(self, monkeypatch):
        events = []
        monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(events))

        MEMORY_TYPE_CLEANUP_REGISTRY[MemoryType.CUPY.value](1)

        assert events == [
            ("enter", 1),
            ("free", "regular"),
            ("free", "pinned"),
            ("synchronize", None),
            ("exit", 1),
        ]

    def test_declaration_rejects_undeclared_cleanup_device(self, monkeypatch):
        events = []
        monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(events, device_count=1))

        with pytest.raises(ValueError, match="device 3 is unavailable"):
            MemoryType.CUPY.cleanup_loaded(3)

        assert events == []

    def test_compatibility_cleanup_contains_undeclared_device(self, monkeypatch, caplog):
        events = []
        monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(events, device_count=1))

        MEMORY_TYPE_CLEANUP_REGISTRY[MemoryType.CUPY.value](3)

        assert events == []
        assert "device 3 is unavailable" in caplog.text

    def test_jax_compilation_cache_is_not_misrepresented_as_gpu_cleanup(self, monkeypatch):
        events = []
        module = SimpleNamespace(
            clear_caches=lambda: events.append("clear"),
            devices=lambda: [SimpleNamespace(platform="gpu")],
        )
        monkeypatch.setitem(sys.modules, "jax", module)

        MemoryType.JAX.cleanup_loaded(0)

        assert events == []

    def test_numpy_cleanup_is_a_noop_without_import(self, monkeypatch):
        from arraybridge import types

        monkeypatch.delitem(sys.modules, "numpy", raising=False)

        def fail_import(name):
            raise AssertionError(f"cleanup imported {name}")

        monkeypatch.setattr(types.importlib, "import_module", fail_import)
        MEMORY_TYPE_CLEANUP_REGISTRY[MemoryType.NUMPY.value](0)
