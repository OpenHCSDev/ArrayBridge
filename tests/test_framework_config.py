"""Tests for the deprecated declaration-derived operation view."""

from types import MappingProxyType

from arraybridge.framework_config import _FRAMEWORK_CONFIG
from arraybridge.framework_ops import _FRAMEWORK_OPS
from arraybridge.types import MemoryType


def test_compatibility_view_is_read_only_and_exhaustive():
    assert isinstance(_FRAMEWORK_CONFIG, MappingProxyType)
    assert set(_FRAMEWORK_CONFIG) == set(MemoryType)
    assert _FRAMEWORK_OPS is _FRAMEWORK_CONFIG


def test_compatibility_view_projects_member_owned_operations():
    expected_names = {"to_numpy", "from_numpy", "stack_arrays", "scale_dtype"}
    for memory_type, operations in _FRAMEWORK_CONFIG.items():
        assert isinstance(operations, MappingProxyType)
        assert set(operations) == expected_names
        for name in expected_names:
            operation = operations[name]
            assert callable(operation)
            assert operation.__self__ is memory_type


def test_compatibility_view_contains_no_executable_strings():
    for operations in _FRAMEWORK_CONFIG.values():
        assert all(not isinstance(operation, str) for operation in operations.values())
