"""The ArrayOperations family: one leaf per MemoryType, found by its declaration."""

import ast
from pathlib import Path

import numpy as np
import pytest

from arraybridge import MemoryType
from arraybridge.array_operations import ArrayOperations

SOURCE = Path(__file__).parents[1] / "src" / "arraybridge"


def test_every_memory_type_has_exactly_one_leaf():
    assert set(ArrayOperations.__registry__) == {member.value for member in MemoryType}
    for member in MemoryType:
        leaf = ArrayOperations.for_memory(member)
        assert leaf.memory_type == member.value
        assert ArrayOperations.for_memory(member) is leaf
        assert member._operations is leaf


def test_leaves_are_not_named_by_strings_or_module_singletons():
    types_source = (SOURCE / "types.py").read_text()
    assert "_OPERATIONS" not in types_source
    tree = ast.parse((SOURCE / "array_operations.py").read_text())
    module_level_names = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert not {name for name in module_level_names if name.endswith("_OPERATIONS")}


def test_numpy_primitives():
    ops = ArrayOperations.for_memory(MemoryType.NUMPY)
    stack = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    assert ops.module is np
    assert ops.accepts(stack) and not ops.accepts([1, 2])
    np.testing.assert_array_equal(ops.max(stack, 0), stack[1])
    np.testing.assert_array_equal(ops.min(stack, 0), stack[0])
    np.testing.assert_array_equal(ops.mean(stack, 0), stack.mean(axis=0))
    np.testing.assert_array_equal(ops.sum(stack, (1, 2)), stack.sum(axis=(1, 2)))
    assert ops.prepend_axis(stack[0]).shape == (1, 3, 4)
    assert ops.astype(stack, np.float32).dtype == np.float32
    ramp = ops.linspace(0, 1, 4, endpoint=False)
    np.testing.assert_array_equal(ramp, [0, 0.25, 0.5, 0.75])
    weights = ops.assign(ops.ones(5, np.float32), slice(-2, None), ramp[:2])
    np.testing.assert_array_equal(weights, [1, 1, 1, 0, 0.25])
    np.testing.assert_array_equal(ops.outer(ramp, ramp), np.outer(ramp, ramp))
    assert ops.floor(2.7) == 2


def test_jax_primitives_are_functional():
    jnp = pytest.importorskip("jax.numpy")
    ops = ArrayOperations.for_memory(MemoryType.JAX)
    values = jnp.ones(4, dtype=jnp.float32)
    updated = ops.assign(values, slice(None, 2), jnp.zeros(2, dtype=jnp.float32))
    np.testing.assert_array_equal(np.asarray(updated), [0, 0, 1, 1])
    np.testing.assert_array_equal(np.asarray(values), [1, 1, 1, 1])
    stack = jnp.arange(24, dtype=jnp.uint16).reshape(2, 3, 4)
    assert ops.mean(stack, 0).dtype == jnp.float32
