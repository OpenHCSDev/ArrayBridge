"""Retained callable declarations exclude thread-local GPU lifecycle handles."""

import threading
from concurrent.futures import ThreadPoolExecutor

import dill
import numpy as np
import pytest

from arraybridge.decorators import ThreadGPUContext, cupy, numpy
from arraybridge.types import MemoryType


def test_context_is_stable_per_thread_and_never_shared_between_threads():
    current = ThreadGPUContext.current()
    with ThreadPoolExecutor(max_workers=1) as executor:
        other, repeated = executor.submit(
            lambda: (ThreadGPUContext.current(), ThreadGPUContext.current())
        ).result()
    assert ThreadGPUContext.current() is current
    assert other is repeated
    assert other is not current


@pytest.mark.parametrize("decorator", [numpy, cupy])
def test_unpublished_decorated_callable_serializes_without_runtime_context(decorator):
    # Like a replaced custom function retained by an undo snapshot, this
    # declaration has no importable public alias. Dill must persist its body.
    @decorator
    def declared(image, scale=3):
        return image * scale

    context = ThreadGPUContext.current()
    key = (MemoryType.CUPY, -1)
    handle = threading.local()
    context._streams[key] = handle
    try:
        restored = dill.loads(dill.dumps(declared))
        assert restored.__name__ == declared.__name__
        assert restored.__wrapped__.__name__ == declared.__wrapped__.__name__
        # CPU NumPy invocation checks actual restored behavior. GPU behavior
        # keeps its existing focused tests; this receipt starts no GPU runtime.
        if decorator is numpy:
            np.testing.assert_array_equal(
                restored(np.array([1, 2], dtype=np.uint16), scale=4), [4, 8]
            )
        assert ThreadGPUContext.current() is context
        assert context._streams[key] is handle
    finally:
        del context._streams[key]
