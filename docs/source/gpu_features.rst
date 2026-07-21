GPU lifecycle
=============

GPU-backed conversion uses the explicit ``gpu_id`` passed to
``convert_memory``, ``stack_slices``, or ``unstack_slices``. Framework-specific
configuration owns device contexts, movement, streams, OOM recognition, and
cache cleanup.

Decorated GPU callables may execute in a thread-local framework stream and may
retry supported OOM failures when ``oom_recovery=True``. Decorators do not
select a device and do not convert an incompatible input.

``cleanup_all_gpu_frameworks(device_id=None)`` asks every loaded supported GPU
framework to release caches. It is safe when optional frameworks are absent,
but the application decides when values are no longer live and cleanup is
appropriate.

ArrayBridge exposes mechanisms, not scheduling policy. Worker counts, device
assignment, retry scope, and concurrency limits belong to the host runtime.
