GPU lifecycle
=============

GPU-backed conversion uses the explicit ``gpu_id`` passed to
``convert_memory``, ``stack_slices``, or ``unstack_slices``. Framework-specific
``MemoryType`` declarations own device discovery, validation, scopes, movement,
and cache cleanup. Device identifiers are local to one framework. Equal numeric
identifiers from different frameworks do not establish physical-device
identity.

Before importing TensorFlow or JAX, their declarations apply coexistence-safe
defaults with ``setdefault``: TensorFlow memory growth is enabled and JAX GPU
preallocation is disabled. An environment value supplied by the host is never
overwritten. Hosts that import optional frameworks themselves should call
``MemoryType.prepare_import()`` first. ArrayBridge warns when it encounters an
already-loaded framework whose import-time defaults were absent.

``MemoryType.subprocess_environment()`` projects the same member-owned import
defaults into a child-process environment. The CuPy member also derives native
library search paths from installed NVIDIA wheels, so a fresh child interpreter
does not depend on another framework having loaded CUDA libraries first.

JAX float64 output requires x64 mode to be enabled before import. ArrayBridge
raises instead of silently returning float32 when a caller requests float64
while that capability is disabled.

Decorated GPU callables may execute in a thread-local framework stream and may
retry supported OOM failures when ``oom_recovery=True``. Decorators do not
select a device and do not convert an incompatible input.

``cleanup_all_gpu_frameworks(device_id=None)`` invokes allocator cleanup only
for loaded frameworks that declare a real cleanup leaf. Supplying ``device_id``
targets that framework-local identifier. Cleanup never imports an absent
framework. JAX compilation-cache clearing is not presented as GPU-memory
cleanup. The application decides when values are no longer live and cleanup is
appropriate.

ArrayBridge exposes mechanisms, not scheduling policy. Worker counts, device
assignment, retry scope, and concurrency limits belong to the host runtime.
