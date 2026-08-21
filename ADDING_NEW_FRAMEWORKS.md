# Adding a framework

This is a maintainer how-to for extending ArrayBridge's closed framework
taxonomy. Applications should consume `MemoryType`; they should not add their
own framework registry.

## 1. Define typed leaves

Add an `ArrayOperations` bundle in
`src/arraybridge/array_operations.py`. It must provide four ordinary callables:

- project an array to NumPy;
- create an array from NumPy on the requested framework-local device;
- stack prepared 2D arrays into one 3D array;
- apply the framework's dtype-scaling semantics.

Define device, stream, DLPack, cleanup, and OOM leaves in
`src/arraybridge/types.py`. Use the no-op defaults in `FrameworkRuntime` only
when the framework genuinely lacks that capability. Do not encode behavior as
strings or add a parallel `MemoryType`-keyed table.

## 2. Add the declaration

Add one `MemoryType` member carrying:

1. its stable value and import name;
2. its display name and module aliases;
3. whether it is GPU-backed;
4. any environment defaults that must be set before first import;
5. its `FrameworkRuntime` leaves;
6. its `ArrayOperations` bundle.

The converter classes, target conversion methods, compatibility operation view,
dtype-scaling view, cleanup adapters, and framework decorator are generated
from the enum. Do not wire them manually.

## 3. Prove the extension

Add focused tests for:

- module-name detection and optional-dependency failure;
- NumPy round trips and every supported DLPack pair;
- CPU-only installations of a nominally GPU framework;
- invalid, negative, and multiple framework-local device IDs;
- same-framework moves between two devices;
- one-, two-, and multi-slice stack/unstack behavior;
- immutable array types;
- cleanup and OOM behavior without importing an absent framework;
- generated registry and export completeness.

Run:

```bash
ruff check src tests
black --check src tests
mypy src --ignore-missing-imports
pytest
python -m sphinx -E -W --keep-going -b html docs/source /tmp/arraybridge-docs
```

When GPU hardware is available, also exercise every installed framework pair in
one process. This catches import-order, allocator-coexistence, DLPack, and real
API-shape defects that isolated mocks cannot prove.
