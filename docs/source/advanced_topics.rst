Advanced extension topics
=========================

Framework behavior is centralized in the registered conversion and framework
strategy families. Add a new framework by extending those owning declarations:

1. add the ``MemoryType`` member;
2. provide framework configuration and conversion strategies;
3. define stack/allocation/device behavior where it differs;
4. add cleanup and OOM behavior only when the framework supports them;
5. exercise detection and every supported conversion pair.

Do not add a parallel framework-name table in a consumer. Generic consumers use
``MemoryType``, the converter registry, and framework configuration owned by
ArrayBridge.

Conversion caching and batching belong to the caller because value lifetime and
semantic identity are application concerns. ArrayBridge functions remain
stateless at the conversion boundary except for declared thread-local GPU
context.
