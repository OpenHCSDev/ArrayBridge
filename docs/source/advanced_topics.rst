Advanced extension topics
=========================

Framework behavior is carried by each ``MemoryType`` member. Add a framework by
extending that owning declaration:

1. add the ``MemoryType`` member;
2. provide its typed runtime and array-operation leaves;
3. define conversion, stack, scaling, and device behavior;
4. add cleanup and OOM behavior only when the framework supports them;
5. exercise detection and every supported conversion pair.

Do not add a parallel framework-name table in a consumer. Generic consumers use
``MemoryType`` directly. The converter registry and historical framework config
are generated projections, not authorities.

Version 0.3 keeps the private ``_FRAMEWORK_CONFIG`` and ``_FRAMEWORK_OPS`` names
only as read-only operation views. Their historical mapping shape is not a
compatibility contract. Migrate callers to ``MemoryType`` methods.

Conversion caching and batching belong to the caller because value lifetime and
semantic identity are application concerns. ArrayBridge functions remain
stateless at the conversion boundary except for declared thread-local GPU
context.
