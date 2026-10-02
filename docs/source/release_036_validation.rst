ArrayBridge typed thread-local release correction
================================================

Parent integration owner. Actual failed original 0.3.5 publisher:
https://github.com/OpenHCSDev/ArrayBridge/actions/runs/36946585391.
Ruff and Black passed; mypy rejected decorators.py:284 because current()
returned Any from the dynamic threading.local context slot. No wheel was
published. The failure and original v0.3.5 tag are preserved unchanged.

The existing ThreadGPUContext runtime owner now uses a threading.local
subclass declaring its sole context slot. The original per-thread lazy
initialization and stream identity remain; current() consumes the declared
optional value, not hasattr plus an untyped foreign attribute. No cast,
type-ignore, fallback reader, new context registry or serialization path.
BOUND-7: declare the native storage boundary instead of probing raw attributes.
Native threading.local retains thread isolation. Durable decorated callables
continue to refer to the original importable ThreadGPUContext owner.

This patch uses version0.3.6 rather than moving/reusing an existing release tag.
The already reviewed dtype_config_default behavior is unchanged. OpenHCS's
>=0.3.5,<0.4 requirement admits this corrected patch release normally.

Retained bounded local red mypy reproduces the exact hosted no-any-return
error: exit1,2.65s,96.91MiB. Green mypy checks all17 source files with no
errors: exit0,2.75s,97.14MiB. Ruff passes src/scripts; Black leaves20 files
unchanged. Existing interpreter/tooling only; no dependency installation.

Original supervisor logs/commands reside at:
/home/ts/wt/openhcs-issue-batch-20260929/s1-installed-20261001/
arraybridge036-mypy-red and arraybridge036-mypy-green. Kernel512MiB,
no swap, oneCPU/thread,60-second shard. Source behavior and release artifact
verification are recorded below when complete; no registry success claimed
by source checks alone.
