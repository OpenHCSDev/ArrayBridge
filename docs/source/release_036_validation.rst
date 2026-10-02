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

Original focused behavior suite passes11 tests,0.30s pytest/0.82s total,
52.7MiB. It covers thread-local context identity/isolation, framework/device
stream identity, actual NumPy dtype controls, and dill restore of unpublished
decorated callables with a live unpicklable runtime handle. Assertions and the
existing three test files are unchanged; no real GPU runtime is needed here.

Original packaged R0 against merged e9aaa262 at source33d5a99f reports zero
nonzero deltas across both changed product files, exit0,1.93s/43.73MiB. The
original global NRA R1 remains a separate unfinished OpenHCS tool-owner check;
this bounded ratchet does not claim that global analysis completed.

Original cached offline uv build produces wheel and sdist, exit0,
1.26s/66.22MiB. Twine accepts both. Local wheel SHA256:
f7fa23f5dcf1925742592109b396fe17662fb56f15e1755a304a45c90e79a249;
local sdist SHA256:
a558dcb5827876725c854cb367a7598765a55a43e658d82678db4f2c541663d4.
These local bytes are not assumed equal to the independently hosted rebuild.

Addresses ArrayBridge issue5. Original source proof does not establish a full
GPU matrix. The unchanged original publisher owns the complete hosted check,
build and trusted upload. Merge this reviewed source checkpoint without waiting
on optional CI, push a new annotated v0.3.6 at actual merged main, and verify
PyPI installer visibility and the hosted wheel source/API before admitting the
consumer. Do not replace v0.3.5 or replay the failed original publisher.
