# Testing ArrayBridge

This is the practical test guide. The workflow files under `.github/workflows`
are the authority for the current CI matrix.

## Local core checks

```bash
python -m pip install -e ".[dev,docs]"
ruff check src tests
black --check src tests
mypy src stubs
pytest
python -m sphinx -E -W --keep-going -b html docs/source /tmp/arraybridge-docs
```

The default suite uses real optional frameworks when installed and otherwise
exercises declared unavailable paths and focused fakes. Markers such as
`torch`, `tensorflow`, `jax`, `cupy`, `pyclesperanto`, and `gpu` can select or
exclude framework-specific tests.

## Real GPU lattice check

On a GPU host, validate more than a single NumPy round trip:

- import all installed frameworks in one fresh process;
- enumerate each library's devices through `ArrayLibrary.all()` (`available_devices()`);
- convert a small exact array through every available source-target pair;
- stack and unstack one, two, and several planes in every framework;
- verify same-framework movement across multiple local devices when present;
- confirm absent frameworks are not imported by discovery or cleanup;
- exercise DLPack success and explicit CPU fallback paths;
- target OOM cleanup to the device that owned execution.

Use a fresh process and unset TensorFlow/JAX allocator variables when checking
that declaration-owned import defaults work. Framework warnings about shared
CUDA plugin registration should be recorded separately from value, device, or
shape failures.

## Commands for a GPU machine

0.5.0's GPU run predates three later fixes, and 0.5.1 was not run on a GPU.
On a machine with CUDA 12 and about 15 GB of free disk:

```bash
git clone https://github.com/OpenHCSDev/ArrayBridge.git && cd ArrayBridge
git checkout v0.5.1
uv venv -p 3.12 .venv-gpu
uv pip install -p .venv-gpu/bin/python -e ".[dev,gpu]"
.venv-gpu/bin/python -m pytest -q --no-cov -ra
.venv-gpu/bin/python -m pytest -q --no-cov -ra -m "gpu or cupy or torch or tensorflow or jax or pyclesperanto"
.venv-gpu/bin/python -m pytest -q --no-cov -ra tests/test_operation_soundness.py
.venv-gpu/bin/python host_call_check.py
```

`tests/test_operation_soundness.py` calls every array operation of every
installed library on its first device; no skip should name an installed
library. `host_call_check.py` is:

```python
import torch
from arraybridge import torch as on_torch

@on_torch
def add_one(image):
    return image + 1

add_one(torch.ones(2, 3, 3))
assert not torch.cuda.is_initialized(), "a host call initialised CUDA"
if torch.cuda.is_available():
    assert add_one(torch.ones(2, 3, 3, device="cuda:0")).is_cuda
print("ok")
```

## Interpreting CI

The main CI workflow runs the test suite across its declared Python and OS
matrix. The manual GPU-named workflow installs CPU-capable Torch and JAX on a
standard runner; it does not prove CUDA behavior. Real multi-framework and
multi-device results therefore remain a required release check on suitable
hardware.
