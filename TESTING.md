# Testing ArrayBridge

This is the practical test guide. The workflow files under `.github/workflows`
are the authority for the current CI matrix.

## Local core checks

```bash
python -m pip install -e ".[dev,docs]"
ruff check src tests
black --check src tests
mypy src --ignore-missing-imports
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
- enumerate framework-local devices through each `MemoryType`;
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

## Interpreting CI

The main CI workflow runs the test suite across its declared Python and OS
matrix. The manual GPU-named workflow installs CPU-capable Torch and JAX on a
standard runner; it does not prove CUDA behavior. Real multi-framework and
multi-device results therefore remain a required release check on suitable
hardware.
