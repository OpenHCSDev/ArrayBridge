# ArrayBridge documentation source

The active Sphinx source is this directory.

From the repository root:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -E -W --keep-going -b html docs/source docs/source/_build/html
```

Narrative pages document public behavior and ownership. `api_reference.rst`
tracks the root export surface. Keep code examples valid under
`scripts/validate_docs.py` in the OpenHCS superproject and verify signatures
against `src/arraybridge/` when public APIs change.
