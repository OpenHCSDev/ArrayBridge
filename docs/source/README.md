# Sphinx Documentation for arraybridge

This directory contains the Sphinx documentation source files for arraybridge.

## Building the Documentation

### Prerequisites

Install Sphinx and required extensions:

```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
```

Or install all doc dependencies:

```bash
pip install -e ".[docs]"
```

### Build HTML Documentation

```bash
cd docs/sphinx
make html
```

The built documentation will be in `_build/html/`. Open `_build/html/index.html` in your browser.

### Build Other Formats

```bash
# PDF (requires LaTeX)
make latexpdf

# ePub
make epub

# Plain text
make text

# See all options
make help
```

### Clean Build Artifacts

```bash
make clean
```

## Documentation Structure

```
docs/sphinx/
├── conf.py                 # Sphinx configuration
├── index.rst              # Documentation home page
├── quickstart.rst         # Quick start guide
├── api/                   # API reference
│   ├── index.rst
│   ├── converters.rst
│   ├── decorators.rst
│   ├── types.rst
│   ├── exceptions.rst
│   └── utils.rst
├── examples/              # Usage examples
│   ├── index.rst
│   ├── basic_conversion.rst
│   ├── decorators.rst
│   └── multi_framework.rst
├── contributing.rst       # Contributing guide
└── ci-cd.rst             # CI/CD documentation
```

## Contributing to Documentation

1. **Edit .rst files** for content changes
2. **Update docstrings** in source code for API changes
3. **Build locally** to verify changes
4. **Check for warnings** during build

### Writing reStructuredText

Basic syntax:

```rst
Title
=====

Section
-------

Subsection
~~~~~~~~~~

**bold** *italic* ``code``

- Bullet list
- Item 2

1. Numbered list
2. Item 2

.. code-block:: python

   # Python code block
   import arraybridge

:doc:`link-to-other-doc`
```

### Autodoc

API documentation is automatically generated from docstrings using autodoc:

```rst
.. automodule:: arraybridge.converters
   :members:
   :undoc-members:
```

Make sure all public APIs have proper docstrings in Google style.

## Viewing Locally

After building:

```bash
# Open in default browser (Linux)
xdg-open _build/html/index.html

# macOS
open _build/html/index.html

# Windows
start _build/html/index.html
```

Or use Python's built-in server:

```bash
cd _build/html
python -m http.server 8000
# Visit http://localhost:8000
```
