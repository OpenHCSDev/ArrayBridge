Continuous integration
======================

The repository workflows are authoritative. The main test matrix currently
covers Python 3.10 through 3.13 on Linux, Windows, and macOS. Separate jobs
exercise optional GPU paths and code quality.

Before opening a change, run:

.. code-block:: bash

   python -m pytest
   ruff check src tests
   black --check src tests
   mypy src --ignore-missing-imports

Documentation changes should also build with warnings as errors:

.. code-block:: bash

   python -m sphinx -E -W --keep-going -b html docs/source docs/source/_build/html

See the current `GitHub Actions workflows
<https://github.com/OpenHCSDev/arraybridge/actions>`_ instead of copying a
workflow matrix into downstream documentation.
