CI/CD Documentation
===================

Please refer to the `CI/CD Documentation <https://github.com/trissim/arraybridge/blob/main/docs/ci-cd.md>`_ for detailed CI/CD information.

Overview
--------

arraybridge uses GitHub Actions for continuous integration and deployment:

* **CI Workflow**: Runs tests across Python 3.9-3.12, multiple OS, and framework combinations
* **Publish Workflow**: Automatically publishes to PyPI on version tags

Quick Commands
--------------

Run tests locally:

.. code-block:: bash

   pytest
   pytest --cov=arraybridge --cov-report=html

Code quality checks:

.. code-block:: bash

   black src/ tests/
   ruff check src/ tests/
   mypy src/ --ignore-missing-imports

See the full CI/CD documentation for detailed information about workflows, test matrix, and troubleshooting.
