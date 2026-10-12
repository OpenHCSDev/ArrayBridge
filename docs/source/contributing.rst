Contributing
============

Install the editable development environment and run the same core checks used
by the repository:

.. code-block:: bash

   python -m pip install -e ".[dev,docs]"
   ruff check src tests
   black --check src tests
   mypy src stubs
   python -m pytest
   python -m sphinx -E -W --keep-going -b html docs/source docs/source/_build/html

Adding a library
----------------

A new library is one ``ArrayLibrary`` subclass. Add tests for detection,
conversion against the other libraries, stacking, its devices and the case
where the package is not installed.

See :doc:`advanced_topics` for how the class is put together and the
repository's
`contribution guide
<https://github.com/OpenHCSDev/arraybridge/blob/main/CONTRIBUTING.md>`_ for the
review workflow.
