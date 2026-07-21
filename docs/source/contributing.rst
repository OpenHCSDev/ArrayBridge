Contributing
============

Install the editable development environment and run the same core checks used
by the repository:

.. code-block:: bash

   python -m pip install -e ".[dev,docs]"
   ruff check src tests
   black --check src tests
   mypy src --ignore-missing-imports
   python -m pytest
   python -m sphinx -E -W --keep-going -b html docs/source docs/source/_build/html

Framework extensions
--------------------

A framework extension must update the nominal ``MemoryType`` declaration and
the owning conversion/framework strategies. Add detection tests, conversion
pair tests, stack tests, and optional-dependency behavior. Consumers should not
maintain a second framework-name registry.

See :doc:`advanced_topics` for the ownership model and the repository's
`contribution guide
<https://github.com/OpenHCSDev/arraybridge/blob/main/CONTRIBUTING.md>`_ for the
review workflow.
