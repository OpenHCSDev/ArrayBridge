"""Shared fixtures: every test starts with clean process-wide library state."""

import importlib.util
import os
import subprocess
import sys
import threading

import pytest

from arraybridge.conversion import ConversionObserver, ConversionRecord
from arraybridge.libraries import ArrayLibrary, CudaLibrary


@pytest.fixture(autouse=True)
def process_state_restored(monkeypatch):
    """Undo what library calls leave behind in the process.

    Import defaults written to ``os.environ`` are removed again, and the CUDA
    streams a test's fake module created for this thread are forgotten.
    """
    monkeypatch.setattr(CudaLibrary, "_thread_streams", threading.local())
    environment = dict(os.environ)
    yield
    for name in set(os.environ) - set(environment):
        del os.environ[name]
    os.environ.update(environment)


@pytest.fixture
def absent_libraries(monkeypatch):
    """Every array library but NumPy is neither loaded nor installed; imports fail the test."""

    for library in ArrayLibrary.all():
        if library.name != "numpy":
            monkeypatch.delitem(sys.modules, library.import_name, raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    import_module = importlib.import_module

    def fail_import(name, package=None):
        if name == "numpy":
            return import_module(name, package)
        raise AssertionError(f"unexpected import of {name}")

    monkeypatch.setattr(importlib, "import_module", fail_import)


@pytest.fixture
def loaded_libraries_after():
    """Run code in a fresh interpreter; the array-library modules it left loaded."""

    def run(code):
        report = (
            "import sys\n"
            "from arraybridge.libraries import ArrayLibrary\n"
            "names = {library.import_name for library in ArrayLibrary.all()}\n"
            "print(sorted(names & set(sys.modules)))\n"
        )
        finished = subprocess.run(
            [sys.executable, "-c", code + report], capture_output=True, text=True, check=False
        )
        assert finished.returncode == 0, finished.stderr
        return finished.stdout.strip()

    return run


class RecordedConversions(ConversionObserver):
    """Keeps every record it is given."""

    def __init__(self):
        self.records: list[ConversionRecord] = []

    def record(self, record):
        self.records.append(record)


@pytest.fixture
def recorded():
    """The records of the conversions made during the test."""

    with RecordedConversions().installed() as observer:
        yield observer.records
