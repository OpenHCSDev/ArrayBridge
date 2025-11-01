# Exact Resolved File Sections

## File 1: src/arraybridge/decorators.py (Lines 22-36)

Replace lines 22-36 with:

```python
from arraybridge.types import MemoryType, VALID_MEMORY_TYPES
from arraybridge.utils import optional_import
from arraybridge.oom_recovery import _execute_with_oom_recovery
from arraybridge.framework_ops import _FRAMEWORK_OPS
from arraybridge.dtype_scaling import SCALING_FUNCTIONS
from arraybridge.framework_ops import _FRAMEWORK_OPS
from arraybridge.oom_recovery import _execute_with_oom_recovery
from arraybridge.slice_processing import process_slices
from arraybridge.utils import optional_import

logger = logging.getLogger(__name__)
```

**Note**: There appear to be duplicate imports. Check if they're needed or if this is an artifact.

---

## File 2: src/arraybridge/stack_utils.py (Lines 15-21)

Replace lines 15-21 with:

```python
from arraybridge.types import GPU_MEMORY_TYPES, MemoryType
from arraybridge.converters import detect_memory_type
from arraybridge.framework_config import _FRAMEWORK_CONFIG
from arraybridge.utils import optional_import

logger = logging.getLogger(__name__)
```

---

## File 3: src/arraybridge/utils.py (Complete resolved version)

The entire beginning of the file should look like this:

```python
"""
Memory conversion utility functions for arraybridge.

This module provides utility functions for memory conversion operations,
supporting Clause 251 (Declarative Memory Conversion Interface) and
Clause 65 (Fail Loudly).
"""

import importlib
import logging
from typing import Any, Optional

from arraybridge.types import MemoryType

from .exceptions import MemoryConversionError
from .framework_config import _FRAMEWORK_CONFIG

logger = logging.getLogger(__name__)


class _ModulePlaceholder:
    """
    Placeholder for missing optional modules that allows attribute access
    for type annotations while still being falsy and failing on actual use.
    """
    def __init__(self, module_name: str):
        self._module_name = module_name

    def __bool__(self):
        return False

    def __getattr__(self, name):
        # Return another placeholder for chained attribute access
        # This allows things like cp.ndarray in type annotations to work
        return _ModulePlaceholder(f"{self._module_name}.{name}")

    def __call__(self, *args, **kwargs):
        # If someone tries to actually call a function, fail loudly
        raise ImportError(
            f"Module '{self._module_name}' is not available. "
            f"Please install the required dependency."
        )

    def __repr__(self):
        return f"<ModulePlaceholder for '{self._module_name}'>"


def optional_import(module_name: str) -> Optional[Any]:
    """
    Import a module if available, otherwise return a placeholder that handles
    attribute access gracefully for type annotations but fails on actual use.

    This function allows for graceful handling of optional dependencies.
    It can be used to import libraries that may not be installed,
    particularly GPU-related libraries like torch, tensorflow, and cupy.

    Args:
        module_name: Name of the module to import

    Returns:
        The imported module if available, a placeholder otherwise

    Example:
        ```python
        # Import torch if available
        torch = optional_import("torch")

        # Check if torch is available before using it
        if torch:
            # Use torch
            tensor = torch.tensor([1, 2, 3])
        else:
            # Handle the case where torch is not available
            raise ImportError("PyTorch is required for this function")
        ```
    """
    try:
        # Use importlib.import_module which handles dotted names properly
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError, AttributeError):
        # Return a placeholder that handles attribute access gracefully
        return _ModulePlaceholder(module_name)


def _ensure_module(module_name: str) -> Any:
    # ... rest of the file continues unchanged ...
```

**Key points:**
1. Docstring mentions "arraybridge" (not "OpenHCS")
2. Imports are placed RIGHT AFTER the `from typing` line
3. NO duplicate imports at line 94
4. The rest of the file remains unchanged

---

## Quick Copy-Paste Commands

If you want to resolve this quickly using command line:

### For decorators.py:
```bash
# Just keep the arraybridge imports, remove openhcs
sed -i '/<<<<<<< HEAD/,/>>>>>>> origin\/main/c\
from arraybridge.types import MemoryType, VALID_MEMORY_TYPES\
from arraybridge.utils import optional_import\
from arraybridge.oom_recovery import _execute_with_oom_recovery\
from arraybridge.framework_ops import _FRAMEWORK_OPS' src/arraybridge/decorators.py
```

### For stack_utils.py:
```bash
sed -i '/<<<<<<< HEAD/,/>>>>>>> origin\/main/c\
from arraybridge.types import GPU_MEMORY_TYPES, MemoryType' src/arraybridge/stack_utils.py
```

### For utils.py:
This one requires manual editing due to multiple conflicts. Recommended to use a merge tool or edit manually.

---

## Validation Script

After resolving, run this to verify:

```bash
#!/bin/bash
echo "Checking for remaining conflict markers..."
if grep -r "<<<<<<< HEAD\|======= \|>>>>>>> origin/main" src/arraybridge/*.py; then
    echo "❌ Still have conflicts!"
    exit 1
else
    echo "✅ All conflicts resolved!"
fi

echo "Testing imports..."
python3 -c "
from arraybridge.types import MemoryType
from arraybridge.utils import optional_import
from arraybridge.stack_utils import stack_slices
print('✅ All imports work!')
"

echo "Building docs..."
cd docs && make clean && make html
if [ $? -eq 0 ]; then
    echo "✅ Docs build successfully!"
else
    echo "❌ Docs build failed!"
    exit 1
fi
```
