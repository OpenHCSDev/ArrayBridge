# PR Merge Conflicts - Analysis and Resolution

## Summary
Your PR has **3 merge conflicts** in the following files:
1. `src/arraybridge/decorators.py` - 1 conflict (import statements)
2. `src/arraybridge/stack_utils.py` - 1 conflict (import statements)
3. `src/arraybridge/utils.py` - 3 conflicts (docstring and import placement)

All conflicts stem from the same root cause: **package renaming from `openhcs` to `arraybridge`** that happened in the main branch.

---

## Conflict 1: decorators.py (Line 22-31)

### The Conflict
```python
<<<<<<< HEAD
from arraybridge.types import MemoryType, VALID_MEMORY_TYPES
from arraybridge.utils import optional_import
from arraybridge.oom_recovery import _execute_with_oom_recovery
from arraybridge.framework_ops import _FRAMEWORK_OPS
=======
import numpy as np
from openhcs.constants.constants import MemoryType

>>>>>>> origin/main
```

### Analysis
- **Your branch (HEAD)**: Uses `arraybridge.types` (correct - package was renamed)
- **Main branch**: Uses `openhcs.constants.constants` (old package name) and has extra `numpy` import

### Resolution
**Keep HEAD version** but ensure `numpy` import exists elsewhere if needed:
```python
from arraybridge.types import MemoryType, VALID_MEMORY_TYPES
from arraybridge.utils import optional_import
from arraybridge.oom_recovery import _execute_with_oom_recovery
from arraybridge.framework_ops import _FRAMEWORK_OPS
```

Note: Check if `numpy as np` is needed later in the file. If so, add it after these imports.

---

## Conflict 2: stack_utils.py (Line 15-21)

### The Conflict
```python
<<<<<<< HEAD
from arraybridge.types import GPU_MEMORY_TYPES, MemoryType
=======
from openhcs.constants.constants import GPU_MEMORY_TYPES, MemoryType

>>>>>>> origin/main
```

### Analysis
- **Your branch**: Uses `arraybridge.types` (correct)
- **Main branch**: Uses `openhcs.constants.constants` (old name)

### Resolution
**Keep HEAD version** - simple and straightforward:
```python
from arraybridge.types import GPU_MEMORY_TYPES, MemoryType
```

---

## Conflict 3: utils.py (3 sub-conflicts)

This is the most complex conflict with multiple merge markers.

### Sub-conflict 3a: Docstring (Lines 2-10)

```python
<<<<<<< HEAD
Memory conversion utility functions for arraybridge.

This module provides utility functions for memory conversion operations.
=======
Memory conversion utility functions for OpenHCS.

This module provides utility functions for memory conversion operations,
supporting Clause 251 (Declarative Memory Conversion Interface) and
Clause 65 (Fail Loudly).
>>>>>>> origin/main
```

**Resolution**: Keep main's more detailed docstring but update package name:
```python
"""
Memory conversion utility functions for arraybridge.

This module provides utility functions for memory conversion operations,
supporting Clause 251 (Declarative Memory Conversion Interface) and
Clause 65 (Fail Loudly).
"""
```

### Sub-conflict 3b: Import Placement (Lines 19-27 and 94-100)

This is the **critical conflict**. The issue is about import order:

- **Your branch (HEAD)**: Imports are at top (before `optional_import` function)
- **Main branch**: Imports are at bottom (after `optional_import` function)

```python
# HEAD version (top placement):
from typing import Any, Optional

# Then later at line 94:
from arraybridge.types import MemoryType
from .exceptions import MemoryConversionError
from .framework_config import _FRAMEWORK_CONFIG
logger = logging.getLogger(__name__)
```

```python
# Main version (better placement):
from typing import Any, Optional

from arraybridge.types import MemoryType
from .exceptions import MemoryConversionError
from .framework_config import _FRAMEWORK_CONFIG
logger = logging.getLogger(__name__)

# Then _ModulePlaceholder class and optional_import function
```

**Resolution**: Keep **main branch's placement** (imports at line 19, NOT line 94). This is better because:
1. Standard Python convention: imports at the top
2. `optional_import` doesn't depend on these imports
3. Avoids potential circular import issues

---

## Step-by-Step Resolution Instructions

### Option 1: Manual Resolution (Recommended for Learning)

1. **Merge main into your branch:**
   ```bash
   git fetch origin
   git merge origin/main
   ```

2. **Resolve decorators.py:**
   ```bash
   # Edit the file and keep:
   from arraybridge.types import MemoryType, VALID_MEMORY_TYPES
   from arraybridge.utils import optional_import
   from arraybridge.oom_recovery import _execute_with_oom_recovery
   from arraybridge.framework_ops import _FRAMEWORK_OPS

   # Remove conflict markers and the openhcs import
   ```

3. **Resolve stack_utils.py:**
   ```bash
   # Keep:
   from arraybridge.types import GPU_MEMORY_TYPES, MemoryType
   ```

4. **Resolve utils.py:**
   - Update docstring to mention "arraybridge" with full description
   - Place imports at line 19 (after `from typing import Any, Optional`)
   - Remove duplicate imports at line 94

5. **Mark as resolved and commit:**
   ```bash
   git add src/arraybridge/decorators.py
   git add src/arraybridge/stack_utils.py
   git add src/arraybridge/utils.py
   git commit -m "Merge main and resolve package rename conflicts"
   git push
   ```

### Option 2: Accept Main's Changes Then Re-apply Docs

If the conflicts are too complex:

1. **Accept main's version completely:**
   ```bash
   git fetch origin
   git merge origin/main -X theirs
   ```

2. **Verify the code still works:**
   ```bash
   python -c "import arraybridge; print(arraybridge.__version__)"
   ```

3. **Your documentation changes should still be intact** (they don't conflict)

4. **Commit and push:**
   ```bash
   git push
   ```

---

## Why These Conflicts Occurred

The main branch underwent a **package rename from `openhcs` to `arraybridge`**. Your documentation branch was based on an earlier commit (before the rename), so when you try to merge:

- Your branch has the OLD package structure
- Main branch has the NEW package structure
- Git can't auto-merge because import paths are different

---

## Verification After Resolution

After resolving conflicts, verify everything works:

```bash
# 1. Check imports work
python -c "from arraybridge.types import MemoryType; print(MemoryType.NUMPY)"

# 2. Build docs to ensure no broken references
cd docs && make html

# 3. Run a basic test
python -c "from arraybridge import detect_memory_type; import numpy as np; print(detect_memory_type(np.array([1,2,3])))"
```

---

## Recommendation

**Use Option 1 (Manual Resolution)** with this exact resolution:

1. **decorators.py**: Keep HEAD (your imports from `arraybridge.*`)
2. **stack_utils.py**: Keep HEAD (your imports from `arraybridge.*`)
3. **utils.py**:
   - Docstring: Use main's detailed version but say "arraybridge"
   - Imports: Use main's placement (line 19, not line 94)
   - Content: Both versions are essentially the same after imports

This preserves your work while properly integrating with the package rename.
