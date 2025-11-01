# Visual Guide to Merge Conflicts

## The Big Picture

```
                    Initial Commit (e4919ab)
                            |
                            |
                    ┌───────┴────────┐
                    |                |
                    v                v
           Your Branch          Main Branch
    (docs changes only)   (package rename: openhcs→arraybridge)
                    |                |
                    |                |
        - Added .readthedocs.yml     - Renamed all imports
        - Created 10 .rst files      - openhcs.* → arraybridge.*
        - Updated index.rst          - Fixed code quality
        - Removed mkdocs.yml         - Updated all modules
                    |                |
                    └───────┬────────┘
                            |
                            v
                    MERGE CONFLICTS! ❌
                    (3 files affected)
```

## What Happened?

Your branch and main branch **diverged**:
- **Your work**: Documentation improvements (no code changes intended)
- **Main's work**: Package rename from `openhcs` to `arraybridge`

The problem: Your branch was based on code that still had `openhcs` imports!

---

## Conflict Anatomy

### Conflict 1: decorators.py

```
Line 1-21: ✅ No conflict
            |
Line 22:    ├──┬── ❌ CONFLICT START
            |  |
Your code:  |  ├── from arraybridge.types import MemoryType, VALID_MEMORY_TYPES
            |  |   from arraybridge.utils import optional_import
            |  |   from arraybridge.oom_recovery import _execute_with_oom_recovery
            |  |   from arraybridge.framework_ops import _FRAMEWORK_OPS
            |  |
Main code:  |  ├── import numpy as np
            |  |   from openhcs.constants.constants import MemoryType
            |  |
Line 31:    |  └── ❌ CONFLICT END
            |
Line 32+:   ✅ No conflict (both merged automatically)
```

**Why it conflicts**: Git sees completely different import lines and doesn't know which to keep.

**Resolution**: Keep YOUR version (HEAD) because:
- ✅ Uses correct package name (`arraybridge`)
- ✅ Main's version uses old name (`openhcs`)
- ✅ You can add `import numpy as np` if needed

---

### Conflict 2: stack_utils.py

```
Line 1-14: ✅ No conflict
            |
Line 15:    ├──┬── ❌ CONFLICT START
            |  |
Your code:  |  ├── from arraybridge.types import GPU_MEMORY_TYPES, MemoryType
            |  |
Main code:  |  ├── from openhcs.constants.constants import GPU_MEMORY_TYPES, MemoryType
            |  |
Line 21:    |  └── ❌ CONFLICT END
            |
Line 22+:   ✅ No conflict
```

**Why it conflicts**: Same reason - different package names.

**Resolution**: Keep YOUR version (HEAD)
- ✅ Uses `arraybridge.types` (correct)
- ❌ Main uses `openhcs.constants.constants` (old)

---

### Conflict 3: utils.py (Most Complex)

```
Line 1:     ├──┬── ❌ CONFLICT 3a: Docstring
            |  |
Your code:  |  ├── """Memory conversion utility functions for arraybridge.
            |  |    This module provides utility functions..."""
            |  |
Main code:  |  ├── """Memory conversion utility functions for OpenHCS.
            |  |    This module provides utility functions,
            |  |    supporting Clause 251..."""
            |  |
Line 10:    |  └── ❌ CONFLICT 3a END
            |
Line 11-18: ✅ No conflict
            |
Line 19:    ├──┬── ❌ CONFLICT 3b: Import placement
            |  |
Your code:  |  ├── (no imports here - they're at line 94 instead)
            |  |
Main code:  |  ├── from arraybridge.types import MemoryType
            |  |   from .exceptions import MemoryConversionError
            |  |   from .framework_config import _FRAMEWORK_CONFIG
            |  |   logger = logging.getLogger(__name__)
            |  |
Line 27:    |  └── ❌ CONFLICT 3b END
            |
Line 28-93: ✅ No conflict (_ModulePlaceholder and optional_import)
            |
Line 94:    ├──┬── ❌ CONFLICT 3c: Duplicate import placement
            |  |
Your code:  |  ├── from arraybridge.types import MemoryType
            |  |   from .exceptions import MemoryConversionError
            |  |   from .framework_config import _FRAMEWORK_CONFIG
            |  |   logger = logging.getLogger(__name__)
            |  |
Main code:  |  ├── (nothing - already at line 19)
            |  |
Line 100:   |  └── ❌ CONFLICT 3c END
            |
Line 101+:  ✅ No conflict
```

**Why it conflicts**:
1. Docstring: Different wording ("arraybridge" vs "OpenHCS")
2. Import placement: YOUR version puts imports at line 94, MAIN puts them at line 19

**Resolution**:
1. Docstring: Use MAIN's detailed version but change "OpenHCS" → "arraybridge"
2. Imports: Use MAIN's placement (line 19 is standard Python style)
3. Result: Imports appear ONCE at line 19, NOT at line 94

---

## Side-by-Side Comparison

### decorators.py Resolution

```diff
# ❌ WRONG (Main's version - old package name)
- from openhcs.constants.constants import MemoryType

# ✅ CORRECT (Your version - new package name)
+ from arraybridge.types import MemoryType, VALID_MEMORY_TYPES
+ from arraybridge.utils import optional_import
+ from arraybridge.oom_recovery import _execute_with_oom_recovery
+ from arraybridge.framework_ops import _FRAMEWORK_OPS
```

### stack_utils.py Resolution

```diff
# ❌ WRONG
- from openhcs.constants.constants import GPU_MEMORY_TYPES, MemoryType

# ✅ CORRECT
+ from arraybridge.types import GPU_MEMORY_TYPES, MemoryType
```

### utils.py Resolution

```diff
# Part 1: Docstring
# ❌ WRONG (mentions old package)
- """Memory conversion utility functions for OpenHCS."""

# ✅ CORRECT (new package + full description)
+ """
+ Memory conversion utility functions for arraybridge.
+
+ This module provides utility functions for memory conversion operations,
+ supporting Clause 251 (Declarative Memory Conversion Interface) and
+ Clause 65 (Fail Loudly).
+ """

# Part 2: Import placement
# ❌ WRONG (imports at line 94 - unusual)
  from typing import Any, Optional

- # ... 75 lines of code ...
-
- from arraybridge.types import MemoryType
- from .exceptions import MemoryConversionError

# ✅ CORRECT (imports at line 19 - standard)
  from typing import Any, Optional
+
+ from arraybridge.types import MemoryType
+ from .exceptions import MemoryConversionError
+ from .framework_config import _FRAMEWORK_CONFIG
+ logger = logging.getLogger(__name__)

  # Then _ModulePlaceholder and optional_import classes
```

---

## The Resolution Strategy

```
Step 1: Start merge
┌─────────────────────────────┐
│ git merge origin/main       │
└─────────────────────────────┘
              ↓
Step 2: Fix conflicts
┌─────────────────────────────┐
│ decorators.py: Keep HEAD    │ ← Uses arraybridge.*
│ stack_utils.py: Keep HEAD   │ ← Uses arraybridge.*
│ utils.py: Mix both          │ ← Main's placement + Your package name
└─────────────────────────────┘
              ↓
Step 3: Mark resolved
┌─────────────────────────────┐
│ git add <files>             │
└─────────────────────────────┘
              ↓
Step 4: Complete merge
┌─────────────────────────────┐
│ git commit                  │
│ git push                    │
└─────────────────────────────┘
```

---

## Quick Decision Tree

```
For each conflict, ask:

Does it import from openhcs.*?
    YES → ❌ Reject, use arraybridge.* instead
    NO  → Proceed to next question
        ↓
Is it better structured/documented?
    YES → ✅ Keep it (but verify package names)
    NO  → Consider alternative
        ↓
Are imports at top of file (after docstring)?
    YES → ✅ Keep it
    NO  → Move to top
```

---

## Summary Table

| File | Conflict Type | Your Version | Main Version | Resolution |
|------|---------------|--------------|--------------|------------|
| `decorators.py` | Import path | `arraybridge.*` ✅ | `openhcs.*` ❌ | **Keep HEAD** |
| `stack_utils.py` | Import path | `arraybridge.*` ✅ | `openhcs.*` ❌ | **Keep HEAD** |
| `utils.py` (3a) | Docstring | "arraybridge" ✅ | "OpenHCS" ❌ but detailed ✅ | **Mix**: Main's detail + your name |
| `utils.py` (3b,3c) | Import placement | Line 94 ❌ | Line 19 ✅ | **Keep MAIN** |

Legend:
- ✅ = Good choice
- ❌ = Should be changed
- **Mix** = Combine best of both

---

## After Resolution Checklist

- [ ] No conflict markers remain (`<<<<<<<`, `=======`, `>>>>>>>`)
- [ ] All imports use `arraybridge.*` (not `openhcs.*`)
- [ ] Imports are at top of files (standard Python style)
- [ ] Code runs: `python -c "import arraybridge"`
- [ ] Docs build: `cd docs && make html`
- [ ] Tests pass (if applicable)
- [ ] Committed and pushed
