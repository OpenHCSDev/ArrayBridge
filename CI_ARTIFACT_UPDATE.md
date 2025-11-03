# GitHub Actions upload-artifact v3 → v4 Migration

## Issue
GitHub deprecated `actions/upload-artifact@v3` effective April 16, 2024. The CI workflows were failing with:
```
Error: This request has been automatically failed because it uses a deprecated version of `actions/upload-artifact: v3`.
```

## Solution
Updated all `upload-artifact` action references from `v3` to `v4` in CI workflows.

## Files Updated

### 1. `.github/workflows/ci.yml`
**Line 91**: Updated GPU test artifact upload
```yaml
# Before
uses: actions/upload-artifact@v3

# After
uses: actions/upload-artifact@v4
```

### 2. `.github/workflows/gpu-tests.yml`
**Line 41**: Updated standalone GPU test artifact upload
```yaml
# Before
uses: actions/upload-artifact@v3

# After
uses: actions/upload-artifact@v4
```

## Changes Made
- ✅ Both workflow files now use `actions/upload-artifact@v4`
- ✅ Artifact upload configuration remains the same
- ✅ CI workflows will no longer fail due to deprecated action

## Reference
- [GitHub Blog: Deprecation Notice for Artifact Actions](https://github.blog/changelog/2024-04-16-deprecation-notice-v3-of-the-artifact-actions/)
- [Upload Artifact v4 Documentation](https://github.com/actions/upload-artifact/releases/tag/v4)

## Verification
The GPU test CI job should now:
1. ✅ Run without the deprecation error
2. ✅ Successfully upload test results and coverage reports
3. ✅ Display artifacts in the GitHub Actions UI
