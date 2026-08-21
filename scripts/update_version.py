#!/usr/bin/env python3
"""Update ArrayBridge's three version projections without Git side effects."""

import argparse
import re
from pathlib import Path

from packaging.version import InvalidVersion, Version

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION_FILES = {
    PROJECT_ROOT
    / "src/arraybridge/__init__.py": (
        r"(__version__\s*=\s*['\"])([^'\"]+)(['\"])",
        "package",
    ),
    PROJECT_ROOT
    / "pyproject.toml": (
        r"(^version\s*=\s*['\"])([^'\"]+)(['\"])",
        "project",
    ),
    PROJECT_ROOT
    / "docs/source/conf.py": (
        r"(^release\s*=\s*['\"])([^'\"]+)(['\"])",
        "documentation",
    ),
}


def current_version() -> Version:
    """Return the agreed version, failing when a projection has drifted."""
    declared_versions: dict[str, Version] = {}
    for path, (pattern, projection_name) in VERSION_FILES.items():
        match = re.search(pattern, path.read_text(encoding="utf-8"), flags=re.MULTILINE)
        if match is None:
            raise RuntimeError(f"Version declaration is missing from {path}")
        declared_versions[projection_name] = Version(match.group(2))

    unique_versions = set(declared_versions.values())
    if len(unique_versions) != 1:
        declarations = ", ".join(f"{name}={version}" for name, version in declared_versions.items())
        raise RuntimeError(f"Version projections have drifted: {declarations}")
    return unique_versions.pop()


def update_version(version_text: str) -> None:
    try:
        requested = Version(version_text)
    except InvalidVersion as error:
        raise ValueError(f"Invalid version: {version_text}") from error
    current = current_version()
    if requested <= current:
        raise ValueError(f"New version {requested} must be greater than {current}")

    for path, (pattern, _projection_name) in VERSION_FILES.items():
        content = path.read_text(encoding="utf-8")
        updated, count = re.subn(
            pattern,
            rf"\g<1>{requested}\g<3>",
            content,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise RuntimeError(f"Expected one version declaration in {path}, found {count}")
        path.write_text(updated, encoding="utf-8")

    print(f"Updated ArrayBridge version projections to {requested}")
    print("Run scripts/verify_release_ready.py --allow-dirty, then review and commit.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="New release version, for example 0.3.0")
    args = parser.parse_args()
    update_version(args.version)


if __name__ == "__main__":
    main()
