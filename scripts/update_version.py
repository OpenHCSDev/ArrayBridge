#!/usr/bin/env python3
"""Update ArrayBridge's canonical package version without Git side effects."""

import argparse
import re
from pathlib import Path

from packaging.version import InvalidVersion, Version

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = PROJECT_ROOT / "src/arraybridge/__init__.py"
VERSION_PATTERN = r"(__version__\s*=\s*['\"])([^'\"]+)(['\"])"


def current_version() -> Version:
    """Return the package-owned canonical version."""

    match = re.search(
        VERSION_PATTERN,
        VERSION_FILE.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"Version declaration is missing from {VERSION_FILE}")
    return Version(match.group(2))


def update_version(version_text: str) -> None:
    try:
        requested = Version(version_text)
    except InvalidVersion as error:
        raise ValueError(f"Invalid version: {version_text}") from error
    current = current_version()
    if requested <= current:
        raise ValueError(f"New version {requested} must be greater than {current}")

    content = VERSION_FILE.read_text(encoding="utf-8")
    updated, count = re.subn(
        VERSION_PATTERN,
        rf"\g<1>{requested}\g<3>",
        content,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError(f"Expected one version declaration in {VERSION_FILE}, found {count}")
    VERSION_FILE.write_text(updated, encoding="utf-8")

    print(f"Updated ArrayBridge version to {requested}")
    print("Run scripts/verify_release_ready.py --allow-dirty, then review and commit.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="New release version, for example 0.3.0")
    args = parser.parse_args()
    update_version(args.version)


if __name__ == "__main__":
    main()
