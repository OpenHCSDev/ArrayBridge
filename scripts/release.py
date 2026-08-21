#!/usr/bin/env python3
"""Validate, tag, and push an already-published ArrayBridge commit."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from packaging.version import Version

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_INIT = PROJECT_ROOT / "src/arraybridge/__init__.py"
PYPI_METADATA_URL = "https://pypi.org/pypi/arraybridge/json"


def run_git(*arguments: str, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    """Run Git against the project checkout."""
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        capture_output=capture_output,
        text=True,
        check=True,
    )


def get_current_version() -> Version:
    """Read the package version projection."""
    match = re.search(
        r"^__version__\s*=\s*['\"]([^'\"]+)['\"]",
        PACKAGE_INIT.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if match is None:
        raise RuntimeError("Package version declaration is missing")
    return Version(match.group(1))


def get_pypi_version() -> Version:
    """Read the published version, failing closed when PyPI cannot be checked."""
    try:
        with urlopen(PYPI_METADATA_URL, timeout=10) as response:
            metadata = json.load(response)
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError("Could not verify the current PyPI version") from error
    return Version(metadata["info"]["version"])


def ensure_release_commit_is_published() -> None:
    """Require a clean checkout whose HEAD is exactly the published main tip."""
    if run_git("status", "--porcelain", capture_output=True).stdout:
        raise RuntimeError("Commit every release change before tagging")

    run_git("fetch", "origin", "main", "--tags")
    head = run_git("rev-parse", "HEAD", capture_output=True).stdout.strip()
    origin_main = run_git("rev-parse", "origin/main", capture_output=True).stdout.strip()
    if head != origin_main:
        raise RuntimeError("Release HEAD must exactly match origin/main")


def validate_release() -> None:
    """Run the repository's build and metadata release gates."""
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/verify_release_ready.py")],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag the validated ArrayBridge release")
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation")
    args = parser.parse_args()

    current_version = get_current_version()
    pypi_version = get_pypi_version()
    print(f"Current package version: {current_version}")
    print(f"Current PyPI version: {pypi_version}")

    if current_version <= pypi_version:
        raise RuntimeError(
            f"Current version {current_version} must be greater than PyPI version {pypi_version}"
        )

    ensure_release_commit_is_published()
    validate_release()

    tag = f"v{current_version}"
    if run_git("tag", "--list", tag, capture_output=True).stdout.strip():
        raise RuntimeError(f"Tag {tag} already exists")

    response = "y" if args.yes else input(f"Create release for {tag}? [y/N] ")
    if response.lower() != "y":
        print("Aborted.")
        return
    run_git("tag", "-a", tag, "-m", f"Release version {current_version}")
    run_git("push", "origin", tag)

    print(f"\nSuccessfully created and pushed tag {tag}")
    print("GitHub Actions workflow should start automatically.")
    print("Monitor progress at: https://github.com/OpenHCSDev/arraybridge/actions")


def entrypoint() -> None:
    """Present release precondition failures without an implementation traceback."""
    try:
        main()
    except (KeyError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    entrypoint()
