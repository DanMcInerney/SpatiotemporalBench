"""The documents that present the release name the package's own version.

The version is read from `pyproject.toml`, never typed here.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
MAJOR_MINOR = ".".join(VERSION.split(".")[:2])


def read(*parts: str) -> str:
    return ROOT.joinpath(*parts).read_text(encoding="utf-8")


def test_the_readme_installs_and_fetches_this_version():
    """Contract: the README's banner says it describes this version, and its quick start installs a package of at least
    this minor version and downloads this version's tag of the dataset. Regression caught: a version bump that leaves
    the quick start installing an older package or fetching the data of the release before, whose question ids and
    credit the README no longer describes."""
    readme = read("README.md")
    assert f"> **Version {VERSION}.**" in readme
    assert f'pip install "spatiotemporalbench>={MAJOR_MINOR}"' in readme
    assert f"--revision v{VERSION} " in readme


def test_the_technical_report_and_the_changelog_are_for_this_version():
    """Contract: `docs/report.md` is this version's technical report (its title and the dataset revision it names), the
    CHANGELOG's newest released entry is this version (an `[Unreleased]` section above it, Keep a Changelog's draft of
    the next release, is not a version), and the previous report is kept under `docs/history/`. Regression caught: a
    release whose report still describes the one before it, or whose changelog has no entry for it."""
    report = read("docs", "report.md")
    assert report.startswith(f"# SpatiotemporalBench {VERSION}: technical report\n")
    assert f"revision `v{VERSION}`" in report
    newest = re.search(r"^## \[(?!Unreleased\])([^\]]+)\]", read("CHANGELOG.md"), re.MULTILINE)
    assert newest is not None and newest.group(1) == VERSION
    assert list((ROOT / "docs" / "history").glob("report-*.md"))
