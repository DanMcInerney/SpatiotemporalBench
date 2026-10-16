"""The public documents link only to files the repository ships.

Contract: every relative markdown link in the public documents (the repository `README.md`, `docs/`, `examples/`, the
READMEs of `render/` and of the maps) names a file or folder that git tracks. Regression caught: a document that
still links a file that was moved (`VISION.md` is `docs/vision.md`) or taken out of the repository (the development
records: HANDOFF, DECISIONS, FINDINGS, the plans), which a reader of the published repository meets as a dead link.
Outside a git checkout the check falls back to the file system."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ("README.md", "docs/**/*.md", "examples/**/*.md", "render/*.md", "src/stbench/maps/README.md")
LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
FENCE = re.compile(r"^(```|~~~).*?^\1", re.MULTILINE | re.DOTALL)


def tracked() -> set[str] | None:
    """The repository's tracked paths (posix), or None when git cannot say (no checkout, no git)."""
    try:
        done = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return set(filter(None, done.stdout.split("\0"))) if done.returncode == 0 and done.stdout else None


def documents() -> list[Path]:
    return sorted({p for pattern in DOCUMENTS for p in ROOT.glob(pattern) if p.is_file()})


def targets_in(text: str) -> list[str]:
    """The relative targets of the markdown links in `text`, without anchors; code fences are skipped."""
    targets = (unquote(t).split("#", 1)[0] for t in LINK.findall(FENCE.sub("", text)))
    return [t for t in targets if t and not re.match(r"^[a-z][a-z0-9+.-]*:", t, re.IGNORECASE)]


def resolves(doc: Path, target: str, files: set[str] | None) -> bool:
    path = (doc.parent / target).resolve() if not target.startswith("/") else (ROOT / target.lstrip("/")).resolve()
    try:
        rel = path.relative_to(ROOT).as_posix()
    except ValueError:
        return False
    if files is None:
        return path.exists()
    return rel in files or any(f.startswith(rel.rstrip("/") + "/") for f in files)


def test_the_link_reader_sees_links_anchors_and_external_targets():
    """Contract: the reader returns relative targets without their anchors and ignores URLs, bare anchors, images and
    fenced code. Regression caught: a pattern that stops matching (every doc then passes, vacuously) or one that
    follows a URL as if it were a path."""
    text = ("[a](format.md#frames) [b](https://x.org/y.md) [c](#top) [d](mailto:a@b.c) [e](../README.md \"title\")\n"
            "![f](figures/frames.jpg)\n```\n[g](nowhere.md)\n```\n")
    assert targets_in(text) == ["format.md", "../README.md"]


def test_the_documents_to_check_exist():
    """Contract: the globs find the documents (a moved folder would make the link test pass over nothing).
    Regression caught: `docs/` renamed or emptied, so that no link is checked."""
    names = {p.relative_to(ROOT).as_posix() for p in documents()}
    assert {"README.md", "docs/vision.md", "docs/questions.md", "docs/format.md", "render/README.md"} <= names


@pytest.mark.parametrize("doc", documents(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_every_relative_link_names_a_tracked_file(doc):
    """Contract: each relative markdown link of `doc` resolves to a tracked file or folder. Regression caught: a link
    to a moved or untracked file."""
    files = tracked()
    dead = [t for t in targets_in(doc.read_text(encoding="utf-8")) if not resolves(doc, t, files)]
    assert dead == []
