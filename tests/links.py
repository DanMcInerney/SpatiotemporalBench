"""Directory links for the tests of what must not follow them (VISION rule 6): a symlink where the platform lets this
user make one, else, on Windows, a junction (which needs no privilege and is a different kind of reparse point)."""

from __future__ import annotations

import os
from pathlib import Path


def junction(link: Path, target: Path) -> None:
    """A Windows junction at `link` pointing at the existing folder `target` (absolute)."""
    import _winapi

    _winapi.CreateJunction(str(target), str(link))


def dir_link(link: Path, target: Path) -> None:
    """A link at `link` pointing at the existing folder `target` (absolute)."""
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        junction(link, target)
