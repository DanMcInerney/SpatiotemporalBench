"""Tolerant reads of the files of a run folder, for `stbench view` (the one owner of how the viewer reads a file).

A run that was killed, is still going or was made by another version leaves files that are missing, cut in the middle of
a line or not what `docs/format.md` says. `stbench view` exists to look at exactly those runs, so none of these raises:
a missing file is `None` or `[]`, and a line that cannot be read is skipped with a sentence in the caller's `problems`
list that the page shows. `NaN` and `Infinity`, which Python writes and a browser's `JSON.parse` refuses, read as null.
"""

from __future__ import annotations

import json
from pathlib import Path


def _loads(text: str):
    return json.loads(text, parse_constant=lambda _: None)


def json_object(path: Path, problems: list[str] | None = None) -> dict | None:
    """The JSON object in PATH, None when the file is missing or is not an object (said in `problems` when the file
    exists)."""
    try:
        found = _loads(path.read_text(encoding="utf-8", errors="replace"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        if problems is not None:
            problems.append(f"{path.name}: cannot be read ({type(error).__name__})")
        return None
    if isinstance(found, dict):
        return found
    if problems is not None:
        problems.append(f"{path.name}: is not a JSON object")
    return None


def jsonl(path: Path, problems: list[str]) -> list[dict]:
    """The JSON objects of PATH, one a line, in file order; [] for a missing file. A line that is not an object is left
    out and named in `problems`."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return []
    except OSError as error:
        problems.append(f"{path.name}: cannot be read ({type(error).__name__})")
        return []
    out = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            found = _loads(line)
        except ValueError:
            found = None
        if isinstance(found, dict):
            out.append(found)
        else:
            problems.append(f"{path.name} line {number}: not a JSON object")
    return out
