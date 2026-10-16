"""The leak scan: nothing under an agent-facing script folder may carry truth.

`leak_scan(script_dir, names)` takes `DATASET/scripts/<sid>` and the names the layout holds (the script's map: read by
the caller from its timeline, never guessed from the path) and raises LeakError if it finds:
- a file that is not on the allowlist (unlisted_file) or a JSON key that is not on that file's allowlist
  (unlisted_key); the allowlist is the contract of what a robot may be given;
- a file whose path names depth (`depth`): the agent is fed the camera, and the recorded depth is the scorer's, kept in
  `private/<sid>/depth/` (0.6.0);
- an entity id (`E` followed by 4 or more digits), a region id (`R1`), the word "truth" or the map's name, in a file's
  name, its text, or the metadata of its images (PNG text chunks and other ancillary chunks, JPEG COM and APPn).
A pose row's `subgoal` (a job's words for what the robot is doing) is an allowed key whose text is scanned by the second
rule like any other text of the file.

Seeds are not pattern-matched: script ids are random and the key allowlists keep such fields out. `ALLOWED` below is the
one owner of the allowlist: a writer of a new agent-facing file gets its entry here (through the coordinator, plan
§2.8), nothing registers at import, and a scan in a process that imported no writer finds the same table.
"""

from __future__ import annotations

import contextlib
import fnmatch
import json
import re
import zlib
from pathlib import Path
from typing import NamedTuple

from . import robots

TOKEN = r"(?<![A-Za-z0-9])"
RULES = {
    "entity_id": re.compile(TOKEN + r"E\d{4,}"),
    "region_id": re.compile(TOKEN + r"R\d+(?![A-Za-z0-9])"),
    "truth": re.compile("truth", re.IGNORECASE),
}
FREE_KEYS = "attributes"  # the keys inside an `attributes` dict are descriptive words, not part of the schema

# Exactly the keys `perception.build` writes: a row's three, a detection's, and a door's `state` with its `open` flag.
PERCEPTION_KEYS = frozenset({"k", "clock", "detections", "d", "kind", "attributes", "bbox_px", "center_cam",
                             "center_map", "extent_m", "floor_hull", "state", "open"})
ROBOT_KEYS = robots.DOC_KEYS  # every key of `robots.robot_doc`, the phase names and status keys of every robot included
CAMERA_KEYS = frozenset({"width", "height", "hfov_deg", "fx", "fy", "cx", "cy", "depth_unit", "near_m", "far_m"})
# `subgoal` is a job's own words for what the robot is doing (`JobSpec.subgoals`). The key is allowed, and the text
# rules read its value with the rest of the file: an entity id, a region id or the map's name in it is a leak.
POSE_KEYS = frozenset({"k", "clock", "pose", "camera", "pos", "rot", "yaw_deg", "pitch_deg", "status", "phase", "task",
                       "subgoal"} | {key for r in robots.ROBOTS.values() for key in r.status.keys})
# file pattern (posix, relative to the script folder) -> allowed JSON keys; None means free text or binary
ALLOWED: dict[str, frozenset[str] | None] = {
    "robot.json": ROBOT_KEYS,
    "sensors/camera.json": CAMERA_KEYS,
    "sensors/pose.jsonl": POSE_KEYS,
    "sensors/rgb/*.jpg": None,
    "perception/oracle.jsonl": PERCEPTION_KEYS,
}
DEPTH = "depth"   # the agent is fed the camera: a file named for depth under `scripts/` is a leak (0.6.0)
TEXT = {".md", ".txt"}
JSON_LIKE = {".json", ".jsonl"}
BENIGN_PNG = {b"IHDR", b"PLTE", b"IDAT", b"IEND"}


class Leak(NamedTuple):
    file: str
    rule: str
    match: str


class LeakError(Exception):
    def __init__(self, leaks: list[Leak]):
        self.leaks = leaks
        super().__init__("agent-facing files leak: " + "; ".join(f"{x.file}: {x.rule} {x.match!r}" for x in leaks[:5]))


def _text_rules(name: str, text: str, extra: list[str]) -> list[Leak]:
    out = [Leak(name, rule, m.group(0)) for rule, rx in RULES.items() for m in rx.finditer(text)]
    low = text.lower()
    return out + [Leak(name, "map_name", s) for s in extra if s and s.lower() in low]


def _walk_keys(node: object, allowed: frozenset[str], free: bool = False) -> set[str]:
    bad: set[str] = set()
    if isinstance(node, dict):
        for k, v in node.items():
            if not free and k not in allowed:
                bad.add(k)
            bad |= _walk_keys(v, allowed, free or k == FREE_KEYS)
    elif isinstance(node, list):
        for v in node:
            bad |= _walk_keys(v, allowed, free)
    return bad


def _png_texts(data: bytes) -> list[str]:
    out, i = [], 8
    while i + 8 <= len(data):
        n = int.from_bytes(data[i:i + 4], "big")
        kind, body = data[i + 4:i + 8], data[i + 8:i + 8 + n]
        i += 12 + n
        if kind in BENIGN_PNG:
            continue
        if kind == b"zTXt" and b"\x00" in body:
            head, _, rest = body.partition(b"\x00")
            with contextlib.suppress(zlib.error):
                body = head + b"\x00" + zlib.decompress(rest[1:])
        elif kind == b"iTXt":
            parts = body.split(b"\x00", 5)
            if len(parts) == 6 and parts[1][:1] == b"\x01":
                with contextlib.suppress(zlib.error):
                    body = parts[0] + b"\x00" + zlib.decompress(parts[5])
        out.append(kind.decode("latin-1") + " " + body.decode("utf-8", errors="replace"))
    return out


def _jpeg_texts(data: bytes) -> list[str]:
    out, i = [], 2
    while i + 4 <= len(data) and data[i] == 0xFF:
        marker = data[i + 1]
        if marker == 0xDA:
            break
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        n = int.from_bytes(data[i + 2:i + 4], "big")
        if 0xE0 <= marker <= 0xEF or marker == 0xFE:
            out.append(data[i + 4:i + 2 + n].decode("utf-8", errors="replace"))
        i += 2 + n
    return out


def _image_texts(path: Path) -> list[str]:
    data = path.read_bytes()
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return _png_texts(data)
    if data[:2] == b"\xff\xd8":
        return _jpeg_texts(data)
    return []


def find_leaks(folder: str | Path, names: list[str]) -> list[Leak]:
    """Every leak under the agent-facing script folder; none of `names` (the script's map) may appear in any file."""
    root = Path(folder)
    extra = list(names)
    leaks: list[Leak] = []
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = f.relative_to(root).as_posix()
        leaks += _text_rules(rel, rel, extra)
        if DEPTH in rel.lower():
            leaks.append(Leak(rel, "depth", rel))
            continue
        matches = [pat for pat in ALLOWED if fnmatch.fnmatchcase(rel, pat)]
        if not matches:
            leaks.append(Leak(rel, "unlisted_file", rel))
            continue
        allowed = ALLOWED[matches[0]]
        suffix = f.suffix.lower()
        if suffix in TEXT or suffix in JSON_LIKE:
            text = f.read_text(encoding="utf-8", errors="replace")
            leaks += _text_rules(rel, text, extra)
            if suffix in JSON_LIKE and allowed is not None:
                try:
                    docs = [json.loads(line) for line in text.splitlines() if line.strip()] if suffix == ".jsonl" \
                        else [json.loads(text)]
                except json.JSONDecodeError:
                    leaks.append(Leak(rel, "bad_json", rel))
                    continue
                bad = set().union(*(_walk_keys(d, allowed) for d in docs)) if docs else set()
                leaks += [Leak(rel, "unlisted_key", k) for k in sorted(bad)]
        else:
            for chunk in _image_texts(f):
                leaks += _text_rules(rel, chunk, extra)
    return leaks


def leak_scan(folder: str | Path, names: list[str]) -> None:
    """Raise LeakError if anything under the agent-facing script folder carries truth."""
    leaks = find_leaks(folder, names)
    if leaks:
        raise LeakError(leaks)
