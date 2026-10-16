"""A case's stream for `stbench view`'s frame-by-frame section: the steps the harness delivered, packed as numbers.

The steps are `stream.delivered(case, channel)`'s: built by the code that streams a case to a system, from the same
files and for the same channel, so they are the steps a system on that channel received (case-relative `k` from 0, the
case's own clock, the pose and camera, the status, the detections in the order a system got them). They are not read
back from a transport log. This module only packs them for the page, once for each (case, channel) however many arms
ran it (`Streams.items`; a case holds the key of its entry), and draws the depth pictures. One entry:

    {"channel", "sends": [what the channel delivers],
     "k0": the script keyframe of step 0 (the pictures are named by it), "clock0": the first step's clock,
     "t": [seconds since clock0, per step],
     "windows": [the step each look window ends before],
     "tasks": [[task index, its first step, the gap before it in s]],
     "pose": [[x, y, z, yaw_deg, camera x, y, z, pitch_deg], per step] or null,
     "status": {"keys": [...], "rows": [[a value per key], per step]} or null,
     "subgoal": [the text of each step] or null (the job states none, or the brief withheld it),
     "detections": {"tags": [[kind, attributes, state, size x, y, z]],
                    "rows": [[[tag, u0, v0, u1, v1, map x, y, z, camera x, y, z], per detection], per step]} or null,
     "rgb": the folder of the pictures as a `file:` address, or null, "depth": the same for the drawn depth pictures,
     "camera": {"near_m", "far_m", "unit"} or null, "missing": {"rgb": n, "depth": n}, "notes": [sentences]}

A picture of step i is that folder's address plus the six-digit script keyframe `k0 + i` and `.jpg` (`.png` for the
depth). Only a channel that sends depth (`detections`) has depth pictures; they are drawn from the scorer's
`private/<sid>/depth/`, since an agent is fed the camera alone. The detections' keys (`d`), fresh for every frame, and
the poses' quaternions, which the heading and pitch say again, are left out. Everything is rounded as the dataset's
files are, so a number on the page is the delivered one. A case the dataset cannot stream (a file missing or not what
`docs/format.md` says) has no entry and a sentence in its `problems`.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from .camera import Camera
from .case import CasePaths, ScriptPaths, depth_before_0_6_0, read_case
from .stream import delivered
from .stream_protocol import CHANNELS
from .view_depth import write_depth_view

MAX_WHY = 240   # characters of an exception's text kept in a sentence
DRAW_THREADS = 8


def _clock_offsets(steps: list[dict]) -> list[float]:
    """Each step's clock as seconds after the first step's, to the tenth (the clocks are delivered to the tenth)."""
    first = datetime.fromisoformat(steps[0]["clock"])
    return [round((datetime.fromisoformat(s["clock"]) - first).total_seconds(), 1) for s in steps]


def _pose(step: dict) -> list:
    pose, camera = step["pose"], step["camera"]
    return [*pose["pos"], pose["yaw_deg"], *camera["pos"], camera["pitch_deg"]]


def _status(steps: list[dict]) -> dict:
    keys = sorted({key for s in steps for key in s["status"]})
    return {"keys": keys, "rows": [[s["status"].get(key) for key in keys] for s in steps]}


def _detections(steps: list[dict]) -> dict:
    """Each distinct (kind, attributes, state, size) once in `tags` (a thing's size does not change, so it is a tag's,
    not a detection's); a detection is its tag's index and its numbers."""
    tags: dict[str, int] = {}
    table: list = []
    rows = []
    for step in steps:
        row = []
        for d in step["detections"]:
            tag = json.dumps([d["kind"], d.get("attributes") or {}, d.get("state") or {}, d["extent_m"]],
                             sort_keys=True)
            if tag not in tags:
                tags[tag] = len(table)
                table.append(json.loads(tag))
            row.append([tags[tag], *d["bbox_px"], *d["center_map"], *d["center_cam"]])
        rows.append(row)
    return {"tags": table, "rows": rows}


def _folder(path: Path) -> str:
    return path.resolve().as_uri() + "/"


def _draw(script: ScriptPaths, ks: range, folder: Path, cam: Camera) -> int:
    """Draw the depth picture of each script keyframe `ks` into FOLDER; how many cannot be (the image is missing or is
    not a 16-bit PNG)."""
    def one(k: int) -> int:
        source = script.depth(k)
        try:
            write_depth_view(source, folder / f"{k:06d}.png", cam)
        except (OSError, ValueError):
            return 1
        return 0

    with ThreadPoolExecutor(DRAW_THREADS) as pool:
        return sum(pool.map(one, ks))


def _pictures(case: CasePaths, ks: range, sends: frozenset[str], depth_dir: Path | None) -> dict:
    """The pictures part of an entry: the rgb folder, the depth folder (drawn now) and what is missing."""
    script, notes = case.script, []
    out: dict = {"rgb": None, "depth": None, "camera": None, "missing": {"rgb": 0, "depth": 0}}
    if "rgb" in sends:
        out["rgb"] = _folder(script.rgb_dir)
        out["missing"]["rgb"] = sum(not script.rgb(k).is_file() for k in ks)
    if "depth" in sends:
        try:
            cam = Camera.load(script.camera)
        except (OSError, ValueError, KeyError, TypeError):
            cam = None
            notes.append("the depth is not drawn: sensors/camera.json cannot be read, so the range is not known")
        if cam is not None:
            out["camera"] = {"near_m": cam.near_m, "far_m": cam.far_m, "unit": cam.depth_unit}
            if depth_dir is not None:
                folder = depth_dir / script.id
                out["missing"]["depth"] = _draw(script, ks, folder, cam)
                out["depth"] = _folder(folder)
                if out["missing"]["depth"] and depth_before_0_6_0(script).is_dir():
                    notes.append("the depth pictures are not drawn: this dataset keeps its depth under "
                                 "scripts/<id>/sensors/depth (before 0.6.0); the viewer reads private/<id>/depth")
    return out | {"notes": notes}


def _entry(dataset: Path, cid: str, channel: str, depth_dir: Path | None) -> dict:
    case = CasePaths.from_id(dataset, cid)
    window, rec = delivered(case, channel), read_case(case)
    steps, sends = window.steps, CHANNELS[channel]
    return {
        "channel": channel, "sends": sorted(sends), "k0": rec.k0, "clock0": steps[0]["clock"],
        "t": _clock_offsets(steps), "windows": [hi for _, hi in window.windows],
        "tasks": [[t.index, t.k0, t.gap_s] for t in window.tasks],
        "pose": [_pose(s) for s in steps] if "pose" in sends else None,
        "status": _status(steps) if "status" in sends else None,
        "subgoal": [s["subgoal"] for s in steps] if all("subgoal" in s for s in steps) else None,
        "detections": _detections(steps) if "detections" in sends else None,
        **_pictures(case, range(rec.k0, rec.k1), sends, depth_dir),
    }


class Streams:
    """The streams of a view's cases by `"<case id>:<channel>"`: built on first ask and shared by the arms that ran the
    case on that channel. With `depth_dir` the depth pictures are drawn into `depth_dir/<script id>/`."""

    def __init__(self, dataset: Path | None, depth_dir: Path | None = None):
        self.dataset, self.depth_dir = dataset, depth_dir
        self.items: dict[str, dict] = {}
        self._why: dict[str, str] = {}

    def get(self, cid: str, channel: str, problems: list[str]) -> str | None:
        """The key of the case's stream on `channel`, None (and a sentence in `problems`) when it cannot be shown."""
        if self.dataset is None:
            return None
        key = f"{cid}:{channel}"
        if key not in self.items and key not in self._why:
            try:
                if channel not in CHANNELS:
                    raise ValueError(f"{channel!r} is not a channel (the channels are {', '.join(CHANNELS)})")
                self.items[key] = _entry(self.dataset, cid, channel, self.depth_dir)
            except Exception as error:  # noqa: BLE001  (a case the dataset cannot stream is a sentence, not a crash)
                self._why[key] = " ".join(f"{type(error).__name__}: {error}".split())[:MAX_WHY]
        if key in self._why:
            problems.append(f"the stream cannot be shown frame by frame ({self._why[key]})")
            return None
        return key
