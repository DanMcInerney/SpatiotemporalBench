"""A template to start your own agent from: every message it receives and every answer it can give, with TODOs.

Copy this folder, rename the class, fill in the TODOs. It runs as it is (it answers `unknown` to everything, which is
always allowed and scores a little). Run it on one case, `DATASET/private/<case id>`:

    stbench run examples/template_agent/agent.py:TemplateAgent DATASET/private/<case id> --local

Standard library only, so it runs unchanged in the Docker sandbox too. Decoding the pictures needs a library you bring:
`--local` runs in the Python that runs `stbench` (numpy and Pillow are installed there); the Docker sandbox has only the
standard library plus what you put in your own folder. docs/agent.md has the whole contract.

The harness sends five messages, in this order, to one instance that lives for the whole case: `start` once, then for
each task `task` and one `observe` for each of its frames (with an `ask` after the frame at a point it asks about),
then the end `ask` and `finish`. Below, each is a method. An `observe` carries exactly one frame, a list of one step,
and returning from it asks for the next: the harness sends nothing more until you have.

Rules the harness holds you to:
- Keep everything you need in `state_dir` (or in the process: it is never reset). The harness reads only its size, at
  the end of each task and at `finish`. Each case starts with an empty `state_dir` and a new process.
- `workdir` is read-only. `workdir/public` holds `robot.json` and `camera.json`; `workdir/inbox` holds only the files of
  the frame you were just given, and still holds them during the `ask` that follows it.
- Nothing from the future is readable, and no answer is ever checked: there is no feedback.
- `{"type": "unknown"}` is always an allowed answer, and a wrong answer scores below it.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    from system_shim import System  # `stbench run` starts this file through the shim, which provides it
except ImportError:  # run on its own (a test of yours): the same constructor, the one thing this file needs of it

    class System:
        def __init__(self, state_dir, workdir, **opts):
            self.state_dir, self.workdir, self.opts = Path(state_dir), Path(workdir), opts

        def start(self, session, robot, channel):
            self.session, self.robot, self.channel = session, robot, channel


# Every shape an answer can take (answers.py is the strict parser: an extra key or a wrong type has credit 0 and costs
# -1, below `unknown`).
ANSWERS = {
    "position": {"type": "position", "xy": [3.2, 1.5]},  # map metres; or "xyz": [x, y, z]
    # a place from the camera, along the floor: metres ahead of it, then to its left (negative: behind it, to its right)
    "offset": {"type": "offset", "ahead_m": 3.5, "left_m": -1.0},
    "time": {"type": "time", "clock": "2026-12-06T09:40:38"},  # local ISO clock, no zone
    "yesno": {"type": "yesno", "value": True},
    "count": {"type": "count", "value": 2},  # a non-negative integer
    "number": {"type": "number", "value": 1.25},  # a finite number (metres for a distance)
    "set": {"type": "set", "items": [{"xy": [3.2, 1.5], "change": "moved"}]},  # change: added, removed or moved
    # a list of places: one item per thing, each "xy" or "xyz", no "change"
    "positions": {"type": "positions", "items": [{"xy": [3.2, 1.5]}, {"xyz": [4.0, 2.0, 0.9]}]},
    # cells of the 0.5 m answer grid: [i, j] covers x in [0.5i, 0.5i + 0.5) and y in [0.5j, 0.5j + 0.5), integers
    "cells": {"type": "cells", "items": [[6, 3], [7, 3]]},
    # a route: 1 to 50 map points, in the order you would pass them
    "route": {"type": "route", "points": [[5.4, 3.0], [2.4, 3.5]]},
    # a direction to turn in place toward: degrees to the left of the way the robot faces, -180 to 180 (right: negative)
    "heading": {"type": "heading", "deg_left": 35.0},
    "unknown": {"type": "unknown"},
}


class TemplateAgent(System):
    def __init__(self, state_dir, workdir, **opts):
        super().__init__(state_dir, workdir, **opts)  # state_dir: read-write, and persistent; workdir: read-only
        # What never changes is in workdir/public/: robot.json (the robot's name, body, camera, status keys) and
        # camera.json (width, height, hfov_deg, fx, fy, cx, cy, and the scorer's depth sensor: depth_unit, near_m,
        # far_m, which `rgb_pose` never sends).
        self.camera = json.loads((self.workdir / "public" / "camera.json").read_text(encoding="utf-8"))
        self.notes = self.state_dir / "notes.jsonl"
        # TODO: open or build whatever you keep (an index, a database, a notes file) and say what is in it.

    def start(self, session, robot, channel):
        """Called once, before anything else. `session` is a fresh 16-hex token for this run (never the case id: use it
        to namespace any remote memory), `robot` is `public/robot.json` as a dict, `channel` is `rgb_pose` (the default:
        the camera image `rgb` alone: the robot has a camera and no depth sensor) or `detections` (`depth` and what
        a perfect detector saw, no camera image)."""
        super().start(session, robot, channel)
        # TODO: choose what to keep for this robot (its phases, the status keys it reports, how far its camera sees).

    def task(self, index, clock, gap_s, brief):
        """A task begins. `index` counts from 0 in the case, `clock` is the local ISO time of its first step, `gap_s`
        the seconds nobody observed since the last task ended (0 for the first) and `brief` text: the first one says
        who you are, what the robot does, what you will be sent and the cost of a wrong answer; a later one is that
        task's assignment alone."""
        self.note({"task": index, "clock": clock, "gap_s": gap_s, "brief": brief})
        # TODO: decide from the brief what is worth remembering for the questions that will come.

    def observe(self, steps):
        """One frame: a list of one step. Store what you will want to answer from later. Returning asks for the next."""
        for step in steps:
            # step["k"]: the step number from 0, case-relative. step["clock"]: its local ISO time, e.g.
            # "2026-12-06T09:39:02", the form questions quote.
            # step["pose"]: the robot's base in the map frame (metres, x east, y north, z up):
            # {"pos": [x, y, z], "rot": [w, x, y, z] (body frame to map), "yaw_deg"}.
            # step["camera"]: the camera's pose, {"pos", "rot", "pitch_deg"}: it sits on a mount, so it is not the base.
            # step["velocity"]: the base's own velocity [vx, vy], metres a second in the map frame, from its controller.
            # step["status"]: the robot's own readings, {"phase", "task", ...its other keys}.
            note = {"k": step["k"], "clock": step["clock"], "base": step["pose"]["pos"], "status": step["status"]}

            # Image paths are relative to workdir, and the files stay only until the next frame (and the ask after
            # this one): read them now.
            rgb_jpg = self.workdir / step["rgb"] if "rgb" in step else None  # `rgb_pose`
            # Detections (the `detections` channel only): kind, attributes, bbox_px, center_cam, center_map, extent_m.
            note["seen"] = self.look(rgb_jpg, step.get("detections", []), step["camera"])
            self.note(note)

    def look(self, rgb_jpg: Path | None, detections: list[dict], camera: dict):
        """TODO: turn one frame into whatever you want to remember (JSON-serialisable), e.g. with Pillow and numpy:
        decode the image, find objects, and place them in map metres from the camera's pose (not the base's) and your
        own estimate of their distance."""
        return None

    def ask(self, when, now, questions):
        """Answers by qid. `when` is "during" (right after the step `now = {k, clock}`, one question) or "end" (after
        the last step, the whole set). Each question is `{qid, text, answer_type}` and nothing else. A question you
        leave out, or answer with something that is not an object, has credit 0 and costs -1, below `unknown`."""
        return {q["qid"]: self.answer(q) for q in questions}

    def answer(self, question):
        """One question, from what you stored. Return one of the shapes in ANSWERS."""
        # question["text"]: the question in words, e.g. "How far ahead of the robot is the floor clear?"
        # question["answer_type"]: position, time, yesno, count, number, set, positions, cells or route: the shape to
        # return (ANSWERS[...]) or unknown.
        # TODO: work the answer out from self.notes and return ANSWERS[question["answer_type"]] filled in. A wrong
        # answer scores below unknown, so abstain unless you are likely right.
        return ANSWERS["unknown"]

    def finish(self):
        """The run is over: the usage you report, {"models": {id: {tokens_in, tokens_cached_in, tokens_cache_write_in,
        tokens_out}}, "peak_context": int, "remote_store_bytes": int}, each value may be null (docs/agent.md)."""
        return {}

    def note(self, record: dict) -> None:
        with self.notes.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
