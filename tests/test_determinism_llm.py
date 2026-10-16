"""What the reference agent's model is given, in a second process (`determinism_kit.py`: nothing is recorded, the two
are compared).

The argv of each exec kind, the tool list and the tools' fixed replies (`submit_answer`, `next_frame`), the system
prompt, a task, a frame, an ask, the fixed lines and the first message of the thread, as the model reads them.
`outputs` hashes them on fixed inputs; the test checks that the digests are the same in a second process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from codex_kit import CREDIT, ROBOT
from determinism_kit import digest
from stbench import agent_tools, codex_exec, codex_mcp, llm_render
from stbench.codex_asks import AskTable
from stbench.codex_exec import Codex, toml
from stbench.codex_feed import Feed
from stbench.codex_log import Journal
from stbench.codex_system import CodexSystem

CAMERA = {"width": 640, "height": 480, "hfov_deg": 80.0, "fx": 400.0, "fy": 400.0, "cx": 320.0, "cy": 240.0,
          "depth_unit": "mm", "near_m": 0.01, "far_m": 4.0}
FACING_Y = [0.70710678, 0.0, 0.0, 0.70710678]
SUBMIT_CALLS = [{"label": "Q1", "answer": {"type": "count", "value": -1}},
                {"label": "Q1", "answer": {"type": "count", "value": 2}},
                {"label": "Q1", "answer": {"type": "count", "value": 3}},
                {"label": "Q2", "answer": {"type": "count", "value": 1}},
                {"label": "Q2", "answer": {"type": "count", "value": 1}},
                {"label": "Q2", "answer": {"type": "position", "xy": [1, 2]}},
                {"label": "Q9", "answer": {"type": "unknown"}}, {"answer": {"type": "unknown"}}]


def fixed_steps() -> list[dict]:
    """Two frames as the stream delivers them on `rgb_pose`: the base and the camera differ, and the status holds the
    vacuum's keys."""
    return [{"k": k, "clock": f"2026-03-02T08:00:{k:02d}",
             "pose": {"pos": [5.6, 5.05 - 0.1 * k, 0.0], "rot": FACING_Y, "yaw_deg": -90.0},
             "camera": {"pos": [5.6, 4.92 - 0.1 * k, 0.08], "rot": FACING_Y, "pitch_deg": 5.0},
             "status": {"phase": "cleaning", "task": 0, "battery_pct": 97, "brush": True},
             "rgb": f"inbox/rgb/{k:06d}.jpg"} for k in (12, 13)]


def fixed_detection_steps() -> list[dict]:
    """The same two frames as the stream delivers them on `detections`: no `rgb`, and each frame's detections (the
    second sees a door, the first a mug and the door)."""
    mug = {"d": "9f3a01bc", "kind": "mug", "attributes": {"color": "red", "size": "small"},
           "bbox_px": [120, 200, 180, 260], "center_cam": [0.1, -0.3, 1.5], "center_map": [2.0, 1.0, 0.9],
           "extent_m": [0.1, 0.1, 0.12]}
    door = {"d": "01aa77ee", "kind": "door", "attributes": {}, "bbox_px": [300, 10, 420, 470],
            "center_cam": [0.5, 0.0, 2.5], "center_map": [3.1, 1.2, 1.0], "extent_m": [0.9, 0.05, 2.0],
            "state": {"open": True}}
    return [{key: value for key, value in step.items() if key != "rgb"} | {"detections": found}
            for step, found in zip(fixed_steps(), ([mug, door], [door]), strict=True)]


def argv_output() -> dict:
    """The argv of each exec kind, with what varies by machine (the interpreter, the temporary files) as fixed words,
    and the settings every exec shares."""
    codex = Codex("fixed-model", command=["codex"])
    codex.catalog, codex.instructions = Path("CATALOG"), Path("INSTRUCTIONS")
    python = toml(sys.executable)
    server, empty = Path("SERVER"), Path("EMPTY")
    argv = {kind: [a.replace(python, "'PYTHON'") for a in codex.argv(kind, thread, server, empty)]
            for kind, thread in (("start", None), ("resume", "THREAD"))}
    return {"argv": argv, "base": codex_exec.BASE_INSTRUCTIONS, "patch": codex_exec.CATALOG_PATCH,
            "server": codex_exec.SERVER_CONFIG, "tool_timeout": codex_exec.TOOL_TIMEOUT_S}


def tools_output() -> dict:
    """The tool list, and what the tools reply to fixed calls: the submission rules for two labels and the nudge of a
    premature `next_frame` (the ask table), and, from the feed, a submission outside an ask, a call past the call bound
    and a retired tool name."""
    asks = AskTable({"Q1": "count", "Q2": "position"})
    submit = [asks.submit(arguments) for arguments in SUBMIT_CALLS]
    nudges = [asks.pull(), asks.pull()]
    feed = Feed(max_calls=0)
    try:
        replies = {"no_ask": feed.call("submit_answer", SUBMIT_CALLS[2]), "past_max": feed.call("submit_answer", {}),
                   "retired": feed.call("memory_list", {})}
    finally:
        feed.close()
    return {"tools": agent_tools.TOOLS, "specs": codex_mcp.tool_specs(), "submit": submit, "nudges": nudges, **replies}


def messages_output(folder: Path) -> dict:
    """What the model reads, on fixed inputs: the system prompt, the first message of the thread, a first and a later
    task, two frames (and the image each sends), a during and an end ask over every answer type, the nudge, the end
    and the line that resumes a thread; and, for the `detections` channel, its system prompt and the same two frames as
    detections."""
    public = folder / "work" / "public"
    public.mkdir(parents=True)
    (public / "camera.json").write_text(json.dumps(CAMERA), encoding="utf-8")
    (public / "robot.json").write_text(json.dumps(ROBOT), encoding="utf-8")
    system = CodexSystem(folder / "state", folder / "work", codex=Codex("fixed-model", command=["codex"]),
                         journal=Journal(folder / "journal.jsonl"))
    system.start("0123456789abcdef", {}, "rgb_pose")
    system.close()  # the feed it made has no use here
    first = llm_render.task_message(0, "2026-03-02T08:00:00", 0.0, "Your assignment is to clean.")
    questions = llm_render.labelled([{"qid": "x", "text": f"A {kind} question?", "answer_type": kind}
                                     for kind in ("position", "time", "yesno", "count", "number", "set", "positions",
                                                  "cells", "route")])
    now = {"k": 13, "clock": "2026-03-02T08:00:13"}
    return {"system": llm_render.system_prompt(public, "rgb_pose", asks_during=True, **CREDIT),
            "first": system._first(first),
            "tasks": [first, llm_render.task_message(1, "2026-03-02T08:20:31", 612.0, "Next: dock.")],
            "frames": [llm_render.frame_message(step, "rgb_pose") for step in fixed_steps()],
            "text_only": {"system": llm_render.system_prompt(public, "rgb_pose", text_only=True),
                          "frames": [llm_render.frame_message(step, "rgb_pose", text_only=True)
                                     for step in fixed_steps()]},
            "detections": {"system": llm_render.system_prompt(public, "detections"),
                           "frames": [llm_render.frame_message(step, "detections")
                                      for step in fixed_detection_steps()]},
            "asks": [llm_render.ask_message(when, now, questions) for when in ("during", "end")],
            "nudge": agent_tools.unanswered(["Q1", "Q3"]), "end": llm_render.END_MESSAGE,
            "resume": llm_render.RESUME_MESSAGE}


def outputs(folder: Path) -> dict[str, str]:
    """The digest of each part of the reference agent's interface on fixed inputs, and of all of it."""
    parts = {"argv": argv_output(), "tools": tools_output(), "messages": messages_output(folder / "messages")}
    return {**{name: digest(part) for name, part in parts.items()}, "codex": digest(parts)}


def test_the_reference_agent_digest_is_deterministic(tmp_path):
    """Contract: what the reference agent's model is given (the argv of a start and a resume, the tool list and the
    tools' replies to fixed calls, the prompt, a task, a frame (and the same in the text-only option), an ask, the fixed
    lines and the first message of the thread) is the same in a second process (another `PYTHONHASHSEED`): the digest of
    each part, on fixed inputs, and of all of it, and the four digests differ. Regression caught: a prompt, tool or
    argv that depends on dict or set order or on the machine it ran on (an interpreter path in the argv)."""
    here = outputs(tmp_path / "here")
    out = subprocess.run([sys.executable, __file__, str(tmp_path / "there")], capture_output=True, text=True,
                         check=True, env={**os.environ, "PYTHONHASHSEED": "12345"}).stdout
    assert json.loads(out) == here
    assert len(set(here.values())) == 4  # three different parts and the whole, none an empty digest


if __name__ == "__main__":  # the second process of the test above
    print(json.dumps(outputs(Path(sys.argv[1]))))
