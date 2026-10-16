"""Claude Code as the robot's brain: an example agent (not a reference row) that runs `claude -p` once a message, which
since 0.10.0 means once a frame.

The model is `claude-sonnet-5-5`, driven headless through Claude Code's own `stream-json` input and output. A case is
one Claude Code session: the first message (the first `task`) starts it and every later message resumes it by its id,
so the conversation is the agent's short-term memory. Its long-term memory is Claude Code's own: its working folder is
`state_dir`, so the notes it writes there with its tools (Read, Write, Edit, Glob and Grep; no shell, no web, no
subagents) are measured, and its built-in memory (auto memory, `CLAUDE.md`) lives under `CLAUDE_CONFIG_DIR`, which the
shim points into the state folder, so each case starts empty. `start` refuses to run when a `CLAUDE.md` or
`CLAUDE.local.md` sits above the state folder: Claude Code would add it to the context and the run would be another
agent.

The stream sends one frame an `observe` (0.10.0), so a frame is one message: its header line (its k, clock, base,
heading, camera pitch and status), the JPEG attached, and a line naming the image. Each frame is therefore one
`claude -p` invocation, and the start-up of an invocation and the resume of the session are paid once a frame, where a
message carried up to 8 frames before. A case makes an invocation for every frame, task and ask, so it is slower, and
it should be expected to cost more, than it did. The reference Codex agent avoids this by holding one session and
pulling frames inside it (docs/codex.md); a long-lived stream-json session would do the same for this agent, and this
example does not. An ask labels its questions Q1..Qn (never the qid) and asks for one JSON object from label to answer;
the last JSON object of the reply is read, and one more message
asks for the labels it lacked, after which what is still missing is `unknown`. The robot has a camera and no depth
sensor, and there is no `locate` tool: positions come from the poses and the pictures alone, a stated limit of this
example. On the `detections` channel the detections of a frame go in as one JSON line.

Run it on one case (`DATASET/private/<case id>`), unverified because it needs the network:

    stbench run examples/claude_code_agent/agent.py:ClaudeCodeAgent DATASET/private/<case id> --local

It needs the `claude` command (it targets Claude Code 2.1.284) and a credential in the environment as a variable:
`ANTHROPIC_API_KEY`, or `CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`, the subscription). This file never reads,
logs or passes one on by itself: the child inherits the environment. Set `CLAUDE_AGENT_COMMAND` to use another command
(a `.py` file is run with the interpreter that runs this one), and `CLAUDE_AGENT_LOG` to a file to get one JSON line an
invocation (the op, seconds, the session, the result's subtype, turns, images, cost and usage; never a prompt or an
environment value). A short case cost on the order of a dollar with up to 8 frames a message; the cost of one frame a
message has not been measured, and it should be expected to be higher. `finish` reports tokens per model from the last
result's `modelUsage`, assumed cumulative over a resumed session, to be confirmed by part 2's live probe (if it is
not, `finish` reports only the last invocation's tokens); Claude Code's own dollar figure is never reported, the
benchmark prices tokens with its own table. Each invocation passes `--max-turns`, which `claude --help` (2.1.284) does
not list: assumed, to be confirmed by the same probe (if the flag is refused, every invocation fails). A failed
invocation (a non-zero exit, no result, an error result) raises, and the harness ends the case or fails that ask.

Standard library only (and the shim's `system_shim`), like the other examples.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    from system_shim import System  # `stbench run` starts this file through the shim, which provides it
except ImportError:  # run on its own (a test of yours): the same constructor and `start`, which is all this file needs

    class System:
        def __init__(self, state_dir, workdir, **opts):
            self.state_dir, self.workdir, self.opts = Path(state_dir), Path(workdir), opts

        def start(self, session, robot, channel):
            self.session, self.robot, self.channel = session, robot, channel


MODEL = "claude-sonnet-5-5"
TOOLS = "Read,Write,Edit,Glob,Grep"  # its memory tools; no shell, no web, no subagents
MAX_TURNS = 40  # model turns one message may take
TIMEOUT_S = {"task": 540, "observe": 540, "during": 540, "end": 3300}  # under the harness's 600 s and 3,600 s
COMMAND_VAR, LOG_VAR = "CLAUDE_AGENT_COMMAND", "CLAUDE_AGENT_LOG"
QUIET = {"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_TELEMETRY": "1"}
INSTRUCTION_FILES = ("CLAUDE.md", "CLAUDE.local.md")  # what Claude Code reads from the folders above its own
NUDGE_MIN_S = 60  # an ask with less time left than this is not nudged: what is missing is `unknown`
UNKNOWN = {"type": "unknown"}
CUE = "Make any notes you want in your working folder now, then reply with one short line."
UNKNOWN_RULE = ('Answer only if you are more likely right than wrong; otherwise answer {"type": "unknown"}. '
                'A wrong answer costs more than unknown.')

# Every shape an answer can take (copied from the template agent: answers.py is the strict parser, and an extra key or a
# wrong type has credit 0 and costs -1, below `unknown`), and what each means.
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
NOTES = {
    "position": 'map metres; "xyz": [x, y, z] also works',
    "offset": "metres from the camera along the floor: ahead along the robot's heading, then to its left; negative is "
              "behind it or to its right",
    "time": "the local clock time, no time zone",
    "yesno": "true or false",
    "count": "a whole number, 0 or more",
    "number": "a number; metres for a distance",
    "set": "one item per change, a map position and change added, removed or moved; an empty list means no change",
    "positions": "one item per thing, each a map position (xy or xyz); an empty list means none",
    "cells": "[i, j] covers x in [0.5i, 0.5i + 0.5) and y in [0.5j, 0.5j + 0.5), integers; an empty list means none",
    "route": "1 to 50 map points in the order you would pass them",
    "heading": "degrees to the left of the way the robot faces now, from -180 to 180; to its right is negative",
    "unknown": "you cannot know",
}


def introduction(camera: dict, channel: str) -> str:
    """What the first message says before the first brief: the role, the frame of reference, the camera, how messages
    come, the working folder, the answer shapes and the unknown rule. It names no place, dataset path or question id."""
    if channel == "detections":
        sensed = ("each frame has a header line and, after it, a line of JSON with what a perfect detector saw in "
                  "that frame; no pictures come with them")
    else:
        sensed = ("each frame has a header line, and its camera image is attached to the message (the robot has no "
                  "depth sensor); things in the images carry no labels or positions, so judge what they are, and where "
                  "they are in the map frame, from the image, the robot's pose and the camera")
    shapes = [f"- {kind}: {json.dumps(answer)} ({NOTES[kind]})" for kind, answer in ANSWERS.items()]
    return "\n\n".join([
        "You are the onboard assistant of a robot. The robot moves through a place and senses it through its camera "
        "and its own readings. You are told what it is asked to do and shown what it senses, and you must remember "
        "what matters, because later you will be asked questions about the place and about what the robot did, saw "
        "and passed, where and when.",
        "Coordinates are metres in the map frame: x points east, y north and z up. A heading is in degrees, "
        "counter-clockwise from +x; a camera pitch is positive looking up. The camera image is "
        f"{camera['width']} x {camera['height']} pixels with a horizontal field of view of {camera['hfov_deg']:g} "
        "degrees.",
        "\n".join([
            "How this works:",
            "- A task message tells you the clock time a task begins at, how long the robot went unobserved since the "
            "last one, and the assignment. Frames then arrive one at a time, in time order: "
            f"{sensed}. A header gives the base's position and heading, its own velocity in the map frame when it has "
            "one, and the camera's pitch (the camera sits on a mount, so it is not the base).",
            "- Your working folder is yours: keep notes there in files with Write and Edit, and look them up with "
            "Read, Glob and Grep. After each task message and each frame, update your notes, then reply with one "
            "short line.",
            "- Your conversation may be compacted at any time, and what was said is then partly lost. Your working "
            "folder keeps everything exactly, so keep there what you may need later: where the robot has been, what "
            "you saw, where and when, and what changed. Nothing else persists.",
            "- Questions come at some decision points during the job and after it, in a message that numbers them Q1, "
            "Q2, ... Answer each from your notes, using only what the robot has sensed so far. Reply with one JSON "
            "object from each label to its answer, and nothing after it.",
        ]),
        "\n".join(["Answers are JSON objects, one shape for each kind of answer:", *shapes, UNKNOWN_RULE]),
    ])


def _num(v: float) -> str:
    """Two decimals, finer than the first rung (0.25 m) of the position ladder; never "-0.00"."""
    s = f"{v:.2f}"
    return "0.00" if s == "-0.00" else s


def _value(v) -> str:
    """A status value as one word: `true` or `false`, a number without a needless ".0", a string as it is."""
    if isinstance(v, bool):
        return "true" if v else "false"
    return f"{v:g}" if isinstance(v, float) else str(v)


def header(step: dict) -> str:
    """A frame's header line: its number and clock, the base's position and heading, its own velocity when the step has
    one, the camera's pitch, the status (`phase` and `task` first, then the other keys in sorted order)."""
    x, y, z = step["pose"]["pos"]
    status = step["status"]
    keys = [*(k for k in ("phase", "task") if k in status), *sorted(k for k in status if k not in ("phase", "task"))]
    vx, vy = step.get("velocity", (None, None))   # the base's own velocity, m/s in the map frame, when the step has it
    moving = "" if vx is None else f"velocity ({vx + 0.0:.3f}, {vy + 0.0:.3f}) m/s; "
    return (f"[k{step['k']} {step['clock']}] base ({_num(x)}, {_num(y)}, {_num(z)}) "
            f"heading {round(step['pose']['yaw_deg'])} deg; {moving}camera pitch {round(step['camera']['pitch_deg'])} "
            f"deg; status: {', '.join(f'{k} {_value(status[k])}' for k in keys)}")


def find_command() -> list[str]:
    """The `claude` command as an argument list: `CLAUDE_AGENT_COMMAND` when set (a `.py` file is run with this
    interpreter), else `claude` on the path. RuntimeError when there is none."""
    given = os.environ.get(COMMAND_VAR)
    if given and given.endswith(".py"):
        if not Path(given).is_file():
            raise RuntimeError(f"{COMMAND_VAR} names {given}, which is not a file")
        return [sys.executable, given]
    found = shutil.which(given or "claude")
    if found is None:
        raise RuntimeError(f"no `claude` command: install Claude Code, or set {COMMAND_VAR}"
                           + (f" (it names {given}, which is not on the path)" if given else ""))
    return [found]


def instruction_files(state_dir: Path) -> list[Path]:
    """The `CLAUDE.md` and `CLAUDE.local.md` files in the folders above `state_dir`, which Claude Code would read."""
    return [p / name for p in Path(state_dir).resolve().parents for name in INSTRUCTION_FILES if (p / name).is_file()]


def last_json_object(text: str) -> dict | None:
    """The last JSON object in `text` (prose around it is skipped), or None."""
    decoder, found, i = json.JSONDecoder(), None, text.find("{")
    while i != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
        except ValueError:
            i = text.find("{", i + 1)
            continue
        found = obj
        i = text.find("{", end)
    return found


def tail(text: str, n: int = 500) -> str:
    return text.strip()[-n:]


class ClaudeCodeAgent(System):
    def __init__(self, state_dir, workdir, **opts):
        super().__init__(state_dir, workdir, **opts)
        self.command: list[str] = []
        self.camera: dict = {}
        self.conversation: str | None = None  # Claude Code's session id, from the latest `result`
        self.model_usage: dict | None = None  # the latest `result`'s modelUsage (cumulative), once there is one
        self.peak_context: int | None = None

    def start(self, session, robot, channel):
        """Refuses (RuntimeError) when there is no `claude` or a `CLAUDE.md` sits above the state folder."""
        super().start(session, robot, channel)
        self.command = find_command()
        if above := instruction_files(self.state_dir):
            raise RuntimeError(f"{above[0]} would be added to Claude Code's context, which would make this run another "
                               "agent: move it aside for the run")
        self.camera = json.loads((self.workdir / "public" / "camera.json").read_text(encoding="utf-8"))

    def task(self, index, clock, gap_s, brief):
        lines = [f"Task {index} begins at {clock}."]
        if gap_s > 0:
            lines.append(f"The robot went unobserved for {gap_s:.0f} s since the last task's final frame.")
        text = "\n".join([*lines, brief, CUE])
        if self.conversation is None:
            text = f"{introduction(self.camera, self.channel)}\n\n{text}"
        self._invoke("task", "task", text)

    def observe(self, steps):
        if not steps:
            raise ValueError("a batch needs at least one step")
        pictured = [s for s in steps if s.get("rgb")]
        title = f"Frames k{steps[0]['k']} to k{steps[-1]['k']} ({steps[0]['clock']} to {steps[-1]['clock']})"
        if pictured:
            title += f", with {len(pictured)} camera image{'s' * (len(pictured) != 1)} attached, in frame order"
        lines, images = [f"{title}:"], []
        for step in steps:
            lines.append(header(step))
            if step.get("rgb"):
                images.append((self.workdir / step["rgb"]).read_bytes())
                lines.append(f"(image {len(images)} is frame k{step['k']})")
            if "detections" in step:
                lines.append("detections: " + json.dumps(step["detections"], separators=(",", ":")))
        self._invoke("observe", "observe", "\n".join([*lines, CUE]), images)

    def ask(self, when, now, questions):
        """One message for the questions (Q1..Qn, no qid), and one more for the labels the reply lacked."""
        if when not in ("during", "end"):
            raise ValueError(f"an ask is during or end, got {when!r}")
        labelled = {f"Q{n}": q for n, q in enumerate(questions, 1)}
        deadline = time.monotonic() + TIMEOUT_S[when]
        where = ("during the job, at the robot's latest frame" if when == "during"
                 else "after the job, at the robot's last frame")
        lines = [f"Questions {where} (k{now['k']}, {now['clock']}). Answer each from your notes, using only what the "
                 "robot has sensed up to that frame. Reply with one JSON object from each label to its answer object, "
                 'such as {"Q1": {"type": "unknown"}}, and nothing after it.']
        for label, q in labelled.items():
            lines += ["", f"{label}: {q['text']}", f"Answer type: {q['answer_type']}. Shape: "
                      f"{json.dumps(ANSWERS.get(q['answer_type'], UNKNOWN))}"]
        found = self._labelled(self._invoke("ask", when, "\n".join([*lines, "", UNKNOWN_RULE]), deadline=deadline),
                               list(labelled))
        if (missing := [label for label in labelled if label not in found]) and \
                deadline - time.monotonic() > NUDGE_MIN_S:
            nudge = (f"You have not answered {', '.join(missing)}. Reply with one JSON object from each of those "
                     'labels to its answer object, and nothing after it. If you cannot tell, answer {"type": '
                     '"unknown"}.')
            found |= self._labelled(self._invoke("ask", when, nudge, deadline=deadline), missing)
        return {q["qid"]: found.get(label, dict(UNKNOWN)) for label, q in labelled.items()}

    @staticmethod
    def _labelled(reply: str, labels: list[str]) -> dict[str, dict]:
        """The answers of `labels` in the reply's last JSON object: a label whose value is not an object holding a
        string `type` is missing."""
        obj = last_json_object(reply) or {}
        return {label: obj[label] for label in labels
                if isinstance(obj.get(label), dict) and isinstance(obj[label].get("type"), str)}

    def finish(self):
        """The `finish` usage report: tokens per model from the latest result's cumulative `modelUsage`."""
        def count(usage: dict, key: str) -> int | None:
            value = usage.get(key)
            return value if isinstance(value, int) and not isinstance(value, bool) else None

        models = None
        if self.model_usage is not None:
            models = {}
            for model, usage in self.model_usage.items():
                fresh, cached, written, out = (count(usage, k) for k in (
                    "inputTokens", "cacheReadInputTokens", "cacheCreationInputTokens", "outputTokens"))
                models[model] = {"tokens_in": None if None in (fresh, cached, written) else fresh + cached + written,
                                 "tokens_cached_in": cached, "tokens_cache_write_in": written, "tokens_out": out}
        return {"models": models, "peak_context": self.peak_context, "remote_store_bytes": None}

    def _invoke(self, op: str, kind: str, text: str, images=(), deadline: float | None = None) -> str:
        """One `claude -p` run of this message in the session (new for the first, resumed after): the final text of its
        `result`. RuntimeError when it fails (a non-zero exit, no result, an error result, a timeout)."""
        resumed = self.conversation is not None
        argv = [*self.command, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
                "--model", MODEL, "--tools", TOOLS, "--permission-mode", "acceptEdits", "--max-turns", str(MAX_TURNS),
                *(["--resume", self.conversation] if resumed else [])]
        blocks = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                               "data": base64.b64encode(data).decode("ascii")}} for data in images]
        message = {"type": "user", "message": {"role": "user", "content": [*blocks, {"type": "text", "text": text}]}}
        started = time.monotonic()
        limit = (deadline if deadline is not None else started + TIMEOUT_S[kind]) - started
        record = {"op": op, "resumed": resumed, "session": self.conversation, "images": len(images)}
        try:
            run = subprocess.run(argv, input=json.dumps(message) + "\n", capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", cwd=self.state_dir, timeout=limit,
                                 env={**os.environ, **QUIET}, check=False)
        except subprocess.TimeoutExpired:
            self._log(record | {"seconds": round(time.monotonic() - started, 3), "subtype": "timeout"})
            raise RuntimeError(f"claude {op} timed out after {limit:.0f} s") from None
        result = None
        for line in run.stdout.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue  # not an event
            if not isinstance(event, dict):
                continue
            if event.get("type") == "assistant":
                usage = (event.get("message") or {}).get("usage") or {}
                held = sum(usage.get(k) or 0 for k in ("input_tokens", "cache_read_input_tokens",
                                                       "cache_creation_input_tokens"))
                if held:
                    self.peak_context = max(self.peak_context or 0, held)
            elif event.get("type") == "result":
                result = event
        result = result or {}
        if isinstance(result.get("session_id"), str) and result["session_id"]:
            self.conversation = result["session_id"]
        if isinstance(result.get("modelUsage"), dict) and result["modelUsage"]:
            self.model_usage = result["modelUsage"]
        self._log(record | {"seconds": round(time.monotonic() - started, 3), "session": self.conversation,
                            **{k: result.get(k) for k in ("subtype", "is_error", "num_turns", "total_cost_usd",
                                                          "usage", "modelUsage")}})
        text_out = result.get("result") if isinstance(result.get("result"), str) else ""
        if run.returncode != 0 or not result or result.get("is_error"):
            raise RuntimeError(f"claude {op} failed (exit {run.returncode}, subtype {result.get('subtype')}, "
                               f"is_error {result.get('is_error')}): {tail(run.stderr) or tail(text_out)}")
        if self.conversation is None:
            raise RuntimeError(f"claude {op} gave no session id, so its memory could not be resumed")
        return text_out

    @staticmethod
    def _log(record: dict) -> None:
        """One JSON line of `record` in `CLAUDE_AGENT_LOG`, when set. It holds no prompt and no environment value."""
        if path := os.environ.get(LOG_VAR):
            Path(path).parent.mkdir(parents=True, exist_ok=True)  # `--out` is made only when the run is written
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
