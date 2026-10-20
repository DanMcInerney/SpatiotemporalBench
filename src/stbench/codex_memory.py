"""A memory for the Codex agent: a store the agent reaches through an MCP server of its own, fed by the robot's sensors.

`stbench run|eval codex:<model> --memory SPEC` gives the reference agent one memory system beside `next_frame` and
`submit_answer`, so that a memory store is measured under the reference agent's own transport, prompt and images: the
arm is the store, and its baseline is the same agent, model and effort without it (`results.json` names the memory,
which `eval compare` reads as part of the agent's lever). Without `--memory` nothing of the reference agent changes,
byte for byte. SPEC is a JSON file:

    {"name": "mymemory",
     "serve": ["/path/to/python", "-m", "mymemory.server", "--model", "{model}", "{store}"],
     "sense": ["/path/to/python", "-m", "mymemory.feed", "{store}"],
     "prompt": "guide.md"}

- `name`: the MCP server's name (`NAME`: lower-case letters, digits, `_` and `-`), which the model sees its tools under;
  never `stbench`, the tool server's, nor `codex`, the name Codex reports its own helpers under.
- `serve`: the server's argv. Codex starts it for each exec, as it starts the tool server, `required` and pre-approved,
  beside it in the same exec. `{store}` is the case's store folder, `<state_dir>/<name>`, absent at the start of every
  case: the store is the agent's state (counted in `state_bytes`), and nothing carries over from one case to the next.
  `{model}` is the model id. The server must offer no tool that reads the disk outside its store, since the held-out
  truth is on disk. Its argv holds no quote or line break (`codex_exec.toml` could not pass it to Codex).
- `sense` (optional): the robot's own sensor feed into the store, one process for the case (`Sense`), started before
  anything is sent to the model. The harness writes one JSON object a line to its stdin for each message of the stream
  and reads exactly one JSON object line back from its stdout for each, within `SENSE_TIMEOUT` seconds; every line it
  writes is ASCII (`json.dumps`), so the encoding of its stdin is no matter. The messages are what the robot itself
  knows and nothing of any question, since it is sent no ask, no question and no finish:

      {"start": {"camera": <public/camera.json>, "robot": <robot.json>}}
      {"task": {"index": int, "clock": str, "gap_s": float, "brief": str}}
      {"observe": {"camera": <public/camera.json>, "steps": [STEP]}}        exactly one step, `feed_step`
      {"record": <the model's memory argument>}                             only with `carry`

  The `observe` message is written as the frame is served to the model, so that the store has frame k when the model
  has it and never frame k+1 (the runner reads the next message ahead to learn whether questions join a frame, and
  nothing is fed from that read). Its image is a file of the system's own that exists while the message is handled.
- `carry` (optional, a bool, default false; needs `sense`): the agent's memory record rides on its pull. With it,
  `next_frame` takes an optional `memory` object (`agent_tools.tools(True)`), the record of the frame it was just
  shown, as the memory's claim takes it, and the harness writes `{"record": <memory as given>}` to the sense feed
  before it counts the pull, then returns the feed's reply as the last text block of the item that pull gets
  (`"<name>: <compact JSON>"`; `codex_feed`). The feed answers it as it answers any message, one JSON object line; a
  record it refuses is just that reply, and the case goes on. Without `carry` the tools, and every byte the model is
  sent, are as they were, and a `memory` argument sends nothing.
- `prompt`: a UTF-8 text file, relative to SPEC's folder, the memory's guide: how this agent uses this memory. It is
  appended to the exec's base instructions (`codex_exec.Codex`), never to a message, so every message the model is sent
  is the reference agent's.

A failure of the feed (it cannot start, exits, closes its output, answers with a line that is no JSON object, or takes
longer than `SENSE_TIMEOUT`) is a `MemoryFailure`, which `codex_system` journals and ends the case with as the
transport failure `memory`; so is one on a carried `record`, at the runner's next check, which raises before any item
is served (the pull is closed, so the model is told nothing of the failure).
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import re
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from .local_system import kill_tree

NAME = re.compile(r"[a-z][a-z0-9_-]{0,31}")
RESERVED = ("stbench", "codex")  # the tool server's name, and the one Codex reports its own helpers under
KEYS = {"name", "serve", "sense", "prompt", "carry"}
SENSE_TIMEOUT = 120.0  # seconds one sense message may take
CLOSE_WAIT = 5.0  # seconds a sense command has to end after its stdin is closed, before it is killed
EXIT_WAIT = 2.0  # seconds a command whose output ended has to exit, so that its exit code is the one reported
STEP_KEYS = ("k", "clock", "velocity", "status", "subgoal")  # a step's keys the feed is given as they are
STEP_POSE = ("pos", "yaw_deg")  # the base: what the frame's header line shows of it
STEP_CAMERA = ("pos", "rot", "pitch_deg")  # the camera's place and attitude in the map frame, and its pitch
TAIL = 500  # characters of a failed command's output a failure message keeps


class MemoryFailure(ValueError):
    """A memory SPEC that cannot be used, or a sense message that failed; the message says why."""


@dataclass(frozen=True)
class Memory:
    """A loaded SPEC (see the module docstring). `guide` is the prompt file's text, `path` SPEC's absolute path."""

    name: str
    serve: tuple[str, ...]
    sense: tuple[str, ...] | None
    guide: str
    path: Path
    carry: bool = False

    def store(self, state_dir: Path) -> Path:
        """The case's store folder."""
        return Path(state_dir) / self.name

    def server(self, state_dir: Path, model: str) -> tuple[str, list[str]]:
        """(the MCP server's name, its argv) for one case: what `Codex.argv` configures beside the tool server."""
        return self.name, _fill(self.serve, model=model, store=self.store(state_dir))

    def open_sense(self, state_dir: Path, model: str) -> Sense | None:
        """The case's sensor feed, started (None: SPEC has none). Raises MemoryFailure when it cannot be started."""
        if self.sense is None:
            return None
        return Sense(_fill(self.sense, model=model, store=self.store(state_dir)))

    def record(self) -> dict:
        """The memory as the journal's `run` event records it: its name, SPEC's argvs as loaded and the guide's text,
        and `carry` when it is on (a memory without it is recorded as it always was)."""
        return {"name": self.name, "serve": list(self.serve), "sense": list(self.sense) if self.sense else None,
                "guide": self.guide, **({"carry": True} if self.carry else {})}


def _fill(argv: tuple[str, ...], *, model: str, store: Path) -> list[str]:
    return [a.replace("{store}", str(store)).replace("{model}", model) for a in argv]


def feed_step(step: dict, workdir: Path) -> dict:
    """What the sensor feed is given of one step: a whitelist, never the step as it is. The frame's number and clock,
    what its header line shows the model (the base's position and heading, its own velocity, the camera's pitch, the
    status and the subgoal, the last two when the step has them), the camera's position and rotation, and `rgb`, the
    absolute path of the frame's image in the inbox. Nothing privileged (`identity`), no detection and no depth ever
    reaches the feed, whatever the step carries."""
    return {**{key: step[key] for key in STEP_KEYS if key in step},
            "pose": {key: step["pose"][key] for key in STEP_POSE},
            "camera": {key: step["camera"][key] for key in STEP_CAMERA},
            "rgb": str(Path(workdir) / step["rgb"])}


class Sense:
    """A memory's sensor feed: one running process, a JSON line in and a JSON line out for each message (`send`). Its
    stderr goes to a file, so a command that logs a good deal never blocks, and its tail is what a failure reports. A
    thread reads its stdout into a queue, so `send` can wait with a timeout on Windows too."""

    def __init__(self, argv: list[str]):
        self._lines: queue.Queue[bytes | None] = queue.Queue()
        self._err = tempfile.TemporaryFile()  # noqa: SIM115  (it lives as long as the command: `close` closes it)
        try:
            self._proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err,
                                          start_new_session=os.name != "nt")
        except OSError as error:
            self._err.close()
            raise MemoryFailure(f"the memory's sense command cannot be started: {error}") from None
        self._turn = threading.Lock()  # one request and its reply at a time: a reply is its request's, whoever waits
        self._pump = threading.Thread(target=self._read, daemon=True, name="stbench-sense")
        self._pump.start()

    def _read(self) -> None:
        try:
            for raw in self._proc.stdout:
                if raw.strip():
                    self._lines.put(raw)
        finally:
            self._lines.put(None)  # the end of its output

    def _tail(self) -> str:
        try:
            self._err.seek(0, 2)
            self._err.seek(max(0, self._err.tell() - 4 * TAIL))
            return self._err.read().decode("utf-8", "replace").strip()[-TAIL:]
        except (OSError, ValueError):
            return ""

    def send(self, message: dict, timeout: float | None = None) -> dict:
        """Write `message` as one line to the command and return the JSON object it printed back as one line. Raises
        MemoryFailure when it exited, closed its output, printed a line that is no JSON object or took longer than
        `timeout` seconds (default `SENSE_TIMEOUT`); the command is then ended. Callers are served one at a time (the
        write and the read of a request/reply are one turn), so two threads never pair a reply with the other's
        request."""
        with self._turn:
            return self._send(message, SENSE_TIMEOUT if timeout is None else timeout)

    def _send(self, message: dict, limit: float) -> dict:
        kind = next(iter(message))
        try:
            self._proc.stdin.write(json.dumps(message).encode("ascii") + b"\n")
            self._proc.stdin.flush()
            raw = self._lines.get(timeout=limit)
        except queue.Empty:
            self._end(0.0)
            raise MemoryFailure(f"the memory's sense command did not answer {kind!r} within {limit:g} s") from None
        except (OSError, ValueError):
            raw = None  # it is gone (its input is closed): the exit below says how it ended
        if raw is None:
            self._end(EXIT_WAIT)
            raise MemoryFailure(f"the memory's sense command ended (exit {self._proc.returncode}) on {kind!r}: "
                                f"{self._tail()}")
        try:
            found = json.loads(raw)
        except ValueError:
            found = None
        if not isinstance(found, dict):
            self._end(0.0)
            raise MemoryFailure(f"the memory's sense command printed no JSON object on {kind!r}: "
                                f"{raw.decode('utf-8', 'replace').strip()[:TAIL]!r}")
        return found

    def _end(self, grace: float) -> None:
        """End its input, give it `grace` seconds to end and kill it (with what it started) if it has not."""
        with contextlib.suppress(OSError, ValueError):
            self._proc.stdin.close()
        try:
            self._proc.wait(grace)
        except subprocess.TimeoutExpired:
            kill_tree(self._proc)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._proc.wait(CLOSE_WAIT)
        self._pump.join(CLOSE_WAIT)

    def close(self, grace: float = CLOSE_WAIT) -> None:
        """End the command (`_end`) and release its stderr file. Idempotent."""
        self._end(grace)
        with contextlib.suppress(OSError):
            self._err.close()


def load(path: str | Path) -> Memory:
    """The memory of SPEC at `path`. Raises MemoryFailure for a file that is not such a SPEC."""
    path = Path(path).resolve()
    try:
        spec = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise MemoryFailure(f"cannot read the memory SPEC {path}: {error}") from None
    if not isinstance(spec, dict) or (extra := set(spec) - KEYS) or not {"name", "serve", "prompt"} <= set(spec):
        raise MemoryFailure(f"{path}: a memory SPEC is an object with name, serve and prompt (and optionally sense and "
                            "carry)" + (f"; unknown keys {sorted(extra)}" if isinstance(spec, dict) and extra else ""))
    name = spec["name"]
    if not isinstance(name, str) or not NAME.fullmatch(name) or name in RESERVED:
        raise MemoryFailure(f"{path}: name must match {NAME.pattern} and be none of {list(RESERVED)}, not {name!r}")
    serve = _argv(spec["serve"], "serve", path)
    sense = None if spec.get("sense") is None else _argv(spec["sense"], "sense", path)
    carry = spec.get("carry", False)
    if not isinstance(carry, bool):
        raise MemoryFailure(f"{path}: carry is true or false, not {carry!r}")
    if carry and sense is None:
        raise MemoryFailure(f"{path}: carry needs a sense feed, which the record is written to")
    if "{store}" not in " ".join(serve):
        raise MemoryFailure(f"{path}: serve must name the case's store as {{store}}")
    if any("'" in a or "\n" in a for a in serve):
        raise MemoryFailure(f"{path}: serve holds a quote or a line break, which cannot be passed to codex")
    prompt_path = path.parent / str(spec["prompt"])
    try:
        guide = prompt_path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise MemoryFailure(f"{path}: cannot read the prompt {prompt_path}: {error}") from None
    if not guide:
        raise MemoryFailure(f"{path}: the prompt {prompt_path} is empty")
    return Memory(name, serve, sense, guide, path, carry)


def _argv(value, key: str, path: Path) -> tuple[str, ...]:
    if not (isinstance(value, list) and value and all(isinstance(a, str) and a for a in value)):
        raise MemoryFailure(f"{path}: {key} is a non-empty list of strings (an argv)")
    return tuple(value)
