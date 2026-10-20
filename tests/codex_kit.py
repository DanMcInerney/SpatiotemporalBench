"""Helpers for the tests of the Codex route (`test_codex_*`, `test_cmd_codex`): a `Codex` over the fake command
(`codex_fake`), the fake's record of what it was run with and did, and a `CodexSystem` on a working folder laid out as
the harness lays it out.

`Fake` holds the transport and the folder the fake keeps its threads and its record in. `Fake.plan` is the JSON of
`codex_fake`'s per-exec changes. `Fake.execs` gives one dict an exec: what it was started with (`argv`, `prompt`,
`instructions`, the text of its instructions file, `servers`, `history`, ...) and what it did (`calls`, the tool calls
it made, each with the `server` it went to, the reply's `text`, `error`, `images` as the hash of each image block and
`blocks`; `finished`, whether it reached the end of its turn). `pulled` is the replies to its `next_frame` calls.
`memory` loads a memory SPEC (with `carry` when asked) over the two fake programs of a memory (`memory_fake_server`,
`memory_fake_sense`), and a
`Run` given it has the memory's guide in its transport's base instructions, as `cmd_codex` does.
`Run` is a started `CodexSystem` on the fake (or on a `codex` of the test's own): a context manager that closes the
system (a live exec is killed), with `batch` to write the files of a frame into its inbox and give back its step as the
harness delivers it (`stream` to observe frames in order, `serve` to serve the frame that is held after the last).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from stbench import job_registry
from stbench.codex_exec import Codex, Result
from stbench.codex_feed import Feed, Item
from stbench.codex_live import Live
from stbench.codex_log import Journal
from stbench.codex_memory import Memory
from stbench.codex_memory import load as load_memory
from stbench.codex_system import CodexSystem, Settings
from stbench.llm_render import END_MESSAGE
from stbench.questionnaire import asked_ends
from stbench.stream_protocol import CHANNELS, DEFAULT_CHANNEL

FAKE = Path(__file__).with_name("codex_fake.py")
MEMORY_SERVER = Path(__file__).with_name("memory_fake_server.py")
MEMORY_SENSE = Path(__file__).with_name("memory_fake_sense.py")
GUIDE = "Call claim for what you see.\n\nCall view before you answer."
SESSION = "0123456789abcdef"
CREDIT = {"match_m": 0.5, "clearance_m": 0.1, "reacquire_m": 0.5}  # test numbers: the vacuum's C, R = match_m
CAMERA = {"width": 640, "height": 480, "hfov_deg": 80.0, "fx": 381.3, "fy": 381.3, "cx": 320.0, "cy": 240.0,
          "depth_unit": "mm", "near_m": 0.01, "far_m": 4.0}
ROBOT = {"id": "vacuum", "name": "robot vacuum", "flies": False,
         "body": {"shape": "disc", "length_m": 0.34, "width_m": 0.34, "height_m": 0.09, "climb_m": 0.02,
                  "clearance_m": 0.02, "under_margin_m": 0.01},
         "camera": {"width": 640, "height": 480, "hfov_deg": 80.0, "mount_m": [0.13, 0.0, 0.08]}}
STATUS = {"phase": "cleaning", "task": 0, "battery_pct": 97, "brush": True}


class Fake:
    """A `Codex` that runs `codex_fake.py`, with the folder it keeps its state in."""

    def __init__(self, tmp_path: Path, plan: dict | None = None, *, model: str = "fake-model",
                 reasoning: str = "low", guide: str | None = None, compact_at: int | None = None):
        self.home = tmp_path / "fake-codex"
        self.home.mkdir()
        self.plan_file = self.home / "plan.json"
        self.plan(plan or {})
        self.waits: list[float] = []
        self.settings = {"model": model, "reasoning": reasoning, "guide": guide, "compact_at": compact_at}
        self.codex = self.new()

    def new(self, **overrides) -> Codex:
        """Another transport on the same fake (a run makes its own, as it would the real command)."""
        found = {**self.settings, **overrides}
        return Codex(found["model"], reasoning=found["reasoning"], guide=found["guide"], compact_at=found["compact_at"],
                     command=[sys.executable, str(FAKE)], sleep=self.waits.append,
                     env={"FAKE_CODEX_DIR": str(self.home), "FAKE_CODEX_PLAN": str(self.plan_file)})

    def plan(self, plan: dict) -> None:
        self.plan_file.write_text(json.dumps({str(k): v for k, v in plan.items()}), encoding="utf-8")

    def execs(self) -> list[dict]:
        """What the fake was run with and did, one record per exec, in order (a retried exec is there twice)."""
        path = self.home / "execs.jsonl"
        found = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        for record in found:
            log = self.home / f"calls-{record['n']}.jsonl"
            lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
            head = next((line for line in lines if "tools" in line), {})
            record |= {"tools": head.get("tools", []), "config": head.get("config"), "others": head.get("others", {}),
                       "tool_timeout": head.get("tool_timeout"), "calls": [line for line in lines if "tool" in line],
                       "finished": any(line.get("finished") for line in lines)}
        return found


def pulled(record: dict) -> list[dict]:
    """The calls of `next_frame` an exec made, in order."""
    return [call for call in record["calls"] if call["tool"] == "next_frame"]


def memory(folder: Path, *, name: str = "mem", fault: str | None = None, sense: bool = True,
           guide: str = GUIDE, carry: bool = False, instructions: str | None = None) -> Memory:
    """A memory loaded from a SPEC written under `folder`: its server `memory_fake_server.py` over the store (sending
    `instructions` as its `initialize` reply's, when given), its sensor feed `memory_fake_sense.py` over the store (with
    `fault`, `KIND:N`, when given; none with `sense` false) and `guide` as its prompt file; `carry` is the SPEC's."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "guide.md").write_text(guide, encoding="utf-8")
    served = [sys.executable, str(MEMORY_SERVER), "{store}", *([instructions] if instructions else [])]
    spec = {"name": name, "serve": served, "prompt": "guide.md", **({"carry": True} if carry else {}),
            **({"sense": [sys.executable, str(MEMORY_SENSE), "{store}", *([fault] if fault else [])]} if sense else {})}
    (folder / "memory.json").write_text(json.dumps(spec), encoding="utf-8")
    return load_memory(folder / "memory.json")


def job_also_asks(monkeypatch, *families: str) -> None:
    """Every job the registry is asked for also asks the end `families`. The hand cases of the driver's tests hold a
    `number` during question on `vacuum/1`, which asks only `object_where`; the prompt shows the form of the answer
    types the job asks and the driver refuses a case whose questions have another (`brief.answer_types`), so a test that
    runs such a case makes its job ask one more family of the missing type, here `distance_between` (a `number`), and
    nothing else of the job changes."""
    real = job_registry.get

    def get(job_id: str):
        job = real(job_id)
        return replace(job, ends=(*asked_ends(job), *families))

    monkeypatch.setattr(job_registry, "get", get)


def run_exec(fake: Fake, *, kind: str = "start", thread: str | None = None, prompt: str = "hello",
             limit: float = 120.0) -> Result:
    """One exec of the fake, run as `CodexSystem` runs one (`Live`, over a `Feed`) with only the end item to give, so
    that the model's first pull ends the job and its turn completes. Raises what the exec ended in."""
    feed = Feed()
    feed.supply(Item("end", END_MESSAGE))
    live = Live(fake.codex, kind=kind, thread=thread, prompt=prompt, server=feed.server_config())
    try:
        live.wait(limit)
        return live.result()
    finally:
        live.close()
        feed.close()


def jpeg(k: int) -> bytes:
    """A tiny stand-in JPEG, different for each frame."""
    return b"\xff\xd8\xff\xe0" + f"frame-{k}".encode() + b"\xff\xd9"


def jpeg_hash(k: int) -> str:
    """How the fake names the image of frame `k` in what it saw."""
    return hashlib.sha256(jpeg(k)).hexdigest()[:16]


class Run:
    """A started `CodexSystem` over a `Fake`: the working folder has `public/camera.json` and `public/robot.json` (the
    robot's `ROBOT`, or `robot`) and an empty `inbox/`, the state folder is empty, and `journal` is the run's log.
    `settings` and `timeouts` go to the system; `channel` is the one the case is started on (the default one). Use it as
    a context manager: leaving it closes the system, which kills an exec still waiting for its next item."""

    def __init__(self, tmp_path: Path, plan: dict | None = None, *, codex=None, settings: Settings | None = None,
                 channel: str = DEFAULT_CHANNEL, robot: dict = ROBOT, **opts):
        held = opts.get("memory")
        self.fake = Fake(tmp_path, plan, guide=held.guide if held else None) if codex is None else None
        self.work, self.state = tmp_path / "work", tmp_path / "state"
        (self.work / "public").mkdir(parents=True)
        (self.work / "public" / "camera.json").write_text(json.dumps(CAMERA), encoding="utf-8")
        (self.work / "public" / "robot.json").write_text(json.dumps(robot), encoding="utf-8")
        self.state.mkdir()
        self.journal = Journal(tmp_path / "codex.calls.jsonl")
        self.log = self.journal.path
        extra = {"settings": settings} if settings else {}
        self.system = CodexSystem(self.state, self.work, codex=codex or self.fake.codex, journal=self.journal, **extra,
                                  **opts)
        self.channel = channel
        self.system.start(SESSION, robot, channel)

    def __enter__(self) -> Run:
        return self

    def __exit__(self, *_) -> None:
        self.system.close()

    def batch(self, ks, *, depth_mm: int = 2000, base: dict | None = None, camera: dict | None = None,
              detections: dict[int, list] | None = None) -> list:
        """The inbox of the frames `ks` (emptied first, as the harness does before each frame) and their steps as the
        harness delivers them on the run's channel, each frame's clock one second after the last: with `rgb` (a tiny
        JPEG) where the channel sends the camera image; with `depth` (a flat depth image of `depth_mm`, the sensor's
        16-bit file) where it sends depth; with `detections` (by frame, none for a frame left out) where it sends
        those. The inbox holds only what the channel delivers. The stream sends one frame an `observe`: give one k."""
        sends = CHANNELS[self.channel]
        inbox = self.work / "inbox"
        shutil.rmtree(inbox, ignore_errors=True)
        (inbox / "rgb").mkdir(parents=True)
        (inbox / "depth").mkdir()
        steps = []
        for k in ks:
            step = {
                "k": k, "clock": f"2026-03-02T08:00:{k:02d}",
                "pose": base or {"pos": [5.6, 5.05 - 0.1 * k, 0.0], "rot": [0.7071, 0.0, 0.0, -0.7071],
                                 "yaw_deg": -90.0},
                "camera": camera or {"pos": [5.6, 4.92 - 0.1 * k, 0.08], "rot": [0.7071, 0.0, 0.0, -0.7071],
                                     "pitch_deg": 5.0},
                "status": dict(STATUS)}
            if "rgb" in sends:
                (inbox / "rgb" / f"{k:06d}.jpg").write_bytes(jpeg(k))
                step["rgb"] = f"inbox/rgb/{k:06d}.jpg"
            if "depth" in sends:
                Image.fromarray(np.full((480, 640), depth_mm, dtype=np.uint16)).save(inbox / "depth" / f"{k:06d}.png")
                step["depth"] = f"inbox/depth/{k:06d}.png"
            if "detections" in sends:
                step["detections"] = list((detections or {}).get(k, []))
            steps.append(step)
        return steps

    def stream(self, ks, **kw) -> None:
        """Observe the frames `ks` one at a time, in order, each with its own inbox, as the stream does. The last one is
        held, not yet served (`CodexSystem`'s held frame): `serve` serves it, as the next message would."""
        for k in ks:
            self.system.observe(self.batch([k], **kw))

    def serve(self) -> None:
        """Serve the frame that is held, alone, and wait for the pull after it (what an `observe`, a `task` or `finish`
        sent next does first). Raises what that wait raises."""
        self.system._flush()
