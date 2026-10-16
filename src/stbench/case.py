"""The dataset layout, the private index, and cases: a window of a script's tasks, delivered from step 0.

```text
DATASET/
  scripts/<sid>/              agent-facing ONLY; <sid> is 16 random hex characters
    robot.json
    sensors/  camera.json pose.jsonl rgb/%06d.jpg
    perception/  oracle.jsonl
  private/                    scorer only
    index/<id>.json           one entry a script or case; `read_index` aggregates them
    <sid>/  timeline.json depth/%06d.png instances/%06d.png instance_map.json visibility.jsonl perception_ids.jsonl
            render_log.json
    <cid>/  case.json questions.jsonl gold.jsonl
```
The depth the robot recorded is the scorer's, not the agent's (0.6.0): the agent is fed the camera, so no depth file
lives under `scripts/`, and only the `detections` gate channel is sent depth, copied from `private/<sid>/depth/`.
Ids come from the OS random source, not from the inputs, so a folder name reveals nothing; an equal index entry reuses
its id, so a rebuild makes no duplicate. A case's start clock is drawn from its id alone, so it names no input. There is
no mission, floor plan or protocol file: briefs are never files.

Every number a case delivers is case-relative. Private records (evidence, decisions, `visibility.jsonl`,
`perception_ids.jsonl`) stay in script numbering and are converted only by `case_k`, `script_k`, `case_clock` and
`case_task`, which the harness and the compiler both import, so question text and stream agree: a window starts at k 0,
task 0, at its own start clock, whatever came before it in the script.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import secrets
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .seeding import make_rng
from .timeline import Timeline, load

# The keys of an index entry that earlier builds wrote and nothing reads: `read_entry` drops them, so an old entry still
# equals the entry the code builds now (`claim_id` reuses its id).
OLD_ENTRY_KEYS = frozenset({"schema", "generator"})
SCRIPTS, PRIVATE = "scripts", "private"
LENGTHS = {"short": (1, 1), "medium": (2, 3), "long": (4, 6)}  # tasks a window holds
RETRY_S = 5.0   # how long an index file held by another process is waited for (`_retry`)


# -- the private index -------------------------------------------------------------------------------------------
def index_dir(dataset: str | Path) -> Path:
    """`private/index/`: one small file per script or case, so ones written in parallel never rewrite a shared file."""
    return Path(dataset) / PRIVATE / "index"


def _retry(fn):
    """`fn()`, retried on `PermissionError` for up to `RETRY_S`. On Windows a file swapped in by `os.replace` cannot be
    opened, nor the swap made, while another process has the file open (a parallel build reading and writing the
    index), so every index read and write goes through this one retry. The waits double from 5 ms to a cap of 0.4 s
    and each is drawn at random between its half and its whole: a fixed schedule has every process that collided wake
    together and collide again (about 1.6 s in all, which a suite run under load outlasted)."""
    deadline, wait = time.monotonic() + RETRY_S, 0.005
    while True:
        try:
            return fn()
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
        time.sleep(random.uniform(wait / 2, wait))
        wait = min(wait * 2, 0.4)


def read_entry(dataset: str | Path, iid: str) -> dict | None:
    """The index entry of a script or case (without the keys of `OLD_ENTRY_KEYS`), or None when there is none."""
    p = index_dir(dataset) / f"{iid}.json"

    def read() -> dict | None:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:  # not `exists()`: on Windows that says False for a file mid-swap, where open retries
            return None

    doc = _retry(read)
    return None if doc is None else {k: v for k, v in doc.items() if k not in OLD_ENTRY_KEYS}


def read_index(dataset: str | Path) -> dict[str, dict]:
    """Every entry by id, aggregated from the index folder. A file that is gone between the listing and the read was
    caught in the middle of a swap (NTFS replaces non-atomically), so it is left out like one the listing missed."""
    d = index_dir(dataset)
    entries = {p.stem: read_entry(dataset, p.stem) for p in sorted(d.glob("*.json"))} if d.exists() else {}
    return {iid: e for iid, e in entries.items() if e is not None}


def _write_json(path: Path, doc: dict) -> None:
    """A JSON file by temp file and `os.replace`, so a reader never sees half of it; the swap is retried like a read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{secrets.token_hex(4)}.tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    _retry(lambda: os.replace(tmp, path))


def write_entry(dataset: str | Path, iid: str, entry: dict) -> None:
    _write_json(index_dir(dataset) / f"{iid}.json", entry)


def claim_id(dataset: str | Path, entry: dict) -> str:
    """The id of an equal entry already in the index, else a fresh random one."""
    index = read_index(dataset)
    found = next((i for i, e in index.items() if e == entry), None)
    while found is None:
        cand = secrets.token_hex(8)
        if cand not in index:
            found = cand
    return found


# -- paths -------------------------------------------------------------------------------------------------------
class ScriptPaths:
    """Every file a script can hold. `root` is the agent-facing folder, `private` the scorer's."""

    def __init__(self, dataset: str | Path, sid: str):
        self.dataset = Path(dataset)
        self.id = sid
        self.root = self.dataset / SCRIPTS / sid
        self.private = self.dataset / PRIVATE / sid

    @classmethod
    def from_dir(cls, path: str | Path) -> ScriptPaths:
        """From `DATASET/scripts/<sid>` or `DATASET/private/<sid>`; a private folder whose index entry is not a script
        (a case's) is refused."""
        p = Path(path).resolve()
        if p.parent.name not in (SCRIPTS, PRIVATE):
            raise ValueError(f"{path} is not DATASET/{SCRIPTS}/<id> or DATASET/{PRIVATE}/<id>")
        entry = read_entry(p.parent.parent, p.name) if p.parent.name == PRIVATE else None
        if entry is not None and entry.get("kind") != "script":
            raise ValueError(f"{path} is a {entry.get('kind')}, not a script")
        return cls(p.parent.parent, p.name)

    # agent-facing
    robot = property(lambda self: self.root / "robot.json")
    sensors = property(lambda self: self.root / "sensors")
    camera = property(lambda self: self.root / "sensors" / "camera.json")
    pose = property(lambda self: self.root / "sensors" / "pose.jsonl")
    rgb_dir = property(lambda self: self.root / "sensors" / "rgb")
    perception = property(lambda self: self.root / "perception")
    oracle = property(lambda self: self.root / "perception" / "oracle.jsonl")
    # private
    timeline = property(lambda self: self.private / "timeline.json")
    depth_dir = property(lambda self: self.private / "depth")   # the scorer's, never in `scripts/` (0.6.0)
    instances_dir = property(lambda self: self.private / "instances")
    instance_map = property(lambda self: self.private / "instance_map.json")
    visibility = property(lambda self: self.private / "visibility.jsonl")
    perception_ids = property(lambda self: self.private / "perception_ids.jsonl")
    render_log = property(lambda self: self.private / "render_log.json")  # holds the render seed

    def rgb(self, k: int) -> Path:
        return self.rgb_dir / f"{k:06d}.jpg"

    def depth(self, k: int) -> Path:
        return self.depth_dir / f"{k:06d}.png"

    def instances(self, k: int) -> Path:
        return self.instances_dir / f"{k:06d}.png"

    def mkdirs(self) -> None:
        for d in (self.rgb_dir, self.depth_dir, self.perception, self.instances_dir):
            d.mkdir(parents=True, exist_ok=True)


def depth_before_0_6_0(script: ScriptPaths) -> Path:
    """Where a dataset made before 0.6.0 kept a script's depth: under the agent-facing folder, which now holds none."""
    return script.sensors / "depth"


def holds_files(folder: Path) -> bool:
    """Whether `folder` holds a file anywhere under it (an empty folder, or one of empty folders, holds none)."""
    return folder.is_dir() and any(p.is_file() for p in folder.rglob("*"))


def _named(ids: list[str]) -> str:
    return ", ".join(ids[:3]) + (f" (and {len(ids) - 3} more)" if len(ids) > 3 else "")


def depth_move_problem(scripts: Iterable[ScriptPaths]) -> str | None:
    """None, or the sentence `stbench compile`, `check`, `perceive` and a stream refuse with when a script's
    agent-facing folder still holds depth images (the layout before 0.6.0): the code reads depth from
    `private/<sid>/depth/` alone, and a depth file left under `scripts/` is a leak. It names the scripts and the cure:
    `stbench move-depth` moves a folder whose private place is free; a script that holds frames in both places (a
    render made after the move left the old frames behind) needs the old folder deleted, which the command will not do
    for it, since it cannot say which frames are current."""
    old = sorted({s.id: s for s in scripts if holds_files(depth_before_0_6_0(s))}.values(), key=lambda s: s.id)
    if not old:
        return None
    both = [s.id for s in old if holds_files(s.depth_dir)]
    text = (f"{len(old)} script(s) still hold depth images under scripts/<id>/sensors/depth/, the layout before 0.6.0, "
            f"e.g. {_named([s.id for s in old])}: depth is the scorer's and lives in private/<id>/depth/. Run "
            "`stbench move-depth --data DATASET` to move them")
    if both:
        text += (f". {len(both)} of them (e.g. {_named(both)}) hold frames in private/<id>/depth/ too, left by a "
                 "render made after the move: delete their old scripts/<id>/sensors/depth/ folder, which "
                 "move-depth refuses to touch")
    return text


class CasePaths:
    """A case: its private folder, and its script's paths by composition."""

    def __init__(self, script: ScriptPaths, cid: str):
        self.script = script
        self.dataset = script.dataset
        self.id = cid
        self.private = self.dataset / PRIVATE / cid

    @classmethod
    def from_id(cls, dataset: str | Path, cid: str) -> CasePaths:
        """The case with this id, its script read from the case's index entry."""
        entry = read_entry(dataset, cid)
        if entry is None or entry.get("kind") != "case":
            raise ValueError(f"{cid} is not a case in {dataset}")
        return cls(ScriptPaths(dataset, entry["script"]), cid)

    @classmethod
    def from_dir(cls, path: str | Path) -> CasePaths:
        """From `DATASET/private/<cid>`: the folder the commands over one case take."""
        p = Path(path).resolve()
        if p.parent.name != PRIVATE:
            raise ValueError(f"{path} is not DATASET/{PRIVATE}/<cid>")
        return cls.from_id(p.parent.parent, p.name)

    record = property(lambda self: self.private / "case.json")
    questions = property(lambda self: self.private / "questions.jsonl")
    gold = property(lambda self: self.private / "gold.jsonl")


# -- the case record and its offsets -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CaseRecord:
    """`private/<cid>/case.json`: the window of tasks a, a+1 .. b (script indices) and where it sits in the script."""

    case: str  # the case id
    script: str  # the script id
    a: int  # the window's first task (script index)
    b: int  # its last task, inclusive
    k0: int  # = tasks[a].k0
    k1: int  # = tasks[b].k1; the window is keyframes [k0, k1)
    t0: float  # the script time of k0
    start_clock: str  # the clock of the case's first step, drawn per case
    length: str  # "short", "medium" or "long"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CaseRecord:
        """It reads only the keys it names: the `case.json` of an earlier build may carry a `schema`."""
        return cls(d["case"], d["script"], int(d["a"]), int(d["b"]), int(d["k0"]), int(d["k1"]), float(d["t0"]),
                   d["start_clock"], d["length"])


def read_case(paths: CasePaths) -> CaseRecord:
    return CaseRecord.from_dict(json.loads(paths.record.read_text(encoding="utf-8")))


def case_k(rec: CaseRecord, k: int) -> int:
    return k - rec.k0


def script_k(rec: CaseRecord, kc: int) -> int:
    return kc + rec.k0


def start_clock(rng: np.random.Generator) -> str:
    """A date in 2026 and a start time between 07:00 and 10:00, from the given generator: the one owner of the
    distribution, for a case's clock (`write_case`) and a script's own."""
    day = datetime(2026, 1, 1) + timedelta(days=int(rng.integers(0, 365)))
    return (day + timedelta(hours=7, minutes=int(rng.integers(0, 180)))).isoformat(timespec="seconds")


def format_clock(at: datetime) -> str:
    """A delivered clock, `YYYY-MM-DDTHH:MM:SS.s`: the local ISO time with one decimal of a second, always (the tenth is
    kept as it is, "07:12:00.0" and not "07:12:00"), so every clock a system reads has one shape. Tenths of a second,
    because a walking person covers a metre in under a second and the second-rounded clock moved `closest_approach`'s
    gold by more than C on 6% of person rows."""
    return f"{at:%Y-%m-%dT%H:%M:%S}.{at.microsecond // 100_000}"


def case_clock(rec: CaseRecord, t: float) -> str:
    """The case's clock at script time t, as a system is sent it (`format_clock`): its own start clock plus the true
    time since the window's first step, rounded half up to the tenth of a second after the shift. The script's own
    clock (`timeline.clock_at`, the pose rows') stays to the second; only a delivered clock carries tenths."""
    tenths = math.floor((t - rec.t0) * 10 + 0.5)
    return format_clock(datetime.fromisoformat(rec.start_clock) + timedelta(milliseconds=100 * tenths))


def case_task(rec: CaseRecord, i: int) -> int:
    return i - rec.a


def gap_s(tl: Timeline, rec: CaseRecord, i: int) -> float:
    """The unobserved seconds before task i: 0 for the window's first task, else from the last keyframe of task i - 1
    to the first of task i."""
    if i == rec.a:
        return 0.0
    return tl.keyframe(tl.ego.tasks[i].k0).t - tl.keyframe(tl.ego.tasks[i - 1].k1 - 1).t


VELOCITY_DECIMALS = 3   # a delivered velocity is to the millimetre a second: odometry reads no finer


def base_velocity(tl: Timeline, rows: Mapping[int, dict], k: int) -> list[float]:
    """The base's own velocity `[vx, vy]` at script keyframe `k`, metres a second in the map frame: the controller's
    odometry that every delivered step carries (`velocity`). It is the displacement of the delivered base position
    (`rows[k]["pose"]["pos"]`, script numbering) from the previous delivered frame of the same task (`status.task`) to
    `k`, over the true time between the two (`tl.keyframe(..).t`, never the tenth-rounded clock that is delivered),
    rounded to `VELOCITY_DECIMALS`. `[0.0, 0.0]` at a task's first frame, which has no frame before it in its task: the
    robot is not known to be still there, the controller simply has nothing to difference. `rows` holds the window's
    pose rows by `k`, so a window that starts at a task's first frame needs nothing before it."""
    before, row = rows.get(k - 1), rows[k]
    dt = tl.keyframe(k).t - tl.keyframe(k - 1).t if before is not None else 0.0
    if before is None or before["status"]["task"] != row["status"]["task"] or dt <= 0:
        return [0.0, 0.0]
    moved = [now - was for was, now in zip(before["pose"]["pos"][:2], row["pose"]["pos"][:2], strict=True)]
    return [round(d / dt, VELOCITY_DECIMALS) + 0.0 for d in moved]   # + 0.0: no "-0.0"


def case_rows(rec: CaseRecord, tl: Timeline, pose_rows: list[dict]) -> list[dict]:
    """The window's pose rows with k, clock and status.task rebased to the case, and the base's own `velocity`
    (`base_velocity`) added: what a step carries."""
    by_k = {row["k"]: row for row in pose_rows}
    rows = []
    for row in pose_rows:
        if rec.k0 <= row["k"] < rec.k1:
            velocity = base_velocity(tl, by_k, row["k"])
            row = copy.deepcopy(row)
            row["clock"] = case_clock(rec, tl.keyframe(row["k"]).t)
            row["status"]["task"] = case_task(rec, row["status"]["task"])
            row["velocity"] = velocity
            row["k"] = case_k(rec, row["k"])
            rows.append(row)
    return rows


def windows(n_tasks: int, rng, starts: Sequence[int] | None = None) -> list[tuple[int, int, str]]:
    """(a, b, length) of one window per length whose minimum fits in `n_tasks`: its size uniform in
    [lo, min(hi, n_tasks)], its first task uniform among those that leave room for it. Never splits a task.

    `starts` (script task indices) are the only tasks a window may begin at: a job's `continues` kinds (a `resume`) are
    left out, since a window beginning there would show that earlier tasks exist. Then the size is uniform over the
    sizes some allowed start leaves room for, and the first task uniform over the allowed starts that do; a length no
    allowed start fits is left out. With every start allowed it makes the same `rng.integers` calls as `None` (today's
    draws, so a one-task script keeps its window)."""
    allowed = None if starts is None else sorted({a for a in starts if 0 <= a < n_tasks})
    out = []
    for length, (lo, hi) in LENGTHS.items():
        top = min(hi, n_tasks if allowed is None else n_tasks - min(allowed, default=n_tasks))
        if lo > top:
            continue
        size = int(rng.integers(lo, top + 1))
        if allowed is None:
            a = int(rng.integers(0, n_tasks - size + 1))
        else:
            fit = [a for a in allowed if a + size <= n_tasks]
            a = fit[int(rng.integers(0, len(fit)))]
        out.append((a, a + size - 1, length))
    return out


def write_case(dataset: str | Path, sid: str, a: int, b: int, length: str) -> CasePaths:
    """`case.json` and the case's index entry for tasks a..b of script `sid`; an equal entry reuses its id. The case's
    start clock is drawn from the case's own random id alone, never from the script's map, seed, robot or job or from
    the window, so a delivered clock names none of them; an equal entry reuses its id and so its clock."""
    dataset = Path(dataset)
    script = ScriptPaths(dataset, sid)
    tl = load(script.timeline)
    if length not in LENGTHS:
        raise ValueError(f"a case's length is one of {list(LENGTHS)}, not {length!r}")
    lo, hi = LENGTHS[length]
    if not (0 <= a <= b < len(tl.ego.tasks) and lo <= b - a + 1 <= hi):
        raise ValueError(f"tasks {a}..{b} are not a {length} window ({lo}-{hi} tasks) of a {len(tl.ego.tasks)}-task "
                         f"script")
    k0, k1 = tl.ego.tasks[a].k0, tl.ego.tasks[b].k1
    entry = {"kind": "case", "script": sid, "a": a, "b": b, "length": length}
    cid = claim_id(dataset, entry)
    paths = CasePaths(script, cid)
    rec = CaseRecord(cid, sid, a, b, k0, k1, tl.keyframe(k0).t, start_clock(make_rng("case_clock", cid)), length)
    _write_json(paths.record, rec.to_dict())
    write_entry(dataset, cid, entry)
    return paths
