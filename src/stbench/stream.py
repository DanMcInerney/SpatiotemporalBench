"""The harness side of the stream protocol: stream one case's window to a system and collect its answers.

`run_case` makes a run folder, builds the window's steps from the case's script (every number case-relative: `case.py`
owns the offsets, so nothing delivered names where the window sits in its script), starts the system, sends each task's
`task` message, each step as one frame (one `observe` of one step, the inbox holding only that frame's files, copied
plain; a system whose class sets `reads_files` False, one that opens no picture, is given none) and, right after the
frame it is about, each during ask; then the end ask and `finish`. The messages and the failure semantics are in
`stream_protocol.py`'s docstring; this module decides by what a call raised:

- `make`, `start`, `task` or `observe` raising anything ends the case; so does an ask raising `TransportError` (a
  timeout, an exit, a bad line), which fails that ask's questions and every later one;
- an ask raising anything else, or answering with a reply of the wrong shape, fails that ask's questions only;
- `read_answers` fails a question the reply left out, repeated or did not answer with an object;
- `finish` raising leaves the answers standing and `usage` None;
- past `cap_s` (checked before each message) every question not yet answered fails.

A failed question's answer is `stream_protocol.failed(reason)`. Every message is timed here, for every system; the
adapters (`system.py`) enforce each message's timeout, and in-process systems have none. `CaseRun.answers` holds the
questions this run asked: the during questions are left out when `during` is off, and a question the channel cannot
answer is never part of the run.

The velocity. A step on a channel that sends `pose` carries the robot's own `velocity` (`case.case_rows` computes it
from the delivered pose rows of the current task and the true script times: the controller's odometry, 0 at a task's
first frame), the numbers the gold of `closest_approach` reads.

The subgoal. A pose row of a job that states one carries `subgoal` (`episode.pose_rows`), and a step carries it on
every channel, beside `status`. The `neutral` brief withholds it: that brief takes the assignment out, and a subgoal
that names the task's own things would put it back (0.5.0-control, FD-0.5.0-control-impl §15 item 7). `swapped` is
refused for a job that states one: the subgoal says the true task in every frame, and the brief would contradict it.
"""

from __future__ import annotations

import json
import pickle
import secrets
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import job_registry, look_windows
from .brief import compose
from .case import (
    CasePaths,
    CaseRecord,
    case_clock,
    case_k,
    case_rows,
    case_task,
    depth_move_problem,
    gap_s,
    read_case,
    script_k,
)
from .records import check_question
from .safecopy import remove_tree, tree_bytes
from .seeding import stable_hex
from .stream_protocol import (
    CASE_CAP_S,
    CHANNELS,
    DEFAULT_CHANNEL,
    USAGE_KEYS,
    System,
    TransportError,
    ask_view,
    failed,
    provides,
    read_answers,
    structured_view,
    timeout_key,
)
from .timeline import Timeline, load

ASK_ERROR, CASE_CAP = "ask error", "case cap"   # failure reasons beside `read_answers`' and "case ended: <message>"
MAX_STATUS = 300   # characters of an exception's text kept in a case's status
# The reason the swapped brief is refused (here for a job with a subgoal, and by `stbench eval` for every job).
SWAPPED_RETIRED = ("the swapped brief is retired: a subgoal says the true task in every frame, so a swapped brief "
                   "would contradict it")


@dataclass
class CaseRun:
    answers: dict[str, dict]           # qid -> answer, or stream_protocol.failed(reason)
    status: str                        # "complete" or "ended: <reason>"
    failures: dict[str, int]           # failed questions by reason
    wall_s: dict[str, list[float]]     # seconds per message, by timeout key: start, task, observe, ask_*, finish
    state_bytes: dict[str, int]        # per case task end ("0", "1", ...) and "finish"
    usage: dict | None                 # finish's reply, stamped {"self_reported": true, ...}; None when it failed
    session: str                       # secrets.token_hex(8), never the case id


@dataclass(frozen=True)
class _Task:
    index: int       # case-relative
    k0: int          # its first case k
    k1: int          # one past its last
    clock: str
    gap_s: float
    brief: str


@dataclass(frozen=True)
class _Window:
    """What a case delivers: its steps, the files behind them, its tasks and the compiler's decision windows over its
    steps (all case-relative). A system is sent the steps one at a time; the windows are the compiler's cut for a
    decision point's own families (`World.window_ks`), kept here for `stbench view`, and never decide where an ask may
    fall."""

    steps: list[dict]
    files: list[list[tuple[Path, str]]]   # per step: (source file, name under the working folder)
    tasks: list[_Task]
    windows: list[tuple[int, int]]   # `look_windows.windows`: [lo, hi) over the steps, cut at decisions and task ends
    robot: dict


class _Ended(Exception):
    """The case is over: `where` is the message (or "cap") that ended it, `error` what it raised (None for the cap)."""

    def __init__(self, where: str, error: BaseException | None = None):
        super().__init__(where)
        self.where, self.error = where, error


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _short(error: BaseException) -> str:
    return " ".join(f"{type(error).__name__}: {error}".split())[:MAX_STATUS]


def _rekey(cid: str, kc: int, detections: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """A frame's detections as a system gets them, and each old key's new one. The perception oracle draws a keyframe's
    keys and its shuffle from the script's inputs and the keyframe's place in its script, so a system could search them
    for the seed and the window's place; here the frame's detections are put in a canonical order (by content) and the
    i-th gets `d` = `stable_hex(case id, "d", case k, i)`: nothing delivered depends on either. ValueError when two
    collide."""
    ordered = sorted(detections, key=lambda d: json.dumps({k: v for k, v in d.items() if k != "d"}, sort_keys=True))
    keys = [stable_hex(cid, "d", kc, i) for i in range(len(ordered))]
    if len(set(keys)) != len(keys):
        raise ValueError(f"two detections of case step {kc} drew one key")
    return ([{**d, "d": key} for d, key in zip(ordered, keys, strict=True)],
            {d["d"]: key for d, key in zip(ordered, keys, strict=True)})


def _window(src: CasePaths, rec: CaseRecord, tl: Timeline, *, cid: str, channel: str, privileged: bool,
            variant: str) -> _Window:
    """The window of `src` as a system on `channel` gets it, its detections keyed from case `cid`."""
    script, sends = src.script, CHANNELS[channel]
    job = job_registry.get(tl.episode.job)
    if variant == "swapped" and job.subgoals is not None:
        raise ValueError(f"{job.id}: {SWAPPED_RETIRED}")
    robot = json.loads(script.robot.read_text(encoding="utf-8"))
    plans = [tl.ego.tasks[i].plan for i in range(rec.a, rec.b + 1)]
    texts = compose(plans, job=job, robot=robot, domain=job.domain, channel=channel, variant=variant)
    seen = {r["k"]: r["detections"] for r in _jsonl(script.oracle)} if "detections" in sends else {}
    ids = {r["k"]: r["ids"] for r in _jsonl(script.perception_ids)} if privileged and "detections" in sends else {}
    steps, files = [], []
    for row in case_rows(rec, tl, _jsonl(script.pose)):
        kc = row["k"]
        k = script_k(rec, kc)
        step: dict = {"k": kc, "clock": row["clock"]}
        names: list[tuple[Path, str]] = []
        if "pose" in sends:
            step["pose"], step["camera"], step["velocity"] = row["pose"], row["camera"], row["velocity"]
        if "status" in sends:
            step["status"] = row["status"]
        if "subgoal" in row and variant != "neutral":
            step["subgoal"] = row["subgoal"]
        if "rgb" in sends:
            step["rgb"] = f"inbox/rgb/{kc:06d}.jpg"
            names.append((script.rgb(k), step["rgb"]))
        if "depth" in sends:
            step["depth"] = f"inbox/depth/{kc:06d}.png"
            names.append((script.depth(k), step["depth"]))
        if "detections" in sends:
            if k not in seen:
                raise ValueError(f"{script.oracle} has no row for keyframe {k}")
            step["detections"], new = _rekey(cid, kc, seen[k])
            if privileged:
                if k not in ids:
                    raise ValueError(f"{script.perception_ids} has no row for keyframe {k}")
                step["identity"] = {new[old]: entity for old, entity in ids[k].items()}
        elif privileged:
            step["identity"] = {}
        steps.append(step)
        files.append(names)
    if not steps or [s["k"] for s in steps] != list(range(rec.k1 - rec.k0)):
        raise ValueError(f"{script.pose} does not hold the window's keyframes {rec.k0}..{rec.k1 - 1}")
    tasks = [_Task(case_task(rec, i), case_k(rec, tl.ego.tasks[i].k0), case_k(rec, tl.ego.tasks[i].k1),
                   case_clock(rec, tl.keyframe(tl.ego.tasks[i].k0).t), gap_s(tl, rec, i), texts[case_task(rec, i)])
             for i in range(rec.a, rec.b + 1)]
    cuts = {case_k(rec, d.k) for d in tl.decisions_in(rec.k0, rec.k1)} | {t.k1 - 1 for t in tasks}
    return _Window(steps, files, tasks, look_windows.windows(len(steps), cuts), robot)


def delivered(case: CasePaths, channel: str = DEFAULT_CHANNEL, brief: str = "correct") -> _Window:
    """The window `run_case` delivers for `case` on `channel`, built by the same code and read from the same files: its
    steps (case-relative, with their detections keyed as a system gets them), their files, the tasks and the look
    windows.
    For a reader of a finished run (`view_frames`); the harness itself builds the window in `_run`. Whatever the files
    lack raises as it does in a run: FileNotFoundError, KeyError, ValueError."""
    return _window(case, read_case(case), load(case.script.timeline), cid=case.id, channel=channel, privileged=False,
                   variant=brief)


def _questions(case: CasePaths, channel: str, during: bool) -> list[dict]:
    """The case's questions this run asks, in file order: each row is checked, only those the channel can answer
    (`provides`) stay, and the during ones go when `during` is off. ValueError for a bad row or none left."""
    can = provides(channel)
    rows = _jsonl(case.questions)
    for n, row in enumerate(rows):
        if problems := check_question(row):
            raise ValueError(f"{case.questions}: row {n} is not a valid question: {'; '.join(problems)}")
    if len({q["qid"] for q in rows}) != len(rows):
        raise ValueError(f"{case.questions} holds one qid twice")
    asked = [q for q in rows if set(q["needs"]) <= can and (during or q["when"] == "end")]
    if not asked:
        raise ValueError(f"case {case.id} has no question to ask on the {channel} channel"
                         + ("" if during else " with the during questions off"))
    return asked


def _during_at(questions: list[dict], window: _Window, rec: CaseRecord,
               ask_map: dict[int, int] | None) -> dict[int, list[dict]]:
    """The during questions by the case step each is asked after (`ask_map` sends a question's decision point to the
    stream's own). Any step of the window may be asked: a system gets one frame at a time, and the ask follows its
    frame. One that falls outside the window is a harness bug: ValueError."""
    at: dict[int, list[dict]] = {}
    for q in questions:
        if q["when"] != "during":
            continue
        k = q["decision_k"] if ask_map is None else ask_map.get(q["decision_k"])
        kc = None if k is None else case_k(rec, k)
        if kc is None or not 0 <= kc < len(window.steps):
            raise ValueError(f"question {q['qid']} is asked at step {kc} of its stream, which lies outside the window")
        at.setdefault(kc, []).append(q)
    return at


class _Run:
    """One case's book: what is still to answer, the answers and failures so far, wall times, sizes and the status."""

    def __init__(self, questions: list[dict], cap_s: float):
        self.pending = {q["qid"]: q for q in questions}
        self.answers: dict[str, dict] = {}
        self.failures: Counter[str] = Counter()
        self.wall_s: dict[str, list[float]] = {}
        self.state_bytes: dict[str, int] = {}
        self.status, self.usage = "complete", None
        self.cap_s, self.began = cap_s, time.monotonic()

    def settle(self, qid: str, answer: dict) -> None:
        self.pending.pop(qid)
        self.answers[qid] = answer

    def fail(self, questions: list[dict], reason: str) -> None:
        for q in questions:
            self.settle(q["qid"], failed(reason))
            self.failures[reason] += 1

    def end(self, ended: _Ended) -> None:
        """The case is over: the status says why, and every question not yet answered fails."""
        if ended.where == "cap":
            self.status, reason = f"ended: {CASE_CAP}", CASE_CAP
        else:
            self.status, reason = f"ended: {ended.where}: {_short(ended.error)}", f"case ended: {ended.where}"
        self.fail(list(self.pending.values()), reason)

    def message(self, key: str, fn: Callable, *args):
        """`fn(*args)` timed under `key`; `_Ended("cap")` instead when the case has run past its cap."""
        if time.monotonic() - self.began > self.cap_s:
            raise _Ended("cap")
        began = time.monotonic()
        try:
            return fn(*args)
        finally:
            self.wall_s.setdefault(key, []).append(round(time.monotonic() - began, 3))

    def must(self, op: str, fn: Callable, *args) -> None:
        """A message whose failure, of any kind, ends the case."""
        try:
            self.message(op, fn, *args)
        except _Ended:
            raise
        except Exception as error:
            raise _Ended(op, error) from error

    def ask(self, system: System, when: str, now: dict, questions: list[dict], view: Callable[[dict], dict]) -> None:
        """One ask: a transport failure ends the case, any other failure fails these questions, and what the reply
        answers wrongly fails only its question."""
        try:
            reply = self.message(timeout_key("ask", when), system.ask, when, now, [view(q) for q in questions])
            answers, why = read_answers(questions, reply)
        except _Ended:
            raise
        except TransportError as error:
            raise _Ended("ask", error) from error
        except Exception:  # noqa: BLE001  (an error reply, a wrong shape, a system's own exception)
            self.fail(questions, ASK_ERROR)
            return
        for q in questions:
            if q["qid"] in answers:
                self.settle(q["qid"], answers[q["qid"]])
            else:
                self.fail([q], why[q["qid"]])


def _copy(data):
    """An independent copy of plain data (dicts, lists, numbers and strings), as `copy.deepcopy` makes, a good deal
    faster: a system is handed each frame's step and may do what it likes with it."""
    return pickle.loads(pickle.dumps(data, pickle.HIGHEST_PROTOCOL))


def _fill_inbox(work: Path, names: list[tuple[Path, str]]) -> None:
    """Empty `work/inbox` and put one frame's files in it, as plain copies: a file's time never carries the script's."""
    remove_tree(work / "inbox")
    (work / "inbox").mkdir()
    for source, name in names:
        target = work / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _stream(run: _Run, system: System, window: _Window, during_at: dict[int, list[dict]], end: list[dict], *,
            session: str, channel: str, work: Path, state: Path, view: Callable[[dict], dict], files: bool) -> None:
    """Every message of the case, in order; `_Ended` when a failure ends it. One frame at a time: a `task` message when
    the step begins a task, the step's pictures in an emptied inbox when the system reads them (`files`:
    `System.reads_files`), `observe` of that one step, then the during asks at that step."""
    run.must("start", system.start, session, _copy(window.robot), channel)
    starts = {t.k0: t for t in window.tasks}
    ends = {t.k1: t for t in window.tasks}
    for i, step in enumerate(window.steps):
        if i in starts:
            task = starts[i]
            run.must("task", system.task, task.index, task.clock, task.gap_s, task.brief)
        if files:
            _fill_inbox(work, window.files[i])
        run.must("observe", system.observe, _copy([step]))
        for q in during_at.get(i, []):
            run.ask(system, "during", {"k": step["k"], "clock": step["clock"]}, [q], view)
        if i + 1 in ends:
            run.state_bytes[str(ends[i + 1].index)] = tree_bytes(state)
    last = window.steps[-1]
    if end:
        run.ask(system, "end", {"k": last["k"], "clock": last["clock"]}, end, view)


def run_case(make: Callable[[Path, Path], System], case: CasePaths, *, channel: str = DEFAULT_CHANNEL,
             brief: str = "correct", during: bool = True, stream_from: CasePaths | None = None,
             ask_map: dict[int, int] | None = None, cap_s: float = CASE_CAP_S) -> CaseRun:
    """Stream `case`'s window to the system `make(state_dir, workdir)` builds, and return what came back.

    `brief` is the variant `brief.compose` writes (`correct`, `swapped` or `neutral`) and `during` whether the during
    questions are asked; neither is part of a case. With `stream_from`, the steps, cuts, tasks and briefs are the other
    case's (its detections keyed from this case's id), and this case's during questions are asked at
    `ask_map[decision_k]`, a keyframe of the other case's script. ValueError, before anything is sent, for a channel
    that is not one, that the system's class does not list, or that answers no question of the case, and for a case
    whose files do not fit its records; every failure of the system itself is in the returned `CaseRun`."""
    src = stream_from or case
    if stream_from is not None and ask_map is None:
        raise ValueError("a case fed another case's stream needs `ask_map`, where its during questions are asked")
    questions = _questions(case, channel, during)
    rec, tl = read_case(src), load(src.script.timeline)
    run = _Run(questions, cap_s)
    session = secrets.token_hex(8)
    root = Path(tempfile.mkdtemp(prefix="stbench-case-"))
    try:
        state, work = root / "state", root / "work"
        for folder in (state, work / "public", work / "inbox"):
            folder.mkdir(parents=True)
        for source, name in ((src.script.robot, "robot.json"), (src.script.camera, "camera.json")):
            shutil.copyfile(source, work / "public" / name)
        try:
            system = make(state, work)
        except Exception as error:  # noqa: BLE001  (a system that cannot be built fails its case, not the harness)
            run.end(_Ended("start", error))
        else:
            try:
                _run(run, system, case, src, rec, tl, questions, state=state, work=work, session=session,
                     channel=channel, variant=brief, ask_map=ask_map)
            finally:
                try:
                    system.close()
                except Exception as error:  # noqa: BLE001  (the answers stand; the status says what did not close)
                    if run.status == "complete":
                        run.status = f"ended: close: {_short(error)}"
    finally:
        remove_tree(root)
    return CaseRun(run.answers, run.status, dict(run.failures), run.wall_s, run.state_bytes, run.usage, session)


def _run(run: _Run, system: System, case: CasePaths, src: CasePaths, rec: CaseRecord, tl: Timeline,
         questions: list[dict], *, state: Path, work: Path, session: str, channel: str, variant: str,
         ask_map: dict[int, int] | None) -> None:
    """The case on a built system: refuse what cannot run, stream, ask, finish. Failures of the system end the case
    (and are recorded in `run`); a refusal raises."""
    cls = type(system)
    channels = getattr(cls, "channels", tuple(CHANNELS))
    if channel not in channels:
        raise ValueError(f"{cls.__name__} runs on the {', '.join(channels)} channel only, not {channel!r}")
    privileged, structured = bool(getattr(cls, "privileged", False)), bool(getattr(cls, "structured", False))
    window = _window(src, rec, tl, cid=case.id, channel=channel, privileged=privileged, variant=variant)
    missing = [p for names in window.files for p, _ in names if not p.is_file()]
    if missing:
        if moved := depth_move_problem([src.script]):   # an old layout, not a dataset with pictures missing
            raise FileNotFoundError(moved)
        raise FileNotFoundError(f"{len(missing)} image files of the window are missing, for example {missing[0]}")
    during_at = _during_at(questions, window, rec, ask_map)
    run.began = time.monotonic()
    try:
        _stream(run, system, window, during_at, [q for q in questions if q["when"] == "end"], session=session,
                channel=channel, work=work, state=state, view=structured_view if structured else ask_view,
                files=bool(getattr(cls, "reads_files", True)))
    except _Ended as ended:
        run.end(ended)
        return
    try:
        reply = run.message("finish", system.finish)
        if not isinstance(reply, dict):
            raise TypeError(f"finish returned a {type(reply).__name__}, not an object")
        run.usage = {"self_reported": True, **{key: reply.get(key) for key in USAGE_KEYS}}
    except _Ended as ended:  # the cap: finish was not sent
        run.end(ended)
        return
    except Exception as error:  # noqa: BLE001  (the answers stand; no usage)
        run.end(_Ended("finish", error))
    run.state_bytes["finish"] = tree_bytes(state)

