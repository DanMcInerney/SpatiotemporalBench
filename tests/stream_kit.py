"""Helpers for the tests of the harness (`stream.run_case`, `stbench run`): synthetic cases with questions and pictures,
and scripted systems that record every call they get.

`case_on` lays out a `synth.Synth` script as a dataset (with a tiny JPEG per keyframe for the rgb channel) and a case
over some of its tasks, with the hand-written question rows given; `hand20` is the one script most tests share.
`Recorder` is a system that logs each message with what the working folder held at that moment and answers every
question `unknown`; a test subclasses it for the failure it needs, and `play` runs a case on it. `SCRIPTED` is the same
kind of system as standard-library source text, for the adapters. `fake_job` is a job with a swap pair for the brief's
tests. The question rows pass `records.check_question`; `gold_row` passes `records.check_gold`.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image

from stbench import case as case_mod
from stbench import episode as ep
from stbench.job_registry import JobSpec
from stbench.stream import CaseRun, run_case
from stbench.stream_protocol import System
from synth import HOME, Synth

K, L = HOME.rooms   # the home stage's two rooms: "kitchen" and "living room"
UNKNOWN = {"type": "unknown"}


def qid(n: int) -> str:
    """A question id of the compiler's form (16 lowercase hex characters): 16 digits of `n`."""
    return f"{n:016x}"


def during_q(n: int, decision_k: int, *, needs: tuple[str, ...] = ("see", "pose"), **over) -> dict:
    """A hand-written during question asked after script keyframe `decision_k`. It needs the camera and the pose, which
    every channel gives, so it is asked on the default channel (`rgb_pose`, no depth image) as on the others; a test of
    the channel filter passes `needs=("depth", "pose")`."""
    row = {"qid": qid(n), "when": "during", "decision_k": decision_k, "text": f"How far ahead is it clear ({n})?",
           "template": "path_ahead", "slots": {"cap": 2}, "answer_type": "number", "needs": list(needs)}
    return {**row, **over}


def end_q(n: int, *, needs: tuple[str, ...] = ("see", "pose"), **over) -> dict:
    """A hand-written end question."""
    row = {"qid": qid(n), "when": "end", "decision_k": None, "text": f"Where was the thing last seen ({n})?",
           "template": "object_where", "slots": {"kind": "mug", "attributes": {"color": "red"}},
           "answer_type": "position", "needs": list(needs)}
    return {**row, **over}


QUESTIONS = [during_q(1, 12), end_q(2), end_q(3)]   # what `hand20` is asked: one during question, two end ones


def gold_row(row: dict, **over) -> dict:
    """The gold row of a question built by `during_q` or `end_q`: its answer a `number` or a `position`, answerable."""
    gold = ({"type": "number", "value": 1.4} if row["answer_type"] == "number"
            else {"type": "position", "xy": [3.2, 1.5]})
    tolerance = [0.25, 0.5, 1, 2] if row["answer_type"] == "number" else [0.5, 1]
    out = {"qid": row["qid"], "family": row["template"], "gold": gold, "slice": "answerable", "answer_class": None,
           "range_m": None, "credit": "ladder", "region": None, "tolerances": {"ladder": tolerance},
           "metric": None if row["when"] == "during" else "xy",
           "scope": "now" if row["when"] == "during" else "memory", "reference": None, "critical": False,
           "subjects": ["w:0"], "evidence_k": [0], "entities": []}
    return {**out, **over}


def hand20() -> Synth:
    """One task of 20 keyframes in the kitchen, decision points at k 3 and 12, a red mug seen all along and a blue book
    seen from k 5. Its look windows are [0, 4), [4, 12), [12, 13) and [13, 20): a cut after each decision point and
    after the task's last step. It is delivered one frame at a time: 20 observes."""
    s = Synth(tasks=[[(K, 20)]], decisions=[(3, "before_doorway"), (12, "before_doorway")])
    s.item("E01", "mug", {"color": "red"}, [(0, None, (2.0, 1.0, 0.9))])
    s.item("E02", "book", {"color": "blue"}, [(0, None, (2.5, 1.0, 0.9))])
    s.see(range(20), "E01")
    s.see(range(5, 20), "E02")
    return s


def write_pictures(paths: case_mod.ScriptPaths, n_frames: int) -> None:
    """A tiny JPEG per keyframe whose red level is its number, so a test can tell which frame a file holds."""
    for k in range(n_frames):
        Image.new("RGB", (8, 8), (10 * k % 256, 0, 0)).save(paths.rgb(k), quality=95)


def case_on(dataset: Path, synth: Synth, questions: list[dict], *, a: int = 0, b: int | None = None,
            length: str = "short") -> case_mod.CasePaths:
    """`synth`'s script written under `dataset` with its pictures, and the case over its tasks a..b, holding `questions`
    (hand-written rows) as its `questions.jsonl`."""
    case = synth.write_case(dataset, a, b, length)
    write_pictures(case.script, synth.n_frames)
    ep.write_jsonl(case.questions, questions)
    return case


def add_case(case: case_mod.CasePaths, questions: list[dict], *, a: int = 0, b: int | None = None,
             length: str = "short") -> case_mod.CasePaths:
    """Another case over `case`'s script (a test that patches `case.claim_id` gives it another id), with `questions`."""
    other = case_mod.write_case(case.dataset, case.script.id, a, a if b is None else b, length)
    ep.write_jsonl(other.questions, questions)
    return other


def clock(case: case_mod.CasePaths, seconds: float) -> str:
    """The case's clock `seconds` into the window: its start clock plus them, as the steps write it, to the tenth of a
    second with its one decimal always (`YYYY-MM-DDTHH:MM:SS.s`), written out here and not by `case.format_clock`."""
    at = datetime.fromisoformat(case_mod.read_case(case).start_clock) + timedelta(seconds=seconds)
    return at.strftime("%Y-%m-%dT%H:%M:%S") + f".{at.microsecond // 100_000}"


def fake_job(**over) -> JobSpec:
    """A job of two task kinds that are a swap pair, with hand-written assignment texts: what the brief's tests compose
    from, so they pin the brief and not any real job's words."""
    nothing = lambda *a, **k: None  # noqa: E731  (a job's callables, which a brief never calls)
    spec = JobSpec(id="fake/1", robot="vacuum", domain="home", task_kinds=("tidy", "check"), decisions={}, events={},
                   plan=nothing, script=nothing, relevance=nothing, checks=nothing, anchors=frozenset(),
                   assignments={"tidy": "Its current assignment is to tidy the {room}.",
                                "check": "Its current assignment is to check the {room}."},
                   swaps=(("tidy", "check"),))
    return replace(spec, **over)


def names_under(folder: Path) -> list[str]:
    """The files and folders under `folder`, as sorted POSIX paths from it."""
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*"))


class Recorder(System):
    """A system that appends `(op, arguments..., what the folders held)` to `opts["log"]` for every message and answers
    each question `unknown`. What the folders held is `{"inbox": the inbox's file names, "mtimes": their modification
    times, "all": every name under the run folder}` at that moment, so a test can check what was readable when.
    Arguments are deep copies. `privileged` and `structured` are class flags, as the harness reads them."""

    def __init__(self, state_dir, workdir, **opts):
        super().__init__(state_dir, workdir, **opts)
        self.log: list = opts["log"]
        self.root = self.workdir.parent
        self.closed = False

    def _note(self, op: str, *args) -> None:
        files = [p for p in (self.workdir / "inbox").rglob("*") if p.is_file()]
        held = {"inbox": sorted(p.relative_to(self.workdir).as_posix() for p in files),
                "mtimes": {p.relative_to(self.workdir).as_posix(): p.stat().st_mtime for p in files},
                "all": names_under(self.root)}
        self.log.append((op, *copy.deepcopy(args), held))

    def start(self, session, robot, channel):
        super().start(session, robot, channel)
        self._note("start", session, robot, channel)

    def task(self, index, clock, gap_s, brief):
        self._note("task", index, clock, gap_s, brief)

    def observe(self, steps):
        self._note("observe", steps)

    def ask(self, when, now, questions):
        self._note("ask", when, now, questions)
        return {q["qid"]: dict(UNKNOWN) for q in questions}

    def finish(self):
        self._note("finish")
        return {}

    def close(self):
        self.closed = True


def play(case: case_mod.CasePaths, system: type[Recorder] = Recorder, **kw) -> tuple[CaseRun, list]:
    """`run_case` of `case` on `system` (a `Recorder` or a subclass), with `kw` for it: the `CaseRun` and the system's
    log of every message."""
    log: list = []
    return run_case(lambda state, work: system(state, work, log=log), case, **kw), log


AGENT_CALLS: list[dict] = []   # what `fake_agent` was handed, one entry a call


def fake_agent(model: str, case: case_mod.CasePaths, args) -> CaseRun:
    """A runner as `cmd_run.AGENTS` names one (`"stream_kit.fake_agent"`): it notes what `stbench run` handed it, runs
    the case on a `Recorder` as the Codex driver runs its system, and raises what its model says: `refuse` the
    harness's `Refused`, `crash` a plain `RuntimeError`."""
    from stbench.cmd_run import Refused

    AGENT_CALLS.append({"model": model, "case": case.id, "reasoning": args.reasoning, "channel": args.channel,
                        "brief": args.brief, "during": args.during, "timeout": args.timeout, "out": args.out,
                        "out_exists": args.out.is_dir()})
    if model == "refuse":
        raise Refused("no such login")
    if model == "crash":
        raise RuntimeError("the agent fell over")
    return run_case(lambda state, work: Recorder(state, work, log=[]), case, channel=args.channel, brief=args.brief,
                    during=args.during == "on")


def ops(log: list) -> list[str]:
    return [entry[0] for entry in log]


def observed(log: list) -> list[list[int]]:
    """The case steps of each observe, in order."""
    return [[s["k"] for s in entry[1]] for entry in log if entry[0] == "observe"]


def messages(log: list) -> str:
    """What a system was sent, as one JSON text: every entry but the folders it held and the session (a fresh token
    each run)."""
    return json.dumps([[op, *(args[1:] if op == "start" else args)] for op, *args, _held in log])


# A scripted system in standard-library source, for the adapters: it writes 1,000 bytes in its first observe, counts the
# steps it has seen and answers every question `unknown` with what it knew when it was asked.
SCRIPTED = '''
from pathlib import Path


class Scripted:
    def __init__(self, state_dir, workdir, **opts):
        self.state, self.work, self.seen = Path(state_dir), Path(workdir), []

    def start(self, session, robot, channel):
        self.meta = [len(session), robot["name"], channel]

    def observe(self, steps):
        if not self.seen:
            (self.state / "memory.bin").write_bytes(b"x" * 1000)
        self.seen += [s["k"] for s in steps]

    def ask(self, when, now, questions):
        return {q["qid"]: {"type": "unknown", "when": when, "seen": len(self.seen), "meta": self.meta}
                for q in questions}

    def finish(self):
        return {"models": {"scripted": {"tokens_in": 5, "tokens_cached_in": 0, "tokens_cache_write_in": 0,
                                        "tokens_out": 1}}}
'''


def scripted_case(tmp_path: Path) -> tuple[case_mod.CasePaths, Path]:
    """`hand20` with its questions under `tmp_path/data`, and the `Scripted` system's file in a folder of its own."""
    case = case_on(tmp_path / "data", hand20(), QUESTIONS)
    (tmp_path / "agent").mkdir()
    file = tmp_path / "agent" / "scripted.py"
    file.write_text(SCRIPTED, encoding="utf-8")
    return case, file


def check_scripted_run(run: CaseRun) -> None:
    """What `Scripted` on `scripted_case` must give, whatever carried the messages: the case complete, each question
    answered when it was asked (the during one after 13 steps, the end ones after 20), its usage and its sizes."""
    assert run.status == "complete" and run.failures == {} and len(run.session) == 16
    # the session's length, the robot's name and the channel `start` carried (the default one)
    meta = [16, "robot vacuum", "rgb_pose"]
    assert {q: (a["when"], a["seen"], a["meta"]) for q, a in run.answers.items()} == {
        qid(1): ("during", 13, meta), qid(2): ("end", 20, meta), qid(3): ("end", 20, meta)}
    assert run.usage["models"] == {"scripted": {"tokens_in": 5, "tokens_cached_in": 0, "tokens_cache_write_in": 0,
                                                "tokens_out": 1}} and run.usage["self_reported"] is True
    assert set(run.wall_s) == {"start", "task", "observe", "ask_during", "ask_end", "finish"}
    assert run.state_bytes["finish"] == 1000
