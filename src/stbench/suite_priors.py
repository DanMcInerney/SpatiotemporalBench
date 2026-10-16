"""What each baseline may have beyond the stream: the priors and the options.

`options_for` is `suite.OptionsFn`: from the system's `baselines.Baseline` flags and nothing else it builds the system's
`RunOptions`. A `needs_prior` system gets other scripts' records (`prior_records`, in a `blind.Prior` that says whether
it keeps its pool's own best response: `Baseline.in_pool`), a `needs_map` one the public map of
the case's layout and with it the job's zone rules (`zones`: decision kind or event wording -> `ZoneRule`, and
`slot_zones`; neutral, they pickle: a memoryless baseline that held no rule would answer every zone question `unknown`
and make the memory-needed gate vacuous) and its `task_things` predicate (`task_predicate.TaskThings`, None for a job
that asks none), a `needs_events` one its job's event table, and an `other_stream` one another
script's case as its stream (`other_case_for`) with the map of where its during questions are asked (`ask_map`) and
the maps that put a clock or a task a question names onto that stream (`clock_map`, `task_map`: the same moment of the
task, as the plan's §4.8 says: "clocks in slots translate by task-relative offset") and its time answers, clocks of
that stream, back onto the asking case's (`clock_back`). It never runs a system: the
`nm` of a record that carries `baselines.NM_SYSTEM`'s answers comes from `done`, the runs of an earlier wave, and a run
that is missing raises (`stbench check` runs that wave first).

A record (`answer_keys.RECORD_KEYS`) is what `record_for` builds of one asked question, from what a system on the gate
channel is sent of it: its template, slots and `when`; the clock, phase, base, heading and subgoal of the step the ask
is about (`ask_step`: the decision's step of a during question, the window's last of an end one, read from the script's
pose rows, the clock as the stream delivers it); the rooms the briefs so far named; `nm`; the case's other end
questions; the anchor of a `task_things` question's task (`task_predicate.anchor_of` over the task's pose rows as
delivered, None for any other question); the camera pose of every step of the window up to the ask's (`cameras`, as the
stream delivers them: what `path_log` faced, which its ego rule is judged on); `own`, `path_log`'s own-log answer to a
`clear_path` or `open_heading` question (`baselines_path_own.own_answer`; None for any other), which its rule for that
family is judged on; the gold.
`answer_keys.levels` keys it exactly as it keys what a system builds at the ask, so the pool a prior system finds is the
pool written here.

Records come from other scripts only (map X-H14): two cases of one script are windows of one world and one path, so a
prior that took one for the other would hold its own answer. Truth side: this module reads gold. A case's script facts
are read once a process (`_window`) and its records once per version of its questions and gold (`_records`).

Two rules for a held-out layout (VISION rule 5), which an agent has never seen:
- a system that models an agent (`Baseline.models_agent`) gets the layout's region kinds (`kinds`, for the rooms its
  briefs name) and never its map; the instruments keep the map on every layout, a perfect perceiver's knowledge;
- a family whose reference is `other_case` (a fact of the layout, the same in every case of it) is learned from public
  layouts only: `prior_records` keeps no record of it from a held-out layout, so the priors that read it abstain, and
  the form adversary pools the same way (`form_check`). Every other family keeps the pool of its own layout and job,
  which on a held-out layout is stronger than an honest agent's prior, so the gate it feeds is the stricter.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from . import (
    answer_keys,
    baselines,
    baselines_path_own,
    gates,
    job_registry,
    map_files,
    mapspec,
    questions_text,
    robots,
    suite,
    task_predicate,
)
from .blind import Prior
from .brief import compose
from .case import CasePaths, CaseRecord, case_clock, case_task, format_clock, read_case
from .seeding import stable_hash
from .stream import CaseRun
from .stream_protocol import GATE_CHANNEL
from .suite import RunOptions
from .timeline import load
from .timeline_ego import TaskPlan


@dataclass(frozen=True)
class _Window:
    """What a case's records read of its script: the window, its pose rows (script numbering), where each step lies in
    its task, the decision points, the plans, the briefs a system on the gate channel is sent, the layout's rooms and
    the clocks the stream delivers."""

    rec: CaseRecord
    rows: dict[int, dict]                     # the window's pose rows by script k
    tasks: dict[int, tuple[int, int]]         # script k -> (its task in the case's numbering, steps since it began)
    points: tuple[tuple[int, str], ...]       # (k, kind) of the window's decision points, in order
    plans: tuple[TaskPlan, ...]               # the window's tasks'
    briefs: tuple[str, ...]                   # one a task, as the stream sends them on `GATE_CHANNEL`
    kinds: tuple[str, ...]                    # the layout's region kinds
    clocks: tuple[tuple[str, ...], ...]       # per task (case numbering), its steps' delivered clocks in step order
    at: dict[int, str]                        # script k -> the clock the stream delivers at that step
    model: robots.Robot                       # the robot as its `start` message gives it


@lru_cache(maxsize=1024)
def _read_info(dataset: str, cid: str) -> dict:
    return suite.case_info(CasePaths.from_id(dataset, cid))


def _info(case: CasePaths) -> dict:
    """`suite.case_info` (case, script, layout, robot, job, length), read once a process."""
    return _read_info(str(case.dataset), case.id)


@lru_cache(maxsize=1024)
def _read_window(dataset: str, cid: str) -> _Window:
    case = CasePaths.from_id(dataset, cid)
    info, rec, tl = _read_info(dataset, cid), read_case(case), load(case.script.timeline)
    job = job_registry.get(info["job"])
    robot = json.loads(case.script.robot.read_text(encoding="utf-8"))
    plans = tuple(tl.ego.tasks[i].plan for i in range(rec.a, rec.b + 1))
    rows = {r["k"]: r for r in suite.read_jsonl(case.script.pose) if rec.k0 <= r["k"] < rec.k1}
    tasks = {k: (case_task(rec, tl.task_of(k).index), k - tl.task_of(k).k0) for k in rows}
    at = {k: case_clock(rec, tl.keyframe(k).t) for k in sorted(rows)}
    clocks: dict[int, list[str]] = {i: [] for i in range(len(plans))}
    for k, clock in at.items():
        clocks[tasks[k][0]].append(clock)
    decisions = tl.decisions_in(rec.k0, rec.k1)
    return _Window(rec, rows, tasks, tuple((d.k, d.kind) for d in decisions), plans,
                   tuple(compose(list(plans), job=job, robot=robot, domain=job.domain, channel=GATE_CHANNEL)),
                   tuple(r.kind for r in mapspec.load_map(info["layout"]).regions),
                   tuple(tuple(c) for _, c in sorted(clocks.items())), at, robots.from_doc(robot))


def _window(case: CasePaths) -> _Window:
    return _read_window(str(case.dataset), case.id)


class AskStep(NamedTuple):
    row: dict        # the step's pose row as delivered: `pose`, `camera`, `status`; its `clock` is the script's own
    task: int        # its task, in the case's numbering
    k_in_task: int   # steps since its task began
    clock: str       # the clock the stream delivers at it (`now["clock"]` of the ask): the case's, not `row["clock"]`
    k: int           # its keyframe, in the script's numbering


def ask_step(case: CasePaths, question: dict) -> AskStep:
    """The step an ask of `question` is about: a during question's decision keyframe, an end question's last step of the
    window. ValueError when the case's script holds no such step."""
    w = _window(case)
    k = question["decision_k"] if question["when"] == "during" else w.rec.k1 - 1
    if k not in w.rows:
        raise ValueError(f"question {question['qid']} is asked at keyframe {k}, which is no step of case {case.id}")
    return AskStep(w.rows[k], *w.tasks[k], w.at[k], k)


def _anchor(case: CasePaths, question: dict) -> task_predicate.Anchor | None:
    """The anchor of the task a `task_things` question names, built as `path_log` builds it from its own log: the pose
    rows of the task as delivered, under the job's predicate and the layout's public map. None for any other question,
    and for a job that asks no predicate or a task that has no anchor."""
    info = _info(case)
    pred = job_registry.get(info["job"]).task_things
    if question["template"] != "task_things" or pred is None:
        return None
    w = _window(case)
    steps = [w.rows[k] for k in sorted(w.rows) if w.tasks[k][0] == question["slots"]["task"]]
    return task_predicate.anchor_of(pred, steps, mapspec.load_map(info["layout"]))


def _own(case: CasePaths, question: dict, step: AskStep) -> dict | None:
    """`path_log`'s own-log answer to a question of a template with an own-log rule (`baselines_path_own.rules`), from
    the pose rows of every step of the window up to the ask's as delivered, the base and heading of the ask's step, as
    it computes it at the ask (`baselines_path_own.own_answer`, the function its answer is). None for any other."""
    template = question["template"]
    if template not in baselines_path_own.rules():
        return None
    w = _window(case)
    return baselines_path_own.own_answer(w.model, template, question["slots"],
                                         [r for k, r in sorted(w.rows.items()) if k <= step.k],
                                         tuple(step.row["pose"]["pos"][:2]), step.row["pose"]["yaw_deg"])


def _record(case: CasePaths, question: dict, gold: dict, nm: dict | None, siblings: list[dict]) -> dict:
    w, step = _window(case), ask_step(case, question)
    during = question["when"] == "during"
    briefs = w.briefs[:step.task + 1] if during else w.briefs   # a during ask has been sent the tasks begun so far
    return {"script": case.script.id, "template": question["template"], "slots": question["slots"],
            "when": question["when"], "clock": step.clock, "phase": step.row["status"]["phase"],
            "base": tuple(step.row["pose"]["pos"][:2]), "yaw": step.row["pose"]["yaw_deg"],
            "subgoal": step.row.get("subgoal"), "rooms": answer_keys.rooms_named(list(briefs), w.kinds), "nm": nm,
            "siblings": [] if during else siblings, "anchor": _anchor(case, question),
            "cameras": tuple(w.rows[k]["camera"] for k in sorted(w.rows) if k <= step.k),
            "own": _own(case, question, step), "gold": gold}


def _ends(questions: list[dict]) -> list[dict]:
    """The `{template, slots}` of each end question."""
    return [{"template": q["template"], "slots": q["slots"]} for q in questions if q["when"] == "end"]


def record_for(case: CasePaths, question: dict, gold: dict, nm: dict | None) -> dict:
    """The record (`answer_keys.RECORD_KEYS`) of one question of `case` with its gold row and `nm` (`NM_SYSTEM`'s answer
    to it; None for a system that does not key on it): the keys the systems build at the ask, so `levels` of either
    agree. ValueError for a question the case's script gives no ask step."""
    others = _ends([q for q in suite.read_jsonl(case.questions) if q["qid"] != question["qid"]])
    return _record(case, question, gold, nm, others)


def _stamp(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=1024)
def _records(dataset: str, cid: str, stamp: tuple) -> tuple[dict, ...]:
    """The case's records with no `nm`; `stamp` (the questions' and the gold's file stamps) keeps a recompile fresh."""
    case = CasePaths.from_id(dataset, cid)
    golds = {g["qid"]: g for g in suite.read_jsonl(case.gold)}
    questions = suite.read_jsonl(case.questions)
    return tuple(_record(case, q, golds[q["qid"]], None, _ends([o for o in questions if o is not q]))
                 for q in questions)


def nm_answers(case: CasePaths, done: Mapping[tuple[str, str], CaseRun]) -> dict[str, dict]:
    """`NM_SYSTEM`'s answers to `case`'s questions by qid, from the earlier wave's runs in `done`. ValueError when it
    never ran the case: the records that carry its answers need the first wave."""
    run = done.get((baselines.NM_SYSTEM, case.id))
    if run is None:
        raise ValueError(f"no {baselines.NM_SYSTEM} run of case {case.id}: the records of a system that keys on its "
                         f"answers (`prior_nm`) are built from a first wave that runs it, as `stbench check` does")
    return run.answers


def case_records(case: CasePaths, nm: Mapping[str, dict] | None = None) -> list[dict]:
    """One record per question of `case`, in question order. `nm` (qid -> answer, as `nm_answers` gives) fills each
    record's `nm`; None leaves it None. ValueError for a question with no answer in `nm`, or whose answer is a failure:
    a record keyed on an answer the system never gave would be a prior of nothing."""
    base = _records(str(case.dataset), case.id, (_stamp(case.questions), _stamp(case.gold)))
    if nm is None:
        return [dict(r) for r in base]
    out = []
    for r in base:
        qid = r["gold"]["qid"]
        if qid not in nm or nm[qid].get("type") == "failed":
            raise ValueError(f"{baselines.NM_SYSTEM} gave no answer to question {qid} of case {case.id}: "
                             f"{nm.get(qid, 'missing')}")
        out.append({**r, "nm": nm[qid]})
    return out


def others(case: CasePaths, cases: list[CasePaths]) -> list[CasePaths]:
    """The cases among `cases` of other scripts than `case`'s, on its layout and job, in the order given."""
    mine = _info(case)
    return [c for c in cases if c.script.id != case.script.id
            and all(_info(c)[key] == mine[key] for key in ("layout", "job"))]


def prior_records(case: CasePaths, cases: list[CasePaths], done: Mapping[tuple[str, str], CaseRun], *,
                  nm: bool) -> list[dict]:
    """The records of every question of the cases of other scripts (`others`), in `cases` order. With `nm` each carries
    `NM_SYSTEM`'s answer from `done` (see `nm_answers`, `case_records`); without, None. A record of a family whose
    reference is `other_case` is kept only from a script on a public layout (the module docstring); in a root of
    held-out layouts that pool is empty."""
    out: list[dict] = []
    for other in others(case, cases):
        records = case_records(other, nm_answers(other, done) if nm else None)
        if _info(other)["layout"] in map_files.HELD_OUT:
            records = [r for r in records if r["gold"]["reference"] != gates.OTHER]
        out += records
    return out


def _tasks(case: CasePaths) -> int:
    """How many tasks the case's window holds."""
    w = _window(case)
    return w.rec.b - w.rec.a + 1


def other_case_for(case: CasePaths, cases: list[CasePaths]) -> CasePaths:
    """The case `other_case` is fed: of another script, on `case`'s layout, job and robot, of its length (`short`,
    `medium` or `long`), holding as many tasks when some case does, else the nearest count (the fewer on a tie); among
    those, preferring the same task plans (the same room), then the first by `stable_hash(case id, other id)`. A task or
    a clock the case's questions name past the partner's last task is put on it by `task_map` and `clock_map`.
    ValueError naming what is missing when no case of another script has the length: a window's size within its length
    is no reason to refuse a dataset (a long window of 5 tasks beside others of 4 and 6)."""
    w, info = _window(case), _info(case)
    same = [c for c in others(case, cases) if _info(c)["robot"] == info["robot"]
            and _info(c)["length"] == info["length"]]
    if not same:
        raise ValueError(f"no case of another script on layout {info['layout']}, job {info['job']} and robot "
                         f"{info['robot']} of length {info['length']} among the {len(cases)} cases, to feed case "
                         f"{case.id} to `other_case`")
    count = min((abs(_tasks(c) - _tasks(case)), _tasks(c)) for c in same)[1]
    nearest = [c for c in same if _tasks(c) == count]
    pool = [c for c in nearest if _window(c).plans == w.plans] or nearest
    return min(pool, key=lambda c: (stable_hash(case.id, c.id), c.id))


def task_map(own: CasePaths, other: CasePaths) -> dict[int, int]:
    """Where `other_case`, fed `other`'s stream, is asked about the tasks `own`'s questions name: each of `own`'s case
    task indices to itself, or to `other`'s last task when `other` holds fewer. Case numbering on both sides, as the
    `task` messages carry them."""
    last = _tasks(other) - 1
    return {i: min(i, last) for i in range(_tasks(own))}


def clock_map(own: CasePaths, other: CasePaths) -> dict[str, str]:
    """Where `other_case` is asked the clocks `own`'s questions name: for each `clock` slot of `own`'s questions, the
    task i that holds the clock (the last whose first delivered clock is not after it, the first task for an earlier
    clock) and the seconds since task i's first delivered clock; `other`'s task `task_map[i]`'s first delivered clock
    plus those seconds; and, of that task's delivered clocks, the one nearest (the earlier on a tie). The same moment of
    the task, a clock `other`'s stream delivered. ValueError for a clock that is not an ISO time."""
    mine, theirs, to = _window(own), _window(other), task_map(own, other)
    firsts = [datetime.fromisoformat(c[0]) for c in mine.clocks]
    out: dict[str, str] = {}
    for q in suite.read_jsonl(own.questions):
        clock = q["slots"].get("clock")
        if clock is None or clock in out:
            continue
        at = datetime.fromisoformat(clock)
        i = max((n for n, first in enumerate(firsts) if first <= at), default=0)
        held = theirs.clocks[to[i]]
        want = datetime.fromisoformat(held[0]) + (at - firsts[i])
        out[clock] = min(held, key=lambda c: (abs(datetime.fromisoformat(c) - want), datetime.fromisoformat(c)))
    return out


def clock_back(own: CasePaths, other: CasePaths) -> dict[str, str]:
    """`clock_map`'s inverse for the clocks `other_case` answers in: for each clock `other`'s stream delivers, task j's
    c, the clock `own`'s stream holds the same number of seconds after the first of its task j (the last task `own`
    holds, when `other` holds more): the same moment of the task, on the asking case's own day. The answer of a table
    fed `other`'s stream is a clock of that stream, which scored against `own`'s gold, a clock of another day, could
    only lose."""
    mine, theirs = _window(own), _window(other)
    firsts = [datetime.fromisoformat(c[0]) for c in mine.clocks]
    out: dict[str, str] = {}
    for j, held in enumerate(theirs.clocks):
        start, home = datetime.fromisoformat(held[0]), firsts[min(j, len(firsts) - 1)]
        out |= {c: format_clock(home + (datetime.fromisoformat(c) - start)) for c in held}
    return out


def _mapped(out: dict[int, int], k: int, target: int) -> None:
    """`out[k] = target`; ValueError when two questions asked at keyframe `k` map it to different keyframes of the other
    script (a decision point and a look point at one keyframe would, and `other_case` is asked at one place a
    keyframe)."""
    if out.get(k, target) != target:
        raise ValueError(f"the questions asked at keyframe {k} map it to keyframes {out[k]} and {target} of the other "
                         f"script; a keyframe maps to one")
    out[k] = target


def ask_map(own: CasePaths, other: CasePaths) -> dict[int, int]:
    """Where `other_case`, fed `other`'s stream, is asked `own`'s during questions: each asked decision keyframe of
    `own` to the keyframe of `other`'s script that is its j-th decision point of the same kind (j: its index among the
    window's points of that kind), else `other`'s last point of that kind, else the last step of `other`'s window; each
    asked look keyframe (a question of a look family, `questions_text.LOOKS`) to the frame of `other`'s stream in task
    `task_map[i]` nearest the same number of steps into it as `own`'s (the earlier on a tie), else the last step of
    `other`'s window. A keyframe of each script's own numbering, and every one is a frame of `other`'s stream (a look
    point is any frame), so every question is asked (map X-H15)."""
    mine, theirs, to = _window(own), _window(other), task_map(own, other)
    kinds = dict(mine.points)
    out = {}
    for q in suite.read_jsonl(own.questions):
        if q["when"] != "during":
            continue
        k = q["decision_k"]
        if q["template"] in questions_text.LOOKS:
            if k not in mine.tasks:
                raise ValueError(f"question {q['qid']} is asked at keyframe {k}, which is no frame of case {own.id}")
            i, steps = mine.tasks[k]
            near = [(abs(steps_in - steps), steps_in, e) for e, (task, steps_in) in theirs.tasks.items()
                    if task == to[i]]
            _mapped(out, k, min(near)[2] if near else theirs.rec.k1 - 1)
            continue
        if k not in kinds:
            raise ValueError(f"question {q['qid']} is asked at keyframe {k}, which is no decision point of case "
                             f"{own.id}")
        j = [p for p, kind in mine.points if kind == kinds[k]].index(k)
        same = [p for p, kind in theirs.points if kind == kinds[k]]
        _mapped(out, k, same[j] if j < len(same) else same[-1] if same else theirs.rec.k1 - 1)
    return out


def options_for(name: str, case: CasePaths, cases: list[CasePaths],
                done: Mapping[tuple[str, str], CaseRun]) -> RunOptions:
    """The options of the baseline `name` over `case`: exactly what its `Baseline` flags ask for (see the module).
    `cases` are the cases of the run, `done` the runs of the waves before this one."""
    if (b := baselines.lookup(name)) is None:
        raise ValueError(f"no baseline named {name!r}; the baselines are {', '.join(sorted(baselines.BASELINES))}")
    info = _info(case)
    opts: dict = {}
    if b.needs_prior:
        opts["prior"] = Prior(prior_records(case, cases, done, nm=b.prior_nm), in_pool=b.in_pool)
    if b.needs_map:
        job, layout = job_registry.get(info["job"]), mapspec.load_map(info["layout"])
        if b.models_agent and info["layout"] in map_files.HELD_OUT:
            opts["kinds"] = tuple(r.kind for r in layout.regions)
        else:
            opts["map"] = layout
        opts |= {"zones": dict(job.zones), "slot_zones": job.slot_zones, "task_things": job.task_things}
    if b.needs_events:
        opts["events"] = dict(job_registry.get(info["job"]).events)
    if not b.other_stream:
        return RunOptions(opts)
    other = other_case_for(case, cases)
    return RunOptions({**opts, "clock_map": clock_map(case, other), "task_map": task_map(case, other),
                       "clock_back": clock_back(case, other)}, other, ask_map(case, other))
