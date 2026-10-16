"""The reader of JSON-lines files, and the runner.

The runner (below the table) runs scripted baselines over cases through `stream.run_case` and scores a run. It knows
nothing of priors: a caller hands `run_suite` an `OptionsFn`, called in this process to build each run's `RunOptions`
(constructor options, and for `other_case` the stream it is fed and where its during questions are asked), and the
`waves` to run in order. A wave's options are built only after every earlier wave has finished, from those waves' runs,
so a prior keyed on answers finds them. With `workers` above 1 the runs of a wave go to a spawn pool; the result lists
follow the order of `cases`, so a pooled run equals a serial one. This module may read gold (`score_rows`); the
baselines come from `baselines.BASELINES`, read inside the functions, so the registry is not needed to import it.
"""

from __future__ import annotations

import json
import multiprocessing
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from . import case as case_mod
from . import score
from .case import CasePaths
from .stream import CaseRun, run_case
from .stream_protocol import failed


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# -- the runner ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RunOptions:
    """What one (system, case) run is given. `opts` are the constructor options (priors, the public map, the job's
    event table); `stream_from` and `ask_map` make it `other_case`: the system is fed that case's stream, and this
    case's during questions are asked at `ask_map[decision keyframe]`, a keyframe of that stream's script. It is
    pickled into spawned workers, so everything in it must pickle."""

    opts: dict
    stream_from: CasePaths | None = None
    ask_map: dict[int, int] | None = None


# (system, case, every case of the run, the earlier waves' runs by (system, case id)) -> its options; called in the
# parent process, once a run, before its wave starts
OptionsFn = Callable[[str, CasePaths, list[CasePaths], Mapping[tuple[str, str], CaseRun]], RunOptions]


def _record(name: str):
    """The registry entry of a baseline (or of a learned one's in-pool twin: `baselines.lookup`), read here so this
    module imports without the baselines."""
    from .baselines import BASELINES, lookup

    if (found := lookup(name)) is None:
        raise ValueError(f"no baseline named {name!r}; the baselines are {', '.join(sorted(BASELINES))}")
    return found


def _run(cls: type, case: CasePaths, channel: str, ro: RunOptions, brief: str = "correct", during: bool = True
         ) -> CaseRun:
    """One run of the system class `cls`, built with `ro.opts`. The pool is sent the class itself, which pickles by
    reference, so a worker never depends on what this process put in the registry."""
    return run_case(lambda state, work: cls(state, work, **ro.opts), case, channel=channel, brief=brief,
                    during=during, stream_from=ro.stream_from, ask_map=ro.ask_map)


def run_baseline(name: str, case: CasePaths, channel: str, ro: RunOptions, *, brief: str = "correct",
                 during: bool = True) -> CaseRun:
    """The baseline `name` over `case` with `ro` (`stream.run_case` takes `brief` and `during`). ValueError for a name
    that is none, and whatever `run_case` refuses."""
    return _run(_record(name).cls, case, channel, ro, brief, during)


# (system, class, case, options, whether it asks the during questions)
Job = tuple[str, type, CasePaths, RunOptions, bool]


def _run_bundle(jobs: list[Job], channel: str, brief: str) -> list[CaseRun]:
    """The runs of `jobs`, one after the other, in this process."""
    return [_run(cls, case, channel, ro, brief, during) for _, cls, case, ro, during in jobs]


def _steps(case: CasePaths) -> int:
    """How many steps the case's window holds, 0 when its record cannot be read (the run says why)."""
    try:
        rec = case_mod.read_case(case)
    except (OSError, ValueError, KeyError):
        return 0
    return rec.k1 - rec.k0


def _bundles(jobs: list[Job]) -> list[list[int]]:
    """The indices of `jobs` grouped by the case whose files each run reads (the case it is fed the stream of, else its
    own), the groups with the most work first. The runs of a group are one pool task: they read the same images, and
    what a run works out of one (`table.SEEN`) the next finds in the process, where as separate tasks they would meet
    in different processes."""
    found: dict[str, list[int]] = {}
    for i, (_, _, case, ro, _) in enumerate(jobs):
        found.setdefault((ro.stream_from or case).id, []).append(i)
    work = {key: len(idx) * _steps(jobs[idx[0]][3].stream_from or jobs[idx[0]][2]) for key, idx in found.items()}
    return [found[key] for key in sorted(found, key=lambda key: -work[key])]


def _run_wave(jobs: list[Job], channel: str, pool: ProcessPoolExecutor | None, log: Callable[[str], None],
              brief: str) -> dict[tuple[str, str], CaseRun]:
    """Every job (system, class, case, options, whether it asks the during questions) of one wave, serially or in
    `pool` (the jobs on one case's files as one task: `_bundles`), logged as each finishes: their runs by (system, case
    id)."""
    runs: dict[tuple[str, str], CaseRun] = {}
    if pool is None:
        started = ((name, case, _run(cls, case, channel, ro, brief, during)) for name, cls, case, ro, during in jobs)
    else:
        futures = {pool.submit(_run_bundle, [jobs[i] for i in idx], channel, brief): idx for idx in _bundles(jobs)}
        started = ((jobs[i][0], jobs[i][2], run) for future in as_completed(futures)
                   for i, run in zip(futures[future], future.result(), strict=True))
    for name, case, run in started:
        runs[name, case.id] = run
        log(f"{name} {case.id}: {run.status}")
    return runs


def run_suite(cases: list[CasePaths], names: list[str], channel: str, *,
              options: OptionsFn | None = None, waves: list[list[str]] | None = None,
              log: Callable[[str], None] = lambda msg: None, workers: int = 1, brief: str = "correct",
              during: bool = True) -> dict[str, list[CaseRun]]:
    """Every system of `waves` (default: `[names]`) over every case: `{system: [CaseRun per case, in `cases` order]}`,
    systems in wave order. The waves run in order; a wave's `RunOptions` are built here, by `options` (none: `RunOptions
    ({})`), only after every earlier wave has finished, from `cases` and those waves' runs (never a later wave's, nor
    its own). A wave may name a system `names` does not (a helper whose runs feed the next wave's options). With
    `workers` above 1 the runs of a wave go to that many spawned processes; a failed run cancels the queued ones, waits
    for the running ones and raises its error here. Every run is given `brief` (the variant `run_baseline` takes) and
    every run of a system of `names` `during` (whether the during questions are asked); a helper system's runs always
    ask them, because the records that carry its answers need one for every question of the other cases. ValueError,
    before anything runs, for a system that is none, one named twice among the waves, one of `names` in no wave, or
    fewer than one worker; `log` gets one line a run."""
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    cases = list(cases)
    waves = [list(names)] if waves is None else [list(wave) for wave in waves]
    ran = [name for wave in waves for name in wave]
    if len(set(ran)) < len(ran) or not set(names) <= set(ran):
        raise ValueError(f"the waves {waves} must run every system of {list(names)} exactly once")
    records = {name: _record(name) for name in ran}
    done: dict[tuple[str, str], CaseRun] = {}
    pool = (ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
            if workers > 1 else None)
    try:
        for wave in waves:
            earlier = MappingProxyType(dict(done))
            jobs = [(name, records[name].cls, case, options(name, case, cases, earlier) if options else RunOptions({}),
                     during or name not in names) for name in wave for case in cases]
            done.update(_run_wave(jobs, channel, pool, log, brief))
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)  # after a failed run the queued ones never start
    return {name: [done[name, case.id] for case in cases] for name in ran}


# -- cases and their scores ----------------------------------------------------------------------------------------
def case_info(case: CasePaths) -> dict:
    """What a result row says of its case (`score_stats.INFO_KEYS`), from the case's and its script's index entries: the
    case and script ids, the layout (the script's map), robot and job, and the case's length. ValueError for an entry
    the index does not hold."""
    entry, script = (case_mod.read_entry(case.dataset, iid) for iid in (case.id, case.script.id))
    if entry is None or script is None:
        raise ValueError(f"{case.dataset} has no index entry for case {case.id} or for its script {case.script.id}")
    return {"case": case.id, "script": case.script.id, "layout": script["map"], "robot": script["robot"],
            "job": script["job"], "length": entry["length"]}


def cases_like(case: CasePaths) -> list[CasePaths]:
    """The compiled cases (questions and gold on disk) of `case`'s dataset on its script's layout and job, `case`'s own
    among them, in id order: what a baseline's options are built from."""
    mine = case_info(case)
    found = (CasePaths.from_id(case.dataset, iid) for iid, entry in case_mod.read_index(case.dataset).items()
             if entry.get("kind") == "case")
    return [c for c in found if c.questions.is_file() and c.gold.is_file()
            and all(case_info(c)[key] == mine[key] for key in ("layout", "job"))]


def score_rows(case: CasePaths, run: CaseRun, *, during: bool = True) -> list[dict]:
    """One result row (`score.row`) per gold of the case, in gold order. A gold the run has no answer for scores as
    `failed("unanswered")`: malformed, so a wrong answer on an answerable gold and a missed `unknown`, and never a gold
    without a row (the gates need every one). With `during` false (the run did not ask them) the golds of the during
    questions are left out. ValueError for a gold whose question the case does not hold."""
    info = case_info(case)
    when = {q["qid"]: q["when"] for q in read_jsonl(case.questions)}
    rows = []
    for gold in read_jsonl(case.gold):
        if gold["qid"] not in when:
            raise ValueError(f"{case.gold} has a gold for {gold['qid']}, which {case.questions.name} does not ask")
        if during or when[gold["qid"]] != "during":
            rows.append(score.row(run.answers.get(gold["qid"], failed("unanswered")), gold, when=when[gold["qid"]],
                                  info=info))
    return rows
