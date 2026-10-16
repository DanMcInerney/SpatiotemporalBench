"""Running a selection of cases: what is selected, what is finished, the arms, the budget and the pool (docs/eval.md).

The selection is the dataset's compiled cases that ask a question (`check.compiled_cases`), narrowed by `--cases` and by
`--select` on `suite.case_info`'s `layout`, `robot`, `job` and `length` (a key given twice is either value; different
keys must all hold). Refused (`cmd_run.Refused`, exit 2) before anything is written: none left, the retired `swapped`
brief.

Resume. A case is finished when its `summary.json` says `complete` and its rows were written (`eval_results.finished`);
the rows' qids must all be in the gold on disk, or a rerun is refused (`eval_results.check_resume`). Any other case's
folder is removed and the case run again from its start (a Codex call cannot be replayed). What was finished is read
from the folders, never remembered, so a run killed at any point resumes.

Arms (`--arms LEVER=V1,V2`): every case is run once per value, into a run folder of its own (`RUNDIR/<lever>-<value>`),
the arms of one case one after another in the order `sorted(values, key=stable_hash(case id, lever, value))`: no arm
always runs first. Everything else is equal by construction (the brief's during sentence is in every arm).

The budget. `--max-usd TOTAL` with `--case-usd X` (the reservation of a case in flight): a case starts only while what
the folders' summaries spent plus what the cases in flight reserve plus `X` stays within `TOTAL`. A case's spend is
`prices.cost_usd` of its usage, and `X` when that is None (so the cap holds for a case that cannot be priced). When a
case may not start, none starts until one in flight ends and frees its reservation; with none in flight and none
allowed, the run stops, partial.

The pool. A baseline runs the whole selection in one `suite.run_suite` (`--workers` processes). Anything else runs one
case a thread, `--workers` at a time (a `codex:` case is a Codex process each).
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

from . import case_select, check, prices
from . import eval_results as results
from . import eval_systems as systems
from .brief import VARIANTS
from .case import CasePaths
from .cmd_run import Refused
from .eval_systems import CASES, Arm, Config, Worker
from .seeding import stable_hash
from .stream import SWAPPED_RETIRED

LEVERS = {"during": ("on", "off"), "brief": tuple(v for v in VARIANTS if v != "swapped")}   # `swapped` is retired
EPS = 1e-9


# -- the selection -----------------------------------------------------------------------------------------------
def parse_select(items: list[str]) -> dict[str, set[str]]:
    """`--select KEY=VALUE` (repeated): the values wanted for each key (`case_select.parse`). Refused otherwise."""
    try:
        return case_select.parse(items)
    except case_select.Bad as error:
        raise Refused(str(error)) from None


def select(data: Path, ids: list[str] | None, picks: dict[str, set[str]]) -> list[CasePaths]:
    """The compiled cases of `data` that ask a question, in id order, narrowed by the ids and the picks
    (`case_select.narrow`). Refused when none is left."""
    asking, silent = check.compiled_cases(data)
    if not asking:
        none = f" ({len(silent)} compiled cases ask nothing)" if silent else ""
        raise Refused(f"no case with questions and gold in {data}{none}; compile some first (`stbench build`, or "
                      "`stbench compile`)")
    try:
        return case_select.narrow(asking, ids, picks, data)
    except case_select.Bad as error:
        raise Refused(str(error)) from None


def brief_problem(variant: str) -> str | None:
    """Why the brief `variant` is refused, else None: `swapped` is retired (a job's subgoal says the true task in every
    frame, so a swapped brief would contradict it: `stream.SWAPPED_RETIRED`), as an option and as an arm, before a case
    is run."""
    return SWAPPED_RETIRED if variant == "swapped" else None


# -- the arms ----------------------------------------------------------------------------------------------------
def parse_arms(text: str) -> tuple[str, list[str]]:
    """`--arms LEVER=V1,V2[,...]`: (the lever, its values). Refused for another lever, a value it lacks, a value twice
    or fewer than two values."""
    lever, equals, rest = text.partition("=")
    values = [v for v in rest.split(",") if v] if equals else []
    if lever not in LEVERS or len(values) < 2 or len(set(values)) < len(values):
        levers = ", ".join(f"{name} ({'|'.join(allowed)})" for name, allowed in LEVERS.items())
        raise Refused(f"--arms takes LEVER=V1,V2[,...] with two or more different values and LEVER one of {levers}, "
                      f"not {text!r}")
    if lever == "brief":
        for value in values:
            if problem := brief_problem(value):
                raise Refused(problem)
    if unknown := [v for v in values if v not in LEVERS[lever]]:
        raise Refused(f"--arms {lever} takes {', '.join(LEVERS[lever])}, not {', '.join(unknown)}")
    return lever, values


def make_arms(out: Path, lever: str | None, values: list[str], brief: str, during: bool) -> list[Arm]:
    """The run folders: RUNDIR itself without arms, else `RUNDIR/<lever>-<value>` for each value, the other options
    those of the command."""
    if lever is None:
        return [Arm(out, brief, during)]
    return [Arm(out / f"{lever}-{value}", value if lever == "brief" else brief,
                value == "on" if lever == "during" else during, value) for value in values]


def arm_order(cid: str, lever: str | None, arms: list[Arm]) -> list[Arm]:
    """The arms of one case in the order they run: seeded by the case, so no arm always runs first."""
    return sorted(arms, key=lambda arm: stable_hash(cid, lever, arm.value))


# -- the budget --------------------------------------------------------------------------------------------------
def spent(cfg: Config) -> float:
    """What the summaries of every run folder say was spent: each case's cost, or `--case-usd` where it has none."""
    total = 0.0
    for arm in cfg.arms:
        for path in sorted((arm.path / CASES).glob("*/summary.json")):
            try:
                usage = json.loads(path.read_text(encoding="utf-8")).get("usage")
            except (OSError, ValueError, AttributeError):
                usage = None
            cost = prices.cost_usd(usage) if isinstance(usage, dict | None) else None
            total += cfg.case_usd if cost is None else cost
    return total


class Budget:
    """The cap of a run (none without `--max-usd`): `admit` a case before it starts, `release` it after."""

    def __init__(self, cfg: Config):
        self.cap, self.each = cfg.max_usd, cfg.case_usd
        self.spent = spent(cfg) if self.cap is not None else 0.0
        self.reserved: dict[tuple[str, str], float] = {}

    def admit(self, key: tuple[str, str]) -> bool:
        if self.cap is not None and self.spent + sum(self.reserved.values()) + self.each > self.cap + EPS:
            return False
        self.reserved[key] = self.each or 0.0
        return True

    def release(self, key: tuple[str, str], cost: float | None) -> None:
        del self.reserved[key]
        if self.cap is not None:
            self.spent += self.each if cost is None else cost


# -- the pool ----------------------------------------------------------------------------------------------------
@dataclass
class _Chain:
    """The arms of one case still to run, in order: one runs after another."""

    case: CasePaths
    arms: deque[Arm] = field(default_factory=deque)


def _schedule(cfg: Config, work: Worker, run: results.Run, log: Callable[[str], None]) -> None:
    chains = deque(_Chain(case, deque(arms)) for case in cfg.cases
                   if (arms := [a for a in arm_order(case.id, cfg.lever, cfg.arms)
                                if not results.finished(a, case.id)]))
    budget, running = Budget(cfg), {}
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        while chains or running:
            while chains and run.fatal is None and len(running) < cfg.workers:
                chain = chains[0]
                key = (chain.arms[0].path.name, chain.case.id)
                if not budget.admit(key):
                    break
                chains.popleft()
                arm = chain.arms.popleft()
                running[pool.submit(work, arm, chain.case)] = (chain, arm, key)
            if not running:
                break
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                chain, arm, key = running.pop(future)
                outcome = future.result()  # anything but a verdict on the case is a bug, and propagates
                budget.release(key, outcome.cost)
                run.outcomes[key] = outcome
                log(f"{arm.path.name} {chain.case.id}: {outcome.status}")
                if outcome.fatal:
                    run.fatal = outcome.status.removeprefix("refused: ")
                elif chain.arms:
                    chains.appendleft(chain)
    run.capped = bool(chains) and run.fatal is None


def execute(cfg: Config, work: Worker | None, run: results.Run, log: Callable[[str], None] = print) -> None:
    """Run every case of every arm that is not finished, under the budget and `--workers`, recording in `run` how each
    ended (so a failure part-way still leaves what was done). `work` is `systems.prepare`'s; None runs a baseline."""
    if work is not None:
        _schedule(cfg, work, run, log)
        return
    for arm in cfg.arms:
        todo = {case.id for case in cfg.cases if not results.finished(arm, case.id)}
        if todo:
            for cid, outcome in systems.run_baselines(cfg, arm, todo).items():
                run.outcomes[arm.path.name, cid] = outcome
                log(f"{arm.path.name} {cid}: {outcome.status}")
