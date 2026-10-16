"""The systems `stbench eval` runs, and how one case of each is run (docs/eval.md).

SYSTEM is one of the three kinds `cmd_run` knows (`cmd_run.classify`): a baseline, `FILE.py:Class` (an external system:
the Docker sandbox, or with `--local` a plain subprocess, stamped `sandbox: none`), or `<agent>:<model>` through
`cmd_run.AGENTS`. Each case of each run writes `RUNDIR/cases/<cid>/` through `cmd_run.write_run`; what resume and the
results read is those folders, never anything kept in memory.

- A baseline learns from, or streams, other cases of the selection (`suite_priors.options_for`), so it always runs
  against the whole selection (`suite.run_suite`, in processes, the way `stbench check` runs them) and only the folders
  of the unfinished cases are written: a baseline is deterministic and free.
- An agent or an external system runs one case a thread (`cmd_run.resolve` builds its runner once): an agent is given a
  namespace of its own with `out` its case folder, made first. A run that cannot start is `refused`, one that finished
  but cannot be scored `invalid`, one that ran says how it ended (`complete`, or `ended: <why>`); a case's `cost_usd` is
  always `prices.cost_usd` of what it reported, never a figure of its own.
"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import baselines, cmd_run, prices, suite, suite_priors
from .case import CasePaths
from .cmd_run import BASELINE, Refused, RunInvalid
from .safecopy import remove_tree
from .stream import CaseRun
from .stream_protocol import CHANNELS

CASES = "cases"   # RUNDIR/cases/<cid>/


@dataclass(frozen=True)
class Arm:
    """One run folder: where it is and the two options that can differ between arms (`value` is the arm's value of
    the lever, None for a run without arms)."""

    path: Path
    brief: str
    during: bool
    value: str | None = None

    def case_dir(self, cid: str) -> Path:
        return self.path / CASES / cid


@dataclass(frozen=True)
class Config:
    """What one `eval` runs: the system, the selected cases (compiled, current), the run folders and the options."""

    system: str
    kind: str                      # cmd_run.BASELINE, EXTERNAL or AGENT
    data: Path
    out: Path
    channel: str
    cases: list[CasePaths]
    arms: list[Arm]
    lever: str | None              # "during" or "brief" with `--arms`
    workers: int
    local: bool
    timeout: float | None
    reasoning: str | None
    max_usd: float | None
    case_usd: float | None
    text_only: bool = False        # an agent is sent no image (`--text-only`)
    cells_from: Path | None = None  # the dataset whose cells (and weights) the scores use; None: `data` itself

    @property
    def sandbox(self) -> str:
        return cmd_run.sandbox_of(self.kind, self.local)

    def namespace(self, arm: Arm, out: Path | None = None) -> argparse.Namespace:
        """What `stbench run` hands a runner for one case of `arm` (`out`: the case's folder)."""
        return argparse.Namespace(system=self.system, channel=self.channel, brief=arm.brief,
                                  during="on" if arm.during else "off", local=self.local, timeout=self.timeout,
                                  reasoning=self.reasoning, text_only=self.text_only, out=out)


@dataclass(frozen=True)
class Outcome:
    """How one case of one arm ended: its status as `results.json` says it, why (for a case that is `invalid`), what
    it cost in dollars (None: not known, which the budget counts as the case's reservation) and whether no later case
    can start either (the agent itself refused)."""

    status: str
    error: str | None
    cost: float | None
    fatal: bool = False


Worker = Callable[[Arm, CasePaths], Outcome]


def channel_problem(system: str, kind: str, channel: str) -> str | None:
    """Why a baseline cannot run on `channel`, else None (an agent says so itself, an external system declares its
    channels to the shim only)."""
    if kind != BASELINE:
        return None
    channels = getattr(baselines.BASELINES[system].cls, "channels", tuple(CHANNELS))
    if channel in channels:
        return None
    return f"{system} runs on the {', '.join(channels)} channel only, not {channel!r}"


def _fresh(folder: Path) -> Path:
    remove_tree(folder)
    folder.mkdir(parents=True)
    return folder


def _written(cfg: Config, arm: Arm, case: CasePaths, run: CaseRun, where: str, folder: Path) -> Outcome:
    """The case's three files from `run`, and how it ended."""
    try:
        cmd_run.write_run(folder, case, cfg.namespace(arm), run, where)
    except ValueError as error:  # a gold the scorer rejects: the answers and the summary are kept to inspect
        return Outcome("invalid", f"not scored: {error}", None)
    return Outcome(run.status, None, prices.cost_usd(run.usage))


def prepare(cfg: Config, stack: contextlib.ExitStack) -> Worker | None:
    """The function that runs one case of one arm on an agent or an external system, built once (`cmd_run.resolve`: an
    external system's sandbox program lives as long as `stack`); None for a baseline, which runs the whole selection
    at once (`run_baselines`). Refused as `cmd_run.classify` and `resolve` say, before any case is run."""
    if cfg.kind == BASELINE:
        return None
    where, runner = cmd_run.resolve(cfg.namespace(cfg.arms[0]), cfg.cases[0], stack)

    def work(arm: Arm, case: CasePaths) -> Outcome:
        folder = _fresh(arm.case_dir(case.id))
        try:
            run = runner(case, cfg.namespace(arm, folder))
        except Refused as error:  # the agent cannot run at all: no later case can do better
            remove_tree(folder)
            return Outcome(f"refused: {error}", None, 0.0, fatal=True)
        except RunInvalid as error:  # its files stay: they say why
            return Outcome("invalid", str(error), None)
        except (ValueError, OSError) as error:  # `run_case` refuses before anything is sent
            remove_tree(folder)
            return Outcome(f"refused: {error}", None, 0.0)
        return _written(cfg, arm, case, run, where, folder)

    return work


def run_baselines(cfg: Config, arm: Arm, todo: set[str]) -> dict[str, Outcome]:
    """The baseline over the whole selection (`suite.run_suite`: its priors need every case, and the baselines are
    deterministic and free), with the folders of the cases in `todo` (their ids) written: their outcomes by case id. A
    system keyed on another's answers runs that one first (`baselines.NM_SYSTEM`)."""
    prior_nm = baselines.BASELINES[cfg.system].prior_nm
    runs = suite.run_suite(cfg.cases, [cfg.system], cfg.channel, options=suite_priors.options_for,
                           waves=[[baselines.NM_SYSTEM], [cfg.system]] if prior_nm else None, workers=cfg.workers,
                           brief=arm.brief, during=arm.during)[cfg.system]
    return {case.id: _written(cfg, arm, case, run, cfg.sandbox, _fresh(arm.case_dir(case.id)))
            for case, run in zip(cfg.cases, runs, strict=True) if case.id in todo}
