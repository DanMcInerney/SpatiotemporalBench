"""Helpers for the tests of `stbench eval` (`test_eval_*`): hand-written compiled cases, fake systems and `cli`.

`compiled` lays a `stream_kit` script out as a dataset with a gold row for each of its questions, some of them
`unknown`: every score a test expects is worked out from these numbers, never
from the compiler. `hand_dataset` is the one two-case dataset most tests share: two scripts of the
20-step hand case, asked `QUESTIONS` (a during question, `path_ahead`, and two end ones, `object_where`), whose golds
are answerable except two of the end ones. Under `all_unknown` an answerable gold scores 0 and an unknown one 1.

`fake_agent` is a runner as `cmd_run.AGENTS` names one (`register_agent` adds it as `fake`): its model says what the
run does (`ok`, `usage`: it reports tokens of two models and a remote store, `stop`: it fails in `observe`, `refuse`,
`invalid`), and every case it runs is noted in `RUNS` with the messages it was sent.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from run_kit import golds
from stbench import case as case_mod
from stbench import cmd_run
from stbench.baselines import BASELINES, Baseline
from stbench.cli import main
from stbench.stream import CaseRun, run_case
from stbench.stream_protocol import System
from stream_kit import QUESTIONS, K, Recorder, case_on, during_q, end_q, gold_row, hand20
from synth import Synth

UNKNOWN = {"type": "unknown"}
RUNS: list[dict] = []   # one entry a case `fake_agent` ran: the case, the options, the messages it was sent


def gold(question: dict, *, unknown: bool = False) -> dict:
    """`stream_kit.gold_row`; an `unknown` gold when asked."""
    row = gold_row(question)
    return {**row, "gold": UNKNOWN, "slice": "unknown"} if unknown else row


def compiled(dataset: Path, synth, questions: list[dict] = QUESTIONS, unknown: tuple[int, ...] = (), *,
             cid: str | None = None, **kw) -> case_mod.CasePaths:
    """`synth` as a case under `dataset` (id `cid` when given) asking `questions`, with a gold row each; the questions
    at the indices `unknown` have an `unknown` gold."""
    with mock.patch.object(case_mod, "claim_id", (lambda dataset, entry: cid) if cid else case_mod.claim_id):
        case = case_on(dataset, synth, questions, **kw)
    golds(case, rows=[gold(q, unknown=i in unknown) for i, q in enumerate(questions)])
    return case


def hand_script(seed: int):
    """The 20-step hand script of a seed (its own world: a script of its own)."""
    synth = hand20()
    synth.seed = seed
    return synth


def hand_dataset(root: Path) -> list[case_mod.CasePaths]:
    """Two cases of two scripts under `root`. The first has golds [answerable, answerable, unknown], the second
    [answerable, unknown, unknown]."""
    return [compiled(root, hand_script(1), unknown=(2,)), compiled(root, hand_script(2), unknown=(1, 2))]


SHORT_QUESTIONS = [during_q(1, 3), end_q(2), end_q(3)]   # a during question at step 3 and two end ones


def short_script(seed: int) -> Synth:
    """Six steps of one task in the kitchen with a decision point at step 3 and a mug in view all along: a case a
    scripted system that builds a map from depth images (`memory_table`) runs in a fraction of a second."""
    synth = Synth(tasks=[[(K, 6)]], decisions=[(3, "before_doorway")], seed=seed)
    synth.item("E01", "mug", {"color": "red"}, [(0, None, (2.0, 1.0, 0.9))])
    synth.see(range(6), "E01")
    return synth


def short_dataset(root: Path) -> list[case_mod.CasePaths]:
    """Two cases of two `short_script`s, asked `SHORT_QUESTIONS`, with golds [answerable, answerable, unknown]."""
    return [compiled(root, short_script(seed), SHORT_QUESTIONS, unknown=(2,)) for seed in (1, 2)]


class Counting(System):
    """`all_unknown` that counts the `observe` calls of every instance (the class attribute; a test resets it)."""

    observed = 0

    def observe(self, steps):
        type(self).observed += 1

    def answer(self, question):
        return dict(UNKNOWN)


def register(monkeypatch) -> None:
    """Add `counting` to the baselines for one test."""
    monkeypatch.setitem(BASELINES, "counting", Baseline(Counting))
    Counting.observed = 0


class Reporting(Recorder):
    """A `Recorder` whose `finish` reports `opts["usage"]` and which fails in `observe` when `opts["fail"]`."""

    def observe(self, steps):
        if self.opts.get("fail"):
            raise RuntimeError("out of memory")
        super().observe(steps)

    def finish(self):
        super().finish()
        return self.opts.get("usage", {})


TOKENS = {"m1": {"tokens_in": 1000, "tokens_cached_in": 200, "tokens_cache_write_in": 0, "tokens_out": 100},
          "m2": {"tokens_in": 500, "tokens_cached_in": 0, "tokens_cache_write_in": 0, "tokens_out": 50}}
PRICES = {"m1": {"in": 2.0, "cached_in": 0.2, "cache_write_in": 2.5, "out": 10.0},
          "m2": {"in": 1.0, "cached_in": 0.1, "cache_write_in": 1.25, "out": 5.0}}   # per million: by hand, $0.00339
COST = 0.00339   # m1: 800 x 2 + 200 x 0.2 + 100 x 10 = 2,640; m2: 500 x 1 + 50 x 5 = 750; over a million


def fake_agent(model: str, case: case_mod.CasePaths, args) -> CaseRun:
    """The runner of `fake:<model>`: `refuse` raises the harness's `Refused`, `invalid` its `RunInvalid`; `usage`
    reports `TOKENS` and a remote store of 4,096 bytes; `stop` fails in `observe`; anything else is an `ok` run."""
    if model == "refuse":
        raise cmd_run.Refused("no such login")
    if model == "invalid":
        raise cmd_run.RunInvalid("the leak scan hit")
    usage = ({"models": TOKENS, "peak_context": 777, "remote_store_bytes": 4096} if model == "usage"
             else {"models": {"m1": TOKENS["m1"]}})
    log: list = []
    RUNS.append({"model": model, "case": case.id, "brief": args.brief, "during": args.during, "out": args.out,
                 "reasoning": args.reasoning, "memory": getattr(args, "memory", None),
                 "compact_at": getattr(args, "compact_at", None), "log": log})
    return run_case(lambda state, work: Reporting(state, work, log=log, usage=usage, fail=model == "stop"), case,
                    channel=args.channel, brief=args.brief, during=args.during == "on")


def register_agent(monkeypatch) -> None:
    monkeypatch.setitem(cmd_run.AGENTS, "fake", "eval_kit.fake_agent")
    RUNS.clear()


def cli(*argv) -> int:
    return main(["eval", *(str(a) for a in argv)])


def results(folder: Path) -> dict:
    return json.loads((folder / "results.json").read_text(encoding="utf-8"))


def case_files(folder: Path) -> dict[str, bytes]:
    """Every file under a run folder but `results.*` (which say when they were written), by relative path."""
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*"))
            if p.is_file() and not p.name.startswith("results.")}
