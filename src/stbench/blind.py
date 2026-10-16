"""The shortcut baselines that answer from other scripts' records alone: `blind` and `questionnaire_blind` (V0 §10), and
the pool machinery every prior system shares (`blind`, `path_log`, `current_view_prior`).

A prior is a list of records (`answer_keys.RECORD_KEYS`) about other scripts' asked questions, each with its gold row.
`pool_at` finds the pool of a question: the prior's records that share its key at the first level of
`answer_keys.levels` whose records are enough to learn from (`MIN_POOL_SCRIPTS` scripts, `MIN_POOL_GOLDS` golds), else
none: a level held by one or two records is one world's accident, and an answer from it scores below abstaining.
`best_response` answers with whichever candidate (`answer_candidates`, then any extra) has the best mean penalised score
over gold rows; `unknown` comes first, so it wins every tie. That mean is read on the golds the candidate was chosen
from, so the winner of a small pool looks better there than it is on any other script: the best of many candidates on a
few correlated golds is partly luck, and a pool that is only noise still has a winner that beats abstaining in-sample.
`learned_answer`, which every shortcut that learns from other scripts answers through, therefore keeps the best
response only when the same rule, run leaving each script out in turn, beats abstaining on the scripts it left out
beyond what the draw of scripts explains: the lower end of the 95% interval of its held-out gain over abstaining, from a
bootstrap over the scripts (`held_out_gain_low`), is above 0, else `unknown`. A mean that merely beats abstaining on a
few scripts is not enough: it answered at a loss on other scripts (the AMR's `task_things`, the drone's `ego_now`), and
a shortcut that does worse than abstaining is BROKEN. A pool of an `in_pool` prior skips that round and answers with the
pool's own winner:
`check` scores every learned shortcut under both rules (`baselines.in_pool_twins`) and a gate reads the larger, since
the held-out round abstains where a fair learner would not on a small pool near the boundary.

A time gold is a clock on its own case's day (drawn per case), so it is learned as a lag: how far it lies behind the
clock of the step its question was asked at (a record's `clock`, which a system reads from the ask). A pool whose
question has a clock moves every record's time gold onto `EPOCH` (`lagged`) and the asker's own time candidates with it
(`to_lag`), learns there, and moves the answer onto the asker's clock (`at_clock`): a gold that lay 15 s behind its
case's last clock is answered 15 s behind the asking case's, on any day. Read as clocks, golds of other days never
repeat, and no time answer would be credited.

`blind` keys on the template and the question's slots (clocks dropped, then each slot of `SLOT_DROP_ORDER` in turn, in
the slots and in the descriptions nested in them); it never sees a stream. `questionnaire_blind` knows one more thing:
the other questions of the end ask it is in, and answers from a feature's records only under the same support floor
(`supported`). Neither reads a picture, a pose or a map. Every system here builds the record it keys on from what it
was sent (`build_record`) and exposes it as `ask_record`, so the harness's prior records and the systems' lookups
agree.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from datetime import datetime

import numpy as np

from . import answer_keys
from .answer_candidates import UNKNOWN, candidates
from .score import penalised
from .score_stats import N_BOOT
from .stream_protocol import System

Memo = dict[tuple[str, int], float]   # (candidate's JSON, id of a gold row) -> its penalised score in one selection
# What a key level's records must hold before a prior answers from it (VISION rule 5: the best answer learned from other
# cases, not the memory of one). Three scripts are three worlds, so one world's accident cannot be the pool; five golds
# are the least that a majority can be told from its exceptions. Fixed before data.
MIN_POOL_SCRIPTS = 3
MIN_POOL_GOLDS = 5
TIE = 1e-12   # a mean this far above another is better; within it the earlier candidate (or abstaining) stands
EPOCH = datetime(2000, 1, 1)   # the clock every record's own ask clock is moved onto, to read its time gold as a lag
EPOCH_CLOCK = EPOCH.isoformat()


class Prior(list):
    """The records of other scripts a shortcut learns from (`opts["prior"]`) and the rule it keeps its best response by:
    `in_pool` False (the default) is `learned_answer`'s held-out round, True the pool's own winner (`best_response`)."""

    def __init__(self, records=(), in_pool: bool = False):
        super().__init__(records)
        self.in_pool = in_pool


class Pool(list):
    """The records one question's key found (`pool_at`) and what `learned_answer` needs of the question: its `clock`
    (the delivered clock of the step it is asked at: time golds are then read as lags; None reads them as clocks;
    `EPOCH_CLOCK` learns the lags and leaves the answer there, for the caller to `at_clock`) and the prior's `in_pool`.
    A plain list is a pool with neither."""

    def __init__(self, records=(), clock: str | None = None, in_pool: bool = False):
        super().__init__(records)
        self.clock, self.in_pool = clock, in_pool


def to_lag(answer: dict, clock: str) -> dict:
    """A time answer in the clock of a case whose question was asked at `clock`, moved onto `EPOCH`: where it would lie
    were that ask at `EPOCH`. Any other answer as it stands."""
    if answer.get("type") != "time" or "clock" not in answer:
        return answer
    moved = datetime.fromisoformat(answer["clock"]) + (EPOCH - datetime.fromisoformat(clock))
    return {**answer, "clock": moved.isoformat()}


def at_clock(answer: dict, clock: str | None) -> dict:
    """`to_lag`'s inverse: a time answer learned for an ask at `EPOCH`, moved onto the asker's `clock` (None leaves it).
    Any other answer as it stands."""
    if clock is None or answer.get("type") != "time" or "clock" not in answer:
        return answer
    moved = datetime.fromisoformat(answer["clock"]) + (datetime.fromisoformat(clock) - EPOCH)
    return {**answer, "clock": moved.isoformat()}


def _is_time(p: dict) -> bool:
    """Whether the gold of the record `p` is a time."""
    return p["gold"]["gold"]["type"] == "time"


def ask_clock(p: dict) -> str | None:
    """The `clock` of the record `p`, the one its time gold is read against; None for a record with none (missing or
    empty) whose gold is no time. ValueError for a time gold of a record with no clock: read as a clock on its own day
    it could never be credited, which is the fault the lag removes."""
    if clock := p.get("clock"):
        return clock
    if _is_time(p):
        raise ValueError(f"the record of question {p['gold'].get('qid')!r} has a time gold and no `clock` to read it "
                         "against")
    return None


def lagged(p: dict) -> dict:
    """The record with a time gold moved by its own `clock` (`to_lag`), so that golds of cases on other days compare;
    any other record as it stands. ValueError for a time gold of a record with no clock (`ask_clock`)."""
    if not _is_time(p):
        return p
    gold = p["gold"]
    return {**p, "gold": {**gold, "gold": to_lag(gold["gold"], ask_clock(p))}}


def supported(golds: int, scripts: int) -> bool:
    """Whether `golds` gold rows from `scripts` scripts are enough to learn from: the support floor every prior and the
    form adversary answer under."""
    return golds >= MIN_POOL_GOLDS and scripts >= MIN_POOL_SCRIPTS


def build_record(question: dict, when: str, step: dict | None = None, **keys) -> dict:
    """The `answer_keys` record a prior system keys on at an ask: the question's `template` and `slots`, `when`, and,
    from `step` (the step the ask is about: the log's entry at the ask's `now["k"]`), its `clock`, its `phase`, its
    `base` (the delivered pose's x and y), its `yaw` (the delivered `yaw_deg`) and its `subgoal` (None when the step
    has none: the job states none, or the brief withheld it); `keys` adds the rest (`clock` for a system that holds no
    step, `rooms`, `nm`). A system holds only the keys its levels read and its time golds are read against (`clock`),
    built from what it was sent."""
    rec = {"template": question["template"], "slots": question["slots"], "when": when}
    if step is not None:
        rec |= {"clock": step["clock"], "phase": step["status"]["phase"], "base": tuple(step["pose"]["pos"][:2]),
                "yaw": step["pose"]["yaw_deg"], "subgoal": step.get("subgoal")}
    return rec | keys


def pool_at(system: str, rec: dict, prior: list[dict]) -> Pool:
    """The prior's records that share `rec`'s key at the first level of `answer_keys.levels(system, ...)` where they
    come from at least `MIN_POOL_SCRIPTS` scripts and number at least `MIN_POOL_GOLDS`; a level that holds fewer gives
    way to the next coarser one. Empty when no level has enough (an empty or thin prior, or one with no record of this
    template). Each record's `script` and `gold` are what `learned_answer` reads; the pool also holds `rec`'s `clock`
    (None without one) and the prior's `in_pool`."""
    want = answer_keys.levels(system, rec)
    keyed = [(answer_keys.levels(system, p), p) for p in prior]
    clock, in_pool = rec.get("clock"), getattr(prior, "in_pool", False)
    for depth, key in enumerate(want):
        hits = [p for keys, p in keyed if depth < len(keys) and keys[depth] == key]
        if supported(len(hits), len({p["script"] for p in hits})):
            return Pool(hits, clock, in_pool)
    return Pool((), clock, in_pool)


def _score(candidate: dict, gold: dict, memo: Memo | None) -> float:
    """`penalised(candidate, gold)`, kept in `memo` (when given) under the candidate's JSON and the gold row's id."""
    if memo is None:
        return penalised(candidate, gold)
    key = (json.dumps(candidate, sort_keys=True), id(gold))
    if key not in memo:
        memo[key] = penalised(candidate, gold)
    return memo[key]


def best_response(golds: list[dict], extra: Sequence[dict] = (), memo: Memo | None = None) -> dict:
    """The candidate with the highest mean penalised score over the gold rows: `answer_candidates.candidates(golds)`,
    then `extra`. The first of equals wins, so `unknown` wins ties; `memo` (when given) keeps scores between calls
    over the same gold rows. The mean is read on the rows the candidates come from, so it flatters the winner of a
    small pool: a shortcut answers through `learned_answer`."""
    best, best_mean = dict(UNKNOWN), -math.inf
    for candidate in [*candidates(golds), *extra]:
        if (mean := statistics.fmean(_score(candidate, g, memo) for g in golds)) > best_mean + TIE:
            best, best_mean = candidate, mean
    return best


def _held_out_scores(pool: list[dict], extra: Sequence[dict], memo: Memo) -> dict[str, list[float]]:
    """Each script's rows' penalised scores under the best response over the other scripts' rows alone (`unknown` when
    there are none), in the pool's order of scripts."""
    out = {}
    for script in dict.fromkeys(p["script"] for p in pool):
        others = [p["gold"] for p in pool if p["script"] != script]
        answer = best_response(others, extra, memo) if others else UNKNOWN
        out[script] = [_score(answer, p["gold"], memo) for p in pool if p["script"] == script]
    return out


def held_out_mean(pool: list[dict], extra: Sequence[dict] = (), memo: Memo | None = None) -> float:
    """The mean penalised score of `best_response` over the pool's gold rows when each script's rows are answered by
    the best response over the other scripts' rows alone (`unknown` when there are none): the rule's score on a script
    it did not learn from. `pool` is records with a `script` and a `gold` (`pool_at`)."""
    held = _held_out_scores(pool, extra, {} if memo is None else memo)
    return sum(sum(scores) for scores in held.values()) / len(pool)


def gain_low(gains: Sequence[float], rows: Sequence[int]) -> float:
    """The lower end of the 95% interval of a gain over abstaining read per script: `gains[i]` is script i's summed gain
    over its `rows[i]` rows. A bootstrap over the scripts (`score_stats.N_BOOT` resamples of whole scripts, a ratio of
    sums, seeded). Above 0 when the gain holds beyond what the draw of scripts explains. Few scripts leave it near the
    least gain of one script, so three scripts need every one to gain."""
    gains, rows = np.asarray(gains, dtype=float), np.asarray(rows)
    picks = np.random.default_rng(0).integers(0, len(gains), size=(N_BOOT, len(gains)))
    return float(np.percentile(gains[picks].sum(axis=1) / rows[picks].sum(axis=1), 2.5))


def held_out_gain_low(pool: list[dict], extra: Sequence[dict] = (), memo: Memo | None = None) -> float:
    """The lower end of the 95% interval of the rule's held-out gain over abstaining (`held_out_mean` less `unknown`'s
    mean over the same rows), from a bootstrap over the pool's scripts (`gain_low`): above 0 when the rule beats
    abstaining on the scripts it left out beyond what the draw of scripts explains."""
    memo = {} if memo is None else memo
    held = _held_out_scores(pool, extra, memo)
    gains = [sum(scores) - sum(_score(UNKNOWN, p["gold"], memo) for p in pool if p["script"] == script)
             for script, scores in held.items()]
    return gain_low(gains, [len(scores) for scores in held.values()])


def learned_answer(pool: list[dict], extra: Sequence[dict] = (), memo: Memo | None = None) -> dict:
    """What a shortcut that learns from other scripts answers from `pool` (records with a `script` and a `gold`, as
    `pool_at` gives): the best response over the pool's gold rows, kept only where the rule beats abstaining on scripts
    it did not learn from beyond what the draw of scripts explains (the lower end of the 95% interval of its held-out
    gain over abstaining, `held_out_gain_low`, is above 0), else `unknown`: a rule that beats abstaining on average over
    the scripts left out, by what the luck of a few scripts can do, is not kept. The in-pool mean cannot decide that: it
    picks the best of many candidates on the very rows it scores them on. An empty pool is `unknown`. `extra`
    candidates stand as given in every held-out round (a raycast or a claim is the case's own; the places `path_log`
    takes from the pool's density have seen the script left out, a small leak); `memo` keeps scores between calls over
    the same gold rows, which must stay alive while it does. A pool that is `in_pool` answers with its best response as
    it is, not asked. A pool with a `clock` learns its time golds as lags (the module's docstring): the extra time
    candidates are the asker's own clocks and move with them, and the answer comes back on the asker's clock. A pool
    with no clock reads them as clocks; an empty clock over time golds is refused (ValueError), as is a record of the
    pool with a time gold and no clock (`ask_clock`)."""
    if not pool:
        return dict(UNKNOWN)
    clock, in_pool = getattr(pool, "clock", None), getattr(pool, "in_pool", False)
    if clock == "" and any(_is_time(p) for p in pool):   # no clock reads clocks as clocks; an empty one is no clock
        raise ValueError("the ask has an empty `clock` and the pool holds time golds, which are learned as lags "
                         "behind it")
    if clock is not None:
        moved = [lagged(p) for p in pool]
        if any(m is not p for m, p in zip(moved, pool, strict=True)):
            pool, memo = moved, None   # new rows: scores kept under their identity stay out of a shared memo
        extra = [to_lag(e, clock) for e in extra]
    memo = {} if memo is None else memo
    answer = best_response([p["gold"] for p in pool], extra, memo)
    if answer["type"] != UNKNOWN["type"] and not in_pool and held_out_gain_low(pool, extra, memo) <= TIE:
        answer = dict(UNKNOWN)
    return at_clock(answer, clock)


class Blind(System):
    """Answers from the question alone: what `learned_answer` says of the other scripts' records at the first `blind`
    level that has enough of them (`pool_at`), `unknown` with no pool. `opts["prior"]` is the records (default none)."""

    structured = True
    reads_files = False   # it sees no step, let alone a picture

    def __init__(self, state_dir, workdir, **opts):
        super().__init__(state_dir, workdir, **opts)
        self.prior: list[dict] = opts.get("prior", [])

    def observe(self, steps: list[dict]) -> None:
        pass

    def ask_record(self, question: dict, now: dict, *, when: str) -> dict:
        """The record it keys on: `template`, `slots` and `when`, and the ask's `clock` (`now["clock"]`), which only its
        time golds are read against. It sees no step, so nothing else."""
        return build_record(question, when, clock=now["clock"])

    def ask(self, when: str, now: dict, questions: list[dict]) -> dict[str, dict]:
        return {q["qid"]: learned_answer(pool_at("blind", self.ask_record(q, now, when=when), self.prior))
                for q in questions}


def features(others: list[dict]) -> dict[str, tuple]:
    """What an end ask's other questions (`{template, slots}`) say: `count` (how many), `templates` (their sorted
    templates) and, for each slot name any of them holds, `slot:<name>` (the sorted `json.dumps` of their values)."""
    out: dict[str, tuple] = {"count": (len(others),), "templates": tuple(sorted(o["template"] for o in others))}
    for name in sorted({n for o in others for n in o["slots"]}):
        out[f"slot:{name}"] = tuple(sorted(json.dumps(o["slots"][name], sort_keys=True) for o in others
                                           if name in o["slots"]))
    return out


class QuestionnaireBlind(Blind):
    """`blind` plus the other questions of its end ask. For an end question it picks, among the features of the ask's
    other questions, the one whose learned answer (`learned_answer`) scores highest leaving one script out (the first
    on a tie), and answers with the learned answer over the records of that template that share the feature's value:
    `unknown` when they are fewer than the support floor (`supported`: `MIN_POOL_GOLDS` golds of `MIN_POOL_SCRIPTS`
    scripts; the leave-one-out score reads the same floor). Every during question is `unknown`."""

    def __init__(self, state_dir, workdir, **opts):
        super().__init__(state_dir, workdir, **opts)
        self._memo: Memo = {}
        self._learned: dict[tuple[int, ...], dict] = {}   # (identities of a pool's records) -> its learned answer
        self._means: dict[tuple[str, str], float] = {}   # (template, feature) -> its leave-one-script-out mean
        self.in_pool: bool = getattr(self.prior, "in_pool", False)

    def ask(self, when: str, now: dict, questions: list[dict]) -> dict[str, dict]:
        if when != "end":
            return {q["qid"]: dict(UNKNOWN) for q in questions}
        return {q["qid"]: self._answer(q, [{"template": o["template"], "slots": o["slots"]} for o in questions
                                           if o["qid"] != q["qid"]], now["clock"]) for q in questions}

    def _same(self, template: str) -> list[dict]:
        """The prior's records of end questions of the template."""
        return [p for p in self.prior if p["when"] == "end" and p["template"] == template]

    def chosen(self, template: str, others: list[dict]) -> str | None:
        """The feature (a name of `features(others)`) whose learned answer, leaving one script out, scores highest over
        the prior's end records of the template (the first of equals); None when the prior has none."""
        same = self._same(template)
        return max(features(others), key=lambda name: round(self._mean(template, same, name), 9)) if same else None

    def _answer(self, question: dict, others: list[dict], clock: str) -> dict:
        name = self.chosen(question["template"], others)
        if name is None:
            return dict(UNKNOWN)
        value = features(others)[name]
        found = [p for p in self._same(question["template"]) if features(p["siblings"]).get(name) == value]
        return at_clock(self._learn(found), clock)

    def _learn(self, found: list[dict]) -> dict:
        """The learned answer over the records `found` (`unknown` under the support floor), kept per set of records. A
        time answer is learned for an ask at `EPOCH`, one lag to be put on an asker's clock (`at_clock`)."""
        key = tuple(map(id, found))
        if key not in self._learned:
            held = supported(len(found), len({p["script"] for p in found}))
            self._learned[key] = (learned_answer(Pool(found, EPOCH_CLOCK, self.in_pool), memo=self._memo) if held
                                  else dict(UNKNOWN))
        return self._learned[key]

    def _mean(self, template: str, same: list[dict], name: str) -> float:
        """The mean penalised score of answering each record of `same` from the other scripts' records that share its
        value of the feature `name` (`unknown` when they are under the support floor). ValueError for a record of `same`
        with a time gold and no clock (`ask_clock`), whether or not a pool reads it."""
        if (template, name) not in self._means:
            scores = []
            for r in same:
                value = features(r["siblings"]).get(name)
                found = [p for p in same if p["script"] != r["script"] and features(p["siblings"]).get(name) == value]
                scores.append(penalised(at_clock(self._learn(found), ask_clock(r)), r["gold"]))
            self._means[template, name] = statistics.fmean(scores)
        return self._means[template, name]
