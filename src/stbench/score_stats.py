"""Aggregation over result rows, bootstrap intervals over scripts, paired differences and gap closed (plan §2.6).

A result row is a plain dict with `ROW_KEYS` (`score.row` makes one): `(case, qid)` is unique, `script` the cluster.

The headline is the mean of two columns, `now` and `memory` (VISION rule 4, 0.6.0): a column is the mean of its cells by
their weights, a cell being an (ability, scope) pair (`abilities.ability_of`, the row's `scope`), each cell the
reported score pooled over its own rows, a ratio of sums, so a cell's mean weighs the same whatever its number of
questions and the two columns weigh equally whatever theirs. An ability whose rows are in both scopes (clearance: the
corridor ahead is answered from the present view, a person from memory) has a cell in each column and each counts on
its own. Over rows of one scope the headline is that column.
**Weights** (0.11.0, FD-0.11 §3.7). A row weighs what its (job, ability, column) cell weighs in the dataset,
`min(1, n / MIN_ABILITY_N)` for its `n` questions (`ability_counts.weights`: counted over the whole dataset's golds, or
another's, `--cells-from`, so a subset run keeps the release's cells). `thin` holds the cells below 1, `{cell:
weight}`; every other cell weighs 1, so a dataset whose cells hold 20 questions or more is unweighted. A cell's mean
pools its rows by their weights, the cell weighs in its column the largest weight of its rows and the column in the
headline the largest weight of its cells, so a cell, and a column of thin cells alone, move in and out continuously
where they once vanished under 20 questions. A set of cells reads as weight 0 each (the rule of
0.6.0 to 0.10.0: counted in no mean, which the results of an earlier build record). A diagnostic ability
(`abilities.DIAGNOSTIC`) is counted in no mean, and `by.ability` lists it flagged, an ability's `cells` apart. Whether
the rule applies at all is decided once for the dataset (`ability_counts.thin_rule`: not when every non-diagnostic cell
is thin), never from the rows of one run, so a subset and the full run agree; a caller that does not apply it passes no
`thin` and says so with `thin_ignored`. A `thin` entry of two names, (job, ability), is thin in both columns.
Beside the headline, from the same resample: `columns` (the two columns), `abilities_mean` (the mean of all the counted
cells by their weights) and `pooled` (the counted rows pooled). The splits (`DIMS`) are pooled inside each group, so a
family of eight questions weighs eight times a family of one within a split.
Confidence intervals are 95% percentile intervals from a cluster bootstrap that resamples whole scripts (every case of a
script, and so every question, moves together: the cases of one script are windows of one world and one path), with a
seeded generator, so a result is deterministic and independent of row order. A resample that lacks every row of a group
leaves that group out of its interval. The headline's interval is the mean's own: both columns and their mean are
computed on every resample, from one set of script weights, and a resample that lacks a column takes the other alone.
With fewer than two scripts an interval is `None`.

Four splits read the row's own fields and add none to it: `slice` (answerable or unknown gold), `relevance` (an
answerable row is `critical` or `situational` by its tag; a row with an unknown gold is `unknown`, whatever its tag),
`ability` (the family's ability) and `kind` (the family's kind, harm, help, situation or reuse, the vision's
four; `abilities.kind_of`). `kind` is a split like the others, pooled within each group; it is not a cell, so the
headline and every cell count as they did. A kind none of whose rows the means count still reports its mean, flagged
`diagnostic` as an ability is.

**Two scores** (0.10.0, FD-questions-0.10 §4, review B4). A row holds `score`, the reported score (the penalised one
less a catastrophe: `score.reported`), and `score_gate`, the penalised score alone (`score.penalised`). Everything here
reads `score` unless it is told `key="score_gate"`, which the gates, the balance check and the form adversary do
(`gate_score`): no weight of a catastrophe ever reaches a validity verdict. A row of an earlier build holds no
`score_gate` (its `score` was the penalised one: read as that) and no `catastrophe` (read as false).
Two reports stand beside the headline, from the same resample: `by.age_band` and `job_age_band` (the reported score and
the count by the evidence age's band, `records.AGE_BANDS`, `none` for a row with no band) and `catastrophe_rate` (the
catastrophes over the harm rows answered, overall and per job).
`above_floor` is the headline less the unknown share: what the system scores beyond answering `unknown` throughout,
which scores exactly that share (1 on an unknown gold, 0 on an answerable one), weighted and resampled like the
headline. `paired_difference(headline=True)` takes the headline's statistic of the differences; by default it pools
them.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .abilities import DIAGNOSTIC, ability_of, kind_of
from .cell_weights import Thin, thin_weights, weight_of
from .records import AGE_BANDS, SCOPES

OUTCOMES = ("correct", "partial", "wrong", "abstained")
SLICES = ("answerable", "unknown")
N_BOOT = 2000
INFO_KEYS = ("case", "script", "layout", "robot", "job", "length")  # what the caller knows of a case
ROW_KEYS = ("case", "script", "qid", "family", "when", "scope", "reference", "critical", "slice", "answer_class",
            "range_m", "age_band", "s", "score", "score_gate", "catastrophe", "outcome", "layout", "robot", "job",
            "length")
# A row of an earlier run lacks these: `answer_class` and `range_m` (before 0.6.0) are read as null, `age_band` (before
# 0.10.0) as null, `score_gate` as its `score` and `catastrophe` as false (`_defaults`).
OPTIONAL_ROW_KEYS = ("answer_class", "range_m", "age_band", "score_gate", "catastrophe")
SCORE_KEYS = ("score", "score_gate")   # the row fields a score is read from: the reported one, the penalised one
GATE = "score_gate"
DIMS = ("scope", "own_path", "family", "ability", "kind", "when", "length", "layout", "robot", "job", "critical",
        "slice", "relevance", "answer_class", "age_band")  # `by`'s splits
NEW_DIMS = ("age_band",)   # the splits an aggregate of an earlier build lacks
DERIVED = ("relevance", "ability", "kind")   # the splits (and `where` keys) that are not a row's own field

Row = dict
Interval = tuple[float, float] | None


def relevance(row: Row) -> str:
    """The row's relevance class: `unknown` for an unknown gold, else `critical` or `situational` by its `critical`
    tag."""
    if row["slice"] == "unknown":
        return "unknown"
    return "critical" if row["critical"] else "situational"


def ability(row: Row) -> str:
    """The row's ability: its family's (`abilities.ability_of`)."""
    return ability_of(row["family"])


def gate_score(row: Row) -> float:
    """The row's gate score: its `score_gate`, the penalised score; a row of an earlier build has none, and its `score`
    was that."""
    return row["score_gate"] if row.get("score_gate") is not None else row["score"]


def _defaults(row: Row) -> Row:
    """What an `OPTIONAL_ROW_KEYS` key is read as when the row lacks it."""
    return {"answer_class": None, "range_m": None, "age_band": None, "catastrophe": False}


def _key(key: str) -> str:
    if key not in SCORE_KEYS:
        raise ValueError(f"key must be one of {list(SCORE_KEYS)}, got {key!r}")
    return key


def _check(rows: Iterable[Row]) -> list[Row]:
    """The rows, validated, each with every `OPTIONAL_ROW_KEYS` key present (`_defaults` when the row lacks it)."""
    rows, seen = list(rows), set()
    for r in rows:
        if missing := [k for k in ROW_KEYS if k not in r and k not in OPTIONAL_ROW_KEYS]:
            raise ValueError(f"result row {r!r} lacks {missing}")
        if r["outcome"] not in OUTCOMES or r["slice"] not in SLICES:
            raise ValueError(f"result row {r['qid']!r}: outcome must be in {OUTCOMES}, slice in {SLICES}")
        if r["scope"] not in SCOPES:
            raise ValueError(f"result row {r['qid']!r}: scope must be in {SCOPES}, got {r['scope']!r}")
        if (r["case"], r["qid"]) in seen:
            raise ValueError(f"duplicate result for case {r['case']!r}, qid {r['qid']!r}")
        seen.add((r["case"], r["qid"]))
    return [{**_defaults(r), **r, "score_gate": gate_score(r)} for r in rows]


def _weights(n_scripts: int, n_boot: int, seed: int) -> np.ndarray:
    """Row 0 is all ones (the point estimate); then `n_boot` bootstrap multiplicities over scripts."""
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    draws = np.random.default_rng(seed).integers(0, n_scripts, size=(n_boot, n_scripts))
    w = np.zeros((n_boot, n_scripts))
    np.add.at(w, (np.arange(n_boot)[:, None], draws), 1.0)
    return np.vstack([np.ones(n_scripts), w])


def _interval(samples: np.ndarray, n_scripts: int) -> Interval:
    samples = samples[~np.isnan(samples)]
    if n_scripts < 2 or samples.size == 0:
        return None
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return (float(lo), float(hi))


def _point(x: float) -> float | None:
    return None if np.isnan(x) else float(x)


def _mean(parts: list[np.ndarray], size: int, weights: list[float] | None = None) -> np.ndarray:
    """The mean of the arrays `parts` at each position by their `weights` (equal by default), an array that is NaN
    there left out; NaN where all are (or there are none)."""
    if not parts:
        return np.full(size, np.nan)
    stack = np.vstack(parts)
    w = np.ones((len(parts), 1)) if weights is None else np.array(weights, float)[:, None]
    live = ~np.isnan(stack)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(live.any(axis=0), (np.where(live, stack, 0.0) * w).sum(axis=0) / (live * w).sum(axis=0), np.nan)


class _Scripts:
    """Rows grouped by script, with the weights of the point estimate and of each resample over those scripts, and the
    rows' cells: the rows the means count (not a diagnostic ability, not a cell of weight 0), grouped by (scope,
    ability), each cell in the column of its scope. `weight` is each counted row's weight (0 for another)."""

    def __init__(self, rows: list[Row], n_boot: int, seed: int, thin: Thin = frozenset()):
        self.n = len(rows)
        self.scripts = sorted({r["script"] for r in rows})
        at = {s: i for i, s in enumerate(self.scripts)}
        self.script_of = np.array([at[r["script"]] for r in rows], dtype=int)
        self.scope = np.array([r["scope"] for r in rows])
        self.w = _weights(len(self.scripts), n_boot, seed)
        self.ability = np.array([ability(r) for r in rows])
        diagnostic = np.array([a in DIAGNOSTIC for a in self.ability], dtype=bool)
        held = thin_weights(thin)
        weight = np.array([weight_of(held, r["job"], a, r["scope"]) for r, a in zip(rows, self.ability, strict=True)])
        self.counted = ~diagnostic & (weight > 0)
        self.weight = np.where(self.counted, weight, 0.0)
        cells = sorted(set(zip(self.scope[self.counted].tolist(), self.ability[self.counted].tolist(), strict=True)))
        self.groups = []   # (scope, ability, the cell's rows' weights, the cell's weight in its column)
        for sc, ab in cells:
            picked = self.weight * ((self.scope == sc) & (self.ability == ab))
            self.groups.append((sc, ab, picked, float(picked.max())))
        self.columns = {sc: max(x for c, _, _, x in self.groups if c == sc) for sc, _, _, _ in self.groups}

    def column_n(self, scope: str) -> int:
        """The counted rows in the column `scope`."""
        return int((self.counted & (self.scope == scope)).sum())

    def total(self, values: np.ndarray) -> np.ndarray:
        """Resampled sums of per-row `values`, shape (1 + n_boot,): element 0 is the point estimate."""
        return self.w @ np.bincount(self.script_of, weights=values, minlength=len(self.scripts))

    def ratio(self, num: np.ndarray, den: np.ndarray) -> dict:
        """`{mean, ci}` of resampled sums `num` over `den`: a ratio of sums, NaN (left out) where `den` is 0."""
        with np.errstate(invalid="ignore", divide="ignore"):
            samples = self.total(num) / self.total(den)
        return {"mean": _point(samples[0]), "ci": _interval(samples[1:], len(self.scripts))}

    def pooled(self, values: np.ndarray, picked: np.ndarray | None = None) -> dict:
        """The mean of `values` over the `picked` rows (all by default; a weight each) and its interval."""
        picked = np.ones(self.n) if picked is None else picked.astype(float)
        return self.ratio(values * picked, picked)

    def means(self, values: np.ndarray) -> dict[str, np.ndarray]:
        """The headline's statistics over per-row `values`, each an array over the point estimate and the resamples
        (element 0 is the point estimate), from one set of script weights: `now` and `memory` (the mean of the counted
        cells of the scope by their weights, each a ratio of sums over its own rows by theirs), `headline` (the mean of
        the two columns, each weighing the largest weight of its cells), `abilities_mean` (the mean of all the counted
        cells) and `pooled` (the counted rows pooled). A cell with no row in a resample, and a column with none, are
        left out of it: the other stands alone."""
        size = self.w.shape[0]
        each = []
        for scope, _, picked, weight in self.groups:
            with np.errstate(invalid="ignore", divide="ignore"):
                each.append((scope, self.total(values * picked) / self.total(picked), weight))
        out = {scope: _mean([m for sc, m, _ in each if sc == scope], size, [x for sc, _, x in each if sc == scope])
               for scope in SCOPES}
        with np.errstate(invalid="ignore", divide="ignore"):
            pooled = self.total(values * self.weight) / self.total(self.weight)
        return {**out, "headline": _mean([out[scope] for scope in SCOPES], size,
                                         [self.columns.get(scope, 1.0) for scope in SCOPES]),
                "abilities_mean": _mean([m for _, m, _ in each], size, [x for _, _, x in each]), "pooled": pooled}

    def stat(self, samples: np.ndarray) -> dict:
        """`{mean, ci}` of an array of the point estimate and the resamples."""
        return {"mean": _point(samples[0]), "ci": _interval(samples[1:], len(self.scripts))}

    def headline(self, values: np.ndarray) -> dict:
        """`{mean, ci}` of the headline's statistic over per-row `values` (`means`); the interval is the mean's own."""
        return self.stat(self.means(values)["headline"])


def pooled(rows: Iterable[Row], *, n_boot: int = N_BOOT, seed: int = 0, key: str = "score") -> dict:
    """`{mean, ci, n, scripts}` of the score pooled over `rows` (a ratio of sums), the interval from a bootstrap over
    scripts; `key` is the row field it is read from (`SCORE_KEYS`: the gates pass `GATE`). The mean and the interval
    are `None` for no rows."""
    rows, key = _check(rows), _key(key)
    if not rows:
        return {"mean": None, "ci": None, "n": 0, "scripts": 0}
    s = _Scripts(rows, n_boot, seed)
    return {**s.pooled(np.array([r[key] for r in rows])), "n": len(rows), "scripts": len(s.scripts)}


def _count(rows: Iterable[Row]) -> dict:
    out = dict.fromkeys(OUTCOMES, 0)
    for r in rows:
        out[r["outcome"]] += 1
    return out


def _value(row: Row, dim: str) -> str:
    match dim:
        case "own_path":
            return str(row["reference"] == "path_log")
        case "relevance":
            return relevance(row)
        case "ability":
            return ability(row)
        case "kind":
            return kind_of(row["family"])
        case "answer_class" | "age_band":
            return "none" if row[dim] is None else str(row[dim])
    return str(row[dim])


def _empty() -> dict:
    return {
        "n": 0, "scripts": 0, "cases": 0, "headline": {"mean": None, "ci": None}, "columns": {},
        "abilities_mean": {"mean": None, "ci": None}, "pooled": {"mean": None, "ci": None}, "thin": [],
        "thin_ignored": False, "above_floor": {"mean": None, "ci": None}, "by": {d: {} for d in DIMS},
        "counts": _count([]), "confidently_wrong_rate": {"rate": None, "ci": None}, "job_age_band": {},
        "catastrophe_rate": {"rate": None, "ci": None, "n": 0, "of": 0, "jobs": {}},
        "abstention": {"precision": None, "ci_precision": None, "recall": None, "ci_recall": None,
                       "answered_unknown": 0, "unknown_gold": 0},
    }


def _ages(s: _Scripts, rows: list[Row], score: np.ndarray) -> dict[str, dict]:
    """`job_age_band`: for each job, each evidence-age band its rows hold (`AGE_BANDS` order, `none` last), the pooled
    score and the count."""
    jobs, bands = np.array([r["job"] for r in rows]), np.array([_value(r, "age_band") for r in rows])
    out = {}
    for job in sorted(set(jobs.tolist())):
        out[job] = {}
        for band in (*AGE_BANDS, "none"):
            picked = (jobs == job) & (bands == band)
            if picked.any():
                out[job][band] = {**s.pooled(score, picked), "n": int(picked.sum())}
    return out


def _catastrophes(s: _Scripts, rows: list[Row], said: np.ndarray) -> dict:
    """`catastrophe_rate`: the catastrophes over the harm rows answered (the family's kind is harm, the answer is not
    `unknown`), with the counts, overall and for each job that has a harm row."""
    harm = np.array([kind_of(r["family"]) == "harm" for r in rows], dtype=float)
    answered = harm * (1 - said)
    hit = np.array([bool(r["catastrophe"]) for r in rows], dtype=float) * answered
    jobs = np.array([r["job"] for r in rows])

    def stat(here: np.ndarray) -> dict:
        found = s.ratio(hit * here, answered * here)
        return {"rate": found["mean"], "ci": found["ci"], "n": int((hit * here).sum()),
                "of": int((answered * here).sum())}

    return {**stat(np.ones(len(rows))),
            "jobs": {job: stat((jobs == job).astype(float)) for job in sorted(set(jobs.tolist()))
                     if (harm * (jobs == job)).any()}}


def _cells(s: _Scripts, score: np.ndarray, name: str) -> dict[str, dict]:
    """The cells of the ability `name`, one per scope it has rows in: the mean over all the cell's rows and over the
    counted ones, their intervals, the rows (`n`) and the counted ones, whether none counts (`diagnostic`) and the
    cell's `weight` in the headline: its column's share of the columns' weights, times its share of its own column's
    cell weights (0 for a cell that counts in none)."""
    in_column = {sc: sum(x for c, _, _, x in s.groups if c == sc) for sc in s.columns}
    held = {(sc, ab): x for sc, ab, _, x in s.groups}
    out = {}
    for scope in SCOPES:
        mine = (s.ability == name) & (s.scope == scope)
        if not mine.any():
            continue
        counted, kept = mine & s.counted, s.pooled(score, mine * s.weight)
        out[scope] = {**s.pooled(score, mine), "n": int(mine.sum()), "counted": int(counted.sum()),
                      "diagnostic": not counted.any(), "counted_mean": kept["mean"], "counted_ci": kept["ci"],
                      "weight": held[scope, name] / in_column[scope] * s.columns[scope] / sum(s.columns.values())
                      if counted.any() else 0.0}
    return out


def aggregate(results: Iterable[Row], n_boot: int = N_BOOT, seed: int = 0, *, thin: Thin = frozenset(),
              thin_ignored: bool = False) -> dict:
    """Report one system's rows. No rows give `n` 0 and null means. `thin` is the (job, ability, scope) cells with fewer
    than `MIN_ABILITY_N` questions in the whole dataset, each with its weight below 1 (`ability_counts.thin_rule`; a
    set of cells weighs 0 each, counted in no mean). `thin_ignored` records that the dataset is too small for the rule,
    which the caller then did not apply (it passes no `thin`). Rows that are all diagnostic give null means and no
    columns.

    Returns `{n, scripts, cases, headline, columns, abilities_mean, pooled, thin, thin_ignored, above_floor, by, counts,
    confidently_wrong_rate, abstention, job_age_band, catastrophe_rate}`. Every mean is of the reported `score`:
    - `headline` = `{mean, ci}`, the mean of the `now` and `memory` columns (one column when the rows hold one scope),
      each the mean of its cells (ability, scope) by their weights, each cell pooled over its own rows by theirs; the
      interval from one resample of scripts for all of them;
    - `columns` = `{scope: {mean, ci, n}}`, those two columns, `n` the rows counted in each;
    - `abilities_mean` and `pooled` = `{mean, ci}`, the mean of all the counted cells by their weights, and the counted
      rows pooled (the secondary numbers published beside the headline);
    - `thin` = the (job, ability, scope) cells of the given weights that this score applied, each as a list ending in
      its weight (`thin_from` reads them back), and `thin_ignored` as given;
    - `above_floor` = `{mean, ci}`, the headline less the unknown share (the score of answering `unknown` throughout),
      weighted the same way and resampled by script;
    - `by[dim][str(value)]` = `{mean, ci, n}`, pooled within each group of the splits `DIMS`: `scope`, `own_path`
      (`reference == "path_log"`: "True" or "False"), `family`, `ability`, `when`, `length`, `layout`, `robot`, `job`,
      `critical`, `slice` (answerable or unknown), `relevance` (`relevance`: critical, situational or unknown) and
      `answer_class` (`none` where the gold declares none); `kind` is the family's kind (harm, help, situation or
      reuse), pooled within itself over every one of its rows beside `ability`, and its entry also holds `counted`
      (its rows in the means) and `diagnostic` (none counted). An ability's entry also holds `scope` (the scope of its
      rows, `now/memory` if both), `counted` (its rows in the means), `diagnostic` (none counted:
      `abilities.DIAGNOSTIC`, or a weight of 0 on every job and column it was asked on), `counted_mean`,
      `counted_ci` (pooled over its counted rows alone, by their weights; `mean` pools every row of the ability) and
      `cells`, `{scope: {mean, ci, n, counted, diagnostic, weight, counted_mean, counted_ci}}`, the same for each
      column the ability has rows in, `weight` the cell's share of the headline (0 where none of its rows count);
    - `counts` = correct, partial, wrong (answered with s = 0, malformed included) and abstained totals;
    - `confidently_wrong_rate` = `{rate, ci}`, wrong over all questions;
    - `abstention` = `{precision, ci_precision, recall, ci_recall, answered_unknown, unknown_gold}`: of `unknown`
      answers, the share on unknown-gold questions, and of unknown-gold questions, the share answered `unknown`
      (`None` when the denominator is 0);
    - `by.age_band[band]` = `{mean, ci, n}`, the splits' pooled reported score by the evidence age's band
      (`records.AGE_BANDS`; `none` for a row with no band: an unknown gold, or a gold of an earlier build) and
      `job_age_band[job][band]` the same within each job;
    - `catastrophe_rate` = `{rate, ci, n, of, jobs}`: the rows whose answer is a catastrophe (`n`) over the harm rows
      answered (`of`: the family's kind is harm and the answer is not `unknown`), and `jobs[job]` = `{rate, ci, n, of}`
      for each job with a harm row.
    """
    rows = _check(results)
    if not rows:
        return _empty()
    s = _Scripts(rows, n_boot, seed, thin)
    score = np.array([r["score"] for r in rows])
    by = {}
    for dim in DIMS:
        labels = [_value(r, dim) for r in rows]
        by[dim] = {}
        for label in sorted(set(labels)):
            picked = np.array([x == label for x in labels])
            by[dim][label] = {**s.pooled(score, picked), "n": int(picked.sum())}
    for label, entry in by["ability"].items():
        mine = s.ability == label
        entry |= {"scope": "/".join(sorted(set(s.scope[mine]))), "counted": int((mine & s.counted).sum())}
        entry["diagnostic"] = entry["counted"] == 0
        kept = s.pooled(score, mine * s.weight)
        entry |= {"counted_mean": kept["mean"], "counted_ci": kept["ci"], "cells": _cells(s, score, label)}
    kinds = np.array([kind_of(r["family"]) for r in rows])
    for label, entry in by["kind"].items():   # pools every row, as `family` does; says how many of them the means count
        entry["counted"] = int(((kinds == label) & s.counted).sum())
        entry["diagnostic"] = entry["counted"] == 0

    def flag(test) -> np.ndarray:
        return np.array([bool(test(r)) for r in rows], dtype=float)

    wrong = s.ratio(flag(lambda r: r["outcome"] == "wrong"), np.ones(len(rows)))
    hit = flag(lambda r: r["slice"] == "unknown" and r["outcome"] == "correct")
    said = hit + flag(lambda r: r["outcome"] == "abstained")
    truth = flag(lambda r: r["slice"] == "unknown")
    precision, recall = s.ratio(hit, said), s.ratio(hit, truth)
    means, floor = s.means(score), s.means(score - truth)
    columns = {scope: {**s.stat(means[scope]), "n": s.column_n(scope)} for scope in SCOPES if s.column_n(scope)}
    jobs = {r["job"] for r in rows}
    applied = sorted(([*cell, w] for cell, w in thin_weights(thin).items() if cell[0] in jobs),
                     key=lambda entry: [str(x) for x in entry])
    return {
        "n": len(rows), "scripts": len(s.scripts), "cases": len({r["case"] for r in rows}),
        "headline": s.stat(means["headline"]), "columns": columns, "abilities_mean": s.stat(means["abilities_mean"]),
        "pooled": s.stat(means["pooled"]), "thin": applied, "thin_ignored": thin_ignored,
        "above_floor": s.stat(floor["headline"]), "by": by, "counts": _count(rows),
        "confidently_wrong_rate": {"rate": wrong["mean"], "ci": wrong["ci"]},
        "job_age_band": _ages(s, rows, score), "catastrophe_rate": _catastrophes(s, rows, said),
        "abstention": {
            "precision": precision["mean"], "ci_precision": precision["ci"],
            "recall": recall["mean"], "ci_recall": recall["ci"],
            "answered_unknown": int(said.sum()), "unknown_gold": int(truth.sum()),
        },
    }


def paired_difference(results_a: Iterable[Row], results_b: Iterable[Row], *, where: dict | None = None,
                      headline: bool = False, counted: bool = False, thin: Thin = frozenset(), n_boot: int = N_BOOT,
                      seed: int = 0, key: str = "score") -> dict:
    """System A minus system B on the score (the reported one, or with `key="score_gate"` the penalised one the gates
    read: `SCORE_KEYS`), over the questions both answered.

    Rows pair on (case, qid). The statistic is the pooled mean of the differences (a ratio of sums), with a 95% CI
    from a bootstrap over scripts; the gates and every split read this. With `headline=True` it is the headline's
    statistic of the differences (`thin` as for `aggregate`), which is A's headline less B's over the shared
    questions; with `counted=True` it pools only the rows the means count (a thin cell's rows and a diagnostic ability's
    left out). Returns `{mean, ci, n, scripts}`; positive means A scored higher. `where` keeps only
    the rows of A whose keys (`ROW_KEYS`, or `relevance` or `ability`) equal its values, such as
    `{"family": "object_where"}` or `{"relevance": "critical"}`. Two sets that share no question raise ValueError.
    """
    where, key = where or {}, _key(key)
    if unknown := [k for k in where if k not in ROW_KEYS and k not in DERIVED]:
        raise ValueError(f"where names {unknown}, which are not among the row keys {list(ROW_KEYS)} or {list(DERIVED)}")
    b = {(r["case"], r["qid"]): r for r in _check(results_b)}
    rows = []
    for a in _check(results_a):
        other = b.get((a["case"], a["qid"]))
        if other is None or any((_value(a, k) if k in DERIVED else a[k]) != v for k, v in where.items()):
            continue
        if (a["script"], a["family"]) != (other["script"], other["family"]):
            raise ValueError(f"qid {a['qid']!r} of case {a['case']!r} has a different script or family in the two sets")
        rows.append({**a, "d": a[key] - other[key]})
    if not rows:
        raise ValueError("the two result sets share no questions (after the `where` filter)")
    s = _Scripts(rows, n_boot, seed, thin)
    d = np.array([r["d"] for r in rows])
    stat = s.headline(d) if headline else s.pooled(d, s.weight) if counted else s.pooled(d)
    return {**stat, "n": int(s.counted.sum()) if counted else len(rows), "scripts": len(s.scripts)}


def gap_closed(S: float, floor: float, oracle: float) -> float:
    """(S - floor) / (oracle - floor): how much of the distance from the floor to the oracle a system covers."""
    if oracle <= floor:
        raise ValueError(f"oracle ({oracle}) must exceed floor ({floor})")
    return (S - floor) / (oracle - floor)
