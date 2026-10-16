"""The evidence reading of the shortcut-type gates (FD-0.5.0-control §8): a system's gain over `all_unknown` is held
against the gate's margin only when it is shown, in the whole group, in each length tier and in each slice.

The point verdict of `memory_needed`, `shortcuts` and `time_needed` fails a system whose pooled mean is more than the
margin over `all_unknown`. On a coin of about a hundred questions that fails a shortcut for chance: the drone's `blind`
scored +0.104 on `zone_clear` (2026-10-08) with an interval that reached below 0. And a pooled mean can hide a gain:
the helper's short `distance_between` was answered +0.375 over `all_unknown` on its answerable questions, a quarter of
the rows, which pools to under the margin. So each system S is read in cells, the rows of the group (the rows of its
`memory` scope for `memory_needed`):

| cell               | rows                                              |
| ------------------ | ------------------------------------------------- |
| `("all",)`         | all of them                                       |
| `("length", L)`    | the cases of each length present                  |
| `("slice", s)`     | the answerable rows, and the unknown ones, if any |

Every score is the row's `score_gate`, the penalised score (`score_stats.GATE`), never its reported `score`.

In each cell, d is the paired difference of S and `all_unknown` (`score_stats.paired_difference`: pairs on (case,
question), the interval a bootstrap over scripts, N_BOOT resamples, seed 0). S **counts** in the cell when d's mean is
over the gate's margin and the interval is above 0: the gain is more than the margin and not chance. The gate's
evidence verdict fails when some system counts in some cell. A cell of fewer than two scripts has no interval, so it
cannot count; `read` says which cells those are. On the unknown slice `all_unknown` scores 1, so nothing counts there:
the slice reading bites on the answerable slice.
"""

from __future__ import annotations

from collections.abc import Mapping

from .score_stats import GATE, N_BOOT, SLICES, paired_difference, pooled

ALL = ("all",)
LENGTHS = ("short", "medium", "long")   # `case.LENGTHS`' order; a gate module may not import `case` (the fences)
Cell = tuple[str, ...]


def cells_of(rows: list[dict]) -> list[Cell]:
    """The cells that `rows` has something in, in order: all, each length (`case.LENGTHS` order), each slice."""
    lengths, slices = {r["length"] for r in rows}, {r["slice"] for r in rows}
    order = [*LENGTHS, *sorted(lengths - set(LENGTHS))]
    return [ALL, *(("length", n) for n in order if n in lengths), *(("slice", s) for s in SLICES if s in slices)]


def label(cell: Cell) -> str:
    """A cell in words: `all rows`, `length short`, `slice answerable`."""
    return "all rows" if tuple(cell) == ALL else " ".join(cell)


def sentence(found: dict) -> str:
    """One of `read`'s cells in words: `blind beats all_unknown by +0.300 (95% interval +0.120 to +0.480)`."""
    ci = found["ci"]
    interval = "" if ci is None else f" (95% interval {ci[0]:+.3f} to {ci[1]:+.3f})"
    return f"{found['worst']} beats all_unknown by {found['mean']:+.3f}{interval}"


def _in(cell: Cell, row: dict) -> bool:
    return cell == ALL or row[cell[0]] == cell[1]


def read(rows: Mapping[str, list[dict]], floor: list[dict], *, bar: float, scope: str | None = None,
         n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Read each system of `rows` (name -> its result rows) against `floor` (the rows of `all_unknown`) in every cell of
    the floor's rows, only those of `scope` when given. Returns
    - `ok`: no system counts in any cell, that is, none has a paired gain over `bar` (the gate's margin and tolerance)
      with an interval above 0;
    - `cells`: for each cell with rows, `{cell, worst, mean, ci, n, scripts, counts}`, the system that counts there
      (the largest gain among those that count) or, when none does, the one with the largest gain; `mean` and `ci` are
      its paired gain, `n` the questions and `scripts` the scripts they pair on, `ci` None under two scripts;
    - `pooled`: `{system: {mean, ci}}`, each system's own score pooled over the rows it was read on;
    - `note`: None, or the cells whose worst system has under two scripts, which cannot count."""
    kept = (lambda rs: [r for r in rs if r["scope"] == scope]) if scope else list
    floor = kept(floor)
    asked = {(r["case"], r["qid"]) for r in floor}
    mine = {name: kept(rs) for name, rs in rows.items()}
    mine = {name: rs for name, rs in mine.items() if rs}
    found = []
    for cell in cells_of(floor):
        reads = []
        for name, rs in mine.items():
            picked = [r for r in rs if _in(cell, r) and (r["case"], r["qid"]) in asked]
            if picked:
                d = paired_difference(picked, floor, n_boot=n_boot, seed=seed, key=GATE)
                reads.append({"worst": name, "mean": d["mean"], "ci": d["ci"], "n": d["n"], "scripts": d["scripts"],
                              "counts": d["mean"] > bar and d["ci"] is not None and d["ci"][0] > 0})
        if reads:
            found.append({"cell": list(cell), **max(reads, key=lambda x: (x["counts"], x["mean"]))})
    short = [label(c["cell"]) for c in found if c["scripts"] < 2]
    note = f"{', '.join(short)}: fewer than two scripts, so no interval and the cell cannot count" if short else None
    own = {name: pooled(rs, n_boot=n_boot, seed=seed, key=GATE) for name, rs in mine.items()}
    return {"ok": not any(c["counts"] for c in found), "cells": found, "note": note,
            "pooled": {name: {"mean": p["mean"], "ci": p["ci"]} for name, p in own.items()}}
