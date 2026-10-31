"""Scoring one answer (V0 §8, plan §2.6): credit s, the penalised score, the outcome class, and the result row.

A gold record is a `records.GOLD_KEYS` row: `{qid, family, gold, slice, credit, region, tolerances, metric, scope,
reference, critical, ...}`. This module reads:
- `gold`: an answer-shaped value (`answers.parse_gold`); `{"type": "unknown"}` exactly when `slice` is `unknown`;
- `credit`: how the answer is scored (`records.CREDIT_FITS` says which credit scores which gold type): `ladder`
  (position, offset, number, time), `mra` (count), `exact` (yesno), `f1` (set, positions, cells), `route`, `region`
  (a position inside the room polygon `region`, at least `tolerances.erode_m` from its walls) and `relative` (a
  number within a share of the gold's value) and `graded` (an offset, number or position), `graded_set`
  (a list of places), `regret` (a heading) and `route` (a route of points, and since 0.11.0 a heading to a remembered
  thing: `score_graded.route_credit`), the four quarters of `credit_kernel` read by `score_graded` and
  `score_set`. A record without the key is scored by its gold's type: the ladder, `mra`,
  `exact`, `f1` or `route`. A credit that does not score the gold's type raises ValueError;
- `metric`: `xy` (default) or `xyz`, the distance a `ladder` position answer is judged by; an `offset` answer is
  judged by the 2-D error `hypot(d_ahead, d_left)` in the camera's frame, against `tolerances.ladder` in metres;
- `tolerances`: `{"ladder": [...]}` replaces the default thresholds for position, number (metres) or time (seconds);
  `{"match_m": 0.5}` is the matching radius of a `set` or `positions` answer (positions are matched in the floor plane,
  whatever `metric` says); `{"erode_m": 0.2}` is a `region` credit's margin; a `graded`, `graded_set` or `regret` row
  holds its keys in full (`score_graded`). A threshold is met when the error is at
  most the threshold, up to 1e-9. A gold row written by the compiler holds its tolerance in full, so the defaults here
  are reached only by a record without a `credit`.

Credit is 0 for a malformed answer (a `failed` answer included: it parses as malformed under every type), an answer of
another type than the gold's, and `unknown` against an answerable gold. Aggregation lives in `score_stats`,
re-exported here.

Two scores stand apart (0.10.0, FD-questions-0.10 §4, review B4). `penalised` (`2s - 1`, `unknown` 0) is the gate score:
every gate, `knowable` and the evidence verdict read it, as the row's `score_gate`, so no harm weight ever moves a
validity verdict. `reported` is `penalised` except for a catastrophe (`score_catastrophe`), which scores minus its
weight; it is the row's `score`, what the reports, the headline and the reference rows read.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from stbench.answers import (
    Answer,
    Cells,
    Count,
    Heading,
    Number,
    Offset,
    Position,
    Positions,
    Route,
    SetAnswer,
    Time,
    Unknown,
    YesNo,
    answer_type_of,
    is_polygon,
    parse_answer,
    parse_gold,
)
from stbench.records import CREDIT_FITS, WHEN
from stbench.score_catastrophe import weight as catastrophe_weight
from stbench.score_graded import graded_credit, graded_set_credit, regret_credit
from stbench.score_graded import route_credit as heading_route_credit
from stbench.score_set import cells_f1, positions_f1, region_credit, route_credit, set_f1
from stbench.score_stats import INFO_KEYS, aggregate, gap_closed, paired_difference

__all__ = ["aggregate", "catastrophe", "credit", "gap_closed", "outcome", "paired_difference", "penalised",
           "reported", "row"]

EPS = 1e-9
DISTANCE_LADDER = (0.25, 0.5, 1.0, 2.0)
TIME_LADDER = (3.0, 10.0, 30.0, 90.0)
MATCH_M = 0.5
MRA_PERCENT = range(50, 100, 5)
# The credit a gold record without a `credit` key gets, by its gold's type.
DEFAULT_CREDIT = {"position": "ladder", "offset": "ladder", "number": "ladder", "time": "ladder", "count": "mra",
                  "yesno": "exact", "set": "f1", "positions": "f1", "cells": "f1", "route": "route"}


def _ladder(record: dict, default: tuple[float, ...]) -> tuple[float, ...]:
    ladder = (record.get("tolerances") or {}).get("ladder", default)
    if not ladder or any(isinstance(x, bool) or not isinstance(x, int | float) or x <= 0 for x in ladder):
        raise ValueError(f"gold {record.get('qid')!r}: tolerances.ladder must be a non-empty list of positive numbers")
    return tuple(ladder)


def _match_m(record: dict) -> float:
    return (record.get("tolerances") or {}).get("match_m", MATCH_M)


def _share_met(error: float, ladder: tuple[float, ...]) -> float:
    return sum(error <= th + EPS for th in ladder) / len(ladder)


def _mra(estimate: int, truth: int) -> float:
    """VSI-Bench mean relative accuracy: [|est - y| / y < 1 - theta] over theta = 0.50, 0.55, ..., 0.95, in integers."""
    if truth == 0:
        return 1.0 if estimate == 0 else 0.0
    return sum(abs(estimate - truth) * 100 < (100 - p) * truth for p in MRA_PERCENT) / len(MRA_PERCENT)


def _position_error(answer: Position, gold: Position, metric: str) -> float | None:
    if metric == "xyz":
        if not (answer.has_z and gold.has_z):
            return None
        return math.dist(answer.coords, gold.coords)
    return math.dist(answer.xy, gold.xy)


def _ladder_credit(answer: Answer, gold: Answer, record: dict) -> float:
    match gold:
        case Position():
            error = _position_error(answer, gold, record.get("metric") or "xy")
            return 0.0 if error is None else _share_met(error, _ladder(record, DISTANCE_LADDER))
        case Offset():
            error = math.hypot(answer.ahead_m - gold.ahead_m, answer.left_m - gold.left_m)
            return _share_met(error, _ladder(record, DISTANCE_LADDER))
        case Number():
            return _share_met(abs(answer.value - gold.value), _ladder(record, DISTANCE_LADDER))
        case Time():
            return _share_met(abs((answer.clock - gold.clock).total_seconds()), _ladder(record, TIME_LADDER))
    raise ValueError(f"unscorable gold type {answer_type_of(gold)!r}")


def _mra_credit(answer: Count, gold: Count, record: dict) -> float:
    return _mra(answer.value, gold.value)


def _exact_credit(answer: YesNo, gold: YesNo, record: dict) -> float:
    return 1.0 if answer.value == gold.value else 0.0


def _f1_credit(answer: Answer, gold: Answer, record: dict) -> float:
    match gold:
        case SetAnswer():
            return set_f1(answer, gold, _match_m(record))
        case Positions():
            return positions_f1(answer, gold, _match_m(record))
        case Cells():
            return cells_f1(answer, gold)
    raise ValueError(f"unscorable gold type {answer_type_of(gold)!r}")


def _route_credit(answer: Answer, gold: Answer, record: dict) -> float:
    if isinstance(gold, Heading):   # the way to a remembered thing (`way_to`): the outcome of the heading
        return heading_route_credit(answer, gold, record)
    # a parsed route gold handed in as an answer has no points: it is not an answer
    return route_credit(answer, gold) if isinstance(answer, Route) else 0.0


def _region_credit(answer: Position, gold: Position, record: dict) -> float:
    region, erode = record.get("region"), (record.get("tolerances") or {}).get("erode_m")
    if not is_polygon(region) or isinstance(erode, bool) or not isinstance(erode, int | float) or erode < 0:
        raise ValueError(f"gold {record.get('qid')!r}: credit region needs a polygon `region` and tolerances.erode_m "
                         f"of at least 0, got {region!r} and {erode!r}")
    return region_credit(answer.xy, [tuple(p) for p in region], erode)


def _relative_credit(answer: Number, gold: Number, record: dict) -> float:
    """VSI-Bench mean relative accuracy of a number, in floats: [|e - y| * 100 < (100 - p) * y] over p = 50, 55, ...,
    95. Strict, so an error exactly on a bound is not met; a negative estimate is at least y away, so it scores 0."""
    if not gold.value > 0:
        raise ValueError(f"gold {record.get('qid')!r}: credit relative needs a positive gold value, got {gold.value}")
    error = abs(answer.value - gold.value) * 100
    return sum(error < (100 - p) * gold.value for p in MRA_PERCENT) / len(MRA_PERCENT)


_CREDITS: dict[str, Callable[[Answer, Answer, dict], float]] = {
    "ladder": _ladder_credit, "mra": _mra_credit, "exact": _exact_credit, "f1": _f1_credit, "route": _route_credit,
    "region": _region_credit, "relative": _relative_credit, "graded": graded_credit,
    "graded_set": graded_set_credit, "regret": regret_credit,
}


def _coerce(answer: object) -> Answer:
    return answer if isinstance(answer, Answer) else parse_answer(answer)


def _gold(record: dict) -> Answer:
    gold = parse_gold(record["gold"])
    if isinstance(gold, Unknown) != (record["slice"] == "unknown"):
        raise ValueError(f"gold {record.get('qid')!r}: slice {record['slice']!r} disagrees with the gold value")
    return gold


def credit(answer: object, gold_record: dict) -> float:
    """Credit s in [0, 1] for an answer (parsed, or a raw JSON object) against a gold record, by the record's `credit`
    (its gold's default when the record has none)."""
    answer, gold = _coerce(answer), _gold(gold_record)
    if isinstance(gold, Unknown):
        return 1.0 if isinstance(answer, Unknown) else 0.0
    kind = answer_type_of(gold)
    name = gold_record.get("credit") or DEFAULT_CREDIT[kind]
    if name not in CREDIT_FITS:
        raise ValueError(f"gold {gold_record.get('qid')!r}: credit must be one of {list(CREDIT_FITS)}, got {name!r}")
    if kind not in CREDIT_FITS[name]:
        raise ValueError(f"gold {gold_record.get('qid')!r}: credit {name!r} scores {list(CREDIT_FITS[name])} golds, "
                         f"not {kind!r}")
    if answer_type_of(answer) != kind:
        return 0.0
    return _CREDITS[name](answer, gold, gold_record)


def _k(t: float) -> float:
    if not 0 <= t < 1:
        raise ValueError(f"t must be in [0, 1), got {t}")
    return t / (1 - t)


def penalised(answer: object, gold_record: dict, t: float = 0.5) -> float:
    """The penalised score: s - (1-s)k for an answer on an answerable gold, 0 for `unknown` there, 1 for correct
    `unknown`, -k for an answer on an unknown gold; k = t / (1 - t)."""
    k, answer = _k(t), _coerce(answer)
    if gold_record["slice"] == "unknown":
        _gold(gold_record)
        return 1.0 if isinstance(answer, Unknown) else -k
    if isinstance(answer, Unknown):
        _gold(gold_record)
        return 0.0
    s = credit(answer, gold_record)
    return s - (1 - s) * k


def catastrophe(answer: object, gold_record: dict) -> int:
    """The weight of the catastrophe the answer is against the gold record (`score_catastrophe`), 0 when it is none."""
    return catastrophe_weight(_coerce(answer), _gold(gold_record), gold_record)


def _charged(weight: int, gate: float) -> float:
    """The reported score from a catastrophe's `weight` (0: none) and the penalised score `gate`: the one place that
    says a catastrophe replaces the score and does not add to it."""
    return -float(weight) if weight else gate


def reported(answer: object, gold_record: dict, t: float = 0.5) -> float:
    """The reported score: minus the catastrophe's weight for an answer that is one (`catastrophe`), else `penalised`.
    A row without a `catastrophe` tolerance, an `unknown` answer and an answer to an `unknown` gold score as
    `penalised` does."""
    answer = _coerce(answer)
    return _charged(catastrophe(answer, gold_record), penalised(answer, gold_record, t))


def outcome(answer: object, gold_record: dict) -> str:
    """`correct` (s = 1, or `unknown` on an unknown gold), `abstained` (`unknown` on an answerable gold), `wrong`
    (answered with s = 0, malformed included: a confident error) or `partial`."""
    answer, s = _coerce(answer), credit(answer, gold_record)
    if isinstance(answer, Unknown):
        return "correct" if gold_record["slice"] == "unknown" else "abstained"
    return "correct" if s >= 1 - EPS else "wrong" if s <= EPS else "partial"


def row(answer: object, gold: dict, *, when: str, info: dict) -> dict:
    """One result row, the input of `aggregate` and the gates (`score_stats.ROW_KEYS`): the question's `when` and the
    case's `info` (`score_stats.INFO_KEYS`: case, script, layout, robot, job, length), beside what the gold row holds
    (qid, family, scope, reference, critical, slice, answer_class, range_m, age_band), the credit `s`, the reported
    `score` and the penalised `score_gate` at t = 0.5, whether the answer is a `catastrophe` and the outcome."""
    if when not in WHEN:
        raise ValueError(f"when must be one of {list(WHEN)}, got {when!r}")
    if missing := [k for k in INFO_KEYS if k not in info]:
        raise ValueError(f"info lacks {missing}")
    answer = _coerce(answer)
    weight = catastrophe(answer, gold)
    gate = penalised(answer, gold)
    return {
        "case": info["case"], "script": info["script"], "qid": gold["qid"], "family": gold["family"], "when": when,
        "scope": gold["scope"], "reference": gold["reference"], "critical": gold["critical"], "slice": gold["slice"],
        "answer_class": gold.get("answer_class"),   # null for a gold of a compile before 0.6.0
        "range_m": gold.get("range_m"),             # the slant range a relative ladder was built from, else null
        "age_band": gold.get("age_band"),           # the evidence age's band; null on an unknown row and before 0.10.0
        "s": credit(answer, gold), "score": _charged(weight, gate), "score_gate": gate,
        "catastrophe": bool(weight), "outcome": outcome(answer, gold),
        "layout": info["layout"], "robot": info["robot"], "job": info["job"], "length": info["length"],
    }
