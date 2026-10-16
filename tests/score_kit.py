"""Hand-made answers, gold records and result rows for the tests that score, aggregate or gate: no compiler runs, so
every row is what the test says."""

from __future__ import annotations


def places(*points) -> dict:
    """A `positions` gold or answer value from (x, y) points."""
    return {"type": "positions", "items": [{"xy": [x, y]} for x, y in points]}


# the default gold
GOLD = places((0, 0), (3, 0))


def gold_row(qid: str, *, family: str = "inventory", gold: dict | None = None, slice: str = "answerable",
             twin: str | None = None) -> dict:
    """A Perceive gold row; the default gold is `GOLD`."""
    return {"qid": qid, "family": family, "gold": gold or GOLD, "slice": slice, "evidence_k": [3],
            "entities": [], "tolerances": {}, "metric": "xy", "retention": False, "twin": twin}


def rec(gold, *, slice="answerable", metric="xy", tolerances=None, family="where_last", qid="q1", **extra):
    """A gold record as the scorer reads one (a record with no `credit` key, so its gold's type decides); `extra`
    adds keys such as `credit` and `region`."""
    return {"qid": qid, "family": family, "gold": gold, "slice": slice, "evidence_k": [3], "entities": [],
            "tolerances": tolerances or {}, "metric": metric, **extra}


def pos(x, y, z=None):
    return {"type": "position", **({"xyz": [x, y, z]} if z is not None else {"xy": [x, y]})}


def off(ahead, left):
    """An `offset` answer or gold value: metres ahead of the camera and to its left."""
    return {"type": "offset", "ahead_m": ahead, "left_m": left}


def clock(text):
    return {"type": "time", "clock": text}


def num(v):
    return {"type": "number", "value": v}


def count(v):
    return {"type": "count", "value": v}


def items(*its):
    return {"type": "set", "items": [{"xy": [x, y], "change": c} for x, y, c in its]}


def cells(*cs, dont_care=()):
    """A `cells` answer, or a gold with don't-care cells, from (i, j) pairs."""
    return {"type": "cells", "items": [list(c) for c in cs],
            **({"dont_care": [list(c) for c in dont_care]} if dont_care else {})}


UNKNOWN = {"type": "unknown"}
ORIGIN = rec(pos(0, 0))
SQUARE = [[0, 0], [4, 0], [4, 4], [0, 4]]   # a 4 m room with a corner at the origin


def result_row(case, qid, score, outcome_, *, script=None, family="where_last", when="end", scope="memory",
               reference=None, critical=False, slice_="answerable", layout="apartment_a", robot="vacuum",
               job="vacuum/1", length="short", answer_class=None, range_m=None, score_gate=None, catastrophe=False,
               age_band=None):
    """A result row (`score.row`'s shape) with the credit `s` its outcome implies; `script` defaults to the case.
    `score` is the reported score; `score_gate` (the penalised one) is the same unless given; `catastrophe` flags the
    row (the caller gives it the lowered reported score)."""
    s = {"correct": 1.0, "partial": 0.5, "wrong": 0.0, "abstained": 0.0}[outcome_]
    return {"case": case, "script": script or case, "qid": qid, "family": family, "when": when, "scope": scope,
            "reference": reference, "critical": critical, "slice": slice_, "answer_class": answer_class,
            "range_m": range_m, "age_band": age_band, "s": s, "score": score,
            "score_gate": score if score_gate is None else score_gate, "catastrophe": catastrophe,
            "outcome": outcome_, "layout": layout, "robot": robot, "job": job, "length": length}
