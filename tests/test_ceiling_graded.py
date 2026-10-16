"""The ceiling of the graded credit on the compiled rows of every existing family (0.9.0): the gold answered as itself
scores 1, whatever resolution and need its row carries. Rows are made as the compiler makes them
(`questions.records`, which refuses an invalid row) from the same fixtures the family tests use."""

from __future__ import annotations

from dataclasses import replace

import pytest

import family_kit as fk
import person_kit as pk
import task_kit as tk
from ego_kit import EGO_JOB, K, put, scene
from space_kit import things_run
from stbench import questions, score
from stbench.questions_ego import ego_mem, ego_now
from stbench.questions_people import person_mem
from stbench.questions_where import pools
from task_kit import BIG, cand, y
from test_closest_person import case_with, only
from test_closest_person import scene as closest_scene


def ego_cands() -> list:
    """`ego_now` and `ego_mem` rows: three bags at three ranges, one seen only before the look window, the last on a job
    whose clearance is under its pixel (a resolution larger than the need), and an unknown `ego_mem` row."""
    s = scene()
    put(s, "T", "bag", "red", (3.0, 4.13), [K], pts=[(2.9, 3.93, 0.0), (3.1, 3.93, 0.0), (3.0, 4.33, 0.0)])
    put(s, "N", "bag", "blue", (3.0, 2.1), [K], pts=[(2.9, 2.0, 0.0), (3.1, 2.0, 0.0), (3.0, 2.2, 0.0)])
    put(s, "A", "bag", "green", (2.0, 4.6), [2], pts=[(1.9, 4.5, 0.0), (2.1, 4.5, 0.0), (2.0, 4.7, 0.0)])
    put(s, "U", "bag", "black", (1.0, 1.0), [])
    found = ego_now(fk.world(s, job=EGO_JOB), K).answerable
    moment = ego_mem(fk.world(s, job=EGO_JOB), K)
    tight = replace(EGO_JOB, clearance=0.1, screen_fallback=frozenset({"ego_now"}))
    far = scene()   # a bag whose nearest points are (2.9, 3.93), (3.0, 3.93) and (3.1, 3.93): a pixel of 0.26 m
    put(far, "L", "bag", "red", (3.0, 4.13), [K], pts=[(2.9, 3.93, 0.0), (3.0, 3.93, 0.0), (3.1, 3.93, 0.0),
                                                         (3.0, 4.33, 0.0)])
    loose = ego_now(fk.world(far, job=tight), K).answerable
    assert len(loose) == 1
    return [*found, *moment.answerable, *moment.unknown, *loose]


def people_cands() -> list:
    """A `person_mem` row, an unknown one, a `closest_person` row and a "no one" one."""
    s = pk.scene()
    pk.put_person(s, "A", (2.0, 3.63), [2])
    seen = person_mem(pk.cut(s, K, pk.one_person()), K).answerable
    nobody = person_mem(pk.cut(pk.scene(), K, pk.one_person()), K).unknown
    person = closest_scene()
    pk.put_person(person, "B", [(0, 10.0, (2.0, 3.63, 0.0)), (10.0, None, (2.0, 2.63, 0.0))], [2, K])
    nearest = only(pk.world(person))[1]
    none = only(pk.world(closest_scene(), cid=case_with(True, tasks=("0",))))[1]
    return [*seen, *nobody, nearest, none]


def place_cands() -> list:
    """`object_where` rows, answerable and unknown, and a `task_things` row with two items and a don't-care place."""
    answerable, unknown = pools(fk.world(things_run()))
    s = tk.script()
    tk.put(s, "A", (2, y(0.5)))
    tk.put(s, "B", (5, y(1.0)))
    tk.put(s, "E", (14, y(0.3, 1.0)), BIG, at=(5,))
    return [*answerable, *unknown[:3], cand(s)]


def answer_as(gold: dict) -> dict:
    """The answer that is the gold: the same object, a list without its don't-care places."""
    return {"type": "positions", "items": gold["items"]} if gold["type"] == "positions" else dict(gold)


def test_the_gold_answered_as_itself_scores_one_on_every_row_of_every_family():
    """Contract (the ceiling gate, brief 3.8: an ideal reader reaches full credit): for every existing family, the row
    the compiler writes scores 1 when the answer is its gold, and `unknown` scores 1 on an unknown gold; a row also
    passes `records.check_gold` (the compile refuses an invalid row). Rows include a resolution larger than the need (a
    job with C = 0.1 m and a pixel of 0.26 m), the floor resolution, a "no one" and a list with don't-care places.
    Regression caught: a resolution under the gold's own quantum or a threshold above the gold that makes the exact
    answer lose a quarter, a list item matched to nothing, or a row the scorer cannot read."""
    cands = [*ego_cands(), *people_cands(), *place_cands()]
    _, golds = questions.records(cands)
    assert {g["family"] for g in golds} == {"ego_now", "ego_mem", "person_mem", "closest_person", "object_where",
                                            "task_things"}
    assert {g["credit"] for g in golds} == {"graded", "graded_set"}
    assert {g["slice"] for g in golds} == {"answerable", "unknown"}
    for g in golds:
        assert score.credit(answer_as(g["gold"]), g) == pytest.approx(1.0, abs=1e-12), (g["family"], g["gold"])
