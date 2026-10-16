"""`answer_class` and `range_m` on the rows (0.6.0): written by the compiler from the candidate, read as null from older
rows, and keyed by the prior systems; and the `task_things` option reaching the table systems."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from baseline_kit import offset
from family_kit import compiled, fake_cand, fake_spec, multi_run, register, things_by_task
from score_kit import result_row
from stbench import baselines, questions, table_answers
from stbench.answer_keys import quantise
from stbench.records import BALANCE_POOLS, GOLD_KEYS, GOLD_OPTIONAL, check_gold
from stbench.score_stats import ROW_KEYS, aggregate
from stbench.task_predicate import TaskThings


def test_the_compiler_writes_the_candidates_class_and_range_into_the_gold_row(monkeypatch):
    """Contract: `questions.records` puts the candidate's `answer_class` and `range_m` in its gold row, after `slice`,
    and the row passes `check_gold`; a candidate that declares neither gives null in both. Regression caught: the class
    dropped between the draw and the file (the balance check would find every row unclassed and pass vacuously), or a
    range written on the wrong row."""
    register(monkeypatch, fake_spec("fake_c"), lambda w: [])
    monkeypatch.setitem(BALANCE_POOLS, "fake_c", {"moved": frozenset({"moved"}), "stayed": frozenset({"yes"})})
    classed = fake_cand("fake_c", 1, answer_class="moved", range_m=3.25)
    plain = fake_cand("fake_c", 2)
    _, golds = questions.records([classed, plain])
    assert [(g["answer_class"], g["range_m"]) for g in golds] == [("moved", 3.25), (None, None)]
    assert list(golds[0]).index("answer_class") == list(golds[0]).index("slice") + 1
    assert tuple(golds[0]) == (*GOLD_KEYS, *GOLD_OPTIONAL) and all(check_gold(g) == [] for g in golds)


def test_every_compiled_gold_row_carries_a_class_key_even_when_the_family_declares_none():
    """Contract (plan §4, A0): every gold row a compile writes has `answer_class` and `range_m` (null for a family that
    declares none), so no row is unclassed by omission. Hand-checked on a compile of the vacuum's synthetic world.
    Regression caught: a family's candidate path that builds its row without the keys (`check_gold` would refuse the
    compile, or an older writer would leave rows a reader pools silently)."""
    s = things_by_task(multi_run(2), 2)
    _, golds = compiled(s, 0, 1, length="medium")
    assert golds and all("answer_class" in g and "range_m" in g for g in golds)
    assert all(g["answer_class"] == (g["slice"] if g["family"] == "object_where" else None) for g in golds)


@pytest.mark.parametrize("answer, key", [
    (offset(1.2, -0.3), "2,-1"),     # floor(1.2 / 0.5) = 2 ahead, floor(-0.3 / 0.5) = -1 left
    (offset(0.49, 0.0), "0,0"),
    (offset(0.5, 0.5), "1,1"),       # on an edge: the upper cell
    (offset(-0.01, 0.99), "-1,1"),
    (offset(-3.0, -3.0), "-6,-6"),
])
def test_an_offset_is_keyed_by_its_half_metre_cell_in_the_cameras_frame(answer, key):
    """Contract: `answer_keys.quantise` of an offset is `i,j` = (floor(ahead / 0.5), floor(left / 0.5)), so the priors
    that learn from other scripts' golds pool offsets that fall in one cell of the camera's frame. By hand. Regression
    caught: an offset that raises (a prior that cannot key an `ego_now` record), or the axes swapped between the
    record and the lookup."""
    assert quantise(answer) == key


def test_a_result_row_without_a_class_reads_as_null_and_the_aggregate_still_runs():
    """Contract: `ROW_KEYS` holds `answer_class` after `slice`, and a row of a run before 0.6.0 that lacks it is read as
    null (an older run stays scorable): `aggregate` takes it and answers as for the same rows with the key. Regression
    caught: a run kept from `m8` becoming unreadable because its rows predate the key, or the missing key read as a
    class of its own."""
    assert ROW_KEYS.index("answer_class") == ROW_KEYS.index("slice") + 1
    rows = [result_row(f"c{i}", f"q{i}", 1.0 if i % 2 else 0.0, "correct" if i % 2 else "wrong", script=f"s{i % 3}")
            for i in range(6)]
    older = [{k: v for k, v in r.items() if k != "answer_class"} for r in rows]
    assert "answer_class" not in older[0]
    assert aggregate(older, n_boot=20)["headline"]["mean"] == aggregate(rows, n_boot=20)["headline"]["mean"]


def test_the_task_things_option_reaches_the_answerers_as_ctx(monkeypatch):
    """Contract (plan §3): the table systems pass the job's `TaskThings` to their answerers as `ctx["task_things"]`
    beside `map` and `events` (None when the run has none). Regression caught: an option built by the harness and
    dropped by the system, so `table_task` would answer every question from no predicate."""
    seen = []
    monkeypatch.setattr(table_answers, "answer", lambda store, question, ctx: seen.append(ctx) or {"type": "unknown"})
    predicate = TaskThings("near", "point", frozenset({"inspect"}), reach_m=10.0)
    for opts, expected in (({"events": {}, "task_things": predicate}, predicate), ({"events": {}}, None)):
        table = object.__new__(baselines.MemoryTable)
        table.opts, table.store, table.latest, table.model = opts, SimpleNamespace(steps={0: {}}), [0], None
        table.answer_at({"qid": "q", "template": "task_things", "slots": {}}, {"k": 0})
        assert seen[-1]["task_things"] is expected
