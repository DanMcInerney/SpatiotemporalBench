"""`stbench view` for scoring by ability and the offset answer: the layouts table's columns and abilities, a question's
class and range, and an offset shown in the camera's frame."""

from __future__ import annotations

import json
import shutil
import subprocess
from importlib import resources

import pytest

from ability_kit import results_doc, scored
from runview_kit import make_run
from stbench import view, view_page
from stbench.view_questions import merge
from test_view_page import run_stub

UTIL = resources.files("stbench").joinpath(view_page.PACKAGE_DIR, "util.js")


def test_the_layout_carries_the_columns_the_secondary_means_and_each_ability(tmp_path):
    """Contract (`view._layouts`): a layout holds `results.json`'s headline, its `columns`, `abilities_mean`,
    `pooled`, `thin` and a line per ability with its count and flag, copied and not computed. By hand the all-unknown
    run of the hand dataset holds `object_where` (the ability `memory`: 4 questions, 3 unknown golds, 3/4) and
    `path_ahead` (an ability of its own: 2 answerable, 0), in a dataset too small for the rule, so none is a
    diagnostic; the equal mean is 3/8 and the pooled one 3/6. Regression caught: the page's columns read from the
    pooled `by.scope` (not the columns the headline is the mean of), or an ability left out."""
    _, run = make_run(tmp_path)
    (arm,) = view.read_run(run, tmp_path)["arms"]
    layout = arm["layouts"]["apartment_a"]
    assert {k: c["mean"] for k, c in layout["columns"].items()} == pytest.approx({"now": 0.0, "memory": 0.75})
    assert {k: c["n"] for k, c in layout["columns"].items()} == {"now": 2, "memory": 4}
    abilities = layout["abilities"]
    assert {k: a["mean"] for k, a in abilities.items()} == pytest.approx({"memory": 0.75, "path_ahead": 0.0})
    assert {k: (a["n"], a["diagnostic"]) for k, a in abilities.items()} == {
        "memory": (4, False), "path_ahead": (2, False)}
    assert layout["abilities_mean"]["mean"] == pytest.approx(0.375) and layout["pooled"]["mean"] == pytest.approx(0.5)
    assert layout["thin"] == []


def test_a_question_carries_its_answer_class_and_range():
    """Contract: `merge` puts the gold's `answer_class` and `range_m` on the question the page shows (the row's class
    when the gold row is absent, null range). Regression caught: the class and the range the credit was built from
    invisible to a reader checking a distance score."""
    gold = {"qid": "q", "gold": {"type": "offset", "ahead_m": 1, "left_m": 0}, "answer_class": "answerable",
            "range_m": 3.5, "credit": "ladder"}
    (shown,) = merge(None, None, 0, [{"qid": "q", "template": "ego_mem"}], {"q": gold}, [], [])
    assert (shown["answer_class"], shown["range_m"]) == ("answerable", 3.5)
    (bare,) = merge(None, None, 0, [{"qid": "q", "template": "ego_mem"}], {}, [{"qid": "q", "answer_class": "yes"}], [])
    assert (bare["answer_class"], bare["range_m"]) == ("yes", None)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_page_names_the_abilities_and_shows_an_offset_in_the_cameras_frame(tmp_path):
    """Contract (page.js, util.js): the layouts table lists each ability with its mean, its count and a diagnostic flag,
    and an offset answer or gold reads `x m ahead, y m left`, with how far off it is, as a share of the range when the
    question has one, and the range in the credit line. By hand: gold (3, 4) answered (3, 1) is 3 m off, 3/5 of a 5 m
    range. Regression caught: an offset shown as raw JSON, an error shown without the range it is judged against, or an
    ability line missing from the table."""
    _, run = make_run(tmp_path)
    out = run_stub(tmp_path, view_page.render(view.read_run(run, tmp_path)))
    assert "memory +0.750 over 4, path_ahead 0.000 over 2" in out["text"]
    script = (f"const vm = require('vm'); const ctx = {{document: {{}}, Node: class {{}}}}; vm.createContext(ctx);"
              f"vm.runInContext(require('fs').readFileSync({json.dumps(str(UTIL))}, 'utf8'), ctx);"
              "console.log(JSON.stringify(vm.runInContext(`["
              "formatAnswer({type: 'offset', ahead_m: 3, left_m: -1.5}),"
              "offBy({answer: {type: 'offset', ahead_m: 3, left_m: 1}, "
              "gold: {type: 'offset', ahead_m: 3, left_m: 4}, range_m: 5}),"
              "creditText({credit: 'ladder', tolerances: {ladder: [0.5, 1]}, answer_type: 'offset', "
              "range_m: 5})]`, ctx)))")
    done = subprocess.run(["node", "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr[-1500:]
    shown, off_by, credit = json.loads(done.stdout)
    assert shown == "3 m ahead, -1.5 m left"
    assert off_by == "off by 3 m (0.6 of the range)"
    assert credit == "credit: ladder 0.5 / 1 m, range 5 m"


def test_the_viewer_data_of_a_results_document_holds_no_statistic_of_its_own():
    """Contract (view._layouts): the layout block is a copy of the aggregate's own fields. A document with a diagnostic
    ability and a thin pair shows both flags as `results.json` has them. Regression caught: the viewer recomputing a
    mean and so disagreeing with the report."""
    thin = frozenset({("amr/1", "change")})
    rows = [*scored("ego_now", [1, 1]), *scored("mover_next", [-1]), *scored("object_was", [1], job="amr/1")]
    doc = results_doc(rows, thin=thin)
    block = view._layouts(doc)["apartment_a"]
    assert block["abilities"]["prediction"]["diagnostic"] is True and block["abilities"]["change"]["diagnostic"] is True
    assert block["thin"] == [["amr/1", "change", 0.0]]
    assert block["headline"] == doc["layouts"]["apartment_a"]["headline"]
