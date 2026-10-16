"""`tools/fill_report.py`: the `now` and `memory` cells of the results tables. They are read from the aggregate's
`columns`, the counted cells the headline averages, and every number is worked by hand from the rows written here."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from ability_kit import r
from stbench.score_stats import aggregate

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fill_report", ROOT / "tools" / "fill_report.py")
fr = importlib.util.module_from_spec(SPEC)
sys.modules["fill_report"] = fr
SPEC.loader.exec_module(fr)


def test_the_now_and_memory_cells_are_the_counted_columns_the_headline_averages_not_every_question_of_a_scope():
    """Contract (the 0.9.0 release review, S1): the `now` and `memory` cells of a row are the aggregate's `columns`, the
    counted cells the headline is the mean of, with the questions those cells rest on; they are not `by.scope`, which
    pools every question of a scope, a diagnostic ability's too. By hand: `clear_path` 1.0 and 0.5 is a `now` cell of
    3/4 over 2; `object_where` 1, 0, 0, 0 is a `memory` cell of 1/4 over 4; `mover_next` (a diagnostic) -1.0 and -1.0
    in `now` and -0.5 in `memory` counts in neither. So the cells read +0.750 over 2 and +0.250 over 4, and their mean
    is the headline, +0.500. Pooled by scope they would read (1 + 0.5 - 1 - 1) / 4 = -0.125 over 4 and (1 - 0.5) / 5 =
    +0.100 over 5, which do not average to the headline. Regression caught: the cells read from `by.scope` again, so a
    reader cannot rebuild the headline from the table."""
    rows = [r("c1", "q1", 1.0, "clear_path", scope="now", script="s1"),
            r("c1", "q2", 0.5, "clear_path", scope="now", script="s1"),
            *(r("c1", f"q{i}", score, "object_where", script="s2") for i, score in enumerate((1, 0, 0, 0), start=3)),
            r("c1", "q7", -1.0, "mover_next", scope="now", script="s3"),
            r("c1", "q8", -1.0, "mover_next", scope="now", script="s3"),
            r("c1", "q9", -0.5, "mover_next", scope="memory", script="s3")]
    head, now, memory, own = fr.split_cells(aggregate(rows, n_boot=20))
    assert [x.split(" ")[0] for x in (head, now, memory)] == ["+0.500", "+0.750", "+0.250"]
    assert now.endswith(" over 2") and memory.endswith(" over 4") and own == "-"
    assert fr.split_cells(None) == ["n/a", "-", "-", "-"]
