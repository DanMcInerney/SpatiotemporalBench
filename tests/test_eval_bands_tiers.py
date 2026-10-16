"""The credit bands and the kind table of the report (`eval_report`): the bands are the tiers of the four
quarters the scorer awards, and the kinds are a table of their own beside the abilities. Hand rows; every number is
worked from the scorer's tiers, never read from the code under test."""

from __future__ import annotations

from ability_kit import r, results_doc
from score_kit import result_row
from stbench.eval_report import BANDS, credit_band, render


def band(s, **kw):
    return credit_band({**result_row("c", "q", 0.0, "partial", **kw), "s": s})


def test_a_credit_is_banded_by_the_quarters_it_holds():
    """Contract (§3.1): `credit_band` names the tier an answer's credit reaches: full at 1; within need from 3/4 (the
    need met and no pixel); right thing from 1/2 (within twice the need); direction from 1/4 (the right side and
    nothing else); a miss below. By hand: 1.0 full; 0.9 and 0.75 within need; 0.74 and 0.5 right thing; 0.49 and 0.25
    direction; 0.24 and 0 miss. A tier is met up to rounding (0.7499999999 is within need), an `unknown` is its own band
    whatever the credit. Regression caught: the old thirds (2/3 and 1/3) still deciding a band, so a direction-only
    answer counts as a miss, or a boundary answer falling a band for a rounding error."""
    got = [band(s) for s in (1.0, 0.9, 0.75, 0.74, 0.5, 0.49, 0.25, 0.24, 0.0)]
    assert got == ["full", "within need", "within need", "right thing", "right thing", "direction", "direction", "miss",
                   "miss"]
    assert band(0.7499999999) == "within need"
    assert credit_band(result_row("c", "q", 0.0, "abstained")) == "unknown"
    assert credit_band(result_row("c", "q", 1.0, "correct", slice_="unknown")) == "unknown"
    assert BANDS == ("full", "within need", "right thing", "direction", "miss", "unknown")


def kind_rows():
    return [r("c1", "q1", 1.0, "clear_path", scope="now"), r("c1", "q2", 0.5, "closest_person", scope="now"),
            r("c1", "q3", -0.5, "open_heading"), r("c1", "q4", 1.0, "ego_now"), r("c1", "q5", 0.0, "ego_mem"),
            r("c1", "q6", -1.0, "object_where")]


def test_the_report_gives_a_table_of_the_mean_by_kind_beside_the_abilities():
    """Contract (§5): `results.md` has a `Per kind` table after the per-ability one with a line a kind, its question
    count and its mean pooled over them. By hand: harm = clear_path 1.0 and closest_person 0.5, +0.750 over 2; help =
    open_heading -0.5 over 1, counted (0.11.0: `open_heading` is clearance); reuse = object_where -1.0 over 1;
    situation = ego_now 1.0 and ego_mem 0.0, +0.500 over 2. The headline is the abilities': now = clearance 0.75 and
    distance 1.0, 0.875; memory = clearance (open_heading -0.5) and memory (ego_mem 0.0, object_where -1.0: -0.5),
    -0.5; (0.875 - 0.5) / 2 = +0.188 over the 6 counted questions. Regression caught: a report that drops the split,
    orders it before the abilities the headline is made of, lets it move the headline, or flags the help kind as
    counted in no mean."""
    lines = render(results_doc(kind_rows(), n_boot=50)).splitlines()
    at = lines.index("### Per kind")
    assert at > lines.index("### Per ability")
    assert lines[at + 2] == "| kind | n | mean | 95% CI |"
    assert lines[at + 4:at + 8] == ["| harm | 2 | +0.750 | n/a |",
                                    "| help | 1 | -0.500 | n/a |",
                                    "| reuse | 1 | -1.000 | n/a |", "| situation | 2 | +0.500 | n/a |"]
    assert "| headline | 6 | +0.188 | n/a |" in lines


def test_the_credit_band_table_has_a_column_a_tier():
    """Contract: the shares table under `Credit bands by ability` has one column per band of `BANDS`. By hand, three
    `distance` rows of credit 1, 0.75 and 0.25 (penalised 1, 0.5, -0.5) are full, within need and direction a third
    each; the mean is +0.333, the credit (1 + 0.75 + 0.25) / 3 = +0.667 and the bound (1 + 0.5 + 0) / 3 = +0.500.
    Regression caught: a header out of step with the bands."""
    rows = [{**r("c1", f"q{i}", score, "ego_now"), "s": s} for i, (score, s) in enumerate(
        ((1.0, 1.0), (0.5, 0.75), (-0.5, 0.25)))]
    lines = render(results_doc(rows, n_boot=50)).splitlines()
    assert "| ability | n | " + " | ".join(BANDS) + " | mean | credit | abstention bound |" in lines
    assert "| distance | 3 | 0.333 | 0.333 | 0.000 | 0.333 | 0.000 | 0.000 | +0.333 | +0.667 | +0.500 |" in lines
