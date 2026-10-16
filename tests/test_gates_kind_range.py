"""`kind_range` in the gates (FD-0.6.0 §2.1, §13; VISION rule 5): the declared reference of `ego_now` is no shortcut and
is a rival of `separation`, so distance is measured above it. Hand rows of a synthetic `now` family whose rows name
`kind_range` as their reference (the real `ego_now` is lane A3's); the layouts and systems as in
`tests/test_gates.py`."""

from __future__ import annotations

import pytest

from stbench.gates import BASELINE_NAMES, BLIND, KIND_RANGE, ORACLE, SHORTCUTS, evaluate, failing
from stbench.map_files import HELD_OUT
from test_gates import N_BOOT, gates_of, rows, system_set

FAMILY = "ego_here"
NOW = {"scopes": ["now"] * 4, "reference": KIND_RANGE}   # four `now` questions whose reference is `kind_range`


def ego_set(kind_range, **options):
    """`system_set` of a family of `now` rows that name `kind_range` as their reference, with `kind_range`'s rows
    scoring `kind_range` (one per question; four scripts of four questions, `all_unknown` 0.25, `blind` 0.30, the table
    1)."""
    options = {**NOW, **options}
    keep = ("scopes", "reference", "layout", "job")
    own = rows(FAMILY, kind_range, **{k: v for k, v in options.items() if k in keep})
    for r in own:
        r["outcome"] = "partial"   # it answered, whatever the answer scored (`rows` calls a score in [0, 1) abstained)
    return system_set(FAMILY, **options) | {KIND_RANGE: own}


@pytest.mark.parametrize("layout", ["apartment_a", *sorted(HELD_OUT)[:1]])
def test_the_reference_is_a_rival_of_separation_and_never_a_shortcut(layout):
    """Contract (FD-0.6.0 §2.1): a family whose every oracle row names `kind_range` is gated on `separation` against
    the best of the shortcuts and `kind_range` ("distance is measured above it"), and `kind_range` is not among the
    shortcuts (`shortcuts` never reads it: it is the soft floor of the family). Hand rows, the table at 1.0, `blind` at
    0.30 and `all_unknown` at 0.25: `kind_range` at 0.5 leaves a gap of 0.5, which passes (>= 0.40, lower bound 0.5 >
    0.30), and names `kind_range` as the best rival; at 0.7 the gap is 0.3 and `separation` fails, naming it, while
    `shortcuts` still passes (the worst shortcut is `blind`, 0.05 over; `kind_range`'s 0.45 over the floor would fail
    it were it read), and nothing else fails. Regression caught: the reference counted as a shortcut (the family fails
    the gate its own reference sits above), or left out of `separation` (distance never measured above it)."""
    passing = gates_of(ego_set([0.5] * 4, layout=layout))
    assert passing["separation"]["pass"] is True and passing["separation"]["best"] == KIND_RANGE
    assert passing["separation"]["value"] == pytest.approx(0.5)
    report = evaluate(ego_set([0.7] * 4, layout=layout), n_boot=N_BOOT)
    g = report["rows"][0]["gates"]
    assert g["separation"]["pass"] is False and g["separation"]["best"] == KIND_RANGE
    assert g["separation"]["value"] == pytest.approx(0.3)
    assert g["shortcuts"]["pass"] is True and g["shortcuts"]["worst"] == BLIND
    assert KIND_RANGE not in g["shortcuts"]["scores"] and set(g["shortcuts"]["scores"]) <= set(SHORTCUTS)
    assert list(failing(report)) == [(layout, "vacuum/1", FAMILY, "separation")]
    assert report["rows"][0]["scores"][KIND_RANGE]["mean"] == pytest.approx(0.7)


def test_the_shortcut_still_counts_when_it_is_the_better_rival():
    """Contract: `separation` compares the table against the best of the shortcuts and `kind_range`, whichever is the
    higher. Hand rows: `blind` at 0.60 against `kind_range` at 0.5, the table at 1.0: the best rival is `blind` and the
    gap 0.4; `kind_range` at 0.60 against `blind` at 0.3 names `kind_range`. Regression caught: the reference replacing
    the shortcuts as the rival, so a shortcut above it would go unseen in `separation`."""
    blind_up = gates_of(ego_set([0.5] * 4, blind=[0.6] * 4))["separation"]
    assert blind_up["best"] == BLIND and blind_up["value"] == pytest.approx(0.4)
    assert gates_of(ego_set([0.6] * 4))["separation"]["best"] == KIND_RANGE


def test_it_is_a_rival_only_where_every_oracle_row_names_it():
    """Contract (plan §4 A6: "where every oracle row's reference is `kind_range`"): a family whose rows name no
    reference, or a group where only some oracle rows do, has `kind_range` read as nothing, whatever it scores there,
    and does not list it as missing. Hand rows: `kind_range` at 0.9 beside the table at 1.0 passes `separation` (the
    gap to `blind`, 0.7) in a family with no reference; with one oracle row of the reference family unmarked it does
    too. Regression caught: `kind_range` made a rival of every family, or of a group where only some rows name it
    (here it would be the best rival at 0.9 and fail `separation`), or `missing` naming the reference on families that
    do not declare it."""
    plain = ego_set([0.9] * 4, reference=None, scopes=["memory"] * 4)
    (row,) = evaluate(plain, n_boot=N_BOOT)["rows"]
    assert row["gates"]["separation"]["best"] == BLIND and row["missing"] == []
    some = ego_set([0.9] * 4)
    some[ORACLE][0]["reference"] = None   # the oracle's first row names none
    assert gates_of(some)["separation"]["best"] == BLIND
    assert KIND_RANGE in BASELINE_NAMES


def test_a_reference_family_with_no_rows_for_it_cannot_pass_separation():
    """Contract (the gates' rule for a missing system): a group that names `kind_range` as its reference and holds no
    rows for it has `separation` `n/a` with a note naming it, and lists it in `missing`; a pass proven without the
    rival is no pass. Hand rows: the healthy set, without `kind_range`'s rows. And the same set with `kind_range`
    scoring 0.7 (a gap of 0.3) fails, so the `n/a` is the absent system and not a pass given. Regression caught: a
    reference that is silently skipped when its run is missing (a check that did not run it passes every distance
    family)."""
    without = system_set(FAMILY, **NOW)
    (row,) = evaluate(without, n_boot=N_BOOT)["rows"]
    g = row["gates"]["separation"]
    assert g["pass"] is None and "kind_range" in g["note"] and row["missing"] == [KIND_RANGE]
    assert gates_of(ego_set([0.7] * 4))["separation"]["pass"] is False


def test_a_reference_that_answered_nothing_is_no_rival():
    """Contract: a reference that abstained on every question of its family is not a rival `separation` can be passed
    against (a small dev build, three scripts or fewer a layout and job, leaves every `ego_now` prior under the support
    floor, so `kind_range` answers `unknown` throughout): `separation` is `n/a` with a note that it answered no
    question, and the group is not `missing` it. Hand rows: the healthy set with `kind_range` abstaining on all sixteen
    questions, where the table's gap of 1.0 over it would pass; with one of them answered (wrong) it is a rival again,
    and the best rival is `blind`, 0.30, still above its mean of 0. Regression caught: a reference that says nothing
    counted as a rival beaten, so a distance family passes `separation` above a floor that was never measured."""
    mute = ego_set([0.0] * 4)
    for r in mute[KIND_RANGE]:
        r["outcome"] = "abstained"
    (row,) = evaluate(mute, n_boot=N_BOOT)["rows"]
    g = row["gates"]["separation"]
    assert g["pass"] is None and "answered no question" in g["note"] and row["missing"] == []
    mute[KIND_RANGE][0]["outcome"] = "wrong"
    assert gates_of(mute)["separation"]["best"] == BLIND
