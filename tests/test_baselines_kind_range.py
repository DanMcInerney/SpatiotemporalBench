"""`kind_range`, `ego_now`'s declared reference (FD-0.6.0 §2.1, VISION rule 5): the current view's bearing to the thing
at the typical floor range of its kind. Hand-fed steps through the real `table_ego` answerer, which measures from the
camera position of the ask's step (`kind_range_kit.seen_at`)."""

from __future__ import annotations

import pytest

from baseline_kit import UNKNOWN, ahead, offset, record, started, step
from kind_range_kit import ego_question, prior_of, seen_at
from stbench.baselines import BASELINES, KindRange
from stbench.baselines_kind_range import Ranges, placed
from stbench.gates import LEARNED

# floor distances 2, 5, 7, 10 and 13 (median 7, mean 7.4): the axis and the 3-4-5, 6-8-10 and 5-12-13 triangles
BINS = [(2.0, 0.0), (3.0, 4.0), (0.0, -7.0), (6.0, 8.0), (-5.0, 12.0)]


def view(tmp_path, prior, *, kind="bin", ahead_m=4.0, left_m=3.0):
    """A `kind_range` over `prior`, fed an empty step 0 and step 1 that shows one red `kind` `ahead_m` ahead of the
    camera and `left_m` to its left (4 and 3 by default: a floor distance of 5)."""
    system = started(KindRange, tmp_path, prior=prior)
    system.observe([step(0, 1.0, 0.0, 90.0), seen_at(1, ahead_m, left_m, kind)])
    return system


def answer(system, kind="bin", k=1):
    return system.ask("during", {"k": k, "clock": ""}, [ego_question(kind)])["q1"]


def test_it_keeps_the_bearing_and_takes_the_kinds_median_range_not_the_true_one(tmp_path):
    """Contract (FD-0.6.0 §2.1): the answer lies on the bearing of the thing in the current view, at the median floor
    distance of its kind over other scripts' `ego_now` golds, whatever its true range. Hand case: the bin in view is 4
    ahead and 3 left of the camera (5 m, bearing 36.87 degrees left of ahead); other scripts' bins stood 2, 5, 7, 10 and
    13 m away (median 7, mean 7.4), so the answer is (4 x 7 / 5, 3 x 7 / 5) = (5.6, 4.2). A bin 1.6 ahead and 1.2 left
    (2 m) on the same bearing gets the same answer. Regression caught: the true range used (the answer would be
    (4, 3)), the mean for the median (5.92, 4.44), or the bearing lost."""
    assert answer(view(tmp_path, prior_of("bin", BINS))) == offset(5.6, 4.2)
    nearer = view(tmp_path / "near", prior_of("bin", BINS), ahead_m=1.6, left_m=1.2)
    assert answer(nearer) == offset(5.6, 4.2)


def test_the_bearing_is_the_cameras_not_the_bases(tmp_path):
    """Contract (`table_ego`: the offset is measured from the camera position of the ask's step): the bearing kept is
    the camera's. Hand case: the vacuum's camera is mounted forward of its base, so a thing 4 m ahead of the camera and
    1 m to its left stands at a different bearing from the base; the answer on it, with the median 5 (golds on the axis)
    and the thing 4.123 m away, is (4 x 5 / 4.123, 1 x 5 / 4.123) = (4.8507, 1.2127), not the base's. Regression caught:
    a bearing taken from the base position."""
    system = view(tmp_path, prior_of("bin", [(5.0, 0.0)] * 5), ahead_m=4.0, left_m=1.0)
    got = answer(system)
    assert got["ahead_m"] == pytest.approx(4.8507, abs=1e-4) and got["left_m"] == pytest.approx(1.2127, abs=1e-4)


def test_a_bearing_behind_the_camera_keeps_its_signs(tmp_path):
    """Contract: the bearing is kept whole, behind and to the right as well as ahead and to the left. Hand case: a thing
    3 m behind and 4 m to the right (offset (-3, -4), 5 m) of a kind whose median is 10 is answered (-6, -8).
    Regression caught: a magnitude-only rescale that drops a sign."""
    system = view(tmp_path, prior_of("bin", [(10.0, 0.0)] * 5), ahead_m=-3.0, left_m=-4.0)
    assert answer(system) == offset(-6.0, -8.0)


@pytest.mark.parametrize(("golds", "expected"), [
    ([(50.0, 0.0)], (40.0, 30.0)),
    ([(20.0, 0.0), (30.0, 0.0)], (20.0, 15.0)),
    ([(20.0, 0.0), (30.0, 0.0), (90.0, 0.0)], (24.0, 18.0)),
])
def test_the_median_is_over_any_gold_of_the_kind_however_few(tmp_path, golds, expected):
    """Contract (the coordinator's rule: a reference is the strongest held-out "bearing plus typical range" answer): the
    range is the median floor distance over any gold of the kind, a single one included, with no support floor. Hand
    cases, a drum asked at (4, 3) (5 m) beside the bins of five scripts: one drum gold at 50 m gives 50, so (4 x 50 /
    5, 3 x 50 / 5) = (40, 30); golds at 20 and 30 m a median of 25, (20, 15); at 20, 30 and 90 m a median of 30 (not
    the mean 46.7), (24, 18). Regression caught: a support floor that sends a thinly held kind to a range of other
    kinds (it scored under `all_unknown` on release-c), or a mean for the median."""
    prior = [*prior_of("bin", BINS), *prior_of("drum", golds, scripts=len(golds), start=3)]
    assert answer(view(tmp_path, prior, kind="drum"), "drum") == offset(*expected)


def test_each_kind_takes_its_own_median(tmp_path):
    """Contract: a kind with enough golds takes its own median, however the other kinds stand. Hand case: bins at 7 m
    (median of 2, 5, 7, 10, 13) beside five drums at 40 m: the bin is answered at 7 (5.6, 4.2), the drum at 40
    (32, 24). Regression caught: one median for every kind, which a gate would read as the kind's range."""
    prior = [*prior_of("bin", BINS), *prior_of("drum", [(40.0, 0.0)] * 5, start=3)]
    assert answer(view(tmp_path, prior)) == offset(5.6, 4.2)
    assert answer(view(tmp_path / "d", prior, kind="drum"), "drum") == offset(32.0, 24.0)


def test_a_kind_no_other_script_has_a_gold_of_is_unknown(tmp_path):
    """Contract: with no other-script gold of the kind there is no typical range and no borrowing from other kinds, so
    `unknown`. Hand cases: no prior at all; a prior of bins asked about a drum (the bins' median 7 is not lent to it);
    a prior whose only drum records are `unknown` golds, which carry no range. Regression caught: a fallback to
    every kind's range, which answered kinds nobody had seen at a range that is another thing's, or `unknown` golds
    counted as a range of 0."""
    assert answer(view(tmp_path / "a", [])) == UNKNOWN
    assert answer(view(tmp_path / "b", prior_of("bin", BINS), kind="drum"), "drum") == UNKNOWN
    unknown = [record(ego_question("drum", qid=f"u{i}"), UNKNOWN, script=f"s{i}", when="during") for i in range(3)]
    assert answer(view(tmp_path / "c", [*prior_of("bin", BINS), *unknown], kind="drum"), "drum") == UNKNOWN


def test_it_reads_only_ego_now_records_and_answers_only_ego_now(tmp_path):
    """Contract: other templates are `unknown`, whatever the table could say of them, and other templates' records are
    no support. Hand case: the prior holds five bin golds of `ego_now` and the case is asked, beside the `ego_now`
    question, a `path_ahead` question and an `ego_mem` one about the bin in view (which `table_ego` answers with the
    bin's offset, as it does `ego_now`): `unknown` for both. Five offset-gold records of another template
    (`object_where`) give a drum no range. Regression caught: a
    reference that answers the family's neighbours (a gate would then read it as a shortcut on them), or a range
    learned from golds of another family."""
    system = view(tmp_path, prior_of("bin", BINS))
    memory = {**ego_question("bin", qid="q3"), "template": "ego_mem"}
    both = system.ask("during", {"k": 1, "clock": ""}, [ego_question(), ahead(qid="q2"), memory])
    assert both == {"q1": offset(5.6, 4.2), "q2": UNKNOWN, "q3": UNKNOWN}
    stray = [record({**ego_question("drum", qid=f"o{i}"), "template": "object_where"}, offset(40.0, 0.0),
                    script=f"s{i % 3}") for i in range(5)]
    assert Ranges([*prior_of("bin", BINS), *stray]).of("drum") is None


def test_a_thing_out_of_the_frame_of_the_ask_is_unknown(tmp_path):
    """Contract (`table_ego`: `ego_now` reads the sighting at the ask step itself): the bin is in view at step 1 and
    not at step 3, so it is answered at the first ask (5.6, 4.2) and `unknown` at the second, a thing no longer in
    view having no bearing to keep. Regression caught: a reference that answers from an earlier sighting, which
    would hand a `now` family's answer to a system that remembers."""
    system = view(tmp_path, prior_of("bin", BINS))
    assert answer(system) == offset(5.6, 4.2)
    system.observe([step(2, 1.0, 0.0, 90.0), step(3, 1.0, 0.0, 90.0)])
    assert answer(system, k=3) == UNKNOWN


def test_a_bearing_with_no_direction_goes_straight_ahead_and_an_unknown_stays():
    """Contract (`placed`): a thing exactly at the camera's ground point has no bearing and is put straight ahead at the
    range; an `unknown`, or any answer that is no offset, stays `unknown`, as does any with no range to place it at.
    Hand values. Regression caught: a division by zero on a drone's thing straight below it, or a position handed back
    as an offset."""
    assert placed(offset(0.0, 0.0), 7.0) == offset(7.0, 0.0)
    assert placed(UNKNOWN, 7.0) == UNKNOWN and placed(offset(3.0, 4.0), None) == UNKNOWN
    assert placed({"type": "position", "xy": [1.0, 2.0]}, 7.0) == UNKNOWN


def test_it_is_a_reference_with_a_prior_and_no_twin():
    """Contract (VISION rule 5): `kind_range` is a registered baseline that takes the other scripts' records and the map
    and nothing else, and is not among `gates.LEARNED`, which is why it has no in-pool twin (`tests/test_baselines.py`).
    Regression caught: the reference registered without its prior (it would answer `unknown` everywhere), or listed
    among the learned shortcuts."""
    b = BASELINES["kind_range"]
    assert b.cls is KindRange and b.needs_prior and b.needs_map and "kind_range" not in LEARNED
    assert not (b.prior_nm or b.needs_events or b.other_stream or b.models_agent or b.in_pool)
