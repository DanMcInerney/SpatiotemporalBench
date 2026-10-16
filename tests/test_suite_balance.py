"""The balance rule (`suite_balance`, FD-0.6.0 §8): a family fails on a job when its held-out best constant is shown to
beat `all_unknown`; the shares of the classes are reported against the draw and never gated."""

from __future__ import annotations

import pytest

from balance_kit import numbers, yesno
from stbench import suite_balance
from stbench.suite_balance import best_constant_rows, class_shares, read


def only(asked) -> dict:
    (entry,) = read(asked)
    return entry


def test_a_constant_that_beats_the_floor_fails_and_is_named():
    """Contract (FD-0.6.0 §8): a family whose yes share is 70% fails: always saying yes scores above `all_unknown`
    with an interval above 0, and the family does not ship on that job. Four scripts of ten yes/no questions, seven yes
    and three no each: the best constant answered from the other three scripts (21 yes of 30) is yes, which is right
    seven times (+1) and wrong three (-1) a script, a gain of (7 - 3) / 10 = +0.4 over `all_unknown`'s 0 on answerable
    golds, the same in every script, so the interval is +0.4 to +0.4. Regression caught: a balance rule that lets a
    constant answer through (the verdict never failing, or an interval that is not read)."""
    entry = only(yesno([7] * 4, 10))
    assert entry["constant"] == "yes" and entry["scripts"] == 4 and entry["n"] == 40
    assert entry["mean"] == pytest.approx(0.4) and entry["ci"] == pytest.approx((0.4, 0.4))
    assert entry["fails"] is True
    assert suite_balance.line(entry) == (
        "BALANCE FAIL apartment_a vacuum/1 same_place: the best constant (over every script: yes) scores +0.400 held "
        "out, over all_unknown (95% interval +0.400 to +0.400; 4 scripts, 40 questions)")
    assert suite_balance.failure(entry).startswith(suite_balance.line(entry) + "; a constant answer must not beat")


def test_a_fair_coin_that_came_up_slightly_tilted_passes():
    """Contract (plan §1.6): a draw that is fair in expectation is not failed for the tilt chance gives it. Eight
    scripts of five yes/no questions, 21 yes of 40 (52.5%), the yes golds three a script in five scripts and two in
    three. Leaving a script out, the best constant is yes in every pool (18 yes of 35 or 19 of 35), so a script of three
    yes scores 3 - 2 = +1 and a script of two yes 2 - 3 = -1: a gain of (5 - 3) / 40 = +0.05, and the interval over the
    eight scripts (each +0.2 or -0.2) reaches below 0. Regression caught: a gate that fails the mean alone, so that a
    fair coin fails whenever it lands over 50%."""
    entry = only(yesno([3, 3, 3, 3, 3, 2, 2, 2], 5))
    assert entry["mean"] == pytest.approx(0.05) and entry["scripts"] == 8
    assert entry["ci"][0] < 0 < entry["ci"][1] and entry["fails"] is False
    assert suite_balance.line(entry).startswith("BALANCE ok apartment_a vacuum/1 same_place: the best constant")


def test_the_interval_is_taken_over_scripts_and_the_shares_are_not_gated():
    """Contract (plan §2 call 5): the interval resamples scripts, the unit that moves together (a script's cases are
    windows of one world), and the class shares decide nothing. Four scripts of twenty yes/no questions with 20, 14, 14
    and 4 yes (52 of 80, 65% against a coin's 50%). Held out, the constant is yes in every pool (A: 32 of 60, B and C:
    38 of 60, D: 48 of 60), scoring +1.0, +0.4, +0.4 and -0.6 a question in the four scripts: a gain of +0.3. Over
    questions that is 0.3 with a standard error of about 0.11, an interval above 0; over scripts (SD 0.66 over four)
    it reaches below 0, so the family passes, though its yes share is 15 points off the draw. Regression caught: a
    bootstrap over questions, which fails families whose scripts disagree, or a band on the shares."""
    asked = yesno([20, 14, 14, 4], 20)
    entry = only(asked)
    assert entry["mean"] == pytest.approx(0.3) and entry["ci"][0] < 0 < entry["ci"][1]
    assert entry["fails"] is False
    shares = class_shares("same_place", [g.gold for g in asked])
    assert shares["pools"] == {"yes": pytest.approx(0.65), "no": pytest.approx(0.35)}
    assert shares["draw"] == {"pool": "no", "share": 0.5, "bound": "="}


def test_the_constant_is_chosen_leaving_the_script_out():
    """Contract (FD-0.6.0 §8: "over the other scripts' golds"): a script's questions are answered by a constant learned
    from the other scripts, never its own. Script s0 asks ten questions whose answer is 1.0, s1 to s3 two each whose
    answers are 11, 21 and 31 (a ladder of 2 m at most). Over all 16 golds the answer 1.0 is right ten times and wrong
    six, +0.25, which would give s0 ten correct answers; without s0 the best of the others scores (2 - 4) / 6 below
    abstaining, so s0 is answered `unknown` (0 on each of its ten). s1 to s3 are answered 1.0 (ten right of 14 golds:
    (10 - 4) / 14 above 0), wrong, -1 each. Regression caught: a pool that includes the script scored, which finds a
    constant only because it memorised the script."""
    asked = numbers([1.0, 11.0, 21.0, 31.0], [10, 2, 2, 2])
    rows = best_constant_rows(asked)
    by_script = {s: [r["score"] for r in rows if r["script"] == s] for s in ("s0", "s1", "s2", "s3")}
    assert by_script == {"s0": [0.0] * 10, "s1": [-1.0] * 2, "s2": [-1.0] * 2, "s3": [-1.0] * 2}
    entry = only(asked)
    assert entry["mean"] == pytest.approx(-6 / 16) and entry["fails"] is False


def test_a_one_script_group_cannot_fail():
    """Contract: with one script there is nothing to hold out and no interval (`score_stats`), so the rule neither
    answers nor fails. Regression caught: a constant learned from the script's own golds, or a crash on no pool."""
    entry = only(yesno([9], 10))
    assert entry["mean"] == 0 and entry["ci"] is None and entry["fails"] is False
    assert "no interval: under two scripts" in suite_balance.line(entry)


@pytest.mark.parametrize("layout, fails", [("apartment_a", True), ("home_large", False)])
def test_a_fact_of_the_layout_is_excused_on_a_held_out_layout_only(layout, fails):
    """Contract (the form adversary's rule, `form_check`): a family whose reference is `other_case` (the same answer in
    every case of its layout) is learned from public layouts only, so on a held-out layout there is nothing to learn
    from and its constant is `unknown` (a gain of 0); on a public layout the constant is learned and, all golds being
    yes, scores +1. Regression caught: a layout-constant family failed on held-out layouts, where it is the design."""
    entry = only(yesno([5] * 4, 5, layout=layout, reference="other_case"))
    assert entry["fails"] is fails and entry["mean"] == pytest.approx(1.0 if fails else 0.0)


def test_a_family_that_leans_one_way_is_judged_on_its_own_family():
    """Contract: each family of a group is read on its own golds, and the entries come in family order. A coin that
    leans (70% yes) and a coin that does not (four scripts alternating yes and no) in one group: the first fails, the
    second passes. Regression caught: families pooled into one constant, which a lean in one hides in the other."""
    lean, even = yesno([7] * 4, 10, "same_place"), yesno([5] * 4, 10, "object_was")
    first, second = read(lean + even)
    assert (first["family"], first["fails"], second["family"], second["fails"]) == ("object_was", False, "same_place",
                                                                                    True)
    assert second["mean"] == pytest.approx(0.4) and first["mean"] == pytest.approx(0.0)


@pytest.mark.parametrize("family, classes, pools, draw", [
    ("object_where", ["answerable"] * 4 + ["unknown"], {"answerable": 0.8, "unknown": 0.2}, ("unknown", 0.2, "=")),
    ("ego_mem", ["answerable"] * 4 + ["unknown"], {"answerable": 0.8, "unknown": 0.2}, ("unknown", 0.2, "=")),
    ("object_was", ["earlier"] * 3 + ["later"], {"earlier": 0.75, "later": 0.25}, ("later", 0.5, "=")),
    ("same_place", ["yes", "moved", "vacated", "yes"], {"yes": 0.5, "no": 0.5}, ("no", 0.5, "=")),
    ("task_things", ["empty", "items", "items"], {"empty": 1 / 3, "items": 2 / 3}, ("empty", 1 / 3, "<=")),
])
def test_class_shares_are_counted_by_hand_against_the_declared_draw(family, classes, pools, draw):
    """Contract (FD-0.6.0 §8 table): the report gives each class's share of the questions asked, each pool's (a pool is
    one or more classes: `same_place`'s no is moved and vacated) and the draw's share for the family: an unknown slice
    of 0.2 (`object_where`, `ego_mem`), fair coins (`object_was`, `same_place`), empties capped at a third
    (`task_things`). Counted by hand on a handful of golds. Regression caught: a pool read from another family's
    declaration, a moved and a vacated gold counted as two pools, or a draw changed in one place and not the other."""
    shares = class_shares(family, [{"answer_class": c} for c in classes])
    assert shares["n"] == len(classes)
    assert shares["classes"] == pytest.approx({c: classes.count(c) / len(classes) for c in set(classes)})
    assert shares["pools"] == pytest.approx(pools)
    assert shares["draw"] == {"pool": draw[0], "share": pytest.approx(draw[1]), "bound": draw[2]}


def test_a_family_with_no_classed_gold_or_no_declaration_reports_no_draw():
    """Contract: an unclassed family (`ego_now`, or any gold row of a compile before 0.6.0) reports its count and no
    shares, and a classed family with no declared draw (none today; the family is not in `DRAWS`) reports its classes
    with draw null. Regression caught: a share invented for a family that declares none."""
    assert class_shares("ego_now", [{"answer_class": None}] * 3) == {"n": 3, "classes": {}, "pools": {}, "draw": None}
    assert class_shares("ego_now", [{"answer_class": "yes"}])["draw"] is None


@pytest.mark.parametrize("yes, per_script, thin", [
    ([7, 7, 7], 10, True),    # three scripts: each held-out pool has two scripts, under the floor of three
    ([1] * 4, 1, True),       # four scripts of one question: each pool has three golds, under the floor of five
    ([2] * 4, 2, False),      # four scripts of two questions: three scripts and six golds, the least that reads
])
def test_a_group_too_thin_to_read_is_reported_and_cannot_fail(yes, per_script, thin):
    """Contract (coordinator, review of A9): a group whose held-out pools fall under the priors' support floor
    (`blind.supported`: 5 golds of 3 scripts) is reported `thin` and cannot fail, however far the constant gains, like
    the one-script group; the smallest group above the floor reads and fails. The constant gains +0.4 a question in
    the first group (seven yes of ten) and +1.0 in the other two (every gold yes). Regression caught: a family
    dropped on a job for a constant that is the best of many on a handful of golds."""
    entry = only(yesno(yes, per_script))
    assert entry["thin"] is thin and entry["fails"] is (not thin) and entry["mean"] > 0
    assert suite_balance.line(entry).startswith("BALANCE " + ("thin" if thin else "FAIL"))
    assert ("too thin to read" in suite_balance.line(entry)) is thin


def test_a_set_constant_is_described_by_its_size():
    """Contract: a report names a set constant as `positions (empty)` or `positions of 3`, not as a bare type or a list
    of coordinates, and a route by its points; numbers and offsets keep their values. Regression caught: a set
    printed as its type alone, which hides whether the best constant is the empty set."""
    assert suite_balance.describe({"type": "positions", "items": []}) == "positions (empty)"
    assert suite_balance.describe({"type": "positions", "items": [{"xy": [1.0, 2.0]}] * 3}) == "positions of 3"
    assert suite_balance.describe({"type": "set", "items": [{"xy": [1.0, 2.0], "change": "moved"}]}) == "set of 1"
    assert suite_balance.describe({"type": "cells", "items": []}) == "cells (empty)"
    assert suite_balance.describe({"type": "route", "points": [[0.0, 0.0], [1.0, 1.0]]}) == "route of 2"
    assert suite_balance.describe({"type": "number", "value": 1.4}) == "number 1.40"
    assert suite_balance.describe({"type": "offset", "ahead_m": 2.0, "left_m": -0.5}) == "offset 2.00 -0.50"
    assert suite_balance.describe({"type": "yesno", "value": False}) == "no"
    assert suite_balance.describe({"type": "unknown"}) == "unknown"
