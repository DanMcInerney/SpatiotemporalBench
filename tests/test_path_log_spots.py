"""`path_log`'s recurring-spots rule for `task_things` (`baselines_path_task`: `spots_claim`, `spots_held_out`):
the places other scripts' golds put items, within the predicate's reach of this task's anchor, judged on each script
left out. Hand maps and hand golds; every expected value is worked out in the test. The way is `task_kit.PATH`:
within 1 m of the docking leg (reach 1.0 m), here along y = 5 for x from 0 to 40 (the log) or along y = 20."""

from __future__ import annotations

import pytest

import task_kit as tk
from baseline_kit import EVENTS, gold, question, record, started, step
from stbench import baselines_path_task as bpt
from stbench.baselines_path import PathLog
from stbench.blind import Prior
from stbench.task_predicate import anchor_of
from task_kit import PATH

SLOTS = {"phrase": PATH.phrase, "task": 0}
ASKED = question("task_things", SLOTS, answer_type="positions")
UNKNOWN = {"type": "unknown"}


def leg(y: float) -> list[dict]:
    """The steps of a task that docked along y = `y`: cleaning at the origin, then docking at x = 0, 20 and 40."""
    return [step(0, 0.0, 0.0, phase="cleaning"), step(1, 0.0, y, phase="docking"), step(2, 20.0, y, phase="docking"),
            step(3, 40.0, y, phase="docking")]


def listed(*places: tuple[float, float]) -> dict:
    return {"type": "positions", "items": [{"xy": list(p)} for p in places]}


def positions_gold(*places: tuple[float, float]) -> dict:
    """A `task_things` gold row listing `places` (F1 within 0.5 m)."""
    return {**gold(listed(*places)), "family": "task_things", "credit": "f1", "tolerances": {"match_m": 0.5},
            "metric": None}


def prior_of(rows: list[tuple[str, float, list[tuple[float, float]]]], in_pool: bool = False) -> Prior:
    """A `task_things` record per row (script, the y the task docked along, the places its gold lists), with the anchor
    the harness would write of that task."""
    out = []
    for script, y, places in rows:
        rec = record(ASKED, listed(*places), script=script)
        out.append({**rec, "anchor": anchor_of(PATH, leg(y), None), "gold": positions_gold(*places)})
    return Prior(out, in_pool=in_pool)


def log_with(tmp_path, prior: Prior, y: float = 5.0) -> PathLog:
    """`path_log` over the hand map, with the prior, having logged one task that docked along y = `y`."""
    pl = started(PathLog, tmp_path, map=tk.room_map(tk.script()), events=dict(EVENTS), task_things=PATH, prior=prior)
    pl.task(0, "2026-03-02T08:00:00", 0.0, "brief")
    pl.observe(leg(y))
    return pl


def asked(pl: PathLog) -> dict:
    return pl.ask("end", {"k": 3, "clock": ""}, [ASKED])["q1"]


def two_ways() -> list[tuple[str, float, list[tuple[float, float]]]]:
    """Three scripts that docked along y = 5 and put an item near (10.2, 5.2) each, spread by hand within 0.45 m of the
    first (10.0, 5.0), (10.2, 5.4) and (10.4, 5.2); three that docked along y = 20 and put theirs near (10.2, 20.2)."""
    return [("a0", 5.0, [(10.0, 5.0)]), ("a1", 5.0, [(10.2, 5.4)]), ("a2", 5.0, [(10.4, 5.2)]),
            ("b0", 20.0, [(10.0, 20.0)]), ("b1", 20.0, [(10.2, 20.4)]), ("b2", 20.0, [(10.4, 20.2)])]


def test_the_claim_lists_the_recurring_places_within_reach_of_the_anchor_and_no_others():
    """Contract (0.6.0 repair, the reviewer's `task_things` shortcut): `spots_claim` lists the places the records' golds
    put items that lie within the predicate's reach of the anchor, those within `MATCH_M` (0.5 m) of one another merged
    into their mean, sorted, and the empty list when none is in reach. Hand values, the anchor the leg along y = 5 from
    x = 0 to 40 (reach 1.0 m, a spot judged by a 0.05 m footprint): (10.0, 5.3) and (10.2, 5.3) are one spot at (10.1,
    5.3); (30.0, 5.9) is in reach; (30.0, 6.4) is 1.4 m off and (41.1, 5.0) is 1.075 m past the end, so neither is
    listed; (10.0, 20.0) is another way's. Regression caught: a spot outside the reach listed (the claim is the pool's
    places, not the ones near this anchor), the same spot twice (a false positive for every script that put an item
    there), or a spot at the way's end dropped."""
    anchor = anchor_of(PATH, leg(5.0), None)
    golds = [positions_gold((10.0, 5.3), (30.0, 6.4)), positions_gold((10.2, 5.3), (30.0, 5.9), (41.1, 5.0)),
             positions_gold((10.0, 20.0), (40.9, 5.0))]
    records = [{"script": f"s{n}", "gold": g} for n, g in enumerate(golds)]
    assert bpt.spots_claim(PATH, anchor, records) == listed((10.1, 5.3), (30.0, 5.9), (40.9, 5.0))
    assert bpt.spots_claim(PATH, anchor_of(PATH, leg(60.0), None), records) == listed()


def test_path_log_answers_the_spots_near_its_own_anchor_where_the_pool_cannot(tmp_path):
    """Contract (0.6.0 repair): where items recur at spots that depend on the task's anchor, `path_log` answers the
    spots near the anchor it logged, because the rule beats abstaining on a script left out. Hand case: three scripts
    docked along y = 5 and put items around (10.2, 5.2), three along y = 20 around (10.2, 20.2) (`two_ways`). No answer
    fixed over the six golds can score for both ways: one cluster alone is right for three and wrong for three (mean 0),
    and both clusters score 2/3 F1 on each, 1/3 penalised; the spots rule, its claim made from the other scripts' golds
    within the reach of the left-out script's anchor, scores 1 on all six (left out, a0's claim is the mean of a1's and
    a2's, (10.3, 5.3), 0.42 m from its (10.0, 5.0)). So a log along y = 5 is answered (10.2, 5.2), the mean of the
    three, and a log along y = 20 (10.2, 20.2). The same under an `in_pool` prior (the twin). Regression caught: the
    rule left out of the answer, or an answer made from the pool's places whatever the anchor."""
    for twin in (False, True):
        prior = prior_of(two_ways(), in_pool=twin)
        assert asked(log_with(tmp_path / f"a{twin}", prior)) == listed((10.2, 5.2))
        assert asked(log_with(tmp_path / f"b{twin}", prior, y=20.0)) == listed((10.2, 20.2))
    assert bpt.spots_held_out(PATH, prior_of(two_ways()))[0] == pytest.approx(1.0)


def test_the_rule_never_reads_the_golds_of_the_script_it_is_scored_on(tmp_path):
    """Contract (0.6.0 repair, VISION rule 5): a record's score is made from the claim of the other scripts' golds
    alone, whatever other records its own script holds, so a spot only one script knows is never credited. Hand case:
    seven records on the leg along y = 5 from six scripts, each putting one item at its own spot 6 m from the next
    ((0..30, 5.0)), script s0 holding two records with the same item (6.0, 5.0). Left out, a record's claim lists the
    other scripts' five spots and none within 0.5 m of its own item: F1 0, penalised -1, whose mean over the seven is
    -1. Were s0's second record read for the first (or the record's own gold for itself) its claim would hold the item.
    A mean under abstaining's (0) is not answered: `path_log` stays `unknown`. Regression caught: a leave-one-record-out
    round in place of leave-one-script-out, or a rule answered on its in-sample score."""
    rows = [("s0", 5.0, [(6.0, 5.0)]), ("s0", 5.0, [(6.0, 5.0)]), ("s1", 5.0, [(0.0, 5.0)]),
            ("s2", 5.0, [(12.0, 5.0)]), ("s3", 5.0, [(18.0, 5.0)]), ("s4", 5.0, [(24.0, 5.0)]),
            ("s5", 5.0, [(30.0, 5.0)])]
    prior = prior_of(rows)
    assert bpt.spots_held_out(PATH, prior)[0] == pytest.approx(-1.0)
    assert asked(log_with(tmp_path, prior)) == UNKNOWN


def test_a_pool_answer_that_beats_the_rule_stands(tmp_path):
    """Contract (0.6.0 repair): `path_log` answers with the best of its candidates, so a pool answer that scores higher
    on the scripts left out than the spots rule is kept, under the held-out prior and the `in_pool` one (whose pool is
    held to its own best response). Hand case: six scripts docked along y = 5 and each put one item at (10.0, 8.0), 3 m
    off the way, past the 1 m reach. The rule lists nothing near the anchor, the empty list, F1 0 on every gold, -1; the
    pool's densest place (10.0, 8.0) scores 1 on every script left out. So the answer is that place. Regression caught:
    the rule answered whenever it has a claim, over a pool answer that is better (a rule that can only lose, here)."""
    rows = [(f"s{n}", 5.0, [(10.0, 8.0)]) for n in range(6)]
    assert bpt.spots_held_out(PATH, prior_of(rows))[0] == pytest.approx(-1.0)
    for twin in (False, True):
        assert asked(log_with(tmp_path / f"t{twin}", prior_of(rows, in_pool=twin))) == listed((10.0, 8.0))


def test_path_log_answers_a_set_family_only_where_its_held_out_gain_is_sure(tmp_path):
    """Contract (0.6.0 repair, the AMR's `task_things` BROKEN line): `path_log`'s pooled answer for a set family
    (`task_things` here) is kept only where its held-out gain over abstaining has an interval over the scripts above 0,
    as `blind.learned_answer` asks of every learned answer; the `in_pool` twin is in-sample by design and answers.
    Hand case: five scripts docked along y = 5 and put one item at (10.0, 8.0), 3 m off the way (past the 1 m reach, so
    the spots rule has no claim), a sixth put its item at (30.0, 8.0). The pool's best response is (10.0, 8.0): left
    out of a script of the five it scores 1, left out of the sixth -1, a held-out mean of (5 - 1) / 6 = +0.67; but the
    gains are 1, 1, 1, 1, 1 and -1, and a resample of six holds the sixth script three times or more (a gain of 0 or
    under) with probability 6.2%, so the interval is not above 0 and `path_log` is `unknown`. With the sixth also at
    (10.0, 8.0) every gain is 1 and it answers. Regression caught: the pooled answer kept on a positive average
    held-out gain, or the bar applied to the twin."""
    rows = [(f"s{n}", 5.0, [(10.0, 8.0)]) for n in range(5)]
    odd = [*rows, ("s5", 5.0, [(30.0, 8.0)])]
    assert asked(log_with(tmp_path / "odd", prior_of(odd))) == UNKNOWN
    assert asked(log_with(tmp_path / "odd_twin", prior_of(odd, in_pool=True))) == listed((10.0, 8.0))
    agree = [*rows, ("s5", 5.0, [(10.0, 8.0)])]
    assert asked(log_with(tmp_path / "agree", prior_of(agree))) == listed((10.0, 8.0))


def test_the_spots_rule_is_answered_only_where_its_held_out_gain_is_sure(tmp_path):
    """Contract (0.6.0 repair): the recurring-spots rule is held to the bar of every learned answer, the lower end of
    its held-out gain over abstaining (an interval over scripts) above 0, and not to a positive mean alone. Hand case:
    six scripts docked along y = 5, five put one item at (10.0, 5.0) and a sixth at (30.0, 5.0), both within the 1 m
    reach. Left out of one of the five, the claim lists both spots (four and the sixth): F1 2/3, penalised 1/3;
    left out of the sixth it lists (10.0, 5.0) alone: F1 0, penalised -1. The mean is (5 / 3 - 1) / 6 = 1/9 above
    abstaining's 0, but a resample of six holds the sixth script twice or more (a gain of 0 or under) with probability
    26%, so the interval is not above 0; the pool's own answer fails the same bar (see the test above), so `path_log` is
    `unknown`. Regression caught: the rule answered on a positive held-out mean, which scored under `all_unknown`."""
    rows = [(f"s{n}", 5.0, [(10.0, 5.0)]) for n in range(5)] + [("s5", 5.0, [(30.0, 5.0)])]
    mean, low = bpt.spots_held_out(PATH, prior_of(rows))
    assert mean == pytest.approx(1 / 9) and low <= 0
    assert asked(log_with(tmp_path, prior_of(rows))) == UNKNOWN


def test_the_in_pool_twin_answers_the_spots_rule_without_the_bar(tmp_path):
    """Contract (0.6.0 repair): the bar on the spots rule is the held-out prior's; an `in_pool` prior (the twin,
    in-sample by design) chooses the rule as before, on its mean beating the pool's own best response. Hand case:
    `two_ways` and a seventh script, `c`, that docked along y = 5 and put its item at (30.0, 5.0). Left out of one
    of the three along y = 5, the claim lists their other two's mean (10.3, 5.3) and (30.0, 5.0): F1 2/3, penalised
    1/3 each, 1 in all; left out of one of the three along y = 20, the claim lists the other two's mean, 1 each, 3
    in all; left out of `c` the claim lists the cluster along y = 5 alone: F1 0, -1. The mean is (1 + 3 - 1) / 7 =
    3/7; the gains are 1/3, 1/3, 1/3, 1, 1, 1 and -1, and a resample of seven holds `c` three times or more (a mean
    of 0 or under) with probability 6.5%, so the interval is not above 0. The held-out prior abstains (the pool's
    answer is `unknown` as well), the twin answers the spots rule along y = 5: the cluster's mean (10.2, 5.2) and
    (30.0, 5.0). Regression caught: the twin held to the held-out bar, which makes the gate's larger-of-two score
    vacuous."""
    rows = [*two_ways(), ("c", 5.0, [(30.0, 5.0)])]
    mean, low = bpt.spots_held_out(PATH, prior_of(rows))
    assert mean == pytest.approx(3 / 7) and low < 0
    assert asked(log_with(tmp_path / "held", prior_of(rows))) == UNKNOWN
    assert asked(log_with(tmp_path / "twin", prior_of(rows, in_pool=True))) == listed((10.2, 5.2), (30.0, 5.0))
