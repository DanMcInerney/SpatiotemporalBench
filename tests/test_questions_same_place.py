"""`same_place` (questions_change): was a thing in the same place in two named tasks, by hand on two tasks of
`vacuum_run` (28 keyframes each). A bag is seen from the camera at (2.13, 1.5, 0.08) at keyframes 3 and 31; the render
of the last look window of each task (keyframes 19 to 21 and 47 to 49) sees the floor ahead of the front edge at x = 0,
so the place (1.0, 0.0) is seen empty in task 1 (by keyframe 49) and (6.0, 2.0) is not."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import change_draw_kit as dk
import change_kit as ck
from change_draw_kit import only
from change_kit import GAP_S, thing
from space_kit import vacuum_run
from stbench import questionnaire
from stbench.questions_change import GOLDS, MOVED, VACATED, YES, same_place
from stbench.questions_text import COINS, FAMILIES, SLICED
from stbench.records import BALANCE_POOLS, pool_of
from stbench.robots import ROBOTS
from stbench.seeding import stable_hex

SEEN = dk.SEEN


def script(base_z: float = 0.0, height: float = 0.314):
    """Two tasks and bags that each stand in a place in task 0 and another in task 1 (the moves are in the gap):
    - E0001 red 5.13 -> 5.13 and E0002 blue 3.13 -> 3.23 (0.10 m): the same place (yes);
    - E0003 green 5.13 -> 5.73 (0.60 m): moved, past the 0.50 m of twice the job's unit;
    - E0004 purple 3.13 -> 3.33 (0.20 m) and E0005 orange 5.13 -> 5.62 (0.49 m): between the two, not asked;
    - E0006 yellow (1.0, 0.0) -> (7.0, 2.5), `base_z` up and `height` high, seen in task 0 only, its place seen empty in
      task 1: vacated;
    - E0007 pink (6.0, 2.0) -> (7.5, 2.5), seen in task 0 only, its place never seen in task 1: not asked;
    - E0008 teal (1.0, 0.5) -> (7.0, 1.0), seen in task 0, a 500 px glimpse of it in task 1: not asked."""
    s = vacuum_run(2)
    for eid, colour, a, b in (("E0001", "red", (5.13, 1.5), (5.13, 1.5)), ("E0002", "blue", (3.13, 1.5), (3.23, 1.5)),
                              ("E0003", "green", (5.13, 1.5), (5.73, 1.5)),
                              ("E0004", "purple", (3.13, 2.0), (3.33, 2.0)),
                              ("E0005", "orange", (5.13, 2.0), (5.62, 2.0))):
        thing(s, eid, "bag", colour, [a, b], SEEN)
    s.item("E0006", "bag", {"color": "yellow"}, [(0.0, GAP_S, (1.0, 0.0, base_z)), (GAP_S, None, (7.0, 2.5, 0.0))],
           size=(0.374, 0.304, height))
    s.see(3, "E0006", 1500)
    thing(s, "E0007", "bag", "pink", [(6.0, 2.0), (7.5, 2.5)], [3])
    thing(s, "E0008", "bag", "teal", [(1.0, 0.5), (7.0, 1.0)], [3])
    s.see(31, "E0008", 500)
    return s


def window(s=None, job=None):
    return ck.window(script() if s is None else s, 0, 1, job=job or only("same_place"))


def rows(w) -> dict[str, list[tuple]]:
    """The candidates of the pool of each side of the first draw as (entity, class, answer, evidence, entities)."""
    first = same_place(w)[0]
    return {side: sorted((c.subjects[0][2:], c.answer_class, c.gold["value"], c.evidence_k, c.entities)
                         for c in pool) for side, pool in (("yes", first.answerable), ("no", first.unknown))}


def test_yes_moved_and_vacated_candidates_with_their_evidence():
    """Contract: for two tasks i < j, a thing with a place in i (its last) and a first place in j under 0.15 m apart is
    a yes, one at least max(0.45 m, twice the job's unit 0.25 m) = 0.50 m apart is a moved no, one with a place in i and
    no sighting at all in j whose place was seen empty in j (`grid.place_free` over j's cells) is a vacated no; the
    places between, a place never seen empty, and a thing glimpsed in j are not candidates. Hand values on `script`: yes
    E0001 and E0002 (evidence keyframes 3 and 31), no E0003 (0.60 m, moved, evidence 3 and 31) and E0006 (vacated:
    evidence 3 and keyframe 49, the render that saw (1.0, 0.0) empty, no entity named for it); not E0004 (0.20 m), E0005
    (0.49 m), E0007 (place never seen empty) or E0008 (glimpsed in task 1). Regression caught: a threshold off by the 2u
    or the between-gap, a vacated no without the space seen empty (an invented fact), or a glimpsed thing declared
    gone."""
    assert rows(window()) == {
        "yes": [("E0001", YES, True, (3, 31), ("E0001", "E0001")), ("E0002", YES, True, (3, 31), ("E0002", "E0002"))],
        "no": [("E0003", MOVED, False, (3, 31), ("E0003", "E0003")), ("E0006", VACATED, False, (3, 49), ())]}


def vacated(w) -> list[str]:
    """The entities of the vacated candidates of the window's first draw."""
    return sorted(c.subjects[0][2:] for c in same_place(w)[0].unknown if c.answer_class == VACATED)


def test_a_vacated_no_needs_a_robot_whose_band_can_see_the_thing_gone():
    """Contract: a vacated no is asked only when the robot's free-space cells can show the thing gone: the robot does
    not fly (a flier reads every ground point as floor, so a box seen from the air reads empty), the thing stood on the
    floor the robot drives on (`World.on_floor`) and its top is above the band's floor (`grid.band`: the vacuum's is
    0.02 m; a flatter thing leaves no obstacle point). Hand-worked on `script`, whose yellow E0006 (a bag, 0.314 m high,
    on the floor) is vacated: with the world's robot a drone, with E0006 on a table (base 0.8 m) or with E0006 a mat
    0.01 m high, nothing is vacated, and the moved no (E0003) is unchanged. Regression caught: a vacated gold that says
    the thing left while it stands there (the drone's ground band, a shelf above the band, a rug under it)."""
    assert vacated(window()) == ["E0006"]
    w = window()
    w.robot = ROBOTS["drone"]
    assert vacated(w) == [] and rows(w)["no"][0][0] == "E0003"

    def with_yellow(base_z: float, height: float):
        return window(script(base_z, height))

    assert vacated(with_yellow(0.0, 0.314)) == ["E0006"]
    assert vacated(with_yellow(0.8, 0.314)) == [] and vacated(with_yellow(0.0, 0.01)) == []


def test_a_thing_in_view_at_the_end_is_not_asked_about_its_place_in_two_tasks():
    """Contract: as for `object_was` and `object_where`, a thing with any visibility row in the window's last look
    window is not asked, since the current view would hold its place and lift `current_view` above the floor.
    Hand-worked on `script` with the red E0001, a yes candidate, also seen as a 200 px glimpse at keyframe 52 (the last
    look window is 50 to 55): E0002 is the only yes. Regression caught: `same_place` naming a thing the robot sees at
    the end."""
    s = script()
    s.see(52, "E0001", 200)
    w = window(s)
    assert w.in_end_window("E0001") and [r[0] for r in rows(w)["yes"]] == ["E0002"]


def test_the_first_place_in_a_task_is_contradicted_by_a_glimpse_elsewhere_before_it():
    """Contract: a thing's first place in task j is its first recognisable sighting only when nothing was sighted
    elsewhere, recognisable or not, from the task's start to it, since the memory table's first sighting is that glimpse
    and the gold must be the table's. Hand-worked on a navy bag at (5.13, 1.5) in task 0 (seen at keyframe 3), moved to
    (7.13, 1.5) in the gap, glimpsed there at 500 px at keyframe 29 (offset 1 of task 1), and back at (5.13, 1.5) from
    t 7203 s, recognisable at keyframe 31: with the glimpse the first place is contradicted and the bag is no candidate;
    without it the bag is a yes. Regression caught: the rule left out, a yes whose memory-table answer (the glimpse, 2 m
    away) is no."""
    def navy_yes(glimpse: bool) -> bool:
        s = script()
        thing(s, "E0009", "bag", "navy", [(5.13, 2.0), (7.13, 2.0), (5.13, 2.0)], [3, 31], gaps=(GAP_S, 7203.0))
        if glimpse:
            s.see(29, "E0009", 500)
        return "E0009" in [r[0] for r in rows(window(s))["yes"]]

    assert navy_yes(False) and not navy_yes(True)


def test_a_candidate_is_a_full_row_and_its_classes_fit_the_pools():
    """Contract: a `same_place` candidate has slots `kind`, `attributes` and `tasks` (the two case task indices, the
    earlier first), a yes/no gold with slice `answerable` and no tolerances or range, the thing as its one subject (no
    named time), and the class of its pool: `yes` in the first pool and `moved` or `vacated` in the second, which is how
    `records.BALANCE_POOLS` reads them. Hand values: E0003's slots are bag, green, tasks [0, 1]; a no is `value` false.
    Regression caught: a named time among the subjects (rule 6), a class outside its pool (the balance check would find
    the share by class wrong), or a range on a row that has no rungs."""
    first = same_place(window())[0]
    assert {pool_of("same_place", c.answer_class) for c in first.answerable} == {"yes"}
    assert {pool_of("same_place", c.answer_class) for c in first.unknown} == {"no"}
    assert set(BALANCE_POOLS["same_place"]) == {"yes", "no"}
    (c,) = [c for c in first.unknown if c.subjects == ("e:E0003",)]
    assert c.slots == {"kind": "bag", "attributes": {"color": "green"}, "tasks": [0, 1]}
    assert (c.slice, c.tolerances, c.range_m, c.critical, c.family) == ("answerable", {}, None, True, "same_place")


def test_the_no_distance_is_twice_the_jobs_unit_and_widens_with_the_kind(monkeypatch):
    """Contract: a moved no needs max(0.45 m, 2u) plus the kind's widening. Hand values on E0005, which moves 0.49 m:
    with the home's unit 0.25 m the bar is 0.50 m and it is no candidate; with a job whose ladder starts at 0.125 m the
    bar is 0.45 m and it is a moved no; and with the bag's catalog size replaced by 1.0 x 0.4 x 0.5 (widening 0.25 m)
    the bar is 0.75 m and E0003 (0.60 m) is no longer one. Regression caught: the fixed 0.45 m of 0.5.0 (the home's unit
    ignored), or a widening dropped."""
    def asked(w):
        return {e for _, _, _, _, ents in sum(rows(w).values(), []) for e in ents}

    assert "E0005" not in asked(window())
    fine = replace(only("same_place"), ladder=(0.125, 0.25, 0.5, 1.0))
    assert "E0005" in asked(window(job=fine)) and "E0004" not in asked(window(job=fine))
    monkeypatch.setattr("stbench.questions_change.item_size", lambda kind: (1.0, 0.4, 0.5))
    assert "E0003" not in asked(window())


def test_a_moment_is_a_task_pair_a_kind_and_a_draw_and_one_draw_a_thing_of_the_smaller_pool():
    """Contract: the moments are "<i>|<j>|<kind>|<draw>", one draw for each thing of the smaller pool, each holding both
    pools whole, and only for a kind with a yes and a no candidate in that pair. Hand values on `script`: bags have two
    yes and two no things, so draws 0|1|bag|0 and 0|1|bag|1; dropping the vacated bag leaves one no and one draw; a
    window of one task has no pair, and a window of three tasks has a moment for each of its three pairs. Regression
    caught: a moment per kind only (one question a pair however many things), a draw per candidate, or a task pair
    outside the window."""
    assert [m.key for m in same_place(window())] == ["0|1|bag|0", "0|1|bag|1"]
    s = vacuum_run(2)
    thing(s, "E0001", "bag", "red", [(5.13, 1.5), (5.13, 1.5)], SEEN)
    thing(s, "E0002", "bag", "blue", [(3.13, 1.5), (3.23, 1.5)], SEEN)
    thing(s, "E0003", "bag", "green", [(5.13, 1.5), (5.73, 1.5)], SEEN)
    assert [m.key for m in same_place(window(s))] == ["0|1|bag|0"]
    assert same_place(ck.window(s, 0, 0, job=only("same_place"))) == []
    t = vacuum_run(3)
    thing(t, "E0001", "bag", "red", [(5.13, 1.5)] * 3, [3, 31, 59])
    thing(t, "E0003", "bag", "green", [(5.13, 2.0), (5.73, 2.0), (6.33, 2.0)], [3, 31, 59])
    keys = [m.key for m in same_place(ck.window(t, 0, 2, job=only("same_place")))]
    assert keys == ["0|1|bag|0", "0|2|bag|0", "1|2|bag|0"]


def test_each_draw_is_asked_in_the_pool_its_coin_gives():
    """Contract: every draw is asked once, in the pool the hand coin (`stable_hash(case id, family, key, "slice") % 100
    < 50`: the no pool) gives it, and the asked yes count is the number of draws whose coin is the yes side. Hand-worked
    on `script` over 12 case ids (two draws): two questions each, a yes for each draw whose coin lands on the first
    pool. Regression caught: `same_place` out of `COINS` (the no pool at 20%, so an all-yes notebook scores), or a draw
    whose coin is the case's alone."""
    for n in range(12):
        cid = stable_hex("same", n, n=16)
        asked = questionnaire.end(ck.window(script(), 0, 1, cid=cid, job=only("same_place")))
        no = sum(dk.hand_coin(cid, "same_place", f"0|1|bag|{j}") for j in range(2))
        assert len(asked) == 2 and len({c.subjects for c in asked}) == 2, cid
        assert Counter(c.answer_class == YES for c in asked) == {k: v for k, v in ((True, 2 - no), (False, no)) if v}


def test_the_yes_share_is_a_fair_coin_over_two_thousand_cases():
    """Contract: over 2,000 synthetic case ids the no share of a `same_place` draw is within 0.03 of one half (the
    binomial standard deviation is 0.011), so a notebook that always says yes scores about zero. The family is a coin
    family. Regression caught: `same_place` out of `COINS`, or the yes share tilted by a key that fixes the coin."""
    assert "same_place" in COINS and "same_place" not in SLICED and "same_place" in GOLDS
    ids = [stable_hex("share", i, n=16) for i in range(2000)]
    share = sum(questionnaire.unknown_slice(cid, "same_place", "0|1|bag|0") for cid in ids) / len(ids)
    assert abs(share - 0.5) <= 0.03


def test_same_place_is_temporal_with_an_exact_credit_and_its_own_words():
    """Contract: `same_place` is a temporal end family of `memory`, a yes/no with an exact credit, with the slots kind,
    attributes and tasks, and its text names two tasks by ordinal. Hand text: 'Was the red bag in the same place in your
    second task as in your first?' for tasks [0, 1] and 'Was the red bag in the same place in your third task as in your
    second?' for [1, 2]. Regression caught: a wording that names an event of the job, or a pair of tasks given in the
    wrong order."""
    spec = FAMILIES["same_place"]
    assert (spec.when, spec.scope, spec.answer_type, spec.credit, spec.temporal) == (
        "end", "memory", "yesno", "exact", True)
    assert spec.slots == ("kind", "attributes", "tasks")
    slots = {"kind": "bag", "attributes": {"color": "red"}, "tasks": [0, 1]}
    assert spec.render(slots) == "Was the red bag in the same place in your second task as in your first?"
    assert spec.render({**slots, "tasks": [1, 2]}) == (
        "Was the red bag in the same place in your third task as in your second?")
