"""The table systems' and `path_log`'s answers to `drop_zone_people` and `tangle_ahead` (`table_hazard`,
`baselines_path_hazard`), and that the oracle reproduces the golds over a compiled case. Hand-worked: a vacuum at (1.87,
1.5) heading east has its camera at (2.0, 1.5); one at (3.0, 1.0) heading north at (3.0, 1.13); a person is a 0.34 x
0.56 m footprint and a cord 0.5 x 0.18 m."""

from __future__ import annotations

import hazard_kit as hk
import person_kit as pk
from baseline_kit import EVENTS, ROBOT, det, play, question, scores, started, step
from ego_kit import K, put, scene
from stbench import episode as ep
from stbench import questions, questions_hazard, table_answers
from stbench.baselines import MemoryTable, Oracle
from stbench.baselines_path import PathLog
from stbench.robots import ROBOTS
from stbench.score import penalised
from stbench.table import Store


def rect(x, y, sx, sy):
    return [[x - sx / 2, y - sy / 2], [x + sx / 2, y - sy / 2], [x + sx / 2, y + sy / 2], [x - sx / 2, y + sy / 2]]


def person(key, x, y, hull):
    return {**det(key, "person", "", x, y, hull=hull), "attributes": {}}


def thing(key, kind, x, y, z=0.1, hull="rect"):
    """A detection of `kind` at (x, y, z) with the hull of a cord's footprint there (None: not recognisable)."""
    return det(key, kind, "black", x, y, z, hull=rect(x, y, 0.5, 0.18) if hull == "rect" else hull)


def positions(*xy):
    return {"type": "positions", "items": [{"xy": list(p)} for p in xy]}


def ask(store, template, k=0):
    ctx = {"now": store.steps[k], "robot": ROBOT}
    return table_answers.answer(store, question(template, {}, answer_type="positions"), ctx)


def fed(*frames, yaw=90.0, base=(3.0, 1.0)):
    store = Store()
    store.begin_task(0, "2026-03-02T08:00:00", 0.0, "task 0")
    for k, dets in enumerate(frames):
        store.add_step(step(k, *base, yaw, detections=dets, identity={d["d"]: "E" + d["d"] for d in dets}))
    return store


def test_drop_zone_people_lists_the_centroids_of_the_hulls_within_4_m():
    """Contract (`table_hazard`): the floor centroid of the hull, in the ask's frame, of every person track whose hull
    has a point within 4 m of the camera's floor point, sorted, to 4 decimals. By hand from (2.0, 1.5): A at (3.5, 1.5)
    has its nearest point (3.33, 1.5), 1.33 m; R at (2.0, 5.0) (2.0, 4.72), 3.22 m; Q at (6.5, 1.5) (6.33, 1.5), 4.33 m,
    out. A person detected without a hull and a bag with one count for nothing. Regression caught: a person outside the
    zone, a thing taken for a person, a glimpse taken for evidence, or the hull's corner in place of its centroid."""
    store = fed([person("a", 3.5, 1.5, rect(3.5, 1.5, 0.34, 0.56)), person("r", 2.0, 5.0, rect(2.0, 5.0, 0.34, 0.56)),
                 person("q", 6.5, 1.5, rect(6.5, 1.5, 0.34, 0.56)), person("g", 3.0, 2.0, None),
                 det("t", "bag", "red", 2.5, 1.5, hull=rect(2.5, 1.5, 0.3, 0.3))], yaw=0.0, base=(1.87, 1.5))
    assert ask(store, "drop_zone_people") == positions((2.0, 5.0), (3.5, 1.5))
    assert ask(fed([], yaw=0.0, base=(1.87, 1.5)), "drop_zone_people") == positions()


def test_tangle_ahead_lists_the_middles_of_the_tangle_things_last_recognised_in_reach():
    """Contract: the middle of every track of a tangle kind whose last hull has its nearest point in reach of the
    camera's floor point and heading and that stood on the floor then. By hand from (3.0, 1.13) heading north: the cord
    C1 at (3.0, 1.8) is 0.58 m ahead and is listed; C2 at (3.0, 2.6) is 1.38 m away and C4 at (3.0, 0.2) is behind; the
    bath mat (a kind that recurs at one spot), a cord at z 0.6 (its base 0.5 m up, off the floor), a cord with no hull
    (not recognisable) and a person are not listed. Regression caught: a layout prior listed, a thing on a shelf listed,
    a thing never recognised listed, or the reach measured from the base."""
    store = fed([thing("c1", "extension_cord", 3.0, 1.8), thing("c2", "extension_cord", 3.0, 2.6),
                 thing("c4", "extension_cord", 3.0, 0.2), thing("m", "bath_mat", 3.0, 1.6),
                 thing("s", "extension_cord", 3.0, 1.5, z=0.6), thing("n", "extension_cord", 2.5, 1.5, hull=None),
                 person("p", 3.0, 1.9, rect(3.0, 1.9, 0.34, 0.56))])
    assert ask(store, "tangle_ahead") == positions((3.0, 1.8))
    assert ask(fed([]), "tangle_ahead") == positions()


def test_a_tangle_thing_seen_elsewhere_later_has_moved_and_is_left_out_as_for_the_gold():
    """Contract: a track whose later detections put it 0.15 m or more from the place of its last hull has moved since
    and is not listed; one 0.1 m off is listed at the place of the hull. By hand: the cord seen at (3.0, 1.8) with a
    hull at step 0 and detected without one at (3.0, 1.5) at step 1 has moved 0.3 m; at (3.0, 1.9) 0.1 m. Regression
    caught: a place the robot knows is stale listed as a tangle ahead."""
    moved = fed([thing("c", "extension_cord", 3.0, 1.8)], [thing("c", "extension_cord", 3.0, 1.5, hull=None)])
    assert ask(moved, "tangle_ahead", 1) == positions()
    settled = fed([thing("c", "extension_cord", 3.0, 1.8)], [thing("c", "extension_cord", 3.0, 1.9, hull=None)])
    assert ask(settled, "tangle_ahead", 1) == positions((3.0, 1.8))


def case_of(path, s, cands):
    case = s.write_case(path)
    qs, gs = questions.records(cands)
    ep.write_jsonl(case.questions, qs, sort_keys=True)
    ep.write_jsonl(case.gold, gs)
    return case, qs[0]["qid"], gs[0]


def test_the_oracle_reproduces_the_tangle_gold_over_a_compiled_case(tmp_path):
    """Contract (the ceiling gate): over a compiled case the oracle and `memory_table` answer the `tangle_ahead` gold
    exactly (score 1) because the channel's floor hulls of the recognisable things are the gold's. Hand case: two cords
    in reach (C1 at (3.0, 1.8), C3 at (2.2, 1.13)), a bath mat in reach (a don't-care circle in the gold, nothing in the
    answer), and a cord out of reach. Regression caught: the oracle channel drifting from the gold's sightings, the
    table and the gold reading the kinds or the floor differently."""
    s = scene()
    for eid, xy in (("C1", (3.0, 1.8)), ("C3", (2.2, 1.13)), ("C4", (3.0, 2.6))):
        put(s, eid, "extension_cord", "black", xy, [K], size=(0.5, 0.18, 0.03), mobility="clutter")
    put(s, "M", "bath_mat", "beige", (3.0, 1.6), [K], size=(0.4, 0.6, 0.015), mobility="clutter")
    (cand,) = questions_hazard.tangle_ahead(hk.tangle_world(s), K).answerable
    case, qid, row = case_of(tmp_path, s, [cand])
    assert row["gold"]["items"] == [{"xy": [2.2, 1.13]}, {"xy": [3.0, 1.8]}] and row["gold"]["dont_care"]
    for cls in (Oracle, MemoryTable):
        run, _ = play(case, cls, events=EVENTS)
        assert run.answers[qid] == {"type": "positions", "items": row["gold"]["items"]}, cls.__name__
        assert set(scores(case, run).values()) == {1.0}, cls.__name__


def test_the_oracle_table_reproduces_the_drop_zone_gold():
    """Contract (the ceiling gate): the table that follows the oracle channel (`table_hazard`) answers the
    `drop_zone_people` gold exactly, because the channel's person hulls are the gold's evidence sightings. Hand case:
    the drone over (2.0, 1.5) with a person at (3.5, 1.5) and one at (2.0, 5.0) in the zone and one at (6.5, 1.5)
    outside, and a 500 px glimpse at (3.0, 2.0) that is a don't-care circle in the gold and nothing in the answer; the
    answer scores 1 against the gold row. Regression caught: the oracle channel's person hull differing from the gold's,
    or the zone read from another point."""
    s = hk.drone()
    for pid, xy in (("A", (3.5, 1.5)), ("R", (2.0, 5.0)), ("Q", (6.5, 1.5))):
        pk.put_person(s, pid, xy, [hk.LOOK])
    pk.put_person(s, "B", (3.0, 2.0), [hk.LOOK], px=500)
    (cand,) = questions_hazard.drop_zone_people(hk.zone_world(s, 11), hk.LOOK).answerable
    (_,), (row,) = questions.records([cand])
    assert row["gold"]["items"] == [{"xy": [2.0, 5.0]}, {"xy": [3.5, 1.5]}] and row["gold"]["dont_care"]
    store = hk.store_of(s, hk.LOOK)
    answer = table_answers.answer(store, question("drop_zone_people", {}, answer_type="positions"),
                                  {"now": store.steps[hk.LOOK], "robot": ROBOTS["drone"]})
    assert answer == {"type": "positions", "items": row["gold"]["items"]} and penalised(answer, row) == 1.0


def test_path_log_has_no_picture_and_abstains_without_other_scripts(tmp_path):
    """Contract (`baselines_path_hazard`): `path_log` reads no picture, so with no prior it answers `unknown` to both
    families, and they are registered for it. Regression caught: a family complete for the table systems and unreachable
    for the shortcut that gates it."""
    system = started(PathLog, tmp_path, kinds=("kitchen",), events=EVENTS, prior=[])
    system.task(0, "2026-03-02T08:00:00", 0.0, "Clean the kitchen.")
    system.observe([step(0, 3.0, 1.0, 90.0)])
    asks = [question("drop_zone_people", {}, "q1", "positions"), question("tangle_ahead", {}, "q2", "positions")]
    assert system.ask("during", {"k": 0, "clock": ""}, asks) == {"q1": {"type": "unknown"}, "q2": {"type": "unknown"}}
