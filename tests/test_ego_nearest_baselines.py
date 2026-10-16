"""The references and shortcuts of `ego_now` and `ego_mem` on the nearest-part gold (FD-0.7.0 review, changes 7-8): the
oracle and `memory_table` read the same floor hulls the gold takes, `kind_range` keeps the bearing of the nearest part,
and `path_log` answers the nearest point of the cluster of nearest parts it learned. Hand-worked from `ego_kit`'s scene
(the look point 13 at (3.0, 1.13, 0.08) heading north, the look point 27 heading north-east)."""

from __future__ import annotations

import pytest

from baseline_kit import EVENTS, offset, play, question, record, started, step
from ego_kit import K, put, scene, world
from kind_range_kit import ego_question, prior_of, seen_at
from stbench import episode as ep
from stbench import questions
from stbench.baselines import FirstTable, KindRange, LatestTable, MemoryTable, Oracle
from stbench.baselines_path import PathLog
from stbench.questions_ego import ego_mem, ego_now

WEST, SOUTH, NORTH = ([(4.95, 3.08, 0.0), (4.95, 3.18, 0.0)], [(4.95, 3.08, 0.0), (5.05, 3.08, 0.0)],
                      [(4.95, 3.18, 0.0), (5.05, 3.18, 0.0)])


def fixture_case(tmp_path):
    """A case whose ego questions are every answerable candidate and the first real and absent unknown ones at the look
    points 13 and 27 of a hand scene: A, a red bag at (5.0, 3.13), seen at keyframes 2 to 5 as three faces of its
    footprint (west, south, north, west), so only the union of its hulls holds its nearest corner; M, a green toy
    that stood at (2.0, 2.2) when seen at 2 and stands at (6.0, 2.5) when seen at 4 and 5, whose old place the union
    leaves out; B, a blue bag at (1.0, 1.13) seen at 13 whose nearest point is on the middle of its east edge; U, a
    yellow bag nobody saw."""
    s = scene()
    put(s, "A", "bag", "red", (5.0, 3.13), [2, 3, 4, 5], pts=lambda k: {2: WEST, 3: SOUTH, 4: NORTH, 5: WEST}[k])
    put(s, "M", "toy", "green", (0, 0), [2, 4, 5], where=[(0, 3.75, (2.0, 2.2, 0.0)), (3.75, None, (6.0, 2.5, 0.0))])
    put(s, "B", "bag", "blue", (1.0, 1.13), [K])
    put(s, "U", "bag", "yellow", (6.0, 3.0))
    w = world(s)
    cands = list(ego_now(w, K).answerable)
    for k in (K, 27):
        moment = ego_mem(w, k)
        cands += [*moment.answerable, next(c for c in moment.unknown if c.subjects[0].startswith("e:")),
                  next(c for c in moment.unknown if c.subjects[0].startswith("d:"))]
    case = s.write_case(tmp_path)
    qs, gs = questions.records(cands)
    ep.write_jsonl(case.questions, qs, sort_keys=True)
    ep.write_jsonl(case.gold, gs)
    return case, gs


def test_the_oracle_and_memory_table_equal_the_gold_on_every_ego_question_from_the_channel(tmp_path):
    """Contract (FD-0.7.0 review, change 8): over a compiled case the oracle and `memory_table`, which read only the
    `detections` channel (each recognisable detection's `floor_hull`), answer every ego question with the gold's own
    offset, not merely a credited one: the nearest point of a hull edge (B: 0.0 ahead, 1.95 left), the corner of the
    union of three sightings (A: 1.95 ahead, 1.95 right; the last hull alone is 2.05 ahead), a thing at its new place
    only (M), and the unknown ones. The offsets agree to 1e-4, the gold's own rounding. Regression caught: a table that
    answers the detection's middle (the old answer), the latest hull alone, a hull from a place the thing has left, or a
    channel that cannot carry what the gold reads (the oracle gate then fails or is a tautology)."""
    case, golds = fixture_case(tmp_path)
    offsets = [g for g in golds if g["gold"]["type"] == "offset"]
    assert {g["family"] for g in offsets} == {"ego_now", "ego_mem"} and len(offsets) >= 5
    assert any(g["family"] == "ego_mem" and len(g["evidence_k"]) == 4 for g in offsets)   # A: all four sightings
    for cls in (Oracle, MemoryTable):
        run, _ = play(case, cls, events=EVENTS)
        assert run.status == "complete" and run.failures == {}
        for g in golds:
            got = run.answers[g["qid"]]
            if g["gold"]["type"] == "unknown":
                assert got == g["gold"], cls.__name__
            else:
                assert (got["ahead_m"], got["left_m"]) == pytest.approx((g["gold"]["ahead_m"], g["gold"]["left_m"]),
                                                                        abs=1e-4), (cls.__name__, g["family"])
    a_now = next(g for g in offsets if g["family"] == "ego_mem" and g["entities"][0] == "A" and g["evidence_k"][-1] == 5
                 and g["gold"]["ahead_m"] == pytest.approx(1.95, abs=1e-4))
    assert a_now["gold"]["left_m"] == pytest.approx(-1.95, abs=1e-4)


def face_prior(step_m: float, count: int):
    """`count` other scripts, each asked `ego_now` about a chair from the base (x, 1.0) heading north (camera (x, 1.13))
    with x = 2.8 + `step_m` i, whose nearest part was the point of the face y = 4.0 straight ahead of its camera, (x,
    4.0): 2.87 ahead and 0.0 left, on rungs of 0.02, 0.04 and 0.08 m, with a matching radius (`match_m`) of 0.5 m that
    their golds state themselves (ego golds state none and take their last rung, tested in
    `test_path_log_ego_places`)."""
    ask = question("ego_now", {"kind": "chair", "attributes": {}}, answer_type="offset")
    records = [record(ask, offset(2.87, 0.0), script=f"s{i}", when="during", phase="cleaning",
                      base=(round(2.8 + step_m * i, 3), 1.0), yaw=90.0) for i in range(count)]
    for r in records:
        r["gold"]["tolerances"] = {"ladder": [0.02, 0.04, 0.08], "match_m": 0.5}
    return ask, records


def test_path_log_keeps_a_place_only_if_the_rule_it_answers_wins_on_scripts_left_out(tmp_path):
    """Contract (an independent review, finding 1; `baselines_path_ego.learned_place`): the held-out round judges
    the rule that is answered, the cluster's point nearest the record's own camera with the record's script left out,
    not a fixed point. By hand: 21 scripts saw a chair's face y = 4.0 from cameras 2 cm apart along x (2.8 to 3.2) and
    were each asked for the point straight ahead of them, on rungs of 2, 4 and 8 cm. A fixed place, the cluster's mean
    (3.0, 4.0), is up to 20 cm from those golds (a miss, -1, on most rows), so a test on it abstains; the rule's answer
    for a record is the neighbouring script's point, 2 cm off, within the first rung, so the rule wins every held-out
    script and is kept: asked from the base (3.9, 1.0) it answers the member (3.2, 4.0), 2.87 ahead and 0.7 left. With
    five scripts 10 cm apart the neighbour is 10 cm off, outside every rung, so the same rule loses on the scripts left
    out and is not kept: the answer falls to the pool's constant offset (2.87 ahead, 0.0 left). Regression caught: a
    place tested as a fixed point and answered as another rule (kept when the answer loses, dropped when it wins)."""
    ask, close = face_prior(0.02, 21)
    _, wide = face_prior(0.1, 5)

    def ask_from(path, records):
        system = started(PathLog, path, kinds=("kitchen",), events=EVENTS, prior=records)
        system.task(0, "2026-03-02T08:00:00", 0.0, "Clean the kitchen.")
        system.observe([step(0, 3.9, 1.0, 90.0)])
        return system.ask("during", {"k": 0, "clock": ""}, [ask])["q1"]

    got = ask_from(tmp_path / "close", close)
    assert (got["ahead_m"], got["left_m"]) == pytest.approx((2.87, 0.7), abs=1e-3)
    got = ask_from(tmp_path / "wide", wide)
    assert got == offset(2.87, 0.0)


def test_the_time_blind_notebooks_answer_the_ego_questions_without_failing(tmp_path):
    """Contract (`table.LatestStore`, `FirstStore`): the notebooks that keep one sighting a track still answer an ego
    question, since the hulls a track keeps carry their own centres and do not depend on the sightings kept. Hand case:
    the fixture of the test above, run on `latest_table` and `first_table`; every question gets an answer (none fails),
    and a track keeps the hulls of the sightings it keeps and no others. Regression caught: an answerer that reads a
    hull's centre from the thinned sightings (a KeyError on every ego question, which `stbench check` reports as a bug
    of the instrument)."""
    case, golds = fixture_case(tmp_path)
    for cls in (LatestTable, FirstTable):
        run, system = play(case, cls)
        assert run.status == "complete" and run.failures == {} and set(run.answers) == {g["qid"] for g in golds}
        assert any(t.hulls for t in system.store.tracks)
        assert all(set(t.hulls) <= {x[0] for x in t.sightings} for t in system.store.tracks)


def test_kind_range_keeps_the_bearing_of_the_nearest_part_not_the_middle(tmp_path):
    """Contract (FD-0.7.0 review, change 8; FD-0.6.0 §2.1): `kind_range` is the current view's bearing to what the
    question asks for, put at the kind's typical floor range, and the range is read from other scripts' nearest-part
    golds. Hand case: a bin whose floor hull is the segment from 4 ahead and 3 left of the camera to 12 ahead and 3
    left (a long thing seen end-on, its middle 8 ahead) beside other scripts' bins whose nearest parts stood 2, 5, 7, 10
    and 13 m away (median 7): the answer is on the bearing of the nearest end, (4 x 7 / 5, 3 x 7 / 5) = (5.6, 4.2), not
    on the middle's (6.55, 2.46). Regression caught: the bearing of the detection's middle, or a range prior read from
    the middles."""
    system = started(KindRange, tmp_path, prior=prior_of("bin", [(2.0, 0.0), (3.0, 4.0), (0.0, -7.0), (6.0, 8.0),
                                                                 (-5.0, 12.0)]))
    long = seen_at(1, 8.0, 3.0)
    cx, cy = long["camera"]["pos"][:2]
    long["detections"][0]["floor_hull"] = [[round(cx - 3.0, 3), round(cy + 4.0, 3)],
                                           [round(cx - 3.0, 3), round(cy + 12.0, 3)]]
    system.observe([step(0, 1.0, 0.0, 90.0), long])
    assert system.ask("during", {"k": 1, "clock": ""}, [ego_question("bin")])["q1"] == offset(5.6, 4.2)


def test_path_log_answers_the_learned_clusters_point_nearest_the_ask_camera(tmp_path):
    """Contract (FD-0.7.0 review, change 8, `baselines_path_ego`): the nearest parts other scripts' cameras were asked
    for lie along a fixed thing's near face, so `path_log` answers the point of that cluster nearest the ask's camera,
    the nearest point of the footprint as far as the golds have shown it, which keeps the shortcut as strong as the
    layout allows. By hand: five scripts asked from the base (3.0, 1.0) heading north (camera (3.0, 1.13)) saw a chair's
    nearest part at (2.8, 4.0), (2.9, 4.0), (3.0, 4.0), (3.1, 4.0) and (3.2, 4.0): 2.87 ahead and 0.2, 0.1, 0.0, -0.1,
    -0.2 to the left. Asked from (3.9, 1.0) heading north (camera (3.9, 1.13)), with the cluster in view, it answers
    the member (3.2, 4.0): 2.87 ahead and 0.7 left, where the cluster's mean (3.0, 4.0) would give 0.9. Regression
    caught: the mean of the cluster answered (up to the thing's half-width from the nearest part), which makes the
    shortcut weaker than it can be."""
    chair = {"kind": "chair", "attributes": {}}
    ask = question("ego_now", chair, answer_type="offset")
    seen = [record(ask, offset(2.87, round(0.2 - 0.1 * i, 3)), script=f"s{i}", when="during", phase="cleaning",
                   base=(3.0, 1.0), yaw=90.0) for i in range(5)]
    system = started(PathLog, tmp_path, kinds=("kitchen",), events=EVENTS, prior=seen)
    system.task(0, "2026-03-02T08:00:00", 0.0, "Clean the kitchen.")
    system.observe([step(0, 3.9, 1.0, 90.0)])
    got = system.ask("during", {"k": 0, "clock": ""}, [ask])["q1"]
    assert (got["ahead_m"], got["left_m"]) == pytest.approx((2.87, 0.7), abs=1e-3)
