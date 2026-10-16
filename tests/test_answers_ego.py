"""The answers to `ego_now` and `ego_mem`: the table systems (`table_ego`) from the sightings of their tracks, the
oracle and `memory_table` over a compiled case (they must reproduce the gold), and `path_log` (`baselines_path_ego`)
from a prior and the public map. Hand-worked values; the vacuum's camera is mounted 0.13 m ahead of its base and 0.08 m
up."""

from __future__ import annotations

import math

import pytest

from baseline_kit import EVENTS, det, offset, play, question, record, scores, started, step
from ego_kit import K, put, scene, world
from stbench import episode as ep
from stbench import questions, table_answers
from stbench.baselines import CurrentView, MemoryTable, Oracle
from stbench.baselines_path import PathLog
from stbench.questions_ego import ego_mem, ego_now
from stbench.table import Store

SLOTS = {"kind": "bag", "attributes": {"color": "red"}}


def ask(store: Store, template: str, slots: dict, k: int) -> dict:
    return table_answers.answer(store, question(template, slots, answer_type="offset"), {"now": store.steps[k]})


def fed(*steps: tuple[int, tuple, list[dict]]) -> Store:
    """A store fed the steps (k, (x, y, yaw_deg) of the base, detections)."""
    store = Store()
    for k, (x, y, yaw), dets in steps:
        store.add_step(step(k, x, y, yaw, detections=dets))
    return store


def box(x: float, y: float) -> list[list[float]]:
    """The floor hull of a 0.1 m square thing centred at (x, y), as the oracle channel sends it (counter-clockwise)."""
    return [[round(x - 0.05, 3), round(y - 0.05, 3)], [round(x + 0.05, 3), round(y - 0.05, 3)],
            [round(x + 0.05, 3), round(y + 0.05, 3)], [round(x - 0.05, 3), round(y + 0.05, 3)]]


def test_ego_now_is_the_nearest_point_of_the_one_hull_at_the_ask_step_in_its_camera_frame():
    """Contract (FD-0.7.0 review, change 8): `ego_now` is the point of the floor hull of the described thing's unique
    sighting at the step the ask is about that lies nearest the camera, put in that step's camera frame (position and
    heading as delivered, the mount included). By hand: a base at (3.0, 1.0) heading 90 degrees puts the camera at
    (3.0, 1.13); a red bag whose hull is the 0.1 m square about (3.0, 4.13) has its nearest point (3.0, 4.08) on an
    edge: 2.95 ahead and 0.0 left; a blue bag about (1.0, 1.13) has (1.05, 1.13): 0.0 ahead and 1.95 left. A second red
    bag in the frame makes 'the red bag' two sightings: unknown; a red bag seen only at an earlier step is not in this
    frame: unknown; a sighting with no hull (the oracle sends none for a thing that is not recognisable) is unknown
    too. Regression caught: the base used as the camera (3.13), left taken as right, a twin's sighting chosen, the
    hull's vertex in place of its edge, or the centre of the detection used (the old answer)."""
    base = (3.0, 1.0, 90.0)
    store = fed((1, base, [det("a", "bag", "red", 3.0, 4.13, hull=box(3.0, 4.13))]),
                (2, base, [det("b", "bag", "red", 3.0, 4.13, hull=box(3.0, 4.13)),
                           det("c", "bag", "blue", 1.0, 1.13, hull=box(1.0, 1.13)),
                           det("d", "bag", "red", 6.0, 2.0, hull=box(6.0, 2.0))]),
                (3, base, [det("e", "bag", "blue", 1.0, 1.13, hull=box(1.0, 1.13))]),
                (4, base, [det("f", "bag", "green", 2.0, 3.0)]))
    blue = {"kind": "bag", "attributes": {"color": "blue"}}
    assert ask(store, "ego_now", blue, 3) == offset(0.0, 1.95)
    assert ask(store, "ego_now", SLOTS, 1) == offset(2.95, 0.0)
    assert ask(store, "ego_now", SLOTS, 2) == {"type": "unknown"}     # two red bags in the frame
    assert ask(store, "ego_now", SLOTS, 3) == {"type": "unknown"}     # none in this frame
    assert ask(store, "ego_now", {"kind": "bag", "attributes": {"color": "green"}}, 4) == {"type": "unknown"}  # no hull
    assert ask(store, "ego_now", {"kind": "chair", "attributes": {}}, 3) == {"type": "unknown"}


def test_ego_mem_is_the_union_of_the_hulls_at_the_latest_place_in_the_ask_steps_camera_frame():
    """Contract (FD-0.7.0 review, changes 7-8): `ego_mem` is the point nearest the ask's camera of the union of the
    hulls of the track's sightings at the place of its latest hulled one, in the camera frame of the step the ask is
    about, not of the sighting. By hand: a red bag at (5.0, 3.13) seen from the base (2.0, 1.5) at step 1 as the
    west-face edge (4.95, 3.08)-(4.95, 3.18) and at step 2 as the north-face edge (4.95, 3.18)-(5.05, 3.18); at step 0
    it was seen elsewhere (2.0, 2.0), which the union leaves out. Asked at step 3 from (3.0, 1.0) heading 90 degrees
    (camera (3.0, 1.13)) the nearest point of the triangle is its vertex (4.95, 3.08): 1.95 ahead and 1.95 to the right
    (the latest hull alone would give 2.05 ahead); from a base turned to 0 degrees (heading east, camera (3.13, 1.0)) it
    is 1.82 ahead and 2.08 to the left. A thing never seen is unknown. Regression caught: the latest hull alone, a hull
    from before a move in the union, or the offset taken in the frame of the sighting."""
    west, north = [[4.95, 3.08], [4.95, 3.18]], [[4.95, 3.18], [5.05, 3.18]]
    store = fed((0, (2.0, 1.5, 90.0), [det("z", "bag", "red", 2.0, 2.0, hull=box(2.0, 2.0))]),
                (1, (2.0, 1.5, 90.0), [det("a", "bag", "red", 5.0, 3.13, hull=west)]),
                (2, (2.0, 1.5, 90.0), [det("a", "bag", "red", 5.0, 3.13, hull=north)]),
                (3, (3.0, 1.0, 90.0), []), (4, (3.0, 1.0, 0.0), []))
    assert ask(store, "ego_mem", SLOTS, 3) == offset(1.95, -1.95)
    assert ask(store, "ego_mem", SLOTS, 4) == offset(1.82, 2.08)
    assert ask(store, "ego_mem", {"kind": "box", "attributes": {"color": "red"}}, 3) == {"type": "unknown"}


def ego_case(tmp_path, s):
    """`s` as a case with the ego questions of the look points 13 and 27 written by hand from the gold functions: every
    answerable candidate, and the first real and the first absent unknown ones."""
    w = world(s)
    cands = list(ego_now(w, K).answerable)
    for k in (K, 27):
        moment = ego_mem(w, k)
        real = next(c for c in moment.unknown if c.subjects[0].startswith("e:"))
        absent = next(c for c in moment.unknown if c.subjects[0].startswith("d:"))
        cands += [*moment.answerable, real, absent]
    case = s.write_case(tmp_path)
    qs, gs = questions.records(cands)
    ep.write_jsonl(case.questions, qs, sort_keys=True)
    ep.write_jsonl(case.gold, gs)
    return case, gs


def world_with_things():
    s = scene()
    put(s, "A", "bag", "red", (5.0, 3.13), [2, 3, 4, 5, 6, 7, 8, 9], px=1500)
    put(s, "B", "bag", "blue", (1.0, 1.13), [2])
    put(s, "U", "bag", "yellow", (6.0, 2.0))
    put(s, "T", "bag", "green", (3.0, 4.13), [K])
    return s


def test_the_oracle_and_memory_table_reproduce_the_gold_and_current_view_forgets(tmp_path):
    """Contract (plan 4 A3, 'oracle = table = gold'): over a compiled case the oracle and `memory_table` score 1 on
    every ego question, the answerable ones (the offsets at the look points 13 and 27, whose cameras face north and
    north-east) and the unknown ones (a real thing no frame shows and an absent description); `current_view`, which
    keeps the latest look window alone, answers the thing in view and abstains on the things it saw before, so scores 0
    on the `ego_mem` answerable ones. Hand case: A seen at 2 to 9, B at 2, T at 13, the questions asked at 13 and 27.
    Regression caught: the base pose used as the camera (the mount is 0.13 m), the sighting's frame used for the ask's,
    or a memory-free system reading old frames."""
    case, golds = ego_case(tmp_path, world_with_things())
    kinds = {(g["family"], g["gold"]["type"]) for g in golds}
    assert {("ego_now", "offset"), ("ego_mem", "offset"), ("ego_mem", "unknown")} <= kinds
    for cls in (Oracle, MemoryTable):
        run, _ = play(case, cls, events=EVENTS)
        assert run.status == "complete" and run.failures == {}
        assert set(scores(case, run).values()) == {1.0}, cls.__name__
    run, _ = play(case, CurrentView)
    got = scores(case, run)
    for g in golds:
        want = 0.0 if g["family"] == "ego_mem" and g["gold"]["type"] == "offset" else 1.0
        assert got[g["qid"]] == want, (g["family"], g["gold"])


def test_the_ask_step_camera_turns_the_gold_at_a_diagonal_heading(tmp_path):
    """Contract (plan 4 A3): at the look point 27 the base heads 45 degrees (camera (2 + 0.13 cos 45, 1.5 + 0.13 sin
    45)); the nearest corner of the red bag's 0.1 m footprint about (5.0, 3.13), (4.95, 3.08), is then ((dx + dy) /
    sqrt 2, (dy - dx) / sqrt 2) = (3.0732, -0.9688) from it, and the oracle answers exactly that. Regression caught: a
    rotation by the wrong sign or the wrong angle, which the axis-aligned look point 13 cannot show."""
    case, golds = ego_case(tmp_path, world_with_things())
    gold = next(g for g in golds if g["family"] == "ego_mem" and g["gold"]["type"] == "offset"
                and g["entities"][0] == "A" and g["evidence_k"][-1] == 9)
    cam = (2.0 + 0.13 * math.cos(math.pi / 4), 1.5 + 0.13 * math.sin(math.pi / 4))
    dx, dy = 4.95 - cam[0], 3.08 - cam[1]
    run, _ = play(case, Oracle, events=EVENTS)
    got = run.answers[gold["qid"]]
    assert (got["ahead_m"], got["left_m"]) == pytest.approx(((dx + dy) / math.sqrt(2), (dy - dx) / math.sqrt(2)),
                                                            abs=1e-3)
    assert gold["gold"]["ahead_m"] == pytest.approx(got["ahead_m"], abs=1e-3)


# -- path_log ---------------------------------------------------------------------------------------------------
PLACE = (3.0, 4.13)   # where the prior's scripts saw a red bag, as 5 offsets worked out by hand from their cameras:
# (base x, base y, heading, ahead, left). Heading 90: the camera is at (bx, by + 0.13), so ahead = 4.0 - by, left =
# bx - 3.0. Heading 0: the camera is at (bx + 0.13, by), so ahead = 2.87 - bx, left = 4.13 - by.
SEEN_FROM = ((1.0, 1.0, 90.0, 3.0, -2.0), (0.5, 1.0, 0.0, 2.37, 3.13), (3.0, 1.0, 90.0, 3.0, 0.0),
             (1.0, 2.5, 0.0, 1.87, 1.63), (5.0, 1.5, 90.0, 2.5, 2.0))
ASKED = question("ego_mem", SLOTS, answer_type="offset")


def prior(spread: bool = False, asked: dict = ASKED):
    """Five other scripts that each saw a red bag at PLACE from the bases and headings of `SEEN_FROM`; `spread` shifts
    each offset by 2 m ahead and 2.5 m left a script, so no two scripts saw it in one place."""
    return [record(asked, offset(ahead + (2.0 * i if spread else 0.0), left + (2.5 * i if spread else 0.0)),
                   script=f"s{i}", when="during", phase="cleaning", base=(bx, by), yaw=yaw)
            for i, (bx, by, yaw, ahead, left) in enumerate(SEEN_FROM)]


def ask_path_log(tmp_path, steps, prior_records, asked: dict = ASKED):
    system = started(PathLog, tmp_path, kinds=("kitchen",), events=EVENTS, prior=prior_records)
    system.task(0, "2026-03-02T08:00:00", 0.0, "Clean the kitchen.")
    system.observe(steps)
    return system.ask("during", {"k": steps[-1]["k"], "clock": ""}, [asked])["q1"]


def test_path_log_answers_the_kinds_place_as_an_offset_from_the_ask_pose(tmp_path):
    """Contract (plan 4 A3): `path_log` has a reach on `ego_mem`: five other scripts that saw a red bag at one map place
    from different bases and headings (north and east) agree on it once put on the map, and asked from a base at (3.5,
    0.5) heading north (camera (3.5, 0.63)) while its camera faced the place, it answers that place as 3.5 ahead and
    0.5 left; it is credited in full, so its gate is not vacuous. The constant offsets of the pool (five different
    ones) could not reach it. With the places spread (nothing dense), or the camera turned away from the place, it
    abstains. Regression caught: the offsets of other cameras used as if they were this one's (a vacuous shortcut), the
    camera mount left out (the two headings do not cancel it), or a place guessed that it never faced.
    """
    facing = [step(0, 3.5, 0.5, 90.0), step(1, 3.5, 0.5, 90.0)]
    away = [step(0, 3.5, 0.5, -90.0), step(1, 3.5, 0.5, -90.0)]
    got = ask_path_log(tmp_path / "facing", facing, prior())
    assert got == offset(3.5, 0.5)
    assert ask_path_log(tmp_path / "away", away, prior()) == {"type": "unknown"}
    assert ask_path_log(tmp_path / "spread", facing, prior(spread=True)) == {"type": "unknown"}
    from stbench.score import penalised
    gold_row = {"qid": "g", "family": "ego_mem", "gold": offset(3.5, 0.5), "slice": "answerable", "answer_class": None,
                "range_m": 3.5, "credit": "ladder", "region": None, "metric": None, "scope": "memory",
                "tolerances": {"ladder": [0.35, 0.875, 1.75]}, "reference": None, "critical": True}
    assert penalised(got, gold_row) == 1.0


def test_path_log_ego_now_is_the_pool_alone_and_unknown_without_one(tmp_path):
    """Contract (plan 4 A3): `ego_now` is the pool's learned answer: with no pool of other scripts it is `unknown`, and
    constant offsets that beat abstaining on a script left out are answered. Hand case: no prior is `unknown`; five
    scripts whose red bag was always 2.0 ahead and 0.5 left of the camera at one phase give that offset. Regression
    caught: a `path_log` that cannot answer `ego_now` at all (a family complete in the registry but unreachable)."""
    now = question("ego_now", SLOTS, answer_type="offset")
    steps = [step(0, 3.5, 0.5, 90.0)]

    def run(path, prior_records):
        system = started(PathLog, path, kinds=("kitchen",), events=EVENTS, prior=prior_records)
        system.task(0, "2026-03-02T08:00:00", 0.0, "Clean the kitchen.")
        system.observe(steps)
        return system.ask("during", {"k": 0, "clock": ""}, [now])["q1"]

    same = [record(now, offset(2.0, 0.5), script=f"s{i}", when="during", phase="cleaning", base=(float(i), 0.0))
            for i in range(5)]
    assert run(tmp_path / "none", []) == {"type": "unknown"}
    assert run(tmp_path / "same", same) == offset(2.0, 0.5)


def test_path_log_ego_now_reads_the_map_for_a_layout_fixed_thing_in_view(tmp_path):
    """Contract (review of A3): a thing that sits at one spot of the layout (a roll cage at the end of an aisle, a stool
    in a kitchen) is a no-picture shortcut for `ego_now` too, so `path_log` puts the prior's `ego_now` golds of the kind
    on the map and answers the dense place that is in view at the ask step alone. Hand case: five scripts saw a chair at
    PLACE from the bases of `SEEN_FROM`; asked from (3.5, 0.5) heading north with the camera facing it, it answers 3.5
    ahead and 0.5 left. With the camera turned away at the ask step, though it faced the place at the step before,
    `ego_now` falls to the pool (five different offsets: unknown) while `ego_mem` still answers the place seen from the
    camera at (3.5, 0.37) heading south: -3.76 ahead and 0.5 to the right. Regression caught: `ego_now` answered from
    the pool alone (a near-vacuous gate), or a place out of view now offered as one in view."""
    chair = {"kind": "chair", "attributes": {}}
    now, mem = question("ego_now", chair, answer_type="offset"), question("ego_mem", chair, answer_type="offset")
    facing = [step(0, 3.5, 0.5, 90.0), step(1, 3.5, 0.5, 90.0)]
    turned = [step(0, 3.5, 0.5, 90.0), step(1, 3.5, 0.5, -90.0)]
    assert ask_path_log(tmp_path / "now", facing, prior(asked=now), now) == offset(3.5, 0.5)
    assert ask_path_log(tmp_path / "turned", turned, prior(asked=now), now) == {"type": "unknown"}
    assert ask_path_log(tmp_path / "mem", turned, prior(asked=mem), mem) == offset(-3.76, -0.5)
