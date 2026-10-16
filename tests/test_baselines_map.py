"""`map_reader` (`baselines_map`), the reference of a structure family: the public map's raycast from the delivered
pose (the pose-keyed pool where there is no map), `unknown` on every other template. Hand-drawn map and poses
(test_path_log's), every expected value worked by hand: the vacuum's 0.34 m disc has its front edge 0.17 m ahead of
its base, a 0.1 m fine cell stops the body at its near edge, and the drone is a 1.3 m disc whose edge is 0.65 m from
its base."""

from __future__ import annotations

import json

import pytest

from baseline_kit import EVENTS, UNKNOWN, number, question, record, started, step
from stbench import robots
from stbench.baselines import BASELINES, MapReader
from stbench.gates import MAP_READER
from test_path_log import hand_map

DRONE = robots.ROBOTS["drone"]


def clear(direction: str = "ahead", cap: float = 3.0, qid: str = "q1") -> dict:
    """A `clear_path` question as the structured view gives it."""
    return question("clear_path", {"mode": "ground", "direction": direction, "cap": cap}, qid, "number")


def ask(system, steps, q: dict) -> dict:
    system.observe(steps)
    return system.ask("during", {"k": steps[-1]["k"], "clock": ""}, [q])[q["qid"]]


def test_the_reference_is_registered_as_path_logs_system_with_a_map_and_no_in_pool_twin():
    """Contract (review §2.3): `map_reader` holds what a layout-memorising agent holds, so it is `path_log`'s system:
    the other scripts' records, the public map and the job's events, modelling an agent (no map on a held-out layout).
    Regression caught: a reference registered without its map, which would answer `unknown` and make the group's
    reported score a floor."""
    b = BASELINES[MAP_READER]
    assert b.cls is MapReader and (b.needs_prior, b.needs_map, b.needs_events, b.models_agent) == (True,) * 4


def test_the_ground_raycast_runs_from_the_front_edge_and_the_back_from_the_rear_edge(tmp_path):
    """Contract: on the hand map with its furniture, the vacuum at (0.5, 1.7) heading +x has the table's leg 1.33 m
    ahead of its front edge (test_path_log). Behind it the corridor starts at the rear edge, x = 0.5 - 0.17 = 0.33, and
    runs west to the wall's inner face at x = 0.05: the sub-cell with centre 0.05 is in the wall (on its edge, which is
    in), 0.28 m from the start, so it stops the body 0.28 - 0.01 = 0.27 m along. Regression
    caught: a back corridor started from the front edge (it would read 0.27 + 0.34), or from the base."""
    steps = [step(0, 0.5, 1.7, phase="transit")]
    reader = started(MapReader, tmp_path / "a", map=hand_map(), events=EVENTS)
    assert ask(reader, steps, clear("ahead")) == number(1.33)
    assert ask(started(MapReader, tmp_path / "b", map=hand_map(), events=EVENTS), steps, clear("back")) == number(0.27)


def test_a_map_holder_answers_the_ray_whatever_the_pool_holds(tmp_path):
    """Contract: a reader that holds the map answers the raycast (the reference is as strong as a map reader
    can be): other scripts' golds do not stand beside it, and a pool never makes it abstain. Hand case: the vacuum at
    (1.5, 2.7) heading +x on the map without furniture, front edge at x = 1.67; the partition's face is x = 3.98 and
    the first sub-cell centre in it 3.99, 2.32 m from the edge: the body stops 2.32 - 0.01 = 2.31 m along. The pool
    holds six scripts' golds of varied lengths at six other poses of the same phase: the level it is found at is the
    phase alone, where no constant beats abstaining held out (a number each is 1.0, 2.0, ... 6.0 and every other
    script's best response is wrong on the one left out). Regression caught: the ray offered to
    `blind.learned_answer` as a candidate, which scored it on other cases' golds and answered `unknown` (map_reader
    equalled all_unknown on the helper and vacuum), or a pool of unknown golds overriding the ray."""
    steps = [step(0, 1.5, 2.7, phase="transit")]
    q = clear()
    others = [record(q, number(float(i + 1)), script=f"s{i}", phase="transit", base=(0.5 + i, 0.5), when="during")
              for i in range(6)]
    unknowns = [record(q, UNKNOWN, script=f"u{i}", phase="transit", base=(1.5, 2.7), when="during") for i in range(5)]

    def read(name, prior):
        return ask(started(MapReader, tmp_path / name, map=hand_map(furniture=False), events=EVENTS, prior=prior),
                   steps, q)

    assert read("none", []) == number(2.31)
    assert read("varied", others) == number(2.31)
    assert read("unknown", unknowns) == number(2.31)


def test_a_reader_with_no_map_answers_from_the_pool_alone(tmp_path):
    """Contract: on a held-out layout the reader holds the region kinds and no map, so its answer is the pose-keyed
    pool's learned answer: five scripts' golds at the same phase and pose, all 1.5, are a number to answer, three are
    no pool (`unknown`). Regression caught: a pool consulted on a handful of records, or one never consulted."""
    steps = [step(0, 1.5, 2.7, phase="transit")]
    q = clear()

    def read(name, n):
        pool = [record(q, number(1.5), script=f"s{i}", phase="transit", base=(1.5, 2.7), when="during")
                for i in range(n)]
        return ask(started(MapReader, tmp_path / name, kinds=("kitchen",), events=EVENTS, prior=pool), steps, q)

    assert read("five", 5) == number(1.5)
    assert read("three", 3) == UNKNOWN


def test_every_other_template_is_unknown_and_a_held_out_layout_has_no_ray(tmp_path):
    """Contract: `map_reader` answers `clear_path` alone (`unknown` on `path_ahead`, which the same map would answer),
    and on a held-out layout, where it is given the region kinds and no map, it has no ray: `unknown` with no pool.
    Regression caught: a reference that answers other families (a second shortcut under another name), or one that
    reads a map it was not given."""
    steps = [step(0, 0.5, 1.7, phase="transit")]
    reader = started(MapReader, tmp_path / "a", map=hand_map(), events=EVENTS)
    assert ask(reader, steps, question("path_ahead", {"cap": 3.0}, "q1", "number")) == UNKNOWN
    unseen = started(MapReader, tmp_path / "b", kinds=("kitchen",), events=EVENTS)
    assert ask(unseen, steps, clear()) == UNKNOWN


def started_drone(tmp_path, **opts):
    """`MapReader` built over the drone's camera and sent its `start`."""
    (tmp_path / "public").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "public" / "camera.json").write_text(json.dumps(robots.camera_doc(DRONE)), encoding="utf-8")
    system = MapReader(tmp_path / "state", tmp_path, **opts)
    system.start("0123456789abcdef", robots.robot_doc(DRONE), "rgb_pose")
    return system


def flying(x: float, y: float, z: float) -> dict:
    s = step(0, x, y, phase="inspect")
    s["pose"]["pos"] = [x, y, z]
    return s


@pytest.mark.parametrize("z, expected", [(1.0, 2.511), (3.0, 60.0)])
def test_a_fliers_ray_meets_the_walls_that_reach_its_height_and_nothing_else(tmp_path, z, expected):
    """Contract: the flier's reference is the wall raycast at the base's height: the hand map's walls are 2.5 m tall,
    so a drone at 1.0 m meets the partition at x = 4.03 and one at 3.0 m flies over it to the cap; the table, the
    chair and the door leaf are not in its corridor at any height. By hand: from (1.0, 1.5) the front edge is 0.65 on,
    at 1.65; the body is a disc 1.3 m across (radius 0.65) and the 0.9 m door opening (y 1.05 to 1.95) is narrower, so
    the wall pieces either side stop it: the partition's face is x = 3.98, the first sub-cell centre in it 3.99, 2.34
    from the edge, and the piece's end at y = 1.95 puts a centre at 0.45 off the axis, which the disc touches after
    2.34 + 0.65 - sqrt(0.65^2 - 0.45^2) = 2.521 (less half a sub-cell: 2.511). Regression caught: a flier stopped by the
    ground robot's obstacles (furniture, a door leaf), by a wall below it, or a disc taken for its strip (2.33)."""
    reader = started_drone(tmp_path, map=hand_map(), events=EVENTS)
    q = question("clear_path", {"mode": "fly", "direction": "ahead", "cap": 60.0}, "q1", "number")
    assert ask(reader, [flying(1.0, 1.5, z)], q) == number(expected)
