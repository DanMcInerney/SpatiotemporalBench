"""`map_reader`'s answer to `open_heading` (`baselines_map_heading`), the reference of a structure family: the best
heading of the public map's raycast profile from the delivered pose, answered as it is; on a held-out layout, the
pose-keyed pool, `unknown` with no pool. Hand map and poses of `test_path_log.hand_map` without its furniture (two
4 x 3 m rooms side by side, the wall between them with faces at x 3.98 and 4.08 and an interior door 0.9 m wide
centred on y 1.5, read open); the vacuum is a 0.34 m disc (half width 0.17), cap 3 m."""

from __future__ import annotations

from baseline_kit import EVENTS, UNKNOWN, question, record, started, step
from own_kit import heading, heading_gold
from stbench.baselines import MapReader
from test_baselines_map import ask
from test_path_log import hand_map


def heading_q(qid: str = "q1") -> dict:
    return question("open_heading", {"cap": 3.0}, qid, "heading")


def at(tmp_path, yaw_deg: float = 90.0, **opts):
    """What `map_reader` answers to `open_heading` with the vacuum at (2.0, 1.5) facing `yaw_deg`, in the west room."""
    reader = started(MapReader, tmp_path, map=hand_map(furniture=False), events=EVENTS, **opts)
    return ask(reader, [step(0, 2.0, 1.5, yaw_deg, phase="transit")], heading_q())


def test_the_map_reader_names_the_nearest_heading_that_runs_to_the_cap(tmp_path):
    """Contract: the answer is the best heading of the map's raycast profile, in 1 degree steps from the robot's own
    (the gold's profile), the one nearest ahead of those within a sub-cell of the longest. Hand case: the vacuum at
    (2.0, 1.5) facing north (yaw 90), in a room 3.9 m wide whose only way on is the door in the partition (faces x 3.98
    and 4.08, opening y 1.05 to 1.95; the sub-cell centred at (3.99, 1.95) is in the wall). A heading runs to the 3 m
    cap when its ray from the front edge, 0.17 m out, passes 0.17 m clear of that corner: at world 7 degrees the edge is
    (2.1687, 1.5207), the corner (1.8213, 0.4293) from it, and its distance to the line is 1.8213 sin 7 - 0.4293 cos 7 =
    -0.204, so 0.204 clear; at world 8 degrees it is 0.169, which meets it. The nearest ahead of the headings that run
    is world 7 degrees, 83 to the right of north: -83.0 (a 5 degree sweep read -85 and, at the door of a cluttered home,
    missed openings narrower than its step). Regression caught: a raycast from the base or a heading in the world's
    frame, a nearest-ahead tie broken to the farthest, the door read shut, or a coarse sweep."""
    assert at(tmp_path) == heading(-83.0)


def test_a_map_holder_answers_the_ray_whatever_the_pool_holds(tmp_path):
    """Contract (the final review, S1; `clear_path`'s reference does the same): a reader that holds the map answers its
    ray as it is, so the reference is as strong as a map reader can be: other scripts' golds do not stand
    beside it and a pool never makes it abstain or answer for it. Hand case: the case above (-83.0) with a prior of
    five scripts, two records each, at the same phase and pose, all with the best heading straight ahead (0): the pool
    is found at the pose level and its learned answer is 0 (every record gains +1 on it), where the ray is -83. A
    thin prior of two scripts and no prior change nothing. Regression caught: the ray offered to
    `blind.learned_answer` as a candidate (it answered the pool's heading, or `unknown` where the pool was coarse, so
    `map_reader` equalled `all_unknown` on both jobs), or a pool consulted before the map."""
    q = heading_q()
    here = {"phase": "transit", "base": (2.0, 1.5), "yaw": 90.0, "when": "during"}

    def pool(scripts: int) -> list[dict]:
        return [{**record(q, heading(0.0), script=f"s{i}", **here), "gold": heading_gold(0.0)}
                for i in range(scripts) for _ in range(2)]

    for name, prior in (("rich", pool(5)), ("thin", pool(2)), ("none", [])):
        assert at(tmp_path / name, prior=prior) == heading(-83.0), name


def test_a_reader_with_no_map_answers_from_the_pool_alone(tmp_path):
    """Contract: on a held-out layout the reader is given the region kinds and no map, so it has no ray: its answer is
    the pose pool's learned answer, `unknown` with no pool. Hand case: the pool of the test above (five scripts at the
    pose, all with the best heading 0) is a heading to answer, 0; two scripts (`blind.MIN_POOL_SCRIPTS` is 3), or none,
    are no pool. Regression caught: a reader that reads a map it was not given, a pool consulted on a handful of
    records, or one never consulted."""
    q = heading_q()
    here = {"phase": "transit", "base": (2.0, 1.5), "yaw": 90.0, "when": "during"}

    def read(name: str, scripts: int) -> dict:
        pool = [{**record(q, heading(0.0), script=f"s{i}", **here), "gold": heading_gold(0.0)}
                for i in range(scripts) for _ in range(2)]
        unseen = started(MapReader, tmp_path / name, kinds=("kitchen",), events=EVENTS, prior=pool)
        return ask(unseen, [step(0, 2.0, 1.5, 90.0, phase="transit")], q)

    assert read("five", 5) == heading(0.0)
    assert read("two", 2) == UNKNOWN
    assert read("none", 0) == UNKNOWN
