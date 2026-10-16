"""The direction a `clear_path` question asks (0.9.0, `clear_move.draw`, `clear_ground.moment`,
`clear_fly.direction_of`): drawn from the case id and the look point over the job's declared directions, never from the
world. The expected draws are worked here from the hash's definition (SHA-256 of the parts joined by the unit separator,
its first 8 bytes as a big-endian integer) and not by the code under test."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace

from clear_kit import JOB, LAST, room_run, world
from stbench import clear_fly, clear_ground, clear_move

ALL = ("ahead", "back", "left", "right")


def index(case: str, k: int, n: int) -> int:
    """`stable_hash(case, k, "direction") % n`, by hand."""
    data = "\x1f".join([case, str(k), "direction"]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big") % n


def test_the_direction_is_the_hash_of_the_case_and_the_point_over_the_declared_directions():
    """Contract (brief 4.1): the direction asked at point `k` of a case is `directions[hash(case id, k, "direction") %
    len(directions)]`, each declared direction equally and no other. Hand-worked on forty (case, point) pairs: the draw
    equals the hash's index into the four directions, all four occur, a job that declares ahead alone always asks
    ahead, and a job that declares two draws among those two. Regression caught: a draw that ignores the declaration
    (a left asked of a job that cannot turn), a draw by the point alone or by the case alone, or one direction never
    drawn."""
    pairs = [(f"c{i}", k) for i in range(10) for k in (7, 15, 19, 27)]
    drawn = [clear_move.draw(case, k, ALL) for case, k in pairs]
    assert drawn == [ALL[index(case, k, 4)] for case, k in pairs]
    assert set(drawn) == set(ALL)
    assert {clear_move.draw(case, k, ("ahead",)) for case, k in pairs} == {"ahead"}
    assert {clear_move.draw(case, k, ("ahead", "back")) for case, k in pairs} == {"ahead", "back"}
    assert len({clear_move.draw("c0", k, ALL) for k in range(7, 400, 8)}) > 1   # the point matters, not the case alone


def test_the_drones_draw_is_the_0_8_0_one():
    """Contract (plan G1): a flier that declares ahead and back draws exactly what 0.8.0 drew, `ahead` for an even hash
    and `back` for an odd one of `(case id, k, "direction")`, so no flier gold moves. Hand-worked from the hash on forty
    pairs. Regression caught: a drone gold moved by the new draw (a modulus over another tuple, or a reordered pair)."""
    for i in range(10):
        for k in (9, 19, 29, 39):
            job = SimpleNamespace(clear_directions=("ahead", "back"))
            w = SimpleNamespace(rec=SimpleNamespace(case=f"c{i}"), job=job)
            assert clear_fly.direction_of(w, k) == ("ahead", "back")[index(f"c{i}", k, 2)]


def case_drawing(direction: str) -> str:
    """The first case id of `c0`, `c1`, ... whose draw at the look point is `direction` (found from the hash by
    hand)."""
    return next(f"c{i}" for i in range(200) if ALL[index(f"c{i}", LAST, 4)] == direction)


def test_two_worlds_that_differ_only_behind_the_robot_draw_the_same_direction():
    """Contract (brief 4.1, validity): the direction is drawn from the case and the point alone, so what stands behind
    the robot cannot change which way is asked. Hand scene: the same closed room with and without the wall behind the
    robot (the back corridor 2.40 m to a wall, or open and `unknown`), for a case id whose hash draws each of the four
    directions in turn: the asked direction is that one in both worlds. Regression caught: a direction chosen from the
    corridors (the question asked back only where a wall is known), which would leak the world."""
    for direction in ALL:
        job, cid = replace(JOB, clear_directions=ALL), case_drawing(direction)
        asked = []
        for back_wall in (True, False):
            walls = ("ahead", "left", "right") + (("back",) if back_wall else ())
            m = clear_ground.moment(world(room_run(walls), [], job, cid), LAST)
            (cand,) = [*m.answerable, *m.unknown]
            asked.append(cand.slots["direction"])
        assert asked == [direction, direction], direction
