"""What the tests of the reworded change families (lane A5: `object_was`, `same_place`, `object_where`'s range) share,
on `change_kit` and `space_kit`.

`vacuum_run(n)`'s camera stands at (2.13, 1.5, 0.08) at offsets 0, 3, 23 and 26 of every task (keyframes 3 and 31 are
offset 3 of tasks 0 and 1) and at (6.13, 1.5, 0.08) at offsets 7 and 15 (keyframes 7 and 35). A bag (the catalog's size,
0.314 m high, so its middle is 0.157 m up) at (x, 1.5) is then `hypot(x - 2.13, 0.077)` metres away from the first
camera: 1.0030 m at x = 3.13, 3.0010 m at 5.13, 5.0006 m at 7.13. `SLANT` and `RUNGS` are those ranges and the relative
rungs (`rules.relative_ladder`: 10%, 25% and 50% of the range, at least 5, 10 and 20 cm) worked out here by hand.

`moved_bags()` is two tasks and four bags that move in the gap, each seen once in each task from the first camera:
- E0001 red, 5.13 -> 7.13 (2.00 m): beyond task 0's last rung (1.5005 m) but not task 1's (2.5003 m);
- E0002 blue, 3.13 -> 6.13 (3.00 m): beyond the last rung of both tasks (0.5015 and 2.0004 m);
- E0003 green, 5.13 -> 3.62 (1.51 m): beyond task 0's last rung (1.5005 m) by 9.5 mm, and task 1's (0.7460 m);
- E0004 purple, 5.13 -> 3.63 (1.50 m): not beyond task 0's last rung (1.5005 m), beyond task 1's (0.7510 m).
So the earlier state (asked in task 0) is E0001, E0002 and E0003, the later (asked in task 1) E0002, E0003 and E0004.
"""

from __future__ import annotations

import change_kit as ck
from change_kit import thing
from space_kit import vacuum_run

SEEN = [3, 31]   # offset 3 of tasks 0 and 1: the camera at (2.13, 1.5, 0.08)
SLANT = {1.0: 1.003, 1.49: 1.492, 1.5: 1.502, 3.0: 3.001, 4.0: 4.0007, 5.0: 5.0006}  # by dx, metres, to 4 places
RUNGS = {1.0: [0.1003, 0.25075, 0.5015], 1.49: [0.1492, 0.373, 0.746], 1.5: [0.1502, 0.3755, 0.751],
         3.0: [0.3001, 0.75025, 1.5005], 4.0: [0.40007, 1.000175, 2.00035],
         5.0: [0.50006, 1.25015, 2.5003]}   # by dx, the three rungs of that range


def moved_bags(seen=SEEN):
    """Two tasks and the four bags of the module docstring, each seen at the keyframes `seen`."""
    s = vacuum_run(2)
    for eid, colour, (a, b) in (("E0001", "red", (5.13, 7.13)), ("E0002", "blue", (3.13, 6.13)),
                                ("E0003", "green", (5.13, 3.62)), ("E0004", "purple", (5.13, 3.63))):
        thing(s, eid, "bag", colour, [(a, 1.5), (b, 1.5)], seen)
    return s


def only(*families: str):
    """`change_kit.CHANGE_JOB` asking only these end families."""
    from dataclasses import replace
    return replace(ck.CHANGE_JOB, ends=families)


def hand_coin(cid: str, family: str, key: str) -> bool:
    """The questionnaire's coin written out by hand: a moment of a coin family takes its second pool for half of the
    (case id, family, moment key) triples."""
    from stbench.seeding import stable_hash
    return stable_hash(cid, family, key, "slice") % 100 < 50


def many_moved(n: int = 12):
    """Two tasks and `n` (at most 12) bags of distinct colours (`change_kit.MANY_COLOURS`), each at (1.0, 0.3 + 0.2 i)
    in task 0 and 6.5 m on at (7.5, 0.3 + 0.2 i) from the gap, seen at keyframes 3 and 31 from the camera at (2.13, 1.5,
    0.08): at most 5.8 m away, so the last rung (half the range, 2.9 m) is passed by the 6.5 m move at either state, and
    a notebook at the wrong task's place scores -1."""
    s = vacuum_run(2)
    for i in range(n):
        thing(s, f"E{i + 1:04d}", "bag", ck.MANY_COLOURS[i], [(1.0, 0.3 + 0.2 * i), (7.5, 0.3 + 0.2 * i)], SEEN)
    return s
