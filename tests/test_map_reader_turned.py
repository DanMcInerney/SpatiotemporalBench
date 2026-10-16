"""`map_reader`'s answer to a turned `clear_path` (0.9.0, `baselines_map_clear`): the public map's raycast from the
moved start along the moved heading. Hand map of `test_path_log` (two 4 x 3 m rooms, the west room x 0.05 to 3.98 and y
0.05 to 2.95, no furniture), the vacuum (a 0.34 m disc, front edge 0.17 m from its base) at (1.5, 2.5); a wall's inner
face lies on a sub-cell centre (0.05 and 2.95), which counts as in, so a corridor stops at the distance to it less
0.01."""

from __future__ import annotations

import pytest

from baseline_kit import EVENTS, number, question, started, step
from stbench.baselines import MapReader
from test_path_log import hand_map


def clear(direction: str, cap: float = 3.0) -> dict:
    return question("clear_path", {"mode": "drive", "direction": direction, "cap": cap}, "q1", "number")


def read(tmp_path, direction: str, yaw_deg: float) -> dict:
    reader = started(MapReader, tmp_path, map=hand_map(furniture=False), events=EVENTS)
    steps = [step(0, 1.5, 2.5, yaw_deg=yaw_deg, phase="transit")]
    reader.observe(steps)
    return reader.ask("during", {"k": 0, "clock": ""}, [clear(direction)])["q1"]


@pytest.mark.parametrize("yaw, want", [
    (0.0, {"ahead": 2.31, "back": 1.27, "left": 0.27, "right": 2.27}),
    (90.0, {"ahead": 0.27, "back": 2.27, "left": 1.27, "right": 2.31})])
def test_the_ray_runs_from_the_moved_edge_along_the_moved_heading(tmp_path, yaw, want):
    """Contract (brief 4.1): the reference casts the ray from the edge of the asked move along the heading it runs, at
    the delivered pose. Hand values, base (1.5, 2.5), above the door's opening (y 1.05 to 1.95): heading +x, ahead from
    (1.67, 2.5) to the partition's face (centre 3.99) 2.32, so 2.31; back from (1.33, 2.5) west to the wall (centre
    0.05) 1.28, so 1.27; left from (1.5, 2.67) north to the wall at 2.95, 0.28, so 0.27; right from (1.5, 2.33) south to
    the wall at 0.05, 2.28, so 2.27. Heading +y (90 degrees) the same moves turn with the robot: ahead is the north wall
    (0.27), back the south (2.27), left west (1.27) and right east (2.31). Regression caught: a turned ray cast on the
    delivered heading (every direction 2.31), left and right swapped, or the start left at the front edge."""
    for i, (direction, value) in enumerate(want.items()):
        assert read(tmp_path / f"{yaw}-{i}", direction, yaw) == number(value), (yaw, direction)
