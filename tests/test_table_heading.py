"""The table systems' answer to `open_heading` (`table_heading`): the gold's own rule over the cells the system's mapper
made of everything it was sent up to the ask. The vacuum (0.34 m disc, half width 0.17, cap 3 m) at the origin; a wall
of sub-cells with centres at x 1.01 across every row, free floor west of it (fine cells u < 10) and nothing east.

By hand: the disc's centre touches the wall's centre line when it is 0.17 m short of it, after 0.84 / cos(theta) at a
heading theta from +x, so a heading runs to the cap (3 m, the wall's 1 cm of half a sub-cell aside) when 0.84 / cos
theta >= 3.01: theta >= 73.8 degrees. At 74 degrees the run is the cap, at 73 it is 0.84 / cos 73 - 0.01 = 2.86."""

from __future__ import annotations

from heading_kit import VACUUM, occupied, wall_cells
from stbench import table_heading
from stbench.table import Store

SLOTS = {"cap": 3.0}


def ask(cells_by_step: dict[int, dict], k: int = 5, yaw_deg: float = 0.0, slots=SLOTS) -> dict:
    store = Store()
    store.cells = cells_by_step
    now = {"k": k, "pose": {"pos": [0.0, 0.0, 0.0], "yaw_deg": yaw_deg}}
    return table_heading.open_heading(store, slots, {"now": now, "robot": VACUUM})


def test_the_table_answers_the_best_heading_nearest_ahead():
    """Contract (the ceiling): the answer is the heading of the profile over the store's cells with the longest run, the
    one nearest the robot's own among those within a sub-cell of it. Hand value (module docstring): the cap is reached
    from 74 degrees left or right on, and the wall is symmetric, so the two are as near and the left wins: 74.0.
    With the robot facing 45 degrees (yaw 45) the same wall is turned in its frame: the headings that reach the cap are
    those at absolute 74 to 286 degrees, nearest ahead the one at absolute 74, which is 29 degrees to the left: 29.0;
    facing -45 it is the mirror, absolute 286 = -74, 29 degrees to the right: -29.0. Facing the wall's own direction
    (yaw 90) ahead already runs to the cap: 0.0. Regression caught: a heading in the wrong frame (absolute, or
    mirrored), the farthest-ahead heading, or a tie broken to the right."""
    assert ask({0: wall_cells()}) == {"type": "heading", "deg_left": 74.0}
    assert ask({0: wall_cells()}, yaw_deg=45.0) == {"type": "heading", "deg_left": 29.0}
    assert ask({0: wall_cells()}, yaw_deg=-45.0) == {"type": "heading", "deg_left": -29.0}
    assert ask({0: wall_cells()}, yaw_deg=90.0) == {"type": "heading", "deg_left": 0.0}


def test_the_table_reads_only_the_cells_up_to_the_ask():
    """Contract: a table system holds the stream up to the ask and answers from no step after it (truth rule 2, and the
    ask is made before the later steps exist). Hand case: the same wall seen at step 0, and at step 9 (after the ask at
    step 5) the floor ahead of the robot seen blocked by a closer wall at x 0.41 (sub-cells a = 20 across the rows
    u = 4): asked at step 5 the answer is still 74, asked at step 9 it is the wall that has since been seen:
    0.24 / cos(theta) >= 3.01 from 85.4 degrees, so 86. Regression caught: a store read to its end, which a stream
    could never have given."""
    near = occupied((20, b) for b in range(-300, 300))
    cells = {0: wall_cells(), 9: near}
    assert ask(cells, k=5) == {"type": "heading", "deg_left": 74.0}
    assert ask(cells, k=9) == {"type": "heading", "deg_left": 86.0}


def test_nothing_seen_is_unknown():
    """Contract: with no corridor seen empty there is no heading to name: `unknown` (the gold is never `unknown`, so a
    table that answers it loses nothing it was owed). Regression caught: a default heading made up from an empty
    store."""
    assert ask({}) == {"type": "unknown"}
