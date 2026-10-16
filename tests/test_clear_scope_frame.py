"""The scope of a ground `clear_path` row (`clear_ground.moment`, `clear_ground.frame_agrees`): `now` exactly when the
look point's own frame, the latest one a system holds when it is asked (it gets one frame at a time), gives the same
number within the row's resolution, else `memory`, so that an answerable `memory` row is one the current view cannot
answer. `clear_kit`'s tote AMR at `BASE` (0.6, 0) heading +x; the point is keyframe 19, which ends the look window of
keyframes 16 to 19 (the window decides where a point may fall, no longer the scope)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from clear_kit import AMR, JOB, LAST, drive_run, room_run, wall, world
from space_kit import box
from stbench import clear_ground, grid

START = (1.0, 0.0)   # the front edge at the point


def only(w):
    m = clear_ground.moment(w, LAST)
    (cand,) = [*m.answerable, *m.unknown]
    return cand


def view(stop_sub: tuple[int, int] | None, free: bool = True) -> dict:
    """The cells a frame showed of the AMR's ahead corridor: free fine cells x 1.0 to 4.0, y -0.3 to 0.3, and, when
    given, one obstacle sub-cell (a, b) = its 0.02 m index (centre (a + 0.5) x 0.02)."""
    cells = {(u, v): ("free", 0) for u in range(10, 40) for v in range(-3, 3)} if free else {}
    if stop_sub is not None:
        cells[(stop_sub[0] // 5, stop_sub[1] // 5)] = (grid.Occupied([stop_sub]), 0)
    return cells


@pytest.mark.parametrize("stop_sub, free, agrees", [
    ((150, 0), True, True),     # centre 3.01: 2.01 from the edge, 2.00: the gold itself
    ((151, 0), True, True),     # centre 3.03: 2.02, 0.02 away, inside the resolution 0.03
    ((154, 0), True, False),    # centre 3.09: 2.08, 0.08 away
    (None, True, False),        # nothing in the way: the cap, 5.0
    (None, False, False),       # nothing seen: `unknown`
])
def test_the_frame_answers_when_it_gives_the_gold_within_its_resolution(stop_sub, free, agrees):
    """Contract (brief 4.1): `now` exactly when the corridor over frame k's cells alone is a number within the row's
    resolution (here 0.03) of the window's gold (here 2.0), and those cells are read for keyframe 19 only. Hand values
    from the sub-cell centres, the front edge at x 1.0: centre 3.01 gives 2.01 - 0.01 = 2.00; 3.03 gives 2.02; 3.09
    gives 2.08; no obstacle the cap 5.0; no cell seen `unknown`. Regression caught: the equality made exact (a 1 cm
    difference a camera cannot tell sending a row to `memory`), made loose (a row the frame misses by 8 cm called
    `now`), `unknown` counted as agreeing, or the cells of the look window (keyframes 16 to 19) read, not the
    frame's."""
    asked = []

    def latest_cells(ks):
        asked.append(list(ks))
        return view(stop_sub, free)

    w = SimpleNamespace(latest_cells=latest_cells, robot=AMR)
    gold = {"type": "number", "value": 2.0}
    assert clear_ground.frame_agrees(w, LAST, (START, 0.0, 5.0, 0.0), gold, 0.03) is agrees
    assert asked == [[LAST]]


@pytest.mark.parametrize(("seen_at", "scope"), [(19, "now"), (18, "memory"), (17, "memory"), (16, "memory"),
                                                (15, "memory")])
def test_a_corridor_only_an_earlier_frame_of_the_look_window_showed_is_memory(seen_at, scope):
    """Contract: the scope reads frame k alone. Hand scene: a wall across the corridor with its near face at x 3.01
    (2.01 m from the front edge at x 1.0, in the sub-cell centred 3.01: the gold is 2.01 - 0.01 = 2.00), the base at
    `BASE` heading +x in every frame, and a render in one frame only. The window's gold is 2.0 whichever frame it was
    rendered in (it reads the latest cells up to keyframe 19); rendered at 19, the point's own frame: `now`; rendered
    at 18, 17 or 16, frames of the same look window (keyframes 16 to 19) that the old rule counted: `memory`; at 15,
    before the window: `memory`. Regression caught: the scope reading the look window again (a frame a system was sent
    one to three messages ago called `now`)."""
    c = only(world(drive_run(boxes=[wall(3.01).at], depth_ks={seen_at}), [], JOB))
    assert (c.gold, c.evidence_k, c.scope) == ({"type": "number", "value": 2.0}, (seen_at,), scope)


def test_a_corridor_the_frame_shows_is_now_and_one_only_the_past_showed_is_memory():
    """Contract: in the closed room the point's frame (keyframe 19, facing +x) shows the ahead corridor whole: `now`;
    the back, left and right corridors were seen only before it: the same numbers, `memory`. A corridor never seen
    empty is `unknown`, and an `unknown` gold is `memory` (the agent must hold what it has and has not seen), where
    0.8.0's flier and ground rows were `now` by evidence. Regression caught: a `memory` row the current view answers
    (a ground row scoped by its evidence frames), or a never-seen corridor counted as perception."""
    for direction, scope in (("ahead", "now"), ("back", "memory"), ("left", "memory"), ("right", "memory")):
        c = only(world(room_run(), [], replace(JOB, clear_directions=(direction,))))
        assert (c.gold["type"], c.scope) == ("number", scope), direction
    blind = only(world(drive_run(depth_ks=()), None))
    assert (blind.gold, blind.slice, blind.scope) == ({"type": "unknown"}, "unknown", "memory")


def test_a_corridor_the_frame_cannot_see_and_the_past_saw_differently_is_memory():
    """Contract: `memory` means the current view cannot answer, even when it answers something. Hand scene: a crate in
    the first 0.3 m past the front edge (x 1.11 to 1.31, the patch the camera cannot see ahead of the body, 0.53 m) seen
    in keyframes 0 to 11 from (-1, 0), then removed while the robot drove to the base; the wall at x 3.01. The window's
    last sighting of the crate's face (sub-cell centre 1.11, 0.11 from the edge) stops the corridor at 0.10; the frame
    sees only the wall, 2.00. They differ by more than the 0.02 m resolution: `memory`, with the gold 0.10. Regression
    caught: a row scoped `now` because the frame gives a number, whatever number."""
    crate = box(1.11, 1.31, -0.2, 0.2, 0.0, 0.5)
    script = drive_run(boxes=[wall(3.01).at], frame_boxes={k: [crate] for k in range(12)}, depth_ks={3, 19},
                       poses={**dict.fromkeys(range(20), (0.6, 0.0)), **dict.fromkeys(range(12), (-1.0, 0.0))},
                       yaws=dict.fromkeys(range(20), 0.0))
    c = only(world(script, [], JOB))
    assert (c.gold, c.scope) == ({"type": "number", "value": 0.1}, "memory")
