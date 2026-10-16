"""The resolution a ground `clear_path` row carries (0.9.0, `corridor_screens.screen`, `clear_ground.moment`):
`tolerances.resolution_m` is the larger of the grid's 2 cm sub-cell, the stopper's floor contact's one-pixel error in
the frame the stop rests on, and the gap from the stop to that contact, so that a reader of the frames can reach full
credit exactly where the pictures allow. `clear_kit`'s tote AMR at `BASE` heading +x, front edge x 1.0, camera 0.5 m up
at x 0.98 with fy 381.36 px; the one-pixel error of a floor point at distance d is (d^2 + 0.5^2) / (0.5 x 381.36)."""

from __future__ import annotations

import pytest

from clear_kit import LAST, Owned, drive_run, thing, wall, world
from space_kit import box
from stbench import clear_ground


def resolution(w) -> float:
    m = clear_ground.moment(w, LAST)
    (cand,) = m.answerable
    return cand.tolerances["resolution_m"]


def test_the_resolution_is_the_contacts_pixel_error_when_it_is_the_largest():
    """Contract (brief 3.7): a wall 3.53 m from the camera has its foot read to one pixel at (3.53^2 + 0.25) / 190.68 =
    0.0667 m, more than the sub-cell and with no gap (the foot is the stop): that is the resolution. A wall 0.92 m from
    the camera reads to 0.0057 m, under the sub-cell: the resolution is the 0.02 m floor. Hand values (the wall faces at
    x 4.51 and 1.90, the stops at 3.50 and 0.89). Regression caught: a resolution under what the grid can tell (a full
    credit no reader could reach) or one that ignores the camera's distance."""
    far, near = wall(4.51), wall(1.9)
    assert resolution(world(drive_run(boxes=[far.at]), [far])) == pytest.approx(0.0667, abs=0.003)
    assert resolution(world(drive_run(boxes=[near.at]), [near])) == 0.02


def table(leg_x: float):
    """A table whose top edge is the stop (its front at x 3.01, 0.45 to 0.50 m up) on legs standing from `leg_x`, seen
    in the look window."""
    top = Owned(box(3.01, 4.01, -0.4, 0.4, 0.45, 0.5), "E0002")
    legs = [Owned(box(leg_x, leg_x + 0.05, y, y + 0.05, 0.0, 0.45), "E0002") for y in (-0.4, 0.35)]
    script = drive_run(boxes=[top.at, *(leg.at for leg in legs)])
    thing(script, "E0002", (1.0, 0.8, 0.5), (3.51, 0.0, 0.0))
    script.see(list(range(16, 20)), "E0002", 1500)
    return world(script, [top, *legs])


def test_the_resolution_includes_the_gap_from_the_stop_to_its_contact():
    """Contract (brief 3.7): a stop whose floor contact lies beyond it is tied (within C/2 = 0.225) but only to that
    distance: a table top whose edge stops the corridor at 2.01 from the front edge on legs at x 3.11, 0.10 m behind
    the edge, is read at the legs' feet, so the resolution is the gap, 0.10 m (more than the 0.0251 m pixel at 2.13 m);
    with the legs at the edge it is the pixel, 0.0229 m. Regression caught: a tied stopper credited as if its contact
    were at the stop (a full credit for a corridor the frames place only to within the gap)."""
    assert resolution(table(3.11)) == pytest.approx(0.10, abs=0.006)
    assert resolution(table(3.01)) == pytest.approx(0.0229, abs=0.003)
