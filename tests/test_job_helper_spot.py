"""`helper/1`'s place spot (`job_helper_spot`): the window the helper faces, the coin that sets it blocked or clear, the
response that follows the world, and where the thing goes. The worlds are the apartment's seeds 101-103 as the
generator builds one before it plans a path (`process_kit.fresh`). Hand values are the map's: the helper stands at
(5.3875, 5.3625) heading +90 degrees at the hall console F20 (at (5.4, 6.0), turned -90 degrees; its surface S13 is 0.26
m deep and 0.86 m long at 0.8 m), and the window is 0.4 m of the surface about the heading ray's hit point, 0.2 m deep
after the 0.03 m inset: x 5.2875-5.4875, y 5.8-6.2."""

from __future__ import annotations

import math

import numpy as np
import pytest

from process_kit import fresh
from stbench import job_helper_spot as spot
from stbench.catalog import FLOOR_ITEM_Z
from stbench.clutter_surface import standing
from stbench.geometry import Rect, corners, point_in_rect, rects_overlap
from stbench.job_helper_reach import ROBOT_ID, free_space_for
from stbench.mapspec import load_map
from stbench.robots import ROBOTS
from stbench.seeding import make_rng

pytestmark = pytest.mark.xdist_group(name=__name__)  # the worlds are cached per worker process
M = load_map("apartment_a")
POSE = ((5.3875, 5.3625), math.pi / 2)  # the reach pose at the hall console (tests/test_job_helper_paths.py)
T = 5000.0
ROOMS = frozenset({"R1", "R2"})


def floor_cells(w) -> np.ndarray:
    """The helper's free cells in the world at `T`: what every planned path of the helper is made of."""
    return free_space_for(M, w, ROBOTS[ROBOT_ID], ROOMS, T).free.copy()


def written(w) -> list:
    """The placements the ledger holds that start at `T`."""
    return [p for p in w.ledger.finish()[0] if p.t0 == T]


def test_the_window_is_forty_centimetres_of_the_surface_the_base_faces():
    """Contract (`zone_shapes.surface_window`, as the helper reads it): the window of a base at the hall console's reach
    pose is on the console F20's surface S13, 0.2 m deep and 0.4 m long, centred at x 5.3875, y 6.0 (the heading ray
    hits the console at x 5.2875 + ..., by hand 0.2 m deep after the inset), and the world's slot for that surface
    carries it; a base facing open floor has no window and the call refuses it, naming the pose. Regression caught: a
    window on the wrong surface (the sofa's, a fridge's), or a coin set on a place that is no surface."""
    w = fresh(101)
    win = spot.window_at(M, w, POSE)
    assert (win.piece.id, win.surface.id, win.slot.surface_id, win.ground) == ("F20", "S13", "S13", 0.8)
    assert win.rect.sx == pytest.approx(0.2) and win.rect.sy == pytest.approx(0.4)
    assert (win.rect.cx, win.rect.cy) == pytest.approx((5.3875, 6.0), abs=1e-3)
    with pytest.raises(ValueError, match="faces no surface"):
        spot.window_at(M, w, ((5.4, 3.0), 0.0))


def test_a_clear_coin_empties_the_window_on_surfaces_alone():
    """Contract: a clear coin moves everything that stands in the window grown by 0.05 m onto other surfaces, and the
    response then is `go`; nothing it writes is on the floor and the helper's free floor is cell for cell what it was.
    The thing the helper carries is never moved. Hand case: seed 101's hall console holds picture frames and an
    ornament in the window (the response is `elsewhere` before), and after the coin nothing stands in the window grown
    by 0.05 m, every placement written at the coin's time is a surface clutter piece or a carried item on a support, at
    least 0.1 m up, and the free cells of the helper's floor are equal. Regression caught: a coin that leaves a blocker
    in a window it calls clear (the response would lie), one that drops an item on the floor (every planned path would
    cross it), or one that moves the thing it is about to carry."""
    w = fresh(101)
    win = spot.window_at(M, w, POSE)
    assert spot.response(w, win) == spot.ELSEWHERE
    floor = floor_cells(w)
    keep = spot.blockers(w, win)[0]
    kept = (w.clutter.get(keep), w.carried[keep].spot if keep in w.carried else None)
    spot.enforce(w, M, make_rng("test_clear", 1), win, False, T, keep)
    assert [e for e in standing(w, win.rect, win.ground, spot.BAND, spot.DEEP_M) if e != keep] == []
    assert keep in spot.blockers(w, win)
    for p in written(w):
        assert p.entity != keep and p.on is not None and p.pos[2] >= FLOOR_ITEM_Z
    assert (w.clutter.get(keep), w.carried[keep].spot if keep in w.carried else None) == kept
    assert np.array_equal(floor_cells(w), floor)
    spot.enforce(w, M, make_rng("test_clear", 2), win, False, T + 1.0, "no such thing")
    assert spot.blockers(w, win) == [] and spot.response(w, win) == spot.GO


def test_a_blocked_coin_puts_one_raised_item_deep_in_a_clear_window():
    """Contract: when nothing stands in the window shrunk by 0.05 m, a blocked coin moves exactly one carried item that
    lies on a raised surface (not the thing) onto it, its footprint reaching more than 0.05 m into the window, and the
    response then is `elsewhere`; the helper's free floor is cell for cell what it was; a window that is already blocked
    is left alone. Hand case: seed 101's hall console after a clear coin. Regression caught: a blocker the zone
    families would not see (a sliver at the window's edge), an item taken off the floor (the floor would change), two
    items moved for one coin, or the thing moved onto its own target."""
    w = fresh(101)
    win = spot.window_at(M, w, POSE)
    spot.enforce(w, M, make_rng("test_clear", 1), win, False, T - 10.0, "no such thing")
    assert spot.blockers(w, win) == []
    floor = floor_cells(w)
    raised = {e for e, c in w.carried.items() if c.spot.pos[2] >= FLOOR_ITEM_Z}
    spot.enforce(w, M, make_rng("test_blocked", 1), win, True, T, "no such thing")
    moved = [p for p in written(w) if p.entity in w.carried]
    assert len(moved) == 1 and moved[0].entity in raised and moved[0].on == win.slot.furniture_entity
    c = w.carried[moved[0].entity]
    box = Rect(c.spot.pos[0], c.spot.pos[1], c.spec.size[0], c.spec.size[1], c.spot.yaw)
    assert rects_overlap(box, spot.grow(win.rect, -spot.DEEP_M))
    assert spot.response(w, win) == spot.ELSEWHERE and np.array_equal(floor_cells(w), floor)
    spot.enforce(w, M, make_rng("test_blocked", 2), win, True, T + 1.0, "no such thing")
    assert [p.entity for p in w.ledger.finish()[0] if p.t0 == T + 1.0] == []  # already blocked: nothing more moves
    other = fresh(101)
    spot.enforce(other, M, make_rng("test_clear", 1), win, False, T - 10.0, "no such thing")
    spot.enforce(other, M, make_rng("test_blocked", 1), win, True, T, moved[0].entity)
    assert moved[0].entity not in [p.entity for p in written(other)]


def test_the_thing_goes_to_the_windows_centre_or_beside_the_occupied_spot():
    """Contract: with the window clear (`go`) a thing small enough is put down at the window's centre, a quarter turn to
    the surface with its longer side along the surface's; with the window blocked (`elsewhere`) it is put on the free
    spot of the same surface nearest the window's centre that lies outside the window, clear of everything by 0.015 m
    and wholly on the surface. Hand case: seed 101's hall console and a mug (0.131 x 0.097 m): at the centre (5.3875,
    6.0) with its long side along the console's 0.86 m side (yaw 0 on the console turned -90 degrees: +90 degrees to
    it), then, after a blocked coin, off the window by at least 5 mm (y outside 5.8-6.2, or x outside 5.2875-5.4875),
    `ITEM_MARGIN` from the blocker, wholly on the console. Regression caught: a thing put on the occupied spot, on
    another thing, half off the surface, or inside the window the response called occupied."""
    w = fresh(101)
    win = spot.window_at(M, w, POSE)
    mug = next(e for e, c in w.carried.items() if c.spec.kind == "mug" and c.spot.pos[2] >= FLOOR_ITEM_Z)
    spot.enforce(w, M, make_rng("test_clear", 1), win, False, T, mug)
    go = spot.place_spot(w, mug, win, outside=False)
    assert go.pos[:2] == pytest.approx((5.3875, 6.0), abs=1e-3) and go.pos[2] == win.ground
    assert math.cos(go.yaw - win.slot.rect.yaw - math.pi / 2) == pytest.approx(1.0)
    spot.enforce(w, M, make_rng("test_blocked", 1), win, True, T + 1.0, mug)
    away = spot.place_spot(w, mug, win, outside=True)
    size = w.carried[mug].spec.size
    box = Rect(away.pos[0], away.pos[1], size[0], size[1], away.yaw)
    assert not rects_overlap(box, win.rect, spot.OUTSIDE_M - 1e-9) and rects_overlap(win.slot.rect, box)
    assert all(point_in_rect(win.slot.rect, x, y) for x, y in corners(box))
    rest = [r for r, lo, hi in w.without(mug).occupied() if lo < win.ground + size[2] and win.ground < hi]
    assert not any(rects_overlap(box, r, 0.015 - 1e-9) for r in rest)


def test_a_thing_wider_than_the_surface_has_no_place_on_it():
    """Contract: `fits` says whether a box lies on a surface a quarter turn to it either way with 0.01 m to the edges,
    and `place_spot` finds no spot for one that does not. By hand on the hall console (0.26 m deep): a 0.40 x 0.30 m box
    needs 0.32 m of depth turned either way, so it does not fit; the mug (0.131 x 0.097 m) does. Regression caught: a
    thing chosen that no surface of its target can hold (the carry would then end half off the console)."""
    w = fresh(101)
    win = spot.window_at(M, w, POSE)
    assert not spot.fits((0.40, 0.30, 0.267), win.slot) and spot.fits((0.131, 0.097, 0.115), win.slot)
    box = next(e for e, c in w.carried.items() if c.spec.kind == "box")
    assert not spot.fits(w.carried[box].spec.size, win.slot)
    assert spot.place_spot(w, box, win, outside=False) is None
