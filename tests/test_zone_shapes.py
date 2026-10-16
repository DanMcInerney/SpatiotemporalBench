"""The shapes of the new zone sources (`zone_shapes`): an interior opening's way and the window of the surface the robot
faces, by hand on `apartment_a` (its doors and arch) and on `shape_kit`'s small synthetic map (a table, a fridge, a rug,
a light chair), and by a sweep of every surface of `apartment_a` seen from the four sides of its piece."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from shape_kit import APARTMENT, CHAIR, NORTH, RUG, SYNTHETIC, extents, piece, top
from stbench import validate, zone_shapes
from stbench.geometry import Rect, point_in_rect, rect_inside_rect, rects_overlap
from stbench.mapspec import MapSpec, surface_world_rect


def way(opening_id: str, m: MapSpec = APARTMENT) -> Rect:
    return zone_shapes.doorway_way(m, next(o for o in m.openings if o.id == opening_id))


def test_a_way_by_hand():
    """Contract: `doorway_way` is the opening's width less 0.1 m (`JAMB_M`) from each jamb across, and along the way
    from 0.05 m (`SWING_IN_M`) inside the wall face on a door's swing side to 0.3 m (`WAY_M`) beyond the other face; an
    arch runs 0.3 m beyond both faces. `apartment_a`'s walls are 0.1 m thick, so by hand: D1 (x = 4.6, 0.85 m wide at
    y 1.9, swings west) is 0.65 across, y 1.575..2.225, and x 4.6 (the wall's middle: 0.05 inside the west face) to
    4.95 (0.3 past the east face); D3 (x = 6.2, 0.8 m wide at y 1.3, swings east) is y 1.0..1.6 and x 5.85..6.2, the
    other way round; D0 (y = 0, 0.95 m at x 5.4, swings north) is x 5.025..5.775 and y -0.35..0; the arch O1 (y = 2.8,
    1.2 m wide at x 3.6) is x 3.1..4.1 and y 2.45..3.15. The wall's own thickness is read: D1's wall at 0.2 m gives x
    4.55 (0.05 inside its west face at 4.5) to 5.0 (0.3 past its east face at 4.7). An opening the jamb inset leaves
    no width is refused. Regression caught: a way on the wrong side of a door (the open leaf's side), a swing sign read
    the wrong way round for a right-swinging door, an arch given a swing side, or a wall thickness taken from a
    constant."""
    assert extents(way("D1")) == (4.6, 4.95, 1.575, 2.225)
    assert extents(way("D3")) == (5.85, 6.2, 1.0, 1.6)
    assert extents(way("D0")) == (5.025, 5.775, -0.35, 0.0)
    assert extents(way("O1")) == (3.1, 4.1, 2.45, 3.15)
    thick = replace(APARTMENT, walls=tuple(replace(w, thickness=0.2) if w.id == "W05" else w
                                           for w in APARTMENT.walls))
    assert extents(way("D1", thick)) == (4.55, 5.0, 1.575, 2.225)
    too_wide = replace(APARTMENT, openings=tuple(replace(o, width=0.15) if o.id == "O1" else o
                                                 for o in APARTMENT.openings))
    with pytest.raises(ValueError, match="O1"):
        way("O1", too_wide)


def test_no_way_holds_a_jamb_or_an_open_leaf():
    """Contract: no interior opening's way comes within 0.09 m of a jamb (the ends of the wall pieces `wall_pieces`
    cuts round it: the way keeps 0.1 m) or within 0.05 m of the open leaf of a door, drawn as thickly as the renderer
    can put it (0.04 m thick, standing 0.02 m off the hinge jamb: 0.08 m about the hinge line covers it), so that
    neither a wall sub-cell nor a leaf sub-cell (0.02 m) falls in a way. By hand the gap to the leaf is 0.06 m and to
    a jamb 0.1 m; an inset of 0.05 m (the first form) leaves 0.01 m to the leaf, which the check refuses. Regression
    caught: the hinge corner of the bathroom door D3 read as blocked in 9 of 10 empty views, which made a doorway
    question's gold `no` where nothing stood in it."""
    interior = [o for o in APARTMENT.openings if o.interior]
    assert [o.id for o in interior] == ["D1", "D2", "D3", "D4", "O1"]
    walls = APARTMENT.wall_pieces()
    for o in interior:
        w = zone_shapes.doorway_way(APARTMENT, o)
        assert not any(rects_overlap(w, piece_, margin=0.09) for piece_ in walls), o.id
        if o.type == "door":
            leaf = validate.door_leaf_rect(APARTMENT, o, True, 0.08)
            assert not rects_overlap(w, leaf, margin=0.05), o.id
            assert rects_overlap(w._replace(sx=w.sx + 0.1), leaf, margin=0.05), o.id  # the old 0.05 m inset fails


# -- the window -----------------------------------------------------------------------------------------------
def window(xy, yaw: float, m: MapSpec = SYNTHETIC):
    return zone_shapes.surface_window(m, xy, yaw)


def test_the_window_by_hand():
    """Contract: `surface_window` finds the first blocking fixed piece the heading ray meets within 1.5 m, its largest
    surface, and a window 0.4 m long along the surface's long side, centred on the ray's hit point (moved inward until
    it fits), across the whole inset depth, the surface inset 0.03 m. By hand on the table (inset surface x 1.48..2.52,
    y 2.78..3.22): from (2.0, 2.0) heading north the hit is (2.0, 2.7), so the window is x 1.8..2.2 and y 2.78..3.22
    (0.4 x 0.44 m); from (2.45, 2.0) the hit is at x 2.45, past where a 0.4 m window fits, so it is moved in to x
    2.12..2.52 (and from (1.52, 2.0) to 1.48..1.88); from (1.0, 2.0) heading to (2.0, 2.7) it enters the south face at x
    2.0, so the window is the same as the first. A surface shorter than 0.4 m is the whole inset surface (the stool:
    0.26 m less 0.06, a square of 0.2 m); of the bench's two surfaces the larger is used (1.4 m wide); of the shelf's
    two equal ones the first (0.4 m up). Regression caught: a window not centred on the hit, one that leaves its
    surface at an end, a surface chosen by its height, or a window cut to less than the surface's depth."""
    found = window((2.0, 2.0), NORTH)
    assert (found[0].id, found[1].id, found[1].z) == ("F01", "S1", 0.75)
    assert extents(found[2]) == (1.8, 2.2, 2.78, 3.22) and round(found[2].sy, 6) == 0.4
    assert extents(window((2.45, 2.0), NORTH)[2]) == (2.12, 2.52, 2.78, 3.22)
    assert extents(window((1.52, 2.0), NORTH)[2]) == (1.48, 1.88, 2.78, 3.22)
    assert extents(window((1.0, 2.0), math.atan2(0.7, 1.0))[2]) == (1.8, 2.2, 2.78, 3.22)
    assert extents(window((4.0, 2.0), NORTH)[2]) == (3.9, 4.1, 2.9, 3.1)
    bench = window((4.0, 5.0), NORTH)
    assert (bench[0].id, bench[1].id) == ("F07", "S7") and round(bench[2].sy, 6) == 0.4
    assert window((8.0, 2.0), NORTH)[1].id == "S2"


def test_no_window_where_the_ray_meets_no_surface():
    """Contract: `surface_window` is None when no blocking fixed piece lies within 1.5 m along the heading (1.7 m away,
    or facing the other way), and when the piece first met has no surface (a fridge in front of the robot is not a
    table behind it); a rug (it blocks nothing) and a light chair (it moves) in the ray's way are not met. A piece the
    ray starts inside is met at 0. Regression caught: a window on a piece beyond the reach, on the table behind a
    fridge, on a rug's or a chair's surface, or a robot facing a wall given a spot."""
    assert window((2.0, 1.0), NORTH) is None  # the table's face is 1.7 m ahead
    assert window((2.0, 2.0), -NORTH) is None  # facing away
    assert window((6.0, 2.0), NORTH) is None  # the fridge has no surface
    behind = replace(SYNTHETIC, furniture=(*SYNTHETIC.furniture, piece(
        "F08", "table", (6.0, 3.9), (0.6, 1.2, 0.75), NORTH, surfaces=[top("S8", (0, 0, 0.5, 1.1), 0.75)])))
    assert window((6.0, 2.0), NORTH, behind) is None
    assert window((2.0, 1.6), NORTH)[0].id == "F01"  # the light chair (1.7..2.1) and the rug (1.9..2.7) are passed
    assert window((2.0, 3.0), NORTH)[0].id == "F01"  # starting inside the table
    assert (CHAIR.mobility, RUG.blocks) == ("light", False)  # the kit's two pieces are what the test says they are


def test_the_nearer_piece_is_met_first():
    """Contract: of two pieces on the ray, the nearer is met, whatever the map order. Regression caught: the first piece
    in map order, or the last, chosen instead of the nearest."""
    far = piece("F10", "table", (2.0, 3.0), (0.6, 1.2, 0.75), NORTH, surfaces=[top("S9", (0, 0, 0.5, 1.1), 0.9)])
    near = replace(far, id="F11", center=(2.0, 2.4), surfaces=(top("S10", (0, 0, 0.5, 1.1), 0.8),))
    for order in ((far, near), (near, far)):
        got = window((2.0, 1.5), NORTH, replace(SYNTHETIC, furniture=order))
        assert (got[0].id, got[1].z) == ("F11", 0.8)


def passes_through(rect: Rect, base, yaw: float) -> bool:
    """Whether the ray from `base` heading `yaw` passes through the rectangle in its first 3 m, sampled every 2 mm."""
    return any(point_in_rect(rect, base[0] + s * math.cos(yaw), base[1] + s * math.sin(yaw), 1e-9)
               for s in np.arange(0.0, 3.0, 0.002))


def test_every_window_of_apartment_a_lies_on_its_surface_and_the_ray():
    """Contract: for every surface of `apartment_a`, seen from 0.8 m out of each of its four sides, straight on, at five
    places along the side (the middle and 40% and 80% of the way to each end), the window is the first piece's (the one
    stood before), lies inside the surface inset 0.03 m (the inset surface worked out here from `surface_world_rect`),
    is 0.4 m long along the surface's long side or the whole of it when that is shorter, spans the inset surface's whole
    other side, and the heading ray passes through it. Regression caught: a window that leaves its surface, shifts off
    the ray, or loses its depth on a turned piece or one whose surface is off its centre (the sink unit, the sofa)."""
    seen = 0
    for f in zone_shapes.blocking_pieces(APARTMENT):
        s = zone_shapes.largest_surface(f)
        if s is None:
            continue
        full = surface_world_rect(f.center, f.yaw, s)
        inner = Rect(full.cx, full.cy, full.sx - 0.06, full.sy - 0.06, full.yaw)
        x_long = inner.sx >= inner.sy
        for facing in ((1, 0), (-1, 0), (0, 1), (0, -1)):  # the side faced, in the surface's frame: +x, -x, +y, -y
            across, along = (inner.sx, inner.sy) if facing[0] else (inner.sy, inner.sx)
            for share in (0.0, 0.4, -0.4, 0.8, -0.8):
                out, side = (across / 2 + 0.8) * (facing[0] or facing[1]), share * along / 2
                local = (out, side) if facing[0] else (side, out)
                c, sn = math.cos(inner.yaw), math.sin(inner.yaw)
                base = (inner.cx + c * local[0] - sn * local[1], inner.cy + sn * local[0] + c * local[1])
                yaw = inner.yaw + math.atan2(-facing[1], -facing[0])  # heading at the surface
                found = zone_shapes.surface_window(APARTMENT, base, yaw)
                if found is None or found[0].id != f.id or not APARTMENT.contains(base):
                    continue  # another piece or a wall is in the way, or the base is outside the rooms
                seen += 1
                _, got, w = found
                assert got == s
                assert rect_inside_rect(w, inner, 1e-9), (f.id, base, yaw)
                length, depth = (w.sx, w.sy) if x_long else (w.sy, w.sx)  # along the long side, across it
                assert round(length, 6) == round(min(0.4, max(inner.sx, inner.sy)), 6), (f.id, w)
                assert round(depth, 6) == round(min(inner.sx, inner.sy), 6), (f.id, w)
                assert passes_through(w, base, yaw), (f.id, base, yaw)
    assert seen > 150  # the sweep reaches most of the 17 surfaces' sides, so the checks above are not vacuous
