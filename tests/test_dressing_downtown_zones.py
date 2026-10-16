"""Where the downtown's render-only objects may not stand because the drone flies there (`render/delivery_keepout.py`,
used by `dressing_layout_downtown.py` and `scenery_layout.py`; lane M3-RY): the column round each drop zone and the
nest, and the nest's bare yard. Free: plain Python over `downtown_a.json`. The numbers are worked out by hand from the
map and the drone's body (a disc 1.30 m across, so a radius of 0.65 m, and a planner clearance of 3.0 m: 3.65 m from
everything it does not fly over). What `validate` keeps off a flight (the map's walls and pieces) is
`test_job_drone*.py`'s; what the generic dressing guard keeps off the open floor is `test_dressing.py`'s."""

from __future__ import annotations

import math
from functools import cache

import pytest

from dressing_kit import layout_of, raw_map, render_module
from stbench.geometry import corners
from stbench.mapspec import load_map, surface_world_rect

K = render_module("delivery_keepout")
SCENERY = render_module("scenery_layout")
LAYOUT = render_module("dressing_layout_downtown")
REACH_M, TOP_M, FLUSH_M = 3.65, 18.0, 0.02
# (centre, size) of the nest and the five drop zones (M3-maps: F25 and F26-F30), and each column's radius by hand: the
# half-diagonal of the piece, sqrt(3.0^2 + 3.0^2) / 2 = 2.121 (the square's three zones), sqrt(2.6^2 + 3.0^2) / 2 =
# 1.985 (the two street zones), sqrt(1.6^2 + 1.6^2) / 2 = 1.131 (the nest), plus 3.65
PIECES = {"F25": ((40.0, -18.5), (1.6, 1.6)), "F26": ((40.0, -6.0), (3.0, 3.0)), "F27": ((60.0, -6.0), (3.0, 3.0)),
          "F28": ((60.0, -18.0), (3.0, 3.0)), "F29": ((20.3, 44.1), (2.6, 3.0)), "F30": ((79.7, 44.1), (2.6, 3.0))}
RADIUS = {pid: math.hypot(*size) / 2 + REACH_M for pid, (_, size) in PIECES.items()}
# the nest's yard as M3-maps cut it: the square's two pieces of paving leave x 35.005 to 45.005, y -23.995 to -12.995
YARD = (35.005, 45.005, -23.995, -12.995)
ARM_M = {"lamp": 1.7, "signal": 4.5, "hydrant": 0.18, "tree": 1.5}  # how far a post reaches from its pole (scenery.py)


@cache
def _layout() -> tuple:
    return tuple(layout_of("downtown", raw_map("downtown_a")))


def gap(centre: tuple[float, float], rect: tuple[float, float, float, float]) -> float:
    """The distance in plan from a point to a rectangle (xmin, xmax, ymin, ymax), 0 inside it."""
    dx = max(rect[0] - centre[0], 0.0, centre[0] - rect[1])
    dy = max(rect[2] - centre[1], 0.0, centre[1] - rect[3])
    return math.hypot(dx, dy)


def plan(rect) -> tuple[float, float, float, float]:
    """A rectangle of the map (turned a quarter turn) as (xmin, xmax, ymin, ymax)."""
    xs, ys = [p[0] for p in corners(rect)], [p[1] for p in corners(rect)]
    return min(xs), max(xs), min(ys), max(ys)


def meets(rect: tuple[float, float, float, float], tol: float = 0.0) -> bool:
    """Whether a plan rectangle shares area with the yard (by more than `tol` on each axis)."""
    return rect[0] < YARD[1] - tol and YARD[0] + tol < rect[1] and rect[2] < YARD[3] - tol and YARD[2] + tol < rect[3]


def test_no_dressing_in_the_drones_reach():
    """Contract (`delivery_keepout`: the drone hovers 7 m over a drop zone and keeps 3.65 m from everything it does not
    fly over, and `validate` cannot see a render-only object): no object that stands (more than a 2 cm plate) and no
    lamp post, signal pole or hydrant of the scenery with its arm comes within a drop zone's or the nest's column, the
    disc about its centre of its half-diagonal plus 3.65 m (5.77 m for a square zone, 5.63 for a street zone, 4.78 for
    the nest), from the ground up to 18 m; a tree goes with its crown or stays with it. The objects that stood there
    in the stationing (read off the layout without the rule: four trees with their crowns, six benches, a bike rack, a
    planter, the East Street lamp) are gone, and the pieces beside them that stand clear are kept (the two street
    trees that stood 5.5 and 4.3 m from a street zone's centre when it was in the middle of its loading zone stand 10.8
    and 9.4 m from it at the south end). Regression caught: a drone drawn inside a tree's crown or against a lamp's arm
    in a frame, or a zone that a bollard or bench blocks in every world (a fact of the layout, which `blind` learns)."""
    solid = [d for d in _layout() if d.box[5] > FLUSH_M]
    assert len(solid) >= 40  # 15 trees with their crowns, 4 benches, 8 planters and 3 bike racks stand
    for pid, (centre, _) in PIECES.items():
        for d in solid:
            rect = (d.box[0], d.box[3], d.box[1], d.box[4])
            assert gap(centre, rect) >= RADIUS[pid] or d.box[2] >= TOP_M, (pid, d.cls, d.pos)
        for p in SCENERY.layout(load_map("downtown_a")).posts:
            arm = ARM_M[p.kind]
            assert gap(centre, (p.x - arm, p.x + arm, p.y - arm, p.y + arm)) >= RADIUS[pid], (pid, p.kind, p.x, p.y)
    stems = [d for d in _layout() if d.cls == "street_tree"]
    crowns = [d for d in _layout() if d.cls == "tree_crown"]
    for s in stems:  # every stem keeps its crown, 0.6 m from it on a sidewalk and over it in the square
        assert min(math.dist(s.pos[:2], c.pos[:2]) for c in crowns) <= 0.6 + 1e-6, s.pos
    gone = {("street_tree", (37.3, -9.5)), ("street_tree", (62.7, -9.5)), ("street_tree", (54.51, -21.7)),
            ("street_tree", (60.51, -21.7)), ("bench", (35.91, -6.5)), ("bench", (64.09, -6.5)),
            ("bench", (57.51, -23.1)), ("bench", (64.09, -17.5)), ("bench", (18.5, 42.38)), ("bench", (81.5, 38.84)),
            ("bike_rack", (81.5, 45.62)), ("planter", (63.8, -23.0))}
    left = {(d.cls, (round(d.pos[0], 2), round(d.pos[1], 2))) for d in _layout()}
    assert not gone & left, sorted(gone & left)
    assert {("street_tree", (48.51, -21.7)), ("bench", (51.51, -23.1)), ("planter", (46.21, -23.0)),
            ("street_tree", (18.5, 54.7)), ("street_tree", (81.5, 53.36))} <= left
    lamps = {(round(p.x, 1), round(p.y, 1)) for p in SCENERY.layout(load_map("downtown_a")).posts if p.kind == "lamp"}
    # the East Street lamp (arm box x 79.8 to 83.2, y 36.3 to 39.7: 4.4 m from the zone's centre) went; the West Street
    # lamp (arm box y 50.3 to 53.7: 6.2 m from its zone's) and the high street's stay
    assert (81.5, 38.0) not in lamps and (18.5, 52.0) in lamps and (10.0, 23.5) in lamps


def test_the_yard_is_bare():
    """Contract (M3-maps: the nest stands in a bare yard in the square's south-west corner, the ground the square's two
    pieces of paving leave, 10 x 11 m, so a landing drone has 3.65 m clear): in the yard x 35.005 to 45.005, y -23.995
    to -12.995 no render-only object stands (the only things in it are the two white lines that mark its open edges, 2
    cm plates inside the outer 0.25 m of the yard's north and east sides), no piece of sidewalk or paving has any of its
    footprint in it, and no piece of the map has a receptacle surface in it (no thing is ever put down there). The
    nest's own centre lies in it, and the layout's keep-out finds the same yard from the pieces and walls (the walls'
    faces are 5 mm outside it on the west and south). Regression caught: a landing pad that people, items or street
    furniture stand on, or a yard cut larger than the bare ground (a line drawn on the paving)."""
    layout = _layout()
    inside = [d for d in layout if meets((d.box[0], d.box[3], d.box[1], d.box[4]))]
    assert sorted((d.cls, d.recipe) for d in inside) == [("lane_marking", "proc:paint:bar:x"),
                                                         ("lane_marking", "proc:paint:bar:y")]
    north, east = sorted(inside, key=lambda d: d.recipe)
    assert north.box == pytest.approx((35.0, -13.245, 0.0, 44.905, -13.095, 0.012), abs=1e-6)
    assert east.box == pytest.approx((44.755, -24.0, 0.0, 44.905, -13.245, 0.012), abs=1e-6)
    m = load_map("downtown_a")
    for f in m.furniture:
        if f.kind in ("sidewalk", "plaza"):
            assert not meets(plan(f.footprint()), tol=1e-6), f.id
        for s in f.surfaces:
            assert not meets(plan(surface_world_rect(f.center, f.yaw, s))), (f.id, s.id)
    nest = next(f for f in m.furniture if f.kind == "nest")
    assert YARD[0] < nest.center[0] < YARD[1] and YARD[2] < nest.center[1] < YARD[3]
    (found,) = LAYOUT.keep_out(raw_map("downtown_a")).yards
    assert found.rect == pytest.approx((35.0, 45.005, -24.0, -12.995), abs=1e-6) and found.ground_sides == ("+x", "+y")


def test_a_yard_runs_out_to_the_nearest_ground_or_wall_and_a_nest_in_the_open_has_none():
    """Contract (`delivery_keepout.yard`, generic over a map's pieces: a nest is found by its kind): on a hand-made
    square with a nest at (5, 5), a wall slab ending at x 2 abeam of it, a raised piece of ground starting at x 9 abeam
    of it, a piece of ground starting at y 8 over it and a wall slab ending at y 1 under it, the yard is x 2 to 9, y 1
    to 8, bounded by ground on its +x and +y sides; a piece of ground that is not abeam (its y range misses the nest's)
    does not stop it; and with no wall under the nest the yard has no south side to stop on, which is a ValueError.
    Regression caught: a yard cut by a piece it does not touch, one that ignores a wall, or a nest in the open
    given a yard the size of the map."""
    ground_east, ground_north = (9.0, 12.0, 4.0, 6.0), (3.0, 7.0, 8.0, 11.0)
    off_abeam = (6.0, 7.0, 6.5, 7.5)  # between the nest and the east piece, but its y range misses y = 5
    walls = [(0.0, 2.0, 0.0, 10.0), (0.0, 12.0, 0.0, 1.0)]
    y = K.yard((5.0, 5.0), [ground_east, ground_north, off_abeam], walls)
    assert y.rect == (2.0, 9.0, 1.0, 8.0) and y.ground_sides == ("+x", "+y")
    with pytest.raises(ValueError, match="nothing to stop its yard"):
        K.yard((5.0, 5.0), [ground_east, ground_north], walls[:1])
