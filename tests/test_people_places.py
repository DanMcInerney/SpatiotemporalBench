"""People where M3's robots work (lane M3-people): the warehouse's pickers and the downtown's pedestrians through the
overlay's additions (`people`: walk surfaces, keep-offs, the flier rule, the robot's own clearance, several people),
and the home's resident untouched. The scripts are `people_kit`'s hand scripts (an AMR driving across the dock, a drone
hovering over the plaza) with their people present for every task. A downtown walk grid is a million cells, so the
module builds each catalog's overlay once (`people_kit.with_people`, one xdist group), and the forced walks (a start and
a waypoint pinned, as `test_people.py` pins the home's) run on a map cropped to the action. Hand coordinates are
`downtown_a`'s and `warehouse_a`'s own."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from conftest import script_on_map
from people_kit import (
    drone_hover,
    footprints,
    generated,
    perimeter,
    person_ids,
    robot_gaps,
    with_people,
)
from stbench import asset_manifest, people, people_downtown, people_warehouse
from stbench import timeline as T
from stbench.geometry import Rect, point_in_rect, quat_z, rects_overlap
from stbench.mapspec import load_map
from stbench.mover_walk import Walk, Walker
from stbench.timeline import Entity, Placement
from stbench.validate import validate

pytestmark = pytest.mark.xdist_group("people_places")

PLAZA = (34.0, -25.0, 66.0, 5.0)  # the downtown's square, for the forced walks: a map cut to it, a 0.1 M-cell grid
SEAM_M = 0.011  # walk pieces stand up to 0.01 m apart (M3-maps insets neighbours by 0.005 m) and are one surface
BAD = {"mover_collision", "mover_camera_clearance", "body_clearance"}  # what a person may not break in a hand script


def covered(rects, x: float, y: float) -> bool:
    return any(point_in_rect(r, x, y) for r in rects)


def codes(tl, m) -> set[str]:
    return {p.code for p in validate(tl, m)}


def forced(m, mod, tl, start, *, waypoint=None, yaw=0.0, keep_off=None):
    """`people.add` with one person present for the task who starts at `start` facing `yaw` and, with a `waypoint`,
    walks to it after its first dwell (else stands where it started); `keep_off` replaces the module's. The overlay's
    timeline: the person is in `person_ids` unless the walk found no place to start."""
    stub = {"start": lambda self, keepout: Walk(start, yaw, 1e9 if waypoint is None else 0.0, (), 0.0)}
    if waypoint is not None:
        stub["_waypoint"] = lambda self, pos: waypoint
    with ExitStack() as stack:
        stack.enter_context(patch.object(mod, "P_HOME", 1.0))
        stack.enter_context(patch.object(mod, "COUNT", 1))
        stack.enter_context(patch.multiple(Walker, **stub))
        if keep_off is not None:
            stack.enter_context(patch.object(mod, "KEEP_OFF", keep_off))
        return people.add(tl, m, np.random.default_rng(1))


def test_the_homes_resident_is_untouched():
    """Contract (M3-people moves nothing of the home: the worker's recipe imports the resident's unchanged, and the
    overlay's additions are no-ops for a module that declares none): the manifest's `resident` row is as M2's merge left
    it (the file's name and the box), and a generated `vacuum/1` script of seed 102 holds a resident, so the overlay
    runs on a flat floor. Regression caught: the worker's recipe or the manifest edit moving the resident, or a home
    script generated with no one in it."""
    m = asset_manifest.packaged()
    resident = m.asset("resident")
    (body,) = resident.files
    assert body.path == "resident.glb"
    assert list(resident.native_bbox_m) == [0.33573, 0.56036, 1.76497]
    assert person_ids(generated(102))


def test_the_walk_surface_is_the_pieces_with_their_seams_closed():
    """Contract (M3-people §1, the review's minor 5: a body keeps wholly on the walk pieces): the rectangles a walk
    never enters (the map off its walk pieces) cover the roadway, the nest's yard and everything beyond, and none of
    the sidewalks and the plaza, whose 5 mm and 10 mm seams (M3-maps insets neighbouring pieces) are closed. Hand
    points from `downtown_a`: the high street's south sidewalk is y 0-4, the plaza y -12.995 to -0.005 (x 35.005-64.995)
    and its south half y -23.995 to -13.005 (x 45.005-64.995), the nest's yard x 35.005-45.005 below it. Regression
    caught: a pedestrian walking into the roadway or the yard, or a seam that cuts the plaza off the high street."""
    off = people._off_walk(load_map("downtown_a"), people_downtown.WALK_ON)
    on_the_pieces = [(50, 2), (50, -6), (55, -18), (17, 49.5), (50, 3.95), (17, 27.0)]
    seams = [(50, -0.0025), (55, -13.0)]  # sidewalk to plaza, plaza to its south half
    off_the_pieces = [(50, 10), (50, 4.05), (40, -18.5), (40, -14), (30, -6), (25, 50), (50, 72.5)]
    assert not any(covered(off, x, y) for x, y in on_the_pieces + seams)
    assert all(covered(off, x, y) for x, y in off_the_pieces)


def test_walk_pieces_must_be_one_height_and_axis_aligned():
    """Contract (M3-people §1): people stand at the one top of the pieces they walk on (`ValueError` naming the map when
    the tops differ by over 0.01 m or no piece exists), and the walk surface is built of axis-aligned pieces
    (`ValueError` for a piece turned off a quarter turn). A module with no `WALK_ON` (the warehouse's) stands at 0.
    Regression caught: a pedestrian sunk into a raised pavement or floating over a low one, a rotated piece quietly
    ignored by the exact surface."""
    m, mod = load_map("downtown_a"), people_downtown
    assert people.walk_height(m, mod) == 0.15 and people.walk_height(load_map("warehouse_a"), people_warehouse) == 0.0
    edited = lambda **change: replace(m, furniture=tuple(  # noqa: E731
        replace(f, **change) if f.id == "F05" else f for f in m.furniture))
    assert people.walk_height(edited(size=(4.0, 100.0, 0.155)), mod) == 0.155  # within 0.01 m: one height
    with pytest.raises(ValueError, match="downtown_a.*one height"):
        people.walk_height(edited(size=(4.0, 100.0, 0.25)), mod)
    bare = replace(m, furniture=tuple(f for f in m.furniture if f.kind not in mod.WALK_ON))
    with pytest.raises(ValueError, match="no fixed piece"):
        people.walk_height(bare, mod)
    with pytest.raises(ValueError, match="downtown_a.*quarter turns"):
        people._ground(edited(yaw=0.3), mod)


def test_the_walkers_obstacles_stand_above_its_floor():
    """Contract (M3-people §1, the review's minor 6): at the plaza's height (0.15 m) a fixed piece is an obstacle only
    above it (the sidewalks and the plaza, 0.15 m, are ground, and every fixed blocking piece of the downtown is one of
    them), a light piece always is, and a carried item or clutter only when its base is within 0.1 m of the floor: a
    box on the plaza at 0.15 m or 0.2 m is, one at 0 m (in the roadway below) or 0.5 m (on a shelf) is not; at 0 m (the
    home's floor) every fixed blocking piece is one. Hand case on the hover script's world, four boxes and a cart set
    out apart. Regression caught: a pedestrian that cannot step on a sidewalk, or that walks through a cart."""
    m, tl = load_map("downtown_a"), drone_hover(3)
    cart = Rect(50.0, -6.0, 1.2, 0.8, 0.0)
    boxes = {0.0: (60.0, -20.0), 0.15: (62.0, -20.0), 0.2: (64.0, -20.0), 0.5: (62.0, -2.0)}
    things = [("E0900", "food_cart", "light", (1.2, 0.8, 1.1), (50.0, -6.0, 0.15))]
    things += [(f"E09{i:02d}", "box", "carried", (0.3, 0.3, 0.3), (x, y, z))
               for i, (z, (x, y)) in enumerate(boxes.items(), 1)]
    for eid, kind, mobility, size, pos in things:
        tl.entities.append(Entity(eid, kind, {}, mobility, size))
        tl.placements.append(Placement(eid, 0.0, None, pos, quat_z(0.0), None))
    pieces = [f.footprint() for f in m.furniture if f.blocks and f.mobility == "fixed"]
    on_floor = people._obstacles(tl, m, 0.0, math.inf, 0.15)
    assert len(pieces) == 10 and not any(r in on_floor for r in pieces)
    assert all(r in people._obstacles(tl, m, 0.0, math.inf, 0.0) for r in pieces)
    assert cart in on_floor
    for z, (x, y) in boxes.items():
        assert (Rect(x, y, 0.3, 0.3, 0.0) in on_floor) is (z in (0.15, 0.2)), z


# -- the people the overlay draws --------------------------------------------------------------------------------------
def test_pedestrians_walk_on_sidewalks_and_the_plaza():
    """Contract (M3-people §1; `docs/people.md`): every pedestrian placement stands at the walk pieces' height (0.15 m)
    with its whole footprint on the sidewalks and the plaza (to their 0.01 m seams), never over the nest's yard (x
    35.005-45.005, y -23.995 to -12.995) or a thing lying in the roadway (the parked cars and vans, dumpsters, at base
    0): the world's own roadway items, 21 of them in this script, and two pedestrians who walk. Regression caught: a
    kerbside pedestrian overhanging 0.28 m into the cars parked along the street (the review's minor 5), one in the
    road, one in the yard, or one sunk into the pavement."""
    m, tl = with_people("downtown")
    walk = [f.footprint() for f in m.furniture if f.kind in people_downtown.WALK_ON]
    yard = Rect(40.005, -18.495, 10.0, 11.0, 0.0)
    road = [(p.t0, math.inf if p.t1 is None else p.t1, Rect(p.pos[0], p.pos[1], e.size[0], e.size[1], p.yaw))
            for e in tl.entities if e.mobility in ("carried", "light")
            for p in tl.placements_of(e.id) if p.pos[2] < 0.1]
    assert len(road) >= 20 and len(person_ids(tl)) == 2
    for eid in person_ids(tl):
        stands = footprints(tl, eid)
        assert len(stands) > 3 and len({(r.cx, r.cy) for _, _, r, _ in stands}) > 3  # it walks
        for t0, t1, rect, z in stands:
            assert z == pytest.approx(0.15, abs=1e-9)
            assert all(any(point_in_rect(w, x, y, SEAM_M) for w in walk) for x, y in perimeter(rect)), (eid, t0)
            assert not rects_overlap(rect, yard)
            assert not any(t0 < u1 and u0 < t1 and rects_overlap(rect, r) for u0, u1, r in road), (eid, t0)


def test_several_people_never_overlap():
    """Contract (M3-people §1: with `COUNT` above 1 each person treats the ones before as obstacles): the warehouse's
    three pickers and the downtown's two pedestrians never have footprints that overlap while both stand (written out
    from the placements), and `validate` finds no mover collision, camera clearance or body clearance in either hand
    script. Regression caught: two people drawn through each other, or a person the robot's body or camera comes
    up to."""
    for catalog in ("warehouse", "downtown"):
        m, tl = with_people(catalog)
        ids = person_ids(tl)
        assert len(ids) == (3 if catalog == "warehouse" else 2)
        stands = {eid: footprints(tl, eid) for eid in ids}
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                assert not any(a0 < b1 and b0 < a1 and rects_overlap(ra, rb)
                               for a0, a1, ra, _ in stands[a] for b0, b1, rb, _ in stands[b]), (catalog, a, b)
        assert not BAD & codes(tl, m), catalog


def test_people_keep_off_the_robots_zones():
    """Contract (M3-people §1 `KEEP_OFF`; the zone coins of `amr/1` and `drone/1`): no picker's footprint meets a bay
    and no pedestrian's a drop zone at any keyframe of the overlay's scripts; and pinned at the edge, on a cut map: a
    pedestrian who starts in the drop zone `Square West` (3 x 3 m at (40, -6)) finds no walk, one who starts 0.3 m
    outside its edge (x 41.8) does, and with the keep-off turned off the first does too. Regression caught: a zone coin
    that a person turns from clear to blocked (making `no` the likelier answer: VISION rule 5)."""
    for catalog, kind in (("warehouse", "bay"), ("downtown", "drop_zone")):
        m, tl = with_people(catalog)
        zones = [f.footprint() for f in m.furniture if f.kind == kind]
        assert len(zones) >= 4
        for eid in person_ids(tl):
            stands = [rect for _, _, rect, _ in footprints(tl, eid)]
            assert not any(rects_overlap(rect, z) for rect in stands for z in zones), (catalog, eid)
    m, mod, tl = replace(load_map("downtown_a"), bounds=PLAZA), people_downtown, drone_hover(5, bare=True)
    assert not person_ids(forced(m, mod, tl, (40.0, -6.0)))
    assert person_ids(forced(m, mod, tl, (41.8, -6.0)))
    assert person_ids(forced(m, mod, tl, (40.0, -6.0), keep_off=()))


def test_people_keep_off_things_on_their_floor():
    """Contract (M3-people §1): a pedestrian walking the plaza goes round a food cart (a light piece 1.2 x 0.8 m at
    0.15 m) that stands on the straight line from its start (46, -6) to its waypoint (54, -6): it reaches the far side
    (x past 52), no footprint meets the cart and `validate` finds no `mover_collision`; with the cart off the line the
    same walk passes through the cart's place, so the check can fail. Regression caught: a person walking through street
    furniture."""
    m, mod, full = replace(load_map("downtown_a"), bounds=PLAZA), people_downtown, load_map("downtown_a")
    cart = Rect(50.0, -6.0, 1.2, 0.8, 0.0)

    def walk(cart_at):
        tl = drone_hover(70, bare=True)
        tl.entities.append(Entity("E0900", "food_cart", {}, "light", (1.2, 0.8, 1.1)))
        tl.placements.append(Placement("E0900", 0.0, None, (*cart_at, 0.15), quat_z(0.0), None))
        out = forced(m, mod, tl, (46.0, -6.0), waypoint=(54.0, -6.0))
        (eid,) = person_ids(out)
        return out, [rect for _, _, rect, _ in footprints(out, eid)]

    out, stands = walk((50.0, -6.0))
    assert max(r.cx for r in stands) > 52.0 and not any(rects_overlap(r, cart) for r in stands)
    assert "mover_collision" not in codes(out, full)
    assert any(rects_overlap(r, cart) for r in walk((50.0, -12.0))[1])


def test_people_pass_under_a_flier():
    """Contract (M3-people §1 flier rule and `validate`'s own, the review's minor 7): where the drone flies over a
    pedestrian (`robots.flies_over`, its base height less its 3 m clearance against the person's own top: the plaza's
    0.15 m plus the box's 1.8 m) nothing of it keeps the pedestrian off, so one stands right under a drone hovering at
    7 m; where it does not (a hover at 4 m clears 1.0 m) the pedestrian keeps the robot's own clearance, 3 m and 1 cm
    beyond the disc of 0.65 m, which is more than the 0.5 m of `ROBOT_KEEP_M`: with its near edge at 2.98 m (centre 3.80
    m east) it finds no walk, at 3.04 m (3.86 m) it does, and under it it does not. And the overlay and `validate`
    agree on the person's own floor: standing at 0.15 m under a drone at 4.9 m (clears 1.9 m, under the person's top
    1.95 m) is a camera clearance problem, under one at 5.5 m it is not. The agreement needs the coordinator's two
    `validate_ego` lines (`_mover_camera`: the mover's own box and `p.pos[2] + top`; the review's minor 7): until then
    its last assertion alone fails. Regression caught: a crowd frozen by a drone 7 m up, one standing on its landing
    pad, or a flier passing over a person by the wrong height."""
    full, under = load_map("downtown_a"), (50.0, -8.0)
    m, mod = replace(full, bounds=PLAZA), people_downtown
    high, low = drone_hover(5, z=7.0, at=under, bare=True), drone_hover(5, z=4.0, at=under, bare=True)
    stands = forced(m, mod, high, under)
    (eid,) = person_ids(stands)
    assert {(*p.pos[:2], p.pos[2]) for p in stands.placements_of(eid)} == {(50.0, -8.0, 0.15)}
    assert not BAD & codes(stands, full)
    assert not person_ids(forced(m, mod, low, (53.80, -8.0)))
    assert person_ids(forced(m, mod, low, (53.86, -8.0)))
    assert not person_ids(forced(m, mod, low, under))

    def hand(z):
        tl = drone_hover(3, z=z, at=under)
        tl.entities.append(Entity("E0900", "person", {}, "mover", people_downtown.BOX_M))
        tl.placements.append(Placement("E0900", 0.0, None, (*under, 0.15), quat_z(0.0), None))
        return codes(tl, full)

    assert "body_clearance" in hand(4.9) and "body_clearance" not in hand(5.5)
    assert "mover_camera_clearance" not in hand(5.5)
    assert "mover_camera_clearance" in hand(4.9)  # fails until `validate_ego` reads the mover's own floor and box


def test_pickers_keep_the_robots_clearance():
    """Contract (M3-people §1: the keep from the robot's body is `max(ROBOT_KEEP_M, its clearance + 0.01)`): the AMR (a
    0.80 x 0.58 m body, circumscribed radius 0.494 m, clearance 0.10 m) drives 4 m east between two keyframes, from
    (4, 5) to (8, 5), and a picker stands at (6, 5 + d) facing east (half width 0.28 m), so the gap to the swept body is
    d - 0.28 - 0.494: d = 1.05 leaves 0.276 m and finds no walk (the keep is 0.3 m, the module's `ROBOT_KEEP_M`, not
    the 0.11 m of the AMR's own clearance), d = 1.10 leaves 0.326 m and stands; and the overlay's own pickers keep 0.3 m
    (less the outline's 0.01 m) from the body wherever they come within 2 m of it. Regression caught: a person the AMR
    drives through, or a margin that the robot's own clearance would alone set."""
    m, mod = load_map("warehouse_a"), people_warehouse
    tl = script_on_map("warehouse_a", 1, "amr", [(4.0, 5.0, 0.0, 0.0), (8.0, 5.0, 0.0, 0.0)])
    for d, stands in ((1.05, False), (1.10, True)):
        assert bool(person_ids(forced(m, mod, tl, (6.0, 5.0 + d)))) is stands, d
    m, tl = with_people("warehouse")
    near = [g for eid in person_ids(tl) for g in robot_gaps(tl, eid)]
    assert near and min(near) >= 0.3 - 0.01


def digests() -> dict[str, str]:
    """The timeline digests of the two catalogs' hand scripts with their people."""
    return {catalog: T.digest(with_people(catalog)[1]) for catalog in ("warehouse", "downtown")}


def test_people_are_deterministic():
    """Contract: a warehouse and a downtown hand script with people are byte-identical in a second process with another
    `PYTHONHASHSEED`, and the two catalogs' differ. Regression caught: an unseeded draw, or a set's or dict's iteration
    order in the walk surface or the obstacles."""
    env = {**os.environ, "PYTHONHASHSEED": "4242"}
    out = subprocess.run([sys.executable, __file__], capture_output=True, text=True, check=True, env=env).stdout.strip()
    here = digests()
    assert here == json.loads(out) and len(set(here.values())) == 2


if __name__ == "__main__":
    print(json.dumps(digests()))
