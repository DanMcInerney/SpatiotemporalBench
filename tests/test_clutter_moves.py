"""The two functions a later move of clutter is built on (`clutter.floor_spot`, `clutter.move_floor`): where a thing may
lie on the floor in a gap, and that a floor clutter piece re-drawn by the world's processes keeps off the light
furniture at both its places and in its kind's rooms. The worlds are the apartment's seeds 101-103 (`process_kit`);
hand values are the map's: the hall R1 runs x 4.65-6.15 and y 0.05-6.15, the laundry basket L06 has its map place at
(7.8, 1.95), 0.4 m square."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from process_kit import SEEDS, fresh, guard_off
from realism_kit import apartment, keepout_hits
from stbench import clutter
from stbench import world_processes_home as home
from stbench.catalog import place_of
from stbench.clutter_reach import FloorGuard
from stbench.generate import map_place
from stbench.geometry import Rect, rect_distance, rect_inside_rect, rects_overlap
from stbench.route import blocking_rects
from stbench.seeding import make_rng

# One xdist group (`--dist loadgroup`): the worlds are cached per worker process.
pytestmark = pytest.mark.xdist_group(name=__name__)


def test_floor_spot_obeys_clutter_rules():
    """Contract (M2-A2W §2, `place`'s floor rules): a spot `floor_spot` gives for a 0.3 x 0.2 x 0.3 m thing (taller than
    a robot climbs) in the hall, inside the area x 4.65-6.15, y 1.0-5.0, with a swept body at x 5.25-5.55, y 2.5-3.5 to
    avoid, lies wholly inside the area, 0.25 m (`WORLD_CLEARANCE_M`) or more from the swept body, off every keep-out
    (the doors' bands and swings, the dock's apron: `realism_kit.keepout_hits`), off every light piece, fixed piece and
    thing in the world, and `FloorGuard` allows it; with `anchors` ("sofa") it lies within 0.8 m of the sofa
    (`DROP_REACH_M`); and a guard that refuses everything leaves no spot. Regression caught: a drop on the robot's
    path, on a keep-out or on another thing, or a floor cut in two."""
    m, w = apartment(), fresh(101)
    size, area, swept = (0.3, 0.2, 0.3), Rect(5.4, 3.0, 1.5, 4.0, 0.0), Rect(5.4, 3.0, 0.3, 1.0, 0.0)
    sofa, guard, found = next(f for f in m.furniture if f.kind == "sofa").footprint(), FloorGuard(w), []
    for i in range(8):
        spot = clutter.floor_spot(w, m, make_rng("test_floor_spot", i), size, area=area, avoid=[swept])
        if spot is not None:
            found.append(Rect(spot[0], spot[1], size[0], size[1], spot[2]))
    assert len(found) >= 5, len(found)
    stands = [r for r, z0, _ in w.occupied() if z0 < 0.3]
    for box in found:
        assert rect_inside_rect(box, area) and rect_distance(box, swept) >= 0.25
        assert keepout_hits(m, box) == [] and guard.allows(box, size[2])
        assert not any(rects_overlap(box, r) for r in [*w.light_rects.values(), *blocking_rects(m), *stands])
    near = [clutter.floor_spot(w, m, make_rng("test_anchors", i), size, anchors=("sofa",)) for i in range(4)]
    assert any(near) and all(rect_distance(Rect(x, y, size[0], size[1], yaw), sofa) <= 0.8
                             for x, y, yaw in filter(None, near))
    with pytest.MonkeyPatch.context() as refuse:
        refuse.setattr(FloorGuard, "allows", lambda self, box, height: False)
        assert all(clutter.floor_spot(w, m, make_rng("test_refused", i), size, area=area) is None for i in range(3))


@pytest.fixture(scope="module")
def redrawn():
    """One round of `move_floor` over every floor clutter piece of each world (the guard off: the redraws are the
    subject here), each new place as (seed, piece, kind, room, footprint), and each world's light furniture, at its
    map places and where the start left it (it does not move here)."""
    m, out, light = apartment(), [], {}
    with pytest.MonkeyPatch.context() as mp:
        guard_off(mp)
        for seed in SEEDS:
            w, rng = fresh(seed), make_rng("test_redraws", seed)
            light[seed] = [*(map_place(w, m, eid) for eid in w.light_rects), *w.light_rects.values()]
            kinds = {e.id: e.kind for e in w.entities}
            for eid in home.floor_clutter(w):
                if clutter.move_floor(w, m, rng, eid, 100.0):
                    box = w.clutter[eid][0]
                    out.append(SimpleNamespace(seed=seed, kind=kinds[eid], box=box, room=m.region_of((box.cx, box.cy))))
    return SimpleNamespace(moves=out, light=light)


def test_clutter_keeps_off_both_places_of_light_furniture(redrawn, monkeypatch):
    """Contract (M2-A2W §2): a floor clutter piece placed again keeps off every light piece at its map place and where
    it stands now (`place` kept it off the map places, and a gap's draw of a chair may put it back). On seed 103, with
    the laundry basket put 0.77 m from its map place (7.8, 1.95), 0.4 m square, at (7.16, 1.51): in an area on that
    place a 0.15 m cube finds no spot in 40 draws, and finds one on a map without light furniture (the control: the
    place is open floor that only the basket's map place keeps); and over a round of re-draws of every floor
    clutter piece of the three worlds no new place overlaps a light piece at either of its places. Regression
    caught: clutter under a chair's start that a later process puts back."""
    guard_off(monkeypatch)
    m, w, size = apartment(), fresh(103), (0.15, 0.15, 0.2)
    basket, place = [e.id for e in w.entities if e.kind == "laundry_basket"][0], Rect(7.8, 1.95, 0.4, 0.4, 0.0)
    w.light_rects[basket] = Rect(7.16, 1.51, 0.4, 0.4, 1.53)
    assert rect_distance(w.light_rects[basket], place) > 0.2  # 0.24 m of floor between the two
    no_light = replace(m, furniture=tuple(f for f in m.furniture if f.mobility != "light"))
    assert all(clutter.floor_spot(w, m, make_rng("test_map_place", i), size, area=place) is None for i in range(40))
    assert any(clutter.floor_spot(w, no_light, make_rng("test_map_place", i), size, area=place) for i in range(40))
    assert len(redrawn.moves) >= 60
    for move in redrawn.moves:
        assert not any(rects_overlap(move.box, light) for light in redrawn.light[move.seed]), (move.seed, move.kind)


def test_a_moved_floor_piece_stays_in_its_kinds_rooms_and_only_floor_clutter_moves(redrawn):
    """Contract (M2-A2W §2, "placed again by its kind's places"): a floor clutter piece re-drawn by `move_floor` lies
    in a room its kind stands in (`ClutterKind.places`: the plunger and the bath bin in the bathroom, the pedal
    bin in the kitchen, a floor fan in the bedroom or the living room, a door stop anywhere), and a clutter piece that
    is not floor clutter (the toaster on a counter) is refused with a ValueError. Regression caught: a bin moved to
    the bedroom, or a toaster put on the floor."""
    rooms = {r.id: r.name for r in apartment().regions}
    catalog = fresh(101).catalog
    seen = set()
    for move in redrawn.moves:
        allowed = {place_of(p)[1] for p in catalog.clutter_kind(move.kind).places}
        assert None in allowed or rooms[move.room] in allowed, (move.kind, rooms[move.room], allowed)
        seen.add(move.kind)
    assert {"pedal_bin", "floor_fan", "plunger"} <= seen  # the kinds this checks are among the ones moved
    w = fresh(101)
    toaster = next(e.id for e in w.entities if e.kind == "toaster")
    with pytest.raises(ValueError, match="not floor clutter"):
        clutter.move_floor(w, apartment(), make_rng("test_toaster"), toaster, 100.0)
