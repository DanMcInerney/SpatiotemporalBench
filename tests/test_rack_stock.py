"""`rack_stock` (the warehouse racks' floor-level stock, read from the map) against the renderer that draws it
(`render/rack_layout.py`, loaded from its file as `test_asset_render` loads it: render/ is not a package and has no
bpy). Expected places are worked out by hand from the bay arithmetic, with made-up racks where a map has none of a
shape."""

from __future__ import annotations

import importlib.util
import math
from dataclasses import replace
from pathlib import Path

import pytest

from stbench import rack_stock
from stbench.geometry import Rect, to_local
from stbench.mapspec import Furniture, Surface, load_map

RENDER = Path(__file__).resolve().parent.parent / "render"
SIZE = (1.1, 10.8, 3.9)  # the warehouse's rack row: depth, length (4 bays of 2.7 m), height (3 levels of 1.3 m)


def render_layout():
    """`render/rack_layout.py`, loaded from its file."""
    spec = importlib.util.spec_from_file_location("rack_layout", RENDER / "rack_layout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def racks(m):
    return [f for f in m.furniture if f.kind == "pallet_rack"]


def surfaces_of(f) -> list[tuple[float, float, float, float, float]]:
    return [(*s.rect, s.z) for s in f.surfaces]


def rack(yaw: float, center=(10.0, 10.0), free=()) -> Furniture:
    """A rack row of the warehouse's size with a surface at each (bay, level) of `free`, in the rack's frame."""
    surfaces = tuple(Surface(f"S{b}{v}", "rack", (0.0, (b + 0.5) * 2.7 - 5.4, 1.08, 2.6), v * 1.3) for b, v in free)
    return Furniture("F1", "pallet rack", "pallet_rack", "fixed", center, SIZE, yaw, "grey", True, surfaces)


def test_the_bay_levels_agree_with_the_renderer_on_the_warehouse():
    """Contract (plan-warehouse section 1): `rack_stock.layout` finds the same bays, levels and free bay-levels as the
    renderer's `rack_layout.layout` for every rack row of `warehouse_a`, and rejects the same surfaces (one off the
    bay's middle, one off a level's height) with the same error, so the stock the question treats as don't-care stands
    exactly where the renderer draws it. By hand F13's six surfaces are bays 0..3 by levels 0..2 at (0, 0), (0, 2),
    (1, 1), (2, 0), (2, 2) and (3, 1). Regression caught: the two drifting apart (a bay length, a level height or the
    snap changed in one place), which would put a don't-care place where nothing is drawn or leave drawn stock
    unprotected."""
    ref, wh = render_layout(), load_map("warehouse_a")
    assert len(racks(wh)) == 6
    for f in racks(wh):
        assert rack_stock.layout(f.size, surfaces_of(f)) == ref.layout(f.size, surfaces_of(f)), f.id
    f13 = next(f for f in racks(wh) if f.id == "F13")
    assert rack_stock.layout(f13.size, surfaces_of(f13)) == (4, 3, {(0, 0), (0, 2), (1, 1), (2, 0), (2, 2), (3, 1)})
    assert (rack_stock.BAY_M, rack_stock.LEVEL_M, rack_stock.SNAP_M) == (ref.BAY_M, ref.LEVEL_M, ref.SNAP_M)
    for bad in ([(0.0, -3.0, 1.08, 2.6, 1.3)], [(0.0, -4.05, 1.08, 2.6, 1.0)], [(0.0, -4.05, 1.08, 2.6, 3.9)],
                [(0.0, 6.0, 1.08, 2.6, 0.0)]):
        with pytest.raises(ValueError, match="not a bay-level"):
            ref.layout(f13.size, bad)
        with pytest.raises(ValueError, match="not a bay-level"):
            rack_stock.layout(f13.size, bad)


def test_the_floor_stock_stands_in_the_bays_the_renderer_draws_stock_in_at_level_0():
    """Contract: `floor_stock` has two places in each bay at level 0 that the renderer's `stock` lists, and none in a
    bay that has a floor surface or only stock above the floor, for every rack row of `warehouse_a`. Each place is read
    back into its rack's frame (`to_local`) and sorted into its bay; F13's level-0 stock is bays 1 and 3 (its surfaces
    take bays 0 and 2 at the floor), so 2 bays x 2 pallets x 6 racks = 24 places. Regression caught: a stock bay with no
    place (a reader naming it is charged), a place in a free bay or at the wrong level (an item standing on a surface
    next to it swallowed for nothing), or a rack's yaw ignored."""
    ref, wh = render_layout(), load_map("warehouse_a")
    found: dict[tuple[str, int], int] = {}
    places = rack_stock.floor_stock(wh)
    for xy, _ in places:
        for f in racks(wh):
            lx, ly = to_local(Rect(f.center[0], f.center[1], f.size[0], f.size[1], f.yaw), *xy)
            if abs(lx) < 1e-3 and abs(ly) < f.size[1] / 2:
                bay = int((ly + f.size[1] / 2) // 2.7)
                found[(f.id, bay)] = found.get((f.id, bay), 0) + 1
    expected = {(f.id, bay): 2 for f in racks(wh) for bay, level in ref.stock(f.size, surfaces_of(f)) if level == 0}
    assert found == expected and len(places) == sum(expected.values()) == 24
    assert {r for _, r in places} == {rack_stock.STOCK_RADIUS}


def test_the_floor_stock_by_hand():
    """Contract: a stock pallet's place is its bay's middle, a quarter bay (0.675 m) either side along the rack, in the
    map frame to 4 decimals, with the radius 0.5 m plus the widening of a 1.0 x 1.16 m pallet (0.33 m). Hand-worked on
    the warehouse: F13 (centre (0.65, 14.5), yaw 0, bays 1 and 3 at the floor, middles at local y -1.35 and 4.05) has
    pallets at local y -2.025, -0.675, 3.375 and 4.725, so (0.65, 12.475), (0.65, 13.825), (0.65, 17.875) and (0.65,
    19.225); F14 (centre (5.35, 14.5), yaw pi) is the same row turned round: (5.35, 16.525), (5.35, 15.175), (5.35,
    11.125) and (5.35, 9.775). On an invented rack at (10, 10) turned a quarter circle (yaw pi/2, its y axis pointing
    -x) with a surface at the floor of bay 1 and none of bay 2, 3 or 0: pallets at local y -4.725, -3.375, 0.675, 2.025,
    3.375 and 4.725 (bays 0, 2 and 3), so x = 10 - y: 14.725, 13.375, 9.325, 7.975, 6.625 and 5.275 on y 10. Regression
    caught: a rack's frame mirrored or turned the wrong way (a don't-care place on the far side of the aisle), a slot
    offset other than a quarter bay, or a `-0.0` (float noise at a rack centred on an axis) reaching a gold."""
    got = rack_stock.floor_stock(load_map("warehouse_a"))
    near = lambda x: [p for p, _ in got if abs(p[0] - x) < 1e-6]  # noqa: E731
    assert sorted(near(0.65)) == [(0.65, 12.475), (0.65, 13.825), (0.65, 17.875), (0.65, 19.225)]
    assert sorted(near(5.35)) == [(5.35, 9.775), (5.35, 11.125), (5.35, 15.175), (5.35, 16.525)]
    assert {r for _, r in got} == {0.83}
    wh = load_map("warehouse_a")
    turned = replace(wh, furniture=(rack(math.pi / 2, free=[(1, 0)]),))
    places = rack_stock.floor_stock(turned)
    assert [p for p, _ in places] == [(5.275, 10.0), (6.625, 10.0), (7.975, 10.0), (9.325, 10.0), (13.375, 10.0),
                                     (14.725, 10.0)]
    origin = replace(wh, furniture=(rack(math.pi / 2, center=(0.0, 0.0), free=[(b, 0) for b in (1, 2, 3)]),))
    zero = rack_stock.floor_stock(origin)  # bay 0 only: x = -ly, and y = 0 up to float noise (-2.9e-16 at ly -4.725)
    assert [p for p, _ in zero] == [(3.375, 0.0), (4.725, 0.0)] and "-0.0" not in str(zero)


def test_a_map_without_racks_has_no_floor_stock():
    """Contract: only a rack row has stock, so every other bundled map answers an empty list and a rack whose floor is
    all surfaces answers none. Regression caught: stock invented on the apartment or the downtown city (a don't-care
    place in a question that has no such scenery)."""
    for name in ("apartment_a", "downtown_a"):
        assert rack_stock.floor_stock(load_map(name)) == []
    full = replace(load_map("warehouse_a"), furniture=(rack(0.0, free=[(b, 0) for b in range(4)]),))
    assert rack_stock.floor_stock(full) == []
