"""The zone rule's two new sources (`zones.DOORWAY`, `zones.SPOT`) and its band's top (`ZoneRule.top_m`), on
`apartment_a` and `shape_kit`'s synthetic map: which zones `zones_of` lists, what `zone_of` returns, and that the
whole-map list holds every cell of every window."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from shape_kit import APARTMENT, NORTH, SYNTHETIC
from stbench import zones
from stbench.zones import DOORWAY, SPOT, TOP_M, ZoneRule

WAY = ZoneRule("the doorway ahead of you", DOORWAY, height_m=0.02)
SPOT_RULE = ZoneRule("the spot in front of you where you are putting it down", SPOT, height_m=0.01, top_m=0.5)


def zone_box(zone: zones.Zone) -> tuple[float, float, float, float]:
    """The x and y extents (x0, x1, y0, y1) of the zone's polygon, to 6 decimals."""
    xs, ys = zip(*zone.polygon, strict=True)
    return round(min(xs), 6), round(max(xs), 6), round(min(ys), 6), round(max(ys), 6)


def test_the_band_top_is_the_rules_and_defaults_to_3_5_m():
    """Contract: a `ZoneRule` has `top_m`, the band's top above the zone's ground, 3.5 m (`TOP_M`) unless the rule says
    otherwise, and a zone's band is `(height_m, top_m)` whatever the source: a piece kind with `top_m` 1.0 has the band
    (0.3, 1.0); a spot rule asking 0.01 to 0.5 m keys `band_cells` by its own band, beside a default rule's 0.3 to
    3.5 m. Regression caught: a band top still read from the module constant, which would judge a thing put down on a
    table by a 3.5 m column (the ceiling lamp in the zone)."""
    assert ZoneRule("a", "bay").top_m == TOP_M == 3.5 and ZoneRule("a", "bay") == ZoneRule("a", "bay", top_m=TOP_M)
    assert {ZoneRule("a", "bay", top_m=0.5): 1}[ZoneRule("a", "bay", top_m=0.5)] == 1  # still hashable
    (kind,) = zones.zones_of(SYNTHETIC, ZoneRule("the table", "table", top_m=1.0))
    assert kind.band == (0.3, 1.0)
    keep = zones.band_cells(SYNTHETIC, [SPOT_RULE, replace(SPOT_RULE, height_m=0.3, top_m=TOP_M)])
    assert {band for _, band in keep} == {(0.01, 0.5), (0.3, 3.5)}


def test_the_doorway_zones_are_the_interior_ways():
    """Contract: `zones_of` for `DOORWAY` lists the way of every interior opening in map order, named by the opening's
    id, on the floor, with the rule's band: `apartment_a`'s D1, D2, D3, D4 and the arch O1, not the entrance D0. A way's
    cells are the fine cells (0.1 m) whose centres lie in it, a centre on an edge in: by hand D1's way (x 4.6..4.95,
    y 1.575..2.225) holds centres x 4.65..4.95 and y 1.65..2.15, 4 x 6 cells. A margin grows a way on every side (0.1
    m: D1 x 4.5..5.05), and one that leaves it no area is refused. Regression caught: the entrance listed (a robot never
    crosses it), a way named by an entity or its wall, or a margin taken off instead of added."""
    ways = zones.zones_of(APARTMENT, WAY)
    assert [z.piece for z in ways] == ["D1", "D2", "D3", "D4", "O1"]
    assert all((z.base_z, z.band) == (0.0, (0.02, 3.5)) for z in ways)
    assert zone_box(ways[0]) == (4.6, 4.95, 1.575, 2.225)
    assert zones.zone_cells(ways[0]) == [(u, v) for u in range(46, 50) for v in range(16, 22)]
    assert zone_box(zones.zones_of(APARTMENT, replace(WAY, margin_m=0.1))[0]) == (4.5, 5.05, 1.475, 2.325)
    with pytest.raises(ValueError, match="D1"):
        zones.zones_of(APARTMENT, replace(WAY, margin_m=-0.5))


def test_the_spot_zones_are_the_inset_surfaces():
    """Contract: `zones_of` for `SPOT` lists the whole inset largest surface of every blocking fixed piece that has a
    surface, in map order, named by the piece's id, at that surface's height with the rule's band. `apartment_a` has 17
    (seven blocking pieces have none: fridge, stove, washer-dryer, dock, wardrobe, toilet, bath), the dining table F07
    among them (x 1.46..2.74, y 1.56..2.24 at 0.75 m, by hand from its 0.74 x 1.34 m surface turned north, less 0.03 m
    a side), the bookshelf F13 at the first of its three equal shelves, 0.4 m. On the synthetic map the shelf is at 0.4
    m and the bench at its larger surface, 0.55 m, the chair (it moves), the rug and the fridge are out. Regression
    caught: a list of windows (it would hold one cell block per piece), a surface of another height, or a piece with
    no surface listed."""
    spots = zones.zones_of(APARTMENT, SPOT_RULE)
    assert [z.piece for z in spots] == [
        "F02", "F03", "F04", "F06", "F07", "F10", "F11", "F12", "F13", "F20", "F21", "F30", "F31", "F32", "F33", "F40",
        "F43"]
    by_id = {z.piece: z for z in spots}
    assert zone_box(by_id["F07"]) == (1.46, 2.74, 1.56, 2.24) and by_id["F07"].base_z == 0.75
    assert (by_id["F13"].base_z, by_id["F43"].base_z, by_id["F10"].base_z) == (0.4, 0.5, 0.45)
    assert all(z.band == (0.01, 0.5) for z in spots)
    here = zones.zones_of(SYNTHETIC, SPOT_RULE)
    assert [(z.piece, z.base_z) for z in here] == [("F01", 0.75), ("F03", 0.4), ("F04", 0.45), ("F07", 0.55)]
    assert zone_box(here[0]) == (1.48, 2.52, 2.78, 3.22)


def test_zone_of_a_spot_is_the_window_the_base_faces():
    """Contract: `zone_of` for `SPOT` takes the base and its heading and returns the window of the surface faced
    (`zone_shapes.surface_window`) as a zone named by the piece, at the surface's height with the rule's band; None
    where nothing is faced; ValueError without a heading; a margin grows the window. By hand from (2.0, 2.0) facing
    north at the synthetic table: x 1.8..2.2, y 2.78..3.22 at 0.75 m, and 1.75..2.25, 2.73..3.27 with a margin of
    0.05 m. Regression caught: a nearest-piece choice (the base's nearest table, not the one it faces), a heading
    ignored, or a `SPOT` rule that returns the whole surface."""
    zone = zones.zone_of(SYNTHETIC, SPOT_RULE, (2.0, 2.0), NORTH)
    assert (zone.piece, zone.base_z, zone.band) == ("F01", 0.75, (0.01, 0.5))
    assert zone_box(zone) == (1.8, 2.2, 2.78, 3.22)
    grown = zones.zone_of(SYNTHETIC, replace(SPOT_RULE, margin_m=0.05), (2.0, 2.0), NORTH)
    assert zone_box(grown) == (1.75, 2.25, 2.73, 3.27)
    assert zones.zone_of(SYNTHETIC, SPOT_RULE, (2.0, 2.0), -NORTH) is None
    with pytest.raises(ValueError, match="yaw"):
        zones.zone_of(SYNTHETIC, SPOT_RULE, (2.0, 2.0))


def test_zone_of_a_doorway_is_the_nearest_way():
    """Contract: `zone_of` for `DOORWAY` is the way whose centre is nearest the point, as for a piece kind, and ignores
    the heading. By hand: the ways' centres are D1 (4.775, 1.9), D2 (4.775, 3.9), D3 (6.025, 1.3), D4 (6.025, 3.3) and
    O1 (3.6, 2.8); (4.2, 1.9) is nearest D1, (5.7, 3.3) D4, (3.6, 2.5) the arch. Regression caught: the nearest
    opening's centre rather than the nearest way's, or a heading required."""
    def near(xy, yaw=None):
        return zones.zone_of(APARTMENT, WAY, xy, yaw).piece

    assert [zones.centroid(z) for z in zones.zones_of(APARTMENT, WAY)] == [
        (4.775, 1.9), (4.775, 3.9), (6.025, 1.3), (6.025, 3.3), (3.6, 2.8)]
    assert (near((4.2, 1.9)), near((5.7, 3.3)), near((3.6, 2.5), 1.0)) == ("D1", "D4", "O1")


def window_zones(m, rule, step: float = 0.5, headings: int = 8):
    """Every zone `zone_of` returns for `rule` from a grid of bases over the map (every `step` m) and `headings`
    headings, with where it came from."""
    x0, y0, x1, y1 = m.bounds
    for i in range(int((x1 - x0) / step) + 1):
        for j in range(int((y1 - y0) / step) + 1):
            for h in range(headings):
                base, yaw = (x0 + i * step, y0 + j * step), 2 * math.pi * h / headings
                if (zone := zones.zone_of(m, rule, base, yaw)) is not None:
                    yield zone, base, yaw


def test_the_zone_list_holds_every_cell_of_every_window():
    """Contract: for every zone `zone_of` returns for a `SPOT` rule, from a half-metre grid of bases over `apartment_a`
    and eight headings from each (with the rule's margin 0 and 0.05 m), its cells are among `band_cells`' cells at its
    (ground, band): `band_cells` lists the whole surfaces, so a window anywhere on one loses no cell; the sweep meets
    windows on at least ten pieces. Regression caught: `zones_of` listing windows, or one surface per piece other than
    the one `zone_of` takes, so a spot question's gold would read cells the stream had dropped."""
    for margin in (0.0, 0.05):
        rule = replace(SPOT_RULE, margin_m=margin)
        keep = zones.band_cells(APARTMENT, [rule])
        seen = set()
        for zone, base, yaw in window_zones(APARTMENT, rule):
            assert set(zones.zone_cells(zone)) <= keep[(zone.base_z, zone.band)], (zone.piece, base, yaw, margin)
            seen.add(zone.piece)
        assert len(seen) >= 10, (margin, sorted(seen))
