"""The downtown's street paint (`render/street_marks.py`, plain Python: no Blender needed): where the centre lines, lane
dashes, crosswalks and stop bars go, worked out by hand from `downtown_a.json` (a high
street 100 m long whose kerbs are at y = 4 and y = 23, 2.6 m parking lanes along both, and two side streets 12 m between
their kerbs at x = 19 to 31 and 69 to 81 that join its north side; a square, 30 m wide, behind the south sidewalk).
Where paint may be at all (flush, off every receptacle surface) is `test_dressing.py`'s; here is only what it is."""

from __future__ import annotations

import pytest

from dressing_kit import raw_map, render_module
from stbench.geometry import corners
from stbench.mapspec import load_map, surface_world_rect

S = render_module("street_marks")


@pytest.fixture(scope="module")
def spec() -> dict:
    return raw_map("downtown_a")


def rounded(rect) -> tuple[float, ...]:
    return tuple(round(v, 3) for v in rect)


def test_a_street_is_a_region_with_a_sidewalk_along_each_long_side_and_its_carriageway_lies_between_the_kerbs(spec):
    """Contract (the module's: a street is read from the map's own pieces, never from a name): the high street, which is
    100 x 27 m with sidewalks 4 m deep along y = 0 to 4 and y = 23 to 27, is a street along x with its carriageway
    between the kerbs at y = 4 and 23 (19 m); the two side streets, 20 m wide with 4 m sidewalks, are streets along y
    with carriageways x 19 to 31 and 69 to 81; each has a 2.6 m parking lane on each side. The square (30 x 24 m with no
    sidewalk along its sides) is no street. Travel lanes are 3.45 m: (19 - 5.2) / 3.45 = 4 on the high street and (12 -
    5.2) / 3.45 = 2 on a side street. Regression caught: the square taken for a street (a centre line across its
    paving), a carriageway measured to the building instead of the kerb, or the parking lanes left out of the lanes."""
    got = [(s.region, s.axis, s.lo, s.hi, s.c_lo, s.c_hi, s.park, s.lanes) for s in S.streets(spec)]
    assert got == [("R1", "x", 0.0, 100.0, 4.0, 23.0, 2.6, 4), ("R2", "y", 27.0, 72.0, 19.0, 31.0, 2.6, 2),
                   ("R3", "y", 27.0, 72.0, 69.0, 81.0, 2.6, 2)]


def test_the_paint_of_the_downtown_is_where_the_map_puts_its_streets_crossings_and_junctions(spec):
    """Contract (the module's rules, applied to this map by hand): across each side street's mouth, on the sidewalk's
    line (y 23.5 to 26.5, 0.5 m in from the kerb and from the region edge at y = 27, x 0.5 m in from the sidewalk ends
    at 19 and 31, or 69 and 81) a zebra crossing, and a stop bar 0.7 to 1.1 m past the region edge (y 27.7 to 28.1)
    across the half of the side street that its outgoing traffic uses when it keeps right: heading south, the west half,
    x 21.6 (kerb 19 plus the parking lane) to 24.8 (the centre line 25 less 0.2), and on R3 71.6 to 74.8. Beside the
    square's mouth (the shared edge at y = 0 runs x 35 to 65, so 3.5 m in from each end the crossing is 3 m deep: x 37
    to 40 and 60 to 63) a zebra crossing across the carriageway from kerb to kerb (y 4.3 to 22.7), with a stop bar
    either way 1.2 m before it: eastbound traffic on its right, the south half (y 6.6 to 13.3) at x 35.4 to 35.8 and
    58.4 to 58.8, westbound on the north half (y 13.7 to 20.4) at x 41.2 to 41.6 and 64.2 to 64.6. Then the double
    yellow centre line on y = 13.5 (13.3 to 13.7) of the high street, broken 3.2 m either side of each crossing, and of
    each side street on x = 25 and 75 from 29.7 (1.6 m beyond the stop bar) to the end at y = 72; and, between the two
    lanes of each way, the white dashes on y = 13.5 -+ 3.45 (the lane width), broken at the crossings and across the
    side streets' mouths (18 to 32 and 68 to 82). Regression caught: a crosswalk off its sidewalk line or across a
    parking lane, a stop bar on the wrong half of the road (traffic stopping on the wrong side), lines through a
    crossing, or dashes through a junction."""
    marks = S.marks(spec)
    crosswalks = sorted(rounded(m.rect) for m in marks if m.kind == "crosswalk")
    assert crosswalks == [(19.5, 30.5, 23.5, 26.5), (37.0, 40.0, 4.3, 22.7), (60.0, 63.0, 4.3, 22.7),
                          (69.5, 80.5, 23.5, 26.5)]
    bars = sorted(rounded(m.rect) for m in marks if m.kind == "stop_bar")
    assert bars == [(21.6, 24.8, 27.7, 28.1), (35.4, 35.8, 6.6, 13.3), (41.2, 41.6, 13.7, 20.4),
                    (58.4, 58.8, 6.6, 13.3), (64.2, 64.6, 13.7, 20.4), (71.6, 74.8, 27.7, 28.1)]
    centre = sorted(rounded(m.rect) for m in marks if m.kind == "lane_marking" and m.style == "double")
    assert centre == [(0.0, 33.8, 13.3, 13.7), (24.8, 25.2, 29.7, 72.0), (43.2, 56.8, 13.3, 13.7),
                      (66.2, 100.0, 13.3, 13.7), (74.8, 75.2, 29.7, 72.0)]
    dashes = sorted(rounded(m.rect) for m in marks if m.kind == "lane_marking" and m.style == "dash")
    assert dashes == [(0.0, 18.0, 9.975, 10.125), (0.0, 18.0, 16.875, 17.025), (43.2, 56.8, 9.975, 10.125),
                      (43.2, 56.8, 16.875, 17.025), (82.0, 100.0, 9.975, 10.125), (82.0, 100.0, 16.875, 17.025)]


def aabb(rect) -> tuple[float, float, float, float]:
    """The axis-aligned box of a rectangle of the map (every piece is turned a quarter turn)."""
    pts = corners(rect)
    return min(p[0] for p in pts), max(p[0] for p in pts), min(p[1] for p in pts), max(p[1] for p in pts)


def test_no_mark_lies_on_a_sidewalk_on_a_receptacle_surface_or_outside_a_region(spec):
    """Contract (paint is on the carriageway and off everything a question can put a thing on): every mark lies wholly
    inside one region of the map (the crossing over a side street's mouth lies in the high street's region, y 23.5 to
    26.5, and its stop bar in the side street's), overlaps no sidewalk or paving footprint and no receptacle surface
    (the parking bays, loading zones, sidewalk strips and the square's pads, from the map's furniture and
    `surface_world_rect`). Regression caught: a crosswalk painted over a parking bay (a car parked on it), a stop bar on
    a sidewalk, or paint that runs out of the map."""
    m = load_map("downtown_a")
    surfaces = [aabb(surface_world_rect(f.center, f.yaw, s)) for f in m.furniture for s in f.surfaces]
    ground = [aabb(f.footprint()) for f in m.furniture if f.kind in ("sidewalk", "plaza")]
    regions = [(min(p[0] for p in r.polygon), max(p[0] for p in r.polygon), min(p[1] for p in r.polygon),
                max(p[1] for p in r.polygon)) for r in m.regions]

    def meets(a, b) -> bool:
        return a[0] < b[1] - 1e-9 and b[0] < a[1] - 1e-9 and a[2] < b[3] - 1e-9 and b[2] < a[3] - 1e-9

    marks = S.marks(spec)
    assert len(marks) == 21  # 4 crossings, 6 stop bars, 5 centre lines and 6 runs of dashes (the test above)
    for mark in marks:
        a = mark.rect
        assert any(r[0] <= a[0] and a[1] <= r[1] and r[2] <= a[2] and a[3] <= r[3] for r in regions), (mark.kind, a)
        assert not any(meets(a, b) for b in ground), (mark.kind, "on a sidewalk or the paving", a)
        assert not any(meets(a, b) for b in surfaces), (mark.kind, "on a surface", a)


def test_paint_is_a_pattern_of_rectangles_dashes_zebra_stripes_and_double_lines(spec):
    """Contract (`paint_rects`, what the dressing builder and the scenery draw): a dash mark 18 m long starts its first
    3 m dash 1 m in and repeats every 12 m, so its dashes are x 1 to 4 and 13 to 16 (the third would start at 25, past
    the end); a zebra crossing 11 m wide across its stripes (the side street's 19.5 to 30.5) is eleven stripes 0.5 m
    wide a metre apart, centred (the first x 19.75 to 20.25, the last 29.75 to 30.25), each running the 3 m depth (y
    23.5 to 26.5); a double line is two yellow lines 0.14 m wide with 0.12 m between, round the middle of its rectangle
    (13.5): y 13.30 to 13.44 and 13.56 to 13.70; a bar is its rectangle. Regression caught: dashes that start on the
    seam or ignore the phase, stripes of the wrong pitch or off-centre, or a double line wider than its mark."""
    marks = S.marks(spec)
    dash = next(m for m in marks if m.style == "dash" and m.rect[0] == 0.0 and m.rect[2] < 10.0)
    assert [(round(a, 3), round(b, 3)) for a, b, _, _ in S.paint_rects(dash)] == [(1.0, 4.0), (13.0, 16.0)]
    assert [(round(a, 3), round(b, 3)) for a, b, _, _ in S.paint_rects(dash, phase=2.0)] == [(3.0, 6.0), (15.0, 18.0)]
    zebra = next(m for m in marks if m.style == "zebra" and m.rect[0] == 19.5)
    stripes = S.paint_rects(zebra)
    assert len(stripes) == 11 and rounded(stripes[0]) == (19.75, 20.25, 23.5, 26.5)
    assert rounded(stripes[-1]) == (29.75, 30.25, 23.5, 26.5)
    assert all(round(b[0] - a[0], 3) == 1.0 for a, b in zip(stripes, stripes[1:], strict=False))
    double = next(m for m in marks if m.style == "double" and m.rect[0] == 0.0)
    assert [rounded(r) for r in S.paint_rects(double)] == [(0.0, 33.8, 13.3, 13.44), (0.0, 33.8, 13.56, 13.7)]
    bar = next(m for m in marks if m.kind == "stop_bar")
    assert S.paint_rects(bar) == [bar.rect]


def test_a_street_end_carries_its_centre_line_and_dashes_on(spec):
    """Contract (`extension`, `scenery_layout`'s continued streets): past the high street's east end (x = 100) the
    centre line carries on over `lo..hi` as one double line (13.3 to 13.7) and both lane dashes (y 9.975 to 10.125 and
    16.875 to 17.025) over the same range; a two-lane side street has no dashes. Regression caught: a street that loses
    its lines at the edge of the map, or a side street given dashes it has no lanes for."""
    streets = {s.region: s for s in S.streets(spec)}
    east = S.extension(streets["R1"], "hi", 100.0, 350.0)
    assert [(m.style, rounded(m.rect)) for m in east] == [("double", (100.0, 350.0, 13.3, 13.7)),
                                                         ("dash", (100.0, 350.0, 9.975, 10.125)),
                                                         ("dash", (100.0, 350.0, 16.875, 17.025))]
    north = S.extension(streets["R2"], "hi", 72.0, 322.0)
    assert [(m.style, rounded(m.rect)) for m in north] == [("double", (24.8, 25.2, 72.0, 322.0))]
