"""The warehouse's render side reads a region by its kind, never by its name, and keeps the places the AMR uses clear
(M3-RW): its dressing and its floor markings. Free: `render/` is not a package, so its modules load by name through
`dressing_kit`, and the numbers are worked out by hand from `warehouse_a` (three aisles of kind `aisle`, each between
two rack rows of four bays; the dock of kind `dock`; the AMR's charger and four bays)."""

from __future__ import annotations

from collections import Counter

import pytest

from dressing_kit import layout_of, raw_map, render_module
from stbench import dressing_boxes as db
from stbench import dressing_zones as dz

floor_marks = render_module("floor_marks")


def with_regions(spec: dict, changes: dict[str, dict]) -> dict:
    """A copy of the map spec whose regions, by id, take the given fields."""
    return {**spec, "regions": [{**r, **changes.get(r["id"], {})} for r in spec["regions"]]}


def test_the_dressing_reads_kinds_not_names():
    """Contract (a job and the render read the warehouse's places by region kind, so M4's warehouses need no code): the
    dressing and the floor markings of `warehouse_a` are the same boxes and rectangles when the three aisles (R2 to R4,
    kind "aisle") carry other names, and when the dock (R1, whose name is its kind) is called "inbound aisle" with the
    kind "dock" set, a name that holds the word "aisle". A region that is named an aisle but is of another kind is no
    aisle: by hand, each aisle has two rack rows of four bays with a beam label on each of 2 beams a bay (16 labels, one
    sign), and the floor marks give it one pair of 14 cm tyre streaks and one polished band, so without R2 the layout
    has 32 labels and 2 signs and the plan 14 streaks (of 16) and 3 bands (of 4). Regression caught: an M4 warehouse
    whose aisles are named otherwise dressed as a dock, or not at all; a dock named for its traffic dressed as an
    aisle."""
    spec = raw_map("warehouse_a")
    dressing, plan = layout_of("warehouse", spec), floor_marks.layout(spec)
    assert Counter(d.cls for d in dressing)["beam_label"] == 48 and Counter(d.cls for d in dressing)["aisle_sign"] == 3
    renamed = with_regions(spec, {"R2": {"name": "pick lane one"}, "R3": {"name": "pick lane two"},
                                  "R4": {"name": "pick lane three"}, "R1": {"name": "inbound aisle", "kind": "dock"}})
    assert layout_of("warehouse", renamed) == dressing and floor_marks.layout(renamed) == plan
    no_aisle = with_regions(spec, {"R2": {"kind": "bulk"}})  # still named "west aisle"
    fewer = Counter(d.cls for d in layout_of("warehouse", no_aisle))
    assert fewer["beam_label"] == 32 and fewer["aisle_sign"] == 2
    shorter = floor_marks.layout(no_aisle)
    assert (len(shorter.tracks), len(shorter.lanes)) == (14, 3) and (len(plan.tracks), len(plan.lanes)) == (16, 4)


# The keep-outs of `warehouse_a` by hand: the bays F20 to F23 (1.6 m deep from a station's face, 2.2 m along it) and the
# apron of the charger F19, which stands on the east wall at x 17.38 to 17.5 and y 1.1 to 1.7 facing west: 1.6 m out of
# its face (x 15.78 to 17.38) and 0.5 m beyond each side (y 0.6 to 2.2). The bays F20 (15.9, 6.5) and F22 (1.5, 6.4) are
# y 5.4 to 7.6 and 5.3 to 7.5. Every one reaches 3.5 m up (`zones.TOP_M`).
KEEP_OUTS = {
    "keep-out: bay F20 (up to 3.5 m)": (15.1, 5.4, 0.0, 16.7, 7.6, 3.5),
    "keep-out: bay F21 (up to 3.5 m)": (15.1, 3.0, 0.0, 16.7, 5.2, 3.5),
    "keep-out: bay F22 (up to 3.5 m)": (0.7, 5.3, 0.0, 2.3, 7.5, 3.5),
    "keep-out: bay F23 (up to 3.5 m)": (0.7, 3.1, 0.0, 2.3, 5.3, 3.5),
    "keep-out: apron of charger F19 (up to 3.5 m)": (15.78, 0.6, 0.0, 17.38, 2.2, 3.5),
}


def test_no_dressing_in_a_bay_or_on_the_charger_apron():
    """Contract (rule 5: a zone the render-only dressing always blocks has a gold the layout gives, which a layout-
    learning agent answers from; the AMR's undock must find its apron clear): the guard keeps every fixed `bay`'s
    footprint and every `charger`'s apron (1.6 m out of its face, 0.5 m beyond each side) free of dressing from the
    floor to 3.5 m, found by piece kind, and `warehouse_a`'s layout puts nothing there. By hand the five volumes are
    above; a 0.2 m box on the floor in bay F20 or on the apron is refused for that reason, one 2.5 to 2.9 m up in a
    bay too, and one 3.1 to 3.4 m up, one above the 3.5 m is not, and a fitting hung on the east wall's plane over the
    charger (x 17.38 to 17.5) stands clear of the apron it touches. Regression caught: a bay blocked in every case by
    render-only stock or a cable tray's drop, a piece on the apron the AMR backs over, or a keep-out read off a region's
    or a piece's name."""
    spec = raw_map("warehouse_a")
    guard = dz.DressingGuard(spec, "warehouse")
    keep = {why: box for why, box in guard.forbidden if why.startswith("keep-out")}
    assert keep.keys() == KEEP_OUTS.keys()
    for why, box in KEEP_OUTS.items():
        assert keep[why] == pytest.approx(box), why
    layout = layout_of("warehouse", spec)
    assert [d.cls for d in layout for box in KEEP_OUTS.values() if db.overlaps(d.box, box)] == []

    def keep_outs_met(box: db.Box) -> list[str]:
        return [p.removeprefix("meets ") for p in guard.problems(box) if p.startswith("meets keep-out")]

    assert keep_outs_met((15.8, 6.4, 0.0, 16.0, 6.6, 0.2)) == ["keep-out: bay F20 (up to 3.5 m)"]
    assert keep_outs_met((15.8, 6.4, 2.5, 16.0, 6.6, 2.9)) == ["keep-out: bay F20 (up to 3.5 m)"]
    assert keep_outs_met((15.8, 6.4, 3.1, 16.0, 6.6, 3.4)) == ["keep-out: bay F20 (up to 3.5 m)"]
    assert keep_outs_met((15.8, 6.4, 3.6, 16.0, 6.6, 3.8)) == []
    assert keep_outs_met((16.4, 1.3, 0.0, 16.6, 1.5, 0.2)) == ["keep-out: apron of charger F19 (up to 3.5 m)"]
    assert keep_outs_met((17.38, 1.9, 1.0, 17.5, 2.1, 1.45)) == []
    named = {**spec, "furniture": [{**f, "name": "painted square"} if f["kind"] in ("bay", "charger") else f
                                   for f in spec["furniture"]]}
    assert {why: box for why, box in dz.forbidden(named) if why.startswith("keep-out")} == keep
