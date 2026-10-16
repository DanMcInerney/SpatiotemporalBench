"""The warehouse's floor markings (`render/floor_marks.py`): where the yellow lines, the
hatched aprons and the tyre marks of the slab's material layer lie. The numbers are worked out by hand from
`warehouse_a` (17.4 x 19.8 m inside; four dock doors 2.4 m wide at x = 3.7, 7.1, 10.5 and 13.9 on the south wall, each
with a 1.4 x 2.8 m staging lane in front of it from y = 0.6 to 3.4; three aisles 3.6 m wide between racks 1.1 m deep
whose fronts are at x = 1.2 and 4.8, 7.0 and 10.6, 12.8 and 16.4, running from y = 9.1 to the north wall at 19.9). The
plan needs no Blender: `render/` is not a package, so it is put on the path."""

from __future__ import annotations

import json
import sys
from importlib import resources
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "render"))  # its modules import each other by bare name

import floor_marks as F  # noqa: E402

RACK_FRONTS = (1.2, 4.8, 7.0, 10.6, 12.8, 16.4)
DOORS = (3.7, 7.1, 10.5, 13.9)
FLOOR = (0.1, 0.1, 17.5, 19.9)  # the inside of the walls


def load(name: str) -> dict:
    return json.loads(resources.files("stbench.maps").joinpath(f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def spec() -> dict:
    return load("warehouse_a")


@pytest.fixture(scope="module")
def plan(spec):
    return F.layout(spec)


def overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def racks(spec) -> list[tuple]:
    """The footprint (x0, y0, x1, y1) of each rack row: 1.1 m deep across x, 10.8 m long along y."""
    return [(f["center"][0] - 0.55, f["center"][1] - 5.4, f["center"][0] + 0.55, f["center"][1] + 5.4)
            for f in spec["furniture"] if f["kind"] == "pallet_rack"]


def test_a_yellow_line_runs_along_every_rack_front_30_cm_out_of_the_face(plan):
    """Contract (plan-realism section 4, the warehouse floor): each of the six rack rows has one 10 cm line parallel to
    it whose centre is 30 cm out from the rack's face (1.2 m + 0.3 m = 1.5 m for the first row, 4.8 m - 0.3 m = 4.5 m
    for the one facing it) and as long as the row (y = 9.1 to 19.9). Regression caught: a line on the wrong side of a
    rack (a back-to-back pair has fronts that face away from each other), one under the rack or in the bay, a missing
    row, or a line that stops short of the row's end."""
    rack_lines = [r for r in plan.lines if r[1] == pytest.approx(9.1) and r[3] == pytest.approx(19.9)]
    centres = sorted((r[0] + r[2]) / 2 for r in rack_lines)
    expected = sorted(x + 0.3 if i % 2 == 0 else x - 0.3 for i, x in enumerate(RACK_FRONTS))
    assert centres == pytest.approx(expected)
    assert expected == pytest.approx([1.5, 4.5, 7.3, 10.3, 13.1, 16.1])
    assert all(r[2] - r[0] == pytest.approx(0.10) for r in rack_lines)


def test_a_walkway_of_two_lines_1_2_m_apart_lies_across_the_aisle_mouths_in_the_dock(plan):
    """Contract (plan-realism section 4): two 10 cm lines 1.2 m apart run along x between the dock's end walls (inset
    0.6 m), the nearer to the aisles 0.2 m short of the racks' south end (y = 9.1): their centres are y = 7.7 and 8.9.
    Regression caught: a walkway drawn over the aisles or racks, one that is not 1.2 m wide, or one that runs into the
    walls."""
    walk = sorted((r for r in plan.lines if r[2] - r[0] > 10.0), key=lambda r: r[1])
    assert [round((r[1] + r[3]) / 2, 3) for r in walk] == [7.7, 8.9]
    assert all(r[0] == pytest.approx(0.7) and r[2] == pytest.approx(16.9) for r in walk)
    assert all(r[3] - r[1] == pytest.approx(0.10) for r in walk)


def lanes(spec) -> list[tuple]:
    """The footprint (x0, y0, x1, y1) of each staging lane: 1.4 m across x and 2.8 m along y (a quarter turn)."""
    return [(f["center"][0] - 0.7, f["center"][1] - 1.4, f["center"][0] + 0.7, f["center"][1] + 1.4)
            for f in spec["furniture"] if f["kind"] == "staging_lane"]


def test_each_dock_door_has_an_apron_as_wide_as_the_door_past_the_end_of_its_staging_lane(plan, spec):
    """Contract (plan-realism section 4, hatching at the doors): four hatched aprons, each as wide as its 2.4 m door and
    centred on it, 0.7 m deep and starting 0.2 m beyond the end of the 2.8 m staging lane (y = 3.4), i.e. y = 3.6 to
    4.3: clear of the lane, so the yellow lane outline (furniture) and the hatching never cover one another, even where
    the lane stands off its door's axis (the lanes of doors 1 and 4 are at x 4.5 and 12.9, M3-maps). Regression caught:
    an apron over a staging lane, one off its door, or the apron of a door left out."""
    assert len(plan.hatches) == 4 and len(lanes(spec)) == 4
    for hatch, door in zip(sorted(plan.hatches), DOORS, strict=True):
        assert (hatch[0], hatch[2]) == pytest.approx((door - 1.2, door + 1.2))
        assert (hatch[1], hatch[3]) == pytest.approx((3.6, 4.3))
        assert not any(overlap(hatch, lane) for lane in lanes(spec)), door


def moved_lanes(spec, moves: dict[str, tuple[float, float]]) -> dict:
    """A copy of the spec with the staging lanes of the given ids at new centres."""
    furniture = [{**f, "center": list(moves[f["id"]])} if f["id"] in moves else f for f in spec["furniture"]]
    return {**spec, "furniture": furniture}


def test_a_doors_apron_follows_its_lane(spec):
    """Contract (plan-realism section 4: the apron is past the end of the door's staging lane, so the hatching and the
    lane's outline never cover one another): the lane that is a door's is the one whose span along the wall overlaps the
    door's, wherever it stands in front of it, and the apron lies `HATCH_GAP` (0.2 m) beyond its far end, as wide as the
    door and 0.7 m deep. By hand, each lane 1.4 m across x and 2.8 m long along y, each door 2.4 m wide: lane F06 moved
    to (8.0, 2.4) spans x 7.3 to 8.7 against door 2's 5.9 to 8.3 and ends at y 3.8, so door 2's apron is y 4.0 to 4.7
    over x 5.9 to 8.3; lane F07 moved to x = 12.45 spans 11.75 to 13.15, 0.05 m clear of door 3's 9.3 to 11.7 (and over
    door 4's 12.7 to 15.1, where F08 ends at the same 3.4), so door 3 has no lane, the 0.6 m of bare floor, and its
    apron is y 0.95 to 1.65; the other two doors keep y 3.6 to 4.3. No apron lies on a lane, in the map or moved.
    Regression caught: the apron painted across a lane that stands off its door's axis (the lane found only on the
    axis, as M3-maps' moved lanes F05 and F08 were missed), or taken from a neighbour's lane that is not the door's."""
    moved = moved_lanes(spec, {"F06": (8.0, 2.4), "F07": (12.45, 2.0)})
    hatches = sorted(F.layout(moved).hatches)
    assert [(round(h[1], 3), round(h[3], 3)) for h in hatches] == [(3.6, 4.3), (4.0, 4.7), (0.95, 1.65), (3.6, 4.3)]
    for hatch, door in zip(hatches, DOORS, strict=True):
        assert (hatch[0], hatch[2]) == pytest.approx((door - 1.2, door + 1.2))
    for s in (spec, moved):
        assert not any(overlap(h, lane) for h in F.layout(s).hatches for lane in lanes(s))


def test_nothing_is_painted_or_worn_under_a_rack_or_outside_the_walls(plan, spec):
    """Contract (plan-realism section 3, the dressing's validity): every line, apron, tyre track and polished band is on
    open floor: inside the walls and clear of every rack's footprint (a mark under a rack is invisible at best and, at
    worst, a line a pallet stands on). Regression caught: a track or band that runs into a rack row, or past a wall."""
    footprints = racks(spec)
    assert len(footprints) == 6
    for name in ("lines", "hatches", "tracks", "lanes"):
        for r in getattr(plan, name):
            assert FLOOR[0] - 1e-6 <= r[0] < r[2] <= FLOOR[2] + 1e-6, (name, r)
            assert FLOOR[1] - 1e-6 <= r[1] < r[3] <= FLOOR[3] + 1e-6, (name, r)
            assert not any(overlap(r, f) for f in footprints), (name, r)


def test_tyre_marks_come_in_pairs_a_metre_apart_down_each_aisle_and_from_each_door(plan):
    """Contract (plan-realism section 4): a pair of 14 cm streaks 1.1 m apart (a fork truck's wheels) runs along the
    centre of each of the three aisles (x = 3.0, 8.8 and 14.6, so streaks at 2.45 and 3.55 and so on) and in from each
    of the four doors, and one pair runs across the dock; 3 + 4 aisle and door pairs and the cross pair are 16 streaks.
    Regression caught: a single streak (a skid, not traffic), streaks the width of a lane, or pairs the wrong gap."""
    assert len(plan.tracks) == 16
    along_y = sorted(r for r in plan.tracks if r[3] - r[1] > r[2] - r[0])
    centres = [round((r[0] + r[2]) / 2, 3) for r in along_y]
    for c in (3.0, 8.8, 14.6):
        assert any(abs(x - (c - 0.55)) < 1e-6 for x in centres) and any(abs(x - (c + 0.55)) < 1e-6 for x in centres)
    for r in plan.tracks:
        assert min(r[2] - r[0], r[3] - r[1]) == pytest.approx(0.14)


def test_a_map_of_another_catalog_has_no_markings():
    """Contract: only a warehouse has painted floors; the apartment and the downtown keep the floors they have (the
    downtown's road paint is its own). Regression caught: the apartment's floor given rack lines, or a layout that
    raises on a map with no racks."""
    assert F.layout(load("apartment_a")) is None
    assert F.layout(load("downtown_a")) is None


def test_the_plan_depends_on_the_map_alone(spec):
    """Contract (plan-realism section 3): a map's markings are the same every time it is read, so every episode of a
    map has the same floor. Regression caught: a layout drawn from unseeded randomness or from the spec's list order."""
    reversed_spec = {**spec, "furniture": list(reversed(spec["furniture"]))}
    assert F.layout(spec) == F.layout(spec)
    assert sorted(F.layout(reversed_spec).lines) == sorted(F.layout(spec).lines)
    assert sorted(F.layout(reversed_spec).hatches) == sorted(F.layout(spec).hatches)
