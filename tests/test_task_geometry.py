"""The geometry of a `task_things` predicate (`task_predicate`): the anchor from a task's delivered steps, a footprint's
distance and verdict, and what a table lists. Every expected value is worked out by hand on small shapes."""

from __future__ import annotations

import math

import pytest

import family_kit as fk
import task_kit as tk
from stbench.geometry import Rect
from stbench.task_predicate import (
    EDGE,
    IN,
    OUT,
    Anchor,
    TaskThings,
    anchor_of,
    detected_rect,
    distance,
    judge,
    on_floor,
    reaches,
    verdict,
)

LINE = Anchor("path", segments=(((0.0, 1.0), (20.0, 1.0)),))   # the line y = 1, x from 0 to 20


def steps(phases: list[str], xys: list[tuple[float, float]]) -> list[dict]:
    """Delivered steps as the gold and a table both see them: a base position and a phase, nothing of any plan."""
    return [{"k": k, "pose": {"pos": [x, y, 0.0]}, "status": {"phase": p}}
            for k, (p, (x, y)) in enumerate(zip(phases, xys, strict=True))]


def pred(anchor: str, phases: set[str], **kw) -> TaskThings:
    return TaskThings("p", anchor, frozenset(phases), **kw)


def test_a_path_anchor_is_the_matched_steps_up_to_the_last_step_of_until():
    """Contract (plan §4 A4): a `path` anchor is the base positions of the steps whose phase is in `phases`, up to the
    task's last step of `until`, a segment between two matched steps that follow one another and none across a step that
    does not match. By hand on eight steps, phases docked, undocking, transit, transit, at_station, transit, at_station,
    loading at (0,0) (1,0) (2,0) (3,0) (4,0) (4,3) (5,3) (9,9): with {undocking, transit, at_station} until at_station
    the last at_station is step 6, so steps 1-6 match (docked and loading do not) and the chain runs
    (1,0)-(2,0)-(3,0)-(4,0)-(4,3)-(5,3); with phases {transit} alone, steps 2-3 and 5 match: the segment (2,0)-(3,0) and
    the lone point (4,3) with no chord between. A task without a step of `until`, and one with no matched step, has no
    anchor. Regression caught: an anchor read from a plan (the table, fenced from plans, could not rebuild it), a chord
    drawn across a phase that does not match, or steps after `until` counted."""
    phases = ["docked", "undocking", "transit", "transit", "at_station", "transit", "at_station", "loading"]
    xys = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0), (4.0, 3.0), (5.0, 3.0), (9.0, 9.0)]
    s = steps(phases, xys)
    chain = anchor_of(pred("path", {"undocking", "transit", "at_station"}, until="at_station"), s)
    assert chain.segments == (((1.0, 0.0), (2.0, 0.0)), ((2.0, 0.0), (3.0, 0.0)), ((3.0, 0.0), (4.0, 0.0)),
                              ((4.0, 0.0), (4.0, 3.0)), ((4.0, 3.0), (5.0, 3.0)))
    split = anchor_of(pred("path", {"transit"}), s)
    assert split.segments == (((2.0, 0.0), (3.0, 0.0)), ((4.0, 3.0), (4.0, 3.0)))
    assert anchor_of(pred("path", {"transit"}, until="placing"), s) is None
    assert anchor_of(pred("path", {"charging"}), s) is None


def test_a_point_anchor_is_the_median_of_each_axis():
    """Contract (plan §4 A4): a `point` anchor is the median base x and the median base y over the matched steps, each
    axis on its own. By hand: three inspect steps at (0,0), (10,2) and (4,8) give x 0, 4, 10 -> 4 and y 0, 2, 8 -> 2,
    the point (4,2), which is none of the three; four steps at x 0, 2, 6, 100 give 4 (the mean of the middle two), the
    outlier unmoved. Regression caught: the mean (pulled by a far step), one step's position, or the median of
    positions taken as whole points."""
    s = steps(["inspect"] * 3, [(0.0, 0.0), (10.0, 2.0), (4.0, 8.0)])
    assert anchor_of(pred("point", {"inspect"}), s).segments == (((4.0, 2.0), (4.0, 2.0)),)
    s = steps(["inspect"] * 4, [(0.0, 0.0), (2.0, 0.0), (6.0, 0.0), (100.0, 0.0)])
    assert anchor_of(pred("point", {"inspect"}), s).segments == (((4.0, 0.0), (4.0, 0.0)),)


def test_a_rooms_anchor_is_the_regions_of_the_matched_steps_and_the_ways_through_their_openings():
    """Contract (plan §4 A4): a `rooms` anchor is the regions that hold the base of a matched step, from the public map,
    and the doorway way of each opening of those regions (`zone_shapes.doorway_way`). On the hand map, a door at
    (4, 1.5) in the wall x = 4, 0.9 m wide, swinging left: its way is the rectangle centred (4.175, 1.5), 0.7 across
    the wall and 0.35 along x, turned with the wall (worked out in `task_kit.room_map`). Cleaning steps at (6, 1.5)
    and (7, 2) hold the living room only; a step at (20, 20) is in no room and counts for nothing; with no matched step
    in a room there is no anchor, and a map is required. Regression caught: rooms read from the plan, a way left out
    (a thing in a doorway counted as in the room), or a step outside every room making a region of None."""
    m = tk.room_map(tk.script(room="living room"))
    p = pred("rooms", {"cleaning"})
    where = [(6, 1.5), (6, 1.5), (7, 2), (20, 20)]
    anchor = anchor_of(p, steps(["docked", "cleaning", "cleaning", "cleaning"], where), m)
    assert [r.id for r in anchor.rooms] == ["R2"]
    (way,) = anchor.ways
    assert (way.cx, way.cy, way.sx, way.sy, way.yaw) == pytest.approx((4.175, 1.5, 0.7, 0.35, math.pi / 2))
    assert anchor_of(p, steps(["cleaning"], [(20, 20)]), m) is None
    with pytest.raises(ValueError, match="map"):
        anchor_of(p, steps(["cleaning"], [(6, 1.5)]))


def test_the_distance_is_to_the_footprints_nearest_point_not_its_centre():
    """Contract (plan §4 A4): the distance to a path or point is from the footprint's nearest point. Against the line
    y = 1 (x 0 to 20): a 1 x 1 m square centred (5, 2.6) has its nearest edge at y 2.1, 1.1 m from the line, where its
    centre is 1.6 m; turned 45 degrees its nearest corner is half a diagonal (0.7071 m) nearer its centre, at
    2.6 - 0.7071 = 1.8929, 0.8929 m; one that crosses the line is 0; against the point (3, 3) a 2 x 2 square centred
    (6, 3) is 2 m (its edge at x = 5); and against the segment (0,0)-(2,0) a 2 x 2 square centred (5, 1) is 2 m from
    the segment's end (its corner at (4, 0)). Regression caught: the distance from the centre (a large thing
    misjudged by half its size), a corner mistaken for the nearest point, or a segment's end ignored."""
    assert distance(LINE, Rect(5, 2.6, 1, 1, 0)) == pytest.approx(1.1)
    assert distance(LINE, Rect(5, 2.6, 1, 1, math.pi / 4)) == pytest.approx(2.6 - math.sqrt(0.5) - 1)
    assert distance(LINE, Rect(5, 1.3, 1, 1, 0)) == 0.0
    assert distance(Anchor("point", segments=(((3.0, 3.0), (3.0, 3.0)),)), Rect(6, 3, 2, 2, 0)) == pytest.approx(2.0)
    assert distance(Anchor("path", segments=(((0.0, 0.0), (2.0, 0.0)),)), Rect(5, 1, 2, 2, 0)) == pytest.approx(2.0)


def rect_at(d: float) -> Rect:
    """A 0.4 m square whose nearest point is `d` from the line y = 1."""
    return Rect(5.0, 1.0 + 0.2 + d, 0.4, 0.4, 0.0)


@pytest.mark.parametrize("d, expected", [(0.75, IN), (0.76, EDGE), (1.25, EDGE), (1.26, OUT), (0.0, IN), (3.0, OUT)])
def test_the_verdict_has_an_edge_band_around_the_reach(d, expected):
    """Contract (plan §4 A4): within `reach - band` of the anchor a footprint is inside, within `reach + band` it is in
    the band (don't-care), beyond that it is outside; the limits are inside their class. Reach 1.0, band 0.25:
    0.75 m is `IN`, 0.76 and 1.25 are `EDGE`, 1.26 `OUT`. Regression caught: no band (the boundary scored), a band on
    one side only, or an open limit."""
    p = pred("path", {"docking"}, reach_m=1.0, band_m=0.25)
    assert verdict(p, LINE, rect_at(d)) == expected


def test_a_thing_whose_footprints_disagree_is_in_the_band():
    """Contract (plan §4 A4): a detection carries no heading, so a thing is judged by its footprint as placed and as a
    detection gives it, and wherever they disagree it is `EDGE` (don't-care), so the oracle, which sees only the
    detected one, is never charged. A 1.2 x 0.4 m thing turned 90 degrees centred (5, 2.0), reach 1.0, band 0.25: placed
    it spans y 1.4 to 2.6, 0.4 m from the line (`IN`); detected unturned it spans y 1.8 to 2.2, 0.8 m (`EDGE`); `judge`
    says `EDGE`. Centred (5, 2.5): placed 0.9 (`EDGE`), detected 1.3 (`OUT`): `EDGE`. Two footprints that agree keep
    the verdict. Regression caught: a thing judged by its true heading only, which a table could not reproduce."""
    p = pred("path", {"docking"}, reach_m=1.0, band_m=0.25)
    size = (1.2, 0.4, 0.3)
    for y, placed, seen in [(2.0, IN, EDGE), (2.5, EDGE, OUT)]:
        rect = Rect(5, y, size[0], size[1], math.pi / 2)
        assert (verdict(p, LINE, rect), verdict(p, LINE, detected_rect((5, y), size))) == (placed, seen)
        assert judge(p, LINE, [rect, detected_rect((5, y), size)]) == EDGE
    assert judge(p, LINE, [rect_at(0.5), rect_at(0.6)]) == IN
    assert judge(p, LINE, [rect_at(2.0), rect_at(3.0)]) == OUT


def test_a_table_lists_what_reaches_the_predicate():
    """Contract (plan §4 A4): `reaches` is a table's test on a detection: within `reach` of a path or point (the band
    included on the inside), and for rooms only `IN`. Reach 1.0: 0.9 m reaches (it is `EDGE` to the gold), 1.1 does not;
    on the hand map's living room a thing wholly in it reaches, one on the doorway's way or across the room's edge or in
    the kitchen does not. Regression caught: a table listing past the reach, or a thing in a doorway listed as in the
    room (the gold makes it don't-care, so listing it is harmless, but a table that cannot place it should not claim
    it)."""
    p = pred("path", {"docking"}, reach_m=1.0, band_m=0.25)
    assert reaches(p, LINE, rect_at(0.9)) and not reaches(p, LINE, rect_at(1.1))
    m = tk.room_map(tk.script(room="living room"))
    r = anchor_of(pred("rooms", {"cleaning"}), steps(["cleaning"], [(6, 1.5)]), m)
    wholly, on_way, across, kitchen = (Rect(6.5, 0.8, 0.4, 0.4, 0), Rect(4.2, 1.5, 0.2, 0.2, 0),
                                       Rect(4.0, 0.5, 0.4, 0.4, 0),
                                       Rect(2.0, 1.0, 0.4, 0.4, 0))
    rooms = pred("rooms", {"cleaning"})
    assert [verdict(rooms, r, x) for x in (wholly, on_way, across, kitchen)] == [IN, EDGE, EDGE, OUT]
    assert [reaches(rooms, r, x) for x in (wholly, on_way, across, kitchen)] == [True, False, False, False]


def test_standing_on_the_floor_is_within_the_robots_climb_of_its_base():
    """Contract (plan §4 A4): a thing stands on the floor the robot drives on when its bottom is within the robot's
    `climb_m` of the base height. Base at 0, climb 0.02: a bottom at 0.0 and at 0.02 are on it; 0.05, a rack's 1.3 and
    -0.05 are not. Regression caught: a rack-held thing taken for floor stock, or an open limit."""
    assert [on_floor(b, 0.0, 0.02) for b in (0.0, 0.02, 0.05, 1.3, -0.05)] == [True, True, False, False, False]


@pytest.mark.parametrize("field, value", [("phases", "docking"), ("dont_care_kinds", "bag"), ("phases", {"docking", 3}),
                                          ("dont_care_kinds", {""}), ("item_kinds", "bag"), ("item_kinds", {""})])
def test_phases_and_kinds_are_collections_of_names_never_a_bare_string(field, value):
    """Contract (A0's review; 0.11.0 adds `item_kinds`): `phases`, `dont_care_kinds` and `item_kinds` are collections
    of non-empty strings. A bare string is refused naming the field, since `"docking"` would pass the emptiness check
    and then match a phase by substring (`"dock" in "docking"`), and a non-string member never matches. Regression
    caught: a job declared with `phases="docking"` whose anchor silently matches the phases `docking` and `undocking`
    alike, or `item_kinds="bag"` that lists the kinds `bag` and `bag_of_x` alike."""
    with pytest.raises(ValueError, match=field):
        TaskThings("p", "path", **{"phases": frozenset({"docking"}), field: value})
    ok = TaskThings("p", "path", frozenset({"docking"}), dont_care_kinds=frozenset({"bag"}))
    assert ok.dont_care_kinds == {"bag"}


def test_the_hand_map_the_kit_uses_is_the_family_kits_with_a_wall_and_a_door():
    """Contract: `task_kit.room_map` keeps `family_kit.hand_map`'s two rooms (so the kit's expectations are the rooms of
    the synthetic script) and adds the wall and the door. Regression caught: the kit drifting from the shared map."""
    s = tk.script()
    m, base = tk.room_map(s), fk.hand_map(s)
    assert m.regions == base.regions and len(m.walls) == 1 and [o.id for o in m.openings] == ["D1"]
