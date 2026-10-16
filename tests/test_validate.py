"""validate() must catch each class of world inconsistency in V0 §3, and nothing in a consistent timeline. The ego's
rules are in `test_validate_ego.py`, the body rule, the movers' camera distance and the fliers' in
`test_validate_body.py`.

Each test breaks exactly one thing in the hand-built timeline (tests/builders.py: the home helper in the kitchen of
`apartment_a`) and checks the one code it expects.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from builders import CHAIR, DOOR, TABLE, minimal_timeline
from stbench.geometry import quat_z
from stbench.timeline import Entity, Placement, State
from stbench.validate import TimelineError, check, validate
from validate_kit import codes, with_keyframes, with_mover


def test_consistent_timeline_has_no_problems(apt):
    """Negative control: the fixture is valid, so every other test here fails for the one thing it breaks. Regression
    caught: a rule that flags the valid fixture, which would make every test below pass for the wrong reason."""
    assert validate(minimal_timeline(), apt) == []


def test_check_raises_with_the_problems(apt):
    """Contract: check() turns problems into an exception that names them. Regression caught: a check that returns
    quietly, which lets bad timelines through."""
    tl = minimal_timeline()
    tl.entities.append(tl.entities[0])
    with pytest.raises(TimelineError, match="duplicate_entity"):
        check(tl, apt)


def test_overlapping_placements_of_one_entity(apt):
    """Contract: an entity cannot be in two places at once: a placement that starts before the previous one ends is
    `placement_overlap`. Regression caught: a mug in two places, which a keyframe's world would then place twice."""
    tl = minimal_timeline()
    tl.placements[2] = replace(tl.placements[2], t1=4000.0)  # mug's second placement starts at 3600
    assert codes(tl, apt) == {"placement_overlap"}


def test_open_ended_placement_followed_by_another(apt):
    """Contract: t1=None means "until the end", so any later placement of the same entity overlaps it. Regression
    caught: an open end read as a finite time, which lets an entity stand in two places after it."""
    tl = minimal_timeline()
    tl.placements[2] = replace(tl.placements[2], t1=None)
    assert "placement_overlap" in codes(tl, apt)


def test_overlapping_state_intervals_of_one_key(apt):
    """Contract: two values of one state key at the same time are `state_overlap`; different keys may coexist.
    Regression caught: a door both open and shut at one time, or a rule that rejects a door both open and locked."""
    tl = minimal_timeline()
    tl.states.append(State(DOOR, "open", 100.0, 200.0, False))
    assert codes(tl, apt) == {"state_overlap"}
    tl2 = minimal_timeline()
    tl2.states.append(State(DOOR, "locked", 100.0, 200.0, True))
    assert validate(tl2, apt) == []


def test_empty_or_negative_intervals(apt):
    """Contract: half-open intervals are non-empty and start at or after zero (`bad_interval`). Regression caught: an
    empty or negative interval, which no keyframe's time falls in."""
    tl = minimal_timeline()
    tl.placements[2] = replace(tl.placements[2], t1=0.0)
    assert "bad_interval" in codes(tl, apt)


def test_supporting_entity_missing_when_item_rests_on_it(apt):
    """Contract: an item "on" something that has been removed is floating, so the support must exist for the whole
    interval (`on_target_absent`). Regression caught: a mug left hanging in the air after its table was removed."""
    tl = minimal_timeline()
    tl.placements[0] = replace(tl.placements[0], t1=1000.0)  # the table vanishes at 1000 s, the mug stays on it
    assert "on_target_absent" in codes(tl, apt)


def test_supporting_entity_unknown(apt):
    """Contract: `on` names a real entity (`on_target_absent`). Regression caught: a support that is a typo, read as
    "nothing to check"."""
    tl = minimal_timeline()
    tl.placements[4] = replace(tl.placements[4], on="E9999")
    assert "on_target_absent" in codes(tl, apt)


def test_item_hovering_beside_its_support(apt):
    """Contract: an item whose centre is off the support's footprint is not resting on it (`off_surface`). Regression
    caught: a mug hovering beside its table, which the renderer would draw in mid-air."""
    tl = minimal_timeline()
    tl.placements[4] = replace(tl.placements[4], pos=(3.5, 2.1, 0.75))  # table spans x 1.4-2.8
    assert codes(tl, apt) == {"off_surface"}


def test_item_floating_above_its_support(apt):
    """Contract: an item rests on or below the top of its support; one at 1.2 m above a 0.75 m table floats
    (`off_surface`). Regression caught: a height that is never compared with the support's top."""
    tl = minimal_timeline()
    tl.placements[4] = replace(tl.placements[4], pos=(2.4, 2.1, 1.2))
    assert codes(tl, apt) == {"off_surface"}


def test_two_carried_items_in_the_same_spot(apt):
    """Contract: carried items cannot interpenetrate: overlapping footprints at the same height are
    `carried_collision`. Regression caught: two mugs in one spot, drawn through each other."""
    tl = minimal_timeline()
    tl.placements[4] = replace(tl.placements[4], pos=(2.12, 1.92, 0.75))  # on top of the first mug
    assert codes(tl, apt) == {"carried_collision"}


def test_clutter_collides_as_carried(apt):
    """Contract (the mobility table): a clutter entity is checked as a carried item, so two clutter pieces (or a
    clutter piece and a mug) cannot occupy one spot. Regression caught: a mobility the table knows but the collision
    rule does not, so floor clutter could be generated inside a mug."""
    tl = minimal_timeline()
    tl.entities.append(Entity("E0006", "sock", {"color": "grey"}, "clutter", (0.1, 0.1, 0.1)))
    tl.placements.append(Placement("E0006", 0.0, None, (2.12, 1.92, 0.75), quat_z(0.0), TABLE))
    assert codes(tl, apt) == {"carried_collision"}


def test_carried_items_at_different_times_may_share_a_spot(apt):
    """Contract: a collision needs time overlap: the first mug's old spot is free once it has moved. Regression
    caught: placements compared across time, which would refuse every item that moves onto a spot another left."""
    tl = minimal_timeline()
    tl.placements[4] = replace(tl.placements[4], t0=3600.0, pos=(2.1, 1.9, 0.75))  # where the mug used to be
    assert validate(tl, apt) == []


def test_carried_items_on_different_shelf_levels_do_not_collide(apt):
    """Contract: footprints may coincide on different surface heights; only same-height overlap collides. Regression
    caught: footprints compared in the plan alone, which refuses a mug set on top of another."""
    tl = minimal_timeline()
    tl.placements[4] = replace(tl.placements[4], pos=(2.1, 1.9, 0.75 + 0.1))  # exactly on top of the first mug
    assert "carried_collision" not in codes(tl, apt)


def test_light_furniture_inside_fixed_furniture(apt):
    """Contract: a light piece (a chair) pushed into a fixed one (the table) is `light_collision`. Regression caught:
    light furniture shifted through fixed furniture."""
    tl = minimal_timeline()
    tl.placements[1] = replace(tl.placements[1], pos=(1.8, 1.9, 0.0))
    assert "light_collision" in codes(tl, apt)


def test_placement_outside_the_map(apt):
    """Contract: poses lie inside the rooms: a mug at x = 20 is `outside_map`. Regression caught: an item placed
    outside the apartment, which no camera could meet."""
    tl = minimal_timeline()
    tl.placements[4] = replace(tl.placements[4], pos=(20.0, 2.1, 0.75))
    assert "outside_map" in codes(tl, apt)


def test_light_furniture_straddling_a_wall(apt):
    """Contract: a chair half through the kitchen's east wall is outside the rooms even though its centre is inside.
    Regression caught: light furniture tested by its centre alone, which lets it straddle a wall."""
    tl = minimal_timeline()
    tl.placements[1] = replace(tl.placements[1], pos=(4.5, 2.5, 0.0))
    assert "outside_map" in codes(tl, apt)


def test_non_unit_rotation(apt):
    """Contract: rotations are unit quaternions, a placement's and a keyframe's. Regression caught: a scaled
    quaternion, which silently shears footprints and camera axes."""
    tl = minimal_timeline()
    tl.placements[0] = replace(tl.placements[0], rot=(2.0, 0.0, 0.0, 0.0))
    assert "bad_rotation" in codes(tl, apt)
    tl2 = minimal_timeline()
    kfs = list(tl2.ego.keyframes)
    kfs[0] = replace(kfs[0], rot=(0.5, 0.0, 0.0, 0.0))
    assert "bad_rotation" in codes(with_keyframes(tl2, kfs), apt)


def test_duplicate_unknown_and_bad_mobility(apt):
    """Contract: entity ids are unique, placements and states name real entities, and a mobility is a known class.
    Regression caught: a placement of an entity nobody declared, or a duplicated id, read as the first one's."""
    tl = minimal_timeline()
    tl.placements.append(Placement("E4040", 0.0, None, (1.0, 1.0, 0.0), quat_z(0.0), None))
    tl.states.append(State("E4041", "open", 0.0, None, True))
    tl.entities.append(replace(tl.entities[0], mobility="floating"))
    assert {"unknown_entity", "duplicate_entity", "bad_mobility"} <= codes(tl, apt)


def test_door_is_not_a_collision_target(apt):
    """Contract: doors are fixed entities that sit in wall gaps; a chair beside the door does not collide with its
    leaf. Regression caught: a door collided as a solid piece, which refuses every chair near a doorway."""
    tl = minimal_timeline()
    tl.placements[1] = replace(tl.placements[1], pos=(4.2, 1.9, 0.0))  # 0.4 m from the door plane, not overlapping
    assert "light_collision" not in codes(tl, apt)


# -- movers and the camera's distance from them ----------------------------------------------------------------------
def test_a_mover_is_a_known_mobility(apt):
    """Contract: "mover" is in the timeline's mobility vocabulary, so a mover entity on free
    kitchen floor is consistent and a made-up mobility still is not. The collision rules for movers are tested below.
    Hand case: (2.0, 1.0) is free floor of the apartment's kitchen (the counters' north faces are at y 0.65, the table
    starts at y 1.5, the fixture's cameras stand 1.4 m away at (3.4, 1.0)), and the vacuum's 0.34 m body turned by 0.4
    rad reaches 0.22 m from its centre. (3.0, 0.4), used before, lies inside counter F04 and passed only because the
    fixture holds no counter. Regression caught: "mover" missing from `timeline.MOBILITIES`, which fails every
    timeline holding one with `bad_mobility`."""
    tl = minimal_timeline()
    tl.entities.append(Entity("E0006", "robot_vacuum", {"color": "white"}, "mover", (0.34, 0.34, 0.09)))
    tl.placements.append(Placement("E0006", 0.0, None, (2.0, 1.0, 0.0), quat_z(0.4), None))
    assert validate(tl, apt) == []
    tl.entities[-1] = replace(tl.entities[-1], mobility="hovering")
    assert codes(tl, apt) == {"bad_mobility"}


def test_a_mover_is_inside_the_map_by_its_centre(apt):
    """Contract: like a carried item or a door, a mover lies inside the map when its centre does
    (a footprint may straddle a doorway), so a mover standing in the kitchen door is fine, and one whose centre is in
    the wall or outside is `outside_map`. Regression caught: a mover tested by `rect_inside`, which rejects every
    mover that stands in a doorway of the apartment (the walks cross all four)."""
    assert validate(with_mover(minimal_timeline(), (4.6, 1.9)), apt) == []
    assert "outside_map" in codes(with_mover(minimal_timeline(), (4.6, 1.2)), apt)  # the wall beside the door
    assert "outside_map" in codes(with_mover(minimal_timeline(), (-1.0, 0.5)), apt)


def door_leaf_midpoint(apt, is_open):
    """The middle of the kitchen door's leaf (the door at x = 4.6, y = 1.9) as the map draws it."""
    door = next(o for o in apt.doors() if abs(o.center[0] - 4.6) < 1e-6 and abs(o.center[1] - 1.9) < 1e-6)
    (hx, hy), (tx, ty) = apt.door_leaf(door, is_open)
    return ((hx + tx) / 2, (hy + ty) / 2)


def test_a_mover_may_not_overlap_what_it_rolls_among(apt):
    """Contract: `mover_collision` for a mover's footprint against fixed furniture, light
    furniture (at the time), a wall piece, a door leaf (in the state it has), a carried item or clutter lying on the
    floor and another mover (while both exist). Each case is a pose chosen to overlap one thing only. Regression
    caught: any one class dropped from the rule, and a rule that reads the wrong time (a chair that has moved away
    still blocking, a closed leaf checked as if open)."""
    table = (2.1, 1.9)  # the dining table spans x 1.4-2.8, y 1.5-2.3
    assert "mover_collision" in codes(with_mover(minimal_timeline(), table), apt)
    # fixed furniture is checked for the whole timeline, light furniture only while the chair stands there
    chair = (1.0, 1.9)  # the chair spans x 0.93-1.38, y 1.68-2.12; the mover x 0.83-1.17, clear of the table
    assert "mover_collision" in codes(with_mover(minimal_timeline(), chair), apt)

    def chair_leaves_at_100(t0):
        tl = minimal_timeline()
        tl.placements[1] = replace(tl.placements[1], t1=100.0)  # the chair stands there until t = 100 only
        tl.placements.append(Placement(CHAIR, 100.0, None, (1.155, 0.7, 0.0), quat_z(0.0), None))
        return with_mover(tl, chair, t0=t0, t1=150.0)

    assert "mover_collision" in codes(chair_leaves_at_100(50.0), apt)
    assert "mover_collision" not in codes(chair_leaves_at_100(100.0), apt)
    # the west wall's piece spans x -0.05-0.05; the mover at x 0.15 reaches -0.02, and its centre is inside the kitchen
    assert "mover_collision" in codes(with_mover(minimal_timeline(), (0.15, 1.0)), apt)
    # the leaf of the kitchen door: open in the fixture, so its middle is blocked; shut, that spot is free
    spot = door_leaf_midpoint(apt, True)
    assert "mover_collision" in codes(with_mover(minimal_timeline(), spot), apt)
    shut = with_mover(minimal_timeline(), spot)
    shut.states[0] = State(DOOR, "open", 0.0, None, False)
    assert "mover_collision" not in codes(shut, apt)
    # a carried item or clutter on the floor blocks; the same item on a shelf (above FLOOR_ITEM_Z) does not
    for mobility in ("carried", "clutter"):
        for z, expect in ((0.01, True), (0.5, False)):
            tl = minimal_timeline()
            tl.entities.append(Entity("E0007", "box", {"color": "brown"}, mobility, (0.3, 0.3, 0.2)))
            tl.placements.append(Placement("E0007", 0.0, None, (3.0, 0.5, z), quat_z(0.0), None))
            assert ("mover_collision" in codes(with_mover(tl, (3.1, 0.5)), apt)) is expect, (mobility, z)
    # another mover blocks while both exist, and not after the first has gone
    both = with_mover(with_mover(minimal_timeline(), (2.7, 0.5)), (2.8, 0.5), eid="E0007")
    assert "mover_collision" in codes(both, apt)
    later = with_mover(with_mover(minimal_timeline(), (2.7, 0.5), t1=100.0), (2.8, 0.5), eid="E0007", t0=100.0)
    assert validate(later, apt) == []


# -- a mover on its own floor (M3-people: a pedestrian stands at 0.15 m on a 0.15 m sidewalk) -------------------------
PERSON = (0.34, 0.56, 1.8)
STEP_AT = (2.0, 1.0)  # free kitchen floor (the fixture's cameras stand 1.4 m east of it, test above)


def with_things(tl, *things):
    """`tl` plus entities placed as (id, kind, mobility, size, x, y, z) over the whole timeline."""
    for eid, kind, mobility, size, x, y, z in things:
        tl.entities.append(Entity(eid, kind, {}, mobility, size))
        tl.placements.append(Placement(eid, 0.0, None, (x, y, z), quat_z(0.0), None))
    return tl


def with_person(tl, z, eid="E0006", at=STEP_AT, t0=0.0, t1=None):
    """`tl` plus a person (a mover of the people kind, 0.34 x 0.56 x 1.8 m) standing at base height z."""
    if eid not in {e.id for e in tl.entities}:
        tl.entities.append(Entity(eid, "person", {}, "mover", PERSON))
    tl.placements.append(Placement(eid, t0, t1, (at[0], at[1], z), quat_z(0.4), None))
    return tl


def step(height):
    """A fixed piece 0.6 m square of this height (a sidewalk's top) where the person stands."""
    return ("E0007", "sidewalk", "fixed", (0.6, 0.6, height), *STEP_AT, 0.0)


def test_a_mover_stands_on_the_floor_it_is_at(apt):
    """Contract (M3-people §3): a fixed piece whose top is at most the mover's base height plus `SURFACE_Z_TOL`
    (1e-6 to spare) is the mover's floor, never in its way; anything taller is, and the home's movers (at height 0) keep
    the old rule. Hand cases on the kitchen floor, a 0.6 m square piece at (2.0, 1.0) and a person 0.34 x 0.56 m at its
    middle: a person at 0.15 m on a 0.15 m piece is valid (the whole timeline, `validate`); at 0 m on it, or at 0.15 m
    on one 0.1601 m tall (over the 0.15 + 0.01 m tolerance), it is `mover_collision`; at 0.15 m on one 0.16 m tall it is
    not (0.15 + 0.01 is 0.16 in floats by luck, the review's minor 6); at 0.06 m on a 0.06 m piece (an ordinary
    blocking piece, over the 0.05 m of a floor covering) it is fine and at 0 m it is not; and a 0.01 m rug is
    never in the way. Regression caught: a rule that forbids sidewalks, one that lets a person stand inside a kerb, or a
    tolerance that holds in floats only by luck."""
    on = lambda height, z: codes(with_person(with_things(minimal_timeline(), step(height)), z), apt)  # noqa: E731
    assert validate(with_person(with_things(minimal_timeline(), step(0.15)), 0.15), apt) == []
    assert "mover_collision" in on(0.15, 0.0)
    assert "mover_collision" not in on(0.16, 0.15) and "mover_collision" in on(0.1601, 0.15)
    assert "mover_collision" not in on(0.06, 0.06) and "mover_collision" in on(0.06, 0.0)
    assert "mover_collision" not in on(0.01, 0.0)


def test_things_on_a_movers_floor_are_in_its_way_and_things_below_are_not(apt):
    """Contract (M3-people §3): a light piece stands in a mover's way whatever its base, and a carried item or clutter
    only when its base is within `FLOOR_ITEM_Z` (0.1 m) of the mover's own base height (a car in the roadway at 0 m is
    not in the way of a pedestrian at 0.15 m; a cart on the plaza at 0.15 m is). Hand cases: a person at 0.15 m on a
    0.15 m piece, with a food cart (a light piece 1.2 x 0.8 x 1.1 m) at (2.1, 1.0) on the piece (base 0.15 m) is a
    collision; an item (0.3 m cube) at the person's feet (2.0, 1.0) is one at base 0.15 m or 0.2 m (0.05 m above the
    floor) and not at base 0 m (0.15 m below it) or 0.5 m (on a shelf). Regression caught: a cart a pedestrian walks
    through, or an item below the kerb that blocks the pavement."""
    cart = ("E0008", "food_cart", "light", (1.2, 0.8, 1.1), 2.1, 1.0, 0.15)
    base = lambda *things: with_person(with_things(minimal_timeline(), step(0.15), *things), 0.15)  # noqa: E731
    assert "mover_collision" in codes(base(cart), apt)
    for mobility in ("carried", "clutter"):
        for z, in_the_way in ((0.0, False), (0.15, True), (0.2, True), (0.5, False)):
            item = ("E0008", "box", mobility, (0.3, 0.3, 0.3), *STEP_AT, z)
            assert ("mover_collision" in codes(base(item), apt)) is in_the_way, (mobility, z)


def test_movers_are_in_each_others_way_only_while_they_stand_together(apt):
    """Contract (`mover_collision` between movers): two different movers whose footprints overlap collide for the time
    both exist, and one mover's own successive placements never do. Hand case (three people at one spot, listed out
    of time order): A over [0, 10), B over [5, 15), C over [10, 20), all at the same place: A and B overlap in time,
    so do B and C, and A and C meet only at t = 10, where A has gone: three collisions exactly (A with B, B with C
    twice: B has a second placement over [15, 25) that C meets over [15, 20)), never A with C or a person with itself.
    Regression caught: a sweep over time that pairs placements that touch end to start, or misses a pair when
    placements are not in time order."""
    tl = with_person(minimal_timeline(), 0.0, eid="E0009", t0=10.0, t1=20.0)  # C first
    with_person(tl, 0.0, eid="E0007", t0=0.0, t1=10.0)  # A
    with_person(tl, 0.0, eid="E0008", t0=5.0, t1=15.0)  # B
    with_person(tl, 0.0, eid="E0008", t0=15.0, t1=25.0)  # B again, later: overlaps C over [15, 20)
    hits = [p.message for p in validate(tl, apt) if p.code == "mover_collision"]
    pairs = sorted(tuple(sorted(e for e in ("E0007", "E0008", "E0009") if e in msg)) for msg in hits)
    assert pairs == [("E0007", "E0008"), ("E0008", "E0009"), ("E0008", "E0009")], hits
