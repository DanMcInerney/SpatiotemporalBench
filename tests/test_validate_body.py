"""validate()'s body rule (a robot's footprint keeps its clearance from walls, pieces, door leaves and floor items), the
movers' distance from the camera, and a robot that flies over what it clears."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from builders import CHAIR, DOOR, TASK2_T, hand_job, keyframe, minimal_timeline
from stbench.catalog import item_size
from stbench.geometry import quat_z
from stbench.mapspec import load_map, parse_map
from stbench.robots import ROBOTS
from stbench.timeline import (
    Ego,
    Entity,
    EpisodeInfo,
    Placement,
    Region,
    State,
    Task,
    TaskPlan,
    Timeline,
)
from stbench.validate import validate
from validate_kit import codes, generic_rules_only, with_keyframes, with_mover  # noqa: F401  (a fixture)

# these tests hand-make vacuum scripts, which the vacuum's own checks (`test_job_vacuum.py`) would refuse
pytestmark = pytest.mark.usefixtures("generic_rules_only")


# -- the body --------------------------------------------------------------------------------------------------------
def room_map(*pieces):
    """A 4 x 3 m room (walls 0.1 m thick, so their faces are at x = 0.05 and 3.95, y = 0.05 and 2.95) with the given
    fixed pieces (id, kind, centre, size, under_m) and no door."""
    wall = {"thickness": 0.1, "height": 2.5, "exterior": True}
    return parse_map({
        "name": "room", "bounds": [0, 0, 4.0, 3.0], "wall_thickness": 0.1,
        "wall_height": 2.5,
        "regions": [{"id": "R1", "name": "room", "polygon": [[0.05, 0.05], [3.95, 0.05], [3.95, 2.95], [0.05, 2.95]],
                     "anchor": [2.0, 1.5], "level": 0, "z0": 0.0, "z1": 2.5}],
        "walls": [{"id": "W01", "a": [0, 0], "b": [4, 0], **wall}, {"id": "W02", "a": [4, 0], "b": [4, 3], **wall},
                  {"id": "W03", "a": [0, 3], "b": [4, 3], **wall}, {"id": "W04", "a": [0, 0], "b": [0, 3], **wall}],
        "openings": [], "windows": [],
        "furniture": [{"id": f"F0{i}", "name": kind, "kind": kind, "mobility": "fixed", "center": list(center),
                       "size": list(size), "yaw": 0.0, "color": "white", "blocks": True, "surfaces": [],
                       "under_m": under} for i, (kind, center, size, under) in enumerate(pieces)],
        "ceiling_lights": [{"region": "R1", "pos": [2.0, 1.5, 2.45]}],
    })


def vacuum_in_room(m, bases, pieces, job="vacuum/1", mobility="fixed"):
    """A one-task timeline of the vacuum (of the job `job`) on `room_map`: its pieces (as entities of this mobility,
    placed at their centres) and a keyframe at each base (x, y), docked, heading east."""
    entities = [Entity(f"E000{i + 1}", kind, {}, mobility, size, under_m=under)
                for i, (kind, _, size, under) in enumerate(pieces)]
    placements = [Placement(f"E000{i + 1}", 0.0, None, (*center, 0.0), quat_z(0.0), None)
                  for i, (_, center, _, _) in enumerate(pieces)]
    kfs = [keyframe("vacuum", k, float(k), (x, y, 0.0), 0.0, task=0, phase="docked") for k, (x, y) in enumerate(bases)]
    info = EpisodeInfo("room", 1, "vacuum", job, "2026-03-02T08:00:00")
    return Timeline(info, [Region("R1", "room", "room", m.regions[0].polygon, 0, 0.0, 2.5)], entities, placements, [],
                    Ego(kfs, [Task(0, 0, len(kfs), TaskPlan("clean_room", {"room": "room"}), ())], []))


def body_problems(piece_kind, base, job="vacuum/1", mobility="fixed"):
    """The codes for the vacuum (a 0.17 m disc keeping 0.02 m) of the job `job` at `base` in a room with one
    0.12 x 0.28 piece of this kind and mobility, centred at (1.0, 1.5), whose east face is at x = 1.06."""
    piece = (piece_kind, (1.0, 1.5), (0.12, 0.28, 0.10), 0.0)
    m = room_map(piece)
    return codes(vacuum_in_room(m, [base], [piece], job, mobility), m)


def test_the_body_keeps_its_clearance():
    """Contract: at every keyframe the footprint at the base keeps `clearance_m` (0.02 m for the vacuum, less 1 mm)
    from a wall, and from a fixed piece, except that a piece of the job's anchor kind (`dock`) may be touched, never
    overlapped. Hand case: the room's east wall face is at x = 3.95, the vacuum's radius 0.17: a base at x = 3.77 is
    0.01 m from it (flagged), at 3.75 it is 0.03 m (fine). The piece's east face is at x = 1.06: a base at x = 1.23
    touches it (a dock: fine; a cabinet: flagged), at x = 1.225 it overlaps by 5 mm (a dock: flagged). The anchor
    comes from the job's `anchors`, not from the kind's name: under a job nobody registered (`vacuum/9`) the same dock
    touched at x = 1.23 is a problem like the cabinet, whether the dock is fixed or a light piece. Regression caught: a
    path through furniture, or a special case for the dock."""
    wall = room_map()
    assert codes(vacuum_in_room(wall, [(3.77, 1.5)], []), wall) == {"body_clearance"}
    assert codes(vacuum_in_room(wall, [(3.75, 1.5)], []), wall) == set()
    assert body_problems("dock", (1.23, 1.5)) == set()
    assert body_problems("cabinet", (1.23, 1.5)) == {"body_clearance"}
    assert body_problems("dock", (1.225, 1.5)) == {"body_clearance"}
    assert body_problems("cabinet", (1.25, 1.5)) == set()  # 0.02 m clear: the clearance exactly
    assert body_problems("dock", (1.23, 1.5), job="vacuum/9") == {"body_clearance"}  # no registered job, no anchor
    assert body_problems("dock", (1.23, 1.5), mobility="light") == set()  # an anchor, whatever it collides as
    assert body_problems("dock", (1.23, 1.5), job="vacuum/9", mobility="light") == {"body_clearance"}


def test_floor_items_and_clutter_obstruct_above_the_climb_height():
    """Contract (plan §2.2, the mobility table): a carried item or clutter lying on the floor (its base under
    `FLOOR_ITEM_Z`, 0.1 m) and taller than the robot's `climb_m` (0.02 m for the vacuum) keeps the body's clearance
    like a piece of furniture; a lower one is driven over, and one off the floor is no obstacle. Hand case: the vacuum
    at x = 2.0 (radius 0.17, keeping 0.02 m less 1 mm) and a 0.1 x 0.1 m item beside it at y = 1.5: centred at
    x = 2.23 its west face is 0.01 m from the body (flagged), at x = 2.25 it is 0.03 m (fine). At x = 2.23 an item
    0.015 m tall and one exactly 0.02 m tall are driven over, one 0.03 m tall is flagged, and a 0.1 m item with its
    base at z = 0.5 passes. Both mobilities, since clutter collides as carried. Regression caught: floor items that
    never obstruct (the robot driven over a shoe), a `climb_m` that is ignored (a 1.5 cm rim refused) or read as a
    strict bound, and a height that is not read (a mug on a shelf stopping the robot)."""
    def beside(mobility, x, *, height=0.1, z=0.0):
        m = room_map()
        tl = vacuum_in_room(m, [(2.0, 1.5)], [])
        tl.entities.append(Entity("E0009", "box", {}, mobility, (0.1, 0.1, height)))
        tl.placements.append(Placement("E0009", 0.0, None, (x, 1.5, z), quat_z(0.0), None))
        return codes(tl, m)

    for mobility in ("carried", "clutter"):
        assert beside(mobility, 2.23) == {"body_clearance"}, mobility
        assert beside(mobility, 2.25) == set(), mobility
        assert beside(mobility, 2.23, height=0.015) == set(), mobility
        assert beside(mobility, 2.23, height=0.02) == set(), mobility
        assert beside(mobility, 2.23, height=0.03) == {"body_clearance"}, mobility
        assert beside(mobility, 2.23, z=0.5) == set(), mobility


def test_a_person_keeps_the_bodys_clearance():
    """Contract (M2's call 21: `validate`'s body rule takes movers): a mover's footprint, at its yaw, keeps the robot's
    body clearance like a piece of furniture. Hand case: the vacuum at x = 2.0 (radius 0.17, keeping 0.02 m less 1 mm)
    and a 0.3 x 0.5 x 1.7 m person (mobility `mover`) beside it at y = 1.5, yaw 0: centred at x = 2.33 its west face is
    0.01 m from the body (flagged), at x = 2.40 it is 0.08 m (fine). Regression caught: the body rule skipping movers,
    so a script could put the robot inside a resident and `validate` would pass it."""
    def beside(x):
        m = room_map()
        tl = vacuum_in_room(m, [(2.0, 1.5)], [])
        tl.entities.append(Entity("E0009", "person", {}, "mover", (0.3, 0.5, 1.7)))
        tl.placements.append(Placement("E0009", 0.0, None, (x, 1.5, 0.0), quat_z(0.0), None))
        return codes(tl, m)

    assert "body_clearance" in beside(2.33)
    assert "body_clearance" not in beside(2.40)


def test_under_furniture_is_driven_through_not_over_its_legs():
    """Contract: a piece with an open underside the robot passes under (`under_m` >= its height + margin, 0.10 m for the
    vacuum) obstructs only at its four 0.15 m leg squares; one it cannot pass under obstructs whole. Hand case: a
    1.0 x 0.8 m table at (2.0, 1.5), x 1.5-2.5, y 1.1-1.9, its first leg square x 1.5-1.65, y 1.1-1.25: a base at the
    centre is 0.18 m from every leg (fine), a base on the leg is flagged, and the same centre under a table whose
    underside is 0.05 m up (a sofa) is flagged. Regression caught: a planner driving through table legs, or never
    under a table."""
    def on(under, base):
        piece = ("table", (2.0, 1.5), (1.0, 0.8, 0.75), under)
        m = room_map(piece)
        return codes(vacuum_in_room(m, [base], [piece]), m)

    assert on(0.6, (2.0, 1.5)) == set()
    assert on(0.6, (1.575, 1.175)) == {"body_clearance"}
    assert on(0.05, (2.0, 1.5)) == {"body_clearance"}
    assert on(0.0, (2.0, 1.5)) == {"body_clearance"}


def test_the_body_keeps_clear_of_a_door_leaf_in_the_state_it_has_then(apt):
    """Contract: a door leaf obstructs in the state the door has at that keyframe's time, a shut leaf closing its
    opening. Hand case: the kitchen door (D1) in the wall x = 4.6, y 1.475-2.325; the helper at (4.45, 1.9) facing
    east reaches x = 4.62, through a shut leaf (x 4.58-4.62) but 0.24 m clear of the open one (y 1.455-1.495). The
    door is open until 100 s and shut after, so keyframe 0 is fine and keyframe 3 (7200 s) is flagged. Regression
    caught: a leaf checked as if always open, or only the open one."""
    tl = minimal_timeline()
    kfs = list(tl.ego.keyframes)
    for k in (0, 3):
        kfs[k] = keyframe("helper", k, kfs[k].t, (4.45, 1.9, 0.0), 0.0, task=k // 3, phase="waiting")
    tl.states = [State(DOOR, "open", 0.0, 100.0, True), State(DOOR, "open", 100.0, None, False)]
    found = validate(with_keyframes(tl, kfs), apt)
    assert [(p.code, "keyframe 3" in p.message) for p in found] == [("body_clearance", True)]
    assert "leaf of door" in found[0].message


def test_the_body_keeps_clear_of_a_chair_that_has_moved_next_to_it(apt):
    """Contract: light furniture is geometry at the keyframe's time: a chair that has moved beside the robot is a
    problem at the keyframes after it moved, not before. Hand case: the chair (0.45 x 0.45) moves at 7100 s to
    (3.8, 1.4), x 3.575-4.025, y 1.175-1.625; the helper (0.34 x 0.33, keeping 0.10 m) at (3.8, 1.0) or (3.4, 1.0)
    reaches y 1.18-1.23 at its headings of 1.0 and 1.5 rad: within 0.10 m of the chair. Regression caught: a chair
    read at its first place only."""
    tl = minimal_timeline()
    tl.placements[1] = replace(tl.placements[1], t1=TASK2_T - 100.0)
    tl.placements.append(Placement(CHAIR, TASK2_T - 100.0, None, (3.8, 1.4, 0.0), quat_z(0.0), None))
    bad = [p.message for p in validate(tl, apt) if p.code == "body_clearance"]
    assert len(bad) == 3 and all(f"keyframe {k}" in b for k, b in zip((3, 4, 5), bad, strict=True))


def test_a_mover_keeps_its_distance_from_the_camera_at_every_keyframe(apt):
    """Contract: `mover_camera_clearance` when a mover's centre is within `MOVER_CAMERA_M`
    (0.6 m) of the camera in the plan at a keyframe, whatever its height; exactly 0.6 m is fine, and a mover that does
    not exist at a keyframe is not tested against it. The fixture's cameras stand at (3.8, 1.0) and (3.4, 1.0) (the
    helper's camera is above its base). Regression caught: the distance measured in 3D (the camera is 1.3 m up, so
    everything passes), a limit applied to the footprint's edge instead of the centre, and a check that reads the
    mover's pose at the wrong keyframe time."""
    near = with_mover(minimal_timeline(), (3.0, 1.0))  # 0.4 m from (3.4, 1.0)
    # two 0.34 m bodies 0.4 m apart leave at most 0.06 m between them, under the helper's 0.10 m: the body rule too (M2)
    assert codes(near, apt) == {"mover_camera_clearance", "body_clearance"}
    assert validate(with_mover(minimal_timeline(), (2.8, 1.0)), apt) == []  # 0.6 m from it
    assert codes(with_mover(minimal_timeline(), (2.801, 1.0)), apt) == {"mover_camera_clearance"}
    assert validate(with_mover(minimal_timeline(), (3.0, 1.0), t0=10.0, t1=7000.0), apt) == []  # in the gap only
    only_first_task = with_mover(minimal_timeline(), (3.0, 1.0), t0=0.0, t1=100.0)
    assert codes(only_first_task, apt) == {"mover_camera_clearance", "body_clearance"}  # the same place, as `near`


def test_a_mover_keeps_the_camera_clearance_of_its_own_kind(apt):
    """Contract (plan-warehouse §2, §3.5; `MoverKind.camera_m`): `mover_camera_clearance` is measured against the
    mover's kind, not one global distance: a forklift (1.76 m) with its centre 1.0 m from the camera is a problem and a
    robot vacuum (0.6 m) at 1.0 m is not; a forklift exactly 1.76 m away is fine and one 1.759 m away is not; the tote
    robot's limit is 0.85 m. The fixture's cameras stand at (3.8, 1.0) and (3.4, 1.0); the footprints of these big
    movers also hit the kitchen's furniture, so only the one code is asked about. Regression caught: the household's
    0.6 m applied to a forklift (the camera inside its 2.6 m body), a limit read
    from the wrong kind, or a kind's limit applied to the footprint's edge instead of the centre."""
    one_metre = (2.4, 1.0)  # 1.0 m from (3.4, 1.0)
    assert "mover_camera_clearance" in codes(with_mover(minimal_timeline(), one_metre, kind="forklift"), apt)
    assert "mover_camera_clearance" not in codes(with_mover(minimal_timeline(), one_metre), apt)  # a robot vacuum
    assert "mover_camera_clearance" not in codes(with_mover(minimal_timeline(), (1.64, 1.0), kind="forklift"), apt)
    assert "mover_camera_clearance" in codes(with_mover(minimal_timeline(), (1.641, 1.0), kind="forklift"), apt)
    assert "mover_camera_clearance" not in codes(with_mover(minimal_timeline(), (2.55, 1.0), kind="tote_robot"), apt)
    assert "mover_camera_clearance" in codes(with_mover(minimal_timeline(), (2.551, 1.0), kind="tote_robot"), apt)


def test_a_mover_of_a_kind_no_catalog_has_is_a_problem(apt):
    """Contract (`validate.unknown_mover_kind`): a mover whose kind no catalog lists has no known box, model or camera
    clearance, so the timeline is inconsistent: exactly that one code, on free kitchen floor. Regression caught: the
    clearance check passing silently with a default for any name (a mistyped "forklit" validating), or raising
    KeyError out of `validate` instead of returning a problem. Hand case: (2.0, 1.0) is free kitchen floor (the
    counters' north faces are at y 0.65, the table starts at y 1.5, the cameras stand 1.4 m away); (3.0, 0.6), used
    before, lies inside counter F04."""
    tl = minimal_timeline()
    tl.entities.append(Entity("E0006", "dragon", {"color": "red"}, "mover", (0.3, 0.3, 0.3)))
    tl.placements.append(Placement("E0006", 0.0, None, (2.0, 1.0, 0.0), quat_z(0.4), None))
    assert codes(tl, apt) == {"unknown_mover_kind"}


# -- a robot that flies (generator 0.7.0, plan-city §2) --------------------------------------------------------------
@pytest.fixture(scope="module")
def city():
    return load_map("downtown_a")


def city_timeline(city, robot, base, *, base_z=None, mover=None, light=None):
    """A hand-built timeline on `downtown_a` for `robot`: the south sidewalk (F05, 4 x 100 x 0.15 m at (50, 2), so x
    0-100, y 0-4), one task of two keyframes with the base at `base` (x, y) at height `base_z` (12 m for a flier, the
    floor for a ground robot), and optionally a mover kind standing 1.0 m east of the base, or a light piece (size)
    0.5 m east of it. The facade of the south block stands at y = 0 (W08 and W10, a wall piece to y = -0.6)."""
    r = ROBOTS[robot]
    z = base_z if base_z is not None else 12.0 if r.flies else 0.0
    kf = [keyframe(robot, k, 1.5 * k, (base[0], base[1], z), 0.0, task=0, phase=r.status.start["phase"])
          for k in range(2)]
    plan = TaskPlan("clean_room" if robot == "vacuum" else "hand", {})  # the vacuum's job takes only its own kind
    entities = [Entity("E0001", "sidewalk", {"name": "sidewalk", "color": "grey"}, "fixed", (4.0, 100.0, 0.15))]
    placements = [Placement("E0001", 0.0, None, (50.0, 2.0, 0.0), quat_z(math.pi / 2), None)]
    if mover:
        entities.append(Entity("E0002", mover, {"color": "blue"}, "mover", item_size(mover)))
        placements.append(Placement("E0002", 0.0, None, (base[0] + 1.0, base[1], 0.0), quat_z(0.0), None))
    if light:
        entities.append(Entity("E0002", "bin", {"color": "green"}, "light", light))
        placements.append(Placement("E0002", 0.0, None, (base[0] + 0.5, base[1], 0.0), quat_z(0.0), None))
    return replace(
        minimal_timeline(),
        episode=EpisodeInfo("downtown_a", 1, robot, hand_job(robot), "2026-03-02T08:00:00"),
        regions=[Region(x.id, x.name, x.kind, x.polygon, x.level, x.z0, x.z1) for x in city.regions],
        entities=entities, placements=placements, states=[],
        ego=Ego(kf, [Task(0, 0, 2, plan, ())], []))


def test_a_flying_robot_ignores_a_sidewalk_under_it_and_not_a_wall(city):
    """Contract (plan-city §2; `robots.over_m`): `body_clearance` for the drone (a 0.65 m disc keeping 3.0 m, flying
    over 9 m) counts the walls and the pieces whose top reaches 9 m, so a drone over the 0.15 m sidewalk is fine at
    3.7 m from the south block's facade (3.05 m clear of it) and a problem 2.0 m from it; the vacuum, which keeps its
    body clear of every piece, has the sidewalk under it as the problem and the facade 12 m away as none. Hand case:
    the sidewalk covers y 0-4 and the facade's wall piece reaches y = 0 at x = 20. Regression caught: the drone checked
    against the sidewalks (every keyframe of every route is a problem), against none at all (a facade 2 m from the
    body), or the ground robots' clearance no longer counting the piece under them."""
    assert validate(city_timeline(city, "drone", (20.0, 3.7)), city) == []
    near = validate(city_timeline(city, "drone", (20.0, 2.0)), city)
    assert {p.code for p in near} == {"body_clearance"} and "a wall" in near[0].message
    assert {p.code for p in validate(city_timeline(city, "vacuum", (20.0, 3.5)), city)} == {"body_clearance"}
    assert validate(city_timeline(city, "vacuum", (20.0, 12.0)), city) == []


def test_a_flying_robot_keeps_its_clearance_from_what_reaches_over_it(city):
    """Contract (plan-city §2): a light piece counts for a flier by its height as a fixed one does: a 1 m bin 0.5 m from
    the drone is no problem (it passes 11 m above it), a 12 m pole there is (its top is above the 9 m the drone clears),
    and the vacuum keeps clear of the bin. At the boundary (`robots.flies_over`) a piece of 8.99 m is flown over and one
    of exactly 9.0 m is not. Regression caught: the height rule applied to fixed furniture only (a drone that hits a
    tall light piece), to none (a drone refused by every bin), or a boundary that differs from the planner's."""
    assert validate(city_timeline(city, "drone", (20.0, 12.0), light=(1.0, 1.0, 1.0)), city) == []
    pole = validate(city_timeline(city, "drone", (20.0, 12.0), light=(1.0, 1.0, 12.0)), city)
    assert {p.code for p in pole} == {"body_clearance"}
    assert {p.code for p in validate(city_timeline(city, "vacuum", (20.0, 12.0), light=(1.0, 1.0, 1.0)), city)} == {
        "body_clearance"}
    assert validate(city_timeline(city, "drone", (20.0, 12.0), light=(1.0, 1.0, 8.99)), city) == []
    edge = validate(city_timeline(city, "drone", (20.0, 12.0), light=(1.0, 1.0, 9.0)), city)
    assert {p.code for p in edge} == {"body_clearance"}


def test_a_bus_under_the_drone_is_no_camera_clearance_problem_but_is_under_the_vacuum(city):
    """Contract (plan-city §2, §3.5; `validate_ego._mover_camera`): `mover_camera_clearance` is silent for a mover that
    the robot flies over (the drone's `over_m`, 9 m, is above the mover's height: the 3.1 m bus 1.0 m from it in the
    plan) and fires for the same bus under the vacuum, whose camera keeps the bus's `camera_m` of 5.02 m. Hand case: the
    base at (40, 13.5) on the high street's roadway, the bus at (41, 13.5). Regression caught: the flying rule missing
    from the validator (every drone script with a bus near the route fails validation), or applied to the ground
    robots (the apartment's movers could stand on the camera)."""
    assert validate(city_timeline(city, "drone", (40.0, 13.5), mover="bus"), city) == []
    ground = validate(city_timeline(city, "vacuum", (40.0, 13.5), mover="bus"), city)
    # the vacuum 1.0 m from the bus's centre is inside its footprint: the body rule fires too (M2: it takes movers)
    camera = [p for p in ground if p.code == "mover_camera_clearance"]
    assert {p.code for p in ground} == {"mover_camera_clearance", "body_clearance"}
    assert "keeps 5.02" in camera[0].message


def test_a_mover_as_tall_as_the_height_a_robot_clears_is_not_flown_over(city, monkeypatch):
    """Contract (plan-city §2; `robots.flies_over`): a flier passes over a mover only when the mover's top is below the
    height it clears, as it does for a piece of furniture (`mover_camera_clearance` is silent for it, and only then).
    Hand case: a drone whose base is 3.6 m up and keeps 0.5 m clears exactly 3.1 m, the bus's height, so it keeps the
    bus's `camera_m` from a bus 1.0 m away, and one 0.1 m higher clears 3.2 m and does not. Regression caught: the
    mover test reading the boundary the other way from the furniture's (a bus as tall as the robot clears is flown
    over by the validator and met by the planner)."""
    bus_top = item_size("bus")[2]
    drone = ROBOTS["drone"]
    monkeypatch.setitem(ROBOTS, "edge", replace(drone, id="edge", body=replace(drone.body, clearance_m=0.5)))
    for z, silent in ((bus_top + 0.5, False), (bus_top + 0.6, True)):
        found = {p.code for p in validate(city_timeline(city, "edge", (40.0, 13.5), base_z=z, mover="bus"), city)}
        # not flown over, the bus 1.0 m away also meets the body (M2: the body rule takes movers)
        assert found == (set() if silent else {"mover_camera_clearance", "body_clearance"}), z
