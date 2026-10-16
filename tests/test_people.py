"""The people overlay (`people`, the home's resident in `people_home`): who is home, where they walk, and what they keep
clear of. A resident is a `mover` entity of kind `person`; `add` runs last in `generate_script`. The scripts are
`people_kit`'s: the apartment's generated `vacuum/1` scripts and a three-task script with gap changes."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from builders import keyframe, minimal_timeline
from conftest import script_on_map
from people_kit import (
    apartment,
    gaps,
    generated,
    multi_task,
    obstacles_at,
    people_rng,
    resident,
    robot_gaps,
)
from stbench import (
    asset_manifest,
    job_registry,
    people,
    people_downtown,
    people_home,
    people_residential,
    people_warehouse,
)
from stbench import episode as ep
from stbench import perception as P
from stbench import timeline as T
from stbench.camera import Camera
from stbench.catalog import MoverKind
from stbench.geometry import points_in_polygon, segments_cross
from stbench.leak import leak_scan
from stbench.mapspec import load_map
from stbench.mover_walk import Walk, Walker
from stbench.people import without_people
from stbench.robots import ROBOTS, camera_doc
from stbench.route import FreeSpace
from stbench.timeline import Ego, Task, TaskPlan
from stbench.validate import validate

SEEDS = (101, 102, 103)
BOX_M = people_home.BOX_M
# the catalogs' people: the home's resident, M3's pickers and pedestrians, M4's residential pedestrians
MODULES = (people_home, people_warehouse, people_downtown, people_residential)


@pytest.fixture
def always_home(monkeypatch):
    """A resident home for every task (the declared share is 0.5)."""
    monkeypatch.setattr(people_home, "P_HOME", 1.0)


def resident_scripts():
    """The scripts a resident is added to: the generated `vacuum/1` scripts of seeds 101-103 as they were before the
    overlay, and two three-task scripts with gap changes."""
    return [without_people(generated(s)) for s in SEEDS] + [multi_task(101), multi_task(102)]


def test_scripts_with_people_validate(always_home):
    """Contract (the validate rules `mover_collision`, `mover_camera_clearance`, `unknown_mover_kind`): a resident home
    for every task of the generated `vacuum/1` scripts of seeds 101-103 and of two hand-made three-task scripts, with a
    chair shifted, things moved and a delivery or a removal inside each gap and the bathroom's door shut for ten
    minutes of it, passes `validate` clean (as the scripts do without it), and is there in every task. Its first and
    last poses of a task stand through the gaps around it, through all those changes. Regression caught: a resident
    standing where a gap's change lands, or in a closing door (drawn on the world as it stood when the task began,
    every one of these five scripts fails), or in a wall or a piece of furniture."""
    m = apartment()
    for i, base in enumerate(resident_scripts()):
        assert validate(base, m) == [], i
        tl = people.add(base, m, people_rng(101 + i))
        eid = resident(tl)
        assert eid is not None and validate(tl, m) == [], i
        assert all(tl.placement_of(eid, tl.keyframe(t.k0).t) is not None for t in tl.ego.tasks), i


def test_a_resident_walks_into_each_room():
    """Contract (the free space of a resident, 0.05 m cells with half the box's width and 0.05 m of clearance: 0.33 m,
    since the review's M2 found a circumradius of clearance closed every door): on seed 101's world
    with every interior door open, a path joins the free cell nearest the hall, 0.65 m before each of the hall's four
    doors, to the one 0.65 m beyond it in under 2.5 m (a way round by another door is over 5 m), crossing the door's
    line; and a walk drawn from the hall with its waypoint forced into each of the other rooms in turn arrives in it.
    Regression caught: a clearance that closes every door, so residents never leave the kitchen and living room."""
    m, tl = apartment(), without_people(generated(101))
    free = FreeSpace(m, obstacles_at(tl, m, 0.0), BOX_M[1] / 2 + 0.05, 0.05)
    doors = [o for o in m.doors() if o.interior and "R1" in o.between]
    assert [o.id for o in doors] == ["D1", "D2", "D3", "D4"]
    for o in doors:
        normal = (-math.sin(o.yaw), math.cos(o.yaw))
        sides = [(o.center[0] + s * 0.65 * normal[0], o.center[1] + s * 0.65 * normal[1]) for s in (1, -1)]
        hall, room = (nearest_free(free, p) for p in sorted(sides, key=lambda p: m.region_of(p) != "R1"))
        way = free.path(hall, room)
        assert way is not None and sum(math.dist(a, b) for a, b in zip(way, way[1:], strict=False)) < 2.5, o.id
        tangent = (math.cos(o.yaw), math.sin(o.yaw))
        half = o.width / 2
        ends = [(o.center[0] + s * half * tangent[0], o.center[1] + s * half * tangent[1]) for s in (1, -1)]
        assert any(segments_cross(a, b, *ends) for a, b in zip(way, way[1:], strict=False)), o.id
        target = room_cell(m, free, m.region_of(room), (5.4, 3.1))
        assert m.region_of(target) == m.region_of(room) and free.is_free((5.4, 3.1))
        assert walks_to(free, (5.4, 3.1), target) < 0.06, o.id


def nearest_free(free, p, within: float = 0.3) -> tuple[float, float]:
    """The free cell nearest `p`, which a piece of clutter may have taken (it must be within `within`)."""
    cx, cy = free.cells_xy(free.free)
    i = int(np.hypot(cx - p[0], cy - p[1]).argmin())
    assert math.dist((cx[i], cy[i]), p) <= within, p
    return float(cx[i]), float(cy[i])


def room_cell(m, free, region_id: str, start) -> tuple[float, float]:
    """The cell of the region, connected on the free floor to `start`, nearest the mean of the region's corners (a
    room's middle can be a pocket the furniture walls off)."""
    corners = m.region(region_id).polygon
    mid = (sum(p[0] for p in corners) / len(corners), sum(p[1] for p in corners) / len(corners))
    cx, cy = free.cells_xy(free.reachable_from(start))
    inside = points_in_polygon(cx, cy, [list(p) for p in corners])
    assert inside.any(), region_id
    cx, cy = cx[inside], cy[inside]
    i = int(np.hypot(cx - mid[0], cy - mid[1]).argmin())
    return float(cx[i]), float(cy[i])


def walks_to(free, start, target) -> float:
    """How far from `target` a resident ends 400 s after standing at `start`, when every waypoint it draws is the
    target."""
    kind = people.person_kind(people_home.KIND)

    class Forced(Walker):
        def _waypoint(self, pos):
            return target

    w = Forced(kind, free, np.random.default_rng(1))
    st = Walk(start, 0.0, 0.0, (), 0.0)
    for _ in range(400):
        st = w.advance(st, 1.0)
    return math.dist(st.pos, target)


def test_a_resident_keeps_off_the_robots_sweep_to_the_hand_margin(monkeypatch):
    """Contract (`ROBOT_KEEP_M` 0.2 m): a step is not taken, and a walk that stands there is drawn again, when the
    footprint comes within 0.2 m of the robot's body swept between two keyframes. Hand case: the vacuum (a disc of
    0.17 m radius) drives 3 m east from (0, 0) between two keyframes; a resident of the model's box (0.336 x 0.560 m)
    standing at (1.5, d) facing east has its footprint's near edge d - 0.28 from the path, a gap of d - 0.45: d = 0.63
    leaves 0.18 m and the walk is refused (`None` after every attempt), d = 0.67 leaves 0.22 m and it stands (both
    keyframes' cameras are over 1.3 m away, so no other rule decides). Regression caught: a person the robot drives
    through between frames, or a margin measured from the keyframe poses only."""
    tl = replace(minimal_timeline(), ego=Ego([keyframe("vacuum", k, float(k), (3.0 * k, 0.0, 0.0), 0.0, task=0,
                                                       phase="docked") for k in range(2)],
                                             [Task(0, 0, 2, TaskPlan("hand", {}), ())], []))
    kind = people.person_kind(people_home.KIND)
    for d, stands in ((0.63, False), (0.67, True)):
        monkeypatch.setattr(Walker, "start", lambda self, keepout, d=d: Walk((1.5, d), 0.0, 1e9, (), 0.0))
        rng = np.random.default_rng(1)
        poses, tried = people._walk(tl, apartment(), people_home, kind, ROBOTS["vacuum"], 0, 2, [], rng)
        assert (poses is not None) == stands and tried == (1 if stands else people.ATTEMPTS), d


def test_people_keep_off_the_robot(always_home, monkeypatch):
    """Contract (`ROBOT_KEEP_M`): over the generated scripts of seeds 101-103 and the two three-task scripts, a resident
    home for every task keeps its footprint at every keyframe at least 0.2 m (less the 0.01 m of the outline's
    sampling) from the robot's body and from the body swept to the keyframe before and after, the distances written
    out from the placements and the vacuum's 0.17 m radius; and with the rule off one of them does come nearer (so the
    check can fail). Regression caught: a person the robot drives through between frames."""
    m, keep = apartment(), people_home.ROBOT_KEEP_M
    outs = [people.add(base, m, people_rng(101 + i)) for i, base in enumerate(resident_scripts())]
    near = [g for tl in outs for g in robot_gaps(tl)]
    assert len(near) > 100 and min(near) >= keep - 0.01
    monkeypatch.setattr(people_home, "ROBOT_KEEP_M", -1.0)
    off = [g for i, base in enumerate(resident_scripts()) for g in robot_gaps(people.add(base, m, people_rng(101 + i)))]
    assert min(off) < keep - 0.01


def test_people_come_and_go_unseen(monkeypatch):
    """Contract (`people.add`: a resident arrives and leaves at a whole second inside a gap, 30 s from both its ends,
    t = 0 for the first task): with a resident home for tasks 0 and 2 but not 1, its placements begin at t = 0 and at
    one whole second inside each of the two gaps (the arrival at task 2, and the removal that ends its task 0 pose
    during task 1's absence); every other placement of a task starts at a keyframe time after its first; and it exists
    at every keyframe of tasks 0 and 2 and at none of task 1's. A resident home for all three arrives once at t = 0 and
    jumps once inside each gap. Regression caught: a resident appearing in view, or one left standing through a task it
    is not home for."""
    m, tl = apartment(), multi_task(101)
    (g1, g2), edge = gaps(tl), 30.0
    for pattern in ([True, False, True], [True, True, True]):
        monkeypatch.setattr(people, "presence", lambda rng, n, p, pattern=pattern: pattern)
        out = people.add(tl, m, people_rng(101))
        eid = resident(out)
        marks = {t for p in out.placements_of(eid) for t in (p.t0, p.t1) if t not in (None, 0.0)}
        times = {kf.t for kf in out.ego.keyframes}
        in_gap = sorted(t for t in marks if t not in times)
        assert all(t == round(t) for t in in_gap), pattern
        assert [any(lo + edge <= t <= hi - edge for t in in_gap) for lo, hi in (g1, g2)] == [True, True], pattern
        assert len(in_gap) == 2 and all(any(lo + edge <= t <= hi - edge for lo, hi in (g1, g2)) for t in in_gap)
        assert out.placements_of(eid)[0].t0 == 0.0
        here = [[out.placement_of(eid, out.keyframe(k).t) is not None for k in range(t.k0, t.k1)]
                for t in out.ego.tasks]
        assert [all(h) for h in here] == pattern and [any(h) for h in here] == pattern, pattern
        firsts = {out.keyframe(t.k0).t for t in out.ego.tasks[1:]}
        assert not firsts & {p.t0 for p in out.placements_of(eid)}, pattern


def test_a_gap_with_no_moment_to_change_in_is_refused(always_home):
    """Contract (`people.add`): a resident must arrive or leave inside a gap, 30 s from both its ends, so a gap shorter
    than 60 s (here 47 s between the last keyframe of task 0 and the first of task 1) raises ValueError naming it,
    where a person would otherwise appear in view. Regression caught: numpy's own `low >= high` for the draw, or a
    resident that arrives inside the buffer."""
    tl = minimal_timeline()
    shifted = [replace(kf, t=kf.t - 7150.0) if kf.k >= 3 else kf for kf in tl.ego.keyframes]  # 3.0 s to 53.0 s
    short = replace(tl, ego=Ego(shifted, tl.ego.tasks, []))
    with pytest.raises(ValueError, match="gap before task 1"):
        people.add(short, apartment(), people_rng(1))


@pytest.mark.parametrize("mod", MODULES, ids=lambda mod: mod.__name__.removeprefix("stbench.people_"))
def test_presence_follows_its_rate(mod):
    """Contract (each people module's `P_HOME`: 0.5 the home's resident, 0.8 the warehouse's pickers, 0.7 the downtown's
    pedestrians): over 1,000 draws the share of tasks a person is there for is `P_HOME` within 4 standard errors,
    sqrt(p x (1 - p) / 1000) = 0.016, 0.013 and 0.014 (the draw alone, no walk). Regression caught: a person there in
    every task, so a blind guess learns it, or never."""
    share = sum(people.presence(np.random.default_rng(7), 1000, mod.P_HOME)) / 1000
    assert abs(share - mod.P_HOME) < 4 * math.sqrt(mod.P_HOME * (1 - mod.P_HOME) / 1000)


def test_a_person_is_detected_as_a_person(always_home, tmp_path, monkeypatch):
    """Contract (the oracle channel; docs/people.md): a resident seen in a frame is a detection of kind `person` with no
    attributes and no `state` key, the box of the model as its extent; the written script folder, perception file
    included, passes the leak scan. Hand case: the resident the overlay puts in `minimal_timeline`, in a frame at 500
    px. Regression caught: a colour or a name on a person (it would be described), or the truth reaching an agent
    through one."""
    m, tl = apartment(), minimal_timeline()
    tl = people.add(tl, m, people_rng(1))
    eid = resident(tl)
    assert eid == "E0006"
    cam = Camera.load(camera_doc(ROBOTS["helper"]))
    row = {"id": eid, "px": 500, "bbox": [100, 100, 140, 140], "depth_med_m": 2.0, "lum": 0.3,
           "centroid_cam": [0, 0, 2.0], "centroid_map": [0, 0, 0]}
    oracle, ids = P.build(tl, [{"k": 0, "entities": [row]}], ep.pose_rows(tl)[:1], cam)
    (det,) = oracle[0]["detections"]
    assert det["kind"] == "person" and det["attributes"] == {} and "state" not in det
    assert det["extent_m"] == pytest.approx(list(BOX_M), abs=1e-4) and list(ids[0]["ids"].values()) == [eid]
    paths = ep.write_script(tmp_path, tl)
    ep.write_jsonl(paths.visibility, [{"k": k, "entities": [row]} for k in range(tl.n_keyframes)])
    real = job_registry.get
    monkeypatch.setattr(job_registry, "get", lambda job_id: real("helper/1"))   # the fixture's job is no real one
    P.write(paths)
    leak_scan(paths.root, [tl.episode.map])


def test_a_residents_entity_and_kind_are_the_modules(always_home):
    """Contract (the seam with `validate_ego._kind_of`, the question side and the renderer): a resident is the entity
    `person`, a `mover` with no attributes, the module's box and asset, its id after every other entity's; and
    `person_kind` gives that name its box, speeds, dwell, room share and camera clearance in the fields of `MoverKind`
    (no colours) from the first catalog's module declaring it (the downtown's: catalogs in name order), and raises
    KeyError for any other name, a catalog's mover kind included. Regression caught: a speed range read as the dwell,
    a missing KeyError (an unknown mover kind passing validate), or a described person."""
    tl = people.add(minimal_timeline(), apartment(), people_rng(1))
    e = tl.entity(resident(tl))
    assert (e.kind, e.attributes, e.mobility, e.size, e.asset) == ("person", {}, "mover", BOX_M, people_home.MODEL)
    assert e.id > max(x.id for x in tl.entities if x.id != e.id)
    kind, first = people.person_kind("person"), people_downtown
    assert kind == MoverKind("person", first.BOX_M, (), first.SPEED_MPS, first.DWELL_S, first.SAME_ROOM, first.CAMERA_M)
    for name in ("robot_vacuum", "forklift", "pedestrian", ""):
        with pytest.raises(KeyError):
            people.person_kind(name)


@pytest.mark.parametrize("mod", MODULES, ids=lambda mod: mod.__name__.removeprefix("stbench.people_"))
def test_every_number_of_a_catalogs_people_names_its_source(mod):
    """Contract (each people module's `SOURCES`: each constant with its source or a modelling-choice note): every
    upper-case constant of the module has a non-empty note, and no note is for a constant that does not exist.
    Regression caught: a rate added with no source, or a note left behind by a renamed constant."""
    constants = {n for n in vars(mod) if n.isupper() and n != "SOURCES"}
    assert constants == set(mod.SOURCES) and all(mod.SOURCES.values())


def test_one_person_kind():
    """Contract (one person kind, several bodies: `person_kind` and `validate_ego._kind_of` read the kind with no
    catalog in hand, and the mover path draws each module's model at the module's box, scale 1): every people module
    declares the same `KIND` and `CAMERA_M` (what `validate` reads of a kind, from whichever module comes first), and
    each module's `BOX_M` equals the manifest's `native_bbox_m` of its own `MODEL` within 0.01 m. Regression caught: a
    body drawn stretched, a footprint that is not its body's, or `validate` reading another catalog's camera rule."""
    assert len({mod.KIND for mod in MODULES}) == 1 and len({mod.CAMERA_M for mod in MODULES}) == 1
    for mod in MODULES:
        entry = asset_manifest.packaged().asset(mod.MODEL)
        assert list(entry.native_bbox_m) == pytest.approx(list(mod.BOX_M), abs=0.01), mod.__name__
    assert people_home.MODEL != people_warehouse.MODEL == people_downtown.MODEL  # the home's body is barefoot


def digest(seed: int = 101) -> str:
    """The digest of the overlay's timeline over `minimal_timeline` (two tasks, a gap between them), a resident home for
    every task."""
    return T.digest(people.add(minimal_timeline(), apartment(), people_rng(seed)))


def test_people_are_deterministic(always_home):
    """Contract: the same script and rng give a byte-identical timeline, in this process and in another with another
    `PYTHONHASHSEED`, and another rng gives another. Regression caught: an unseeded draw or a set's iteration order in
    the obstacles or the walk."""
    env = {**os.environ, "PYTHONHASHSEED": "4242"}
    out = subprocess.run([sys.executable, __file__], capture_output=True, text=True, check=True, env=env).stdout.strip()
    assert digest() == out
    assert digest(102) != out


def test_a_catalog_without_people_is_unchanged():
    """Contract (`people.add`: a catalog with no `people_<catalog>` module has no people): the warehouse's and the
    downtown's scripts, their maps naming a catalog no module belongs to, come back with the same world and ego and no
    person entity (every real catalog has its people now). Regression caught: people where no module declares any."""
    scripts = (("warehouse_a", "amr", (8.0, 3.0, 0.0, 0.0)), ("downtown_a", "drone", (40.0, 13.5, 12.0, 0.0)))
    for map_name, robot, pose in scripts:
        tl = script_on_map(map_name, 1, robot, [pose])
        out = people.add(tl, replace(load_map(map_name), catalog="nowhere"), people_rng(1))
        assert (out.world_hash, out.ego_hash) == (tl.world_hash, tl.ego_hash) and resident(out) is None, map_name


if __name__ == "__main__":
    people_home.P_HOME = 1.0
    print(digest())
