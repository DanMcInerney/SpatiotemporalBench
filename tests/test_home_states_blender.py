"""The apartment's look under M2's world processes, in Blender (lane M2-R-home; `render/build_scene.py`'s `Door` and
`Scene.apply_time`, `render/dressing_layout_home.py` and the modules under it): every interior door shows the state the
ledger gives it, and the things that move between tasks are drawn where the timeline puts them and never into the
render-only dressing. One hand-made two-task `vacuum` timeline of seed 101's world, written here and not drawn from the
generator, so that no change of the clutter or the light furniture's draws can move it: D1-D4 shut in task 0 and open
again after the gap, and a floor clutter piece, a tracked thing and a chair each in one place in task 0 and in another
in task 1. It is rendered once (fast: the placeholder drawings, which are the same geometry, at 8 samples) and its scene
built once more, with the dressing, by `blender_home_states_driver.py`. Needs Blender; marked `blender`, skipped without
it. Nothing is downloaded. About twenty seconds. Where the dressing stands and what it keeps clear of is
`test_dressing_zones.py`'s and `test_dressing_layout_home.py`'s, and what the home's finishes are is
`test_wear_home_blender.py`'s."""

from __future__ import annotations

import json
import math
from collections import namedtuple
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from builders import keyframe, start_status
from stbench import blender_run, job_registry
from stbench import episode as ep
from stbench import visibility as V
from stbench.camera import Camera, cam_to_map, unproject
from stbench.catalog import CATALOGS
from stbench.generate import build_world, to_timeline
from stbench.geometry import Rect, dist_to_rect
from stbench.mapspec import load_map
from stbench.render_cli import blender_path, render
from stbench.render_encode import read_png16
from stbench.robots import ROBOTS
from stbench.timeline import Ego, Entity, EpisodeInfo, Task, TaskPlan
from stbench.validate import door_entities, door_leaf_rect

# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
REPO = Path(__file__).resolve().parents[1]
MAP, SEED = "apartment_a", 101
MOVE_T, TASK1_T = 3600.0, 7200.0  # the instant the world changes, and when the second task begins
DOORS = ("D1", "D2", "D3", "D4")
# Where the vacuum stands in the hall to face each interior door, 0.9 m off its wall (x, y, yaw in degrees): the map's
# doors are in the walls x = 4.6 (D1 at y 1.9 and D2 at 3.9, seen facing west) and x = 6.2 (D3 at 1.3 and D4 at 3.3,
# seen facing east); the hall is the strip x 4.65 to 6.15.
DOOR_VIEWS = [(5.5, 1.9, 180.0), (5.5, 3.9, 180.0), (5.3, 1.3, 0.0), (5.3, 3.3, 0.0)]
# The three things that move (x, y, yaw in degrees in task 0 and in task 1), by hand from the map's footprints: each
# lies on the living room's open floor, clear of the coffee table (x 1.85 to 2.95, y 4.1 to 4.7), the TV console (x to
# 0.45), the sofa (y from 5.25), the wall at y = 2.85, the arch and the swing of D2 (x from 3.7, y 3.45 to 4.35), and at
# least 0.12 m from every wall. The vacuum looks at each from 0.3 to 0.5 m, in both tasks, from the floor beside it.
THINGS = ("chair", "purifier", "box")
MOVES = {"chair": ((3.6, 4.9, 180.0), (1.5, 3.25, 90.0)),  # the map's L03 at its map place, then pulled aside
         "purifier": ((1.0, 3.2, 0.0), (3.3, 3.2, 0.0)),  # floor clutter, a kind of the household's catalog
         "box": ((1.7, 3.2, 30.0), (2.4, 3.3, 100.0))}  # a tracked thing, left on the floor
THING_VIEWS = {"chair": ((3.6, 4.1, 90.0), (1.5, 4.05, -90.0)),
               "purifier": ((1.0, 4.0, -90.0), (3.3, 4.0, -90.0)),
               "box": ((1.7, 4.0, -90.0), (2.4, 3.93, -90.0))}
PER_TASK = len(DOORS) + len(THINGS)  # keyframes a task: the four doors, then the three things
FLUSH_M = 0.02  # the flush floor band: grime a thing stands on is no collision
MEET_M = 1e-4  # boxes meet when they share more than this on every axis


@cache
def apartment():
    return load_map(MAP)


def hand_made():
    """The timeline (see the module's docstring) and the entity id of each thing that moves."""
    m, r = apartment(), ROBOTS["vacuum"]
    w = build_world(m, SEED)
    chair = next(e for e in w.entities
                 if e.kind == "chair" and abs(w.light_rects[e.id].cx - MOVES["chair"][0][0]) < 0.01)
    things = {"chair": chair,
              "purifier": Entity(w.new_entity_id(), "air_purifier", {"color": "grey"}, "clutter", (0.25, 0.25, 0.55)),
              "box": Entity(w.new_entity_id(), "box", {"color": "red"}, "carried", CATALOGS["home"].kinds["box"].size)}
    w.entities += [things["purifier"], things["box"]]
    for name, e in things.items():
        for t, (x, y, yaw) in zip((0.0, MOVE_T), MOVES[name], strict=True):
            w.ledger.place(e.id, t, (x, y, 0.0), math.radians(yaw), None)
    for door in DOORS:  # shut for the first task, open again after the gap
        w.ledger.set_state(w.door_entity[door], "open", 0.0, False)
        w.ledger.set_state(w.door_entity[door], "open", MOVE_T, True)
    kfs, tasks = [], []
    for task in (0, 1):
        views = DOOR_VIEWS + [THING_VIEWS[name][task] for name in THINGS]
        k0 = len(kfs)
        kfs += [keyframe("vacuum", k0 + j, task * TASK1_T + float(j), (x, y, 0.0), math.radians(yaw), task=task,
                         phase=r.status.start["phase"], status=start_status("vacuum"))
                for j, (x, y, yaw) in enumerate(views)]
        tasks.append(Task(task, k0, len(kfs), TaskPlan(job_registry.get("vacuum/1").task_kinds[0], {"room": "hall"}),
                          ()))
    info = EpisodeInfo(m.name, SEED, "vacuum", "vacuum/1", "2026-03-02T08:00:00")
    return to_timeline(m, w, Ego(kfs, tasks, []), info), {name: e.id for name, e in things.items()}


States = namedtuple("States", "tl paths things drawn")


@pytest.fixture(scope="module")
def states(tmp_path_factory) -> States:
    """The timeline written and rendered once (fast, every keyframe), and its scene built once with the dressing."""
    tl, things = hand_made()
    paths = ep.write_script(tmp_path_factory.mktemp("home_states"), tl)
    render(paths, fast=True)
    out = paths.root / "drawn.json"
    blender_run.run([blender_path(), "--background", "--factory-startup", "--python-exit-code", "1", "--python",
                     str(REPO / "tests" / "blender_home_states_driver.py"), "--", "--render", str(REPO / "render"),
                     "--src", str(REPO / "src"), "--private", str(paths.private), "--times", f"0,{TASK1_T}", "--out",
                     str(out)], "building the apartment's scene with its dressing")
    return States(tl, paths, things, json.loads(out.read_text(encoding="utf-8")))


def frame_points(s: States, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Every pixel of keyframe `k` that has a depth, unprojected through it and the keyframe's pose to the map frame:
    the points (x, y, z in rows) and the entity id of the instance each belongs to ("" for none)."""
    cam, imap = Camera.load(s.paths.camera), V.load_instance_map(s.paths.instance_map)
    pose = V.load_pose(s.paths.pose)[k]["camera"]
    index, depth = read_png16(s.paths.instances(k)), read_png16(s.paths.depth(k)).astype(float) * cam.depth_scale
    v, u = np.nonzero(depth > 0)
    pts = cam_to_map(unproject(u + 0.5, v + 0.5, depth[v, u], cam).T, pose["pos"], pose["rot"])
    return pts, np.array([imap.get(int(i), "") for i in index[v, u]])


def in_doorway(pts: np.ndarray, opening) -> np.ndarray:
    """Which of the points (x, y, z in rows) lie in the opening's doorway: across the middle 60% of its width, 5 cm
    either side of its wall's centre line and 0.15 to 1.8 m up (the frame's own axes, from the opening's centre and the
    way its leaf runs shut)."""
    hinge, tip = apartment().door_leaf(opening, False)
    along = np.array([tip[0] - hinge[0], tip[1] - hinge[1]]) / opening.width
    rel = pts[:, :2] - np.array(opening.center)
    return ((np.abs(rel @ along) <= 0.3 * opening.width) & (np.abs(rel @ np.array([-along[1], along[0]])) <= 0.05)
            & (pts[:, 2] >= 0.15) & (pts[:, 2] <= 1.8))


def test_a_closed_door_fills_its_doorway(states):
    """Contract (`build_scene.Door`, `Scene.apply_time`: a door leaf is drawn in the state the ledger gives it at the
    keyframe's time): in task 0 D1-D4 are shut (a ledger state set at t = 0), and each of the four frames facing one
    from the hall shows the leaf: at least 1,500 of the pixels in the middle 60% of the doorway (0.15 to 1.8 m up and
    5 cm either side of the wall's centre line) are the door's own instance and no pixel there is anything else, and
    every pixel of that instance, unprojected through the depth, lies within 0.1 m of the shut leaf
    (`validate.door_leaf_rect`) and under its top. In task 1 the same four doors are open again and the same four poses
    show next to nothing across the doorway (at most 15 pixels). Regression caught: a door that shows open while the
    truth says shut (the leaf at its old angle, or the first state kept for the whole script), one that shows shut once
    it is open, or a leaf drawn at the wrong hinge."""
    m, tl = apartment(), states.tl
    for i, door_id in enumerate(DOORS):
        opening, eid = next(o for o in m.doors() if o.id == door_id), door_entities(tl, m)[door_id]
        shut_k, open_k = i, PER_TASK + i
        assert tl.state_of(eid, tl.keyframe(shut_k).t, "open") is False
        assert tl.state_of(eid, tl.keyframe(open_k).t, "open") is True
        pts, owner = frame_points(states, shut_k)
        there = owner[in_doorway(pts, opening)]
        assert (there == eid).sum() >= 1500, f"{door_id}: {(there == eid).sum()} doorway pixels are the door's"
        assert (there != eid).sum() == 0, f"{door_id}: {(there != eid).sum()} doorway pixels are something else"
        leaf, mine = door_leaf_rect(m, opening, False, tl.entity(eid).size[1]), pts[owner == eid]
        far = dist_to_rect(mine[:, 0], mine[:, 1], leaf)
        assert far.max() <= 0.1 and mine[:, 2].max() <= opening.height + 0.1, (door_id, far.max(), mine[:, 2].max())
        pts, _ = frame_points(states, open_k)
        assert in_doorway(pts, opening).sum() <= 15, f"{door_id} shows a leaf across it when open"


def box_meets(a: list[float], b: list[float]) -> bool:
    """Whether two boxes (xmin, ymin, zmin, xmax, ymax, zmax) share more than `MEET_M` on every axis."""
    return all(a[i] < b[i + 3] - MEET_M and b[i] < a[i + 3] - MEET_M for i in range(3))


def test_no_dressing_meets_a_leaf_in_either_state(states):
    """Contract (`dressing_zones.door_zones`, held in the scene: the whole quarter square a leaf sweeps, to the door's
    height, is clear of dressing, and an opening's casing stands beside it): in the built home scene, with all of its
    dressing (over 200 pieces, 245 today), no dressing object's box in the map frame shares more than 0.1 mm on every
    axis with the box of any interior door's leaf slab (its width, 4 cm and its height, laid out by the scene's
    transform), turned open or turned shut. The knobs are not in the box (two 6 cm spheres would swell the box of an
    open leaf by 6 cm on each face, over a jamb's casing that stands 2 mm from the leaf), and the flush plates of grime
    on the floor (at most 2 cm thick: `FLUSH_M`) are left out: a leaf stands on them. Regression caught: a door stop or
    a fitting drawn through a closing door, a casing, a switch plate or a curtain inside the swing, or a leaf wider than
    its opening."""
    doors = [d for d in states.drawn["doors"] if d["interior"]]
    solid = [p for p in states.drawn["dressing"] if p["box"][5] > FLUSH_M]
    assert len(states.drawn["dressing"]) > 200 and len(doors) == len(DOORS) and len(solid) > 150  # it is built, whole
    for door in doors:
        for state in ("open", "shut"):
            hit = [p["name"] for p in solid if box_meets(p["box"], door[state])]
            assert hit == [], f"{door['id']} turned {state} meets {hit}"


def test_things_render_where_they_moved(states):
    """Contract (`Scene.apply_time`: an entity is drawn where its placement at the keyframe's time puts it): a floor
    clutter piece (an air purifier), a tracked thing (a box left on the floor) and a chair (the map's L03) each stand in
    one place in task 0 and in another, 0.7 to 2.7 m away, in task 1; the frame that looks at each shows it (215 pixels
    or more) in both tasks, and its pixels, unprojected through the depth, lie within 0.1 m of its footprint in the
    timeline at that frame's time (at least 98% of them: the pixels at an edge may blend). In the scene with the
    dressing, at either time, no dressing object's box shares more than 0.1 mm on every axis with a thing's, the
    flush plates of grime aside. Regression caught: a thing drawn at its start after it moved (a stale transform, a
    `placement_of` that reads the first interval), one that disappears in the gap, or one moved into dressing."""
    tl, cam = states.tl, Camera.load(states.paths.camera)
    for slot, name in enumerate(THINGS):
        eid = states.things[name]
        for task in (0, 1):
            k = task * PER_TASK + len(DOORS) + slot
            pts, owner = frame_points(states, k)
            mine = pts[owner == eid]
            assert V.status(len(mine), cam) == "seen", f"the {name} is not seen in task {task}: {len(mine)} pixels"
            p, e = tl.placement_of(eid, tl.keyframe(k).t), tl.entity(eid)
            near = dist_to_rect(mine[:, 0], mine[:, 1], Rect(p.pos[0], p.pos[1], e.size[0], e.size[1], p.yaw)) <= 0.1
            assert near.mean() >= 0.98, f"the {name} in task {task}: {(~near).sum()} of {len(mine)} pixels are off"
        for t in (0.0, TASK1_T):
            box = states.drawn["entities"][eid]["boxes"][str(t)]
            assert box is not None, f"the {name} is not drawn at {t}"
            hit = [d["name"] for d in states.drawn["dressing"] if d["box"][5] > FLUSH_M and box_meets(d["box"], box)]
            assert hit == [], f"the {name} at {t} meets {hit}"
