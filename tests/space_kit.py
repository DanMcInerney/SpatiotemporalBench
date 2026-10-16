"""Real geometry for the free-space and compiler tests: a depth image ray-cast from boxes, and scripts that carry it.

`synth.py`'s depth is a flat 6 m image, which is not geometry: under the floor it would certify every far cell free.
`render_depth` ray-casts each pixel of the robot's own camera against the floor plane and axis-aligned boxes, as the
renderer's depth pass does (planar depth, 0 outside the camera's `near_m` to `far_m`), so a test's expected cells are
worked out from the scene by hand. `SpaceScript` is a `synth.Synth` whose keyframes may face any way and whose depth
images are that render; everything else (entities, sightings, decision points, the case, the oracle channel) is
`Synth`'s. A keyframe outside `depth_ks` has no depth at all (every pixel 0), which certifies nothing.

A box is `box(x0, x1, y0, y1, z0, z1)`: its two corners in the map frame, metres.

For the compiler: `JOB` is a job of its own (its own event wordings and relevance kinds, so no test leans on
`vacuum/1`), `vacuum_run` a one-task script of the vacuum with decision points and the phases a task has, `inputs` and
`record` give the compiler its inputs in memory (no files, fixed case id), and `use_job` points `job_registry.get` at a
job for a test that compiles from files.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from builders import keyframe
from stbench import episode as ep
from stbench import grid, job_registry
from stbench.camera import Camera, cam_to_map
from stbench.case import CaseRecord
from stbench.events import EventRule
from stbench.job_registry import JobSpec, Relevance
from stbench.nearest_part import floor_hull
from stbench.questions_world import Inputs
from stbench.recognise import recognisable
from stbench.robots import ROBOTS
from stbench.timeline import Timeline
from synth import START, Synth, camera

Box = tuple[tuple[float, float, float], tuple[float, float, float]]
NO_DEPTH = np.zeros((480, 640), dtype=np.uint16)


def box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> Box:
    return (x0, y0, z0), (x1, y1, z1)


def render_depth(cam: Camera, cam_pos, cam_rot, boxes: list[Box], floor_z: float = 0.0) -> np.ndarray:
    """The planar depth image (uint16 counts of `cam.depth_unit`) of the floor plane at `floor_z` and the boxes, seen
    from a camera at `cam_pos` with orientation `cam_rot`: each pixel centre's ray, the nearest surface it meets, 0
    where it meets none or the surface is nearer than `cam.near_m` or farther than `cam.far_m`."""
    j, i = np.meshgrid(np.arange(cam.width) + 0.5, np.arange(cam.height) + 0.5)
    rays_cam = np.stack([(j - cam.cx) / cam.fx, (i - cam.cy) / cam.fy, np.ones_like(j)], axis=-1).reshape(-1, 3)
    ray = cam_to_map(rays_cam, (0.0, 0.0, 0.0), cam_rot)  # camera z is 1, so the ray parameter is the planar depth
    origin = np.asarray(cam_pos, dtype=float)
    down = ray[:, 2] < 0
    t = np.where(down, (floor_z - origin[2]) / np.where(down, ray[:, 2], -1.0), np.inf)
    safe = np.where(np.abs(ray) < 1e-12, 1e-12, ray)
    for lo, hi in boxes:
        a, b = (np.asarray(lo) - origin) / safe, (np.asarray(hi) - origin) / safe
        enter, leave = np.minimum(a, b).max(axis=1), np.maximum(a, b).min(axis=1)
        t = np.minimum(t, np.where((leave >= enter) & (enter > 0), enter, np.inf))
    metres = np.where(np.isfinite(t), t, 0.0)
    ok = (metres >= cam.near_m) & (metres <= cam.far_m)
    return np.where(ok, np.rint(metres / cam.depth_scale), 0).astype(np.uint16).reshape(cam.height, cam.width)


@dataclass
class SpaceScript(Synth):
    """A `Synth` with hand-set headings and ray-cast depth. `yaws` sets a keyframe's heading in radians (the default is
    `Synth`'s: 45 degrees a frame of a leg), `poses` its base's (x, y) as in `Synth`; `boxes` stand in the scene for
    every rendered keyframe, `frame_boxes` for one more; `depth_ks` names the keyframes with a render (None: all);
    `parts` holds the map-frame points of a thing's pixels at a keyframe, which `inputs` hands the compiler as
    `Inputs.points_of` (no entry: the thing has no pixel there), `gaps` how many of its pixels have no depth
    (`Inputs.gaps_of`; no entry: none)."""

    yaws: dict[int, float] = field(default_factory=dict)
    boxes: list[Box] = field(default_factory=list)
    frame_boxes: dict[int, list[Box]] = field(default_factory=dict)
    depth_ks: set[int] | None = None
    parts: dict[tuple[int, str], np.ndarray] = field(default_factory=dict)   # (keyframe, entity) -> its (N, 3) points
    gaps: dict[tuple[int, str], int] = field(default_factory=dict)   # (keyframe, entity) -> its pixels with no depth

    def timeline(self) -> Timeline:
        tl = super().timeline()
        frames = [keyframe(self.stage.robot, kf.k, kf.t, kf.base, self.yaws.get(kf.k, kf.yaw), task=kf.task,
                           phase=kf.phase, status=kf.status) for kf in tl.ego.keyframes]
        return replace(tl, ego=replace(tl.ego, keyframes=frames))

    def cameras(self) -> dict[int, tuple]:
        """Each keyframe's camera pose (position, rotation), as the pose rows will hold it."""
        return {kf.k: (kf.pos, kf.rot) for kf in self.timeline().ego.keyframes}

    def depth_of(self, k: int) -> np.ndarray:
        if self.depth_ks is not None and k not in self.depth_ks:
            return NO_DEPTH
        pos, rot = (self._cameras or self.cameras())[k]
        return render_depth(camera(self.stage.depth_unit), pos, rot, self.boxes + self.frame_boxes.get(k, []),
                            self.stage.base_z)

    def oracle(self, tl):
        """`Synth.oracle` with each detection's `floor_hull`: the hull of the points `parts` holds for the
        thing at that keyframe when the thing is recognisable there, else null."""
        rows, ids = super().oracle(tl)
        seen = {(r["k"], e["id"]): e for r in self.visibility() for e in r["entities"]}
        cam = camera(self.stage.depth_unit)
        for row, key in zip(rows, ids, strict=True):
            for d in row["detections"]:
                eid, pts = key["ids"][d["d"]], self.parts.get((row["k"], key["ids"][d["d"]]))
                ok = pts is not None and len(pts) and recognisable(seen[row["k"], eid], cam, eid)
                d["floor_hull"] = floor_hull(pts) if ok else None
        return rows, ids

    _cameras = None  # the poses of one `write`, so its renders do not rebuild the timeline for every keyframe

    def write(self, dataset: Path, perception: bool = True):
        self._cameras = self.cameras()
        try:
            return super().write(dataset, perception)
        finally:
            self._cameras = None


# -- a job and a run of it for the compiler ---------------------------------------------------------------------
EVENTS = {"when you set off": EventRule("last", "undocking"), "when you began sweeping": EventRule("first", "cleaning"),
          "when you finished sweeping": EventRule("last", "cleaning"),
          "when you got home": EventRule("first", "docked", after_phase="docking")}
RELEVANT_KINDS = frozenset({"bag", "shoes", "toy", "box"})
JOB = JobSpec(
    id="vacuum/1", robot="vacuum", domain="home", task_kinds=("clean_room",),
    decisions={"before_room": (), "before_doorway": ("path_ahead",), "room_done": ()}, events=EVENTS,
    plan=lambda *a: [], script=lambda *a, **kw: None,
    relevance=lambda plan: Relevance(RELEVANT_KINDS, frozenset({"kitchen", "living room"}),
                                     frozenset({"cleaning", "transit", "docking"})),
    checks=lambda tl, m: [], anchors=frozenset(), assignments={}, swaps=())

# One task of 28 keyframes: docked (k 0), undocking 1-2, transit 3-6, cleaning 7-18, transit 19-22, docking 23-25,
# docked 26-27. `JOB`'s events resolve to keyframes 2, 7, 18 and 26. The stream is cut after the decision points at 5,
# 18 and 21 and after the task's last step: look windows 0-5, 6-13, 14-18, 19-21 and 22-27. Only the point at 21 asks a
# family (`path_ahead`, a `now` one, which the questionnaire asks from `NOW_MIN_STEPS` into a task): the point at 5 is
# a `before_room`, which `JOB` leaves unasked, so a run asks one question during the task.
LEGS = [("kitchen", 3, "undocking"), ("kitchen", 4, "transit"), ("living room", 12, "cleaning"),
        ("living room", 4, "transit"), ("kitchen", 3, "docking"), ("kitchen", 2, "docked")]
DECISIONS = [(5, "before_room"), (18, "room_done"), (21, "before_doorway")]
EVENT_KS = {"when you set off": 2, "when you began sweeping": 7, "when you finished sweeping": 18,
            "when you got home": 26}
FRONT_EDGE_ORIGIN = (-0.17, 0.0)  # the vacuum's base 0.17 m behind x = 0: its front edge is at x 0


def vacuum_run(n_tasks: int = 1, *, seed: int = 1, **fields) -> SpaceScript:
    """`n_tasks` copies of `LEGS`, each with `DECISIONS` (script keyframes shifted by 28 a task). Keyframes 19 to 21 of
    each task, the decision's look window, stand with the front edge at x = 0 heading +x and have a render, a wall's
    front face 2.21 m ahead (so `path_ahead` gives 2.2: its sub-cell centre 2.21 less 0.01); the other keyframes have no
    depth. `fields` replace `SpaceScript`'s (`boxes`, `depth_ks`, `frame_boxes`, `poses`, `yaws`, `decisions` ...)."""
    n = sum(leg[1] for leg in LEGS)
    window = [t * n + k for t in range(n_tasks) for k in (19, 20, 21)]
    setup = {"tasks": [list(LEGS) for _ in range(n_tasks)], "seed": seed,
             "decisions": [(t * n + k, kind) for t in range(n_tasks) for k, kind in DECISIONS],
             "poses": dict.fromkeys(window, FRONT_EDGE_ORIGIN), "yaws": dict.fromkeys(window, 0.0),
             "boxes": [box(2.21, 2.43, -3.0, 3.0, 0.0, 2.0)], "depth_ks": set(window)}
    return SpaceScript(**{**setup, **fields})


BAG_SIZE = (0.374, 0.304, 0.314)  # the catalog's bag: a large thing


def things_run(**fields) -> SpaceScript:
    """`vacuum_run` with things of every fate (E-ids, as the real ids are). E0001 a red bag that moves at t 7.5 s
    (keyframe 5) by 0.9 m, under `STALE_M` (1 m), so `object_where` asks it, recognisable at keyframes 1 (at (1.1, 1))
    and 10 (at (2, 1)); E0002 black shoes at 500 px only; E0003 a
    toy dark (lum 0.005); E0004 a brown box recognisable only from keyframe 12; E0005 a blue bag never seen; E0006 a
    white box that exists only from t 30 s (keyframe 20); E0007 green shoes recognisable at 3 and again at 24, in the
    end look window; E0008 a mug and E0009 a chair (never asked); E0010 and E0011 two identical purple toys, both seen
    (so not unique; not a neutral, which would hide the white box: black, grey and white are one group)."""
    s = vacuum_run(**fields)
    s.item("E0001", "bag", {"color": "red"}, [(0, 7.5, (1.1, 1.0, 0.0)), (7.5, None, (2.0, 1.0, 0.0))], size=BAG_SIZE)
    s.see([1, 10], "E0001", 1500)
    s.item("E0002", "shoes", {"color": "black"}, [(0, None, (1.5, 2.0, 0.0))])
    s.see(3, "E0002", 500)
    s.item("E0003", "toy", {"color": "red"}, [(0, None, (3.5, 2.0, 0.0))])
    s.see(3, "E0003", 1500, lum=0.005)
    s.item("E0004", "box", {"color": "brown"}, [(0, None, (3.5, 1.0, 0.0))])
    s.see(12, "E0004", 1500)
    s.item("E0005", "bag", {"color": "blue"}, [(0, None, (5.5, 1.0, 0.0))])
    s.item("E0006", "box", {"color": "white"}, [(30.0, None, (5.5, 2.0, 0.0))])
    s.item("E0007", "shoes", {"color": "green"}, [(0, None, (6.5, 2.0, 0.0))])
    s.see([3, 24], "E0007", 1500)
    s.item("E0008", "mug", {"color": "red"}, [(0, None, (1.0, 2.5, 0.8))])
    s.see(1, "E0008", 1500)
    s.item("E0009", "chair", {"color": "oak"}, [(0, None, (2.0, 2.5, 0.0))], mobility="light", size=(0.4, 0.4, 0.9))
    s.see(1, "E0009", 1500)
    for eid in ("E0010", "E0011"):
        s.item(eid, "toy", {"color": "purple"}, [(0, None, (4.0, 1.0, 0.0))])
        s.see(1, eid, 1500)
    return s


def inputs(script: SpaceScript, job: JobSpec = JOB) -> Inputs:
    """The compiler's inputs for the script, in memory: its timeline, pose rows and visibility rows, and free-space
    cells from its own ray-cast depth through the pose rows' cameras (what `freespace.script_states` does from
    files)."""
    tl, cam, robot = script.timeline(), camera(script.stage.depth_unit), ROBOTS[script.stage.robot]
    cameras, band, done = script.cameras(), grid.band(robot), {}

    def states_of(ks):  # each keyframe is rendered once per `inputs`, whoever asks
        for k in ks:
            if k not in done:
                done[k] = grid.observe(grid.points(script.depth_of(k), cam, *cameras[k]), tl.keyframe(k).base[2], band)
        return {k: done[k] for k in ks}

    def points_of(k, entity):
        return script.parts.get((k, entity), np.empty((0, 3)))

    return Inputs(tl, cam, robot, job, ep.pose_rows(tl), script.visibility(), states_of, None, points_of,
                  lambda k, entity: script.gaps.get((k, entity), 0))


def record(script: SpaceScript, a: int = 0, b: int | None = None, cid: str = "c0", length: str = "short") -> CaseRecord:
    """A case over the script's tasks a..b with a fixed id, so draws that hash the case id repeat."""
    tl, b = script.timeline(), a if b is None else b
    k0, k1 = tl.ego.tasks[a].k0, tl.ego.tasks[b].k1
    return CaseRecord(cid, "s0", a, b, k0, k1, tl.keyframe(k0).t, START, length)


def use_job(monkeypatch, job: JobSpec = JOB) -> None:
    """Every `job_registry.get` of this test returns `job`, so a compile from files reads the job it was built for."""
    monkeypatch.setattr(job_registry, "get", lambda job_id: job)
