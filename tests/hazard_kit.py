"""What the `drop_zone_people` and `tangle_ahead` tests share: a hand drone that looks straight down from the high
street and `ego_kit`'s vacuum at the look point, with the coverage of the free-space grid set by hand.

**The drone** (`DroneScript`) hovers over (2.0, 1.5), the middle of the stage's first room, `Stage.base_z` up (12 m: its
camera hangs 0.1 m below, so it is at 11.9 m), every keyframe in one phase (`phase`; the first keyframe of a task is the
start phase), its camera pitched `pitch` (nadir by default). From 11.9 m the nadir view covers +-9.98 m east-west and
+-7.49 m north-south (fx = fy = 381.36, 640 x 480), so a disc of radius 4 m lies in it; at 3.9 m it covers +-2.45 m
north-south and does not. `ZONE_JOB` is the downtown's numbers: the clearance 2.4 m and the place unit 1.0 m, so a match
radius of 2.0 m. A person is `person_kit`'s 0.34 x 0.56 m footprint.

**The vacuum** is `ego_kit.scene`: at the look point `K` = 13 its base is (3.0, 1.0) heading north, its camera floor
point (3.0, 1.13) and its front edge (3.0, 1.17). `ego_kit.put` adds things; `covering` gives a world whose free-space
grid observed exactly the cells the test names.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import family_kit as fk
from builders import keyframe
from ego_kit import EGO_JOB
from person_kit import PeopleScript
from space_kit import SpaceScript, record
from stbench import episode as ep
from stbench.questions_world import World
from stbench.table import Store
from synth import DOWNTOWN, Synth

# C = 2.4 m and a match radius of 2.0 m, asked in the drone's phases over the drop point
ZONE_JOB = replace(EGO_JOB, robot="drone", clearance=2.4, ladder=(1.0, 2.0, 4.0, 8.0), looks=("drop_zone_people",),
                   look_phases={"drop_zone_people": frozenset({"inspect", "descend", "lower"})})
# the vacuum's, asked in the phases of its brush and of its drive to the next strip
TANGLE_JOB = replace(EGO_JOB, looks=("tangle_ahead",),
                     look_phases={"tangle_ahead": frozenset({"cleaning", "transit"})})
CENTRE = (2.0, 1.5)   # the camera's floor point of the drone, the middle of the stage's first room
LOOK = 8              # the drone's look point: 8 keyframes in


@dataclass
class DroneScript(PeopleScript):
    """A hand drone: one task of `n` keyframes over `CENTRE` in `phase` (but the first, the start phase), the camera at
    `pitch` radians (nadir by default). Its oracle channel gives a person the evidence rule's hull at C = 2.4 m."""

    pitch: float = -math.pi / 2
    c_m: float = ZONE_JOB.clearance

    def timeline(self):
        tl = Synth.timeline(self)
        frames = [keyframe(self.stage.robot, kf.k, kf.t, kf.base, self.yaws.get(kf.k, kf.yaw), task=kf.task,
                           phase=kf.phase, status=kf.status, pitch=self.pitch) for kf in tl.ego.keyframes]
        return replace(tl, ego=replace(tl.ego, keyframes=frames))


def drone(phase: str = "lower", n: int = 12, base_z: float = 12.0, pitch: float = -math.pi / 2) -> DroneScript:
    """The hand drone `n` keyframes over `CENTRE` in `phase`, `base_z` up, its camera at `pitch`."""
    return DroneScript(tasks=[[("high street", n, phase)]], stage=replace(DOWNTOWN, base_z=base_z), pitch=pitch)


def zone_world(s: SpaceScript, k: int = LOOK) -> World:
    """The window of `s`'s first task ended at keyframe `k`, as the downtown drone's job reads it."""
    return World(fk.inputs(s, ZONE_JOB), replace(record(s), k1=k + 1))


def tangle_world(s: SpaceScript, k: int = 27) -> World:
    """The window of `s`'s first task ended at keyframe `k` (27, the task's last, by default), as the vacuum's job reads
    it."""
    return World(fk.inputs(s, TANGLE_JOB), replace(record(s), k1=k + 1))


def covering(w: World, cells_at: Callable[[int], set[tuple[int, int]]]) -> World:
    """`w` whose free-space grid observed, at keyframe `k`, exactly the fine cells `cells_at(k)` (all `free`)."""
    def states_of(ks, floor_z=None, band=None):
        return {k: dict.fromkeys(cells_at(k), "free") for k in ks}
    return World(replace(w.inp, states_of=states_of), w.rec)


def disc(centre: tuple[float, float], radius: float) -> set[tuple[int, int]]:
    """The fine cells (0.1 m) whose centres lie within `radius` of `centre`, written out here."""
    cells = set()
    for u in range(math.floor((centre[0] - radius) / 0.1) - 1, math.floor((centre[0] + radius) / 0.1) + 2):
        for v in range(math.floor((centre[1] - radius) / 0.1) - 1, math.floor((centre[1] + radius) / 0.1) + 2):
            if math.hypot((u + 0.5) * 0.1 - centre[0], (v + 0.5) * 0.1 - centre[1]) <= radius:
                cells.add((u, v))
    return cells


def store_of(s: SpaceScript, upto: int) -> Store:
    """The oracle's memory table (`table.Store` with perfect identity) fed the script's steps up to keyframe `upto`:
    each delivered pose row with the oracle channel's detections of the keyframe and their entities. No stream is
    run."""
    tl = s.timeline()
    poses = ep.pose_rows(tl)
    rows, ids = s.oracle(tl)
    store = Store()
    store.begin_task(0, poses[0]["clock"], 0.0, "the task")
    for pose, row, key in zip(poses[:upto + 1], rows, ids, strict=False):
        store.add_step({**pose, "detections": row["detections"], "identity": key["ids"]})
    return store
