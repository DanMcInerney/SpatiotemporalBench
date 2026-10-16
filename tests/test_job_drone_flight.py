"""`drone/1`'s flight over `downtown_a`: scripts of 4-6 flights of seeds 101-104, built once for the module
(`drone_kit.generated`, one xdist group). Coordinates are the map's (`drone_kit`): the nest at (40, -18.5) with a drone
standing 0.27 m up on it, the drone a 1.30 m disc keeping 3.0 m, so 3.65 m from every wall; transit at 15 m, lowering
7 m above the zone's ground (7.15 m over a square pad, 7.01 m over a street zone), the gimbal -20 degrees on the pad,
-45 in transit, -90 from the inspection on."""

from __future__ import annotations

import math
from dataclasses import replace
from unittest import mock

import numpy as np
import pytest

import drone_kit
from stbench import job_drone
from stbench.generate import build_world, generate_script
from stbench.geometry import dist_to_rect
from stbench.job_registry import get
from stbench.timeline_ego import TaskPlan
from stbench.validate import validate

pytestmark = pytest.mark.xdist_group(name=__name__)  # one worker builds the scripts once and every test reads them

SEEDS = (101, 102, 103, 104)
M = drone_kit.downtown()
JOB = get("drone/1")
DELIVER, INSPECT = "deliver", "inspect"
DOWN = ("inspect", "descend", "lower", "release", "climb_out", "land")  # the phases that look straight down
TRANSIT = ("takeoff", "approach", "return")  # the phases that look forward and down


@pytest.fixture(scope="module")
def scripts():
    return {seed: drone_kit.generated(seed) for seed in SEEDS}


def flights(scripts, kind: str | None = None):
    """Every (script, task, its keyframes) of the scripts, of one kind if asked."""
    return [(tl, task, tl.ego.keyframes[task.k0:task.k1]) for tl in scripts.values() for task in tl.ego.tasks
            if kind is None or task.plan.kind == kind]


def test_every_flight_runs_validly(scripts):
    """Contract: `validate` (the generic rules, the flier's body rule at every keyframe and the job's own checks) finds
    nothing on the scripts: 4-6 flights each, a drone never in a facade or inside a thing it must clear. Regression
    caught: a way across a block, a keyframe at a height the body rule flies under, or a status outside the
    vocabulary."""
    for seed, tl in scripts.items():
        assert 4 <= len(tl.ego.tasks) <= 6, seed
        assert validate(tl, M) == [], seed
    assert {t.plan.kind for tl in scripts.values() for t in tl.ego.tasks} == {DELIVER, INSPECT}


def test_the_flight_keeps_off_the_walls(scripts):
    """Contract (the review's body rule; M3-Y §3): at every point of every straight line between consecutive keyframes
    of a flight, not only at the keyframes, the drone's 0.65 m footprint is at least 3.0 m from every wall piece (a wall
    obstructs a flier at any height), so the base is 3.65 m from them. Regression caught: a way that cuts a block's
    corner between two keyframes, which a keyframe-only check passes."""
    walls = M.wall_pieces()
    least = math.inf
    for _, _, kfs in flights(scripts):
        for a, b in zip(kfs, kfs[1:], strict=False):
            n = max(math.ceil(math.dist(a.base[:2], b.base[:2]) / 0.25), 1)
            s = np.linspace(0.0, 1.0, n + 1)
            xs, ys = a.base[0] + (b.base[0] - a.base[0]) * s, a.base[1] + (b.base[1] - a.base[1]) * s
            least = min(least, *(float(dist_to_rect(xs, ys, w).min()) for w in walls))
    assert least - 0.65 >= 3.0 - 0.001


def test_each_flight_starts_and_ends_on_the_nest(scripts):
    """Contract: every flight's first keyframe is `preflight` and its last `land`, both with the base on the nest at
    (40.0, -18.5), 0.27 m up (the pad's 0.02 m and the gear's 0.25 m) within a centimetre, `tether_m` 0, the task index
    in order, the battery in 95-100 at the start and never rising: nothing carried from the flight before. Regression
    caught: state carried between flights, a landing beside the pad, or a battery that keeps draining across a gap."""
    for _, task, kfs in flights(scripts):
        for kf, phase in ((kfs[0], "preflight"), (kfs[-1], "land")):
            assert kf.phase == phase and kf.task == task.index
            assert math.dist(kf.base[:2], drone_kit.NEST) <= 0.01 and abs(kf.base[2] - drone_kit.PAD_Z) <= 0.01
            assert kf.status["tether_m"] == 0.0
        assert 95 <= kfs[0].status["battery_pct"] <= 100
        batteries = [kf.status["battery_pct"] for kf in kfs]
        assert batteries == sorted(batteries, reverse=True) and batteries[0] - batteries[-1] <= 10


def test_the_camera_looks_down_from_inspection_on(scripts):
    """Contract (design §4: the gimbal "forward-down in approach; straight down from descent on"; M3-Y §3): the camera's
    pitch is -20 degrees at preflight, -45 in takeoff, approach and return, -90 in inspect, descend, lower, release,
    climb_out and land, on every keyframe, so no frame shows the horizon and none is outside the gimbal's -90 to -20.
    Regression caught: a forward view of the street's far end or of the sky, or a frame outside the gimbal."""
    for _, _, kfs in flights(scripts):
        for kf in kfs:
            want = -20.0 if kf.phase == "preflight" else -45.0 if kf.phase in TRANSIT else -90.0
            assert kf.phase in ("preflight", *TRANSIT, *DOWN)
            assert math.degrees(kf.cam_pitch) == pytest.approx(want, abs=1e-4), (kf.k, kf.phase)
            assert -90.0 <= math.degrees(kf.cam_pitch) <= -20.0


def rate(a, b) -> float:
    """The tether's change between two keyframes, metres a second."""
    return (b.status["tether_m"] - a.status["tether_m"]) / (b.t - a.t)


def test_the_tether_pays_out_only_while_lowering(scripts):
    """Contract (M3-Y §2, `tether_m`): in a delivery the tether is 0 until the `before_lower` keyframe, rises at the
    lowering speed (0.5-1.0 m/s) over `lower` to 7.0 m, holds 7.0 over `release`, falls at the same speed over the
    reel-in (`climb_out` at the lowering height, 7.15 m over a square pad, 7.01 m over a street zone's 0.01 m loading
    zone) to 0 and is 0 from the climb on and in every other phase; an inspection's is 0 throughout, and so is that of
    a delivery aborted over a zone that holds an item. Regression caught: a status that is not the robot's (a tether in
    flight), a lowering that jumps, or a package lowered by an inspection or over an occupied zone."""
    for _, _, kfs in flights(scripts, INSPECT):
        assert {kf.status["tether_m"] for kf in kfs} == {0.0}
    for tl, task, kfs in flights(scripts, DELIVER):
        if drone_kit.blocked_at_start(tl, task):
            assert {kf.status["tether_m"] for kf in kfs} == {0.0}
    deliveries = [f for f in flights(scripts, DELIVER) if not drone_kit.blocked_at_start(f[0], f[1])]
    assert deliveries
    for tl, task, kfs in deliveries:
        before = tl.keyframe(next(d.k for d in tl.decisions_in(task.k0, task.k1) if d.kind == "before_lower"))
        lower, release = ([kf for kf in kfs if kf.phase == p] for p in ("lower", "release"))
        reel = [kf for kf in kfs if kf.phase == "climb_out" and kf.base[2] < 7.16]
        climb = [kf for kf in kfs if kf.phase == "climb_out" and kf.base[2] >= 7.16]
        assert before.status["tether_m"] == 0.0 and lower and release and reel and climb
        low = {"F26": 7.15, "F27": 7.15, "F28": 7.15, "F29": 7.01, "F30": 7.01}[task.plan.params["zone"]]
        assert {kf.base[2] for kf in [before, *lower, *release, *reel]} == {low}
        out = [rate(a, b) for a, b in zip([before, *lower], lower, strict=False)]
        assert 0.5 - 0.01 <= out[0] <= 1.0 + 0.01
        assert out[:-1] == pytest.approx([out[0]] * (len(out) - 1), rel=1e-3) and 0 < out[-1] <= out[0] + 1e-3
        assert lower[-1].status["tether_m"] == pytest.approx(7.0, abs=1e-5)
        assert {kf.status["tether_m"] for kf in release} == {7.0}
        back = [rate(a, b) for a, b in zip([release[-1], *reel], reel, strict=False)]
        assert back[0] == pytest.approx(-out[0], rel=1e-3) and -out[0] - 1e-3 <= back[-1] < 0
        assert reel[-1].status["tether_m"] == pytest.approx(0.0, abs=1e-5)
        ends = {kf.k for kf in [before, *lower, *release, *reel]}
        assert {kf.status["tether_m"] for kf in kfs if kf.k not in ends} == {0.0}


def test_decisions_come_from_the_plan_alone(scripts):
    """Contract (design §8.5): a fixed plan scripted over two different worlds (seeds 101's and 102's) gives the same
    decision kinds, flight by flight, and the very same keyframe times and poses: the way, the times and the poses are
    the map's and the seed's (the phases after `before_lower` and the response follow the zone, which the coin sets in
    both worlds alike: `test_job_drone_control`). Regression caught: a decision point or a pose the world triggered (an
    item in the street, a drop zone's state), which would tell a system of the world through the path."""
    tl = scripts[101]
    n = len(tl.ego.tasks)
    plans = JOB.plan(M, 101, n)
    ego = {}
    for world_seed in (101, 102):
        w = build_world(M, world_seed)
        ego[world_seed] = JOB.script(M, w, plans, 101)
    poses = {s: [(kf.t, kf.pos, kf.rot, kf.base, kf.yaw, kf.cam_pitch, kf.task) for kf in e.keyframes]
             for s, e in ego.items()}
    assert poses[101] == poses[102]
    assert [(d.kind, d.k) for d in ego[101].decisions] == [(d.kind, d.k) for d in tl.ego.decisions]
    assert [(d.kind, d.k) for d in ego[102].decisions] == [(d.kind, d.k) for d in tl.ego.decisions]
    kinds = {d.kind for d in tl.ego.decisions}
    assert kinds == {"before_descent", "before_lower", "before_return"}
    assert [d.kind for d in tl.ego.decisions] == ["before_descent", "before_lower", "before_return"] * n


def test_an_inspection_is_a_deliverys_flight(scripts):
    """Contract (the review's minor 8; M3-Y §2): the same plan with each flight's kind flipped gives the same flight to
    the byte: every pose, keyframe time, camera, battery and decision point equal. Only the hover at the lowering height
    differs, in the status the pair records: a delivery's `lower`, `release` and the reel-in (`climb_out` at 7.15 m) are
    an inspection's `inspect`, and an inspection's tether is 0; a delivery over a zone that holds an item is aborted and
    flies the inspection's hover, so that pair differs in nothing but the delivery's response (`abort`, where an
    inspection's is `go`). Regression caught: a swap pair whose streams differ in more than the recorded status, so a
    brief swap would test a contradicted stream."""
    tl = scripts[101]
    n = len(tl.ego.tasks)
    flipped = [TaskPlan(INSPECT if p.kind == DELIVER else DELIVER, p.params) for p in JOB.plan(M, 101, n)]
    with mock.patch.object(job_drone, "SPEC", replace(JOB, plan=lambda m, seed, n_tasks: flipped)):
        other = generate_script(drone_kit.MAP, 101, "drone", "drone/1", n)
    kinds = [t.plan.kind for t in tl.ego.tasks]
    assert set(kinds) == {DELIVER, INSPECT}
    assert [t.plan.kind for t in other.ego.tasks] == [INSPECT if k == DELIVER else DELIVER for k in kinds]
    assert [(d.k, d.kind) for d in other.ego.decisions] == [(d.k, d.kind) for d in tl.ego.decisions]
    assert other.ego.tasks[0].k0 == tl.ego.tasks[0].k0
    for task in tl.ego.tasks:  # the zone holds the same in both worlds; only a delivery over an item responds abort
        flies_delivery, flies_inspection = (tl, other) if kinds[task.index] == DELIVER else (other, tl)
        want = "abort" if drone_kit.blocked_at_start(tl, task) else "go"
        assert [d.response for d in flies_delivery.decisions_in(task.k0, task.k1)] == [None, want, None]
        assert [d.response for d in flies_inspection.decisions_in(task.k0, task.k1)] == [None, "go", None]
    hovers = 0
    for a, b in zip(tl.ego.keyframes, other.ego.keyframes, strict=True):
        assert (a.k, a.t, a.pos, a.rot, a.base, a.yaw, a.cam_pitch, a.task, a.status["battery_pct"]) == (
            b.k, b.t, b.pos, b.rot, b.base, b.yaw, b.cam_pitch, b.task, b.status["battery_pct"])
        delivery, inspection = (a, b) if kinds[a.task] == DELIVER else (b, a)
        if delivery.phase in ("lower", "release") or (delivery.phase == "climb_out" and delivery.base[2] < 7.16):
            assert (inspection.phase, inspection.status["tether_m"]) == ("inspect", 0.0)
            hovers += 1
        else:
            assert (delivery.phase, delivery.status) == (inspection.phase, inspection.status)
    assert hovers >= 6
