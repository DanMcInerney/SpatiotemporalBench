"""`drone/1`'s plan, gaps, zone coin, briefs, events and checks, over `downtown_a`: scripts of 4-6 flights of seeds
101-104, built once for the module (`drone_kit.generated`, one xdist group). Coordinates and sizes are the map's
(`drone_kit`'s docstring)."""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
from dataclasses import replace
from unittest import mock

import pytest

import drone_kit
from stbench import brief, job_drone
from stbench import timeline as T
from stbench.episode import pose_rows
from stbench.events import resolve
from stbench.generate import generate_script
from stbench.geometry import Rect
from stbench.job_drone_plan import coin_blocked, gaps, settable_zones
from stbench.job_drone_zone import overlap_area, zone_rect
from stbench.job_registry import get
from stbench.robots import ROBOTS, robot_doc
from stbench.timeline_ego import TaskPlan
from stbench.zones import ground_z

pytestmark = pytest.mark.xdist_group(name=__name__)  # one worker builds the scripts once and every test reads them

SEEDS = (101, 102, 103, 104)
M = drone_kit.downtown()
JOB = get("drone/1")
DELIVER, INSPECT = "deliver", "inspect"


@pytest.fixture(scope="module")
def scripts():
    return {seed: drone_kit.generated(seed) for seed in SEEDS}


def poses(tl) -> list[tuple]:
    """Every keyframe's time and pose (the camera's and the base's) and flight: what the way and the clock decide."""
    return [(kf.t, kf.pos, kf.rot, kf.base, kf.yaw, kf.cam_pitch, kf.task) for kf in tl.ego.keyframes]


def digest(seed: int, n: int) -> str:
    return hashlib.sha256(T.dumps(generate_script(
        drone_kit.MAP, seed, "drone", "drone/1", n)).encode()).hexdigest()


def test_scripts_are_deterministic():
    """Contract: a 4-flight script is byte-identical (`timeline.dumps`) in this process and in another with another
    `PYTHONHASHSEED`, and another seed gives another one. Regression caught: an unseeded draw, or a set's or dict's
    iteration order in the plan, the drops, the processes or the path."""
    out = subprocess.run([sys.executable, __file__, "103"], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONHASHSEED": "12345"}).stdout.strip()
    assert digest(103, 4) == out != digest(104, 4)


def test_the_plan_is_the_maps_and_the_seeds():
    """Contract (M3-Y §2): over seeds 1-1000 of five flights, a flight is an inspection with probability 0.25 and a
    delivery otherwise, and its drop zone is uniform among the zones the world can set (the five of the shipped map:
    1/5 each), each share within 4 standard errors; `params` hold the zone's name as `address`, its piece id, and the
    region kinds of the zone (the square for the pads, the street the loading zone lies in for the others) and of the
    nest (the square); one flight's plan is the same call twice. Regression caught: a kind or a zone the world decided,
    a skewed share, or an address that is not the zone's name."""
    n, inspect, picks = 5000, 0, {}
    names = {"F26": "Square West", "F27": "Square East", "F28": "Square South", "F29": "West Street",
             "F30": "East Street"}
    regions = {"F26": "square", "F27": "square", "F28": "square", "F29": "west street", "F30": "east street"}
    for seed in range(1, 1001):
        plans = JOB.plan(M, seed, 5)
        assert [p.kind for p in plans] == [p.kind for p in JOB.plan(M, seed, 5)] and len(plans) == 5
        for p in plans:
            inspect += p.kind == INSPECT
            picks[p.params["zone"]] = picks.get(p.params["zone"], 0) + 1
            assert p.kind in (DELIVER, INSPECT) and p.params["address"] == names[p.params["zone"]]
            assert (p.params["region"], p.params["nest_region"]) == (regions[p.params["zone"]], "square")
    assert abs(inspect / n - 0.25) <= 4 * math.sqrt(0.25 * 0.75 / n)
    assert set(picks) == set(drone_kit.ZONES)
    for count in picks.values():
        assert abs(count / n - 1 / 5) <= 4 * math.sqrt(4 / 25 / n)
    with pytest.raises(ValueError, match="at least one flight"):
        JOB.plan(M, 1, 0)


def test_a_zone_is_planned_only_where_the_world_can_make_it_clear():
    """Contract (M3-Y §4: balance from the world): a drop zone is planned only when the world can put any item that
    stands on it somewhere clear of every zone, from the map and its catalog alone. By hand: the box truck (7.3 x 2.4 m)
    fits only a loading zone, 2.5 x 14.8 m (y 42.1 to 56.9 for the west one), and a street zone's piece is 3.0 m of it,
    so 11.8 m lies beside the piece. At the south end of the loading zone, where the map has the street zones (the piece
    y 42.6 to 45.6), 0.5 m lies south of it and 11.3 m north, where the truck stands clear of the zone: all five zones
    are planned. In the middle of the loading zone (the piece y 48.0 to 51.0) 5.9 m lies each side, short of the truck,
    which covers the zone whichever way it stands: `F29` and `F30` are then not planned. A zone on bare ground has no
    surface for the world to put an item onto and is not. Regression caught: a street zone whose clear coin the world
    cannot honour (about half of those flights came out blocked), or a zone no item can be put on."""
    assert [z.id for z in settable_zones(M)] == list(drone_kit.ZONES)
    middle = tuple(replace(f, center=(f.center[0], 49.5)) if f.id in ("F29", "F30") else f for f in M.furniture)
    assert [z.id for z in settable_zones(replace(M, furniture=middle))] == ["F26", "F27", "F28"]
    bare = tuple(replace(f, center=(10.0, 10.0)) if f.id == "F26" else f for f in M.furniture)
    assert [z.id for z in settable_zones(replace(M, furniture=bare))] == ["F27", "F28", "F29", "F30"]


def test_the_zone_geometry_by_hand():
    """Contract: `overlap_area` is the area two rectangles share, and a marking lies on the ground under it. Hand cases:
    2 x 2 squares a metre apart share 2 m2; a 2 x 2 square and the same turned 45 degrees share the octagon of inradius
    1, 8 tan(22.5 degrees) = 3.3137 m2; apart they share none; `F26` lies on the 0.15 m plaza under it, `F29` on the
    0.01 m loading zone, the nest on bare ground; a zone is its piece less 0.05 m on each side. Regression caught: a
    blocking share read from a wrong overlap, or a zone judged from the street's level."""
    assert overlap_area(Rect(0, 0, 2, 2, 0), Rect(1, 0, 2, 2, 0)) == pytest.approx(2.0)
    assert overlap_area(Rect(0, 0, 2, 2, 0), Rect(0, 0, 2, 2, math.pi / 4)) == pytest.approx(8 * math.tan(math.pi / 8))
    assert overlap_area(Rect(0, 0, 2, 2, 0), Rect(3, 0, 2, 2, 0)) == 0.0
    pieces = M.furniture_by_id()
    assert [ground_z(M, pieces[i]) for i in ("F26", "F29", "F25")] == [0.15, 0.01, 0.0]
    assert zone_rect(pieces["F26"]) == Rect(40.0, -6.0, 2.9, 2.9, 0.0)
    assert zone_rect(pieces["F29"]) == pytest.approx(Rect(20.3, 44.1, 2.5, 2.9, 0.0))


def test_the_zone_coin_holds_both_ways(scripts):
    """Contract (M3-Y §4; the review's M1): a flight's drop zone is blocked when a coin that reads no world says so:
    over 1,000 seeds it is 0.5 within 4 standard errors and is the one `make_rng(map, seed, job, "zone", i)` gives. In a
    script, at every flight's first keyframe the zone holds an item (a footprint overlapping it, taller than 0.3 m over
    its ground) exactly when its coin says blocked, and none within 5 cm when it says clear: the square's three zones
    over seeds 101-104. And the way is the map's: the same script with the drops left out has the same poses, times and
    decision points (a drop moves only the phases and the response, which follow the zone: `test_job_drone_control`).
    Regression caught: a balance the world decides, a zone the drop did not settle, or a path that tells of the drop."""
    flips = [coin_blocked(drone_kit.MAP, s, 0) for s in range(1, 1001)]
    assert flips == [drone_kit.coin_says_blocked(s, 0) for s in range(1, 1001)]
    assert abs(sum(flips) / 1000 - 0.5) <= 4 * math.sqrt(0.25 / 1000)
    seen = set()
    for seed, tl in scripts.items():
        for task in tl.ego.tasks:
            piece = M.furniture_by_id()[task.plan.params["zone"]]
            state = drone_kit.zone_state(tl, M, piece.id, ground_z(M, piece), tl.keyframe(task.k0).t)
            blocked = drone_kit.coin_says_blocked(seed, task.index)
            assert state == ("blocked" if blocked else "clear"), (seed, task.index, piece.name)
            seen.add(blocked)
    assert seen == {True, False}
    n = len(scripts[102].ego.tasks)
    with mock.patch.object(job_drone, "enforce", lambda *args, **kwargs: True):
        undropped = generate_script(drone_kit.MAP, 102, "drone", "drone/1", n)
    assert poses(undropped) == poses(scripts[102])
    assert [(d.k, d.kind) for d in undropped.ego.decisions] == [(d.k, d.kind) for d in scripts[102].ego.decisions]
    assert T.digest(undropped) != T.digest(scripts[102])  # seed 102's worlds differ (items moved), the paths do not


def test_gaps_come_from_the_seed_alone(scripts):
    """Contract: `gaps(map, seed, n)` is `job_gaps`' law with this job's id and 600 s longest flight, needs no world,
    and a script's gaps (the time from a flight's last keyframe to the next one's first) are exactly those; over 1,000
    seeds and 4-6 flights each is 600-10,800 s and, every flight at its longest, the script spans at most 7 hours
    (25,200 s). Regression caught: gaps drawn from the world, or a span that ends a case clock after 17:00."""
    assert gaps(drone_kit.MAP, 7, 5) == gaps(drone_kit.MAP, 7, 5) != gaps(drone_kit.MAP, 8, 5)
    assert gaps(drone_kit.MAP, 7, 1) == []
    for seed in range(1, 1001):
        n = 4 + seed % 3
        drawn = gaps(drone_kit.MAP, seed, n)
        assert len(drawn) == n - 1 and all(600.0 - 1e-6 <= g <= 10800.0 + 1e-6 for g in drawn), seed
        assert sum(drawn) + n * 600.0 <= 25200.0 + 1e-6, seed
    tl = scripts[101]
    want = gaps(drone_kit.MAP, 101, len(tl.ego.tasks))
    got = [tl.keyframe(b.k0).t - tl.keyframe(a.k1 - 1).t for a, b in zip(tl.ego.tasks, tl.ego.tasks[1:], strict=False)]
    assert got == pytest.approx(want, abs=1e-3)
    assert max(kf.t for kf in tl.ego.keyframes) - tl.ego.keyframes[0].t <= 25200.0


def test_assignments_read_as_briefs():
    """Contract: for every plan of seeds 1-200, `brief.compose` renders each flight's text in `correct` and in `swapped`
    with the drop point's name only: by hand "Its current assignment is to deliver a package to the drop point at
    Square West." and, swapped, "... to inspect the drop point at Square West."; no piece id or region word of the
    job's (`F26`, `nest_region`) in it. Regression caught: a swap with no text, or a template naming a param the plan
    lacks."""
    robot = robot_doc(ROBOTS["drone"])
    hand = [TaskPlan(DELIVER, {"address": "Square West", "zone": "F26"}),
            TaskPlan(INSPECT, {"address": "West Street", "zone": "F29"})]
    texts = brief.compose(hand, job=JOB, robot=robot, domain="delivery", channel="rgb_pose")
    lead = "Its current assignment is to {} the drop point at Square West. You will receive"
    assert lead.format("deliver a package to") in texts[0]
    assert texts[1] == "Its current assignment is to inspect the drop point at West Street."
    swapped = brief.compose(hand, job=JOB, robot=robot, domain="delivery", channel="rgb_pose", variant="swapped")
    assert lead.format("inspect") in swapped[0]
    assert swapped[1] == "Its current assignment is to deliver a package to the drop point at West Street."
    for seed in range(1, 201):
        plans = JOB.plan(M, seed, 3)
        for variant in ("correct", "swapped"):
            for plan, text in zip(plans, brief.compose(plans, job=JOB, robot=robot, domain="delivery",
                                                       channel="rgb_pose", variant=variant), strict=True):
                assert f"drop point at {plan.params['address']}." in text
                assert "F2" not in text and "nest_region" not in text


def test_events_resolve_where_they_should(scripts):
    """Contract (plan §2.2): through `events.resolve` over a flight's delivered steps (their `k` and status), a
    delivery that goes on resolves all four events (the first takeoff keyframe, the first lowering one, the first
    release one and the last landing one, the flight's last), and an inspection, or a delivery aborted because its zone
    holds an item (read from the placements), resolves takeoff and landing only: it lowers nothing and releases nothing.
    Regression caught: an event no system can resolve, or one an inspection or an aborted delivery resolves to a
    hover."""
    for tl in scripts.values():
        rows = pose_rows(tl)
        for task in tl.ego.tasks:
            kfs = tl.ego.keyframes[task.k0:task.k1]
            steps = [{"k": r["k"], "status": r["status"]} for r in rows[task.k0:task.k1]]
            got = {w: resolve(rule, steps) for w, rule in JOB.events.items()}
            first = {p: min((kf.k for kf in kfs if kf.phase == p), default=None)
                     for p in ("takeoff", "lower", "release")}
            want = {"when you took off": first["takeoff"], "when you started lowering the package": first["lower"],
                    "when you released the package": first["release"], "when you landed": task.k1 - 1}
            assert got == want
            lowers = task.plan.kind == DELIVER and not drone_kit.blocked_at_start(tl, task)
            assert (got["when you released the package"] is not None) == lowers
            assert (got["when you started lowering the package"] is not None) == lowers


def with_task_times(tl, first: int, shift: float):
    """`tl` with every keyframe from keyframe `first` on `shift` seconds later."""
    kfs = [replace(kf, t=kf.t + shift) if kf.k >= first else kf for kf in tl.ego.keyframes]
    return replace(tl, ego=replace(tl.ego, keyframes=kfs))


@pytest.mark.parametrize(("edit", "code"), [
    ("first keyframe off the nest", "job_nest"),
    ("last keyframe a metre up", "job_nest"),
    ("a phase skipped", "job_phase_order"),
    ("a decision dropped", "job_decisions"),
    ("before_lower off its keyframe", "job_decisions"),
    ("transit at 12 m", "job_heights"),
    ("lowering at 6 m", "job_heights"),
    ("a frame above the transit height", "job_heights"),
    ("a forward view while inspecting", "job_gimbal"),
    ("a gimbal that jumps", "job_gimbal"),
    ("a tether in transit", "job_tether"),
    ("a tether short of 7 m", "job_tether"),
    ("an inspection that lowers", "job_tether"),
    ("a battery that rises", "job_battery"),
    ("a gap of a minute", "job_gap"),
    ("a script of 9 hours", "job_span"),
    ("a flight of 20 minutes", "job_task_time"),
    ("targets that are not the zone's region", "job_targets"),
])
def test_the_jobs_own_checks_have_teeth(scripts, edit, code):
    """Contract: `checks` flags a script that is not a `drone/1` one: a flight that does not start or end on the pad, a
    phase missing, a decision missing or off its keyframe, transit not at 15 m, the lowering not at 7 m over the zone's
    ground, a frame above the transit height, a camera not at its phase's pitch or a gimbal turned faster than 60
    degrees a second, a tether outside the lowering (or short of 7 m, or in an inspection), a rising battery, a gap
    under 10 minutes, a script over 7 hours, a flight over 10 minutes or targets that are not its zone's region.
    Regression caught: a rule of the job that no script can break."""
    tl = scripts[101]
    first = next(t for t in tl.ego.tasks if t.plan.kind == DELIVER)
    insp = next((t for t in tl.ego.tasks if t.plan.kind == INSPECT), None)
    assert insp is not None and first.index != insp.index
    kfs, decisions, tasks = list(tl.ego.keyframes), list(tl.ego.decisions), list(tl.ego.tasks)
    ks = {p: [kf for kf in kfs[first.k0:first.k1] if kf.phase == p] for p in
          ("approach", "inspect", "lower", "climb_out", "takeoff", "preflight")}
    broken = tl

    def put(kf, **fields):
        kfs[kf.k] = replace(kf, **fields)

    if edit == "first keyframe off the nest":
        put(kfs[first.k0], base=(41.0, -18.5, 0.27))
    elif edit == "last keyframe a metre up":
        put(kfs[first.k1 - 1], base=(40.0, -18.5, 1.27))
    elif edit == "a phase skipped":
        for kf in ks["takeoff"]:
            put(kf, phase="approach")
    elif edit == "a decision dropped":
        decisions.pop(0)
    elif edit == "before_lower off its keyframe":
        i = next(i for i, d in enumerate(decisions) if d.kind == "before_lower" and d.k > first.k0)
        decisions[i] = replace(decisions[i], k=decisions[i].k - 1)
    elif edit == "transit at 12 m":
        put(ks["approach"][0], base=(*ks["approach"][0].base[:2], 12.0))
    elif edit == "lowering at 6 m":
        put(ks["lower"][0], base=(*ks["lower"][0].base[:2], 6.0))
    elif edit == "a frame above the transit height":
        put(ks["takeoff"][-1], base=(*ks["takeoff"][-1].base[:2], 16.0))
    elif edit == "a forward view while inspecting":
        put(ks["inspect"][0], cam_pitch=math.radians(-45.0))
    elif edit == "a gimbal that jumps":
        put(ks["preflight"][-1], cam_pitch=math.radians(-90.0))
    elif edit == "a tether in transit":
        put(ks["approach"][0], status={**ks["approach"][0].status, "tether_m": 2.0})
    elif edit == "a tether short of 7 m":
        put(ks["lower"][-1], status={**ks["lower"][-1].status, "tether_m": 6.0})
    elif edit == "an inspection that lowers":
        k = next(kf for kf in kfs[insp.k0:insp.k1] if kf.phase == "inspect" and kf.base[2] < 8.0)
        put(k, status={**k.status, "tether_m": 3.0})
    elif edit == "a battery that rises":
        put(ks["climb_out"][-1], status={**ks["climb_out"][-1].status, "battery_pct": 100})
    elif edit == "a gap of a minute":
        t1 = tl.ego.tasks[1]
        broken = with_task_times(tl, t1.k0, -(tl.keyframe(t1.k0).t - tl.keyframe(t1.k0 - 1).t) + 60.0)
    elif edit == "a script of 9 hours":
        broken = with_task_times(tl, tasks[-1].k0, 32400.0)
    elif edit == "a flight of 20 minutes":
        broken = with_task_times(tl, first.k0 + 20, 1200.0)
    else:
        tasks[0] = replace(tasks[0], targets=("R1",))
    broken = replace(broken, ego=replace(broken.ego, keyframes=kfs if broken is tl else broken.ego.keyframes,
                                         decisions=decisions, tasks=tasks))
    assert code in {p.code for p in job_drone._checks(broken, M)}
    assert job_drone._checks(tl, M) == []


if __name__ == "__main__":  # the second process of `test_scripts_are_deterministic`
    print(digest(int(sys.argv[1]), 4))
