"""`amr/1`'s map contract, plan, script and checks over `warehouse_a`: scripts of 3 tasks of seeds 101-103 built once
for the module (`amr_kit.generated`, one xdist group with the job's other tests). Coordinates are the map's
(`maps/warehouse_a.json`): the charger at (17.44, 1.40) 0.12 m deep facing west (its face at x 17.38), the AMR a 0.80 x
0.58 m rectangle, so its docked pose is (16.98, 1.40) heading 0 (east, head-on); the shelving F11, F12 at x 0.4 (face
0.70) and the packing tables F09, F10 at x 17.1 (face 16.70) are the stations."""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
from dataclasses import replace
from unittest import mock

import numpy as np
import pytest

import amr_kit
from stbench import brief, job_amr, job_amr_plan, questions_zone, route
from stbench import job_amr_drops as drops
from stbench import timeline as T
from stbench.events import resolve
from stbench.generate import add_item, build_world
from stbench.job_registry import get
from stbench.placement import ItemSpec, Spot
from stbench.robots import ROBOTS, robot_doc
from stbench.validate import validate
from stbench.zones import ZoneRule

pytestmark = pytest.mark.xdist_group(name="test_job_amr")  # one worker builds the scripts once, every test reads them

M, JOB = amr_kit.M, get("amr/1")
HOME = (16.98, 1.4)
REPLENISH, COLLECT, PICK = "replenish", "collect", "pick"
TEXTS = {REPLENISH: "Its current assignment is to bring totes from the tote shelves to the packing tables.",
         COLLECT: "Its current assignment is to collect the empty totes from the packing tables and return them to the "
                  "tote shelves.",
         PICK: "Its current assignment is an order-picking run: collect the picks from the racks, then bring them to a "
               "packing table."}


@pytest.fixture(scope="module")
def scripts():
    """{seed: the script}: 3 tasks, the real world processes, bay coin and rear drops."""
    return {seed: amr_kit.generated(seed, 3) for seed in amr_kit.SEEDS}


def digest(tl) -> str:
    return hashlib.sha256(T.dumps(tl).encode()).hexdigest()


def test_scripts_are_deterministic():
    """Contract: a 3-task `amr/1` script is byte-identical (`timeline.dumps`) in this process and in another with
    another `PYTHONHASHSEED`, and another seed gives another one. Regression caught: an unseeded draw, or a set's or
    dict's iteration order in the plan, the bay coin, the drops, the processes or the path."""
    out = subprocess.run([sys.executable, __file__, "103"], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONHASHSEED": "12345"}).stdout.strip()
    assert digest(amr_kit.generated(103, 3)) == out != digest(amr_kit.generated(102, 3))


def test_every_task_runs_validly(scripts):
    """Contract: on every script `validate` (the generic rules and the job's own) finds nothing, and the AMR's true
    rectangle (0.80 x 0.58 m at its heading) keeps its 0.10 m clearance, less a millimetre, from the walls and from
    every fixed piece, light piece and thing on the floor at every point of every segment between keyframes, docking and
    undocking included (a turn in place sweeps the 0.4935 m circumscribed disc). Regression caught: a path through a
    rack, a pallet, a drum in a bay or a rear drop, or a docking that comes nearer a station's face than 0.10 m."""
    for seed, tl in scripts.items():
        assert validate(tl, M) == [], seed
        assert len(tl.ego.tasks) == 3
        for task in tl.ego.tasks:
            assert amr_kit.least_clearance(tl, task) >= 0.10 - 0.001, (seed, task.index)


def test_paths_ignore_lane_loads():
    """Contract (design §8.1, "paths leak"): a task is planned with every staging lane a keep-out, loaded or not, so no
    lane's load changes a path (the robot's own path log would read it, the answer a `slots_empty` question would ask
    for). The first 2 tasks of seed 101 (a replenish from the charger, at y 1.4 in the lanes' band, to the west
    shelving, whose shortest way crosses the lanes) are byte for byte the same on the same world with every lane
    holding a pallet and with every lane empty; with the keep-outs removed the two differ. Regression caught: a detour
    that reads a lane's load."""
    def lanes(w):
        return [eid for eid, c in w.carried.items() if c.spot.slot.kind == "staging lane"]

    def loaded(w):
        for eid in lanes(w):
            w.ledger.remove(eid, 0.0)
            del w.carried[eid]
            w.entities = [e for e in w.entities if e.id != eid]
        slots = [s for s in w.slots if s.kind == "staging lane"]
        assert len(slots) == 4
        for i, slot in enumerate(slots):
            add_item(w, ItemSpec("pallet", {"color": ("blue", "red", "green", "yellow")[i]}),
                     Spot(slot, (slot.rect.cx, slot.rect.cy, slot.z), slot.rect.yaw), [])
        assert len(lanes(w)) == 4

    def emptied(w):
        for eid in lanes(w):
            w.ledger.remove(eid, 0.0)
            del w.carried[eid]
            w.entities = [e for e in w.entities if e.id != eid]
        assert lanes(w) == []

    kw = {"coin": False, "rear": False}
    full, empty = amr_kit.quiet(101, 2, build=loaded, **kw), amr_kit.quiet(101, 2, build=emptied, **kw)
    assert full.world_hash != empty.world_hash
    assert full.ego_hash == empty.ego_hash
    with mock.patch.object(job_amr, "keep_outs", lambda m, robot: []):
        free_full, free_empty = amr_kit.quiet(101, 1, build=loaded, **kw), amr_kit.quiet(101, 1, build=emptied, **kw)
    assert free_full.ego_hash != free_empty.ego_hash  # the control: without keep-outs the path does read the loads


def test_decisions_come_from_the_plan_alone():
    """Contract (design §8.5): a fixed plan scripted over two different worlds (seeds 101 and 102, `plan` replaced by
    seed 102's) gives the same decision kinds, task by task, in the same order, once the bay coins are the same: with
    every coin saying clear (all docked: a `before_undock` after each `before_station`) and with every coin saying
    blocked (all skipped: none). The skips themselves are the coins' and the world's drums', never the plan's.
    Regression caught: a decision point the world triggered (a drop, a lane's load, a light piece in the way)."""
    swapped = replace(JOB, plan=lambda m, seed, n_tasks: JOB.plan(m, 102, n_tasks))
    kinds = lambda tl: [(tl.task_of(d.k).index, d.kind) for d in tl.ego.decisions]  # noqa: E731
    for blocked in (False, True):
        home = amr_kit.quiet(102, 2, coin=blocked)
        with mock.patch.object(job_amr, "SPEC", swapped):
            other = amr_kit.quiet(101, 2, coin=blocked)
        assert other.world_hash != home.world_hash
        assert kinds(other) == kinds(home) != []
        assert [t.plan for t in other.ego.tasks] == [t.plan for t in home.ego.tasks]
        undocks = [d for d in other.ego.decisions if d.kind == "before_undock"]
        assert bool(undocks) == (not blocked)


def test_each_task_starts_and_ends_on_the_charger(scripts):
    """Contract: every task's first and last keyframe stand at the charger's docked pose (16.98, 1.40) heading 0, in
    phase `waiting`; `loaded` is false at the first and, at the last, false unless the last planned stop was skipped
    after a hold made (the robot then still carries what it loaded: docs/job-amr.md); its battery starts in 60-100 and
    never rises; the task index counts from 0 in order: nothing carried over from an earlier task shows where a window
    sits. Regression caught: state carried over from the task before, or a status that reveals a window's place."""
    for tl in scripts.values():
        for task in tl.ego.tasks:
            kfs = tl.ego.keyframes[task.k0:task.k1]
            for kf in (kfs[0], kfs[-1]):
                assert (kf.phase, kf.base, kf.yaw, kf.task) == ("waiting", (*HOME, 0.0), 0.0, task.index)
            last = [d for d in tl.decisions_in(task.k0, task.k1) if d.kind == "before_station"][-1]
            made = any(kf.phase == "at_station" for kf in kfs)
            assert kfs[0].status["loaded"] is False
            assert kfs[-1].status["loaded"] is (made and last.response == "skip")
            assert 60 <= kfs[0].status["battery_pct"] <= 100
            batteries = [kf.status["battery_pct"] for kf in kfs]
            assert batteries == sorted(batteries, reverse=True) and batteries[0] - batteries[-1] <= 3


def test_plans_follow_their_laws():
    """Contract (plan §5.2; M3-W §2): over seeds 1-1000 of five tasks, the kind shares are 0.35, 0.25 and 0.40 within 4
    standard errors; a replenish is one shelving piece then one or two packing tables (distinct), a collect the same
    tables in the other order then the shelving, a pick one packing table with one or two pick faces of one aisle in
    order from its mouth (distinct, of that aisle); the same seed's draws give a collect the stations of a replenish
    reversed. Regression caught: a swap pair that does not use one station set, a pick face off its aisle or out of
    order, or a kind share that is not the declared one."""
    shelves, tables = set(amr_kit.SHELVES), set(amr_kit.TABLES)
    assert (shelves, tables) == ({"F11", "F12"}, {"F09", "F10"})
    aisles = {a.region: a for a in job_amr_plan.aisles(M)}
    assert set(aisles) == {"R2", "R3", "R4"}
    count = {REPLENISH: 0, COLLECT: 0, PICK: 0}
    for seed in range(1, 1001):
        plans = amr_kit.plans_of(seed)
        assert len(plans) == 5
        for plan in plans:
            count[plan.kind] += 1
            ids = plan.params["stations"]
            assert len(set(ids)) == len(ids)
            if plan.kind == REPLENISH:
                assert ids[0] in shelves and 2 <= len(ids) <= 3 and set(ids[1:]) <= tables
            elif plan.kind == COLLECT:
                assert ids[-1] in shelves and 2 <= len(ids) <= 3 and set(ids[:-1]) <= tables
            else:
                aisle = aisles[plan.params["aisle"]]
                faces = [tuple(f) for f in plan.params["faces"]]
                depth = {(f.rack, f.surface): f.depth_m for f in aisle.faces}
                assert len(ids) == 1 and ids[0] in tables and 1 <= len(faces) <= 2 and len(set(faces)) == len(faces)
                assert set(faces) <= set(depth) and [depth[f] for f in faces] == sorted(depth[f] for f in faces)
            assert "dock" in plan.params["regions"] and ("aisle" in plan.params["regions"]) == (plan.kind == PICK)
    total = 5000
    for kind, p in job_amr_plan.KIND_P.items():
        assert abs(count[kind] / total - p) <= 4 * math.sqrt(p * (1 - p) / total), (kind, count[kind])
    assert job_amr_plan.KIND_P == {REPLENISH: 0.35, COLLECT: 0.25, PICK: 0.4}
    for seed in range(1, 61):  # one station set: the same draws, the other kind
        forced = {}
        for kind in (REPLENISH, COLLECT):
            with mock.patch.object(job_amr_plan, "KIND_P", {kind: 1.0}):
                forced[kind] = [p.params["stations"] for p in amr_kit.plans_of(seed)]
        assert forced[COLLECT] == [list(reversed(s)) for s in forced[REPLENISH]], seed
    with pytest.raises(ValueError, match="at least one task"):
        amr_kit.plans_of(1, 0)


def test_assignments_read_as_briefs():
    """Contract: for every plan of seeds 1-200, `brief.compose` (variant "correct") renders each task's text from its
    kind alone, by hand "Its current assignment is to bring totes from the tote shelves to the packing tables." for a
    replenish, and no id or param of the plan is in it; the swapped brief of a replenish is a collect's text and the
    reverse; a pick has no partner and refuses the swapped brief. Regression caught: a template that names a param, a
    plan's station id leaking into a brief, or a swap pair with no text."""
    robot = robot_doc(ROBOTS["amr"])
    for seed in range(1, 201):
        plans = amr_kit.plans_of(seed)
        texts = brief.compose(plans, job=JOB, robot=robot, domain="warehouse", channel="rgb_pose")
        assert len(texts) == len(plans)
        for plan, text in zip(plans, texts, strict=True):
            assert TEXTS[plan.kind] in text
            assert not any(str(i) in text for i in plan.params["stations"])
            for kind, other in ((REPLENISH, COLLECT), (COLLECT, REPLENISH)):
                if plan.kind == kind:
                    swapped = brief.compose([plan], job=JOB, robot=robot, domain="warehouse", channel="rgb_pose",
                                            variant="swapped")
                    assert TEXTS[other] in swapped[0] and TEXTS[kind] not in swapped[0]
    pick = next(p for p in amr_kit.plans_of(1) if p.kind == PICK)
    with pytest.raises(ValueError, match="no swapped brief"):
        brief.compose([pick], job=JOB, robot=robot, domain="warehouse", channel="rgb_pose", variant="swapped")


def test_every_event_resolves_in_every_task(scripts):
    """Contract (plan §2.2; 0.5.0-control): each of the four events resolves, through `events.resolve` over a task's
    delivered rows (their `k` and phase), to the keyframe the plan names: the first transit one (the turn that follows
    the reverse off the charger, whose keyframe before is the last undocking one), the first and the last `at_station`
    ones (absent when the task skipped every stop), and the last `waiting` one, the closing hold's, which stands at the
    charger: a skipped visit waits at T in `waiting` too, so no earlier `waiting` step may be the event. Regression
    caught: an event no system can resolve in a later task, one that lands on the wrong hold, or "got back to the
    charger" resolving to a wait at a blocked bay."""
    mid_waits = 0
    for tl in scripts.values():
        for task in tl.ego.tasks:
            rows = [{"k": kf.k, "status": {"phase": kf.phase}} for kf in tl.ego.keyframes[task.k0:task.k1]]
            got = {wording: resolve(rule, rows) for wording, rule in JOB.events.items()}
            phases = [r["status"]["phase"] for r in rows]
            first_transit = task.k0 + phases.index("transit")
            holds = [task.k0 + i for i, p in enumerate(phases) if p == "at_station"]
            assert got == {"when you left the charger": first_transit,
                           "when you reached your first stop": holds[0] if holds else None,
                           "when you finished at your last stop": holds[-1] if holds else None,
                           "when you got back to the charger": task.k1 - 1}, task.index
            assert tl.keyframe(first_transit - 1).phase == "undocking"
            assert tl.keyframe(task.k1 - 1).base[:2] == HOME and tl.keyframe(task.k1 - 1).phase == "waiting"
            runs = sum(1 for i, p in enumerate(phases) if p == "at_station" and (i == 0 or phases[i - 1] != p))
            docked = sum(d.response == "go" for d in tl.decisions_in(task.k0, task.k1) if d.kind == "before_station")
            assert runs == docked + len(task.plan.params.get("faces", []))
            waits = sum(1 for i, p in enumerate(phases) if p == "waiting" and (i == 0 or phases[i - 1] != p))
            mid_waits += waits > 2  # besides the hold at the start and the closing one
    assert mid_waits > 0  # the fixture skips a visit, so a `waiting` stands before the closing one


def test_the_job_declares_its_bay_zone_and_ladder():
    """Contract (docs/job-amr.md, "The gate"; FD-0.5.0-control-impl §4, §6, §7): the AMR keeps its three decision kinds
    (which families it asks, none since 0.6.0, is `test_jobs_ask_060`'s); the distance ladder is the warehouse's (0.5,
    1, 2, 4 m); the job states subgoals; the bay's zone rule (0.3 m) is the height `job_amr_drops` judges a bay
    blocked by and is the zone of `before_station` alone, and the zone rules pass `questions_zone.check_rules`.
    Regression caught: a ladder of another place's scale, a bay judged at another height than its zone, or a coin
    with no zone to judge (its blocker would be asked)."""
    assert set(JOB.decisions) == {"before_station", "before_undock", "before_crossing"}
    assert JOB.zones == {"before_station": ZoneRule("the bay ahead of you", "bay")}
    assert JOB.zones["before_station"].height_m == drops.ZONE_HEIGHT_M == 0.3
    assert JOB.slot_zones == () and JOB.ladder == (0.5, 1.0, 2.0, 4.0) and JOB.subgoals is not None
    questions_zone.check_rules(JOB)


def test_a_map_without_its_pieces_refuses_the_job():
    """Contract (M3-W §1): the job reads a map by kind, and a map without a charger (or with two), a shelving piece, a
    packing table, a bay on a station's face, or a region of kind `aisle` is refused with a ValueError that names the
    map and the missing kind. Regression caught: a job that plans on a map it cannot work in, or reads a region by its
    name."""
    def without(kind: str, keep: int = 0):
        kept = [f for f in M.furniture if f.kind != kind] + [f for f in M.furniture if f.kind == kind][:keep]
        return replace(M, furniture=tuple(kept))

    for broken, kind in ((without("charger"), "charger"), (without("shelving"), "shelving"),
                         (without("packing_table"), "packing_table"), (without("bay", 3), "bay"),
                         (replace(M, regions=tuple(replace(r, kind="hall") for r in M.regions)), "aisle")):
        with pytest.raises(ValueError, match=f"warehouse_a.*{kind}|{kind}.*warehouse_a"):
            JOB.plan(broken, 1, 2)
    twin = replace(M, furniture=(*M.furniture, replace(next(f for f in M.furniture if f.kind == "charger"), id="F99")))
    with pytest.raises(ValueError, match="exactly one 'charger'"):
        JOB.plan(twin, 1, 2)


def test_a_keep_out_is_an_obstacle_whatever_the_world_holds():
    """Contract (`route.robot_free_space`, additive): each rectangle of `keep_out` is an obstacle the base keeps the
    robot's clearance from (0.4935 m disc + 0.10 m + half a 0.025 m cell's diagonal, 0.6112 m); with none the free space
    is today's. By hand: a 1 x 1 m keep-out at (9, 6.5) frees no cell within 0.61 m of its edge and every cell 0.62 m or
    more from it that was free. Regression caught: a keep-out that is ignored, or that changes the free space when
    empty."""
    w = build_world(M, 101)
    plain = route.robot_free_space(M, w, ROBOTS["amr"])
    same = route.robot_free_space(M, w, ROBOTS["amr"], keep_out=())
    held = route.robot_free_space(M, w, ROBOTS["amr"], keep_out=[amr_kit.Rect(9.0, 6.5, 1.0, 1.0, 0.0)])
    assert (plain.free == same.free).all()
    xs, ys = np.meshgrid(plain.x0 + (np.arange(plain.nx) + 0.5) * plain.res, plain.y0 + (np.arange(plain.ny) + 0.5) *
                         plain.res)
    near = np.hypot(np.maximum(abs(xs - 9.0) - 0.5, 0), np.maximum(abs(ys - 6.5) - 0.5, 0))
    assert not held.free[near < 0.61].any() and plain.free[near < 0.61].any()
    assert (held.free[near >= 0.62] == plain.free[near >= 0.62]).all()


def test_the_keep_outs_are_the_lanes_the_bays_and_the_reverse_corridors():
    """Contract (M3-W §3, with the bays and the corridors added): a task is planned round 12 rectangles, the four
    staging lanes' footprints, each bay's footprint from its station's face to 0.7 m short of T (the table bays 1.2 m
    deep, x 15.5-16.7, centred x 16.1, the planner then keeping 0.612 m from them and T 1.9 m from the face staying
    free) and the reverse corridor's extension behind each station (by hand behind F10: T is x 16.7 - 1.9 = 14.8, its
    rear edge 0.4 m behind it x 14.4, so the extension starts 0.3 m past that, x 14.1, and ends 5 m behind the docked
    rear edge, which is x 16.7 - 0.12 - 0.8 = 15.78, so x 10.78: 3.32 m long, centred x 12.44, 0.58 m across along y
    4.1); and T of every station is free in the world with all of them. Regression caught: a way that crosses a bay or
    sweeps a corridor, or a keep-out that closes a station's T."""
    out = amr_kit.keep_out_rects()
    assert len(out) == 12
    assert out[:4] == amr_kit.lane_rects()
    strip = next(r for r in out[4:8] if abs(r.cy - 4.1) < 1e-9 and r.cx < 15)
    assert (strip.cx, strip.sx, strip.sy) == pytest.approx((12.44, 3.32, 0.58)) and abs(math.cos(strip.yaw)) == 1.0
    bay = next(r for r in out[8:] if abs(r.cy - 4.1) < 1e-9 and r.cx > 15)
    assert (bay.cx, bay.sx, bay.sy) == pytest.approx((16.1, 1.2, 2.2))
    w = build_world(M, 101)
    free = route.robot_free_space(M, w, ROBOTS["amr"], keep_out=out)
    assert all(free.is_free(dock.t) for dock in amr_kit.docks().values())


if __name__ == "__main__":
    print(digest(amr_kit.generated(int(sys.argv[1]), 3)))
