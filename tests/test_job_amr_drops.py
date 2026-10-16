"""`amr/1`'s balance from the world: the bay coin and the rear drop (`job_amr_drops`), on the scripts of `test_job_amr`
(`amr_kit.generated`, one xdist group). By hand from `maps/warehouse_a.json`: the bay of a station is 1.6 m deep and 2.2
m wide, touching its face; its zone is that inset by 0.05 m (1.5 x 2.1 m); the docking line runs through its middle; a
drum is a 0.58 m square 0.88 m high; T is 1.9 m from the face, so behind it along the dock's line the AMR's rear edge at
T is 0.4 m farther, the drop starts 0.3 m past that (0.7 m from T) and the corridor ends 5 m behind the docked rear edge
(which is 0.12 + 0.8 = 0.92 m from the face, so the end is 5.92 m from it, 4.02 m behind T)."""

from __future__ import annotations

import math
from unittest import mock

import pytest

import amr_kit
from stbench import job_amr_drops as drops
from stbench.geometry import Rect, rect_distance, rect_inside_rect, rects_overlap, to_local
from stbench.job_amr_path import bay_keep_out
from stbench.job_amr_plan import pieces
from stbench.seeding import make_rng
from stbench.timeline_ego import Keyframe

pytestmark = pytest.mark.xdist_group(name="test_job_amr")

M, AMR = amr_kit.M, amr_kit.AMR


@pytest.fixture(scope="module")
def scripts():
    return {seed: amr_kit.generated(seed, 3) for seed in amr_kit.SEEDS}


def within(hits: int, n: int, p: float) -> bool:
    """Whether `hits` of `n` independent draws lies within 4 standard errors of the chance `p`."""
    return abs(hits / n - p) <= 4 * math.sqrt(p * (1 - p) / n)


def draw(family: str, seed: int, i: int, j: int) -> float:
    """The documented stream of a coin: (map, seed, job, family, task, visit), its first draw."""
    return float(make_rng("warehouse_a", seed, "amr/1", family, i, j).random())


def edge_time(tl, task) -> float:
    """The instant a task's moves happen: its start less 30 s to the whole second below (0 for the first task)."""
    return 0.0 if task.index == 0 else math.floor(tl.keyframe(task.k0).t - 30.0)


def floor_things(tl, t: float, kinds=("pallet", "drum")) -> list[tuple[object, Rect]]:
    """(entity, footprint) of the pallets and drums that are put on the bare floor at time `t`."""
    return [(e, tl.footprint(e.id, t)) for e in tl.entities if e.kind in kinds and any(
        p.t0 == t and p.on is None and p.pos[2] == 0.0 for p in tl.placements_of(e.id))]


def test_the_bay_coin_holds_both_ways(scripts):
    """Contract (design §8.5, 0.5.0-control): the coin of visit j of task i is one draw from the stream (map, seed,
    "amr/1", "bay", i, j), 0.5 over 1,000 seeds within 4 standard errors, and reads no world; on the fixture, at every
    `before_station` the bay's zone (its footprint inset 0.05 m) holds exactly one drum when its coin says blocked and
    nothing at all when it says clear; that drum stands wholly inside the bay's keep-out (the 1.2 m from the station's
    face that the planner keeps out of) with its centre within 0.10 m of the docking line, so the body cannot pass it;
    and the decision's response is "skip" for exactly the visits that hold a drum and "go" for the rest. Both outcomes
    occur. Regression caught: a balance that the world, not the coin, decides, a drum beside the docking line (the
    robot would dock past it), a drum outside the keep-out (the planner could meet it), or a response that does not
    follow the bay."""
    assert within(sum(draw("bay", s, 0, 0) < 0.5 for s in range(1000)), 1000, 0.5)
    assert all(drops.bay_blocked(M, s, i, j) == (draw("bay", s, i, j) < 0.5)
               for s in (1, 2, 3, 4) for i in range(3) for j in range(3))
    blocked = clear = 0
    for seed, tl in scripts.items():
        for task in tl.ego.tasks:
            for j, dock, response, kfs in amr_kit.visits(tl, task):
                zone, t = amr_kit.bay_zone(dock), kfs[0].t
                inside = [(e, tl.footprint(e.id, t)) for e in tl.entities if e.mobility == "carried"
                          and tl.placement_of(e.id, t) is not None and rects_overlap(tl.footprint(e.id, t), zone)]
                assert drops.bay_blocked(M, seed, task.index, j) == (response == "skip"), (seed, task.index, j)
                if response == "skip":
                    blocked += 1
                    assert [e.kind for e, _ in inside] == ["drum"], (seed, task.index, j, inside)
                    rect = inside[0][1]
                    bay = next(b for b in pieces(M, "bay") if b.id == dock.bay)
                    assert rect_inside_rect(rect, bay_keep_out(bay)), (seed, task.index, j)
                    assert abs(to_local(zone, rect.cx, rect.cy)[1] - to_local(zone, *dock.docked)[1]) <= 0.10 + 1e-6
                else:
                    clear += 1
                    assert response == "go" and inside == [], (seed, task.index, j, inside)
    assert blocked >= 3 and clear >= 3


def test_a_drum_in_a_bay_changes_no_free_cell():
    """Contract (design §8.5, 0.5.0-control): the drum a blocked bay gets stands inside `bay_keep_out`, which every task
    is planned round, so the planner cannot see it: on the world of seed 101 (which holds two drums off the lanes) with
    the job's keep-outs, putting a drum in each of two bays across its docking line gives exactly the free space of the
    same world with those drums gone
    (the robot's free cells are the same grid cell for cell), so a skipped visit is no detour for the path. Regression
    caught: a drum placed beyond the keep-out (T's leg or the arrival closed), or a planner that reads the drum."""
    from process_kit import copy_world  # noqa: PLC0415
    from stbench import SEED_KEY  # noqa: PLC0415
    from stbench.generate import build_world, light_starts  # noqa: PLC0415
    from stbench.route import robot_free_space  # noqa: PLC0415

    with amr_kit.planned():
        base = build_world(M, 101)
        light_starts(base, M, make_rng(M.name, 101, SEED_KEY, "light_start"), frozenset({"charger"}))
        here, bays = amr_kit.docks(), {b.id: b for b in pieces(M, "bay")}
        w, taken = copy_world(base), set()
        want = [(j, bays[d.bay], d.docked) for j, d in enumerate(here[sid] for sid in ("F09", "F10", "F11"))]
        done = drops.block_bays(w, M, 101, 0, 500.0, want, taken)
        assert done == [0, 1] and len(taken) == 2  # the third finds no drum left: it is not blocked
        gone = w
        for eid in sorted(taken):
            gone = gone.without(eid)
        keep = amr_kit.keep_out_rects()
        with_drums = robot_free_space(M, w, AMR, 600.0, keep_out=keep)
        without = robot_free_space(M, gone, AMR, 600.0, keep_out=keep)
        assert (with_drums.free == without.free).all()
        plain = robot_free_space(M, w, AMR, 600.0)  # the control: without the keep-outs the drums close cells
        assert (plain.free != robot_free_space(M, gone, AMR, 600.0).free).any()


def test_the_path_to_a_decision_ignores_its_coin():
    """Contract (design §8.5, 0.5.0-control): the keyframes up to and including a `before_station` are the same
    whichever way its coin falls: forcing the coin of visit j of task i blocked or clear (every other coin the real
    one) gives byte-identical keyframes (time, pose, phase and status) for the whole script up to that decision, the
    same decision list up to it, and then "skip" against "go" at the decision itself; the worlds differ by the drum.
    The check is made on the first visit of the first task and the second visit of the second.
    Regression caught: a planner or a path that reads the bay's drum, a status that tells the coin early, or an
    earlier decision that moves."""
    for i, j in ((0, 0), (1, 1)):
        blocked, clear = amr_kit.forced(101, 3, i, j, True, rear=False), amr_kit.forced(101, 3, i, j, False, rear=False)
        assert blocked.world_hash != clear.world_hash
        db, dc = amr_kit.asked(blocked, i, j), amr_kit.asked(clear, i, j)
        assert db.k == dc.k and (db.response, dc.response) == ("skip", "go")
        assert blocked.ego.keyframes[:db.k + 1] == clear.ego.keyframes[:dc.k + 1]
        assert [d for d in blocked.ego.decisions if d.k < db.k] == [d for d in clear.ego.decisions if d.k < dc.k]
        assert blocked.ego.keyframes[db.k + 1].phase == "waiting" and clear.ego.keyframes[dc.k + 1].phase == "approach"


def test_drops_are_seeded_and_clear_of_the_path(scripts):
    """Contract (design §8.5): the coin of a rear drop is one draw from the stream (map, seed, "amr/1", "rear", i, j),
    0.6 over 1,000 seeds within 4 standard errors; on the fixture at least three in four of the visits whose coin says
    so (a visit that docked: a skipped one has no corridor to reverse along and its coin is not read; every corridor
    lies on open floor, 0.4 m or more beyond the lanes' band, which ends at y 3.4, so a drop finds room unless a thing
    stands in the way; one that finds none is counted, `rear_drops` returns it) have a pallet or a
    drum put on the floor at the task's instant (its start less 30 s, to the second) that overlaps the corridor's
    extension, lies from 0.7 m to 4.02 m behind T along the dock's line, is off every staging lane and at least 0.25 m
    (less a millimetre) from the AMR's body along every point of the task's path; and the first task's keyframes are
    byte for byte those of the same script with the drops off. Regression caught: a drop the robot drives through, a
    drop on a lane, or a path that reveals a drop."""
    assert within(sum(draw("rear", s, 0, 0) < 0.6 for s in range(1000)), 1000, 0.6)
    assert all(drops.rear_dropped(M, s, i, j) == (draw("rear", s, i, j) < 0.6)
               for s in (1, 2, 3, 4) for i in range(3) for j in range(3))
    chosen = realised = 0
    lanes = amr_kit.lane_rects()
    more = {seed: amr_kit.generated(seed, 3) for seed in range(104, 107)}  # a skipped visit has no drop: more scripts
    for seed, tl in {**scripts, **more}.items():
        for task in tl.ego.tasks:
            put = floor_things(tl, edge_time(tl, task))
            for j, dock, _ in amr_kit.docked_visits(tl, task):
                if not drops.rear_dropped(M, seed, task.index, j):
                    continue
                chosen += 1
                d = dock.out
                hits = []
                for _, rect in put:
                    depth = [(c[0] - dock.t[0]) * d[0] + (c[1] - dock.t[1]) * d[1] for c in _corners(rect)]
                    if min(depth) >= 0.7 - 1e-6 and max(depth) <= 4.02 + 1e-6 and abs(to_local(
                            Rect(*dock.t, 1.0, 0.58, math.atan2(d[1], d[0])), rect.cx, rect.cy)[1]) < 0.58:
                        hits.append(rect)
                realised += bool(hits)  # a world with no room for one is counted by `rear_drops`, not hidden
                for rect in hits:
                    assert not amr_kit.overlapping(rect, lanes)
                    gap = amr_kit.path_gap(tl, task, lambda t, rect=rect: [rect])
                    assert gap >= 0.25 - 1e-3, (seed, task.index, j, gap)
    assert chosen >= 6 and 4 * realised >= 3 * chosen  # seeds 101-106: the coin chose 8 of the docked visits
    kw = {"coin": False}
    assert amr_kit.quiet(101, 1, rear=True, **kw).ego_hash == amr_kit.quiet(101, 1, rear=False, **kw).ego_hash


def test_a_rear_drop_lands_behind_a_station_in_the_open_and_is_counted_when_it_cannot():
    """Contract (M3-W §4): with every coin saying drop and nothing swept, one thing is moved into the corridor's
    extension behind each of the four stations (open floor, off the lanes): a pallet or a drum, on the floor from the
    instant `t`, from 0.7 m to 4.02 m behind T along the dock's line (by hand behind F10: x 14.1 to 10.78 at y 4.1),
    over the corridor, and the call counts no miss; with no pallet or drum left to move it counts both as misses and
    moves nothing. Regression caught: a drop that lands outside its extension, a visit that silently gets none, or a
    thing moved twice."""
    from process_kit import copy_world  # noqa: PLC0415
    from stbench import SEED_KEY  # noqa: PLC0415
    from stbench.generate import build_world, light_starts  # noqa: PLC0415

    with amr_kit.planned():
        base = build_world(M, 101)
        light_starts(base, M, make_rng(M.name, 101, SEED_KEY, "light_start"), frozenset({"charger"}))
        w, here = copy_world(base), amr_kit.docks()
        stations = [here[sid] for sid in ("F09", "F10", "F11", "F12")]
        with mock.patch.object(drops, "rear_dropped", lambda m, seed, i, j: True):
            missed = drops.rear_drops(w, M, 101, 0, 500.0, stations, [], AMR, set())
            assert missed == 0
            moved = [eid for eid, c in w.carried.items() if c.spot != base.carried[eid].spot]
            assert len(moved) == 4
            for dock in stations:
                strip = drops.extension(dock, AMR)
                near = [eid for eid in moved if rects_overlap(_footprint(w, eid), strip)]
                assert len(near) == 1, dock.station
                c = w.carried[near[0]]
                assert c.spec.kind in ("pallet", "drum") and c.spot.pos[2] == 0.0
                depth = [(p[0] - dock.t[0]) * dock.out[0] + (p[1] - dock.t[1]) * dock.out[1]
                         for p in _corners(_footprint(w, near[0]))]
                assert min(depth) >= 0.7 - 1e-6 and max(depth) <= 4.02 + 1e-6, (dock.station, depth)
            assert {p.t0 for p in w.ledger.finish()[0] if p.entity in moved} >= {500.0}
            every = {eid for eid, c in w.carried.items() if c.spec.kind in ("pallet", "drum")}
            assert drops.rear_drops(copy_world(base), M, 101, 0, 500.0, stations, [], AMR, every) == 4


def test_every_reverse_corridor_runs_over_open_floor_off_the_lanes():
    """Contract (M3-W §1, the open-corridor map): the reverse corridor's extension behind each of the four stations
    (0.58 m across, along the dock's line) meets no staging lane, so a rear drop finds room and a lane's pallet is never
    the first thing `rear_clear` meets. By hand: the lanes are 2.8 m long across y centred y 2.0, so they end at y 3.4;
    the corridors run along y = the station's y +- 0.29: F09 6.5 (6.21 to 6.79, 6.21 - 3.4 = 2.81 m from the lanes), F11
    6.4 (6.11 to 6.69, 2.71 m), F10 4.1 (3.81 to 4.39, 0.41 m) and F12 4.2 (3.91 to 4.49, 0.51 m). Regression caught:
    a station moved back into the lanes' band, where a corridor over a lane left a rear drop no room (the first gate
    realised 0.32 of the 0.61 drops wanted)."""
    lanes = amr_kit.lane_rects()
    gaps = {sid: min(rect_distance(drops.extension(dock, AMR), lane) for lane in lanes)
            for sid, dock in amr_kit.docks().items()}
    assert gaps == pytest.approx({"F09": 2.81, "F10": 0.41, "F11": 2.71, "F12": 0.51})


def _footprint(w, eid: str) -> Rect:
    c = w.carried[eid]
    return Rect(c.spot.pos[0], c.spot.pos[1], c.spec.size[0], c.spec.size[1], c.spot.yaw)


def _corners(rect: Rect) -> list[tuple[float, float]]:
    from stbench.geometry import corners  # noqa: PLC0415

    return corners(rect)


def test_the_swept_bodies_are_the_robots_by_hand():
    """Contract: `swept_bodies` gives, between consecutive keyframes, the body's rectangle along a straight run (forward
    or reversing: the heading is along the line), the body's rectangle for a hold, and the box round the circumscribed
    square for a turn in place (0.987 m, the diagonal of 0.80 x 0.58). By hand: (0, 0) to (1, 0) at heading 0 is a 1.80
    x 0.58 m rectangle centred (0.5, 0); back again at heading 0 (reversing) the same, turned pi; a hold is 0.80 x 0.58;
    a turn is 0.987 m square. Regression caught: a drop that clears the keyframes' rectangles but not the path between
    them."""
    def kf(k, xy, yaw):
        return Keyframe(k, float(k), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (*xy, 0.0), yaw, 0.0, 0, "transit", {})

    diag = math.hypot(0.80, 0.58)
    swept = drops.swept_bodies([kf(0, (0, 0), 0.0), kf(1, (1, 0), 0.0), kf(2, (0, 0), 0.0), kf(3, (0, 0), 0.0),
                                kf(4, (0, 0), math.pi / 2)], AMR)
    assert swept[0] == pytest.approx(Rect(0.5, 0.0, 1.80, 0.58, 0.0))
    assert swept[1] == pytest.approx(Rect(0.5, 0.0, 1.80, 0.58, math.pi))  # reversing: the run is west, the heading 0
    assert swept[2] == pytest.approx(Rect(0.0, 0.0, 0.80, 0.58, 0.0))
    assert swept[3] == pytest.approx(Rect(0.0, 0.0, diag, diag, 0.0))
    assert rect_distance(Rect(5, 5, 0.5, 0.5, 0.0), swept[0]) > 4
