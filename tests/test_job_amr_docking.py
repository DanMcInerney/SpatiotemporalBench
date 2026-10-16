"""`amr/1`'s manoeuvres: the side arrival, the slow straight approach, the reverse along the same line, `loaded`, and
the crossings of a pick run, on the scripts of `test_job_amr` (`amr_kit.generated`, one xdist group). By hand from
`maps/warehouse_a.json`: the packing tables F09 (y 6.5) and F10 (y 4.1) at x 17.1, 0.8 m deep facing west (face x
16.70); the shelving F11 (y 6.4) and F12 (y 4.2) at x 0.4, 0.6 m deep facing east (face x 0.70); the AMR 0.80 m long, so
docked with its front edge 0.12 m off the face its base is 0.52 m from it, and T is 1.9 m from it, so 1.38 m from the
docked base."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

import amr_kit
from stbench.geometry import wrap_angle
from stbench.job_amr_path import crossing

pytestmark = pytest.mark.xdist_group(name="test_job_amr")

M = amr_kit.M
# station -> (docked base, T, heading): by hand from the face, the 0.12 m gap, half the body and 1.9 m: x 16.7 - 0.52 =
# 16.18 and 16.7 - 1.9 = 14.8 for the tables, x 0.7 + 0.52 = 1.22 and 0.7 + 1.9 = 2.6 for the shelving
DOCKS = {"F09": ((16.18, 6.5), (14.8, 6.5), 0.0), "F10": ((16.18, 4.1), (14.8, 4.1), 0.0),
         "F11": ((1.22, 6.4), (2.6, 6.4), math.pi), "F12": ((1.22, 4.2), (2.6, 4.2), math.pi)}


@pytest.fixture(scope="module")
def scripts():
    return {seed: amr_kit.generated(seed, 3) for seed in amr_kit.SEEDS}


def visits(scripts, docked=True):
    """(seed, task, station id, dock, keyframes from the before_station frame to the end of the undocking) of the visits
    that docked; with `docked` False of every visit (a skipped one ends with its wait at T)."""
    for seed, tl in scripts.items():
        for task in tl.ego.tasks:
            for j, dock, response, kfs in amr_kit.visits(tl, task):
                if response == "go" or not docked:
                    yield seed, task, task.plan.params["stations"][j], dock, kfs


def test_docking_is_straight_and_reverses_its_way(scripts):
    """Contract (M3-W §3): each station's docks are the hand ones (docked base, T, heading); from T the approach runs
    straight along the line to the docked pose at the approach speed (0.2 m/s), the hold stands at the docked pose (the
    front edge 0.12 m off the face), and the undocking retraces the line to T with the heading kept the whole time.
    Regression caught: a reverse that swerves, a dock into the station, an approach that turns on the way."""
    seen = set()
    for seed, task, sid, dock, kfs in visits(scripts):
        seen.add(sid)
        docked, t, heading = DOCKS[sid]
        assert (dock.docked, dock.t) == (docked, t) and abs(wrap_angle(dock.heading - heading)) < 1e-6
        assert all(abs(wrap_angle(kf.yaw - heading)) < 1e-6 for kf in kfs[1:]), (seed, task.index, sid)
        assert kfs[0].base[:2] == t and kfs[-1].base[:2] == t and kfs[-1].phase == "undocking"
        line = [kf for kf in kfs if kf.phase in ("approach", "at_station", "undocking")]
        for kf in line:  # on the straight line from T to the docked pose: the other coordinate does not move
            assert abs(kf.base[1] - t[1]) < 1e-6, (seed, task.index, sid)
        approach, hold, undock = ([kf for kf in kfs if kf.phase == p] for p in ("approach", "at_station", "undocking"))
        assert approach[-1].base[:2] == docked and all(kf.base[:2] == docked for kf in hold)
        along = [abs(kf.base[0] - t[0]) for kf in approach]  # metres from T: 0.2 m/s, so 1 m takes 5 s
        assert along == sorted(along) and along[-1] == pytest.approx(1.38, abs=1e-6)
        assert (approach[-1].t - kfs[0].t) == pytest.approx(1.38 / 0.2, abs=0.01)
        back = [abs(kf.base[0] - t[0]) for kf in undock]  # frames each metre from the docked pose: 1.38 m, 0.38 m, T
        assert back == pytest.approx([0.38, 0.0], abs=1e-6) and undock[-1].base[:2] == t
    assert seen == {"F09", "F10", "F11", "F12"}  # every station is docked at somewhere in the scripts


def test_the_arrival_runs_along_the_face(scripts):
    """Contract (M3-W §3): the leg into T runs along the station's face, parallel to it (1e-6 rad), at least 1.0 m long
    (`SIDE_M` 2.0 m, or 1.5 or 1.0 m when the farther waypoint is not free) and straight toward T, so it crosses the
    dock's line, the axis of the reverse corridor behind the docked rear edge, only at T. Regression caught: an arrival
    that comes in along the corridor (M2's failure: the vacuum drove in along it) or turns on the way to T."""
    legs = 0
    for seed, task, sid, dock, kfs in visits(scripts, docked=False):  # a skipped visit arrives the same way
        tl, t = scripts[seed], dock.t
        back = list(reversed(tl.ego.keyframes[task.k0:kfs[0].k + 1]))  # from the before_station frame back
        back = back[:next((i for i, kf in enumerate(back) if kf.phase != "transit"), len(back))]  # just this leg
        leg = [kf for kf in back if kf.base[:2] != t]  # the turn at T stands at T
        ref = (leg[0].base[0] - t[0], leg[0].base[1] - t[1])
        run = []
        for kf in leg:
            d = (kf.base[0] - t[0], kf.base[1] - t[1])
            if abs(d[0] * ref[1] - d[1] * ref[0]) > 1e-6 * math.hypot(*d) or d[0] * ref[0] + d[1] * ref[1] < 0:
                break
            run.append(math.hypot(*d))
        assert abs(ref[0]) < 1e-6 * abs(ref[1]), (seed, task.index, sid)  # along the face: x does not change
        assert run == sorted(run) and run[-1] >= 1.0 - 1e-6, (seed, task.index, sid)
        assert abs(leg[0].base[1] - t[1]) > 1e-6
        legs += 1
    assert legs >= 12  # three scripts of three tasks of two or three stations, and the picks' tables


def test_loaded_flips_only_at_stations(scripts):
    """Contract (M3-W §2, 0.5.0-control): `loaded` is false at the start and true from the end of the first hold made
    (the shelving of a replenish, the first table of a collect, the first pick face of a pick, a skipped station
    making none) to the end of the last planned stop's hold, then false: the keyframe after the hold's last still has
    the old value, and the flips stand between an `at_station` keyframe and the next, of another phase. A skipped last
    stop makes no such hold, so the robot stays loaded to the task's end (after an earlier hold), and a task that
    skips everything never loads. By hand from the responses: the flips are after `holds[0]`, and after `holds[-1]`
    when the last station docked. Regression caught: a status that reveals the world, a load that appears while the
    robot drives, or a load dropped at a stop the robot never made."""
    first = {"replenish": amr_kit.SHELVES, "collect": amr_kit.TABLES}
    checked = ends_loaded = 0
    for tl in scripts.values():
        for task in tl.ego.tasks:
            kfs = tl.ego.keyframes[task.k0:task.k1]
            loaded = [kf.status["loaded"] for kf in kfs]
            flips = [i for i in range(1, len(kfs)) if loaded[i] != loaded[i - 1]]
            holds = [i for i in range(len(kfs)) if kfs[i].phase == "at_station" and kfs[i + 1].phase != "at_station"]
            stations = task.plan.params["stations"]
            last_docked = amr_kit.visits(tl, task)[-1][2] == "go"
            if not holds:
                want = []  # every stop skipped: nothing is ever loaded
            elif last_docked:
                want = [holds[0] + 1, holds[-1] + 1] if len(holds) > 1 else []  # loaded and unloaded at the one hold
            else:
                want = [holds[0] + 1]  # the last planned stop was skipped: loaded to the end
            assert flips == want and not loaded[0], task.index
            assert loaded[-1] == (bool(holds) and not last_docked), task.index
            for i in flips:
                assert kfs[i - 1].phase == "at_station" and kfs[i].phase != "at_station"
            if holds and task.plan.kind in first and amr_kit.visits(tl, task)[0][2] == "go":
                # the first hold of a docked first station is at that kind of station's docked pose
                assert kfs[holds[0]].base[:2] == DOCKS[stations[0]][0]
                assert stations[0] in first[task.plan.kind]
            checked += 1
            ends_loaded += loaded[-1]
    assert checked == 9 and ends_loaded >= 1  # a skipped last stop is in the fixture: the end is loaded there


def test_crossings_mark_aisle_boundaries(scripts):
    """Contract (M3-W §3): a pick run holds one `before_crossing` 0.5 m before its way enters the aisle's region (the
    base in the dock, facing north across y 9.1: y + 0.5 sin(yaw) = 9.1) and one before it leaves (the base in the
    aisle, facing south the same way), in transit; and the rule is the job's own, never an opening's or a region's name:
    on `warehouse_a` with its dock renamed (a region of another kind than the dock) the same polyline gives the same
    crossing, 0.5 m before it by hand (a way from y 6 to y 12 crosses y 9.1 at 3.1 m: 2.6 m from its start; back, 2.9 m
    less 0.5), and a way that crosses nothing raises. Regression caught: a crossing found by an opening (the warehouse
    has none) or by a region's name."""
    picks, exact = 0, 0
    for tl in scripts.values():
        for task in (t for t in tl.ego.tasks if t.plan.kind == "pick"):
            cross = [tl.keyframe(d.k) for d in tl.decisions_in(task.k0, task.k1) if d.kind == "before_crossing"]
            assert len(cross) == 2
            for kf, n in zip(cross, (0, 1), strict=True):
                assert kf.phase == "transit"
                ahead = (9.1 - kf.base[1]) / math.sin(kf.yaw)  # metres to y 9.1 along the heading
                assert 0 < ahead <= 0.5 + 1e-4  # nearer than 0.5 m only when its segment starts nearer
                exact += abs(ahead - 0.5) < 1e-4
                assert (tl.region_of(kf.base[:2]) == "R1") == (n == 0) and math.sin(kf.yaw) * (1 - 2 * n) > 0.1
            picks += 1
    assert picks >= 3 and exact >= 3
    renamed = replace(M, regions=tuple(replace(r, name="pack area", kind="pack area") if r.id == "R1" else r
                                       for r in M.regions))
    for m in (M, renamed):
        assert not m.openings
        assert crossing(m, [(8.0, 6.0), (8.0, 12.0)], into=True) == (0, pytest.approx(2.6))
        assert crossing(m, [(8.0, 12.0), (8.0, 6.0)], into=False) == (0, pytest.approx(2.4))
        nearer = crossing(m, [(8.0, 6.0), (8.0, 9.4), (8.0, 12.0)], into=True)  # the segment's start is nearer
        assert nearer == (0, pytest.approx(2.6))
        with pytest.raises(ValueError, match="crosses no entry"):
            crossing(m, [(8.0, 12.0), (8.0, 6.0)], into=True)
        with pytest.raises(ValueError, match="crosses no exit"):
            crossing(m, [(8.0, 3.0), (8.0, 6.0)], into=False)
