"""`amr/1`'s balance from the world (design §8.5): the bay coin, the response it brings and the rear drop.

A constant answer to a yes/no or a distance question must earn nothing, so the world is made to vary by coins that read
no world: a station visit's bay is blocked or clear by a coin, and a pallet or a drum is left in the reverse corridor
behind some visits. Every move happens at `t_edge` = the task's start less `changes.EDGE_BUFFER` (0 for task 0), inside
the gap and after every change the gap's processes made, as the processes' own moves do; never onto a staging lane (they
hold the lanes' loads), and no thing taken off one.

**The bay coin and the skip.** For visit j of task i, `make_rng(m.name, seed, JOB_ID, "bay", i, j).random() <
BAY_COIN_P` says the bay is blocked, else clear; the draw reads no world. It is enforced both ways in two steps, so the
path to the decision is the same whatever the coins say. `clear_bays` runs before the task is planned: every thing
standing in the zone of a bay the task visits (the bay's footprint inset by `EDGE_M`, as `zones.zones_of` insets it)
moves out by its start's law (`world_processes_home.move_item`). `block_bays` runs after the free space is computed and
before the route is built: each visit whose coin says blocked gets one of the world's drums (0.88 m tall, far over the
zone's 0.3 m) moved into its bay across the docking line, centred within `LINE_SPREAD_M` of it, and wholly inside the
bay's keep-out (`job_amr_path.bay_keep_out`), which the planner keeps out of whatever stands in it: the free space was
computed before the drum moved, and a drum inside the keep-out would change none of it. Docking there is plainly
impossible, so the route skips the visit (`job_amr_path.visit`): it stops at T, waits `skip_wait_s` and drives on. That
is the scripted robot's response to the zone's state, after the decision; `block_bays` returns the visits it blocked
and the script writes "skip" for them and "go" for the rest. The drum skips `clutter_reach.FloorGuard`: to the
planner's circumscribed disc any drum in a bay closes the docking approach, and the next task that visits the bay
clears it before it plans. A tote (0.32 m) is never the variant: its top is 2 cm over the threshold. A visit whose coin
says blocked and that gets no drum (none left, or no spot) is not blocked: it docks and the response is "go".

**The rear drop.** With `REAR_DROP_P`, after the task is planned, a pallet or a drum is moved (`clutter.floor_spot`,
which keeps the floor connected) into the reverse corridor's extension behind a docked visit, from T's rear edge plus
`EXTENSION_M` to the corridor's end (`REAR_CAP_M` behind the docked rear edge), clear by `WORLD_CLEARANCE_M` of every
body the task sweeps, where nothing of the task passes. A skipped visit has no corridor to reverse along: no drop."""

from __future__ import annotations

import math

import numpy as np

from . import clutter
from .changes import FURNITURE_MARGIN, World
from .geometry import Point, Rect, rect_distance, rect_inside_rect, rects_overlap, to_local, to_world
from .job_amr_path import Dock, bay_keep_out, extension
from .job_amr_plan import BAY_COIN_P, JOB_ID, REAR_DROP_P
from .mapspec import Furniture, MapSpec
from .placement import ITEM_MARGIN, Spot, free_of, occupied_entry
from .robots import Robot
from .seeding import make_rng
from .timeline_ego import Keyframe
from .world_processes_home import move_item

EDGE_M = 0.05  # a zone is a piece's footprint inset by this (`zones.zones_of`)
ZONE_HEIGHT_M = 0.3  # the bay zone rule's height (`job_amr.SPEC.zones`): a drum (0.88 m) is far over it, a tote 2 cm
LINE_SPREAD_M = 0.10  # a bay's drum is centred this near the docking line at most: the body cannot pass it
SKIP_WAIT_S = (10.0, 20.0)  # a skipped visit's wait at T, drawn from a stream of its own (`skip_wait_s`)
DRUM, PALLET = "drum", "pallet"
DROP_KINDS = (PALLET, DRUM)
LANE_KIND = "staging_lane"  # a piece kind: things on one are the processes' alone
PUT_TRIES, SPOT_TRIES, THINGS_TRIED = 30, 3, 4


def bay_blocked(m: MapSpec, seed: int, i: int, j: int) -> bool:
    """Whether the coin says visit j of task i finds its bay blocked: one draw from a stream of its own, no world."""
    return float(make_rng(m.name, seed, JOB_ID, "bay", i, j).random()) < BAY_COIN_P


def skip_wait_s(m: MapSpec, seed: int, i: int, j: int) -> float:
    """How long the robot waits at T before skipping visit j of task i: one draw from a stream of its own, so no other
    draw moves whether the visit is skipped or not."""
    return float(make_rng(m.name, seed, JOB_ID, "hold", i, j).uniform(*SKIP_WAIT_S))


def rear_dropped(m: MapSpec, seed: int, i: int, j: int) -> bool:
    """Whether the coin says a pallet or a drum is left behind visit j of task i (when a spot holds it)."""
    return float(make_rng(m.name, seed, JOB_ID, "rear", i, j).random()) < REAR_DROP_P


def zone_rect(bay: Furniture) -> Rect:
    """The bay's zone: its footprint inset by `EDGE_M` on every side."""
    return Rect(bay.center[0], bay.center[1], bay.size[0] - 2 * EDGE_M, bay.size[1] - 2 * EDGE_M, bay.yaw)


def _on_lane(w: World, eid: str) -> bool:
    lanes = {e.id for e in w.entities if e.kind == LANE_KIND}
    return w.carried[eid].spot.slot.furniture_entity in lanes


def clear_bays(w: World, m: MapSpec, rng: np.random.Generator, t: float, bays: list[Furniture]) -> int:
    """Move out of each bay's zone every carried thing standing in it (a footprint over the zone: nothing on a station's
    top reaches it, the zone being inset), by the start's law; how many found no spot and stayed."""
    stuck = 0
    for bay in bays:
        zone = zone_rect(bay)
        for eid in sorted(w.carried):
            c = w.carried[eid]
            if rects_overlap(occupied_entry(c.spec, c.spot.pos, c.spot.yaw)[0], zone):
                stuck += not move_item(w, m, rng, eid, t)
    return stuck


def _free_for(w: World, eid: str, box: Rect, height: float) -> bool:
    """Whether a thing's box is clear of the carried things, the clutter and the light pieces of the world without it.
    """
    rest = w.without(eid)
    return (free_of(box, 0.0, height, rest.occupied(), ITEM_MARGIN)
            and not any(rects_overlap(box, r, FURNITURE_MARGIN) for r in rest.light_rects.values()))


def _place(w: World, m: MapSpec, eid: str, t: float, x: float, y: float, yaw: float) -> None:
    """The thing on the floor at (x, y) from `t` on: its ledger placement and its record."""
    w.ledger.place(eid, t, (x, y, 0.0), yaw, None)
    w.carried[eid].spot = Spot(clutter.floor_slot(m, x, y), (x, y, 0.0), yaw)


def _candidates(w: World, kinds: tuple[str, ...], taken: set[str]) -> list[str]:
    """The ids of the carried things of these kinds that are not on a lane and not already moved this task."""
    return [eid for eid in sorted(w.carried) if w.carried[eid].spec.kind in kinds and eid not in taken
            and not _on_lane(w, eid)]


def block_bays(w: World, m: MapSpec, seed: int, i: int, t: float, blocked: list[tuple[int, Furniture, Point]],
               taken: set[str]) -> list[int]:
    """Move a drum into each bay of `blocked` ((visit j, the bay, a point on its docking line)): across the docking
    line, centred within `LINE_SPREAD_M` of it, wholly inside the bay's keep-out and so inside its zone, clear of other
    things. The drums are added to `taken`. The visits that got one, in order: a visit with no drum left or no spot is
    not blocked."""
    done = []
    for j, bay, line in blocked:
        rng = make_rng(m.name, seed, JOB_ID, "bay_at", i, j)
        drums = _candidates(w, (DRUM,), taken)
        if not drums:
            continue
        eid = drums[int(rng.integers(len(drums)))]
        size = w.carried[eid].spec.size
        zone, keep = zone_rect(bay), bay_keep_out(bay)
        line_y = to_local(zone, *line)[1]  # across the bay
        k_x = to_local(zone, keep.cx, keep.cy)[0]  # the keep-out has the bay's yaw
        x_lo = max(-zone.sx / 2, k_x - keep.sx / 2) + size[0] / 2  # along the bay: inside the zone and the keep-out
        x_hi = min(zone.sx / 2, k_x + keep.sx / 2) - size[0] / 2
        if x_lo > x_hi:
            continue
        for _ in range(PUT_TRIES):
            lx = float(rng.uniform(x_lo, x_hi))
            ly = line_y + float(rng.uniform(-LINE_SPREAD_M, LINE_SPREAD_M))
            if abs(ly) > zone.sy / 2 - size[1] / 2 + 1e-9:
                continue
            x, y = to_world(zone, lx, ly)
            box = Rect(x, y, size[0], size[1], zone.yaw)
            if rect_inside_rect(box, keep) and _free_for(w, eid, box, size[2]):
                _place(w, m, eid, t, x, y, zone.yaw)
                taken.add(eid)
                done.append(j)
                break
    return done


def swept_bodies(keyframes: list[Keyframe], robot: Robot) -> list[Rect]:
    """The floor the robot's body covers over a task: between consecutive keyframes the body's rectangle along the
    straight run (it always drives along its heading, forward or reversed), a hold the body's rectangle, and where it
    turns or changes heading between two keyframes the box round the circumscribed square of both ends."""
    body = robot.body
    side = math.hypot(body.length_m, body.width_m)  # the circumscribed disc's bounding square
    out = []
    for a, b in zip(keyframes, keyframes[1:], strict=False):
        (ax, ay), (bx, by) = a.base[:2], b.base[:2]
        run, turned = math.hypot(bx - ax, by - ay), abs(math.remainder(a.yaw - b.yaw, math.tau)) > 1e-9
        if not turned and run < 1e-9:
            out.append(Rect(ax, ay, body.length_m, body.width_m, a.yaw))
        elif not turned and abs(math.sin(a.yaw - math.atan2(by - ay, bx - ax))) < 1e-6:
            along = math.atan2(by - ay, bx - ax)
            out.append(Rect((ax + bx) / 2, (ay + by) / 2, run + body.length_m, body.width_m, along))
        else:
            out.append(Rect((ax + bx) / 2, (ay + by) / 2, abs(bx - ax) + side, abs(by - ay) + side, 0.0))
    return out


def rear_drops(w: World, m: MapSpec, seed: int, i: int, t: float, docks: list[Dock | None], swept: list[Rect],
               robot: Robot, taken: set[str]) -> int:
    """For each docked visit whose coin says so (`docks` has visit j's dock at index j, None for a skipped visit: it
    has no corridor and its coin is not read), move a pallet or a drum into the corridor's extension (`extension`):
    `floor_spot` wholly inside the extension widened by the thing's diagonal, clear of every swept body, and the thing
    over the corridor itself. Things are tried in a drawn order, at most `THINGS_TRIED` (a pallet does not fit where a
    drum does). Returns how many visits the coin chose and no thing or spot served."""
    missed = 0
    for j, dock in enumerate(docks):
        if dock is None or not rear_dropped(m, seed, i, j):
            continue
        rng = make_rng(m.name, seed, JOB_ID, "rear_at", i, j)
        things = _candidates(w, DROP_KINDS, taken)
        strip = extension(dock, robot)
        placed = False
        for n in rng.permutation(len(things))[:THINGS_TRIED] if things else ():
            eid = things[int(n)]
            size = w.carried[eid].spec.size
            area = Rect(strip.cx, strip.cy, strip.sx, strip.sy + math.hypot(size[0], size[1]), strip.yaw)
            near = [r for r in swept if rect_distance(r, area) < 1.0]
            for _ in range(SPOT_TRIES):
                at = clutter.floor_spot(w.without(eid), m, rng, size, area=area, avoid=near, t=t)
                if at is None:
                    break
                if rects_overlap(Rect(at[0], at[1], size[0], size[1], at[2]), strip):
                    _place(w, m, eid, t, *at)
                    taken.add(eid)
                    placed = True
                    break
            if placed:
                break
        missed += not placed
    return missed
