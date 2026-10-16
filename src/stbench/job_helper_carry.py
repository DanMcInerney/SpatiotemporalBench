"""`helper/1`'s real carry: the thing a task moves is a world entity that leaves its source and appears at its target.

For a task, once its two reach poses are known (`job_helper._spot`):
- **The thing** (`choose_thing`): the carried item on the source piece nearest the source window's centre and within
  `GRASP_NEAR_M` of it, small enough to lie on the target surface and with a place when the target window is blocked
  (no draw); when the piece holds none, one such item of the world that lies on a raised surface is put in the window at
  `t`, `EDGE_BUFFER` before the task starts (a seeded draw, `make_rng(map, seed, JOB_ID, "thing", i)`, among those whose
  kind lists the source surface as a receptacle, then the rest); failing that, the nearest on the piece at any distance,
  or one put anywhere on the piece's surfaces.
- **The coin and the response** (`job_helper_spot`): the target window is set clear or blocked at `t`, and the response
  follows the world: `go` puts the thing at the window's centre, `elsewhere` at the free spot of the same surface
  nearest the window and outside it, else on the room's other counted piece, in id order (the helper drives there from
  its decision pose and puts it down); a task with no place for it raises ValueError (the seed is skipped).
- **The placements** (`commit`): the thing's placement on the source ends at the task's last `grasping` keyframe (it is
  hidden from then on: the arm and the payload are not drawn), and a new one starts at the last `placing` keyframe on
  the surface the helper put it on.
`Task.targets` of a task is (source room, target room, the thing, the source piece's entity, the target piece's entity):
the pieces are the ones the reach poses face, which the world may move from the plan's preferred ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .catalog import FLOOR_ITEM_Z
from .changes import EDGE_BUFFER, World
from .geometry import Rect, rect_distance
from .job_helper_plan import JOB_ID, park_pose, qualified, spot_blocked
from .job_helper_reach import ROBOT_ID, Pose, reach_pose
from .job_helper_spot import ELSEWHERE, GO, Window, enforce, fits, place_spot, response, window_at
from .keyframes import Frame
from .mapspec import MapSpec
from .placement import Spot, item_rect, occupied_entry
from .robots import ROBOTS
from .route import FreeSpace
from .seeding import make_rng
from .timeline_ego import TaskPlan

Other = tuple[Window, Pose]  # a piece of the target room besides the target, its window and its reach pose
GRASP_NEAR_M = 0.3  # a thing is within the arm's reach of the source window when it lies this near it
CANDIDATES = 12  # items a thing put on the source piece is looked for among


@dataclass(frozen=True)
class Carry:
    """What a task carries and where it ends."""

    eid: str  # the thing
    pieces: tuple[str, str]  # the entities of the source piece and of the target piece the helper faces
    response: str  # `job_helper_spot.GO` or `ELSEWHERE`, the `before_place` decision's
    spot: Spot  # where the thing ends
    alt: Pose | None  # the pose at the room's other piece when the thing goes there, else None


def edge_time(start: float, i: int) -> float:
    """When the world is set for task `i`: `EDGE_BUFFER` before it starts (at 0 for the first)."""
    return 0.0 if i == 0 else start - EDGE_BUFFER


def prepare(m: MapSpec, w: World, free: FreeSpace, mask: np.ndarray, plan: TaskPlan, seed: int, i: int,
            source: Pose, target: Pose, t: float) -> Carry:
    """The thing, the coin, the response and the place of task `i` (the module docstring), changing the world `w` at `t`
    only as it says. ValueError when no thing can be carried or no place holds it."""
    src, dst = window_at(m, w, source), window_at(m, w, target)
    others = _other_pieces(m, w, free, mask, plan, dst, target)
    eid = choose_thing(w, src, dst, others, make_rng(m.name, seed, JOB_ID, "thing", i), t)
    enforce(w, m, make_rng(m.name, seed, JOB_ID, "spot_world", i), dst, spot_blocked(m.name, seed, i), t, eid)
    answer, alt = response(w, dst), None
    spot = place_spot(w, eid, dst, outside=answer == ELSEWHERE)
    if spot is None and answer == GO:
        raise ValueError(f"the {w.carried[eid].spec.kind} has no spot on the free window of {dst.piece.id}")
    if spot is None:
        for win, pose in others:
            if (spot := place_spot(w, eid, win, outside=False)) is not None:
                alt = pose
                break
    if spot is None:
        raise ValueError(f"no place for the {w.carried[eid].spec.kind} on {dst.piece.id} (occupied) or on the other "
                         f"pieces of the room: {', '.join(win.piece.id for win, _ in others) or 'none'}")
    return Carry(eid, (src.slot.furniture_entity, dst.slot.furniture_entity), answer, spot, alt)


def choose_thing(w: World, src: Window, dst: Window, others: list[Other], rng: np.random.Generator,
                 t: float) -> str:
    """The thing of the task (the module docstring). An item is a candidate when it is small enough for both surfaces
    and has a place when the target window is blocked (`placeable`). In turn: the nearest candidate on the source piece
    within `GRASP_NEAR_M` of the window; one put within that reach of the window; the nearest candidate on the piece at
    any distance; one put anywhere on the piece's surfaces. ValueError when there is none."""
    piece = src.slot.furniture_entity
    on_piece = sorted((c for c in w.carried.values() if c.spot.slot.furniture_entity == piece
                       and fits(c.spec.size, dst.slot) and placeable(w, c.entity, dst, others)),
                      key=lambda c: (math.hypot(c.spot.pos[0] - src.rect.cx, c.spot.pos[1] - src.rect.cy), c.entity))
    near = [c for c in on_piece if rect_distance(_box(w, c.entity), src.rect) <= GRASP_NEAR_M]
    if near:
        return near[0].entity
    for in_reach in (True, False):
        if (eid := _put_on(w, src, dst, others, rng, t, in_reach)) is not None:
            return eid
        if in_reach and on_piece:
            return on_piece[0].entity
    raise ValueError(f"no thing for the window on {src.piece.id}: it holds none and no item can be put there that "
                     "fits it and the target")


def placeable(w: World, eid: str, dst: Window, others: list[Other]) -> bool:
    """Whether the item has a place when the target window is blocked: a free spot on the target surface outside the
    window, else a free spot at one of the room's other pieces."""
    return place_spot(w, eid, dst, outside=True) is not None or any(
        place_spot(w, eid, win, outside=False) is not None for win, _ in others)


def _put_on(w: World, src: Window, dst: Window, others: list[Other], rng: np.random.Generator, t: float,
            in_reach: bool) -> str | None:
    """One item of the world that lies on a raised surface, small enough for the source and target surfaces and
    placeable on the target, put on the source piece at `t` on the free spot nearest the window (`place_spot`): within
    `GRASP_NEAR_M` of the window when `in_reach`, else on any of the piece's surfaces. Items whose kind lists the source
    surface as a receptacle come first, each group in a seeded order; at most `CANDIDATES` placeable ones are tried.
    None when none goes."""
    able = [eid for eid, c in w.carried.items()
            if c.spot.pos[2] >= FLOOR_ITEM_Z and fits(c.spec.size, src.slot) and fits(c.spec.size, dst.slot)]
    own = [eid for eid in able if src.slot.kind in w.catalog.kinds[w.carried[eid].spec.kind].receptacles]
    rest = [eid for eid in able if eid not in own]
    pool = [own[int(i)] for i in rng.permutation(len(own))] + [rest[int(i)] for i in rng.permutation(len(rest))]
    slots = [src.slot] if in_reach else [s for s in w.slots if s.furniture_entity == src.slot.furniture_entity]
    for eid in [e for e in pool if placeable(w, e, dst, others)][:CANDIDATES]:
        c = w.carried[eid]
        for slot in slots:
            spot = place_spot(w, eid, src, outside=False, slot=slot)
            if spot is None or (in_reach and rect_distance(item_rect(c.spec, *spot.pos[:2], spot.yaw), src.rect)
                                > GRASP_NEAR_M):
                continue
            w.ledger.place(eid, t, spot.pos, spot.yaw, slot.furniture_entity)
            c.spot = spot
            return eid
    return None


def _box(w: World, eid: str) -> Rect:
    c = w.carried[eid]
    return occupied_entry(c.spec, c.spot.pos, c.spot.yaw)[0]


def _other_pieces(m: MapSpec, w: World, free: FreeSpace, mask: np.ndarray, plan: TaskPlan, dst: Window,
                  target: Pose) -> list[Other]:
    """The target room's other counted pieces, in id order, that the helper can put a thing down on from the target
    pose: each with its window and the pose it stands at (a reach pose, and a way to it from the target pose)."""
    room, robot, pieces = str(plan.params["regions"][1]), ROBOTS[ROBOT_ID], m.furniture_by_id()
    out: list[Other] = []
    for pid in (p for p in qualified(m, park_pose(m)[0])[room] if p != dst.piece.id):
        pose = reach_pose(m, free, mask, pieces[pid], robot)
        if pose is None or free.path(target[0], pose[0], strict=True) is None:
            continue
        try:
            out.append((window_at(m, w, pose), pose))
        except ValueError:
            continue
    return out


def commit(w: World, carry: Carry, steps: list[Frame]) -> None:
    """The thing's placements: its source placement ends at the last `grasping` frame, and it stands at its spot from
    the last `placing` frame on (nothing in between: the arm carries it)."""
    t_grasp = max(f.t for f in steps if f.phase == "grasping")
    t_place = max(f.t for f in steps if f.phase == "placing")
    w.ledger.remove(carry.eid, t_grasp)
    w.ledger.place(carry.eid, t_place, carry.spot.pos, carry.spot.yaw, carry.spot.slot.furniture_entity)
    w.carried[carry.eid].spot = carry.spot


def targets(plan: TaskPlan, carry: Carry) -> tuple[str, ...]:
    """The task's private targets: (source room, target room, the thing, the source piece, the target piece)."""
    src, dst = plan.params["regions"]
    return (str(src), str(dst), carry.eid, *carry.pieces)
