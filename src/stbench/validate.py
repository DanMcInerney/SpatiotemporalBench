"""Timeline validation (V0 §3). `validate` returns problems; an empty list means the timeline is consistent.

World codes (this module): duplicate_entity, unknown_entity, bad_mobility, bad_rotation, bad_interval,
placement_overlap, state_overlap, on_target_absent, off_surface, carried_collision, light_collision, mover_collision,
outside_map, task_placement. The ego's codes (`validate_ego.py`, run by `validate`): keyframe_index, keyframe_order,
task_layout, keyframe_outside_task, frame_spacing, job_robot, task_kind, decision_order, decision_kind, unknown_phase,
bad_status, start_status, camera_mismatch, pitch_limit, unknown_robot, body_clearance, unknown_mover_kind,
mover_camera_clearance, outside_map and bad_rotation again, and whatever the script's job reports through
`JobSpec.checks`.

The mobility table (`timeline.MOBILITY`) decides here which pieces collide with which (`collides_as`), which lie inside
the map by their centre, and which fixed pieces block nothing (`obstructs` "if_blocking": doors, floor coverings). The
body rule (`validate_ego`) reads the fixed, light, carried and mover classes (clutter collides as carried; a mover, such
as a resident, by its footprint at its yaw, M2), so the robot keeps its clearance from a person at every keyframe;
`mover_camera_clearance` keeps a mover off the camera besides. The `structure` row is not applied to a robot's body.

A mover (a thing that rolls by itself, or a person) has a footprint that may touch no wall, no fixed or light piece
that stands above its feet (not floor coverings), no door leaf in the state it has then, no carried item or clutter
lying on its floor (its base within `FLOOR_ITEM_Z` of the mover's own base height) and no other mover, at any time both
exist. A mover reads its own floor: a fixed piece whose top is at most the mover's base height (to `UNDER_TOL`) is under
it, never in its way, so a person standing at 0.15 m on a 0.15 m sidewalk is valid and one standing at 0 m over the same
sidewalk is not. For a mover at height 0 (the home's) that is the rule as it was. Like a carried item or a door, a mover
lies inside the map by its centre (its footprint may straddle a doorway).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from .catalog import FLOOR_ITEM_Z
from .geometry import Rect, point_in_rect, rects_overlap
from .mapspec import MapSpec, Opening
from .timeline import MOBILITIES, MOBILITY, Entity, Placement, Timeline

FLOOR_COVERING_HEIGHT = 0.05  # fixed entities lower than this (rugs, mats) never obstruct anything
SURFACE_Z_TOL = 0.01
UNDER_TOL = SURFACE_Z_TOL + 1e-6  # a surface this close under a mover's feet is its floor (a float's error to spare)
SURFACE_EDGE_TOL = 0.01
QUAT_TOL = 1e-4


class Problem(NamedTuple):
    code: str
    message: str


class TimelineError(ValueError):
    def __init__(self, problems: list[Problem]):
        self.problems = problems
        shown = "; ".join(f"{p.code}: {p.message}" for p in problems[:5])
        super().__init__(shown + (" ..." if len(problems) > 5 else ""))


def check(tl: Timeline, m: MapSpec) -> None:
    """Raise TimelineError if validate finds anything."""
    problems = validate(tl, m)
    if problems:
        raise TimelineError(problems)


def validate(tl: Timeline, m: MapSpec) -> list[Problem]:
    from .validate_ego import ego_problems  # the ego rules read this module's helpers: imported here, so no cycle

    problems: list[Problem] = []
    problems += _entities(tl)
    problems += _intervals(tl)
    problems += _supports(tl)
    problems += _collisions(tl)
    problems += _movers(tl, m)
    problems += _inside_map(tl, m)
    problems += _task_placements(tl)
    problems += ego_problems(tl, m)
    return problems


def collision_class(e: Entity) -> str | None:
    """The class the mobility table collides the entity as ("fixed", "light", "carried", "mover"), None for none."""
    rule = MOBILITY.get(e.mobility)
    return rule.collides_as if rule else None


def non_blocking(e: Entity) -> bool:
    """A piece that collides as fixed but blocks nothing: a door (its leaf is geometry as a leaf, in its state) or a
    floor covering (a rug, a mat)."""
    rule = MOBILITY.get(e.mobility)
    return bool(rule and rule.obstructs == "if_blocking" and (e.kind == "door" or e.size[2] < FLOOR_COVERING_HEIGHT))


def _entities(tl: Timeline) -> list[Problem]:
    out: list[Problem] = []
    seen: set[str] = set()
    for e in tl.entities:
        if e.id in seen:
            out.append(Problem("duplicate_entity", f"entity {e.id} appears twice"))
        seen.add(e.id)
        if e.mobility not in MOBILITIES:
            out.append(Problem("bad_mobility", f"entity {e.id} has mobility {e.mobility!r}; allowed: {MOBILITIES}"))
    for p in tl.placements:
        if p.entity not in seen:
            out.append(Problem("unknown_entity", f"placement refers to unknown entity {p.entity}"))
        if abs(sum(c * c for c in p.rot) - 1.0) > QUAT_TOL:
            out.append(Problem("bad_rotation", f"placement of {p.entity} has non-unit rotation {p.rot}"))
    for s in tl.states:
        if s.entity not in seen:
            out.append(Problem("unknown_entity", f"state refers to unknown entity {s.entity}"))
    return out


def _overlaps(items: list, label: str, code: str) -> list[Problem]:
    """items: records with entity, t0, t1 (and optionally key); flag any entity (and key) whose intervals overlap."""
    out: list[Problem] = []
    by: dict[tuple, list] = {}
    for it in items:
        by.setdefault((it.entity, getattr(it, "key", None)), []).append(it)
    for (ent, key), lst in by.items():
        lst = sorted(lst, key=lambda i: i.t0)
        for a, b in zip(lst, lst[1:], strict=False):
            if a.t1 is None or b.t0 < a.t1:
                what = ent if key is None else f"{ent}.{key}"
                out.append(Problem(code, f"{label} of {what} overlap: [{a.t0}, {a.t1}) and [{b.t0}, {b.t1})"))
    return out


def _intervals(tl: Timeline) -> list[Problem]:
    out: list[Problem] = []
    for it, label in [(p, "placement") for p in tl.placements] + [(s, "state") for s in tl.states]:
        if it.t0 < 0 or (it.t1 is not None and it.t1 <= it.t0):
            out.append(Problem("bad_interval", f"{label} of {it.entity} has interval [{it.t0}, {it.t1})"))
    out += _overlaps(tl.placements, "placements", "placement_overlap")
    out += _overlaps(tl.states, "states", "state_overlap")
    return out


def _coverage(tl: Timeline, entity: str) -> list[tuple[float, float]]:
    """The entity's existence as merged [t0, t1) spans; t1 = inf for open-ended."""
    spans = sorted((p.t0, float("inf") if p.t1 is None else p.t1) for p in tl.placements if p.entity == entity)
    merged: list[tuple[float, float]] = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _supports(tl: Timeline) -> list[Problem]:
    out: list[Problem] = []
    ids = {e.id for e in tl.entities}
    cover: dict[str, list[tuple[float, float]]] = {}
    for p in tl.placements:
        if p.on is None:
            continue
        if p.on not in ids:
            out.append(Problem("on_target_absent", f"{p.entity} is on unknown entity {p.on}"))
            continue
        spans = cover.setdefault(p.on, _coverage(tl, p.on))
        hi = float("inf") if p.t1 is None else p.t1
        if not any(lo <= p.t0 and hi <= end for lo, end in spans):
            msg = f"{p.entity} is on {p.on} during [{p.t0}, {p.t1}) but {p.on} does not exist throughout"
            out.append(Problem("on_target_absent", msg))
            continue
        sup_p = tl.placement_of(p.on, p.t0)
        if sup_p is None:
            continue
        sup = tl.entity(p.on)
        outline = Rect(sup_p.pos[0], sup_p.pos[1], sup.size[0], sup.size[1], sup_p.yaw)
        if not point_in_rect(outline, p.pos[0], p.pos[1], SURFACE_EDGE_TOL):
            out.append(Problem("off_surface", f"{p.entity} at {p.pos[:2]} is outside the footprint of {p.on}"))
            continue
        top = sup_p.pos[2] + sup.size[2]
        if p.pos[2] > top + SURFACE_Z_TOL or p.pos[2] < -SURFACE_Z_TOL:
            msg = f"{p.entity} rests at z={p.pos[2]:.3f}, not on or below the top of {p.on} ({top:.3f})"
            out.append(Problem("off_surface", msg))
    return out


def _time_overlap(a: Placement, b: Placement) -> bool:
    a1 = float("inf") if a.t1 is None else a.t1
    b1 = float("inf") if b.t1 is None else b.t1
    return a.t0 < b1 and b.t0 < a1


def _rect(tl: Timeline, p: Placement) -> Rect:
    e = tl.entity(p.entity)
    return Rect(p.pos[0], p.pos[1], e.size[0], e.size[1], p.yaw)


def _collisions(tl: Timeline) -> list[Problem]:
    out: list[Problem] = []
    known = {e.id for e in tl.entities}
    by_class: dict[str, list[Placement]] = {"carried": [], "light": [], "fixed": []}
    for p in tl.placements:
        if p.entity not in known:
            continue
        e = tl.entity(p.entity)
        if non_blocking(e):
            continue
        if collision_class(e) in by_class:
            by_class[collision_class(e)].append(p)
    carried = by_class["carried"]
    for i, a in enumerate(carried):
        ea = tl.entity(a.entity)
        for b in carried[i + 1:]:
            if a.entity == b.entity or not _time_overlap(a, b):
                continue
            eb = tl.entity(b.entity)
            z_apart = a.pos[2] >= b.pos[2] + eb.size[2] or b.pos[2] >= a.pos[2] + ea.size[2]
            if not z_apart and rects_overlap(_rect(tl, a), _rect(tl, b)):
                out.append(Problem("carried_collision", f"{a.entity} and {b.entity} occupy the same spot"))
    light, fixed = by_class["light"], by_class["fixed"]
    pairs = [(a, b) for a in light for b in fixed] + [(a, b) for i, a in enumerate(light) for b in light[i + 1:]]
    for a, b in pairs:
        if a.entity != b.entity and _time_overlap(a, b) and rects_overlap(_rect(tl, a), _rect(tl, b)):
            out.append(Problem("light_collision", f"{a.entity} intersects {b.entity}"))
    return out


def _inside_map(tl: Timeline, m: MapSpec) -> list[Problem]:
    out: list[Problem] = []
    known = {e.id for e in tl.entities}
    for p in tl.placements:
        if p.entity not in known:
            continue
        e = tl.entity(p.entity)
        if e.kind == "door" or collision_class(e) in ("carried", "mover"):
            inside = m.contains((p.pos[0], p.pos[1]))
        else:
            inside = m.rect_inside(_rect(tl, p))
        if not inside or not 0 <= p.pos[2] <= m.wall_height:
            out.append(Problem("outside_map", f"{p.entity} at {p.pos} is outside the map"))
    return out


def _task_placements(tl: Timeline) -> list[Problem]:
    """task_placement: the world does not change inside a task. A placement of a non-mover entity that starts strictly
    inside a task (after its first keyframe and before its last) belongs to an entity in the task's `targets`: the thing
    a helper carries leaves and lands while it works, and nothing else is placed anew under the robot's camera. (The
    world's processes change it between tasks, `EDGE_BUFFER` from either end of the gap; a mover walks all the while.)
    The `object_where` rules read this: what stands still in a window did not move within it."""
    kfs, ids = tl.ego.keyframes, {e.id: e for e in tl.entities}
    out: list[Problem] = []
    for task in tl.ego.tasks:
        if not 0 <= task.k0 < task.k1 <= len(kfs):
            continue  # `task_layout` reports it
        t_lo, t_hi = kfs[task.k0].t, kfs[task.k1 - 1].t
        for p in tl.placements:
            e = ids.get(p.entity)
            if e is not None and e.mobility != "mover" and t_lo < p.t0 < t_hi and p.entity not in task.targets:
                out.append(Problem("task_placement", f"{p.entity} is placed anew at {p.t0}, inside task {task.index} "
                                                     f"[{t_lo}, {t_hi}], which does not name it among its targets"))
    return out


# -- doors and movers --------------------------------------------------------------------------------------------
def door_entities(tl: Timeline, m: MapSpec) -> dict[str, str]:
    """{map door id: door entity id}, matched by position (the door entity stands at its opening's centre)."""
    out: dict[str, str] = {}
    for d in m.doors():
        for e in tl.entities:
            p = tl.placement_of(e.id, 0.0) if e.kind == "door" else None
            if p is not None and math.hypot(p.pos[0] - d.center[0], p.pos[1] - d.center[1]) < 1e-3:
                out[d.id] = e.id
    return out


def door_leaf_rect(m: MapSpec, opening: Opening, is_open: bool, thickness: float) -> Rect:
    """The door leaf as a thin rectangle from its hinge to its tip, closed (across the opening) or open (square to
    the wall, on the swing side)."""
    (hx, hy), (tx, ty) = m.door_leaf(opening, is_open)
    return Rect((hx + tx) / 2, (hy + ty) / 2, math.hypot(tx - hx, ty - hy), thickness, math.atan2(ty - hy, tx - hx))


def _radius(rect: Rect) -> float:
    return math.hypot(rect.sx, rect.sy) / 2


class Obstacle(NamedTuple):
    """What a mover may not overlap: its rectangle, what it is, the span it stands over and its heights."""

    rect: Rect
    what: str
    t0: float
    t1: float
    base: float  # the height of its lowest point
    top: float  # the height of its highest point (inf for a wall)
    item: bool  # a carried item or clutter: in a mover's way only within FLOOR_ITEM_Z of the mover's base height


def _obstacles(tl: Timeline, m: MapSpec) -> list[Obstacle]:
    """What a mover may not overlap: the map's wall pieces, fixed furniture (not floor coverings or doors), every
    placement of light furniture, carried items and clutter, and each door leaf in each state it has, with the heights
    `_movers` reads against a mover's own floor."""
    inf = float("inf")
    out = [Obstacle(r, "a wall", 0.0, inf, 0.0, inf, False) for r in m.wall_pieces()]
    for e in tl.entities:
        cls = collision_class(e)
        if cls not in ("fixed", "light", "carried") or non_blocking(e):
            continue
        for p in tl.placements_of(e.id):
            out.append(Obstacle(_rect(tl, p), e.id, p.t0, inf if p.t1 is None else p.t1, p.pos[2],
                                p.pos[2] + e.size[2], cls == "carried"))
    for door_id, eid in door_entities(tl, m).items():
        opening, leaf = next(o for o in m.doors() if o.id == door_id), tl.entity(eid)
        for s in tl.states:
            if s.entity == eid and s.key == "open":
                out.append(Obstacle(door_leaf_rect(m, opening, bool(s.value), leaf.size[1]), f"the leaf of door {eid}",
                                    s.t0, inf if s.t1 is None else s.t1, 0.0, leaf.size[2], False))
    return out


def _movers(tl: Timeline, m: MapSpec) -> list[Problem]:
    """mover_collision: a mover's footprint overlaps an obstacle (`_obstacles`) that stands in its way at its base
    height, or another mover, while both exist. An obstacle is in the way unless its top is at most the mover's base
    height (`UNDER_TOL`: it is the mover's floor); a carried item or clutter is in the way only when its base is within
    `FLOOR_ITEM_Z` of the mover's. Each mover placement is read against all the obstacles at once (arrays) and the
    movers against each other by a sweep over time, so a script with many people stays cheap."""
    movers = {e.id for e in tl.entities if collision_class(e) == "mover"}
    if not movers:
        return []
    obstacles = _obstacles(tl, m)
    t0s, t1s = np.array([o.t0 for o in obstacles]), np.array([o.t1 for o in obstacles])
    cxs, cys = np.array([o.rect.cx for o in obstacles]), np.array([o.rect.cy for o in obstacles])
    radii = np.array([_radius(o.rect) for o in obstacles])
    bases, tops, items = np.array([o.base for o in obstacles]), np.array([o.top for o in obstacles]), np.array(
        [o.item for o in obstacles])
    out: list[Problem] = []
    mine = [p for p in tl.placements if p.entity in movers]
    rects = [_rect(tl, p) for p in mine]
    ends = [float("inf") if p.t1 is None else p.t1 for p in mine]
    for a, rect, a1 in zip(mine, rects, ends, strict=True):
        in_the_way = np.where(items, np.abs(bases - a.pos[2]) < FLOOR_ITEM_Z, tops > a.pos[2] + UNDER_TOL)
        close = np.hypot(cxs - rect.cx, cys - rect.cy) <= radii + _radius(rect)  # the circumscribed discs meet
        for i in np.flatnonzero(in_the_way & close & (t0s < a1) & (t1s > a.t0)):
            if rects_overlap(rect, obstacles[i].rect):
                o = obstacles[i]
                msg = f"{a.entity} at {a.pos[:2]} intersects {o.what} during [{max(a.t0, o.t0)}, {min(a1, o.t1)})"
                out.append(Problem("mover_collision", msg))
    live: list[int] = []  # the placements, by index, still standing when the one swept starts
    for j in sorted(range(len(mine)), key=lambda i: mine[i].t0):
        live = [i for i in live if ends[i] > mine[j].t0]
        for i in live:
            if mine[i].entity != mine[j].entity and rects_overlap(rects[i], rects[j]):
                a, b = (mine[i], mine[j]) if i < j else (mine[j], mine[i])
                msg = f"{a.entity} at {a.pos[:2]} intersects {b.entity} at {b.pos[:2]} at the same time"
                out.append(Problem("mover_collision", msg))
        live.append(j)
    return out
