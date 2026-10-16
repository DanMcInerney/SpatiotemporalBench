"""The world timeline (`private/<id>/timeline.json`): the single source of truth.

Conventions:
- Map frame: metres, x east, y north, z up. Times are seconds since the script's `episode.start_clock`.
- An entity's `pos` is the BASE CENTRE: the centre of the bottom face of its bounding box. `size` is
  (extent along the local x, extent along the local y, height), so the footprint is
  `Rect(pos.x, pos.y, size[0], size[1], yaw)`.
- Rotations are unit quaternions `rot = [w, x, y, z]` mapping the body frame (x forward, y left, z up) to the map
  frame. Placements only turn about z. An ego keyframe's `pos` is the CAMERA centre and its `rot` is
  qz(yaw) * qy(-pitch), so pitch is positive looking up; its `base`, `yaw` and `cam_pitch` are the robot's
  (`timeline_ego`).
- All intervals are half-open: placements and states [t0, t1) with `t1 = None` meaning until the end; tasks [k0, k1)
  in keyframe indices. An entity exists only while a placement covers t.
- States are one record per key: {entity, key, t0, t1, value}. Doors are `fixed` entities with an `open` state and the
  `hinge` and `swing` of their map opening in their attributes.
- The file has two sections, `world` (regions, entities, placements, states) and `ego` (keyframes, tasks, decisions).
  `world_hash` and `ego_hash` are in-run content hashes of the two, for comparing timelines in one process; the world's
  does not depend on the robot, and `episode` is outside both. The file records neither.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path

from .geometry import Point, Quat, Rect, point_in_polygon, q, yaw_of
from .timeline_ego import DecisionPoint, Ego, Keyframe, Task, TaskPlan, ego_dict, ego_from_dict  # noqa: F401


@dataclass(frozen=True)
class MobilityRule:
    """What `validate`, the renderer and the compiler read of a mobility. The body rule reads the fixed, light and
    carried classes (clutter collides as carried); movers' and structure's "always" applies when movers return (M2)."""

    collides_as: str | None  # validate's collision class: "fixed", "light", "carried", "mover"; None: never (structure)
    rests_on: str | None  # "floor" or "surface_or_floor"; None for structure
    obstructs: str  # to a ground robot: "always", "if_blocking" (fixed: not rugs or doors), "on_floor_above_climb"
    describable: bool  # a question may name it


MOBILITY: dict[str, MobilityRule] = {
    "structure": MobilityRule(None, None, "always", False),
    "fixed": MobilityRule("fixed", "floor", "if_blocking", False),
    "light": MobilityRule("light", "floor", "always", True),
    "carried": MobilityRule("carried", "surface_or_floor", "on_floor_above_climb", True),
    "mover": MobilityRule("mover", "floor", "always", True),
    "clutter": MobilityRule("carried", "surface_or_floor", "on_floor_above_climb", False),
}
MOBILITIES = tuple(MOBILITY)
DESCRIBABLE = tuple(k for k, r in MOBILITY.items() if r.describable)


# -- records -----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EpisodeInfo:
    map: str
    seed: int
    robot: str  # a `robots.ROBOTS` key
    job: str  # the job module's id: "vacuum/2"
    start_clock: str  # the script's own clock; a case re-bases it (`case.case_clock`)


@dataclass(frozen=True)
class Region:
    id: str
    name: str
    kind: str  # the room's kind word: the map's optional `regions[i].kind`, else the name
    polygon: tuple[Point, ...]
    level: int
    z0: float
    z1: float


@dataclass(frozen=True)
class Entity:
    id: str
    kind: str
    attributes: dict[str, object]
    mobility: str
    size: tuple[float, float, float]
    asset: str | None = None  # the manifest model of a carried or light item; else null
    under_m: float = 0.0  # the height of an open underside; 0 is solid to the floor (`geometry.under_obstacles`)


@dataclass(frozen=True)
class Placement:
    entity: str
    t0: float
    t1: float | None
    pos: tuple[float, float, float]
    rot: Quat
    on: str | None

    @property
    def yaw(self) -> float:
        return yaw_of(self.rot)

    def covers(self, t: float) -> bool:
        return self.t0 <= t and (self.t1 is None or t < self.t1)


@dataclass(frozen=True)
class State:
    entity: str
    key: str
    t0: float
    t1: float | None
    value: object

    def covers(self, t: float) -> bool:
        return self.t0 <= t and (self.t1 is None or t < self.t1)


@dataclass
class Timeline:
    episode: EpisodeInfo
    regions: list[Region]
    entities: list[Entity]
    placements: list[Placement]
    states: list[State]
    ego: Ego

    # -- indexes (the timeline is not edited after construction) ---------------------------------------------
    @cached_property
    def _entity_index(self) -> dict[str, Entity]:
        return {e.id: e for e in self.entities}

    @cached_property
    def _placement_index(self) -> dict[str, tuple[list[float], list[Placement]]]:
        by: dict[str, list[Placement]] = {}
        for p in sorted(self.placements, key=lambda p: p.t0):
            by.setdefault(p.entity, []).append(p)
        return {k: ([p.t0 for p in v], v) for k, v in by.items()}

    @cached_property
    def _state_index(self) -> dict[str, list[State]]:
        by: dict[str, list[State]] = {}
        for s in sorted(self.states, key=lambda s: s.t0):
            by.setdefault(s.entity, []).append(s)
        return by

    # -- queries ---------------------------------------------------------------------------------------------
    def entity(self, entity_id: str) -> Entity:
        return self._entity_index[entity_id]

    def placement_of(self, entity: str, t: float) -> Placement | None:
        """The placement covering t, or None if the entity does not exist then."""
        starts, items = self._placement_index.get(entity, ([], []))
        i = bisect_right(starts, t) - 1
        if i >= 0 and items[i].covers(t):
            return items[i]
        return None

    def placements_of(self, entity: str) -> list[Placement]:
        """Every placement interval of the entity, by start time (empty if it never exists)."""
        return list(self._placement_index.get(entity, ([], []))[1])

    def with_placements(self, replaced: dict[str, list[Placement]]) -> Timeline:
        """A copy in which each listed entity's placements are exactly the given ones (the others are untouched)."""
        kept = [p for p in self.placements if p.entity not in replaced]
        return replace(self, placements=kept + [p for ps in replaced.values() for p in ps])

    def with_states(self, replaced: dict[str, list[State]]) -> Timeline:
        """A copy in which each listed entity's states are exactly the given ones (the others are untouched)."""
        kept = [s for s in self.states if s.entity not in replaced]
        return replace(self, states=kept + [s for ss in replaced.values() for s in ss])

    def state_of(self, entity: str, t: float, key: str | None = None):
        """The states in force at t as {key: value} (empty if none); with `key`, that one value or None."""
        now = {s.key: s.value for s in self._state_index.get(entity, []) if s.covers(t)}
        return now if key is None else now.get(key)

    def values_between(self, entity: str, key: str, t0: float, t1: float) -> list:
        """Every value the state `key` of the entity has at some time in [t0, t1], in start order, without repeats
        (empty if it never has one there)."""
        out: list = []
        for s in self._state_index.get(entity, []):
            if s.key == key and s.t0 <= t1 and (s.t1 is None or s.t1 > t0) and s.value not in out:
                out.append(s.value)
        return out

    def entities_at(self, t: float) -> list[Entity]:
        """Entities that exist at time t, in timeline order."""
        return [e for e in self.entities if self.placement_of(e.id, t) is not None]

    def keyframe(self, k: int) -> Keyframe:
        return self.ego.keyframes[k]

    @property
    def n_keyframes(self) -> int:
        return len(self.ego.keyframes)

    def first_keyframe_at(self, t: float) -> int:
        """The first keyframe at or after time t (n_keyframes if there is none)."""
        return next((kf.k for kf in self.ego.keyframes if kf.t >= t), len(self.ego.keyframes))

    def task_of(self, k: int) -> Task:
        """The task whose keyframes [k0, k1) hold k."""
        for t in self.ego.tasks:
            if t.k0 <= k < t.k1:
                return t
        raise IndexError(f"keyframe {k} lies in no task")

    def decisions_in(self, k0: int, k1: int) -> list[DecisionPoint]:
        """The decision points at keyframes k0 <= k < k1, in order."""
        return [d for d in self.ego.decisions if k0 <= d.k < k1]

    def region_of(self, xy: Point) -> str | None:
        """The region containing the point, or None in a doorway or outside the rooms."""
        for r in self.regions:
            if point_in_polygon(xy[0], xy[1], list(r.polygon)):
                return r.id
        return None

    def clock(self, t: float) -> str:
        return clock_at(self.episode.start_clock, t)

    def footprint(self, entity: str, t: float) -> Rect | None:
        p = self.placement_of(entity, t)
        if p is None:
            return None
        e = self.entity(entity)
        return Rect(p.pos[0], p.pos[1], e.size[0], e.size[1], p.yaw)

    @property
    def world_hash(self) -> str:
        return _hash(_world_dict(self))

    @property
    def ego_hash(self) -> str:
        return _hash(ego_dict(self.ego))


def clock_at(start_clock: str, t: float) -> str:
    """Local ISO time at t seconds after start_clock, rounded half up to the second."""
    return (datetime.fromisoformat(start_clock) + timedelta(seconds=int(t + 0.5))).isoformat(timespec="seconds")


# -- JSON ------------------------------------------------------------------------------------------------------
def _world_dict(tl: Timeline) -> dict:
    return {
        "regions": [{"id": r.id, "name": r.name, "kind": r.kind, "polygon": [[q(x), q(y)] for x, y in r.polygon],
                     "level": r.level, "z0": q(r.z0), "z1": q(r.z1)} for r in tl.regions],
        "entities": [{"id": e.id, "kind": e.kind, "attributes": dict(sorted(e.attributes.items())),
                      "mobility": e.mobility, "size": [q(v) for v in e.size], "asset": e.asset,
                      "under_m": q(e.under_m)} for e in tl.entities],
        "placements": [{"entity": p.entity, "t0": q(p.t0), "t1": None if p.t1 is None else q(p.t1),
                        "pos": [q(v) for v in p.pos], "rot": [q(v) for v in p.rot], "on": p.on}
                       for p in tl.placements],
        "states": [{"entity": s.entity, "key": s.key, "t0": q(s.t0), "t1": None if s.t1 is None else q(s.t1),
                    "value": s.value} for s in tl.states],
    }


def _hash(section: dict) -> str:
    text = json.dumps(section, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def to_dict(tl: Timeline) -> dict:
    ep = tl.episode
    return {
        "episode": {"map": ep.map, "seed": ep.seed, "robot": ep.robot, "job": ep.job, "start_clock": ep.start_clock},
        "world": _world_dict(tl),
        "ego": ego_dict(tl.ego),
    }


def _compact(v: object) -> str:
    return json.dumps(v, separators=(", ", ": "), ensure_ascii=False)


def _emit(v: object, indent: int) -> str:
    """Records of a list go one per line; small values stay on one line. The layout is fixed, so bytes are stable."""
    pad = "  " * indent
    if isinstance(v, dict) and any(isinstance(x, list) and x and isinstance(x[0], dict) for x in v.values()):
        body = ",\n".join(f"{pad}  {json.dumps(k)}: {_emit(x, indent + 1)}" for k, x in v.items())
        return "{\n" + body + "\n" + pad + "}"
    if isinstance(v, list) and v and isinstance(v[0], dict):
        body = ",\n".join(f"{pad}  {_compact(x)}" for x in v)
        return "[\n" + body + "\n" + pad + "]"
    return _compact(v)


def dumps(tl: Timeline) -> str:
    """Canonical JSON text: fixed key order, 6-decimal floats, one record per line, trailing newline."""
    body = ",\n".join(f"  {json.dumps(k)}: {_emit(v, 1)}" for k, v in to_dict(tl).items())
    return "{\n" + body + "\n}\n"


def digest(tl: Timeline) -> str:
    """SHA-256 of the canonical JSON bytes: equal timelines have equal digests."""
    return hashlib.sha256(dumps(tl).encode("utf-8")).hexdigest()


def save(tl: Timeline, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(dumps(tl).encode("utf-8"))


def from_dict(d: dict) -> Timeline:
    """The timeline a dict holds. It reads only the keys it names: a file of an earlier build may carry others (a schema
    string, the episode's digests and generator, a section hash), which nothing reads."""

    def end(x: float | None) -> float | None:
        return None if x is None else float(x)

    ep, world, ego = d["episode"], d["world"], d["ego"]
    return Timeline(
        episode=EpisodeInfo(ep["map"], int(ep["seed"]), ep["robot"], ep["job"], ep["start_clock"]),
        regions=[Region(r["id"], r["name"], r["kind"], tuple((float(x), float(y)) for x, y in r["polygon"]),
                        int(r["level"]), float(r["z0"]), float(r["z1"])) for r in world["regions"]],
        entities=[Entity(e["id"], e["kind"], dict(e["attributes"]), e["mobility"], tuple(float(v) for v in e["size"]),
                         e["asset"], float(e["under_m"])) for e in world["entities"]],
        placements=[Placement(p["entity"], float(p["t0"]), end(p["t1"]), tuple(float(v) for v in p["pos"]),
                              tuple(float(v) for v in p["rot"]), p["on"]) for p in world["placements"]],
        states=[State(s["entity"], s["key"], float(s["t0"]), end(s["t1"]), s["value"]) for s in world["states"]],
        ego=ego_from_dict(ego),
    )


def loads(text: str) -> Timeline:
    return from_dict(json.loads(text))


def load(path: str | Path) -> Timeline:
    return loads(Path(path).read_text(encoding="utf-8"))


def validate(tl: Timeline, map_spec):
    """Check the timeline against the map; returns a list of Problem(code, message), empty when consistent."""
    from .validate import validate as _validate

    return _validate(tl, map_spec)
