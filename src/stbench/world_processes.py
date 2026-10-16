"""The world's own processes between a job's tasks, and the one rule of which doors a task needs open.

A job's `script` calls the callback `gaps` returns once before each task (`Between`), so the world changes what a robot
does not see: `gaps` is the engine, one module of processes a catalog (`world_processes_<catalog>`, the home's is
`world_processes_home`) and the doors' rule here. The doors that may change are decided by `doors_on_ways`: a door on a
way between the rooms a task works in stays open through the task and the planner treats every other door as shut, so
no door's state ever changes the robot's path, and a path log that leaks its detours learns no door (design §8.1). Both
lanes that need the rule, the job (A2) and the gaps (A2W), call it here. Every process re-draws a thing's state by the
law the world started with, at a half-life rate in the gap's length (`half_life_p`; `docs/world.md`).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping
from types import ModuleType

from . import SEED_KEY
from .changes import EDGE_BUFFER, World, when_in_gap
from .mapspec import MapSpec, Opening
from .seeding import make_rng

# between(lo, hi, keep_open): lo None before the first task (the script's start), else the gap (lo, hi) after the last
# keyframe of one task and before the first of the next; keep_open the region ids the next task travels between.
Between = Callable[[float | None, float, frozenset[str]], None]


def doors_on_ways(m: MapSpec, regions: Iterable[str]) -> frozenset[str]:
    """The ids of the interior doors that lie on some way between two of the given regions: a door is on a way when
    it is an edge of some simple path between two of them in the map's room graph (regions are nodes; interior
    openings, doors and arches, are edges; the entrance and the outside are left out). On apartment_a, {hall, kitchen}
    gives {D1, D2} (the kitchen is reached by D1, or by D2, the living room and the arch O1); {hall, bedroom} gives
    {D4}; {hall} gives none. A region the map does not have is a KeyError."""
    wanted = list(dict.fromkeys(regions))
    known = {r.id for r in m.regions}
    for rid in wanted:
        if rid not in known:
            raise KeyError(f"map {m.name} has no region {rid!r}")
    edges = [o for o in m.openings if o.interior and o.between[0] in known and o.between[1] in known]
    on: set[str] = set()
    for i, start in enumerate(wanted):
        for goal in wanted[i + 1:]:
            on |= _edges_on_simple_paths(edges, start, goal)
    return frozenset(o.id for o in edges if o.type == "door" and o.id in on)


def _edges_on_simple_paths(edges: list[Opening], start: str, goal: str) -> set[str]:
    """The ids of the edges (parallel ones apart) that lie on some simple path from `start` to `goal`, by walking every
    such path: the room graphs are a handful of rooms."""
    found: set[str] = set()
    path: list[str] = []

    def walk(at: str, seen: frozenset[str]) -> None:
        if at == goal:
            found.update(path)
            return
        for o in edges:
            if at in o.between:
                nxt = o.between[1] if o.between[0] == at else o.between[0]
                if nxt not in seen:
                    path.append(o.id)
                    walk(nxt, seen | {nxt})
                    path.pop()

    walk(start, frozenset({start}))
    return found


def half_life_p(gap_s: float, half_life_s: float) -> float:
    """The chance that a thing whose state is drawn again at a half-life rate has been drawn again over a gap:
    1 - 2^(-gap / half-life), so 0 for a gap of 0, 1/2 for one half-life and 3/4 for two; it grows with the gap and
    never passes 1, which a count of moves a gap (V0's Poisson draw) did not."""
    return 1.0 - 2.0 ** (-gap_s / half_life_s)


def door_states(w: World) -> dict[str, bool]:
    """Each door's state as the ledger holds it now (its open-ended `open` state): map door id -> open; a door with
    none counts as shut."""
    states = {s.entity: bool(s.value) for s in w.ledger.finish()[1] if s.key == "open" and s.t1 is None}
    return {door: states.get(eid, False) for door, eid in sorted(w.door_entity.items())}


def set_doors(w: World, t: float, wanted: Mapping[str, bool]) -> int:
    """From `t` on, each door of `wanted` (map door id -> open) is in that state; a state is written only for a door
    that is not already in it. Returns how many doors changed."""
    now = door_states(w)
    changed = [door for door in sorted(wanted) if now[door] != wanted[door]]
    for door in changed:
        w.ledger.set_state(w.door_entity[door], "open", t, wanted[door])
    return len(changed)


def _home(catalog: str) -> ModuleType | None:
    """The module of a catalog's processes, `world_processes_<catalog>`, or None when it does not exist; any other
    import error is raised."""
    name = f"{__package__}.world_processes_{catalog}"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name != name:
            raise
        return None


def gaps(w: World, m: MapSpec, seed: int) -> Between:
    """The callback a job's `script` calls once before each task, in task order, as `between(lo, hi, keep_open)`:
    - `lo` None: before the first task, at the script's start (t = 0): what the world's processes set at the start
      (states written at t = 0, which the ledger rule lets replace `build_world`'s);
    - else the unobserved gap (lo, hi) between the last keyframe of one task and the first of the next: every change
      at a time inside it, at least `changes.EDGE_BUFFER` (30 s) from either end, so no change is ever seen happening;
    - `keep_open`: the region ids the next task travels between (its robot's start room and the rooms it works in);
      every door of `doors_on_ways(m, keep_open)` is open at `hi`, and so through the task.
    A job plans the next task on `w` after the call, treating every interior door outside `doors_on_ways(m, keep_open)`
    as shut whatever its state, so no door's state ever changes the robot's path. Draws come from
    `make_rng(m.name, seed, SEED_KEY, "world", i)` for the i-th call. A call needs a gap of at least twice
    `changes.EDGE_BUFFER`: its changes all happen at one instant inside it.
    The catalog's processes are the functions of the module `world_processes_<m.catalog>`, loaded on first use:
    `start(w, m, rng, keep_open)` for `lo` None and `gap(w, m, rng, lo, hi, keep_open, forced)` for a gap, with `forced`
    the doors forced open for the previous task (`doors_on_ways` of the previous call's `keep_open`, which this callback
    keeps; empty on the first gap). A catalog with no module only opens every door of `doors_on_ways(m, keep_open)`
    that is shut, at 0 or at an instant of the gap."""
    home = _home(m.catalog)
    calls = 0
    forced: frozenset[str] = frozenset()

    def between(lo: float | None, hi: float, keep_open: frozenset[str]) -> None:
        nonlocal calls, forced
        if lo is not None and hi - lo < 2 * EDGE_BUFFER:
            raise ValueError(f"the gap ({lo}, {hi}) is shorter than {2 * EDGE_BUFFER} s: no instant is {EDGE_BUFFER} s "
                             "from both ends")
        rng = make_rng(m.name, seed, SEED_KEY, "world", calls)
        calls += 1
        on_ways = doors_on_ways(m, sorted(keep_open))
        if home is None:
            set_doors(w, 0.0 if lo is None else when_in_gap(rng, lo, hi), dict.fromkeys(sorted(on_ways), True))
        elif lo is None:
            home.start(w, m, rng, keep_open)
        else:
            home.gap(w, m, rng, lo, hi, keep_open, forced)
        forced = on_ways

    return between
