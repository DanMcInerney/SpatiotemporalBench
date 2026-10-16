"""`helper/1`'s place spot: the window in front of the helper where it will put a thing down, the coin that sets it
occupied or free, and where the thing then goes (docs/job-helper.md).

The window is `zone_shapes.surface_window` of the base's pose: 0.4 m of the surface the robot faces, centred on its
heading. Before each task a coin that reads no world (`job_helper_plan.spot_blocked`) sets the window, on the world
after the task is planned and `EDGE_BUFFER` before it starts, so no change is ever seen happening, and on surfaces only
(the floor and every planned path stay as they are):
- **clear**: everything that stands in the window grown by `DEEP_M` moves off it, to a place that is not in it (items by
  the world's own law, `clutter_surface.move_to_surface`; clutter by its kind's places, `move_surface`, else slid along
  the surface, `slide_surface`);
- **blocked**: when nothing stands in the window shrunk by `DEEP_M`, one carried item that lies on a raised surface
  moves onto it (`placement.sample_box`).
A failed enforcement is not an error: the response follows the world as it then is. What stands in the window by more
than `SLIVER_M` (the grid of the zone families sees a sub-cell 0.02 m across, centred inside the window) blocks it, and
the response is `ELSEWHERE`; else `GO`. The thing goes to the window's centre on `GO` (the free spot nearest it when it
is wider than the window and meets a neighbour), else to the free spot nearest the window's centre that is outside the
window (`place_spot`, a scan, no draw).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .catalog import FLOOR_ITEM_Z
from .changes import World
from .clutter_surface import Band, free_spot, move_surface, move_to_surface, slide_surface, standing
from .geometry import Rect, rects_overlap
from .job_helper_reach import Pose
from .mapspec import Furniture, MapSpec, Surface
from .placement import EDGE_MARGIN, Slot, Spot, occupied_entry, sample_box
from .zone_shapes import surface_window
from .zones import SPOT, ZoneRule

GO, ELSEWHERE = "go", "elsewhere"  # the responses (`DecisionPoint.response`)
ZONE = ZoneRule("the spot in front of you where you are putting it down", SPOT, height_m=0.01, top_m=0.5)
BAND: Band = (ZONE.height_m, ZONE.top_m)
DEEP_M = 0.05  # a clear window has nothing within this of it; a thing put in a blocked one reaches this far into it
SLIVER_M = 0.01  # a thing blocks the window when it reaches more than this far into it
OUTSIDE_M = 0.005  # a thing put beside the occupied spot is at least this clear of the window (not a rounding off it)
SPOT_TRIES = 40  # spots a thing put into the window is drawn


@dataclass(frozen=True)
class Window:
    """The window and the surface it lies on: the piece and its surface on the map, the window, the world's slot."""

    piece: Furniture
    surface: Surface
    rect: Rect
    slot: Slot

    @property
    def ground(self) -> float:
        return self.surface.z


def grow(rect: Rect, by: float) -> Rect:
    """The rectangle `by` larger on every side (smaller for a negative `by`, never under 2 mm)."""
    return Rect(rect.cx, rect.cy, max(rect.sx + 2 * by, 0.002), max(rect.sy + 2 * by, 0.002), rect.yaw)


def window_at(m: MapSpec, w: World, pose: Pose) -> Window:
    """The window of a base at `pose`; ValueError when it faces no surface (the helper's reach poses face one)."""
    found = surface_window(m, pose[0], pose[1])
    if found is None:
        raise ValueError(f"the base at {pose} faces no surface")
    piece, surface, rect = found
    slot = next((s for s in w.slots if s.surface_id == surface.id), None)
    if slot is None:
        raise ValueError(f"the surface {surface.id} of {piece.id} is no slot of the world")
    return Window(piece, surface, rect, slot)


def blockers(w: World, win: Window) -> list[str]:
    """What blocks the window now: the things standing in it by more than `SLIVER_M`."""
    return standing(w, grow(win.rect, -SLIVER_M), win.ground, BAND)


def response(w: World, win: Window) -> str:
    """What the helper does at the window as the world is: `ELSEWHERE` when it is blocked, else `GO`."""
    return ELSEWHERE if blockers(w, win) else GO


def _in_column(box: Rect, z0: float, z1: float, win: Window, by: float) -> bool:
    """Whether a thing reaches into the window grown by `by`: the band and the footprint."""
    return z1 > win.ground + BAND[0] and z0 < win.ground + BAND[1] and rects_overlap(box, win.rect, by)


def enforce(w: World, m: MapSpec, rng: np.random.Generator, win: Window, blocked: bool, t: float, keep: str) -> None:
    """Make the window blocked or clear from `t` on, as the module docstring says; `keep` (the thing the helper will
    carry) is never moved. Whether it worked is `response`'s to say, from the world as it stands."""
    if not blocked:
        for eid in standing(w, win.rect, win.ground, BAND, DEEP_M):
            if eid != keep:
                _move_out(w, m, rng, win, eid, t)
    elif not standing(w, grow(win.rect, -DEEP_M), win.ground, BAND):
        _put_in(w, rng, win, t, keep)


def _move_out(w: World, m: MapSpec, rng: np.random.Generator, win: Window, eid: str, t: float) -> bool:
    if eid in w.carried:
        spec = w.carried[eid].spec

        def refuse(spot: Spot) -> bool:
            return _in_column(*occupied_entry(spec, spot.pos, spot.yaw), win, DEEP_M)

        return move_to_surface(w, m, rng, eid, t, refuse)
    return (move_surface(w, m, rng, eid, t, lambda box, z0, z1: _in_column(box, z0, z1, win, DEEP_M))
            or slide_surface(w, eid, t, win.slot, (win.rect.cx, win.rect.cy), win.rect, DEEP_M))


def _put_in(w: World, rng: np.random.Generator, win: Window, t: float, keep: str) -> bool:
    """One carried item that lies on a raised surface onto the window, reaching `DEEP_M` into it: of those whose kind
    lists the window's surface as a receptacle, else of any."""
    raised = [eid for eid, c in w.carried.items() if eid != keep and c.spot.pos[2] >= FLOOR_ITEM_Z]
    own = [eid for eid in raised if win.slot.kind in w.catalog.kinds[w.carried[eid].spec.kind].receptacles]
    pool, inner = own or raised, grow(win.rect, -DEEP_M)
    for _ in range(SPOT_TRIES if pool else 0):
        eid = pool[int(rng.integers(len(pool)))]
        c = w.carried[eid]
        found = sample_box(rng, c.spec.size, win.slot.rect, win.slot.z, w.without(eid).occupied(),
                           lambda: float(rng.uniform(-math.pi, math.pi)), SPOT_TRIES,
                           ok=lambda box: rects_overlap(box, inner))
        if found is not None:
            spot = Spot(win.slot, (found[0], found[1], win.slot.z), found[2])
            w.ledger.place(eid, t, spot.pos, spot.yaw, win.slot.furniture_entity)
            c.spot = spot
            return True
    return False


def fits(size: tuple[float, float, float], slot: Slot) -> bool:
    """Whether a box of the size lies on the slot, a quarter turn to it either way, `EDGE_MARGIN` from its edges."""
    return any(ex + 2 * EDGE_MARGIN <= slot.rect.sx and ey + 2 * EDGE_MARGIN <= slot.rect.sy
               for ex, ey in ((size[0], size[1]), (size[1], size[0])))


def place_spot(w: World, eid: str, win: Window, outside: bool, slot: Slot | None = None) -> Spot | None:
    """Where the thing `eid` is put down (it stands in the world now; it is taken out for the look): the free spot on
    `slot` (the window's surface unless given) nearest the window's centre (`clutter_surface.free_spot`), with `outside`
    at least `OUTSIDE_M` clear of the window. None when no spot holds it."""
    slot = slot or win.slot
    found = free_spot(w.without(eid).occupied(), w.carried[eid].spec.size, slot, (win.rect.cx, win.rect.cy),
                      win.rect if outside else None, OUTSIDE_M)
    return None if found is None else Spot(slot, (found[0], found[1], slot.z), found[2])
