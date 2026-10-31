"""The ground corridor's screens (0.8.0, docs/families/clear_path.md): which `clear_path` golds a camera-only reader
could reach, so a question is asked only where a pixel-perfect reading earns the full rung. Each screen answers whether
the question is asked; each reads the timeline and the depth of the frames the gold rests on, never a gold of another
family, and removes questions only (a screen is not a draw: the asking tells that the stopper is tied, visible and clear
of the body's side and the band's edges, which a reader of the frame has without it).

Since 0.9.0 the screens take the free-space cells the gold read (`World.latest_cells`), not the look window's
keyframes, so a gold over a window can call them.

- **The stopper** (`stopper`, its `Stopper` record; `stopper_tied` is its two verdicts together). In the keyframe that
  last observed the stopping fine cell (the frame the gold rests on), the obstacle points the body touches
  (`grid.reach`: inside the corridor and the band), in that cell: the one the body touches first is the stop and its
  instance the stopper. The stopper is *tied* when a point of that instance, within `max(CONTACT_M, slant / fy)` of the
  floor, is touched within C/2 after the stop (inside the corridor widened by C/2), and its one-pixel error `(d^2 + H^2)
  / (H fy)` is at most C/2 (`error_tied`, which a flier's wall passes too: `clear_fly`; the ego families' contact tie:
  a table top whose legs stand 0.4 m behind its edge is not tied, a box on the floor is). It must be *visible*:
  structure (no entity), a `fixed` or `structure` entity, or a thing recognisable in that frame. A gold at the cap
  needs neither. The record also carries the stopper's contact points' least one-pixel error and how far beyond the
  stop that contact lies: the resolution of the gold (`credit_kernel`).
- **The side** (`side_stable`, `CLEAR_SIDE_M`). Not asked when the body 0.01 m narrower or wider on each side (a disc's
  radius 0.01 m smaller or larger) gives a gold more than C/2 from it (an `unknown` there counts as moved): an obstacle
  within 1 cm of the body's side falls either way (`grid.corridor_clear`). For a disc the screen is the more needed
  there: an obstacle tangent to the disc is touched only after the disc has moved its whole radius, one just outside it
  never.
- **The height** (`height_clear`, `CLEAR_HEIGHT_M`). Not asked when an entity the body sweeps before the stop (its
  rectangle, or the disc's path) has a horizontal edge within 0.005 m of an edge of the band `grid.band` reads obstacles
  in: its top and the band's floor (the climb plus `grid.DEPTH_QUANTUM_M`), its underside (`Entity.under_m`) or, when it
  hangs, its bottom edge and the band's top. A camera cannot tell 1.5 cm from 1.3 cm, nor a 10.0 cm underside from the
  vacuum's 9.5 cm. The map's fixed floor coverings (a rug whose top is under the band's floor) are exempt: floor to the
  robot by construction.
"""

from __future__ import annotations

import dataclasses
import math
from typing import NamedTuple

import numpy as np

from . import grid
from .geometry import Rect, rect_distance, rects_overlap
from .nearest_part import CONTACT_M
from .questions_words import cap_of
from .questions_world import World
from .robots import blind_m

CLEAR_SIDE_M = 0.01     # the body's width judged this much narrower and wider on each side
CLEAR_HEIGHT_M = 0.005  # a top, an underside or a bottom edge this near the band's edge is a height no frame shows
EPS = grid.EPS


def _widened(w: World, extra: float):
    """The robot with the body `extra` metres wider (narrower for a negative `extra`)."""
    body = dataclasses.replace(w.robot.body, width_m=w.robot.body.width_m + extra)
    return dataclasses.replace(w.robot, body=body)


def side_stable(w: World, start: tuple[float, float], heading: float, gold: dict, states: dict, blind: float) -> bool:
    """Whether the gold survives the body's side moving by `CLEAR_SIDE_M` either way: the body `2 * CLEAR_SIDE_M`
    narrower and wider (each side moved by `CLEAR_SIDE_M`; a disc's radius by as much) gives a number within C/2 of
    `gold`'s (`unknown` there is a move). `states` are the free-space cells the gold read (`World.latest_cells`)."""
    cap = cap_of(w.robot)
    for extra in (-2 * CLEAR_SIDE_M, 2 * CLEAR_SIDE_M):
        other = grid.corridor_clear(states, _widened(w, extra), start, heading, cap, blind)[0]
        if other["type"] != "number" or abs(other["value"] - gold["value"]) > w.job.clearance_m / 2 + EPS:
            return False
    return True


def _swept(w: World, start: tuple[float, float], heading: float, length: float):
    """Whether a footprint (a `Rect`) meets what the body sweeps moving `length` from `start`, its front edge: a box
    sweeps the rectangle `length` long and its width across; a disc the path of its centre (`length` long, starting a
    radius behind `start`) widened by its radius."""
    body, axis = w.robot.body, (math.cos(heading), math.sin(heading))
    if body.shape == "disc":
        r = body.width_m / 2
        path = Rect(start[0] + axis[0] * (length / 2 - r), start[1] + axis[1] * (length / 2 - r), length, 0.0, heading)
        return lambda rect: rect_distance(rect, path) <= r + EPS
    strip = Rect(start[0] + axis[0] * length / 2, start[1] + axis[1] * length / 2, length, body.width_m, heading)
    return lambda rect: rects_overlap(rect, strip)


def height_clear(w: World, k: int, start: tuple[float, float], heading: float, gold: dict) -> bool:
    """Whether no entity the body sweeps before the stop has a top within `CLEAR_HEIGHT_M` of the band's floor, an
    underside (`Entity.under_m`) or, when it hangs, a bottom edge within `CLEAR_HEIGHT_M` of the band's top (the module
    docstring), at the time of keyframe `k`, heights measured above the base's floor. A fixed piece whose top is under
    the band's floor is exempt."""
    t, floor_z = w.tl.keyframe(k).t, w.rows[k]["pose"]["pos"][2]
    floor, top = grid.band(w.robot)
    meets = _swept(w, start, heading, max(gold["value"], grid.SUB_M))
    for e in w.tl.entities:
        if (rect := w.tl.footprint(e.id, t)) is None or not meets(rect):
            continue
        base = w.tl.placement_of(e.id, t).pos[2] - floor_z
        covering = e.mobility == "fixed" and base + e.size[2] <= floor + EPS
        if not covering and abs(base + e.size[2] - floor) <= CLEAR_HEIGHT_M + EPS:
            return False
        if e.under_m > 0 and abs(base + e.under_m - top) <= CLEAR_HEIGHT_M + EPS:
            return False
        if base > EPS and abs(base - top) <= CLEAR_HEIGHT_M + EPS:
            return False
    return True


def _touched(w: World, along: np.ndarray, across: np.ndarray, widen: float = 0.0) -> np.ndarray:
    """`grid.reach` of the robot's body for each point (along, across), inf where the body never touches it."""
    body = w.robot.body
    out = [grid.reach(body.shape, body.width_m / 2, a, c, widen)
           for a, c in zip(along.tolist(), across.tolist(), strict=True)]
    return np.array([math.inf if t is None else t for t in out])


def error_tied(error_m: float, c_m: float) -> bool:
    """Whether a floor contact read with a one-pixel error of `error_m` is tied: within C/2 of where it is, `c_m` the
    job's clearance. The one tie a corridor's stopper (here) and a flier's wall (`clear_fly`) must pass."""
    return error_m <= c_m / 2 + EPS


class Stopper(NamedTuple):
    """What `stopper` reads of the thing a corridor stops at (the module docstring). `tied` and `visible` are the two
    verdicts of the stopper screen (`stopper_tied` is both); `error_m` is the stopper's contact points' least one-pixel
    error in the frame the stop rests on and `gap_m` how far beyond the stop that contact lies, both metres: the
    resolution a camera-only reader reaches the stop with (`credit_kernel`'s `rho`). A corridor with no stop (at the
    cap, or every cell it crosses seen free) has neither: both 0. Where no contact point was found they are `inf`."""

    tied: bool
    visible: bool
    error_m: float
    gap_m: float


NO_STOP = Stopper(True, True, 0.0, 0.0)           # nothing stops the corridor: nothing to tie or to see
UNTIED = Stopper(False, False, math.inf, math.inf)   # no contact point of the stopper was found


def stopper(w: World, start: tuple[float, float], heading: float, gold: dict, states: dict) -> Stopper:
    """The `Stopper` of an answerable `gold` (a number): the thing the corridor from `start` along `heading` stops at,
    read in the keyframe of `states` (the free-space cells the gold read, `World.latest_cells`) that last observed the
    stopping fine cell. A gold at the cap has no stop (`NO_STOP`). ValueError when the inputs carry no frame
    (`Inputs.frame_of`)."""
    robot, cap, c_m = w.robot, cap_of(w.robot), w.job.clearance_m
    if gold["value"] >= cap - EPS:
        return NO_STOP
    if (found := grid.stop(states, robot, start, heading, cap)) is None:
        return NO_STOP
    if w.inp.frame_of is None:
        raise ValueError("these inputs carry no frames (Inputs.frame_of), so no stopper can be read")
    j = states[found.cell][1]
    frame, row = w.inp.frame_of(j), w.rows[j]
    floor_z, cam_pos = row["pose"]["pos"][2], np.asarray(row["camera"]["pos"], dtype=float)
    pts, axis = frame.points, np.array([math.cos(heading), math.sin(heading)])
    rel = pts[:, :2] - np.asarray(start)
    along, across = rel @ axis, rel @ np.array([-axis[1], axis[0]])
    height, half = pts[:, 2] - floor_z, robot.body.width_m / 2
    lo, hi = grid.band(robot)
    subs = np.floor(pts[:, :2] / grid.SUB_M + grid.EPS).astype(np.int64) // grid.SUB_PER_FINE
    here = (subs[:, 0] == found.cell[0]) & (subs[:, 1] == found.cell[1])
    ahead = np.flatnonzero(here & (np.abs(across) <= half + EPS) & (height > lo + grid.EPS) & (height <= hi + grid.EPS))
    reached = _touched(w, along[ahead], across[ahead])
    reached[(reached < -EPS) | (reached > cap + EPS)] = np.inf
    if not len(ahead) or not np.isfinite(reached).any():
        return UNTIED
    nearest = int(np.argmin(reached))
    stop, who = float(reached[nearest]), str(frame.owner[ahead[nearest]])
    slant = np.linalg.norm(pts - cam_pos, axis=1)
    near = np.flatnonzero((frame.owner == who) & (height <= np.maximum(CONTACT_M, slant / w.cam.fy))
                          & (np.abs(across) <= half + c_m / 2 + EPS))
    touched = _touched(w, along[near], across[near], widen=c_m / 2)
    keep = (touched >= stop - grid.SUB_M) & (touched <= stop + c_m / 2 + EPS)
    contact, reach = near[keep], touched[keep]
    if not len(contact) or (h := abs(cam_pos[2] - floor_z)) <= CONTACT_M:
        return UNTIED
    d2 = np.hypot(pts[contact, 0] - cam_pos[0], pts[contact, 1] - cam_pos[1]) ** 2
    errors = (d2 + h * h) / (h * w.cam.fy)
    least = int(np.argmin(errors))
    error, gap = float(errors[least]), max(0.0, float(reach[least]) - stop)
    if not error_tied(error, c_m):
        return Stopper(False, False, error, gap)
    visible = not who or w.tl.entity(who).mobility in ("fixed", "structure") or j in w.recognised(who)
    return Stopper(True, visible, error, gap)


def stopper_tied(w: World, k: int, start: tuple[float, float], heading: float, gold: dict, states: dict) -> bool:
    """Whether the stopper of an answerable `gold` (a number) is tied and visible (the module docstring): `stopper`'s
    two verdicts together. A gold at the cap is asked (no stopper)."""
    found = stopper(w, start, heading, gold, states)
    return found.tied and found.visible


def screen(w: World, k: int, start: tuple[float, float], heading: float, gold: dict, states: dict,
           blind: float | None = None) -> float | None:
    """The resolution `rho` of an answerable ground corridor gold when it is asked, else None: the height, side and
    stopper screens all pass (the cheap ones first; the stopper reads a frame). `states` are the free-space cells the
    gold read (`World.latest_cells`) and `blind` the floor it did not count (`robots.blind_m` when None: a corridor
    ahead; 0 for another move, `clear_move.blind`). The resolution is the largest of the grid's sub-cell, the stopper's
    least floor-contact pixel error and the gap from the stop to that contact (`Stopper`), so a reader of the frames
    can reach the full credit there; a gold at the cap or over cells seen free throughout has the sub-cell alone."""
    blind = blind_m(w.robot) if blind is None else blind
    if not (height_clear(w, k, start, heading, gold) and side_stable(w, start, heading, gold, states, blind)):
        return None
    found = stopper(w, start, heading, gold, states)
    if not (found.tied and found.visible):
        return None
    return max(grid.SUB_M, found.error_m, found.gap_m)


def asked(w: World, k: int, start: tuple[float, float], heading: float, gold: dict, states: dict,
          blind: float | None = None) -> bool:
    """Whether an answerable ground corridor gold is asked: `screen` gives a resolution."""
    return screen(w, k, start, heading, gold, states, blind) is not None
