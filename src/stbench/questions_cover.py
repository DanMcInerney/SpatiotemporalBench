"""`uncovered` (during, `memory`): which cells of the room the robot is leaving it saw as open floor but did not clean.

Asked at a decision point of a kind that names it (`JobSpec.decisions`; `vacuum/2`: where a room's cleaning ends and
where a task sent home stops cleaning). The room is the public map's region that holds the delivered base at the
decision (`MapSpec.region_of`); a base in a doorway, a wall or outside any room is a dead band, not asked. Nor is a
room in which no cell is wholly inside and clear of every blocked rectangle (the apartment's hall): there the gold is
the empty set whatever the robot did, a fact of the layout a prior learns from the room alone (VISION rule 5).

The gold is `grid_cover.cover`, the one rule the table systems call too, over three things the window holds up to the
decision: the fine cells the robot's `brush` tool swept (`grid_cover.brush_swath` over every delivered step, the brush
on or off as `status.brush` says: an earlier task's sweeps of the room count, since a task that resumes a room finishes
what the stopped one left), the free-space states of every window keyframe (`World.latest_cells`: what was last seen,
nothing unobserved changes it), and the layout's blocked rectangles (`grid_cover.blocked`: the public map's walls and
fixed pieces, never a light piece's true footprint). A cell is named when all 25 of its fine cells were seen free and
none was swept; a cell nobody can judge (partly swept, partly seen, touching a blocked rectangle, not wholly in the
room) is don't-care, and every other cell is a negative (`grid_cover.cover`). So the swath alone cannot answer it, and a
hidden move changes no gold. The evidence is the keyframes that showed the named cells, with the decision's own. The
gold reads the recorded depth, but `needs` is what a system needs to answer: the camera's picture, the pose and the
status (the brush); a system on the camera alone judges the open floor from the picture.

A family module (`questions_text.FAMILY_MODULES`): `SPECS`, `GOLDS` and `SLICED` (none: a during family draws no slice).
"""

from __future__ import annotations

from . import grid_cover
from .questions_select import Cand
from .questions_world import World
from .records import FamilySpec


def _render(slots: dict) -> str:
    return "Which cells of the room you are leaving did you see as open floor but not clean?"


def uncovered(w: World, k: int) -> Cand | None:
    """The question asked at the decision point at script keyframe `k`, with its gold; None when the base is in no room.
    Every during question is about the robot's own manoeuvre, so it is critical; never `unknown`."""
    room = w.map.region_of(w.base(k))
    if room is None:
        return None
    polygon, blocked = w.map.region(room).polygon, grid_cover.blocked(w.map, w.robot)
    if not grid_cover.open_cells(polygon, set(), blocked):  # no cell could ever be named: the layout answers it
        return None
    ks = sorted(j for j in w.rows if j <= k)
    poses = [(*w.base(j), w.yaw(j), bool(w.rows[j]["status"][grid_cover.TOOL])) for j in ks]
    positive, dont_care, evidence = grid_cover.cover(
        w.latest_cells(ks), grid_cover.brush_swath(poses, grid_cover.brush_of(w.robot)), polygon, blocked)
    gold = {"type": "cells", "items": positive, "dont_care": dont_care}
    return Cand("uncovered", {}, gold, "answerable", tuple(sorted({*evidence, k})), (), {}, decision_k=k,
                subjects=(f"w:{k}", f"r:{room}"), critical=True)


SPECS = (FamilySpec("uncovered", "during", "cells", "f1", ("see", "pose", "status"), "memory", None, True, (),
                    _render),)
GOLDS = {"uncovered": uncovered}
SLICED: frozenset[str] = frozenset()
