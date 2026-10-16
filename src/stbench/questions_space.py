"""The two corridor families, both during and both `grid.corridor_clear`: `path_ahead` and `rear_clear`.

`path_ahead` (`now`): how far the robot could move straight ahead, from the depth it was just sent. The gold is
`grid.path_ahead` over the latest free-space cells of the look window that ends at the decision point, at the decision's
delivered pose: the one function a system's mapper calls too, over the same depth, so the oracle reproduces it exactly.
An answer about "now" from older frames would be a guess about space that may have changed (design §8.1), so no other
look window is read. A corridor that was not seen empty is `unknown`; evidence is the look window's keyframes that
observed it, never empty on an answerable gold.

`rear_clear` (scope by evidence): how far the robot could reverse straight back, as it last saw the space behind it. A
robot's camera faces forward, so the corridor behind it was seen on the way in (or not): the gold reads the latest
free-space cells over the window's keyframes up to the decision, nothing before the window (design §5), from the rear
edge along the heading's opposite, with no blind band (a cell was seen, or it was not). The scope is `now` exactly when
every keyframe that observed the corridor is the decision's own frame `k`, the latest one (`questions_world.scope_of`),
else `memory`.
Floor markings are floor to the robot (`grid.band`: under its `climb_m`), so paint never stops the corridor; a pallet, a
tote or a person does.

Both gold rules read the recorded depth, but `needs` is what a system needs to answer (`stream_protocol.provides`): the
camera's picture and the pose. A system on the camera alone (`rgb_pose`) judges the distance from the picture; one given
the depth (`detections`, the gate channel) may use it.

A family module (`questions_text.FAMILY_MODULES`): `SPECS`, `GOLDS`, and `SLICED` (none: a during family draws no
slice).
"""

from __future__ import annotations

import math

from . import grid
from .questions_select import Cand
from .questions_words import DISTANCE_LADDER, cap_of
from .questions_world import World, scope_of
from .records import FamilySpec


def _render(slots: dict) -> str:
    return (f"How far could you move straight ahead now before reaching anything? "
            f"Answer {slots['cap']:g} if nothing is within {slots['cap']:g} m.")


def _render_rear(slots: dict) -> str:
    return (f"If you reversed straight back now, how far could you go before reaching anything you have seen? "
            f"Answer {slots['cap']:g} if nothing is within {slots['cap']:g} m.")


def _corridor_cand(family: str, w: World, k: int, answer: dict, evidence: list[int], scope: str | None = None) -> Cand:
    """The candidate of a corridor question at the decision at script keyframe `k`. Every during question is about the
    robot's own manoeuvre, so it is critical; `unknown` is a gold, not a dead band."""
    return Cand(family, {"cap": cap_of(w.robot)}, answer, "unknown" if answer["type"] == "unknown" else "answerable",
                tuple(evidence), (), {"ladder": list(DISTANCE_LADDER)}, decision_k=k, subjects=(f"w:{k}",),
                critical=True, scope=scope)


def path_ahead(w: World, k: int) -> Cand | None:
    """The question asked at the decision point at script keyframe `k`, with its gold. Never None."""
    answer, evidence = grid.path_ahead(w.latest_cells(w.window_ks(k)), w.robot, w.base(k), w.yaw(k), cap_of(w.robot))
    return _corridor_cand("path_ahead", w, k, answer, evidence)


def rear_clear(w: World, k: int) -> Cand:
    """The question asked at the decision point at script keyframe `k`, with its gold: `grid.corridor_clear` over the
    latest cells of the window's keyframes up to `k`, from the rear edge along the delivered heading plus pi, no blind
    band. Never None: the rule's `unknown` (the corridor was not seen empty) is a gold, not a dead band."""
    heading = w.yaw(k) + math.pi
    answer, evidence = grid.corridor_clear(w.latest_cells(range(w.rec.k0, k + 1)), w.robot,
                                           grid.front_edge(w.robot, w.base(k), heading), heading, cap_of(w.robot), 0.0)
    return _corridor_cand("rear_clear", w, k, answer, evidence, scope_of(w, k, evidence))


SPECS = (FamilySpec("path_ahead", "during", "number", "ladder", ("see", "pose"), "now", None, True, ("cap",),
                    _render),
         FamilySpec("rear_clear", "during", "number", "ladder", ("see", "pose"), "evidence", None, True, ("cap",),
                    _render_rear))
GOLDS = {"path_ahead": path_ahead, "rear_clear": rear_clear}
SLICED: frozenset[str] = frozenset()
