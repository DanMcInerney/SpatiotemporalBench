"""`where_was_i` (during, `memory`, own path): where the robot's base was at a clock the stream delivered a while ago.

The named step is one of the window's steps at least `MIN_AGO_S` before the decision and before the look window that
ends at it (the frames just before it, which a memoryless answer reads), drawn by the case id and the decision alone,
so the clock asked reveals nothing of the world. The gold is the **base** at that step, from the delivered pose row
(never the camera, which stands ahead of it), to 4 decimals: the path the robot drove, recorded in the pose
stream, so it is always answerable and the evidence is the one step. No frame, depth or detection is read.

Why metres and not the room of design §8.2: the room the robot was in at a clock is predicted by the plan's room order
at that offset (`other_case`) and by the room it is in now (`current_view_prior`), so it would fail its gates by
construction; the metres are predicted by neither, and the memoryless answer, where the robot is now, is what
`current_view` gives, so the memory-needed gate measures it.

A family module (`questions_text.FAMILY_MODULES`): `SPECS`, `GOLDS`, and `SLICED` (none: a during family draws no
slice)."""

from __future__ import annotations

from .questions_select import Cand
from .questions_words import DISTANCE_LADDER
from .questions_world import World
from .records import FamilySpec
from .seeding import stable_hash

MIN_AGO_S = 60.0  # the named step lies at least this long before the decision


def where_was_i(w: World, k: int) -> Cand | None:
    """The question asked at the decision point at script keyframe `k`, with its gold; None (a dead band: the point is
    not asked) when no step of the window is old enough. The eligible steps are the window's keyframes before the
    first of the decision's look window with `MIN_AGO_S` or more between them and `k`, in keyframe order; the named
    step is the `stable_hash(case id, k, "where_was_i")`-th of them, modulo their number."""
    now = w.tl.keyframe(k).t
    eligible = [s for s in range(w.rec.k0, w.window_ks(k).start) if now - w.tl.keyframe(s).t >= MIN_AGO_S]
    if not eligible:
        return None
    s = eligible[stable_hash(w.rec.case, k, "where_was_i") % len(eligible)]
    x, y = w.base(s)
    return Cand("where_was_i", {"clock": w.clock(s)}, {"type": "position", "xy": [round(x, 4), round(y, 4)]},
                "answerable", (s,), (), {"ladder": list(DISTANCE_LADDER)}, decision_k=k,
                subjects=(f"w:{k}", f"w:{s}"), critical=True)


SPECS = (FamilySpec("where_was_i", "during", "position", "ladder", ("pose",), "memory", "path_log", False, ("clock",),
                    lambda s: f"Where were you at {s['clock']}?"),)
GOLDS = {"where_was_i": where_was_i}
SLICED: frozenset[str] = frozenset()
