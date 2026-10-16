"""`clear_path` (during, asked at look points): how far the robot's body could move straight before it touched anything,
in every direction its job declares (0.9.0, docs/families/clear_path.md). The family's spec, its words and its
dispatch; the gold of a ground robot is `clear_ground.moment` and a flier's `clear_fly.moment`, each `(w, k) ->
Moment | None`.

The direction is drawn at each look point from the job's `clear_directions` (`clear_move.draw`): a ground robot is asked
ahead, back, or, after turning in place, left or right; a flier ahead or back at its height. The answer is a `number`
in metres from the edge of the body for that move, credited by `graded` (`credit_kernel`): the job's clearance C is the
need, the gold answered longer is the harmful side, the resolution a pixel of the camera that saw the stop. The scope is
by evidence (`"evidence"`): `now` exactly when the latest frame alone gives the same number, else `memory`, set on the
candidate (`Cand.scope`). It is a structure family (`reference` `map_reader`): a system that holds the place's layout
answers it without the sensors, so `stbench check` prints the map reader's score beside it as a design diagnostic, in no
results table (docs/methodology.md, rule 5);
the gold and the table systems call one rule (`mapper_gold`).

The family is asked at look points only (`LOOKS`), never sliced: a point offers its `answerable` golds and, where the
robot has not seen the stop, an `unknown` one (the job's `look_unknown_keep` thins those). The words state one constant,
the cap of a ground robot (`questions_words.cap_of`); the flier's text states none. The slots are built by
`questions_words.clear_slots`.
"""

from __future__ import annotations

from . import clear_fly, clear_ground
from .questions_select import Moment
from .questions_world import World
from .records import FamilySpec

ASK = "how far could your body go before it touched anything?"
SEEN = ("Answer from what you have seen, in metres from the {edge} of your body; answer {cap:g} if nothing is within "
        "{cap:g} m.")


def _render(slots: dict) -> str:
    direction = slots["direction"]
    if slots["mode"] == "fly":
        return (f"If you flew straight {direction} now at this height, how far could your body go before it touched "
                f"anything you have seen? Answer in metres from the edge of your body.")
    seen = SEEN.format(edge="back" if direction == "back" else "front", cap=slots["cap"])
    lead = {"ahead": "If you drove straight ahead now,", "back": "If you reversed straight back now,",
            "left": "If you turned in place to face your left and then drove straight,",
            "right": "If you turned in place to face your right and then drove straight,"}[direction]
    return f"{lead} {ASK} {seen}"


def clear_path(w: World, k: int) -> Moment | None:
    """The look point `k`'s moment: a flier's (`clear_fly.moment`), else a ground robot's (`clear_ground.moment`). None
    is a dead band."""
    return (clear_fly if w.robot.flies else clear_ground).moment(w, k)


SPECS = (FamilySpec("clear_path", "during", "number", "graded", ("see", "pose"), "evidence", "map_reader", True,
                    ("mode", "direction", "cap"), _render),)
GOLDS = {"clear_path": clear_path}
LOOKS = frozenset({"clear_path"})
