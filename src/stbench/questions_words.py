"""The wording and sizes the question families share, so no family module imports another to reach them.

`DISTANCE_LADDER` is the distance credit's rungs, written on every gold row that scores a distance; `ORDINALS` and
`task_phrase` name a task of a window in a question's text ("in your second task"), the task being a case index (the
window's first is 0), so the text names nothing of the script; `MAX_CAP_M` and `cap_of` are the farthest a corridor
question asks about; `clear_slots` builds the slots of a `clear_path` question (0.8.0), which the ground and the flier
golds (`clear_ground`, `clear_fly`) share without importing the family's module.
"""

from __future__ import annotations

from .robots import Robot

DISTANCE_LADDER = (0.25, 0.5, 1.0, 2.0)  # metres: the rungs of the distance credit, written on every gold row
ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth")  # case.LENGTHS' longest window is 6 tasks
MAX_CAP_M = 5.0


def task_phrase(task: int | None) -> str:
    """The words that name a task of a window after a question's own: none for None (the window's only task), else
    " in your second task" for case index 1."""
    return "" if task is None else f" in your {ORDINALS[task]} task"


def cap_of(robot: Robot) -> float:
    """The farthest a corridor question asks about: 5 m, or a metre inside the depth range (3 m for the vacuum), so a
    corridor answered `cap` was seen to its end."""
    return min(MAX_CAP_M, robot.depth.far_m - 1.0)


CLEAR_MODES = ("drive", "fly")  # how the body moves: a ground robot drives, a flier flies
CLEAR_DIRECTIONS = ("ahead", "back", "left", "right")  # a ground robot may ask any; left and right turn in place first
FLY_DIRECTIONS = ("ahead", "back")  # a flier asks no sideways corridor (0.8.0: a held-out constant leads on it)


def clear_slots(mode: str, direction: str, cap: float) -> dict:
    """The slots of a `clear_path` question: `mode` (`CLEAR_MODES`), `direction` (`CLEAR_DIRECTIONS`) and `cap`, metres
    (`cap_of` for a ground robot; the flier's text does not state its cap). ValueError for a mode or direction that is
    not one of them, or for a flier's direction outside `FLY_DIRECTIONS`."""
    allowed = FLY_DIRECTIONS if mode == "fly" else CLEAR_DIRECTIONS
    if mode not in CLEAR_MODES or direction not in allowed:
        raise ValueError(f"a clear_path question is of mode {CLEAR_MODES} and, in mode {mode!r}, direction {allowed}, "
                         f"got {mode!r} and {direction!r}")
    return {"mode": mode, "direction": direction, "cap": cap}
