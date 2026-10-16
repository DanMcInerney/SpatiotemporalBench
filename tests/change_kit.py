"""What the tests of the change families (`questions_change`) share, on `space_kit` and `family_kit`: a job that asks
them, and builders of things and people that move in the gaps between tasks of `vacuum_run`.

`vacuum_run(n)` is n tasks of 28 keyframes, 1.5 s apart inside a task and 7,200 s from one task's start to the next's:
task v's keyframe i is k = 28 v + i at t = 7200 v + 1.5 i. Its events resolve to offsets 2 ('when you set off'), 7
('when you began sweeping'), 18 ('when you finished sweeping') and 26 ('when you got home') of every task, and the
window's last look window is the last task's offsets 22 to 27. A thing or a person moves "in a gap" by a placement that
changes at `GAP_S` (between tasks 0 and 1) or `GAP2_S` (between 1 and 2); a sighting is `see` at a keyframe with 1,500
px (recognisable), 500 px (a glimpse) or 1,500 px at a mean luminance under the floor (dark).

`CHANGE_JOB` is `space_kit.JOB` asking the three families at the end and no during family; `window` is a window of its
tasks as a `World`, `golds` every candidate question of a world by what it asks, for the hidden-change checks;
`many_things` is a script of many moved bags, for the tests of how `object_was` asks them.
"""

from __future__ import annotations

from dataclasses import replace

import family_kit as fk
from space_kit import BAG_SIZE, JOB, SpaceScript, vacuum_run
from stbench import questionnaire
from stbench.questions_world import World

ENDS = ("object_was", "same_place", "person_where")
CHANGE_JOB = replace(JOB, ends=ENDS, decisions={})
GAP_S, GAP2_S = 7000.0, 14100.0   # between tasks 0 and 1, and 1 and 2
DARK = 0.005                      # a mean luminance under the recognisability floor
PERSON_BOX = (0.5, 0.5, 1.7)
SET_OFF, BEGAN, FINISHED, HOME = "when you set off", "when you began sweeping", "when you finished sweeping", \
    "when you got home"
WORDINGS = (SET_OFF, BEGAN, FINISHED, HOME)


def at(x_y: tuple[float, float]) -> tuple[float, float, float]:
    """A floor placement at (x, y)."""
    return (*x_y, 0.0)


def thing(s: SpaceScript, eid: str, kind: str, colour: str, places: list[tuple[float, float]], seen=(),
          gaps: tuple[float, ...] = (GAP_S, GAP2_S), *, px: int = 1500, lum: float | None = None) -> str:
    """A carried thing standing at `places[0]`, then at `places[1]` from the first gap on, and so on (one place a gap
    given), sighted at the keyframes `seen` with `px` pixels at mean luminance `lum` (the kit's default when None). A
    bag has the catalog's size; any other kind the default box."""
    times = [0.0, *gaps[:len(places) - 1], None]
    where = [(times[i], times[i + 1], at(p)) for i, p in enumerate(places)]
    s.item(eid, kind, {"color": colour}, where, size=BAG_SIZE if kind == "bag" else (0.1, 0.1, 0.1))
    s.see(list(seen), eid, px, lum=lum)
    return eid


def person(s: SpaceScript, eid: str, steps: list[tuple[float, tuple[float, float]]], seen=(), *, kind: str = "person",
           colour: str | None = None, px: int = 1500) -> str:
    """A mover whose placements are `steps` ((from time, (x, y)), each until the next), sighted at `seen`; a person has
    no attributes, any other mover the colour given."""
    where = [(t, steps[i + 1][0] if i + 1 < len(steps) else None, at(p)) for i, (t, p) in enumerate(steps)]
    s.item(eid, kind, {} if colour is None else {"color": colour}, where, mobility="mover", size=PERSON_BOX)
    s.see(list(seen), eid, px)
    return eid


def window(s: SpaceScript, a: int = 0, b: int | None = None, cid: str = "c1", job=CHANGE_JOB) -> World:
    """The window of tasks a..b of the script as a `World` of `job` (its length by the task count)."""
    b = a if b is None else b
    return fk.world(s, a, b, cid, "short" if a == b else "medium", job)


def golds(w: World) -> dict[str, dict]:
    """Every candidate question the world could ask by what it asks (family and slots), with its gold: what must not
    move when something the robot never saw changes."""
    return {c.key: c.gold for c in questionnaire.candidates(w)}


# The colours of `many_things`: one neutral only (black, grey and white are one group, so two bags of them would each be
# a look-alike of the other and neither asked).
MANY_COLOURS = ("red", "blue", "green", "black", "brown", "orange", "yellow", "pink", "purple", "cyan", "teal", "beige")


def many_things(n: int = 12, *, y: float = 0.5, dy: float = 2.25, ids: bool = True) -> SpaceScript:
    """Two tasks and `n` (at most 12) bags of distinct colours (`MANY_COLOURS`), each at (0.5 + 0.55 i, `y`) and from
    the gap after task 0 `dy` m further in y (past the distance ladder's last rung, 2 m, so a notebook at the wrong
    place scores -1), seen at keyframe 1 (task 0) and 29 (task 1), nowhere near the last look window (keyframes 50 to
    55). `ids` False numbers the entities backwards: the same descriptions on other ids."""
    s = vacuum_run(2)
    for i in range(n):
        thing(s, f"E{i + 1 if ids else 100 - i:04d}", "bag", MANY_COLOURS[i],
              [(0.5 + 0.55 * i, y), (0.5 + 0.55 * i, y + dy)], [1, 29])
    return s
