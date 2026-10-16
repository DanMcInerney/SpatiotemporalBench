"""What the tests of `questions_things` share, on `family_kit`: a job that asks the four end families, hand worlds of
one flat task to place a thing at a keyframe in, and a window of several tasks with things of every relevance kind.

`THINGS_JOB` is `space_kit.JOB` with `ends` naming `object_where`, `when_seen`, `order` and `distance_between` and no
during family (no registered job asks the three until the coordinator maps `vacuum/2`). `flat(n)` is one task of `n`
keyframes (1.5 s apart) with no decision point: the stream is cut only every 8 steps and after the last, so the end
look window is its last `n % 8` keyframes (56 to 59 of the default 60). `put` places a thing and says how it showed:
`recognised` keyframes (1,500 px), `glimpsed` ones (100 px: a sighting under the sure-seen bound), `detected` ones
(400 px: seen, but under the recognisable floor of 1,024 px) and `dark` ones (1,500 px at a mean luminance under the
floor).

`things_window(n)` is `family_kit.multi_run(n)` with the things of `SETTLED`, by kind and colour: the j-th thing of the
kind (i-th in `KINDS`) is recognisable from keyframe 1 + i of task (j + i) mod n (and again 8 keyframes on, for bags and
shoes) and stands still at (0.5 + 0.8 j, `STANDS_AT[kind]`); `UNSEEN` adds a green bag and an orange toy that no frame
shows. Every kind has several settled things (tasks 7,200 s apart, or 30 s and more apart inside one: the first
sightings of two things of a task are 1.5 s apart, so `order` pairs only across tasks), unseen things, and absent
descriptions left in the catalog's colours; the end look window of the last task (keyframes 22 to 27) shows none of
them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import family_kit as fk
from family_kit import TASK_FRAMES, multi_run
from space_kit import BAG_SIZE, JOB, SpaceScript, record
from stbench.job_registry import JobSpec
from stbench.questions_world import World
from stbench.seeding import stable_hex

ENDS = ("object_where", "when_seen", "order", "distance_between")
THINGS_JOB = replace(JOB, ends=ENDS, decisions={})
RECOG_PX, DETECTED_PX, GLIMPSE_PX, DARK = 1500, 400, 100, 0.005
KINDS = ("bag", "shoes", "box", "toy")
STANDS_AT = {"bag": 1.0, "shoes": 2.0, "box": 3.0, "toy": 4.0}   # the y of a kind's things in `things_window`
SETTLED = {"bag": ("black", "red", "yellow", "orange"), "shoes": ("black", "red", "blue"),
           "box": ("brown", "blue", "beige"), "toy": ("yellow", "orange", "pink", "purple")}  # home's, purple, beige
# No two things of kinds that can be mistaken for one another (bag, box and toy; shoes only for shoes) share a colour,
# and no two are neutral (black, grey and white are one group): a sighted look-alike hides a real thing and a decoy
# (`questions_where.nameable`, `never_seen`), and a toy's stacked blocks are a box's. The black bag is the one neutral
# of its kinds, so the box decoys left are green (no sighted green): red is the sighted red bag's, white the black
# bag's.
UNSEEN = {"bag": "green", "toy": "red"}


def flat(n: int = 60) -> SpaceScript:
    """One task of `n` keyframes in one room, with no decision point and no depth."""
    return SpaceScript(tasks=[[("kitchen", n, "cleaning")]], decisions=[], depth_ks=set())


def put(s: SpaceScript, eid: str, kind: str, color: str, xy: tuple[float, float] = (1.0, 1.0), *,
        recognised: Iterable[int] = (), glimpsed: Iterable[int] = (), detected: Iterable[int] = (),
        dark: Iterable[int] = (), moves: tuple[float, tuple[float, float]] | None = None) -> str:
    """A carried thing at `xy` (z 0), and the keyframes it showed at, by how: see the module docstring. `moves` is a
    (time, xy) it moves to for good: after that the thing stands elsewhere, whatever any frame saw."""
    where = [(0, None, (*xy, 0.0))]
    if moves is not None:
        where = [(0, moves[0], (*xy, 0.0)), (moves[0], None, (*moves[1], 0.0))]
    s.item(eid, kind, {"color": color}, where, size=BAG_SIZE if kind == "bag" else (0.1, 0.1, 0.1))
    s.see(list(recognised), eid, RECOG_PX)
    s.see(list(glimpsed), eid, GLIMPSE_PX)
    s.see(list(detected), eid, DETECTED_PX)
    s.see(list(dark), eid, RECOG_PX, lum=DARK)
    return eid


def things_window(n_tasks: int) -> SpaceScript:
    """The window of `n_tasks` tasks (the module docstring)."""
    s = multi_run(n_tasks)
    count = 0
    for i, kind in enumerate(KINDS):
        for j, color in enumerate(SETTLED[kind]):
            count += 1
            o = TASK_FRAMES * ((j + i) % n_tasks)
            put(s, f"E{count:04d}", kind, color, (0.5 + 0.8 * j, STANDS_AT[kind]),
                recognised=[o + 1 + i] + ([o + 9 + i] if kind in ("bag", "shoes") else []))
    for kind, color in UNSEEN.items():
        count += 1
        put(s, f"E{count:04d}", kind, color, (0.5, 5.0))
    return s


def world(s: SpaceScript, a: int = 0, b: int | None = None, cid: str = "c0", length: str = "short",
          job: JobSpec = THINGS_JOB) -> World:
    """The window a..b of the script as a `World` of `THINGS_JOB` (or `job`)."""
    return fk.world(s, a, b, cid, length, job)


def worlds(s: SpaceScript, ids: Iterable[str], a: int = 0, b: int | None = None,
           length: str = "short") -> Iterable[World]:
    """One `World` of the window a..b for each case id: the inputs are built once, as only the id changes."""
    inp, rec = fk.inputs(s, THINGS_JOB), record(s, a=a, b=b, length=length)
    return (World(inp, replace(rec, case=cid)) for cid in ids)


def ids(tag: str, n: int) -> list[str]:
    """`n` case ids that differ only in their hash, 16 hex characters like a real one."""
    return [stable_hex(tag, i, n=16) for i in range(n)]

