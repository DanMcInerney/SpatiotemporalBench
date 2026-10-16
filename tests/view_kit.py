"""What the `out_of_view` tests share, on `family_kit` and `space_kit`: a job that asks the family, hand-placed things
and a way to put the robot's camera on a place.

`VIEW_JOB` is `space_kit.JOB` asking `out_of_view` at `before_doorway` (and nothing at `room_done`) and no end family
(the things placed here are asked only about at the doorway). The kit's run is
the vacuum's (`space_kit.vacuum_run`, 28 keyframes a task): the decision points are at keyframes 5, 18 and 21 of each
task, the look window of the one at 21 is keyframes 19 to 21 (the one at 18 ends the look window 14-18, the stream's
cuts being those and the task's last step), and the robot stands at (-0.17, 0) heading +x at the look window's
keyframes, with a render. A task's other keyframes carry no depth, so a place counts as seen empty only where `looking`
says the robot looked. A second task's decision is at 49 (look window 47-49) and its keyframes begin at 28. `thing` adds
one thing standing still (or moving), `person` a resident, `looking` points the robot at a place, `ask` is the gold at a
decision point.
"""

from __future__ import annotations

from dataclasses import replace

import family_kit as fk
from space_kit import JOB, SpaceScript, record
from stbench.job_registry import JobSpec
from stbench.questions_select import Cand
from stbench.questions_view import out_of_view
from synth import RECOG

VIEW_JOB: JobSpec = replace(JOB, decisions={"before_doorway": ("out_of_view",), "room_done": ()}, ends=())
K = 21  # task 0's decision before a doorway: the look window 19-21, its first keyframe b0 = 19
K2 = 49  # task 1's, look window 47-49
CHAIR = (0.4, 0.4, 0.9)  # light furniture: tall, an item at any distance inside the radius
SHOES = (0.25, 0.1, 0.1)  # 0.1 m: the fallback height's own, well under 0.3 m
BAG = (0.374, 0.304, 0.314)  # 0.314 m: inside the band 0.25-0.35 m around 0.3 m
TOY = (0.15, 0.15, 0.2)  # 0.2 m: over the fallback's band, under the default's
VASE = (0.1, 0.1, 0.4)  # tall, but stood on a table in the kit
PERSON = (0.5, 0.4, 1.7)
PALLET = (1.2, 1.0, 1.0)  # a large thing: widening max(0, 1.2 / 2 - 0.25) = 0.35
SEEN_PX, GLIMPSE_PX = RECOG, 100  # 1,500 px: a recognisable sighting (over 1,024 px, clear of the edge); 100: a glimpse


def thing(s: SpaceScript, eid: str, xy: tuple[float, float] | None, size: tuple[float, float, float] = CHAIR, *,
          kind: str = "chair", mobility: str = "light", z: float = 0.0, at=(3,), px: int = SEEN_PX,
          attrs: dict | None = None, where: list | None = None) -> str:
    """A thing that stands still at `xy` on a surface `z` high, seen at the keyframes `at` at `px` pixels (never, for
    `at=()`); `where` replaces its placements with [(t0, t1, (x, y, z))] for a thing that moves. Its colour is its id,
    so no two things share a description (a twin) unless `attrs` says so."""
    s.item(eid, kind, {"color": eid} if attrs is None else attrs, where or [(0, None, (*xy, z))],
           mobility=mobility, size=size)
    s.see(list(at), eid, px)
    return eid


def person(s: SpaceScript, eid: str, xy: tuple[float, float], at=(3,), px: int = SEEN_PX) -> str:
    """A resident: a mover standing at `xy`, seen at the keyframes `at`."""
    return thing(s, eid, xy, PERSON, kind="person", mobility="mover", at=at, px=px, attrs={})


def looking(s: SpaceScript, k: int, base: tuple[float, float], yaw: float = 0.0) -> None:
    """The robot at `base` heading `yaw` at keyframe `k`, with a render there (the floor and the kit's wall)."""
    s.poses[k], s.yaws[k] = base, yaw
    if s.depth_ks is not None:
        s.depth_ks.add(k)


def ask(s: SpaceScript, k: int = K, a: int = 0, b: int | None = None, height: float | None = None,
        job: JobSpec = VIEW_JOB) -> Cand:
    """`out_of_view`'s candidate at the decision point `k` of the window of tasks a..b."""
    return out_of_view(fk.world(s, a, b, "c1", job=job), k, height)


def inputs_and_record(s: SpaceScript, a: int = 0, b: int | None = None):
    """What the swap check takes: the window's inputs (with the hand map) and its case record."""
    return fk.inputs(s, VIEW_JOB), record(s, a=a, b=b, cid="c1")
