"""Look points (0.10.0): where the look families (`JobSpec.looks`) ask during a job, and which family a slot draws.

**A look point is any frame of the case window** (FD-questions-0.10 §3): the harness serves one frame per `observe` and
asks right after it, so no frame needs to end anything. Which frames are asked is drawn from the case id and the plan
alone, never from the world, so a point being asked reveals nothing about what is in view (design §8.5):
- the candidates are every frame at least `LOOK_MIN_STEPS` into its task, so the robot has moved and the view is its
  own, except the task's last frame;
- the frames of the job's priority decision kinds (`JobSpec.look_decisions`, the decisions the task turns on: before a
  crossing, a grasp, a lowering) come first in the draw, in `stable_hash(case id, k, "ask")` order, then every other
  candidate in that order;
- a frame is kept when it is at least `LOOK_GAP_STEPS` from every frame kept, until the case holds `ceil(frames /
  LOOK_PER_FRAMES)` points;
- a job that names no look family has none.
The recent view that a memory family reads at a point is `look_windows.recent`: the last `LOOK_WINDOW` frames of the
current task up to the point, so it slides with the point.

What is asked at a point is the questionnaire's draw (`questionnaire.looks`): one question a **slot** of the job's slot
table (`JobSpec.look_slots`, any number of slots; 0.10.0's jobs use three: the harm families, then the situation and
memory families, then the rest), in slot order. Slot `i` draws its family at point `k` by `slot_family`: the family
whose cumulative percent range holds `stable_hash(case id, k, f"slot{i}") % 100`, none when the hash falls past the
slot's total (the rest of a slot asks nothing). The hash reads the case id, the point and the table, never the world, a
candidate or a gold, so which family a point draws reveals nothing about what is in view (design §8.5). A point where
the drawn family finds no candidate is a dead band, and nothing replaces it: no other family of the slot is tried.
"""

from __future__ import annotations

from .questions_world import World
from .seeding import stable_hash

LOOK_MIN_STEPS = 4      # a look point lies at least this many steps into its task (half the recent view)
LOOK_GAP_STEPS = 4      # asked look points are at least this many steps apart
# A case asks at most one look point for this many frames of its window, rounded up. 6, from the replayed draw on the
# dev root (36 cases a job, short and medium): at 8 a point falls once in 7.8 frames, at 6 once in 5.9, at 5.5 once in
# 5.5 and at 5 once in 5.1-5.4, since `LOOK_GAP_STEPS` keeps two points four frames apart and the draw reaches that
# limit; 6 gives the three slots a point 1.3 times the points of 8 and stops short of the gap-limited 5.5, below which
# more cap buys nothing.
LOOK_PER_FRAMES = 6


def points_cap(frames: int) -> int:
    """The most look points a window of `frames` frames asks: `ceil(frames / LOOK_PER_FRAMES)`."""
    return -(-frames // LOOK_PER_FRAMES)


def candidate_points(w: World) -> list[int]:
    """The script keyframes a look family may be asked at, before the draw: every frame of the window at least
    `LOOK_MIN_STEPS` into its task and not the task's last. Empty for a job that asks no look family."""
    if not w.job.looks:
        return []
    found = []
    for k in range(w.rec.k0, w.rec.k1):
        task = w.tl.task_of(k)
        if k - task.k0 >= LOOK_MIN_STEPS and k != task.k1 - 1:
            found.append(k)
    return found


def points(w: World) -> list[int]:
    """The look points asked, in keyframe order: the candidates drawn in order (the decision frames of
    `JobSpec.look_decisions` first, each group in `stable_hash(case id, k, "ask")` order), each kept when it is
    `LOOK_GAP_STEPS` or more from every point kept, up to `points_cap` of the window's frames. The draw reads the
    plan's decision points and the case id alone, never the world."""
    cid, kept = w.rec.case, []
    first = {d.k for d in w.decisions if d.kind in w.job.look_decisions}
    cap = points_cap(w.rec.k1 - w.rec.k0)
    for k in sorted(candidate_points(w), key=lambda k: (k not in first, stable_hash(cid, k, "ask"), k)):
        if len(kept) < cap and all(abs(k - o) >= LOOK_GAP_STEPS for o in kept):
            kept.append(k)
    return sorted(kept)


def slot_family(case_id: str, k: int, slot: int, shares: tuple[tuple[str, int], ...]) -> str | None:
    """The family slot number `slot` draws at look point `k`: the family of `shares` ((family, percent) pairs, in order)
    whose cumulative percent range holds `stable_hash(case id, k, f"slot{slot}") % 100`; None when the hash falls past
    the total of the percents."""
    h, upto = stable_hash(case_id, k, f"slot{slot}") % 100, 0
    for family, percent in shares:
        upto += percent
        if h < upto:
            return family
    return None
