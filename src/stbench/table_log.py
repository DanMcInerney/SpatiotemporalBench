"""The log a scripted system keeps of what it was sent: every delivered step by `k`, and the tasks.

A step is kept as `{k, clock, pose, camera, velocity, status, subgoal}`, whichever of those the channel delivered (a
job that states no subgoal sends none, and the neutral brief withholds it); pictures, depth and detections are never
kept here (the table reads them as they arrive and keeps what it makes of them). A task begins at a `task` message, and
its first step is the next one delivered: tasks are never found from clock gaps (map SG-H2). `task_steps` is what an
event resolves over (`events.resolve`), exactly as the compiler resolves it over the delivered steps: the last task's,
or the one a question names by its case index (a window of several tasks, `slots["task"]`); a task no message carried
has no steps, so an event over it resolves to nothing and the answer is `unknown`, never a raise. `steps_until`,
`nearest` and `task_of` are the other reads an answer makes of the log. A window is the steps one `observe` call carried
(`add_window`, `window_of`): the harness sends one frame a call, so under it a window is one step, and a caller that
observes the compiler's look window in one call (a hand case) gets the window the zone families' golds read for a past
moment. Nothing is written to disk: a long case costs no save per observe (map X-H17). Reads no pictures.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

STEP_KEYS = ("clock", "pose", "camera", "velocity", "status", "subgoal")   # what the log keeps of a step, beside `k`


@dataclass(frozen=True)
class TaskStart:
    index: int      # case-relative, as the `task` message gave it
    clock: str
    gap_s: float
    brief: str
    k0: int         # the first step delivered after the message


class Log:
    def __init__(self) -> None:
        self.steps: dict[int, dict] = {}
        self.tasks: list[TaskStart] = []
        self._pending: tuple[int, str, float, str] | None = None
        self._windows: dict[int, tuple[int, ...]] = {}   # k -> the `k`s of the observe call that delivered it

    def begin_task(self, index: int, clock: str, gap_s: float, brief: str) -> None:
        """A `task` message: the task starts at the next step added."""
        self._pending = (index, clock, gap_s, brief)

    def add(self, step: dict) -> None:
        """Keep the step's `STEP_KEYS` (those it has) under its `k`; a pending task starts here."""
        k = step["k"]
        if self._pending is not None:
            self.tasks.append(TaskStart(*self._pending, k))
            self._pending = None
        self.steps[k] = {"k": k, **{key: step[key] for key in STEP_KEYS if key in step}}

    def add_window(self, ks: Iterable[int]) -> None:
        """One `observe` call carried the steps `ks`: `window_of` of each is all of them."""
        window = tuple(ks)
        self._windows.update(dict.fromkeys(window, window))

    def window_of(self, k: int) -> list[int]:
        """The `k`s of the `observe` call that carried step `k`, in delivery order; `[]` for a step no call carried."""
        return list(self._windows.get(k, ()))

    def briefs(self) -> list[str]:
        """The task messages' briefs, in order."""
        return [t.brief for t in self.tasks]

    def task_steps(self, index: int | None = None) -> list[dict]:
        """The steps of the task whose `task` message carried the case index `index`, in `k` order; None: the last task
        logged (every step when no task message came). An index no message carried gives `[]`."""
        k1 = None
        if index is None:
            k0 = self.tasks[-1].k0 if self.tasks else min(self.steps, default=0)
        else:
            held = [n for n, t in enumerate(self.tasks) if t.index == index]
            if not held:
                return []
            k0 = self.tasks[held[0]].k0
            k1 = self.tasks[held[0] + 1].k0 if held[0] + 1 < len(self.tasks) else None
        return [self.steps[k] for k in sorted(self.steps) if k >= k0 and (k1 is None or k < k1)]

    def steps_until(self, k: int) -> list[dict]:
        """The logged steps up to and including `k`, in `k` order."""
        return [self.steps[j] for j in sorted(self.steps) if j <= k]

    def nearest(self, clock: str) -> dict | None:
        """The logged step whose clock is nearest `clock` (ISO), the earlier on a tie; None when no step has a clock."""
        want = datetime.fromisoformat(clock)
        timed = [(abs(at - want), at, k) for k, s in self.steps.items() if "clock" in s
                 for at in [datetime.fromisoformat(s["clock"])]]
        return self.steps[min(timed)[2]] if timed else None

    def task_of(self, k: int) -> int | None:
        """The case index of the task holding step `k` (the last task begun at or before it), None before any task
        message."""
        held = [t.index for t in self.tasks if t.k0 <= k]
        return held[-1] if held else None
