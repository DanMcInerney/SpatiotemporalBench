"""The evidence age of a gold row (0.10.0, FD-questions-0.10 §4.5, review S5): how old the evidence is when the question
is asked, so that a report can say how a system's score falls with the age of what it must remember.

`age_s` is the clock time in seconds from the gold's oldest evidence keyframe to the ask, read from the timeline's
keyframe times: the ask is a during question's `decision_k`, an end question's the window's last frame. A `now` row is
answered by the latest frame, which is what the system holds at the ask, so its age is 0 whatever evidence keyframe the
candidate names (final review S6: `open_heading` and `clear_path` `now` rows named an earlier frame whose cells alone
gave the gold, and were aged from it). `age_band` is `now` exactly when the row's scope is `now`; else `recent` when
`age_s` is at most `RECENT_S`, `task` when the evidence is in the task the ask falls in, `earlier` when it is in an
earlier task. Both are null on an `unknown` row (it names no evidence) and on a row whose candidate names none.
"""

from __future__ import annotations

from .questions_select import Cand
from .questions_world import World

RECENT_S = 10.0   # a `recent` row's greatest age: declared in the brief (§4.5), never tuned to a score
AGE_ROUND = 3     # decimals of `age_s`


def ask_k(w: World, c: Cand, when: str) -> int:
    """The script keyframe a question is asked at: a during question's decision point, else the window's last frame."""
    return c.decision_k if when == "during" else w.rec.k1 - 1


def age(w: World, c: Cand, when: str, scope: str) -> tuple[float | None, str | None]:
    """`(age_s, age_band)` of the candidate `c`, a question of `when` (`records.WHEN`) and the gold row's `scope`:
    `(None, None)` for an `unknown` slice or a candidate with no evidence keyframe."""
    if c.slice == "unknown" or not c.evidence_k:
        return None, None
    if scope == "now":
        return 0.0, "now"
    ask, oldest = ask_k(w, c, when), min(c.evidence_k)
    seconds = round(w.tl.keyframe(ask).t - w.tl.keyframe(oldest).t, AGE_ROUND)
    if seconds <= RECENT_S:
        return seconds, "recent"
    return seconds, "task" if oldest >= w.tl.task_of(ask).k0 else "earlier"
