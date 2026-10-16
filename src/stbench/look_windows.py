"""The question compiler's views of the robot's recent frames (DECISIONS 2026-10-14, FD-frame-pull; FD-questions-0.10).

Every system gets one frame per `observe`; nothing here is ever sent to a system. Two views, for two kinds of question:
- **The recent view** (`recent`, 0.10.0): the last `LOOK_WINDOW` frames of the current task up to a look point. A look
  point is any frame, so the view slides with it. The memory families read it: `ego_mem` asks of a thing seen before it
  and not in it, `person_mem` of a person last seen before it, and the case's last `LOOK_WINDOW` frames are the view the
  end questions are asked from (`World.end_window`).
- **Decision windows** (`windows`): runs of at most `LOOK_WINDOW` consecutive steps of a case's window, cut after every
  decision point (asked or not) and after every task's last step (until 0.10.0 the stream protocol's batch, then the
  look window a look point had to end). A decision point's own families read the window that ends at it
  (`World.window_ks`: `path_ahead`, `where_was_i`, `out_of_view`, `zone_clear`; no job asks one since 0.6.0), and
  `stbench view` marks them. No look family reads them.

Imports nothing from the package.
"""

from __future__ import annotations

from collections.abc import Iterable

LOOK_WINDOW = 8   # the most steps one view holds


def recent(task_k0: int, k: int) -> range:
    """The recent view at frame k (0.10.0, FD-questions-0.10 §3): the last `LOOK_WINDOW` frames of the current task up
    to and including k, the task beginning at `task_k0`. A memory family asks about a thing not seen in it. Any
    numbering, as long as both arguments share it."""
    if k < task_k0:
        raise ValueError(f"frame {k} lies before its task's first frame {task_k0}")
    return range(max(task_k0, k - LOOK_WINDOW + 1), k + 1)


def windows(n_steps: int, cuts: Iterable[int]) -> list[tuple[int, int]]:
    """Case-relative [lo, hi) decision windows from step 0: a window ends after `LOOK_WINDOW` steps, or after a step in
    `cuts`, whichever is first; the count restarts after every cut. Cuts outside [0, n_steps) are ignored."""
    cut, out, lo = set(cuts), [], 0
    for k in range(n_steps):
        if k + 1 - lo == LOOK_WINDOW or k in cut:
            out.append((lo, k + 1))
            lo = k + 1
    if lo < n_steps:
        out.append((lo, n_steps))
    return out
