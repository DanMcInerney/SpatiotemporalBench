"""`path_log`'s answers to `closest_approach` and `approach_mem` (a module of `baselines_path.PATH_MODULES`; the golds
are `questions_approach`).

It reads no picture, so it never knows who was near or whether anyone was, or who was seen last. What its own log says
of people is nothing: it holds the robot's delivered velocity (half of the gold's arithmetic), but where a person stands
and how they move is no fact of the layout or of the robot's path. The people walk where they like, whatever the robot
does. So, like every question about people (`baselines_path_people`), it answers only what other scripts' records of
the template say, at the first level of `answer_keys.levels("path_log", ...)` that holds enough of them, by
`blind.learned_answer`: a best response is kept only where its held-out gain over abstaining has its interval above 0,
else `unknown`. It is never weaker than `blind`: everything `blind` knows is in `path_log`'s levels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .baselines_path_people import pooled

if TYPE_CHECKING:
    from .baselines_path import PathAnswerer

PATH_ANSWERERS: dict[str, PathAnswerer] = {"closest_approach": pooled, "approach_mem": pooled}
