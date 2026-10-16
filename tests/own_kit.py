"""What the tests of `path_log`'s own-log bar (`baselines_path_own`) share for `open_heading`: a heading, a gold row
whose profile is told by hand, and a prior of other scripts' records that carry the rule's answer (`own`).

`heading_gold(best)`: a `regret` gold row with four headings, 90 degrees apart (the robot's own, 90 left, 180, 90
right), the run 3.0 m at `best` degrees and 0.2 m (under the threshold, so harmful) at the others; the vacuum's need
0.10 m, resolution 0.02 m, threshold 0.44 m. By hand (`score_graded.regret_credit`, `credit_kernel.graded`):
- an answer at `best` earns credit 1, so the penalised score at t = 0.5 is 1 - 0 = +1 against abstaining's 0;
- an answer at another heading runs 0.2 m, a regret of 2.8 m past 4N (N = 0.05 m, the harmful side's halved need) and
  under half the best run: no direction, credit 0, so the penalised score is 0 - 1 = -1.
"""

from __future__ import annotations

from baseline_kit import question, record
from score_kit import rec
from stbench.blind import Prior


def heading(deg: float) -> dict:
    return {"type": "heading", "deg_left": deg}


def heading_gold(best: float) -> dict:
    """The `open_heading` gold row (the module docstring) whose best heading is `best` degrees left."""
    profile = [3.0 if i == (best // 90) % 4 else 0.2 for i in range(4)]
    return rec(heading(best), credit="regret", family="open_heading",
               tolerances={"profile_m": profile, "step_deg": 90, "resolution_m": 0.02, "need_m": 0.10,
                           "threshold_m": 0.44})


def heading_evidence(own: float, best: float, scripts: int = 4, each: int = 2, first: int = 0) -> Prior:
    """A prior of other scripts' `open_heading` records (`each` a script, `scripts` scripts named from s`first`) that
    all carry the rule's own answer `own` degrees left and the gold row `heading_gold(best)`: the evidence `kept` weighs
    the rule by. Each record earns +1 over abstaining when `own` is `best`, else -1."""
    q = question("open_heading", {"cap": 3.0}, answer_type="heading")
    return Prior({**record(q, heading(best), script=f"s{first + i}", phase="transit", when="during"),
                  "own": heading(own), "gold": heading_gold(best)} for i in range(scripts) for _ in range(each))
