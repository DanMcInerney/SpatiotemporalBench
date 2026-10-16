"""Whether an answer is a catastrophe (0.10.0, FD-questions-0.10 §4.3): its action would end in harm, so the reported
score (`score.reported`, never a gate's `score.penalised`) charges the row's weight.

A harm row's gold may carry `tolerances["catastrophe"]` (`records.TOLERANCES`): `{weight, rule, margin_m[, threshold_m |
critical]}`. The rule says what a catastrophic answer is, and each needs a harmful error larger than `margin_m` (the
compile writes `max(C, 2 rho)`: the job's clearance, or twice what one pixel of the camera resolves there, whichever is
larger), so an answer that is merely imprecise is never one:
- `over` (a number): the answer exceeds the gold by more than `margin_m` and, when the row has `threshold_m`, the gold
  is under it and the answer at or over it (a person who came closer than the body has room for, said to be clear);
- `radial` (an offset): the answer places the point farther from the camera than the gold, along the gold's bearing,
  by more than `margin_m` (`score_graded.radial_error`'s sign);
- `miss` (a list of places): some point of `critical` has no answered item within the row's `match_m`.

`unknown` is never a catastrophe (it takes no action), nor is an answer of another type than the gold's; a row without
the key, an `unknown` gold and a gold at the question's cap (the compile writes no key there) are scored as before.
Standard library and the answer types only: the scorer reads it.
"""

from __future__ import annotations

import math

from stbench.answers import Answer, Number, Offset, Positions, Unknown
from stbench.score_graded import radial_error

EPS = 1e-9


def _over(answer: Number, gold: Number, rule: dict) -> bool:
    if not answer.value - gold.value > rule["margin_m"] + EPS:
        return False
    threshold = rule.get("threshold_m")
    return threshold is None or gold.value < threshold <= answer.value


def _radial(answer: Offset, gold: Offset, rule: dict) -> bool:
    return radial_error(answer, gold) > rule["margin_m"] + EPS


def _miss(answer: Positions, record: dict, rule: dict) -> bool:
    match_m = record["tolerances"]["match_m"]
    return any(not any(math.dist(item.xy, point) <= match_m + EPS for item in answer.items)
               for point in rule["critical"])


def weight(answer: Answer, gold: Answer, record: dict) -> int:
    """The weight of the catastrophe the answer is against the gold row `record` (`gold` its parsed gold), 0 when it is
    none: the row has no `catastrophe` tolerance, its gold is `unknown`, or the answer is `unknown` or of another type
    than the gold's."""
    rule = (record.get("tolerances") or {}).get("catastrophe")
    if rule is None or isinstance(gold, Unknown) or isinstance(answer, Unknown) or type(answer) is not type(gold):
        return 0
    match rule["rule"]:
        case "over":
            hit = _over(answer, gold, rule)
        case "radial":
            hit = _radial(answer, gold, rule)
        case "miss":
            hit = _miss(answer, record, rule)
        case _:
            raise ValueError(f"gold {record.get('qid')!r}: unknown catastrophe rule {rule['rule']!r}")
    return rule["weight"] if hit else 0
