"""The bar every own-log rule of `path_log` stands by, and the one place that builds such a rule's answer.

An own-log rule is a fixed rule over the robot's own delivered poses and nothing else (no camera, no map, no pool of
answers): the detour of `clear_path` (`baselines_path_clear`), the longest run of `open_heading`
(`baselines_path_heading`) and the way back of `way_to` (`baselines_path_way`). A rule is no proof of anything, so like
every answer a shortcut takes with the records of other scripts in hand it stands only where it beats abstaining on
them: the records of its template that it answers (`own` a number, a heading, anything but `unknown`) come from enough
scripts (`blind.supported`) and the lower end of the 95% interval of its gain over abstaining, over those scripts, is
above 0 (`kept`; `blind.gain_low`), else it answers `unknown`. The gain is `score.penalised` of the rule's answer less
`unknown`'s on the record's gold row, so it reads whatever credit the gold row names (the ladder of a number, the regret
of a heading). A prior that is `in_pool` keeps the rule unasked, as every in-pool twin keeps its pool's own winner. The
rule learns nothing from the records but whether to stand.

A module that holds a rule defines `OWN_RULES: dict[str, OwnRule]` (template -> rule) and answers the template with
`answer`; `OWN_MODULES` names them. `own_answer` is the rule's answer from the poses the robot was sent, built once for
both: `answer` (the system at the ask) calls it over its log, and `suite_priors` over the delivered rows of a prior
record's script up to the ask's step, and stores it as the record's `own` (`answer_keys.RECORD_KEYS`), so what `kept`
scores is what the system answers. The modules import this one, so the rules are found when first asked for.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

from . import answer_registry
from .answer_candidates import UNKNOWN
from .blind import TIE, gain_low, supported
from .robots import Robot
from .score import penalised

if TYPE_CHECKING:
    from .baselines_path import PathLog

OWN_MODULES = ("baselines_path_clear", "baselines_path_heading", "baselines_path_way")
Pose = tuple[float, float, float]   # a delivered base pose: x and y in metres, yaw in degrees
# (the robot, the delivered poses of every step up to the ask, the question's slots, the ask step's base (x, y) and yaw
# in degrees) -> the answer, `unknown` when the rule has none
OwnRule = Callable[[Robot, Sequence[Pose], dict, tuple[float, float], float], dict]


@lru_cache(maxsize=1)
def rules() -> dict[str, OwnRule]:
    """The own-log rules by template, from the `OWN_RULES` of `OWN_MODULES`."""
    return answer_registry.collect(OWN_MODULES, "OWN_RULES")


def own_answer(model: Robot, template: str, slots: dict, steps: Iterable[dict], base: tuple[float, float],
               yaw_deg: float) -> dict:
    """The own-log rule of `template`'s answer to a question with `slots` asked at the step with the base `base` and the
    heading `yaw_deg`, from the delivered pose of each of `steps` (`pose.pos` and `pose.yaw_deg`), every step the robot
    was sent up to the ask, in order. KeyError for a template with no rule (`rules`)."""
    poses = [(s["pose"]["pos"][0], s["pose"]["pos"][1], s["pose"]["yaw_deg"]) for s in steps]
    return rules()[template](model, poses, slots, base, yaw_deg)


def kept(prior: Sequence[dict], template: str) -> bool:
    """Whether the own-log rule of `template` stands on this prior: the records of `template` it answers (`own` not
    `unknown`) come from enough scripts (`blind.supported`) and the lower end of the 95% interval of its gain over
    abstaining, from a bootstrap over those scripts (`blind.gain_low`), is above 0. A prior that is `in_pool` keeps it
    unasked."""
    if getattr(prior, "in_pool", False):
        return True
    found = [p for p in prior if p["template"] == template and (p.get("own") or UNKNOWN)["type"] != UNKNOWN["type"]]
    if not supported(len(found), len({p["script"] for p in found})):
        return False
    gains: dict[str, list[float]] = {}
    for p in found:
        gains.setdefault(p["script"], []).append(penalised(p["own"], p["gold"]) - penalised(UNKNOWN, p["gold"]))
    return gain_low([sum(g) for g in gains.values()], [len(g) for g in gains.values()]) > TIE


def answer(pl: PathLog, question: dict, rec: dict) -> dict:
    """`path_log`'s answer to a question of a template with an own-log rule: the rule's (`own_answer`) over the steps in
    its log, from the base pose and heading of the ask's step, where it stands on the prior (`kept`, asked once a
    system), else `unknown`."""
    template = question["template"]
    own = own_answer(pl.model, template, question["slots"], [s for _, s in sorted(pl.log.steps.items())], rec["base"],
                     rec["yaw"])
    stands = own["type"] == UNKNOWN["type"] or pl.cached(f"{template}_kept", lambda: kept(pl.prior, template))
    return own if stands else dict(UNKNOWN)
