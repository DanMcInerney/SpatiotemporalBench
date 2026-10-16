"""Does the form of a question predict its answer? (VISION rule 6: nothing about a question's form may.)

The form adversary is the `form` shortcut of the gates. It is judged on a question as it is delivered: the length of its
text, its answer type, `when`, how many steps into its task the step it is asked about lies, its place in the
questionnaire, the phase of that step, and every value its slots hold (`features_of`). For each family it takes the
single feature whose rule scores best: each question is answered by the learned answer (`blind.learned_answer`: the best
response, kept when it beats abstaining on scripts it did not learn from; a time gold is learned as a lag behind the
clock its question was asked at) over the gold rows of the questions of OTHER
scripts that share the feature's value, `unknown` where they are under the support floor (`blind.supported`:
`MIN_POOL_GOLDS` golds of `MIN_POOL_SCRIPTS` scripts, as every prior system's: a value held by one or two worlds is
their accident, not a form that predicts). It has every script's gold and hindsight, so it is an
upper bound over features, not a system, and it learns only from other scripts (map X-H14: the cases of one script
share their world). A question's form must not predict its answer, so this must not
beat all-`unknown` by more than the shortcut margin (`gates`); a feature is named only when it beats the unknown share.
A family whose reference is `other_case` (a fact of the layout, the same in every case of it) is learned from public
layouts only, as every prior system learns it (`suite_priors`): a question of such a family on a held-out layout is
pooled with no held-out question, which is none at all in a root of held-out layouts, so it is answered `unknown`.

This module reads no file and no truth module: the caller (`check`) works out each question's step, place and phase and
hands them in with the question as delivered.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass

from . import map_files, score
from .answer_candidates import UNKNOWN
from .blind import EPOCH_CLOCK, Pool, at_clock, learned_answer, supported
from .gates import OTHER
from .score_stats import gate_score

TEXT_BUCKET = 12   # characters of a question's text per step of its length feature
TIE = 1e-12        # a mean this far above another is better; within it the earlier feature stands


@dataclass(frozen=True)
class Asked:
    """One question as delivered, with what the adversary may score it against."""

    case: str                      # the case's id
    script: str                    # its script's id: the adversary learns from other scripts only
    question: dict                 # the question row
    gold: dict                     # its gold row
    info: dict                     # `score_stats.INFO_KEYS`: what `score.row` says of the case
    features: dict[str, object]    # `features_of` the question
    clock: str | None = None       # the clock delivered at the step the ask is about: its time golds are lags behind it


def _slot_features(name: str, value: object) -> dict[str, object]:
    """A slot's value as features: a dict value contributes one per key (`name.key`, recursively), a list a JSON string,
    a scalar itself."""
    if isinstance(value, dict):
        out: dict[str, object] = {}
        for key in sorted(value):
            out |= _slot_features(f"{name}.{key}", value[key])
        return out
    return {name: value if value is None or isinstance(value, str | int | float | bool)
            else json.dumps(value, sort_keys=True)}


def features_of(question: dict, *, k_in_task: int, place: int, phase: str) -> dict[str, object]:
    """The features of a delivered question: `text_length` (its text's characters // `TEXT_BUCKET`), `answer_type`,
    `when`, `k_in_task` (the case's steps since its task began, at the step the ask is about), `place` (its index in
    the case's questions, in ask order), `phase` (that step's), and `slot:<name>` for each slot, `slot:<name>.<key>` for
    each key of a slot that is an object. Every value is hashable."""
    out: dict[str, object] = {"text_length": len(question["text"]) // TEXT_BUCKET,
                              "answer_type": question["answer_type"], "when": question["when"],
                              "k_in_task": k_in_task, "place": place, "phase": phase}
    for name in sorted(question["slots"]):
        out |= _slot_features(f"slot:{name}", question["slots"][name])
    return out


def learnable(asked: Asked, other: Asked) -> bool:
    """Whether the adversary may learn `asked`'s answer from `other`'s gold: `other` is of another script, and, when the
    family's reference is `other_case`, of a public layout (the module docstring)."""
    return other.script != asked.script and not (asked.gold["reference"] == OTHER
                                                 and other.info["layout"] in map_files.HELD_OUT)


def _feature_rows(group: list[Asked], name: str, answers: dict, memo: dict, in_pool: bool) -> list[dict]:
    """The result rows of the rule on feature `name`: each question answered by the learned answer over the gold of the
    other scripts' questions that share its value (a question without the feature has the value None), `unknown` when
    those are under the support floor. `answers` keeps each pool's answer, by the pool's indices in `group`; a time
    answer is kept as a lag (learned for an ask at `EPOCH`) and put on each asker's clock (`blind.at_clock`)."""
    by_value: dict[object, list[int]] = defaultdict(list)
    for i, a in enumerate(group):
        by_value[a.features.get(name)].append(i)
    rows = []
    for a in group:
        pool = tuple(j for j in by_value[a.features.get(name)] if learnable(a, group[j]))
        if pool not in answers:
            held = supported(len(pool), len({group[j].script for j in pool}))
            records = Pool([{"script": group[j].script, "gold": group[j].gold, "clock": group[j].clock} for j in pool],
                           EPOCH_CLOCK, in_pool)
            answers[pool] = learned_answer(records, memo=memo) if held else dict(UNKNOWN)
        rows.append(score.row(at_clock(answers[pool], a.clock), a.gold, when=a.question["when"], info=a.info))
    return rows


def form_adversary(asked: list[Asked], *, in_pool: bool = False) -> tuple[list[dict], dict[str, dict]]:
    """The best single-feature rule per family: `(rows, table)`.

    `rows` are the result rows (`score.row`) of each family's best feature, the one with the highest mean penalised
    score (the rows' `score_gate`, as the gates read it; the earlier feature on a tie, features in the order the
    questions list them). `table[family]` is `{feature, mean, unknown_share, n, means}`: `means` holds every feature's
    mean, `mean` the best feature's, `unknown_share` the share of the family's golds that are unknown (all-`unknown`'s
    score), and `feature` is the best feature's name, or None when it does not beat the unknown share, so no feature
    is named for a form that predicts nothing. With `in_pool` each pool's answer is its best response as it is,
    without the held-out round (the twin `check` also scores, `gates.in_pool_name`)."""
    groups: dict[str, list[Asked]] = defaultdict(list)
    for a in asked:
        groups[a.gold["family"]].append(a)
    rows: list[dict] = []
    table: dict[str, dict] = {}
    memo: dict = {}
    for family, group in sorted(groups.items()):
        answers: dict[tuple[int, ...], dict] = {}
        by_feature = {name: _feature_rows(group, name, answers, memo, in_pool)
                      for name in dict.fromkeys(name for a in group for name in a.features)}
        means = {name: statistics.fmean(gate_score(r) for r in rs) for name, rs in by_feature.items()}
        best = next(iter(means))
        for name, mean in means.items():
            if mean > means[best] + TIE:
                best = name
        rows += by_feature[best]
        unknown_share = sum(a.gold["slice"] == "unknown" for a in group) / len(group)
        table[family] = {"feature": best if means[best] > unknown_share + TIE else None, "mean": means[best],
                         "unknown_share": unknown_share, "n": len(group), "means": means}
    return rows, table
