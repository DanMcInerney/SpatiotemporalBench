"""The balance rule (FD-0.6.0 §8; plan §2 call 5): no constant answer may beat `all_unknown` on a family.

A family whose answers lean one way lets a system that has never seen the world score by always saying the common thing:
the yes of a coin that is not fair, the position most golds share, the empty set. The compile draws the classes to keep
that from happening (`questionnaire`: coins at 0.5, an unknown slice at 0.2, `task_things`' empties capped), and this
module reads whether the draw worked, from the golds alone, never from a system's answers.

For each family of a (layout, job) group the **best constant** is the one answer, of the types and candidates
`answer_candidates` offers, with the best mean penalised score (the rows' `score_gate`) over the golds of the family's
questions of the OTHER scripts: chosen leaving each script out in turn, then scored on the script left out, so a
constant that is only the best of many on the golds it was chosen from gains nothing. This is the zero-feature rule of
the form adversary (`form_check`) without its support floor and its inner held-out round: the outer round is already
the held-out score, and a rule that abstains on a thin pool would make this check pass for nothing. A time gold is
learned as a lag (`blind`).
Its **gain** is the paired difference of its rows and `all_unknown`'s (`score_stats.paired_difference`: pairs on (case,
question), the interval a bootstrap over scripts, N_BOOT resamples, seed 0). The family **fails** on the group when the
gain's mean is above 0 and its 95% interval lies above 0: some constant is shown to beat abstaining, and a family above
0 does not ship on that job. A group too thin to read (`thin`: some script's held-out pool is under the support floor
of the priors, `blind.supported`, which is fewer than four scripts or a handful of questions) is reported as such and
cannot fail, and a group of one script has no interval at all. The
check is made for every family the group asks, declared pools or not; where a family declares none (`ego_now`) the
constant is expected to score far under `all_unknown` (the offset a constant guesses is outside every rung).

The shares of the classes the questions were drawn in (`class_shares`) are reported beside the verdict, against the
draw's (`DRAWS`), and never gated: at a few dozen questions a job a fair coin's yes share has a standard deviation of
0.08, so a band on it would fail good draws half the time (plan §1.6), whereas the best-constant check is the
binding test of whatever the shares turned out to be.

This is truth-side code (it reads gold rows); the baseline-side parts it uses (`blind`, `answer_candidates`,
`form_check.Asked`) read gold as data.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from . import score
from .answer_candidates import UNKNOWN
from .blind import EPOCH_CLOCK, MIN_POOL_GOLDS, MIN_POOL_SCRIPTS, Pool, at_clock, learned_answer, supported
from .form_check import Asked, learnable
from .gates import TOL
from .questionnaire import COIN_PERCENT, UNKNOWN_PERCENT
from .records import BALANCE_POOLS, pool_of
from .score_stats import GATE, N_BOOT, paired_difference

# The share each pooled family's draw asks of one of its pools: (pool, share, "=" for the draw's own share or "<=" for
# a cap). `same_place`'s and `object_was`'s coins are fair; the sliced families ask the unknown slice
# `UNKNOWN_PERCENT`; `task_things` thins its empty golds to at most a third (plan §2 call 6).
DRAWS = {
    "object_where": ("unknown", UNKNOWN_PERCENT / 100, "="),
    "ego_mem": ("unknown", UNKNOWN_PERCENT / 100, "="),
    "object_was": ("later", COIN_PERCENT / 100, "="),
    "same_place": ("no", COIN_PERCENT / 100, "="),
    "task_things": ("empty", 1 / 3, "<="),
}


def describe(answer: dict) -> str:
    """A constant in words: `unknown`, `yes`, `no`, its type and numbers (`number 1.40`, `position 3.20 1.50`), or, for
    a set (`set`, `positions`, `cells`, `route`), `positions (empty)` or `positions of 3`."""
    kind = answer["type"]
    if kind == "yesno":
        return "yes" if answer["value"] else "no"
    for key in ("items", "points"):
        if key in answer:
            return f"{kind} of {len(answer[key])}" if answer[key] else f"{kind} (empty)"
    numbers = [f"{x:.2f}" for v in answer.values() for x in (v if isinstance(v, list) else [v])
               if isinstance(x, float | int) and not isinstance(x, bool)]
    return " ".join([kind, *numbers])


def _rows(group: list[Asked], answer_of) -> list[dict]:
    """The result rows of `group` (`score.row`), each question answered by `answer_of(asked)`."""
    return [score.row(answer_of(a), a.gold, when=a.question["when"], info=a.info) for a in group]


def best_constant_rows(group: list[Asked]) -> list[dict]:
    """The rows of the best constant of one family's questions, each answered by the best response over the golds of
    the questions of OTHER scripts (`form_check.learnable`: and, for a layout-constant family, public layouts only), as
    it stands (`blind.learned_answer` on an `in_pool` pool: no second held-out round), `unknown` with none."""
    answers: dict[tuple[str, object], dict] = {}
    memo: dict = {}

    def answer_of(a: Asked) -> dict:
        key = (a.script, a.gold["reference"])
        if key not in answers:
            records = Pool([{"script": o.script, "gold": o.gold, "clock": o.clock} for o in group if learnable(a, o)],
                           EPOCH_CLOCK, True)
            answers[key] = learned_answer(records, memo=memo)
        return at_clock(answers[key], a.clock)

    return _rows(group, answer_of)


def thin(group: list[Asked]) -> bool:
    """Whether some script's held-out pool (the family's golds of the other scripts) is under `blind.supported`'s
    floor, `MIN_POOL_GOLDS` golds of `MIN_POOL_SCRIPTS` scripts: with fewer than four scripts, or scripts that ask a
    handful of questions between them, a constant is the best of many on a few correlated golds and its interval is not
    to be read as a fault of the family."""
    return any(not supported(len(others := [o for o in group if o.script != a.script]),
                             len({o.script for o in others})) for a in group)


def in_sample_constant(group: list[Asked]) -> dict:
    """The best response over every script's golds of the family: the constant the held-out rule is named after in a
    report, in sample (its own score is `best_constant_rows`', not this answer's on these golds)."""
    records = Pool([{"script": a.script, "gold": a.gold, "clock": a.clock} for a in group], EPOCH_CLOCK, True)
    return learned_answer(records)


def class_shares(family: str, golds: Iterable[dict]) -> dict:
    """The shares of the family's golds by class, against its draw: `{n, classes, pools, draw}`. `classes` maps each
    `answer_class` present to its share of the golds (golds with no class are counted in `n` and in none of them);
    `pools` maps each pool of `records.BALANCE_POOLS` the family declares to its share; `draw` is `{pool, share, bound}`
    from `DRAWS` (`bound` "=" or "<="), null for a family with none. Empty maps for a family with no classed gold."""
    classes = [g.get("answer_class") for g in golds]
    n = len(classes)
    out: dict = {"n": n, "classes": {}, "pools": {}, "draw": None}
    if not any(classes):
        return out
    out["classes"] = {c: classes.count(c) / n for c in sorted({c for c in classes if c})}
    pools = [pool_of(family, c) for c in classes]
    out["pools"] = {p: pools.count(p) / n for p in BALANCE_POOLS.get(family, {}) if p in pools}
    if family in DRAWS:
        pool, share, bound = DRAWS[family]
        out["draw"] = {"pool": pool, "share": share, "bound": bound}
    return out


def read(asked: list[Asked], *, n_boot: int = N_BOOT, seed: int = 0) -> list[dict]:
    """The balance verdict of each family of ONE (layout, job) group's questions, in family order: `{layout, job,
    family, constant, mean, ci, n, scripts, thin, fails}`. `constant` names the in-sample best constant (`describe`);
    `mean` and `ci` are the held-out best constant's paired gain over `all_unknown` (`ci` None under two scripts);
    `thin` is `thin` of the group; `fails` is `mean` above 0 with `ci` above 0 in a group that is not thin: a thin one
    is reported and cannot fail. `layout` and `job` are the group's (`Asked.info`)."""
    by_family: dict[str, list[Asked]] = defaultdict(list)
    for a in asked:
        by_family[a.gold["family"]].append(a)
    out = []
    for family, group in sorted(by_family.items()):
        gain = paired_difference(best_constant_rows(group), _rows(group, lambda a: UNKNOWN), n_boot=n_boot, seed=seed,
                                 key=GATE)
        out.append({"layout": group[0].info["layout"], "job": group[0].info["job"], "family": family,
                    "constant": describe(in_sample_constant(group)), "mean": gain["mean"], "ci": gain["ci"],
                    "n": gain["n"], "scripts": gain["scripts"], "thin": (small := thin(group)),
                    "fails": not small and gain["mean"] > TOL and gain["ci"] is not None and gain["ci"][0] > 0})
    return out


def line(b: dict) -> str:
    """One entry of `read` in words: `BALANCE ok apartment_a vacuum/1 object_where: the best constant (over every
    script: unknown) scores +0.000 held out, over all_unknown (95% interval +0.000 to +0.000; 4 scripts, 120
    questions)`; the word after BALANCE is `FAIL` when it fails and `thin` when the group is too thin to read (it
    cannot fail, and the line says why)."""
    ci = "no interval: under two scripts" if b["ci"] is None else f"95% interval {b['ci'][0]:+.3f} to {b['ci'][1]:+.3f}"
    word = "FAIL" if b["fails"] else "thin" if b["thin"] else "ok"
    why = (f"; too thin to read, a held-out pool is under {MIN_POOL_GOLDS} golds of {MIN_POOL_SCRIPTS} scripts"
           if b["thin"] else "")
    return (f"BALANCE {word} {b['layout']} {b['job']} {b['family']}: the best constant "
            f"(over every script: {b['constant']}) scores {b['mean']:+.3f} held out, over all_unknown ({ci}; "
            f"{b['scripts']} scripts, {b['n']} questions{why})")


def failure(b: dict) -> str:
    """The line a failing entry adds to a verdict's failures: `line` and the rule."""
    return f"{line(b)}; a constant answer must not beat all_unknown, so the family does not ship on this job."
