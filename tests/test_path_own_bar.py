"""The bar every own-log rule of `path_log` stands by (`baselines_path_own.kept`, `rules`, `answer`): hand cases on
prior records of `open_heading` (`own_kit`: +1 over abstaining when the rule's answer is the gold's best heading, -1
when it is another) and of `clear_path` (`test_path_log_clear.evidence`)."""

from __future__ import annotations

from own_kit import heading_evidence
from stbench import baselines_path, baselines_path_own
from stbench.blind import Prior
from test_path_log_clear import evidence


def test_the_bar_is_the_lower_end_of_the_gain_over_the_scripts_not_its_mean():
    """Contract: `kept` holds when the lower end of the 95% interval of the rule's gain over abstaining, a bootstrap
    over the prior's scripts, is above 0: a mean above 0 is not enough. Hand cases, 2 records a script, each +1 when
    the rule is right and -1 when wrong:
    - 3 scripts right and 1 wrong: the mean gain is (6 - 2) / 8 = +0.5 a record, but a draw of 4 scripts holds 2 or more
      of the wrong one 26% of the time (binomial, p = 1/4), and then the gain is 0 or less, and 3 or more 5% of the
      time, a gain of -0.5: the interval's lower end is -0.5 and the rule is not kept;
    - 7 scripts right and 1 wrong: a draw of 8 holds 4 or more of the wrong one 1% of the time and 3 or more 7%, so
      the lower end is the gain with 3 of 8 wrong, (10 - 6) / 16 = +0.25, and it is kept.
    Regression caught: the rule kept on a mean that the luck of which scripts were drawn explains (the held-out round
    exists for that: the helper's detour lost 0.20 on a prior that beat abstaining on average), or kept only when every
    script gains."""
    right, wrong = heading_evidence(180, 180, scripts=3), heading_evidence(180, 0, scripts=1, first=3)
    assert not baselines_path_own.kept(Prior([*right, *wrong]), "open_heading")
    right, wrong = heading_evidence(180, 180, scripts=7), heading_evidence(180, 0, scripts=1, first=7)
    assert baselines_path_own.kept(Prior([*right, *wrong]), "open_heading")


def test_the_bar_weighs_the_records_of_its_own_template_that_the_rule_answered():
    """Contract: `kept` reads the records of the template it is asked for, and of those only the ones whose `own` is an
    answer (not `unknown`, not absent). Hand cases: eight `open_heading` records of 4 scripts that gain (+1 each) keep
    the heading rule and not the detour's (they are no records of `clear_path`); eight `clear_path` records that gain
    (own 0.6, gold 0.6) keep the detour's rule and not the heading's; eight records of 4 scripts whose `own` is
    `unknown` or missing, which the rule did not answer, are no evidence for either. Regression caught: one template's
    evidence keeping another's rule, or a rule kept on records it never answered (an abstention counted as a gain)."""
    heading, detour = heading_evidence(180, 180), evidence(0.6, 0.6)
    assert baselines_path_own.kept(heading, "open_heading") and not baselines_path_own.kept(heading, "clear_path")
    assert baselines_path_own.kept(detour, "clear_path") and not baselines_path_own.kept(detour, "open_heading")
    silent = [{**r, "own": {"type": "unknown"}} for r in heading]
    missing = [{k: v for k, v in r.items() if k != "own"} for r in heading]
    for records in (silent, missing):
        assert not baselines_path_own.kept(Prior(records), "open_heading")


def test_every_own_log_rule_is_answered_through_the_bar():
    """Contract: a template with an own-log rule (`baselines_path_own.rules`: the detour of `clear_path`, the run of
    `open_heading`) is answered by `baselines_path_own.answer` in `path_log`'s registry, the one place the bar is asked,
    and it has a record answer (`own_answer`) built from the same rule. Regression caught: a new own-log rule (or a
    rewrite of one) that answers without standing by the bar, which would be BROKEN the way the detour was."""
    rules = baselines_path_own.rules()
    assert {"clear_path", "open_heading"} <= set(rules)
    for template in rules:
        assert baselines_path.PATH_ANSWERERS[template] is baselines_path_own.answer, template
