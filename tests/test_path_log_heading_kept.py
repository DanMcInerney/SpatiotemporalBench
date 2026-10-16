"""`path_log`'s own-log rule for `open_heading` stands only where it beats abstaining on other scripts
(`baselines_path_own.kept`, as `clear_path`'s detour does: `test_path_log_clear`). The geometry is
`test_path_log_heading`'s: the vacuum driven along the line y 1.5 from x 0.5 to 2.5 heading +x, which the rule answers
180 degrees left (the run behind it, 2.0 m). The prior's records are `own_kit.heading_evidence`: each carries that
answer and a gold row scored by regret, so a record earns +1 over abstaining when its best heading is 180 and -1 when it
is another."""

from __future__ import annotations

from baseline_kit import EVENTS, question, started, step
from own_kit import heading, heading_evidence
from stbench import baselines_path_heading, baselines_path_own, mapspec
from stbench.baselines_path import PathLog
from stbench.blind import Prior
from test_path_log_heading import DRIVE, ask

UNKNOWN = {"type": "unknown"}


def test_the_heading_stands_only_where_it_beats_abstaining_on_other_scripts(tmp_path):
    """Contract (the rule for every answer a shortcut takes with other scripts' records in hand): the own-log heading
    answers only where the lower end of the 95% interval of its gain over abstaining, over the scripts of the prior, is
    above 0, scored by the heading's regret credit (`score.penalised`). Hand case: the drive answers 180.
    - Records whose best heading is 180 (4 scripts, 8 records): the answer earns credit 1, +1 a record, the interval is
      {+1}: it stands, 180.
    - Records whose best heading is 0, the run behind 0.2 m: credit 0, -1 a record, the interval is {-1}: `unknown`.
    - Records of only 2 scripts (4 records, under `blind.MIN_POOL_SCRIPTS` and `MIN_POOL_GOLDS`) that all gain, or no
      records: `unknown`: a rule judged on a prior too thin to judge by.
    - An `in_pool` prior, in-sample by design, keeps it unasked whatever its records say: 180.
    Regression caught: an own-log heading answered at a loss, like the detour's -0.20 on the helper (BROKEN, below
    `all_unknown`), or one judged on too few records."""
    losing = heading_evidence(180, 0)
    for name, prior, want in (("right", heading_evidence(180, 180), heading(180.0)), ("wrong", losing, UNKNOWN),
                              ("thin", heading_evidence(180, 180, scripts=2), UNKNOWN), ("none", Prior(), UNKNOWN),
                              ("in_pool", Prior(losing, in_pool=True), heading(180.0))):
        assert ask(tmp_path / name, DRIVE, prior=prior) == want, name


def test_the_bar_is_asked_once_a_system_and_only_of_a_rule_that_answers(tmp_path, monkeypatch):
    """Contract: `kept` (a bootstrap over the prior's scripts) is worked out once a system, not once a question, as
    `clear_path`'s is; and not at all when the rule has no answer to give (nothing to stand or fall). Hand case: two
    `open_heading` questions asked after the drive at once are both answered 180 with one call of `kept`, for this
    template; with every run of the rule set to 0 (no swept cell to start from) they are `unknown` and it is never
    called. Regression caught: the bar recomputed for every question of a case (a cost per question that grows with the
    prior), or asked of a rule that has nothing to say."""
    calls: list[str] = []
    real = baselines_path_own.kept
    monkeypatch.setattr(baselines_path_own, "kept",
                        lambda prior, template: calls.append(template) or real(prior, template))
    prior = heading_evidence(180, 180)
    qs = [question("open_heading", {"cap": 3.0}, qid, "heading") for qid in ("q1", "q2")]

    def asked(name: str) -> dict:
        steps = [step(k, x, y, phase="transit") for k, (x, y) in enumerate(DRIVE)]
        system = started(PathLog, tmp_path / name, map=mapspec.load_map("apartment_a"), events=EVENTS, prior=prior)
        system.observe(steps)
        return system.ask("during", {"k": steps[-1]["k"], "clock": ""}, qs)

    assert asked("drive") == {"q1": heading(180.0), "q2": heading(180.0)}
    assert calls == ["open_heading"]
    calls.clear()
    monkeypatch.setattr(baselines_path_heading, "own_run", lambda *args: 0.0)
    assert asked("no-run") == {"q1": UNKNOWN, "q2": UNKNOWN}
    assert calls == []
