"""`stbench check` (`check`, `suite_report`): the numbers a report shows by hand, how a failed gate is named, what is
refused before anything runs, what makes the verdict fail whatever the gates say, and the whole command over a
synthetic dataset through the real stream."""

from __future__ import annotations

import contextlib
import io
import json
import re

import pytest

from check_kit import EVENT, empty_case, hand_case, hand_synth, synthetic_dataset
from run_kit import scored_case
from score_kit import result_row
from stbench import baselines, check, cli, gates, suite, suite_report
from stbench.baselines import BASELINES, AllUnknown, Baseline
from stbench.blind import Blind
from stbench.stream import CaseRun
from stream_kit import end_q, qid
from synth import WAREHOUSE

NM = "current_view"
UNCLASSED = {"n": 4, "classes": {}, "pools": {}, "draw": None}   # four golds that declare no class
SYSTEMS = (*gates.BASELINE_NAMES, gates.FORM)


def number(v: float) -> dict:
    return {"type": "number", "value": v}


def at(x: float, y: float) -> dict:
    return {"type": "position", "xy": [x, y]}


def run_of(answers: dict) -> CaseRun:
    return CaseRun(answers, "complete", {}, {}, {}, None, "0" * 16)


def test_the_report_by_hand(tmp_path):
    """Contract (plan §4.14): a group's numbers are read from the right rows. Four scripts each with one case that
    asks `path_ahead` after step 12 and one `object_where`: the during questions' caps are 2, 2, 2 and 3, the end
    questions' mugs red, red, blue (an unknown gold) and red at another moment, and `current_view` is hand-fed 1.4,
    1.4, 0.3 and 3.0 m and the places (3.2, 1.5), (3.3, 1.6), (3.3, 1.55) and (1.2, 2.2). The hit rates, each over
    four questions, by hand: `blind` keys on the slots, which three `path_ahead` and two `object_where` questions
    share with another script (0.75, 0.5); `path_log` on phase and pose, which all four during questions share (1.0),
    and on the slots with the rooms, all kitchen (0.5); `current_view_prior` on phase, pose and the quantised answer,
    which two of the 1.4 m answers share (1-2 m) and three of the places (the cell 6,3), so 0.5 and 0.75. The reach
    rates (some level above the bare template shared): the same for `path_ahead` (its one slot level, the phase and
    pose, the quantised answer: nothing looser to find), and for `object_where` 1.0 for `blind` and `path_log`, whose
    kind-only level all four mugs share, and 0.75 for `current_view_prior`, whose looser levels hold the same
    quantised place. The now
    share is 4 of 8; `current_view` scores 1 on each 1.4 m answer and meets one rung of four (s = 0.25, penalised 2s - 1
    = -0.5) on 0.3 and on 3.0 against 1.4, a mean of (1 + 1 - 0.5 - 0.5) / 4 = 0.25 over the now rows, under 0.95; one
    `object_where` gold of four is unknown. Regression caught: a rate read from another family's questions or another
    system's key, or a mean over memory rows."""
    specs = [(2, "red", EVENT, False, (1.4, at(3.2, 1.5))), (2, "red", EVENT, False, (1.4, at(3.3, 1.6))),
             (2, "blue", EVENT, True, (0.3, at(3.3, 1.55))),
             (3, "red", "when you started cleaning", False, (3.0, at(1.2, 2.2)))]
    cases = [hand_case(tmp_path, hand_synth(n), cap=cap, color=color, event=event, unknown=unknown)
             for n, (cap, color, event, unknown, _) in enumerate(specs, 1)]
    runs = {(NM, case.id): run_of({qid(1): number(ahead), qid(2): place})
            for case, (*_, (ahead, place)) in zip(cases, specs, strict=True)}
    rows = {gates.CURRENT: [r for case in cases for r in suite.score_rows(case, runs[NM, case.id])]}
    numbers = suite_report.group_numbers(cases, runs, rows)
    assert numbers == {
        "cases": 4, "scripts": sorted(case.script.id for case in cases), "now_share": 0.5,
        "temporal": {"short": {"cases": 4, "answerable": 3, "temporal": 0, "share": 0.0, "per_case": 0.0}},
        "families": {
            "object_where": {"questions": 4, "per_case": 1.0, "unknown_share": 0.25, "empty_set_share": None,
                             "reference": None, "class_shares": UNCLASSED,
                             "hit_rates": {"blind": 0.5, "path_log": 0.5, "current_view_prior": 0.75},
                             "reach_rates": {"blind": 1.0, "path_log": 1.0, "current_view_prior": 0.75},
                             "current_view_now_mean": None},
            "path_ahead": {"questions": 4, "per_case": 1.0, "unknown_share": 0.0, "empty_set_share": None,
                           "reference": None, "class_shares": UNCLASSED,
                           "hit_rates": {"blind": 0.75, "path_log": 1.0, "current_view_prior": 0.5},
                           "reach_rates": {"blind": 0.75, "path_log": 1.0, "current_view_prior": 0.5},
                           "current_view_now_mean": 0.25}}}


def hand_rows(blind: list[list[float]], reference: str | None = None, **others: list[list[float]]
              ) -> dict[str, list[dict]]:
    """Rows of one `object_where` family on four scripts of two questions, the second with an unknown gold, for each
    system: the oracle and `memory_table` exact, `all_unknown` right on the unknown gold only, `blind` as given, the
    rest as `all_unknown` unless `others` gives a system's scores; every row says the family's `reference`."""
    scores = {name: [[0, 1]] * 4 for name in SYSTEMS}
    scores |= {gates.ORACLE: [[1, 1]] * 4, gates.TABLE: [[1, 1]] * 4, gates.BLIND: blind, **others}
    return {name: [result_row(f"c{i}", f"q{i}{j}", s, "correct" if s >= 1 else "wrong" if s < 0 else "abstained",
                              script=f"s{i}", family="object_where", scope="memory", reference=reference,
                              slice_="unknown" if j else "answerable")
                   for i, per_script in enumerate(per) for j, s in enumerate(per_script)]
            for name, per in scores.items()}


def test_a_failed_gate_is_named_with_its_numbers_and_the_verdict_picks_the_reading():
    """Contract (VISION rule 5, plan §4.14): `check` says plainly which gate of which family failed, by how much and
    through which system, and its exit follows `--verdict`: `point` the point estimates, `interval` the intervals. Four
    scripts of two questions, the second unknown, all_unknown 0.5. `blind` exact on all (1.0) beats all_unknown by 0.5
    (the gate allows 0.1) and ties `memory_table`, which beats it by 0 (the gate wants 0.4), failing both readings.
    `blind` exact on two scripts and wrong on two scores 0.5, level with all_unknown, while `memory_table` beats it by
    0.5 on the point estimate but, with two scripts in four unchanged, not on its interval's lower bound (it wants it
    above 0.3): the point reading passes and the interval reading fails, and says the point estimate held. Regression
    caught: an exit that ignores the verdict, or a failure line without the number or the system."""
    every = check.outcome(gates.evaluate(hand_rows([[1, 1]] * 4), n_boot=500), "point")
    assert every == (False, [
        "FAIL apartment_a vacuum/1 object_where shortcuts: blind beats all_unknown by +0.500; the gate allows at "
        "most 0.1.",
        "FAIL apartment_a vacuum/1 object_where separation: memory_table beats blind by +0.000 (95% interval +0.000 "
        "to +0.000); the gate wants at least 0.4 and the interval's lower bound above 0.3."])
    assert check.outcome(gates.evaluate(hand_rows([[1, 1]] * 4), n_boot=500), "interval") == every
    split = gates.evaluate(hand_rows([[0, 0], [0, 0], [1, 1], [1, 1]]), n_boot=500)
    assert check.outcome(split, "point") == (True, [])
    passed, lines = check.outcome(split, "interval")
    assert not passed and len(lines) == 1
    assert lines[0].startswith("FAIL apartment_a vacuum/1 object_where separation: memory_table beats blind by "
                               "+0.500 (95% interval")
    assert lines[0].endswith("The point estimate holds; the interval does not.")


def reach_rates(hits: dict | None = None, **rates: float) -> dict:
    """What `suite_report.group_numbers` holds of the one family of `hand_rows`: each prior system's hit rate (`hits`,
    all 0.5 by default) and reach rate (`rates` over a default of 0.5)."""
    return {("apartment_a", "vacuum/1"): {"families": {"object_where": {
        "hit_rates": {"blind": 0.5, "path_log": 0.5, "current_view_prior": 0.5, **(hits or {})},
        "reach_rates": {"blind": 0.5, "path_log": 0.5, "current_view_prior": 0.5, **rates}}}}}


def test_a_broken_instrument_fails_the_verdict_and_is_named():
    """Contract (VISION rule 5, review F3 and F4): a gate that a broken shortcut passes proves nothing, so the verdict
    fails and says why. Four scripts of two questions, the second unknown: all_unknown scores 0.5 and every gate holds.
    A shortcut that learns from other scripts (`blind`) at 0.40 is 0.10 under it, on the margin and allowed; at 0.39 it
    is 0.11 under and is named, the verdict failing on both readings though no gate does. `other_case` at -1 is not
    named: it answers from another stream, and falling under abstaining is what a wrong stream does. A prior system
    none of whose levels above the template finds a record of any question (reach rate 0) is named though its mean is
    fine, `path_log` and `current_view_prior` each in its own line, and so is a score far under the floor. A family
    that `path_log` is the reference of (every row's `reference`) leaves it out of the shortcut gate, and it is not
    judged there.
    Regression caught: a gate passed by a prior that overfits or never matches, or a stream-fed control taken for a
    broken prior."""
    def judged(blind, hits=None, **kw):
        report = gates.evaluate(hand_rows(blind, **kw), n_boot=500)
        return report, suite_report.broken_instruments(report, hits or reach_rates())

    near, lines = judged([[0.4, 0.4]] * 4)
    assert lines == [] and check.outcome(near, "point") == (True, [])
    under, lines = judged([[0.39, 0.39]] * 4)
    assert lines == ["BROKEN apartment_a vacuum/1 object_where: blind scores 0.390 (95% interval 0.390 to 0.390), "
                     "0.110 under all_unknown (0.500); a shortcut that learns from other scripts and does worse than "
                     "abstaining overfits, and the gate it passes proves nothing."]
    assert check.outcome(under, "point") == (True, [])
    assert check.outcome(under, "point", lines) == (False, lines)
    assert check.outcome(under, "interval", lines) == (False, lines)
    assert judged([[0.4, 0.4]] * 4, other_case=[[-1, -1]] * 4)[1] == []
    never = judged([[0.4, 0.4]] * 4, reach_rates(path_log=0.0, current_view_prior=0.0))[1]
    assert never == [f"BROKEN apartment_a vacuum/1 object_where: no level of {name} above the template finds a "
                     "record of any question (reach rate 0); its gate is vacuous."
                     for name in ("path_log", "current_view_prior")]
    own = judged([[0.4, 0.4]] * 4, reach_rates(path_log=0.0), reference="path_log", path_log=[[-1, -1]] * 4)[1]
    assert own == []
    score_line, hit_line = judged([[0.4, 0.4]] * 4, reach_rates(path_log=0.0), path_log=[[-1, -1]] * 4)[1]
    assert "path_log scores -1.000 (95% interval -1.000 to -1.000), 1.500 under all_unknown (0.500)" in score_line
    assert "no level of path_log above the template finds a record" in hit_line


def test_a_prior_that_pools_nothing_on_a_family_is_not_broken_there():
    """Contract (final review, nit): a reach rate of 0 is a broken instrument only where the prior pools; `path_log`
    answers a structure family (`clear_path`) from its own log alone, so its 0 there is the design, while the same 0
    on another family, or `blind`'s on the structure family, is named. Hand case: the one family of `hand_rows`
    relabelled `clear_path`, every score as in the passing case. Regression caught: the rule naming `path_log` on the
    structure family for levels it does not have, or no longer naming a prior that does pool."""
    report = gates.evaluate(hand_rows([[0.4, 0.4]] * 4), n_boot=500)
    relabelled = {**report, "rows": [{**r, "family": "clear_path"} for r in report["rows"]]}

    def numbers(**rates: float) -> dict:
        group = reach_rates(**rates)["apartment_a", "vacuum/1"]["families"]["object_where"]
        return {("apartment_a", "vacuum/1"): {"families": {"clear_path": group}}}

    assert suite_report.broken_instruments(relabelled, numbers(path_log=0.0)) == []
    (line,) = suite_report.broken_instruments(relabelled, numbers(blind=0.0))
    assert "no level of blind above the template" in line


def test_the_broken_rule_judges_the_held_out_rule_alone_and_not_current_view_prior():
    """Contract (VISION rule 5; DECISIONS 2026-10-08): `BROKEN` says a shortcut that learns from other scripts overfits
    when it scores far under abstaining under the held-out rule. The in-pool twin of a learned shortcut is the rule
    that overfits by design, so it is not judged: `blind` at 0.40 (on the margin) with its twin at 0.00, 0.50 under
    all_unknown's 0.50, gives no line, nor does a `current_view_prior` at -1, which no gate lists among its shortcuts
    (`memory_needed` reads it, but it is not a reading VISION rules 4-5 require: the current view alone is, so its
    deficit does not make the required gate vacuous). The learned shortcuts are `blind`, `path_log`,
    `questionnaire_blind` and the form adversary; `other_case` is not one. Regression caught: a docstring that lists a
    system the code never judges, or the twin's overfit read as a broken instrument."""
    report = gates.evaluate(hand_rows([[0.4, 0.4]] * 4, blind_in_pool=[[0, 0]] * 4,
                                      current_view_prior=[[-1, -1]] * 4), n_boot=500)
    assert suite_report.broken_instruments(report, reach_rates()) == []
    assert suite_report.learned_shortcuts() == {"blind", "path_log", "questionnaire_blind", "form"}
    under = gates.evaluate(hand_rows([[0.39, 0.39]] * 4, blind_in_pool=[[0.6, 0.6]] * 4), n_boot=500)
    (line,) = suite_report.broken_instruments(under, reach_rates())
    assert line.startswith("BROKEN apartment_a vacuum/1 object_where: blind scores 0.390")   # the twin, at 0.6, is not


def test_the_broken_rule_reads_every_level(tmp_path):
    """Contract (the review's M4, VISION rule 5): the broken-instrument rule flags a prior none of whose levels above
    the bare template finds a record (the vacuous gate it exists for), not one whose exact key misses while a looser
    level hits. Hand cases, four scripts each asked one mug at one moment, each in another colour: `blind`'s and
    `path_log`'s first keys (all the slots) recur in no other script, so their hit rate is 0, but the key without the
    colour recurs in all four, so their reach rate is 1: no BROKEN line, the hit rate still in the report. Four scripts
    each asked about another kind: no level above the template is shared by `blind` (reach 0), BROKEN, though
    `path_log`'s rooms-only level, which the four share, still reaches. Regression caught: the check failing a sparse
    exact key (every pair family's, on its first key), or passing a vacuous prior."""
    def numbers(cases) -> dict:
        runs = {(NM, c.id): run_of({qid(1): number(1.4), qid(2): at(3.2, 1.5)}) for c in cases}
        hits, reach = suite_report.hit_rates(cases, runs), suite_report.reach_rates(cases, runs)
        return {("apartment_a", "vacuum/1"): {"families": {"object_where": {
            "hit_rates": hits["object_where"], "reach_rates": reach["object_where"]}}}}

    colours = [hand_case(tmp_path / "colours", hand_synth(n), color=c)
               for n, c in enumerate(("red", "blue", "green", "white"), 1)]
    report = gates.evaluate(hand_rows([[0.4, 0.4]] * 4), n_boot=500)
    sparse = numbers(colours)
    family = sparse["apartment_a", "vacuum/1"]["families"]["object_where"]
    assert (family["hit_rates"]["blind"], family["hit_rates"]["path_log"]) == (0.0, 0.0)
    assert (family["reach_rates"]["blind"], family["reach_rates"]["path_log"]) == (1.0, 1.0)
    assert suite_report.broken_instruments(report, sparse) == []

    kinds = []
    for n, kind in enumerate(("mug", "bowl", "glass", "plate"), 1):
        qs = [end_q(1, slots={"kind": kind, "attributes": {"color": "red"}, "event": EVENT})]
        kinds.append(scored_case(tmp_path / "kinds", hand_synth(n), qs))
    apart = numbers(kinds)
    reach = apart["apartment_a", "vacuum/1"]["families"]["object_where"]["reach_rates"]
    assert (reach["blind"], reach["path_log"]) == (0.0, 1.0)
    assert suite_report.broken_instruments(report, apart) == [
        "BROKEN apartment_a vacuum/1 object_where: no level of blind above the template finds a record of any "
        "question (reach rate 0); its gate is vacuous."]


def test_a_case_that_asks_nothing_is_set_aside_and_counted_per_group(tmp_path):
    """Contract (review F2 and quality finding 1): a compiled case whose `questions.jsonl` holds no row is not checked,
    and is counted: `compiled_cases` splits the cases that ask from those that do not, `yields` gives each (layout, job)
    its compiled cases and how many of them ask nothing, a group of such cases alone included. Hand dataset: two home
    cases that ask, one home case that asks nothing and a warehouse case that asks nothing and is the only one of its
    layout and job. Regression caught: one empty case refusing a whole dataset (`run_case` has no question to ask), or
    a group whose cases all ask nothing vanishing from the report."""
    asking = [hand_case(tmp_path, hand_synth(1)), hand_case(tmp_path, hand_synth(2))]
    warehouse = hand_synth(5, room=WAREHOUSE.rooms[0], stage=WAREHOUSE)
    silent = [empty_case(tmp_path), empty_case(tmp_path, warehouse, "00000000000000ef")]
    found_asking, found_silent = check.compiled_cases(tmp_path)
    assert [c.id for c in found_asking] == sorted(c.id for c in asking)
    assert [c.id for c in found_silent] == sorted(c.id for c in silent)
    assert check.yields(found_asking, found_silent) == [
        {"layout": "apartment_a", "job": "vacuum/1", "compiled": 3, "asked_nothing": 1},
        {"layout": "warehouse_a", "job": "amr/1", "compiled": 1, "asked_nothing": 1}]


class FailsAsks(Blind):
    """`blind`, whose answer to an `object_where` question is an exception."""

    def ask(self, when, now, questions):
        if any(q["template"] == "object_where" for q in questions):
            raise RuntimeError("no answerer")
        return super().ask(when, now, questions)


class FailsObserve(Blind):
    """`blind`, which cannot take a step."""

    def observe(self, steps):
        raise RuntimeError("no steps")


class FailsClose(Blind):
    """`blind`, which answers every question and then cannot close: the case is `ended` with nothing failed."""

    def close(self):
        raise RuntimeError("no close")


def test_a_scripted_system_that_failed_refuses_the_check(tmp_path, monkeypatch, capsys):
    """Contract (review F3): `check` exits 2, naming the system, the case and the failure, when a scripted run failed
    a question (the stream scores a system's own exception as a failed answer and carries on, `complete`) or ended
    early, and writes no report: these systems are the instrument, and a shortcut that fails scores at the floor, which
    a gate that wants it low passes. A `blind` that raises on `object_where` fails its asks (`ask error`, as many as it
    was asked); one that raises on a step ends the case; one that answers all and cannot close is `ended`
    with no question failed. `all_unknown`, run beside it, is not named. Regression caught: a shortcut gate passed by a
    crashing shortcut."""
    synthetic_dataset(tmp_path, 2)
    for cls, why in ((FailsAsks, r"blind on case 0{15}1: \d+ x ask error"),
                     (FailsObserve, r"blind on case 0{15}1: ended: observe: RuntimeError: no steps; "
                                    r"\d+ x case ended: observe"),
                     (FailsClose, r"blind on case 0{15}1: ended: close: RuntimeError: no close; blind on case 0{15}2: "
                                  r"ended: close")):
        monkeypatch.setattr(baselines, "BASELINES", {"all_unknown": Baseline(AllUnknown),
                                                     "blind": Baseline(cls, needs_prior=True)})
        assert cli.main(["check", "--data", str(tmp_path), "--workers", "1"]) == 2
        err = capsys.readouterr().err
        assert re.search(why, err), err
        assert "a scripted system failed" in err and "all_unknown on case" not in err
        assert not (tmp_path / "check_report.json").exists()


def test_a_refusal_for_failed_runs_names_five_and_counts_the_rest(tmp_path, monkeypatch, capsys):
    """Contract (review F3): the refusal for failed scripted runs names the first five (system, case) runs and counts
    the others, so that a crashing system over twenty cases is one readable line. Two crashing systems and their two
    in-pool twins (`baselines.in_pool_twins`, run beside them) over three cases are twelve runs: five named, `(and 7
    more)`. Regression caught: a refusal that prints a line a run (260 for the thirteen systems over twenty cases)."""
    synthetic_dataset(tmp_path, 3)
    monkeypatch.setattr(baselines, "BASELINES", {"blind": Baseline(FailsAsks, needs_prior=True),
                                                 "questionnaire_blind": Baseline(FailsAsks, needs_prior=True)})
    assert cli.main(["check", "--data", str(tmp_path), "--workers", "1"]) == 2
    err = capsys.readouterr().err
    assert len(re.findall(r" on case 0{15}\d: ", err)) == 5 and err.rstrip().endswith("(and 7 more)")


def test_a_dataset_that_cannot_be_checked_is_refused_before_anything_runs(tmp_path, monkeypatch, capsys):
    """Contract (plan §4.8): `check` exits 2, naming why, for a dataset with no case that has questions and gold (before
    any system is run), and for a group with no other script's case to feed `other_case`. Regression caught: an empty
    dataset that passes with no group, or a lone case that is checked against nothing."""
    def ran(*args, **kwargs):
        raise AssertionError("a system ran")

    with monkeypatch.context() as patched:
        patched.setattr(suite, "run_suite", ran)
        assert cli.main(["check", "--data", str(tmp_path / "none")]) == 2
        assert "no case with questions and gold" in capsys.readouterr().err
    hand_case(tmp_path / "lone", hand_synth(2))
    assert cli.main(["check", "--data", str(tmp_path / "lone"), "--workers", "1"]) == 2
    assert "no case of another script on layout apartment_a" in capsys.readouterr().err


def test_a_released_dataset_keeps_its_report_unless_out_is_given(tmp_path, capsys):
    """Contract (docs/format.md, the release): a dataset root that holds `release.json`, the document `export` writes,
    is a released dataset, and `check` over it leaves its `check_report.json` as released: it writes no report there,
    says so, and still prints its numbers (two cases of two scripts); given `--out FILE` it writes that file.
    Regression caught: a check that overwrites the released report, or that no longer finds the marker and writes
    into a released dataset's folder."""
    synthetic_dataset(tmp_path, 2)
    (tmp_path / "release.json").write_text("{}\n", encoding="utf-8")
    assert cli.main(["check", "--data", str(tmp_path), "--workers", "1"]) != 2
    captured = capsys.readouterr()
    assert not (tmp_path / "check_report.json").exists()
    assert "is a released dataset (it has a release.json)" in captured.err
    assert "apartment_a vacuum/1: 2 cases of 2 scripts" in captured.out
    elsewhere = tmp_path.parent / f"{tmp_path.name}-report.json"
    assert cli.main(["check", "--data", str(tmp_path), "--workers", "1", "--out", str(elsewhere)]) != 2
    assert json.loads(elsewhere.read_text(encoding="utf-8"))["groups"] and not (tmp_path / "check_report.json").exists()


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """The synthetic dataset and a case of it that asks nothing, checked once with `--verdict point`: (its folder, the
    exit code, the report, what the command printed)."""
    root = tmp_path_factory.mktemp("synthetic")
    synthetic_dataset(root, 6)
    empty_case(root)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        code = cli.main(["check", "--data", str(root), "--verdict", "point", "--workers", "4"])
    return root, code, json.loads((root / "check_report.json").read_text(encoding="utf-8")), printed.getvalue()


def test_check_on_a_synthetic_dataset(synthetic):
    """Contract (plan §4.8, the exit check): `stbench check --verdict point` over compiled synthetic scripts writes
    `check_report.json` under the dataset, whose gates cover each (layout, job, family) with the oracle passing
    `ceiling` on both families and each prior system's first-level hit rate above 0 (their worlds alike, every key
    finds its records), which carries no `versions`, and whose exit code follows the verdict it names. The worlds
    are alike, so every prior memorises them and the gates fail: exit 1, which is the check able to fail. `path_ahead`
    is read by the mapper and the gold through one `grid` function, so its `knowable` gate is `n/a`, and
    `object_where`'s is not. Regression caught: the command unwired, a family the oracle cannot answer, a prior whose
    key never finds a record (its gate would pass for nothing), or a gate that cannot fail."""
    root, code, report, _ = synthetic
    assert report["verdict"] == "point" and "versions" not in report
    assert code == 1 and report["passed"] is False and report["passed"] == report["passed_point"]
    assert {(g["layout"], g["job"], g["family"]): g["gates"]["ceiling"]["pass"] for g in report["gates"]} == {
        ("apartment_a", "vacuum/1", "path_ahead"): True, ("apartment_a", "vacuum/1", "object_where"): True}
    assert {g["family"]: g["gates"]["knowable"]["pass"] for g in report["gates"]}["path_ahead"] is None
    assert {g["family"]: g["gates"]["knowable"]["pass"] for g in report["gates"]}["object_where"] is not None
    (group,) = report["groups"]
    assert group["cases"] == 6 and set(group["form_adversary"]) == {"path_ahead", "object_where"}
    for family in ("path_ahead", "object_where"):
        hits = group["families"][family]["hit_rates"]
        assert set(hits) == {"blind", "path_log", "current_view_prior"} and min(hits.values()) > 0
    assert group["families"]["path_ahead"]["current_view_now_mean"] == 1.0
    assert {*BASELINES, *baselines.in_pool_twins(), gates.FORM, gates.in_pool_name(gates.FORM)} <= {
        name for g in report["gates"] for name in g["scores"]}   # every system, and the twin of each learned one


def test_check_skips_a_case_that_asks_nothing_and_says_so(synthetic):
    """Contract (review F2): a dataset with a compiled case that asks nothing is checked as the rest of it, the group's
    case count its six that ask, and the report and the printout say that one of seven compiled cases asked nothing.
    Regression caught: the whole check refused (exit 2, no report) for one window that drew no question, or an empty
    case counted among the checked ones or not counted at all."""
    root, code, report, printed = synthetic
    assert code != 2   # checked, not refused
    assert report["asked_nothing"] == [{"layout": "apartment_a", "job": "vacuum/1", "compiled": 7, "asked_nothing": 1}]
    assert report["groups"][0]["cases"] == 6
    assert "apartment_a vacuum/1: 1 of 7 compiled cases ask nothing and are not checked" in printed
