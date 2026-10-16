"""`stbench check --select/--cases` (a scoped check), `--answers` and `--reuse` (`check`, `check_answers`,
`case_select`, `Baseline.reads_scorer`, 0.11 lane C): a change is re-checked in minutes, and the quick check says the
same as the full one. The dataset is two groups (a vacuum of three cases on `apartment_a` and the same vacuum of two on
`house_floor`, both from `check_kit`'s synthetic scripts), small enough that every check below runs in seconds."""

from __future__ import annotations

import contextlib
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest

from baseline_kit import compiled_case
from check_kit import synthetic_dataset
from space_kit import things_run
from stbench import baselines, cli, suite, suite_priors
from stbench.stream_protocol import GATE_CHANNEL
from synth import HOME

APARTMENT = ("apartment_a", "vacuum/1")
HOUSE = ("house_floor", "vacuum/1")
# the systems whose answers never pass through the scorer, by hand (the rest choose a best response by it)
SCORER_FREE = ["current_view", "oracle", "memory_table", "other_case", "all_unknown", "kind_range", "latest_table",
               "first_table"]


def run_check(root: Path, *argv: str) -> tuple[int, str, str]:
    """`stbench check --data root --verdict point --workers 1 ...argv`: (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(["check", "--data", str(root), "--verdict", "point", "--workers", "1", *argv])
    return code, out.getvalue(), err.getvalue()


def report_of(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verdicts(report: dict) -> dict:
    """What a check says, without the time it took or how it was scoped."""
    return {key: report[key] for key in ("verdict", "passed", "passed_point", "passed_interval", "passed_evidence",
                                         "broken", "balance", "asked_nothing", "groups", "gates")}


def of_group(report: dict, group: tuple[str, str]) -> dict:
    """The report's numbers for one (layout, job): its gate rows and its group numbers."""
    layout, job = group
    return {"gates": [g for g in report["gates"] if (g["layout"], g["job"]) == group],
            "groups": [g for g in report["groups"] if (g["layout"], g["job"]) == group],
            "asked_nothing": [y for y in report["asked_nothing"] if (y["layout"], y["job"]) == group],
            "balance": [b for b in report["balance"] if b["layout"] == layout and b["job"] == job]}


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    """The two-group dataset (3 + 2 cases), checked whole once, its answers stored: (the folder, its report, the folder
    of answers, the ids of the cases of each group)."""
    root = tmp_path_factory.mktemp("two_groups")
    apartment = synthetic_dataset(root, 3)
    house = [compiled_case(root, f"{i:016x}", things_run(seed=i, stage=replace(HOME, map_name="house_floor")))
             for i in (4, 5)]
    answers = root / "answers"
    code, _, _ = run_check(root, "--out", str(root / "full.json"), "--answers", str(answers))
    assert code in (0, 1)   # checked, not refused
    return root, report_of(root / "full.json"), answers, {APARTMENT: [c.id for c in apartment],
                                                          HOUSE: [c.id for c in house]}


def test_a_scoped_check_gives_each_group_the_numbers_of_the_full_check(full):
    """Contract (0.11 lane C, design §3.9): `--select layout=house_floor` checks that group alone, and a system's
    options look only at the cases of its own layout and job, so the group's gate rows, numbers, balance entries and
    yield are exactly the full check's for it, and the other group is not in the report at all; run twice it says the
    same. The report holds its scope, by hand: one of two groups, two of five cases, the selection in words. Regression
    caught: a scoped check that pools the other group's scripts into the priors (or drops a group's rows), so that a
    quick check could pass what the full one fails."""
    root, whole, _, _ = full
    first = run_check(root, "--select", "layout=house_floor", "--out", str(root / "a.json"))
    a = report_of(root / "a.json")
    second = run_check(root, "--select", "layout=house_floor", "--out", str(root / "a.json"))
    b = report_of(root / "a.json")
    assert verdicts(a) == verdicts(b) and first[1:] == second[1:]   # twice: the same report and the same words
    assert {g["layout"] for g in a["gates"]} == {"house_floor"}
    assert {k: v for k, v in verdicts(a).items() if k in ("gates", "groups", "asked_nothing", "balance")} == of_group(
        whole, HOUSE)
    assert a["scope"] == {"select": "layout=house_floor", "groups": [{"layout": "house_floor", "job": "vacuum/1"}],
                          "of_groups": 2, "cases": 2, "of_cases": 5}


def test_a_scoped_check_says_what_it_was_scoped_to_and_never_writes_the_datasets_report(full):
    """Contract (0.11 lane C): a scoped check begins and ends its printout with a `SCOPED CHECK` line saying the
    selection and how much of the dataset it read (by hand: `layout=house_floor`, 1 of 2 groups, 2 of 5 cases), and,
    given no `--out`, writes no report: `DATASET/check_report.json` is the whole dataset's (`stbench export` reads it),
    and says so on stderr. Regression caught: a scoped report overwriting the full one, or a scoped pass read as the
    release check's."""
    root, *_ = full
    assert not (root / "check_report.json").exists()
    code, out, err = run_check(root, "--select", "layout=house_floor")
    lines = out.splitlines()
    said = ("SCOPED CHECK (layout=house_floor): 1 of 2 groups and 2 of 5 cases that ask were checked; the verdict "
            "reads those and is not the release check's")
    assert code in (0, 1) and lines[0] == said and lines[-1] == said
    assert "a scoped check leaves" in err and not (root / "check_report.json").exists()


def test_cases_and_select_narrow_together_and_a_bad_scope_is_refused(full):
    """Contract (0.11 lane C): `--cases` takes case ids and, with `--select`, the intersection, as `stbench eval` does
    (`case_select`): the two cases of the house group by id check exactly as `--select layout=house_floor` did; a
    selection by length or by an id narrows the group itself, so the one case of a group left has no other script's
    case to feed `other_case` and the check is refused, as a lone group is; a key that is none, an id the dataset lacks
    and a selection that matches nothing are refused (exit 2) before any system runs, saying why. Regression caught: a
    scope that quietly checks everything, or a bad `--select` that runs the whole dataset."""
    root, whole, _, ids = full
    by_id = run_check(root, "--cases", ",".join(ids[HOUSE]), "--out", str(root / "ids.json"))
    assert by_id[0] in (0, 1) and verdicts(report_of(root / "ids.json"))["gates"] == of_group(whole, HOUSE)["gates"]
    both = run_check(root, "--cases", ",".join(ids[HOUSE]), "--select", "layout=apartment_a")
    assert both[0] == 2 and "no case of the 5 compiled ones matches --cases and --select" in both[2]
    one = run_check(root, "--cases", ids[HOUSE][0], "--out", str(root / "one.json"))
    assert one[0] == 2 and "no case of another script on layout house_floor" in one[2]
    for argv, why in ((["--select", "colour=red"], "--select takes KEY=VALUE"),
                      (["--select", "length=long"], "no case of the 5 compiled ones matches"),
                      (["--cases", "00000000000000ff"], "--cases names 00000000000000ff")):
        code, out, err = run_check(root, *argv)
        assert (code, out) == (2, "") and why in err


def test_a_reused_check_gives_the_verdicts_of_a_fresh_one(full, monkeypatch):
    """Contract (0.11 lane C, design §3.9): `--reuse DIR` reloads the stored answers of the systems that do not read the
    scorer and runs the rest, and the report says the same as the fresh check that stored them: same gates, groups,
    balance and verdict flags, the same exit code, twice over. The systems it did not run are exactly the scorer-free
    ones (by hand, `SCORER_FREE`), seen from `run_suite`, which is never given them; the report names them. Regression
    caught: a reused check that scores another answer than the fresh one (a stale or mixed-up file), runs everything
    anyway (no speed-up), or reloads a learned shortcut whose answers a scorer change moves."""
    root, whole, answers, _ = full
    named = []
    real = suite.run_suite

    def spy(cases, names, channel, **kwargs):
        named.append(list(names))
        return real(cases, names, channel, **kwargs)

    monkeypatch.setattr(suite, "run_suite", spy)
    code, out, _ = run_check(root, "--reuse", str(answers), "--out", str(root / "reused.json"))
    reused = report_of(root / "reused.json")
    again = run_check(root, "--reuse", str(answers), "--out", str(root / "reused.json"))
    twice = report_of(root / "reused.json")
    assert verdicts(reused) == verdicts(whole) == verdicts(twice) and (code, out) == again[:2]
    assert sorted(reused["reused"]) == sorted(SCORER_FREE)
    assert set(named[0]) == {"current_view_prior", "path_log", "blind", "questionnaire_blind", "map_reader",
                             *baselines.in_pool_twins()}
    assert f"REUSED the stored answers of {', '.join(reused['reused'])}" in out
    assert "reused" not in whole and "scope" not in whole   # an unscoped, fresh check reports as it always did


def test_scoped_checks_write_apart_and_their_answers_add_up_to_a_full_reuse(full, tmp_path):
    """Contract (0.11 lane C): one scoped check a job can run side by side. Each writes only its own `--out` and its own
    cases' answer files (`DIR/<system>/<case>.json`; a scoped check never writes the dataset's `check_report.json`), so
    two of them into one `--answers` folder collide in nothing, and the folder then holds every case: a full `--reuse`
    of it reloads all the scorer-free systems (by hand, `SCORER_FREE`) and says what the full check said. Regression
    caught: scoped runs sharing a report path or an answers file, or an answers folder that a per-group run leaves
    unusable for the whole dataset (a group recorded as only the cases of that run)."""
    root, whole, _, ids = full
    shared = tmp_path / "answers"
    for layout in ("apartment_a", "house_floor"):
        code, *_ = run_check(root, "--select", f"layout={layout}", "--answers", str(shared),
                             "--out", str(root / f"{layout}.json"))
        assert code in (0, 1)
    assert not (root / "check_report.json").exists()
    assert sorted(p.name for p in (shared / "oracle").iterdir()) == sorted(f"{c}.json" for g in ids.values() for c in g)
    run_check(root, "--reuse", str(shared), "--out", str(root / "joined.json"))
    joined = report_of(root / "joined.json")
    assert sorted(joined["reused"]) == sorted(SCORER_FREE) and verdicts(joined) == verdicts(whole)


def test_a_group_narrowed_since_the_answers_were_stored_reruns_the_systems_that_read_other_cases(full):
    """Contract (0.11 lane C): `other_case`'s partner and every prior come from the other cases of the group, so their
    stored answers hold only for the group they were stored with; a check narrowed by case id (the first two of the
    apartment's three) reloads the systems that read nothing but their own case (by hand: `current_view`, `oracle`,
    `memory_table`, `all_unknown`, `latest_table`, `first_table`), runs `other_case` and `kind_range`, and says what a
    fresh check of the same two cases says. Regression caught: a reused narrowed check scoring the three-case group's
    `other_case` answers against two cases' golds, a gate passed or failed by a partner that is no longer there."""
    root, _, answers, ids = full
    two = ",".join(ids[APARTMENT][:2])
    run_check(root, "--cases", two, "--out", str(root / "two_fresh.json"))
    run_check(root, "--cases", two, "--reuse", str(answers), "--out", str(root / "two_reused.json"))
    fresh, reused = report_of(root / "two_fresh.json"), report_of(root / "two_reused.json")
    assert sorted(reused["reused"]) == sorted(["current_view", "oracle", "memory_table", "all_unknown", "latest_table",
                                               "first_table"])
    assert verdicts(reused) == verdicts(fresh) and "reused" not in fresh


def test_stored_answers_that_are_missing_or_of_another_compile_are_run_not_trusted(full, tmp_path):
    """Contract (0.11 lane C): a system is reloaded only when every case of the check has a whole file holding an answer
    to exactly the case's questions by qid (a recompile redraws the qids); otherwise it is run, and the check is right
    all the same. By hand, from a copy of the stored answers: the oracle's file for one case is deleted, the table's
    holds one answer fewer, `current_view`'s one qid it never was asked, and `all_unknown`'s is not JSON; those four run
    and the other four scorer-free systems are reloaded. Regression caught: a stale cache scored as answers (a golds
    row `unanswered`, a verdict that moves with the compile), or a corrupt file that stops the check."""
    root, whole, answers, ids = full
    mine = tmp_path / "answers"
    mine.mkdir()
    for folder in answers.iterdir():
        (mine / folder.name).mkdir()
        for file in folder.iterdir():
            (mine / folder.name / file.name).write_bytes(file.read_bytes())
    case = ids[APARTMENT][0]
    (mine / "oracle" / f"{case}.json").unlink()
    for system, change in (("memory_table", lambda d: d["answers"].pop(next(iter(d["answers"])))),
                           ("current_view", lambda d: d["answers"].update({"q-never-asked": {"type": "unknown"}}))):
        file = mine / system / f"{case}.json"
        doc = json.loads(file.read_text(encoding="utf-8"))
        change(doc)
        file.write_text(json.dumps(doc), encoding="utf-8")
    (mine / "all_unknown" / f"{case}.json").write_text("{not json", encoding="utf-8")
    run_check(root, "--reuse", str(mine), "--out", str(root / "stale.json"))
    stale = report_of(root / "stale.json")
    assert sorted(stale["reused"]) == sorted(["other_case", "kind_range", "latest_table", "first_table"])
    assert verdicts(stale) == verdicts(whole)


def test_a_system_that_does_not_read_the_scorer_answers_alike_under_another_one(tmp_path, monkeypatch):
    """Contract (0.11 lane C): `Baseline.reads_scorer` false is a promise that the system's answers do not move with the
    scorer, which `--reuse` stakes the verdict on. Every system that does not declare it answers the synthetic cases
    identically when every `penalised` the baselines import is replaced by its negation, and the sabotage is not idle:
    the in-pool twins of `path_log` and `current_view_prior` (four alike cases give them a pool) answer differently.
    The twins of the learned shortcuts inherit the flag, so they run. Regression caught: a system that starts choosing
    by the scorer while still flagged free, whose stale answers `--reuse` would then score as the new ones."""
    from stbench import baselines_path_ego, baselines_path_own, baselines_path_task, blind, score

    cases = synthetic_dataset(tmp_path, 4)
    learned = ["path_log_in_pool", "current_view_prior_in_pool"]
    names = [*SCORER_FREE, *learned]
    assert sorted(n for n, b in baselines.BASELINES.items() if not b.reads_scorer) == sorted(SCORER_FREE)
    assert all(baselines.lookup(twin).reads_scorer for twin in baselines.in_pool_twins())

    def answers():
        waves = [[baselines.NM_SYSTEM], [n for n in names if n != baselines.NM_SYSTEM]]
        runs = suite.run_suite(cases, names, GATE_CHANNEL, options=suite_priors.options_for, waves=waves)
        return {name: [r.answers for r in rs] for name, rs in runs.items()}

    before, real = answers(), score.penalised
    for module in (score, blind, baselines_path_ego, baselines_path_own, baselines_path_task):
        monkeypatch.setattr(module, "penalised", lambda answer, gold: -real(answer, gold))
    after = answers()
    assert {n: before[n] == after[n] for n in SCORER_FREE} == dict.fromkeys(SCORER_FREE, True)
    assert [before[n] != after[n] for n in learned] == [True, True]
