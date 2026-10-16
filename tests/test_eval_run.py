"""`stbench eval` over baselines and fake agents (`eval_run.py`, `eval_results.py`, plan §5.1's B3): aggregation, the
selection, resume, the refusals, the budget and the resource fields. The dataset is `eval_kit.hand_dataset`: two cases
of two scripts asked a during question (`path_ahead`, golds answerable) and two end ones (`object_where`), whose golds
are [answerable, answerable, unknown] in the first case and [answerable, unknown, unknown] in the second. Under
`all_unknown` an answerable gold scores 0 and an unknown one 1; the expected numbers are worked out from that."""

from __future__ import annotations

import json
import shutil

import pytest

from eval_kit import (
    RUNS,
    Counting,
    case_files,
    cli,
    compiled,
    hand_dataset,
    hand_script,
    register,
    register_agent,
    results,
    short_dataset,
)
from recompile_kit import recompile
from stbench import case as case_mod
from stbench import eval_results, score_stats, suite_priors
from stream_kit import QUESTIONS, K, end_q
from synth import Synth


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("eval")
    hand_dataset(root)
    return root


def test_two_cases_aggregate_by_hand(data, tmp_path):
    """Contract (plan §5.1): `eval` scores every case and reports the headline (the mean of the `now` and `memory`
    columns, each pooled over its own questions), with the splits beside it, in `results.json` and `results.md`.
    `all_unknown` scores 1 on the three unknown golds and 0 on the three answerable ones: the headline is
    (0 + 3/4)/2 = 3/8 (pooled it would be 3/6); `object_where` (four questions, three of
    them unknown) 3/4 and `path_ahead` (two, answerable) 0, which are also the `memory` and `now` splits and the `end`
    and `during` ones; all six answers are `unknown`: three correct, three abstained, precision 3/6 and recall 3/3. The
    `slice` split is answerable 0 and unknown 1, the `relevance` split situational 0 and unknown 1 (no gold is tagged
    critical), and the score above the floor 3/8 - 3/8 = 0 (the unknown share is (0 + 3/4)/2, weighted as the headline
    is): answering `unknown` throughout is the floor. Regression caught: rows lost or pooled wrong, a split that drifts
    from the rows, counts that disagree with the rows, or a headline that is read without the unknown share under it."""
    assert cli("all_unknown", "--data", data, "--out", tmp_path) == 0
    doc = results(tmp_path)
    agg = doc["layouts"]["apartment_a"]
    assert list(doc["layouts"]) == ["apartment_a"]
    assert (agg["n"], agg["scripts"], agg["cases"]) == (6, 2, 2) and agg["headline"]["mean"] == pytest.approx(0.375)
    assert {f: s["mean"] for f, s in agg["by"]["family"].items()} == pytest.approx({"object_where": 0.75,
                                                                                    "path_ahead": 0.0})
    assert {k: s["mean"] for dim in ("scope", "when") for k, s in agg["by"][dim].items()} == pytest.approx(
        {"now": 0.0, "memory": 0.75, "during": 0.0, "end": 0.75})
    assert {k: s["mean"] for k, s in agg["by"]["slice"].items()} == pytest.approx({"answerable": 0.0, "unknown": 1.0})
    assert {k: (s["n"], s["mean"]) for k, s in agg["by"]["relevance"].items()} == pytest.approx(
        {"situational": (3, 0.0), "unknown": (3, 1.0)})   # no gold of the hand dataset is tagged critical
    assert agg["above_floor"]["mean"] == pytest.approx(0.0)   # 3/8 less the unknown share 3/8: no skill above the floor
    assert agg["counts"] == {"correct": 3, "partial": 0, "wrong": 0, "abstained": 3}
    assert agg["abstention"]["answered_unknown"] == 6 and agg["abstention"]["unknown_gold"] == 3
    assert (agg["abstention"]["precision"], agg["abstention"]["recall"]) == (0.5, 1.0)
    assert agg["confidently_wrong_rate"]["rate"] == 0.0
    assert [(c["status"], c["layout"], c["length"]) for c in doc["cases"]] == [("complete", "apartment_a", "short")] * 2
    assert len(doc["rows"]) == 6 and doc["partial"] is False and doc["verified"]
    assert doc["notes"] == [eval_results.RELAXED_NOTE]   # six questions: the dataset is too small for the under-20 rule
    assert (doc["system"], doc["sandbox"], doc["brief"], doc["during"], doc["reasoning"]) == (
        "all_unknown", "in-process", "correct", True, None)
    md = (tmp_path / "results.md").read_text(encoding="utf-8")
    for line in ("## Layout apartment_a: 2 of 2 cases finished", "| headline | 6 | +0.375 |", "| now | 2 | +0.000 |",
                 "| memory | 4 | +0.750 |", "| object_where | 4 | +0.750 |", "| path_ahead | 2 | +0.000 |",
                 "correct 3, partial 0, wrong 0, abstained 3"):
        assert line in md


def test_resume_runs_nothing_twice(data, tmp_path, monkeypatch):
    """Contract (plan §5.1): a case is finished when its summary says `complete`; the same command run again runs
    nothing and writes the same results; a deleted `summary.json` reruns that case alone, and a case whose summary says
    it ended is run again from its start, with the folder it left removed (a Codex case cannot be replayed, and its
    stale journal would refuse the rerun). The fake agent notes each case it runs: two, still two after a rerun, three
    after one case reruns and four after the second. A baseline runs the whole selection (its priors need every case)
    but writes only the unfinished cases' folders. Regression caught: a rerun that repeats finished (paid) cases, or
    one that trusts a folder it should have discarded."""
    register_agent(monkeypatch)
    out = tmp_path / "run"
    assert cli("fake:ok", "--data", data, "--out", out) == 0
    first = results(out)
    assert len(RUNS) == 2
    assert cli("fake:ok", "--data", data, "--out", out) == 0
    assert len(RUNS) == 2 and results(out) == first
    victim, other = (out / "cases" / c["id"] for c in first["cases"])
    (victim / "summary.json").unlink()
    assert cli("fake:ok", "--data", data, "--out", out) == 0
    assert len(RUNS) == 3 and RUNS[-1]["case"] == victim.name
    ended = json.loads((other / "summary.json").read_text(encoding="utf-8"))
    (other / "summary.json").write_text(json.dumps({**ended, "status": "ended: observe: RuntimeError: x"}),
                                        encoding="utf-8")
    (other / "codex.calls.jsonl").write_text("{}", encoding="utf-8")
    assert cli("fake:ok", "--data", data, "--out", out) == 0
    assert len(RUNS) == 4 and RUNS[-1]["case"] == other.name and not (other / "codex.calls.jsonl").exists()
    again = results(out)
    assert again["rows"] == first["rows"] and [c["status"] for c in again["cases"]] == ["complete"] * 2

    register(monkeypatch)
    base = tmp_path / "base"
    assert cli("counting", "--data", data, "--out", base) == 0
    assert Counting.observed == 40   # two cases of 20 frames, one observe a frame
    kept, mine = case_files(base), base / "cases" / first["cases"][0]["id"]
    (mine / "summary.json").unlink()
    assert cli("counting", "--data", data, "--out", base) == 0
    assert Counting.observed == 80   # the whole selection runs, as its priors need every case
    now = case_files(base)
    changed = {name for name in kept if now.get(name) != kept[name]}
    assert changed == {f"cases/{mine.name}/summary.json"}   # only the unfinished case's folder was written


def test_another_run_or_another_compile_is_refused(data, tmp_path, monkeypatch, capsys):
    """Contract (plan §5.1): resuming into a folder that holds the results of another run (another system, channel,
    brief or during flag), or whose finished cases hold rows of qids the gold now on disk lacks (every qid is drawn
    anew by a compile, so the dataset was recompiled since: one case recompiled is enough), is refused (exit 2) and
    changes nothing, because two runs must never be mixed into one result. Regression caught: `counting` finishing a
    folder `all_unknown` started, a result silently relabelled with another brief, or answers of a dataset compiled
    twice reused under new qids."""
    register(monkeypatch)
    out = tmp_path / "run"
    assert cli("all_unknown", "--data", data, "--out", out) == 0
    before, stamp = case_files(out), (out / "results.json").read_bytes()
    capsys.readouterr()
    for what, argv in {"system": ["counting"], "channel": ["all_unknown", "--channel", "detections"],
                       "brief": ["all_unknown", "--brief", "neutral"],
                       "during": ["all_unknown", "--during", "off"]}.items():
        assert cli(argv[0], "--data", data, "--out", out, *argv[1:]) == 2, what
        assert f"{what}: there" in capsys.readouterr().err
    first, second = (c["id"] for c in results(out)["cases"])
    for what, chosen in {"every case": (), "one case": (second,)}.items():
        recompiled = tmp_path / f"recompiled-{len(chosen)}"
        shutil.copytree(data, recompiled)
        recompile(recompiled, *chosen)
        assert cli("all_unknown", "--data", recompiled, "--out", out) == 2, what
        shown = capsys.readouterr().err
        assert "scored against another compile of case(s)" in shown and second in shown, what
        assert (first in shown) == (not chosen), what
    assert case_files(out) == before and (out / "results.json").read_bytes() == stamp


def test_a_run_folder_of_an_earlier_build_resumes_and_loses_its_stamps(data, tmp_path):
    """Contract: a `results.json` and case `summary.json` files written before the stamps went (`schema`, `agent`,
    `versions`, `prices`, a case's `gold_sha256`; a summary's `versions` and `prices`) are accepted by the loader and
    the resume: nothing reads those keys, the finished cases are not run again, and the results.json rewritten from the
    folder equals the one without the stamps. This is how a finished run is re-aggregated with no model call.
    Regression caught: a loader or a resume that requires or compares a stamp that is gone, or a rewrite that keeps
    one."""
    out = tmp_path / "run"
    assert cli("all_unknown", "--data", data, "--out", out) == 0
    new = results(out)
    assert not {"schema", "agent", "versions", "prices"} & set(new)   # a document of this build carries none
    assert all("gold_sha256" not in case for case in new["cases"])
    old = {"schema": "old", "agent": None, "versions": {"compiler": "old", "scorer": "old"},
           "prices": "old", **new, "cases": [{**c, "gold_sha256": "0" * 64} for c in new["cases"]]}
    eval_results.write(out, old)
    for case in new["cases"]:
        path = out / "cases" / case["id"] / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**summary, "versions": {"stream": "old"}, "prices": "old"}),
                        encoding="utf-8")
    assert eval_results.load(out)["schema"] == "old"
    before = case_files(out)
    assert cli("all_unknown", "--data", data, "--out", out) == 0
    assert results(out) == new and case_files(out) == before   # re-aggregated, and no case was run again


def test_gold_of_an_earlier_build_with_a_compiler_key_scores_the_same(data, tmp_path):
    """Contract: a gold row that still carries the `compiler` key of an earlier build is accepted and the key read by
    nothing, so `eval` scores such a dataset exactly as it scores the same rows without the key. Regression caught: a
    reader that refuses a case, or sets it aside as stale, because its gold rows hold a stamp (or lack the present
    one)."""
    old = tmp_path / "old"
    shutil.copytree(data, old)
    for gold in (old / "private").glob("*/gold.jsonl"):
        rows = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines()]
        gold.write_text("".join(json.dumps({**row, "compiler": "old"}) + "\n" for row in rows), encoding="utf-8")
    assert cli("all_unknown", "--data", data, "--out", tmp_path / "now") == 0
    assert cli("all_unknown", "--data", old, "--out", tmp_path / "then") == 0
    then, now = results(tmp_path / "then"), results(tmp_path / "now")
    assert then["rows"] == now["rows"] != [] and then["layouts"] == now["layouts"]


def test_what_cannot_run_is_refused_first(data, tmp_path, capsys):
    """Contract (plan §5.1): the command refuses (exit 2, a message on stderr, no folder made) a system that is none of
    the three kinds, `--reasoning` on a baseline, `--local` on an agent, a channel a baseline does not run on, a spend
    cap on a baseline or without its reservation, a bad `--arms` or `--select`, an arm lever given as an option too, a
    swapped brief (retired: a job's subgoal says the true task in every frame; as `--brief` and as an arm), a case that
    is not there, an empty selection and a RUNDIR under the dataset's truth. Regression caught: a paid or long run that
    starts, or fails deep inside, on an option that was wrong from the start."""
    out = tmp_path / "out"
    base = ["--data", data, "--out", out]
    refused = {
        "SYSTEM must be a baseline": ["nonsense", *base],
        "--reasoning applies to an agent": ["all_unknown", *base, "--reasoning", "high"],
        "--local applies to a FILE.py:Class": ["codex:m", *base, "--local"],
        "runs on the detections channel only": ["memory_table", *base],
        "spends none": ["all_unknown", *base, "--max-usd", "1", "--case-usd", "1"],
        "go together": ["codex:m", *base, "--max-usd", "1"],
        "--arms takes LEVER=V1,V2": ["all_unknown", *base, "--arms", "during=on"],
        "--arms during takes on, off, not maybe": ["all_unknown", *base, "--arms", "during=on,maybe"],
        "do not give --during as well": ["all_unknown", *base, "--arms", "during=on,off", "--during", "off"],
        "--select takes KEY=VALUE": ["all_unknown", *base, "--select", "colour=red"],
        "the swapped brief is retired": ["all_unknown", *base, "--brief", "swapped"],
        "a subgoal says the true task in every frame": ["all_unknown", *base, "--arms", "brief=correct,swapped"],
        "--cases names": ["all_unknown", *base, "--cases", "ffffffffffffffff"],
        "no case of the 2 compiled ones matches": ["all_unknown", *base, "--select", "length=long"],
        "no case with questions and gold": ["all_unknown", "--data", tmp_path / "empty", "--out", out],
        "which a run never writes to": ["all_unknown", "--data", data, "--out", data / "private" / "x"],
        "--data DIR is required": ["all_unknown"],
        "unexpected arguments": ["all_unknown", "a.json", *base],
    }
    for message, argv in refused.items():
        assert cli(*argv) == 2, message
        assert message in capsys.readouterr().err
    assert not out.exists() and not (data / "private" / "x").exists()
    for flag, value in (("--workers", "0"), ("--max-usd", "0"), ("--case-usd", "nan"), ("--timeout", "0.1")):
        with pytest.raises(SystemExit) as bad:
            cli("all_unknown", *base, flag, value)
        assert bad.value.code == 2 and f"argument {flag}: {value!r} must be" in capsys.readouterr().err


def test_the_selection(tmp_path):
    """Contract (plan §5.1): the cases are the dataset's compiled ones that ask a question, narrowed by `--cases` and
    by `--select` on layout, robot, job and length: a key given twice is either value, different keys must all hold.
    Of a short case and a medium one on the apartment and a short one on the warehouse (its script's index entry says
    so), `length=short` picks the two short ones, both lengths all three, `layout=apartment_a` and `length=short` the
    first alone, and `--cases` with `--select` their intersection; a compiled case that asks nothing is not run.
    Regression caught: a selection that ignores a key, ORs two keys, or runs a case that asks nothing."""
    short = compiled(tmp_path, hand_script(1), unknown=(2,))
    two_tasks = Synth(tasks=[[(K, 8)], [(K, 8)]], seed=2)
    medium = compiled(tmp_path, two_tasks, [end_q(1), end_q(2)], b=1, length="medium")
    warehouse = compiled(tmp_path, hand_script(3))
    entry = case_mod.read_entry(tmp_path, warehouse.script.id)
    case_mod.write_entry(tmp_path, warehouse.script.id, {**entry, "map": "warehouse_a"})
    silent = compiled(tmp_path, hand_script(4), [])   # asks nothing: never run
    assert silent.questions.read_text(encoding="utf-8") == ""

    runs = []

    def picked(*argv) -> list[str]:
        runs.append(tmp_path / "runs" / str(len(runs)))
        assert cli("all_unknown", "--data", tmp_path, "--out", runs[-1], *argv) == 0
        return [c["id"] for c in results(runs[-1])["cases"]]

    every = sorted([short.id, medium.id, warehouse.id])
    assert picked() == every
    assert picked("--select", "length=short") == sorted([short.id, warehouse.id])
    assert picked("--select", "length=short", "--select", "length=medium") == every
    assert picked("--select", "layout=apartment_a", "--select", "length=short") == [short.id]
    assert picked("--select", "layout=warehouse_a", "--select", "job=vacuum/1") == [warehouse.id]
    assert picked("--cases", f"{medium.id},{warehouse.id}") == sorted([medium.id, warehouse.id])
    assert picked("--cases", f"{short.id},{medium.id}", "--select", "length=short") == [short.id]


def test_workers_equal_serial(tmp_path, monkeypatch):
    """Contract (plan §5.1): `--workers N` changes nothing but the time. A baseline run in two worker processes
    (`suite.run_suite`'s spawn pool) gives the serial run's rows, in the same order, and so does a fake agent run in
    three threads. Regression caught: a worker reordering or dropping cases, rows of one case written into another's
    folder, `--workers` that never reaches the pool."""
    register_agent(monkeypatch)
    data = tmp_path / "data"
    short_dataset(data)
    for system, extra, workers in (("memory_table", ["--channel", "detections"], "2"), ("fake:ok", [], "3")):
        folder = tmp_path / system.replace(":", "_")
        assert cli(system, "--data", data, "--out", folder / "serial", *extra) == 0
        assert cli(system, "--data", data, "--out", folder / "pool", *extra, "--workers", workers) == 0
        serial, pool = (results(folder / name) for name in ("serial", "pool"))
        assert pool["rows"] == serial["rows"] != []
        assert [c["id"] for c in pool["cases"]] == [c["id"] for c in serial["cases"]]
        assert [c["status"] for c in pool["cases"]] == ["complete"] * 2


def test_a_prior_nm_baseline_runs_its_first_wave(tmp_path, monkeypatch):
    """Contract (plan §5.1): `current_view_prior` keys its priors on `current_view`'s answers, so `eval` (unlike
    `stbench run`) runs `current_view` over every case first and builds the options of the second wave from it:
    `suite_priors.options_for` is asked for it with both cases' `current_view` runs in `done`, and the run completes,
    with the during questions off too (that first wave still asks them: the records it feeds need an answer to every
    question). Regression caught: the prior falling to a first wave that never ran (`options_for` raises), a prior
    baseline that `eval` refuses as `run` does, or a first wave that stops asking under `--during off`."""
    seen = []
    real = suite_priors.options_for

    def watching(name, case, cases, done):
        seen.append((name, sorted(done)))
        return real(name, case, cases, done)

    monkeypatch.setattr(suite_priors, "options_for", watching)
    data, out = tmp_path / "data", tmp_path / "prior"
    short_dataset(data)
    assert cli("current_view_prior", "--data", data, "--out", out, "--channel", "detections") == 0
    ids = sorted(c["id"] for c in results(out)["cases"])
    second = [done for name, done in seen if name == "current_view_prior"]
    assert second == [[("current_view", i) for i in ids]] * 2
    assert len(results(out)["rows"]) == 6
    off = tmp_path / "off"   # the first wave still asks the during questions its records need: `during off` runs
    assert cli("current_view_prior", "--data", data, "--out", off, "--channel", "detections", "--during", "off") == 0
    assert len(results(off)["rows"]) == 4


def test_results_validate(data, tmp_path):
    """Contract (plan §2.7): `results.json` is a document of the present shape: `problems` finds nothing wrong with one
    (nor with one that also holds a key it does not list, the stamps of an earlier build), and finds a missing key, a
    value of the wrong type or outside its choices, a case without its status, a row of an unknown outcome, a damaged
    aggregate and an aggregate without its score above the floor or its relevance split in a damaged copy, which `load`
    refuses naming the file; the row schema names every key `score_stats` says a row has.
    Regression caught: a results file that drifts from its schema unnoticed, a schema that lags the row keys, or a
    validator that accepts anything (the damaged copies prove it can fail)."""
    assert cli("all_unknown", "--data", data, "--out", tmp_path) == 0
    doc = results(tmp_path)
    assert eval_results.problems(doc) == [] and eval_results.load(tmp_path)["system"] == "all_unknown"
    assert eval_results.problems({**doc, "schema": "old", "agent": "old", "versions": {"x": "y"}}) == []
    assert set(eval_results.ROW) == set(score_stats.ROW_KEYS) and set(eval_results.CASE) <= set(doc["cases"][0])
    layout = doc["layouts"]["apartment_a"]
    damaged = {
        "a key missing": {k: v for k, v in doc.items() if k != "rows"},
        "a wrong type": {**doc, "verified": "yes"},
        "outside the choices": {**doc, "sandbox": "somewhere"},
        "a case without its status": {**doc, "cases": [{k: v for k, v in doc["cases"][0].items() if k != "status"}]},
        "a bad row": {**doc, "rows": [{**doc["rows"][0], "outcome": "fine"}]},
        "a damaged aggregate": {**doc, "layouts": {"apartment_a": {**layout, "n": "six"}}},
        "an aggregate without its floor": {**doc, "layouts": {
            "apartment_a": {k: v for k, v in layout.items() if k != "above_floor"}}},
        "an aggregate without its relevance split": {**doc, "layouts": {"apartment_a": {
            **layout, "by": {k: v for k, v in layout["by"].items() if k != "relevance"}}}},
        "a token count that is text": {**doc, "cases": [{**doc["cases"][0], "tokens": {"tokens_in": "many"}}]},
    }
    for what, bad in damaged.items():
        assert eval_results.problems(bad), what
    (tmp_path / "bad.json").write_text(json.dumps(damaged["a wrong type"]), encoding="utf-8")
    refused = r"bad\.json: is not a results document: results\.verified"
    with pytest.raises(ValueError, match=refused):
        eval_results.load(tmp_path / "bad.json")


def test_layouts_are_never_pooled(tmp_path):
    """Contract (VISION rule 13): each layout is aggregated on its own. Of two cases whose scripts are on two layouts
    (the second script's index entry says `warehouse_a`), `all_unknown` scores [0, 0, 1] on the first, the `now` row 0
    and the `memory` rows (0, 1) 1/2, headline 1/4, and [0, 1, 1] on the second, `now` 0 and `memory` 1, headline 1/2;
    the results hold one aggregate each, every row names its layout, and no number is the two layouts' joint headline
    3/8. Regression caught: a headline that averages two worlds, a number about neither."""
    first = compiled(tmp_path, hand_script(1), unknown=(2,))
    second = compiled(tmp_path, hand_script(2), unknown=(1, 2))
    entry = case_mod.read_entry(tmp_path, second.script.id)
    case_mod.write_entry(tmp_path, second.script.id, {**entry, "map": "warehouse_a"})
    out = tmp_path / "run"
    assert cli("all_unknown", "--data", tmp_path, "--out", out) == 0
    doc = results(out)
    assert {k: v["headline"]["mean"] for k, v in doc["layouts"].items()} == pytest.approx(
        {"apartment_a": 1 / 4, "warehouse_a": 1 / 2})
    assert {r["layout"] for r in doc["rows"] if r["case"] == first.id} == {"apartment_a"}
    assert {r["layout"] for r in doc["rows"] if r["case"] == second.id} == {"warehouse_a"}
    md = (out / "results.md").read_text(encoding="utf-8")
    assert "## Layout apartment_a: 1 of 1 case finished" in md and "## Layout warehouse_a: 1 of 1 case finished" in md
    assert "| headline | 3 | +0.250 |" in md and "| headline | 3 | +0.500 |" in md and "| headline | 6 |" not in md


def test_how_a_case_ends_is_its_status(data, tmp_path, monkeypatch, capsys):
    """Contract (plan §5.1): the outcome of a case is what `run_case` returned (`complete`, or `ended: <why>` with the
    failures counted by reason), `refused: <message>` when it raised before anything was sent, or `invalid`; the run
    goes on past all three, exits 1 and says which cases did not finish; only an agent that cannot run at all (it
    raises `Refused`) stops the pool, with exit 2 and nothing left behind. Regression caught: a failed case reported as
    a good one, a refused one that vanishes from the results, or a login failure that leaves a half-written folder."""
    register_agent(monkeypatch)
    out = tmp_path / "stop"
    assert cli("fake:stop", "--data", data, "--out", out) == 1
    doc = results(out)
    assert {c["status"] for c in doc["cases"]} == {"ended: observe: RuntimeError: out of memory"}
    assert all(c["failures"] == {"case ended: observe": 3} for c in doc["cases"]) and doc["rows"] == []
    assert doc["partial"] and doc["layouts"] == {"apartment_a": None}
    assert "Cases not finished" in (out / "results.md").read_text(encoding="utf-8")
    invalid = tmp_path / "invalid"
    assert cli("fake:invalid", "--data", data, "--out", invalid) == 1
    assert {c["status"] for c in results(invalid)["cases"]} == {"invalid"}
    assert {c["error"] for c in results(invalid)["cases"]} == {"the leak scan hit"}
    only_during = compiled(tmp_path / "during", hand_script(7), [QUESTIONS[0]])
    refused = tmp_path / "refused"
    assert cli("fake:ok", "--data", only_during.dataset, "--out", refused, "--during", "off") == 1
    (case,) = results(refused)["cases"]
    assert case["status"].startswith("refused: ") and "no question to ask" in case["status"]
    assert not (refused / "cases" / case["id"]).exists()
    capsys.readouterr()
    login = tmp_path / "login"
    assert cli("fake:refuse", "--data", data, "--out", login) == 2
    assert "no such login" in capsys.readouterr().err and not login.exists()


def test_eval_scores_as_run_does(data, tmp_path):
    """Contract (plan §5.1): a case written by `eval` is the case `stbench run` writes (`cmd_run.write_run`): the same
    three files, its rows are `suite.score_rows`' and its summary's `scores` block is the aggregate of them. Regression
    caught: a second scorer inside `eval` that drifts from `run`'s."""
    assert cli("all_unknown", "--data", data, "--out", tmp_path / "e") == 0
    doc = results(tmp_path / "e")
    for case in doc["cases"]:
        folder = tmp_path / "e" / "cases" / case["id"]
        assert sorted(p.name for p in folder.iterdir()) == ["answers.jsonl", "rows.jsonl", "summary.json"]
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (folder / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
        assert rows == [r for r in doc["rows"] if r["case"] == case["id"]]
        assert summary["scores"]["headline"] == score_stats.aggregate(rows)["headline"]
        assert (summary["system"], summary["sandbox"], summary["cost_note"]) == (
            "all_unknown", "in-process", "no token counts reported")
