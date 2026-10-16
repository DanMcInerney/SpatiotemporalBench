"""`stbench view`'s reader and command (`view.py`, `view_questions.py`, `view_log.py`, `view_cmd.py`, docs/eval.md):
a run folder and its dataset read into one dict. The runs are real `stbench eval` runs over `eval_kit.hand_dataset`;
what a test expects is read from the files the run and the dataset wrote, or worked out by hand from `all_unknown`
(an answerable gold scores 0 and is abstained, an unknown one scores 1 and is correct), never from the reader."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from eval_kit import cli, register_agent
from runview_kit import PageParts, jsonl, make_run, start_stamp
from stbench import view_page
from stbench.cli import main
from stbench.view import find_dataset, read_run
from stbench.view_log import MAX_CHARS


def case_of(doc: dict, cid: str, arm: int = 0) -> dict:
    return next(c for c in doc["arms"][arm]["cases"] if c["id"] == cid)


def test_every_case_and_question_is_there_with_the_answer_and_the_gold(tmp_path):
    """Contract (docs/eval.md, `stbench view`): the data holds every case of the run and, for each, every question in
    ask order with what the solver answered (`answers.jsonl`), the gold (`gold.jsonl`), the text, the outcome and score
    of its row and the link to its evidence frame; the header has the run's options and the headline of the layout.
    By hand: the first case's golds are [answerable, answerable, unknown], so under `all_unknown` its outcomes are
    [abstained, abstained, correct] and its mean 1/3; the second's are [abstained, correct, correct], mean 2/3; the
    headline of the layout is the mean of its columns, now 0/2 and memory 3/4: 0.375 (pooled it would be 3/6), shown
    with each column's mean and count. Regression caught: a question or case left out, an answer paired with the wrong
    question, a gold taken from the wrong row, or a frame link to a picture that is not there."""
    (first, second), run = make_run(tmp_path)
    doc = read_run(run, tmp_path)
    (arm,) = doc["arms"]
    assert sorted(c["id"] for c in arm["cases"]) == sorted([first.id, second.id]) and arm["name"] == ""   # random ids
    assert arm["header"]["system"] == "all_unknown" and arm["header"]["channel"] == "rgb_pose"
    assert arm["layouts"]["apartment_a"]["headline"]["mean"] == pytest.approx(0.375)
    columns = arm["layouts"]["apartment_a"]["columns"]
    assert {k: (c["mean"], c["n"]) for k, c in columns.items()} == pytest.approx({"now": (0.0, 2), "memory": (0.75, 4)})
    assert (arm["totals"]["finished"], arm["totals"]["cases"]) == (2, 2)
    expected = {first.id: (["abstained", "abstained", "correct"], 1 / 3),
                second.id: (["abstained", "correct", "correct"], 2 / 3)}
    for case in (first, second):
        shown = case_of(doc, case.id)
        asked, golds = jsonl(case.questions), jsonl(case.gold)
        answers = jsonl(run / "cases" / case.id / "answers.jsonl")
        assert [q["qid"] for q in shown["questions"]] == [q["qid"] for q in asked]
        assert [q["text"] for q in shown["questions"]] == [q["text"] for q in asked]
        assert [q["answer"] for q in shown["questions"]] == [a["answer"] for a in answers] == [{"type": "unknown"}] * 3
        assert [q["gold"] for q in shown["questions"]] == [g["gold"] for g in golds]
        assert [q["outcome"] for q in shown["questions"]] == expected[case.id][0]
        assert shown["counts"]["mean"] == pytest.approx(expected[case.id][1])
        assert [(q["when"], q["step"]) for q in shown["questions"]] == [("during", 12), ("end", None), ("end", None)]
        assert shown["status"] == "complete" and shown["kind"] == "complete" and shown["layout"] == "apartment_a"
        assert shown["window"]["k0"] == 0 and shown["listed"] is True
        (frame,) = shown["questions"][0]["evidence"]
        assert frame["k"] == 0 and frame["url"].startswith("file:///") and frame["url"].endswith("/000000.jpg")
    first_picture = first.script.rgb(0)
    first_picture.unlink()
    assert case_of(read_run(run, tmp_path), first.id)["questions"][0]["evidence"] == [{"k": 0, "url": None}]


def test_how_a_case_ended_is_its_status_and_its_failure(tmp_path, monkeypatch):
    """Contract: a case that ended, was refused or was invalid shows that as its status, with the failures by reason and
    the reason of the results (`error`), and its questions show as failed answers scored wrong; a case that did not
    finish is not counted finished. `fake:stop` fails in `observe` (three questions fail with `case ended: observe`);
    `fake:invalid` leaves no summary and `error` says why. Regression caught: a failed case listed as a good one, or its
    reason lost."""
    register_agent(monkeypatch)
    (first, _), _ = make_run(tmp_path)
    stop, invalid = tmp_path / "runs" / "stop", tmp_path / "runs" / "invalid"
    assert cli("fake:stop", "--data", tmp_path, "--out", stop) == 1
    assert cli("fake:invalid", "--data", tmp_path, "--out", invalid) == 1
    ended = case_of(read_run(stop, tmp_path), first.id)
    assert ended["status"] == "ended: observe: RuntimeError: out of memory" and ended["kind"] == "ended"
    assert ended["failures"] == {"case ended: observe": 3} and ended["counts"]["wrong"] == 3
    assert [q["answer"] for q in ended["questions"]] == [{"type": "failed", "reason": "case ended: observe"}] * 3
    assert read_run(stop, tmp_path)["arms"][0]["totals"]["finished"] == 0
    bad = case_of(read_run(invalid, tmp_path), first.id)
    assert (bad["status"], bad["kind"], bad["error"], bad["ran"]) == ("invalid", "invalid", "the leak scan hit", False)
    assert [q["asked"] for q in bad["questions"]] == [False] * 3 and bad["counts"]["n"] == 0


def test_arm_folders_are_each_listed_and_the_off_arm_asks_no_during_question(tmp_path, monkeypatch):
    """Contract (docs/eval.md, "Arms"): a folder of arm run folders is read as one arm each, named by its folder, and
    the command finds the dataset above `runs/` by itself and writes `view.html` beside the arms. In the `during=off`
    arm the during question was not asked: it is listed with `asked` false and no outcome, and the `on` arm has all
    three scored. Regression caught: arms merged or dropped, or an unasked question shown as wrong."""
    register_agent(monkeypatch)
    (first, _), _ = make_run(tmp_path)
    arms = tmp_path / "runs" / "arms"
    assert cli("fake:ok", "--data", tmp_path, "--out", arms, "--arms", "during=on,off") == 0
    assert main(["view", str(arms)]) == 0   # no --data: RUN_DIR is under tmp_path/runs
    page = PageParts((arms / "view.html").read_text(encoding="utf-8")).embedded()
    assert [a["name"] for a in page["arms"]] == ["during-off", "during-on"]
    assert page["dataset"] == tmp_path.resolve().as_posix()
    off, on = (case_of(page, first.id, i) for i in (0, 1))
    assert [q["asked"] for q in off["questions"]] == [False, True, True] and off["questions"][0]["outcome"] is None
    assert [q["asked"] for q in on["questions"]] == [True] * 3 and off["counts"]["n"] == 2 and on["counts"]["n"] == 3
    assert [a["header"]["during"] for a in page["arms"]] == [False, True]


def test_a_run_without_results_json_or_a_transport_log_still_renders(tmp_path):
    """Contract: a run that was killed has no `results.json`, one case may have no `summary.json`, a file may be cut in
    the middle of a line, and a scripted system has no transport log: the reader shows what the folders hold, says what
    is wrong in the case's `problems` and takes the header from a `summary.json`; it computes no headline (that is
    `results.json`'s, which eval writes when a run ends), and the page renders. Regression caught: the viewer failing on
    exactly the runs one most needs to look at, or a statistic of its own in place of the file's."""
    (first, second), run = make_run(tmp_path)
    (run / "results.json").unlink()
    (run / "results.md").unlink()
    (run / "cases" / second.id / "summary.json").unlink()
    with (run / "cases" / first.id / "rows.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"qid": "cut in the mid')
    doc = read_run(run, tmp_path)
    (arm,) = doc["arms"]
    assert arm["header"]["system"] == "all_unknown"
    assert arm["layouts"] == {} and arm["results"] == {"present": False, "finished": None}
    assert case_of(doc, first.id)["log"] is None and case_of(doc, first.id)["transport"] is None
    assert any("rows.jsonl line 4" in p for p in case_of(doc, first.id)["problems"])
    lost = case_of(doc, second.id)
    assert (lost["status"], lost["kind"], lost["ran"]) == ("unfinished (no summary.json)", "unfinished", False)
    assert [q["outcome"] for q in lost["questions"]] == ["abstained", "correct", "correct"]   # its rows are there
    assert arm["totals"] == {**arm["totals"], "cases": 2, "finished": 1}
    parts = PageParts(view_page.render(doc))
    assert case_of(parts.embedded(), second.id)["status"] == "unfinished (no summary.json)"


def test_a_run_in_progress_is_counted_from_its_case_folders_not_the_stamp_eval_wrote_at_its_start(tmp_path):
    """Contract (docs/eval.md, `stbench view`; eval_cmd): eval writes `results.json` when a run begins, every case
    pending and `partial: 0 of N cases finished`, and again when it ends, so during the run the file is that stamp. The
    reader counts the finished case folders (`totals`), drops the file's own `partial:` note (the page says how the run
    stands itself), keeps its other notes, records how many cases the file lists as finished (`results`), and keeps its
    layouts as they are, null for a layout it has no aggregate for. Here the first case's folder is complete, the
    second's has lost its `summary.json`, and the stamp lists none finished. Regression caught: a page whose banner says
    0 of 2 beside a header that says 1 of 2, or a headline the viewer made up for a run that has none yet."""
    (first, second), run = make_run(tmp_path)
    (run / "cases" / second.id / "summary.json").unlink()
    start_stamp(run, "sandbox none: not verified")
    (arm,) = read_run(run, tmp_path)["arms"]
    assert (arm["totals"]["finished"], arm["totals"]["cases"]) == (1, 2)
    assert arm["results"] == {"present": True, "finished": 0}
    assert arm["header"]["notes"] == ["sandbox none: not verified"]
    assert arm["layouts"] == {"apartment_a": None}
    assert case_of({"arms": [arm]}, first.id)["kind"] == "complete"
    assert case_of({"arms": [arm]}, second.id)["kind"] == "unfinished"


def test_the_folder_of_one_run_command_is_a_run_of_one_case(tmp_path):
    """Contract (docs/format.md, `runs/<system>-<cid>/`): `stbench run`'s folder, with `summary.json` beside the answers
    and no `cases/`, is read as one case. Regression caught: the viewer refusing the folder `run` writes."""
    (first, _), _ = make_run(tmp_path)
    assert main(["run", "all_unknown", str(first.private), "--out", str(tmp_path / "runs" / "one")]) == 0
    (arm,) = read_run(tmp_path / "runs" / "one", tmp_path)["arms"]
    assert [c["id"] for c in arm["cases"]] == [first.id] and arm["cases"][0]["counts"]["n"] == 3


def test_the_command_refuses_what_it_cannot_show(tmp_path, capsys):
    """Contract (view_cmd): exit 2 with a reason and nothing written when RUN_DIR holds no run, when no dataset is given
    and RUN_DIR is under none, and when `--out` is under the dataset's `scripts/` or `private/` (the page holds the
    gold, which no agent-facing folder may). `--data` makes a run outside a dataset work. Regression caught: a page
    written beside the pictures a system is given, or a traceback where a sentence belongs."""
    (first, _), run = make_run(tmp_path / "data")
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["view", str(empty)]) == 2 and "holds no run" in capsys.readouterr().err
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(run, elsewhere)
    assert find_dataset(elsewhere) is None
    assert main(["view", str(elsewhere)]) == 2 and "pass --data" in capsys.readouterr().err
    assert not (elsewhere / "view.html").exists()
    assert main(["view", str(run), "--out", str(first.private / "view.html")]) == 2
    assert "never writes to" in capsys.readouterr().err and not (first.private / "view.html").exists()
    assert main(["view", str(elsewhere), "--data", str(tmp_path / "data")]) == 0
    assert (elsewhere / "view.html").is_file()


def test_the_same_files_give_the_same_dict_and_the_same_page(tmp_path):
    """Contract (view.py): the reader is a function of the files alone, no clock and no random value, so reading twice
    gives equal dicts and the page of each is the same bytes, and so is the file the command writes twice. Regression
    caught: a timestamp or a set's order in the page, which would make two views of one run differ."""
    (_, _), run = make_run(tmp_path)
    one, two = read_run(run, tmp_path), read_run(run, tmp_path)
    assert json.dumps(one) == json.dumps(two) and view_page.render(one) == view_page.render(two)
    assert main(["view", str(run)]) == 0
    first = (run / "view.html").read_bytes()
    assert main(["view", str(run)]) == 0
    assert (run / "view.html").read_bytes() == first


def test_a_transport_log_is_kept_whole_but_for_two_bulky_fields_and_long_text(tmp_path):
    """Contract (view_log): a case's `<name>.calls.jsonl` is shown record by record as written, with its
    `<name>.summary.json`; the raw `events` and the `argv` are left out (the events that report an error stay, as
    `event_errors`), a string over 20,000 characters is cut and counted, a line that is not JSON is a problem and not a
    crash. Regression caught: a log of tens of megabytes in the page, an error lost with the events, or an unreadable
    line that stops the page."""
    (first, _), run = make_run(tmp_path)
    folder = run / "cases" / first.id
    long = "x" * (MAX_CHARS + 500)
    lines = [json.dumps({"event": "run", "model": "m", "tools": ["locate"]}),
             json.dumps({"i": 0, "purpose": "observe", "kind": "start", "argv": ["agent", "--secret"],
                         "events": [{"type": "turn.started"}, {"type": "error", "message": "boom"}],
                         "tools": [{"tool": "locate", "arguments": {"k": 1}, "result": long}], "messages": ["ok"]}),
             "not json at all",
             json.dumps({"event": "exec_failed", "reason": "error", "message": "at capacity"})]
    (folder / "agent.calls.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (folder / "agent.summary.json").write_text(json.dumps({"model": "m", "tool_errors": 0}), encoding="utf-8")
    shown = case_of(read_run(run, tmp_path), first.id)
    log = shown["log"]
    assert [r.get("event") or r["i"] for r in log["records"]] == ["run", 0, "exec_failed"]
    exec_record = log["records"][1]
    assert "argv" not in exec_record and "events" not in exec_record and exec_record["messages"] == ["ok"]
    assert exec_record["event_errors"] == [{"type": "error", "message": "boom"}]
    result = exec_record["tools"][0]["result"]
    assert result.startswith("x" * MAX_CHARS) and len(result) < MAX_CHARS + 100 and "500 more characters" in result
    assert (log["file"], log["cut"], log["left_out"]) == ("agent.calls.jsonl", 1, ["argv", "events"])
    assert any("line 3" in p for p in shown["problems"])
    assert shown["transport"] == {"name": "agent", "summary": {"model": "m", "tool_errors": 0}}
    assert read_run(run, Path(tmp_path))["arms"][0]["header"]["model"] == "m"
