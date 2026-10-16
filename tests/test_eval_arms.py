"""`stbench eval --arms` (`eval_run.py`, plan §5.1's B3, ready for M6): one lever's arms run on every case, each into a
run folder of its own, in an order seeded per case, differing in that lever alone; and `suite.run_suite` passing the
brief variant and the during flag to every run of a system, which is how a baseline runs under an arm."""

from __future__ import annotations

import pytest

from eval_kit import RUNS, cli, compiled, hand_dataset, hand_script, register_agent, results
from stbench.baselines import BASELINES, Baseline
from stbench.suite import RunOptions, run_suite
from stream_kit import Recorder, messages

# Which arm of `during` runs first for the cases `0000000000000001`, `...2`, `...3` and `...5`: the lower of
# sha256("<case id>\x1fduring\x1f<value>")'s first eight bytes, worked out by hand (not by `stable_hash`).
FIRST = {"0000000000000001": ["on", "off"], "0000000000000002": ["off", "on"], "0000000000000003": ["off", "on"],
         "0000000000000005": ["on", "off"]}
NEUTRAL = "It is carrying on with its usual work."


def test_arms_are_seeded_per_case(tmp_path, monkeypatch):
    """Contract (plan §5.1; design §6.3): `--arms during=on,off` runs every case once in each of two run folders
    (`RUNDIR/during-on`, `RUNDIR/during-off`), the two arms of one case one after another in the order the hash of
    (case id, lever, value) gives, so both orders occur and no arm always runs first; the two results differ only in
    `during` (the off arm holds the end questions' rows and nothing else), and what a case's system was sent, but for
    the during asks, is the same bytes in both arms (the brief's during sentence is in every arm). Regression caught:
    an arm that always runs first, or arms that differ in more than the asks."""
    register_agent(monkeypatch)
    for n in (1, 2, 3, 5):
        compiled(tmp_path, hand_script(n), unknown=(2,), cid=f"{n:016x}")
    out = tmp_path / "run"
    assert cli("fake:ok", "--data", tmp_path, "--out", out, "--arms", "during=on,off") == 0
    assert sorted(p.name for p in out.iterdir()) == ["during-off", "during-on"]
    on, off = results(out / "during-on"), results(out / "during-off")
    assert (on["during"], off["during"]) == (True, False)
    for doc in (on, off):
        assert [c["id"] for c in doc["cases"]] == list(FIRST) and {c["status"] for c in doc["cases"]} == {"complete"}
    order: dict[str, list[str]] = {}
    for run in RUNS:
        order.setdefault(run["case"], []).append(run["during"])
    assert order == FIRST and {tuple(v) for v in order.values()} == {("on", "off"), ("off", "on")}
    keys = ("system", "channel", "brief", "sandbox", "reasoning", "verified", "max_usd")
    assert {k: on[k] for k in keys} == {k: off[k] for k in keys}
    ends = [r for r in on["rows"] if r["when"] == "end"]
    assert off["rows"] == ends and len(ends) == 8 and len(on["rows"]) == 12
    for case in FIRST:
        sent = {r["during"]: [e for e in r["log"] if e[0] in ("task", "observe")] for r in RUNS if r["case"] == case}
        assert messages(sent["on"]) == messages(sent["off"]) and len(sent["on"]) == 21   # a task and 20 frames
        asked = {r["during"]: [e[1] for e in r["log"] if e[0] == "ask"] for r in RUNS if r["case"] == case}
        assert asked == {"on": ["during", "end"], "off": ["end"]}


def test_arm_folders_and_stamps(tmp_path, monkeypatch, capsys):
    """Contract (plan §5.1): a brief lever runs `RUNDIR/brief-<value>` folders each stamped with its value and nothing
    else changed; the default RUNDIR is `DATA/runs/<system>-<channel>` with `-<brief>` and `-during-off` added for the
    options that are not the lever and not their defaults (a system's characters outside `[A-Za-z0-9._-]` become `_`);
    a rerun of the arms skips what is finished in each folder, and a lever may not be given as an option too (the
    refusal is in test_eval_run). Regression caught: arms that share a folder, a stamp of the wrong value, or a rerun
    that repeats an arm."""
    register_agent(monkeypatch)
    hand_dataset(tmp_path)
    assert cli("fake:ok", "--data", tmp_path, "--arms", "brief=neutral,correct", "--during", "off") == 0
    out = tmp_path / "runs" / "fake_ok-rgb_pose-during-off"
    assert sorted(p.name for p in out.iterdir()) == ["brief-correct", "brief-neutral"]
    docs = {p.name: results(p) for p in out.iterdir()}
    assert {name: (d["brief"], d["during"]) for name, d in docs.items()} == {"brief-correct": ("correct", False),
                                                                           "brief-neutral": ("neutral", False)}
    assert len(RUNS) == 4 and {(r["brief"], r["during"]) for r in RUNS} == {("correct", "off"), ("neutral", "off")}
    capsys.readouterr()
    assert cli("fake:ok", "--data", tmp_path, "--arms", "brief=neutral,correct", "--during", "off") == 0
    assert len(RUNS) == 4
    assert cli("fake:ok", "--data", tmp_path, "--arms", "during=on,off", "--brief", "neutral") == 0
    assert sorted(p.name for p in (tmp_path / "runs" / "fake_ok-rgb_pose-neutral").iterdir()) == [
        "during-off", "during-on"]


def test_a_baseline_runs_each_arm_over_the_whole_selection(tmp_path):
    """Contract (plan §5.1): a baseline's arms run as the agents' do, one run folder each: `all_unknown` under
    `during=on,off` scores all six questions in one and the four end ones in the other (0.375 and 0.75 by hand: the
    unknown golds are all end ones, so the on arm is (now 0 + memory 3/4)/2 and the off arm, with no `now` row, is its
    memory column 3/4), and each folder says which. Regression caught: a baseline that cannot run under `--during off`
    (`run_suite` took neither option), or one whose off arm still scores during questions."""
    hand_dataset(tmp_path)
    out = tmp_path / "run"
    assert cli("all_unknown", "--data", tmp_path, "--out", out, "--arms", "during=on,off") == 0
    on, off = results(out / "during-on"), results(out / "during-off")
    assert (len(on["rows"]), len(off["rows"])) == (6, 4)
    assert (on["layouts"]["apartment_a"]["headline"]["mean"], off["layouts"]["apartment_a"]["headline"]["mean"]) == (
        pytest.approx(0.375), pytest.approx(0.75))
    assert {r["when"] for r in off["rows"]} == {"end"}


def test_run_suite_passes_the_brief_and_during_to_the_systems_it_names(tmp_path, monkeypatch):
    """Contract (plan §5.1): `run_suite(brief=, during=)` gives every run of a system of `names` that brief variant and
    that during flag: under `neutral` and `during=False` the system is sent the neutral assignment and no during ask;
    by default the correct brief and the during ask. A helper system of an earlier wave (it feeds the next wave's
    priors) always asks the during questions, because the records that carry its answers need one for every question.
    Regression caught: a baseline that cannot be run under an arm, or a prior built from answers that are missing."""
    case = hand_dataset(tmp_path)[0]
    logs: dict[str, list] = {"helper": [], "main": []}
    for name in logs:
        monkeypatch.setitem(BASELINES, name, Baseline(Recorder))

    def options(name, case, cases, done):
        return RunOptions({"log": logs[name]})

    def sent(name: str) -> tuple[str, list[str]]:
        """The brief of the first task a system was sent, and the `when` of each ask."""
        return [e[4] for e in logs[name] if e[0] == "task"][0], [e[1] for e in logs[name] if e[0] == "ask"]

    run_suite([case], ["main"], "rgb_pose", options=options, waves=[["helper"], ["main"]], brief="neutral",
              during=False)
    assert NEUTRAL in sent("main")[0] and sent("main")[1] == ["end"]
    assert NEUTRAL in sent("helper")[0] and sent("helper")[1] == ["during", "end"]   # the helper still asks them
    logs["main"].clear()
    run_suite([case], ["main"], "rgb_pose", options=options, waves=[["helper"], ["main"]])
    assert NEUTRAL not in sent("main")[0] and sent("main")[1] == ["during", "end"]
