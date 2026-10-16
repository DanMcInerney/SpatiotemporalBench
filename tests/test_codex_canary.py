"""The Codex canary (`tools/codex_canary.py`, M5-d) on the pull flow, over the fake command (`codex_fake`) and synthetic
cases: how it plants a token and probes for it (the live verdict needs the real Codex; the fake recalls nothing). No
model, no `codex`, no network.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from codex_kit import Fake, pulled
from stbench import case as case_mod
from stbench.stream import run_case
from stbench.stream_protocol import DEFAULT_CHANNEL
from stream_kit import K, L, case_on, end_q, qid
from synth import Synth

SPEC = importlib.util.spec_from_file_location("codex_canary", Path(__file__).resolve().parents[1] / "tools" /
                                              "codex_canary.py")
canary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(canary)
TOKEN = "CANARY-0000ABCD"


def test_the_canary_plants_in_the_first_task_repeats_before_the_last_frame_and_probes_on_the_thread(tmp_path,
                                                                                                   monkeypatch):
    """Contract (`codex_canary` docstring): case A's first task message ends with the plant and the plant comes once
    more as a message before its last frame (`repeat`); each case ends, after the job is over, with the probe as a
    resume of its own thread, whose reply the system keeps (`reply`); case B has no plant anywhere; C is one start
    exec whose whole prompt is the probe, on a thread of its own, with nothing pulled. Regression caught: a canary
    that still drives the retired exec-per-message flow (it would run no case at all), a probe sent on another thread
    than the case's, or a plant in B (the verdict would call carry-over that is not)."""
    starts = iter(["2026-03-02T08:00:00", "2026-09-14T09:30:00"])
    monkeypatch.setattr(case_mod, "start_clock", lambda rng: next(starts))
    one = case_on(tmp_path / "a", Synth(tasks=[[(K, 6)]]), [end_q(1)])
    two = case_on(tmp_path / "b", Synth(tasks=[[(L, 6)]]), [end_q(2)])
    fake = Fake(tmp_path)
    ran_a, a = canary.play(fake.codex, one, tmp_path / "out-a", token=TOKEN, repeat=True, during=False)
    ran_b, b = canary.play(fake.codex, two, tmp_path / "out-b", token=None, repeat=False, during=False)
    reply_c = canary.bare(fake.codex, tmp_path / "out-c")
    assert (ran_a.status, ran_b.status) == ("complete", "complete")
    first, probe_a, second, probe_b, bare = fake.execs()
    plant = canary.PLANT.format(token=TOKEN)
    assert first["prompt"].endswith(f"\n{plant}") and a.planted and a.planted.endswith(plant)
    frames = [c["text"] for c in first["calls"] if c["tool"] == "next_frame"]
    assert frames.count(plant) == 1 and frames[frames.index(plant) + 1].startswith("[k5 ")
    assert (probe_a["sub"], probe_a["base"], probe_a["prompt"]) == ("resume", first["thread"], canary.PROBE)
    assert (probe_b["sub"], probe_b["base"], probe_b["prompt"]) == ("resume", second["thread"], canary.PROBE)
    assert (bare["sub"], bare["base"], bare["prompt"]) == (None, None, canary.PROBE) and bare["finished"]
    assert not [c for c in pulled(bare) if not c["text"].startswith("The job is over")]
    assert "CANARY" not in second["prompt"] + "".join(c["text"] for c in second["calls"]) and b.planted is None
    assert (a.reply, b.reply, reply_c) == ("noted", "noted", "noted")


def test_a_dry_run_sends_nothing_and_shows_the_planted_message(tmp_path, monkeypatch):
    """Contract (`codex_canary` docstring): `--dry-run` runs a case over a system that sends nothing to Codex, keeps the
    first task message as it would be planted, counts what it was sent and fails every question. Regression caught: a
    dry run that needs a transport (it would call Codex), or one that plants nothing."""
    monkeypatch.setattr(case_mod, "start_clock", lambda rng: "2026-03-02T08:00:00")
    case = case_on(tmp_path / "a", Synth(tasks=[[(K, 6)]]), [end_q(1)])
    made: list = []

    def make(state, work):
        made.append(canary.Dry(state, work, token=TOKEN))
        return made[0]

    done = run_case(make, case, channel=DEFAULT_CHANNEL, during=False)
    assert done.status == "complete" and done.answers == {qid(1): {"type": "failed", "reason": "dry run"}}
    assert made[0].sent == 1 + 6 + 1  # the task, six frames and the end ask
    assert made[0].planted.startswith("Task 0 begins at 2026-03-02T08:00:00.0.\n")
    assert made[0].planted.endswith(canary.PLANT.format(token=TOKEN))


@pytest.mark.parametrize(("text", "held"), [
    ("the code word is CANARY-0000ABCD.", True), ("0000abcd", True), ("x0000abcdef", False), ("0000abc", False)])
def test_a_text_holds_the_token_when_its_hex_digits_stand_alone(text, held):
    """Contract (`codex_canary.holds`): the token's eight hex digits stand alone in the text, in any case and with or
    without the prefix, not inside a longer run of hex digits. Regression caught: a hash in a journal that matches by
    chance and fails a clean canary."""
    assert canary.holds(text, TOKEN) is held


def test_the_verdict_follows_the_rules_fixed_before_any_exec():
    """Contract (`codex_canary` docstring): `fail` when B's or C's reply, or B's journal, holds the token (or an earlier
    attempt's); else `pass` when A's reply holds it (the positive control: the probe can detect recall); else
    `inconclusive`. Regression caught: a pass given when the probe could not have detected recall, or a carry-over
    missed in B's journal."""
    assert canary.verdict(TOKEN, "CANARY-0000ABCD", "NONE", "NONE", "journal") == "pass"
    assert canary.verdict(TOKEN, "NONE", "NONE", "NONE", "journal") == "inconclusive"
    assert canary.verdict(TOKEN, "0000ABCD", "0000abcd", "NONE", "journal") == "fail"
    assert canary.verdict(TOKEN, "0000ABCD", "NONE", "NONE", "a line holding 0000abcd") == "fail"
    assert canary.verdict(TOKEN, "0000ABCD", "NONE", "ffff0000", "journal", earlier=("CANARY-FFFF0000",)) == "fail"
