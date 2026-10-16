"""The Codex reference agent between cases (M5-d): a second case of a run starts a fresh thread that holds nothing of
the first. Over the fake command (`codex_fake`) and two synthetic cases on one shared transport, as a run of many cases
shares one `Codex`: no model, no `codex`, no network.

The live half of this contract, a token planted in one case's thread that another case's fresh thread must not recall,
needs the real Codex and is `tools/codex_canary.py`. What the fake can show, and what this test pins, is
everything the harness sends and starts: the thread and the prompt (the fake keeps each thread's messages, as Codex
does, and the reference agent has no memory folder or other store that could carry anything between cases).
"""

from __future__ import annotations

from codex_kit import Fake
from stbench import case as case_mod
from stbench.codex_log import Journal
from stbench.codex_system import CodexSystem
from stbench.stream import run_case
from stream_kit import K, L, case_on, clock, end_q
from synth import Synth

FIRST_QUESTION = "Where was the alpha mug last seen?"
SECOND_QUESTION = "Where was the bravo book last seen?"
FRAME_SECONDS = (0, 1.5, 3, 4.5, 6, 7.5)  # six keyframes 1.5 s apart, each clock delivered to the tenth


def shown(record: dict) -> str:
    """Everything the model of an exec was shown: its prompt and the reply to each of its calls."""
    return "\n".join([record["prompt"], *(call["text"] for call in record["calls"])])


def test_a_second_case_starts_a_fresh_thread_with_nothing_of_the_first(tmp_path, monkeypatch):
    """Contract (M5-d, plan 5.4): two cases run one after the other on one `Codex` transport are two threads. The second
    case's exec is a start (not a resume, no base thread) on a thread the first never used, holding no earlier message;
    what it was shown holds its own brief and nothing of the first case (no word of its room, none of its frame clocks,
    none of its question). Positive control: the first case's thread was shown its room, its clocks and its question, so
    the second case's lack of each is a result. Regression caught: a thread id cached on the transport, or state kept
    between cases, which would let the arms of M6's leak check talk to each other."""
    starts = iter(["2026-03-02T08:00:00", "2026-09-14T09:30:00"])  # two fixed clocks: a random draw could collide
    monkeypatch.setattr(case_mod, "start_clock", lambda rng: next(starts))
    first = case_on(tmp_path / "first", Synth(tasks=[[(K, 6)]]), [end_q(1, text=FIRST_QUESTION)])
    second = case_on(tmp_path / "second", Synth(tasks=[[(L, 6)]]), [end_q(2, text=SECOND_QUESTION)])
    fake = Fake(tmp_path)
    journals = {"first": Journal(tmp_path / "first.calls.jsonl"), "second": Journal(tmp_path / "second.calls.jsonl")}
    for name, case in (("first", first), ("second", second)):
        # one system a case, built as `cmd_codex` builds it, on the one shared transport and a journal of its own
        done = run_case(lambda state, work, j=journals[name]: CodexSystem(state, work, codex=fake.codex, journal=j),
                        case)
        assert (name, done.status) == (name, "complete")
    one, two = fake.execs()  # a case is one exec

    # -- positive control: the first case's thread was shown what the second must not have --------------------------
    clocks = [clock(first, s) for s in FRAME_SECONDS]
    assert (one["sub"], one["base"], one["history"]) == (None, None, [])
    assert "kitchen" in one["prompt"] and "living room" not in one["prompt"]
    assert all(c in shown(one) for c in clocks) and FIRST_QUESTION in shown(one)

    # -- the second case: a fresh thread of its own and nothing of the first -----------------------------------------
    assert (two["sub"], two["base"], two["history"]) == (None, None, [])
    assert two["thread"] != one["thread"] and "living room" in two["prompt"]
    for gone in ("kitchen", FIRST_QUESTION, *clocks):
        assert gone not in shown(two), gone
    assert SECOND_QUESTION in shown(two)
