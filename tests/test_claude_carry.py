"""A memory that carries its record on the pull, for the Claude Code agent (`codex_memory` SPEC key `carry`,
`agent_tools.tools(True)`, `codex_feed.Feed(carry=)`, `codex_system`, `claude_system`, `codex_log`): `next_frame` takes
an optional `memory`, the harness writes `{"record": memory}` to the memory's sensor feed before it counts the pull, and
the feed's reply is the last text block of the item that pull gets. The feed, the relay and the tools are the Codex
agent's, tested in `test_codex_carry`; here the case is run through `stbench run claude:sonnet` over the fake command
(`claude_cli_fake`) and a tiny sensor feed (`memory_fake_sense`): no model, no `claude`, no network. Expected values are
written by hand. The invariant that matters most is that an arm without `carry` is sent what it always was.
"""

from __future__ import annotations

import copy
import json

import pytest
from _pytest.monkeypatch import MonkeyPatch

from claude_kit import pulled
from claude_memory_kit import play, sensed
from codex_kit import job_also_asks, memory
from stbench.codex_log import read_log, transcripts
from stbench.llm_scan import scan_log
from test_cmd_codex import jsonl, small_case

RECORD = {"photo": "k{k}", "record": {"label": "mug", "note": "frame {k}"}, "request_id": "r{k}"}
CARRY = {"next_memory": RECORD}   # the fake model records every frame it is shown, on its next pull
REPLY = '{"ok":true,"receipt":{"n":%d}}'   # what the fake sensor feed answers a record, compact


@pytest.fixture(autouse=True)
def the_job_asks_what_the_case_asks(monkeypatch):
    """`small_case` holds a `number` during question, which `vacuum/1` asks none of: see `test_cmd_claude.py`."""
    job_also_asks(monkeypatch, "distance_between")


@pytest.fixture(scope="module")
def played(tmp_path_factory):
    """The case with a carrying memory whose model records every frame it is shown, and the reference agent's, over the
    one case."""
    base = tmp_path_factory.mktemp("claude-carry")
    with MonkeyPatch.context() as patch:
        job_also_asks(patch, "distance_between")
        case = small_case(base / "data")
        carrying = play(base / "carry", patch, {0: CARRY}, memory(base / "spec", carry=True), case=case)
        reference = play(base / "reference", patch, None, case=case)
    return carrying, reference


def calls_of(done) -> list[dict]:
    return [e for e in done.journal if e.get("event") == "call" and e["tool"] == "next_frame"]


def test_each_record_follows_its_frame_into_the_feed_and_is_written_before_the_runner_goes_on(played):
    """Contract (`codex_system._carry`, `codex_feed`): a frame is fed before it is held, the record about it is written
    while the frame is served (the model's next pull) and before the runner's next message, and each frame's is one
    line `{"record": <memory as the model gave it>}` in the feed's one stream, replied `{"ok": true, "receipt": ...}`.
    The case is one task and frames 0..5: each frame is fed as it is served, so the feed's stream is the frame and then
    the record about it, six times over (frame 3 is served with its questions, and the model's pull after it comes
    after its answer). Regression caught: a record written while the runner writes an observe
    (the one pipe shared), written twice, or never."""
    carrying, _ = played
    messages = [m for m in sensed(carrying) if "message" in m]
    order = [(next(iter(m["message"])), m["message"].get("observe", {}).get("steps", [{}])[0].get("k"))
             for m in messages]
    assert order == [("start", None), ("task", None)] + [("observe", k) if i == 0 else ("record", None)
                                                         for k in range(6) for i in range(2)]
    records = [m for m in messages if "record" in m["message"]]
    assert [m["message"] for m in records] == [
        {"record": {"photo": f"k{k}", "record": {"label": "mug", "note": f"frame {k}"}, "request_id": f"r{k}"}}
        for k in range(6)]
    assert [m["n"] for m in records] == [4, 6, 8, 10, 12, 14] and len({m["pid"] for m in messages}) == 1


def test_the_reply_is_the_last_block_of_the_next_item_and_nothing_else_the_model_is_sent_changes(played):
    """Contract (`codex_feed`; a memory changes no message the model is sent): the user messages of the process and
    every item the model is served are the reference agent's, byte for byte, but that the pull after a frame item (the
    fake model records each frame it is shown) is served the item with `<name>: <reply>` added as its last text block,
    on its own line, and nothing is added to a pull that follows anything else (the first pull). Regression caught: the
    reply placed before the item's blocks or merged into its text, a reply on a pull that carried no record, or any
    change to a message of an arm that carries."""
    carrying, reference = played
    (ours,), (theirs,) = carrying.processes, reference.processes
    assert ours["messages"] == theirs["messages"]
    after_frame, receipts = False, iter(sorted(m["n"] for m in sensed(carrying) if "record" in m.get("message", {})))
    for mine, ref in zip(pulled(ours), pulled(theirs), strict=True):
        if after_frame:
            tail = "mem: " + REPLY % next(receipts)
            assert mine["text"] == ref["text"] + "\n" + tail and mine["blocks"] == [*ref["blocks"], "text"]
        else:
            assert (mine["text"], mine["blocks"]) == (ref["text"], ref["blocks"])
        assert mine["images"] == ref["images"]
        after_frame = ref["text"].startswith("[k")
    assert next(receipts, None) is None and len(pulled(ours)) == len(pulled(theirs))


def test_the_journal_keeps_the_record_and_the_reply_and_the_served_text_stays_the_items_own(played):
    """Contract (`codex_log`): the `call` event of a carrying `next_frame` holds `arguments.memory` and `memory_reply`
    (the reply object); the `served` event of the item keeps `text` unchanged and gains `memory_reply` (the text the
    model read), and an item whose pull carried nothing has none; `transcripts` ends the message the model held with
    that block, after the item's image and any questions that came with it. Regression caught: a journal that cannot
    say what the memory answered (the replay cannot tell a refused record), or a reply folded into the served text."""
    carrying, reference = played
    carried = [e for e in calls_of(carrying) if "memory" in e["arguments"]]
    assert [e["arguments"]["memory"]["photo"] for e in carried] == [f"k{k}" for k in range(6)]
    assert [e["memory_reply"] for e in carried] == [{"ok": True, "receipt": {"n": n}} for n in (4, 6, 8, 10, 12, 14)]
    served = [e for e in carrying.journal if e.get("event") == "served"]
    bare = [e for e in reference.journal if e.get("event") == "served"]
    assert [e["text"] for e in served] == [e["text"] for e in bare]
    assert [e["images"] for e in served] == [e["images"] for e in bare]
    mem = ["mem: " + REPLY % n for n in (4, 6, 8, 10, 12, 14)]
    # task 0 (in the prompt), frames 0, 1, 2, then frame 3 with its questions, frames 4 and 5 with the end ask, the end:
    # each reply is to the record the pull that got the item carried (the first pull carries none)
    assert [e.get("memory_reply") for e in served] == [None, None, *mem]
    held = [m for m in transcripts(carrying.journal)[0]["request"]["messages"] if m["role"] == "tool"]
    during = next(m for m in held if isinstance(m["content"], list) and len(m["content"]) == 4)
    assert [b["type"] for b in during["content"]] == ["text", "image_url", "text", "text"]
    assert during["content"][-1]["text"].startswith("mem: ")


def test_the_leak_scan_reads_the_replies_a_memory_carried(played, tmp_path):
    """Contract (`codex_log.transcripts`, `llm_scan`): the reply of a carried record is part of what the model was
    shown, so a question's text in one, before that question's ask, is a hit, as it is in an item the feed served; the
    run as played has none. Regression caught: carried replies invisible to the scan (a store could carry a future
    question to the model through the one channel the scan does not read)."""
    carrying, _ = played
    questions = jsonl(carrying.case.questions)
    end = next(q for q in questions if q["when"] == "end")
    path = carrying.out / "claude.calls.jsonl"
    assert carrying.summary["scan"] == [] and scan_log(path, [], questions=questions) == []
    records = copy.deepcopy(read_log(path))
    planted = next(r for r in records if r.get("event") == "served" and r.get("memory_reply"))
    planted["memory_reply"] += f" {end['text']}"
    bad = tmp_path / "planted.jsonl"
    bad.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    hits = scan_log(bad, [], questions=questions)
    assert hits and all("(tool)" in hit and f"question {end['qid']} before its ask" in hit for hit in hits), hits


def test_the_tool_server_is_offered_the_memory_argument_to_a_carrying_arm_only(played, tmp_path, monkeypatch):
    """Contract (`codex_mcp`, `codex_feed.server_config`): the configuration file of the tool server holds `carry`
    only for a memory whose SPEC says so, and so is the arm that is offered `next_frame`'s `memory` argument; a memory
    without `carry` is launched with the configuration every arm has (`address` and `key`), is sent a `memory` argument
    the model makes up and writes nothing for it, and sees the items the reference agent does. Regression caught: a
    record sent to a feed whose arm did not ask for it (a second factor in the arm), or a new key in the configuration
    every arm is launched with."""
    carrying, reference = played
    (ours,), (theirs,) = carrying.processes, reference.processes
    assert ours["config"]["carry"] is True and set(theirs["config"]) == {"address", "key"}
    plain = play(tmp_path / "plain", monkeypatch, CARRY, memory(tmp_path / "spec"), case=carrying.case)
    (only,) = plain.processes
    assert set(only["config"]) == {"address", "key"}
    assert not [m for m in sensed(plain) if "record" in m.get("message", {})]
    assert [(c["text"], c["blocks"]) for c in pulled(only)] == [(c["text"], c["blocks"]) for c in pulled(theirs)]
    assert not any("memory_reply" in e or e.get("event") == "memory_dropped" for e in plain.journal)


def test_the_pull_the_feed_refuses_writes_no_record(tmp_path, monkeypatch):
    """Contract (`codex_feed`) at the case: a model that pulls early on an ask (sending its record each time) is
    refused once with the nudge and carries nothing on that pull; one record is written for each frame, six, not eight.
    Regression caught: the refused pull's record written too (the same sighting twice in the store)."""
    done = play(tmp_path, monkeypatch, {0: {**CARRY, "premature": 1}}, memory(tmp_path / "spec", carry=True))
    assert len([m for m in sensed(done) if "record" in m.get("message", {})]) == 6
    early = [e for e in calls_of(done) if e.get("premature") and e["error"]]
    assert len(early) == 2 and all("memory_reply" not in e for e in early)   # one for each ask


def test_a_feed_that_fails_on_a_record_ends_the_case_as_memory(tmp_path, monkeypatch):
    """Contract (`codex_system._carry`): a sensor feed that dies while a record is written is journaled as
    `memory_failed` (message `record`) and the runner ends the case as `TransportError("memory")` at its next check,
    before any item is served (the pull is closed: the model is told nothing of it); the frames after the failed record
    are not served. Regression caught: a store that stopped learning with the case going on, or a failure in the feed's
    thread that the runner never hears of."""
    # the feed's messages: 1 start, 2 task, 3 the first frame, 4 the first record
    done = play(tmp_path, monkeypatch, {0: CARRY}, memory(tmp_path / "spec", carry=True, fault="exit:4"), status=1)
    (failed,) = [e for e in done.journal if e.get("event") == "memory_failed"]
    assert failed["message"] == "record" and "ended (exit 3) on 'record'" in failed["error"]
    assert done.summary["status"].endswith("TransportError: memory")
    (only,) = done.processes
    assert [c["text"].split(" ")[0] for c in pulled(only) if c["text"].startswith("[k")] == ["[k0"]
