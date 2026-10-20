"""A memory that carries its record on the pull (`codex_memory` SPEC key `carry`, `agent_tools.tools(True)`,
`codex_feed.Feed(carry=)`, `codex_system`, `codex_log`): `next_frame` takes an optional `memory`, the harness writes
`{"record": memory}` to the memory's sensor feed before it counts the pull, and the feed's reply is the last text block
of the item that pull gets. Driven in process by threads that play the model for the pull (`Feed`), and over the fake
command (`codex_fake`) and a tiny sensor feed (`memory_fake_sense`) for the case; no model, no `codex`, no network.
Expected values are written by hand. The invariant that matters most is that an arm without `carry` is sent what it
always was (`test_a_memory_changes_no_message_the_model_is_sent`, in `test_codex_memory`, and the no-carry cases here).
"""

from __future__ import annotations

import copy
import io
import json
import threading
import time
from types import SimpleNamespace

import pytest

from codex_kit import CREDIT, Fake, Run, memory, pulled, run_exec
from stbench import agent_tools, codex_mcp
from stbench.agent_tools import NEXT, SUBMIT, TOOLS, tools, unanswered
from stbench.codex_exec import CodexError
from stbench.codex_feed import BUSY, CLOSED, Feed, Item
from stbench.codex_log import read_log, transcripts
from stbench.codex_memory import MemoryFailure, load
from stbench.llm_scan import scan_log
from stbench.stream_protocol import TransportError

SOON = 20.0  # seconds a wait may take before a test calls it a hang
FIRST, BRIEF = "2026-03-02T08:00:00", "Your assignment is to clean up the café."
COUNT = {"qid": "a1b2c3d4e5f60718", "text": "How many frames have you seen?", "answer_type": "count"}
POSITION = {"qid": "0f1e2d3c4b5a6978", "text": "Where is the robot now?", "answer_type": "position"}
RECORD = {"photo": "k{k}", "record": {"label": "mug", "note": "frame {k}"}, "request_id": "r{k}"}
CARRY = {"next_memory": RECORD}  # the fake model records every frame it is shown, on its next pull
REPLY = '{"ok":true,"receipt":{"n":%d}}'  # what the fake sensor feed answers a record, compact
MEMORY_DESCRIPTION = (
    "Your memory record about the frame you were just shown, as the memory's claim takes it ({photo, record, "
    "request_id}), instead of a claim call; the memory's reply comes at the end of the next item.")


def deadline(seconds: float = SOON) -> float:
    return time.monotonic() + seconds


def text(value: str) -> dict:
    return {"content": [{"type": "text", "text": value}], "isError": True}


# -- the SPEC and the tools ------------------------------------------------------------------------------------------
OMIT = object()  # a SPEC key left out


def spec(folder, **changes):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "guide.md").write_text("How to use it.\n", encoding="utf-8")
    body = {"name": "mem", "serve": ["python", "serve.py", "{store}"], "sense": ["python", "sense.py", "{store}"],
            "prompt": "guide.md", **changes}
    (folder / "memory.json").write_text(json.dumps({k: v for k, v in body.items() if v is not OMIT}), encoding="utf-8")
    return folder / "memory.json"


def test_a_spec_may_carry_and_a_memory_without_it_is_recorded_as_before(tmp_path):
    """Contract (`codex_memory.load`): `carry` is an optional bool (default false) that needs a sense feed, since the
    record is written to it; the journal's record of a memory holds `carry` only when it is on, so the run of a memory
    without it is the record it always was. Regression caught: a `carry` that is a string, a number or null (truthy or
    falsy and accepted), a carrying memory with nowhere to write, or a new key in every old memory's record."""
    plain, carrying = load(spec(tmp_path / "a")), load(spec(tmp_path / "b", carry=True))
    assert (plain.carry, carrying.carry) == (False, True) and "carry" not in plain.record()
    assert carrying.record()["carry"] is True and load(spec(tmp_path / "c", carry=False)).record() == plain.record()
    for n, bad in enumerate(("yes", 1, None, [])):
        with pytest.raises(MemoryFailure, match="carry is true or false"):
            load(spec(tmp_path / f"d{n}", carry=bad))
    with pytest.raises(MemoryFailure, match="carry needs a sense feed"):
        load(spec(tmp_path / "f", carry=True, sense=OMIT))


def test_the_tools_of_a_carrying_arm_are_the_same_two_and_a_memory_argument_and_other_arms_are_unchanged():
    """Contract (`agent_tools.tools`, `codex_mcp`): without carry the tools are `TOOLS` itself (the reference agent's,
    `next_frame` with no arguments and `submit_answer`), whatever the relay's configuration lacks; with carry
    (`"carry": true` in the relay's configuration) they are the same two, `next_frame` differing only in an optional
    `memory` object with the description below and `submit_answer` untouched; and a call of a tool that is in neither is
    still an unknown-tool error. Regression caught: the memory argument offered to an arm that does not carry (the
    model would send records nobody writes), or a changed description or schema of the tool every arm is sent."""
    assert tools(False) is TOOLS and tools(True) != TOOLS and tools(True)[1] is SUBMIT
    carrying = tools(True)[0]["function"]
    assert (carrying["name"], carrying["description"]) == (NEXT["function"]["name"], NEXT["function"]["description"])
    assert carrying["parameters"] == {"type": "object", "properties": {
        "memory": {"type": "object", "description": MEMORY_DESCRIPTION}}}
    assert "required" not in carrying["parameters"] and agent_tools.MEMORY["description"] == MEMORY_DESCRIPTION
    listing = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    for config, schema in (({}, NEXT["function"]["parameters"]), ({"carry": False}, NEXT["function"]["parameters"]),
                           ({"carry": True}, carrying["parameters"])):
        out = io.StringIO()
        codex_mcp.serve(config, io.StringIO(json.dumps(listing) + "\n"), out)
        listed = json.loads(out.getvalue())["result"]["tools"]
        assert [t["name"] for t in listed] == ["next_frame", "submit_answer"] and listed[0]["inputSchema"] == schema
        assert listed[1]["inputSchema"] == SUBMIT["function"]["parameters"]


# -- the pull ---------------------------------------------------------------------------------------------------------
class Carry:
    """A `carry` for a feed: it keeps each record it is handed and holds it until `gate` is set (not at all when the
    gate starts set), and answers `{"ok": true, "receipt": {"n": N}}`."""

    def __init__(self, gate: bool = True):
        self.sent: list = []
        self.entered, self.gate = threading.Event(), threading.Event()
        if gate:
            self.gate.set()

    def __call__(self, memory: dict) -> dict:
        self.sent.append(memory)
        self.entered.set()
        assert self.gate.wait(SOON)
        return {"ok": True, "receipt": {"n": len(self.sent)}}


def pull(feed: Feed, memory=None) -> tuple[threading.Thread, list[dict]]:
    """A thread that makes one `next_frame` call (with `memory`, when given) and keeps the reply."""
    box: list[dict] = []
    arguments = {} if memory is None else {"memory": memory}
    thread = threading.Thread(target=lambda: box.append(feed.call("next_frame", arguments)), daemon=True)
    thread.start()
    return thread, box


def calls(events: list[dict]) -> list[dict]:
    return [e for e in events if e["event"] == "call"]


def test_the_record_is_carried_before_the_pull_is_counted_and_its_reply_ends_the_next_item():
    """Contract (`codex_feed`): a pull with a `memory` hands it to `carry` once the pull is accepted and counts the
    arrival only after the reply is back, so the runner, which waits for the arrival, cannot use the memory's sense pipe
    while the record is on it; a second pull meanwhile is `BUSY` and carries nothing. The item the pull returns is its
    own blocks unchanged and in order and then one text block, `<name>: <compact JSON>`, the reply; the `served` event's
    `text` is the item's own and gains `memory_reply`; the `call` event keeps the arguments and gains the reply object.
    Regression caught: a record written while the runner uses the pipe (interleaved lines, or replies paired with the
    wrong message), a reply that replaces or precedes the item's blocks, or served text that holds the reply."""
    events: list[dict] = []
    carry = Carry(gate=False)
    feed = Feed(events.append, carry=carry, carry_name="mem")
    try:
        ticket = feed.serve_in_prompt(Item("task", "T0"))
        first, box = pull(feed, {"photo": "k0", "record": {"label": "mug"}})
        assert carry.entered.wait(SOON)
        assert feed.wait(ticket, deadline(0.4), lambda: True) == "timeout"  # the record is on the pipe: not pulled yet
        assert feed.call("next_frame", {"memory": {"photo": "k9"}}) == text(BUSY)
        assert carry.sent == [{"photo": "k0", "record": {"label": "mug"}}]  # the refused pull carried nothing
        carry.gate.set()
        assert feed.wait(ticket, deadline(), lambda: True) == "pulled"
        feed.supply(Item("frame", "[k0 c] h", b"\xff\xd8\xff\xe0jpg\xff\xd9", k=0, ask={"when": "during"},
                         types={}, asked="Q1: ?"))
        first.join(SOON)
    finally:
        feed.close()
    (reply,) = box
    assert [b["type"] for b in reply["content"]] == ["text", "image", "text", "text"] and not reply["isError"]
    assert [b["text"] for b in reply["content"] if b["type"] == "text"] == [
        "[k0 c] h", "Q1: ?", 'mem: {"ok":true,"receipt":{"n":1}}']
    served = [e for e in events if e["event"] == "served"][-1]
    assert served["text"] == "[k0 c] h" and served["memory_reply"] == 'mem: {"ok":true,"receipt":{"n":1}}'
    pulls = calls(events)
    made = next(e for e in pulls if "item" in e)
    busy = next(e for e in pulls if "item" not in e)
    assert made["arguments"] == {"memory": {"photo": "k0", "record": {"label": "mug"}}}
    assert made["memory_reply"] == {"ok": True, "receipt": {"n": 1}} and "result" not in made
    assert "memory_reply" not in busy and busy["result"] == BUSY


def test_a_pull_the_feed_refuses_or_the_job_has_ended_carries_nothing():
    """Contract (`codex_feed`): the first pull made while questions are unanswered is refused with the nudge and carries
    nothing (the second goes through and does); after the end item a pull answers the end again, carries nothing and the
    journal has a `memory_dropped` event (`end`), not a key of the `call` event; the end item that a carrying pull gets
    is served with the reply of the record it carried, like any item. Regression caught: a record written for a pull
    that was refused (the model sends it again on the pull that goes through: a sighting twice), or a record written
    after the store is closed."""
    events: list[dict] = []
    carry = Carry()
    feed = Feed(events.append, carry=carry, carry_name="mem")
    ask = Item("ask", "Q1: ?", k=1, ask={"when": "end", "now_k": 1, "labels": {"Q1": "qa"}}, types={"Q1": "count"})
    try:
        ticket = feed.serve_in_prompt(Item("task", "T0"))
        opened, _ = pull(feed)
        assert feed.wait(ticket, deadline(), lambda: True) == "pulled"
        feed.supply(ask)
        opened.join(SOON)
        early = feed.call("next_frame", {"memory": {"photo": "k1"}})
        assert early["isError"] and early["content"][0]["text"] == unanswered(["Q1"]) and carry.sent == []
        going, box = pull(feed, {"photo": "k1"})  # the second premature pull fails the label and goes through
        assert carry.entered.wait(SOON) and feed.wait(1, deadline(), lambda: True) == "pulled"
        feed.supply(Item("end", "over"))
        going.join(SOON)
        again = feed.call("next_frame", {"memory": {"photo": "k2"}})
    finally:
        feed.close()
    assert carry.sent == [{"photo": "k1"}]
    assert [b["text"] for b in box[0]["content"]] == ["over", 'mem: {"ok":true,"receipt":{"n":1}}']
    assert again == {"content": [{"type": "text", "text": "over"}], "isError": False}
    refused, carried, dropped = [e for e in calls(events) if e["tool"] == "next_frame"][-3:]
    assert refused["premature"] and "memory_reply" not in refused
    assert carried["memory_reply"] == {"ok": True, "receipt": {"n": 1}} and carried["premature"]
    assert "memory_dropped" not in dropped and "memory_reply" not in dropped and dropped["over"]
    (gone,) = [e for e in events if e["event"] == "memory_dropped"]
    assert gone["reason"] == "end" and "memory_reply" not in gone


def test_without_carry_a_memory_argument_sends_nothing_and_changes_nothing():
    """Contract (`codex_feed`): a feed with no `carry` ignores a `memory` argument: the item is its own blocks, the
    `call` event has no `memory_reply`, the journal no `memory_dropped`, and the relay's configuration has no `carry`
    key (so its tool list is the reference agent's). Regression caught: a record written for an arm that does not
    carry, or a new key in the configuration every arm is launched with."""
    events: list[dict] = []
    feed = Feed(events.append)
    carrying = Feed(carry=Carry())
    plain, carries = feed.server_config(), carrying.server_config()
    carrying.close()
    try:
        ticket = feed.serve_in_prompt(Item("task", "T0"))
        thread, box = pull(feed, {"photo": "k0"})
        assert feed.wait(ticket, deadline(), lambda: True) == "pulled"
        feed.supply(Item("frame", "F1", k=1))
        thread.join(SOON)
    finally:
        feed.close()
    assert box[0]["content"] == [{"type": "text", "text": "F1"}]
    assert set(plain) == {"address", "key"} and carries["carry"] is True
    (event,) = calls(events)
    assert event["arguments"] == {"memory": {"photo": "k0"}} and "memory_reply" not in event
    assert all(e["event"] != "memory_dropped" for e in events)
    assert "memory_reply" not in [e for e in events if e["event"] == "served"][-1]


def test_a_carry_that_raises_is_an_error_reply_and_an_exec_that_ends_meanwhile_drops_the_reply():
    """Contract (`codex_feed`): a pull never raises for a request: a `carry` that raises is the reply `{"ok": false,
    "error": <type>, "message"}` in the item, and the pull completes; a pull whose exec ends while its record is
    carried (`begin_exec`) is no call of the model's, answers the closed error, and its reply, which nobody reads, is
    journaled as `memory_dropped` (`abandoned`). Regression caught: a tool server whose call dies on a failing memory
    (the model hears nothing), or a reply lost without a trace."""
    events: list[dict] = []

    def broken(memory):
        raise ValueError("no store")

    feed = Feed(events.append, carry=broken, carry_name="mem")
    try:
        ticket = feed.serve_in_prompt(Item("task", "T0"))
        thread, box = pull(feed, {"photo": "k0"})
        assert feed.wait(ticket, deadline(), lambda: True) == "pulled"
        feed.supply(Item("frame", "F1", k=1))
        thread.join(SOON)
    finally:
        feed.close()
    assert box[0]["content"][-1]["text"] == 'mem: {"ok":false,"error":"ValueError","message":"no store"}'
    gone, carry = [], Carry(gate=False)
    feed = Feed(gone.append, carry=carry, carry_name="mem")
    try:
        feed.serve_in_prompt(Item("task", "T0"))
        thread, box = pull(feed, {"photo": "k0"})
        assert carry.entered.wait(SOON)
        feed.begin_exec()
        carry.gate.set()
        thread.join(SOON)
    finally:
        feed.close()
    assert box[0] == text(CLOSED) and calls(gone) == []
    (dropped,) = [e for e in gone if e["event"] == "memory_dropped"]
    assert dropped["reason"] == "abandoned" and dropped["memory_reply"] == {"ok": True, "receipt": {"n": 1}}


# -- the case ---------------------------------------------------------------------------------------------------------
def calls_of(log: list[dict]) -> list[dict]:
    """The `call` events of `next_frame`, in the order they were journaled."""
    return [e for e in log if e.get("event") == "call" and e["tool"] == "next_frame"]


def play(run: Run) -> None:
    """Six frames in two tasks, a during ask at frame 2 (it joins that frame) and the end ask at frame 5."""
    system = run.system
    system.task(0, FIRST, 0.0, BRIEF)
    run.stream([0, 1, 2])
    system.ask("during", {"k": 2, "clock": "2026-03-02T08:00:02"}, [COUNT])
    run.stream([3])
    system.task(1, "2026-03-02T08:10:00", 596.0, "Now sweep the hall.")
    run.stream([4, 5])
    system.ask("end", {"k": 5, "clock": "2026-03-02T08:00:05"}, [POSITION])
    system.finish()


def messages(run: Run) -> list[dict]:
    path = run.state / "mem" / "sense.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def case(tmp_path, plan: dict | None, held=None, **opts) -> SimpleNamespace:
    """The case of `play` over the fake, with the memory `held` (none: the reference agent), and what it left."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    with Run(tmp_path, {0: plan} if plan is not None else None, memory=held, asks_during=True, **CREDIT, **opts) as run:
        play(run)
    sensed = [m for m in messages(run) if "message" in m] if held else []
    return SimpleNamespace(execs=run.fake.execs(), log=read_log(run.log), sensed=sensed, run=run)


@pytest.fixture(scope="module")
def played(tmp_path_factory):
    """The case with a carrying memory whose model records every frame it is shown, and the reference agent's."""
    base = tmp_path_factory.mktemp("carry")
    carrying = case(base / "carry", CARRY, memory(base / "carry" / "spec", carry=True))
    return SimpleNamespace(carrying=carrying, reference=case(base / "reference", None))


def test_each_record_follows_its_frame_into_the_feed_and_is_written_before_the_runner_goes_on(played):
    """Contract (`codex_system._carry`, `codex_feed`): a frame is fed before it is held, the record about it is written
    while the frame is served (the model's next pull) and before the runner's next message, and each frame's is one line
    `{"record": <memory as the model gave it>}` in the feed's one stream, replied `{"ok": true, "receipt": {...}}`.
    Frames 0..5 of the case and the tasks make this order: each frame is fed as it is served and the record about it
    follows at the model's next pull (frame 2 is served with its questions, frame 3 alone when the second task comes,
    which is then fed after it, and frame 5 with the end ask). Regression caught: a record
    written while the runner writes an observe (the one pipe shared), written twice, or never."""
    order = [(next(iter(m["message"])), m["message"].get("observe", {}).get("steps", [{}])[0].get("k"))
             for m in played.carrying.sensed]
    assert order == [("start", None), ("task", None), ("observe", 0), ("record", None), ("observe", 1),
                     ("record", None), ("observe", 2), ("record", None), ("observe", 3), ("record", None),
                     ("task", None), ("observe", 4), ("record", None), ("observe", 5), ("record", None)]
    records = [m for m in played.carrying.sensed if "record" in m["message"]]
    assert [m["message"] for m in records] == [
        {"record": {"photo": f"k{k}", "record": {"label": "mug", "note": f"frame {k}"}, "request_id": f"r{k}"}}
        for k in range(6)]
    assert [m["n"] for m in records] == [4, 6, 8, 10, 13, 15] and len({m["pid"] for m in played.carrying.sensed}) == 1


def test_the_reply_is_the_last_block_of_the_next_item_and_nothing_else_the_model_is_sent_changes(played):
    """Contract (`codex_feed`, DECISIONS: a memory changes no message the model is sent): the prompts of the execs and
    every item the model is served are the reference agent's, byte for byte, but that the pull after a frame item (the
    fake model records each frame it is shown) is served the item with `<name>: <reply>` added as its last text block,
    on its own line, and nothing is added to a pull that follows anything else (the first pull, a task). Regression
    caught: the reply placed before the item's blocks or merged into its text, a reply on a pull that carried no record,
    or any change to a message of an arm that carries."""
    ours, theirs = played.carrying.execs, played.reference.execs
    assert [e["prompt"] for e in ours] == [e["prompt"] for e in theirs]
    after_frame, receipts = False, iter([4, 6, 8, 10, 13, 15])
    for mine, ref in zip(pulled(ours[0]), pulled(theirs[0]), strict=True):
        if after_frame:
            tail = "mem: " + REPLY % next(receipts)
            assert mine["text"] == ref["text"] + "\n" + tail and mine["blocks"] == [*ref["blocks"], "text"]
        else:
            assert (mine["text"], mine["blocks"]) == (ref["text"], ref["blocks"])
        assert mine["images"] == ref["images"]
        after_frame = ref["text"].startswith("[k")
    assert next(receipts, None) is None and len(pulled(ours[0])) == len(pulled(theirs[0]))


def test_the_journal_keeps_the_record_and_the_reply_and_the_served_text_stays_the_items_own(played):
    """Contract (`codex_log`): the `call` event of a carrying `next_frame` holds `arguments.memory` and `memory_reply`
    (the reply object); the `served` event of the item keeps `text` unchanged and gains `memory_reply` (the text the
    model read), and an item whose pull carried nothing has none; `transcripts` ends the message the model held with
    that block, after the item's image and any questions that came with it. Regression caught: a journal that cannot
    say what the memory answered (the replay cannot tell a refused record), or a reply folded into the served text."""
    log, reference = played.carrying.log, played.reference.log
    carried = [e for e in calls_of(log) if "memory" in e["arguments"]]
    assert [e["arguments"]["memory"]["photo"] for e in carried] == [f"k{k}" for k in range(6)]
    assert [e["memory_reply"] for e in carried] == [{"ok": True, "receipt": {"n": n}} for n in (4, 6, 8, 10, 13, 15)]
    served, bare = [e for e in log if e.get("event") == "served"], [e for e in reference if e.get("event") == "served"]
    assert [e["text"] for e in served] == [e["text"] for e in bare] and [e["images"] for e in served] == [
        e["images"] for e in bare]
    # task 0 (prompt), frame 0, frame 1, frame 2 with its questions, frame 3, task 1, frame 4, frame 5 with its
    # questions, the end: each reply is to the record the pull that got the item carried (the first pull, and the one
    # after a task, carry none)
    mem = ["mem: " + REPLY % n for n in (4, 6, 8, 10, 13, 15)]
    replies = [None, None, *mem[:4], None, *mem[4:]]
    assert [e.get("memory_reply") for e in served] == replies
    held = [m for m in transcripts(log)[0]["request"]["messages"] if m["role"] == "tool"]
    during = next(m for m in held if isinstance(m["content"], list) and len(m["content"]) == 4)
    assert [b["type"] for b in during["content"]] == ["text", "image_url", "text", "text"]
    assert during["content"][-1] == {"type": "text", "text": "mem: " + REPLY % 6}
    assert during["content"][2]["text"] == served[3]["asked"] and during["content"][0]["text"] == served[3]["text"]


def test_the_leak_scan_reads_the_replies_a_memory_carried(played):
    """Contract (`codex_log.transcripts`, `llm_scan`): the reply of a carried record is part of what the model was
    shown, so a question's text in one, before that question's ask, is a hit, as it is in an item the feed served; the
    run as played has none. Regression caught: carried replies invisible to the scan (a store could carry a future
    question to the model through the one channel the scan does not read)."""
    questions = [COUNT, POSITION]
    assert scan_log(played.carrying.run.log, [], questions=questions) == []
    records = copy.deepcopy(played.carrying.log)
    planted = next(r for r in records if r.get("event") == "served" and r.get("memory_reply"))
    planted["memory_reply"] += f" {POSITION['text']}"
    path = played.carrying.run.log.with_name("planted.jsonl")
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    hits = scan_log(path, [], questions=questions)
    assert hits and all("(tool)" in hit and f"question {POSITION['qid']} before its ask" in hit for hit in hits), hits


def test_an_arm_without_carry_is_not_written_to_whatever_the_model_sends(tmp_path):
    """Contract (`codex_system`, `codex_feed`): the fake model sends a `memory` on every pull after a frame, but a
    memory without `carry` is offered the reference agent's tools and writes nothing: the feed sees no record, the
    server's configuration has no `carry`, the items are the reference agent's byte for byte and the `call` events have
    no reply. Regression caught: a record sent to a feed whose arm did not ask for it (a second factor in the arm)."""
    ref = case(tmp_path / "ref", None)
    plain = case(tmp_path / "plain", CARRY, memory(tmp_path / "plain" / "spec"))
    assert not [m for m in plain.sensed if "record" in m["message"]]
    assert [m for m in plain.sensed if next(iter(m["message"])) != "record"]
    assert [(c["text"], c["blocks"]) for c in pulled(plain.execs[0])] == [
        (c["text"], c["blocks"]) for c in pulled(ref.execs[0])]
    assert "carry" not in plain.execs[0]["config"] and set(plain.execs[0]["config"]) == {"address", "key"}
    assert all("memory_reply" not in e for e in calls_of(plain.log))
    assert not any("memory_reply" in e or e.get("event") == "memory_dropped" for e in plain.log)


def test_the_pull_the_feed_refuses_writes_no_record(tmp_path):
    """Contract (`codex_feed`) at the case: a model that pulls early on an ask (sending its record each time) is refused
    once with the nudge and carries nothing on that pull; one record is written for each frame, six, not seven.
    Regression caught: the refused pull's record written too (the same sighting twice in the store)."""
    done = case(tmp_path, {**CARRY, "premature": 1}, memory(tmp_path / "spec", carry=True))
    assert len([m for m in done.sensed if "record" in m["message"]]) == 6
    early = [e for e in calls_of(done.log) if e.get("premature") and e["error"]]
    assert len(early) == 2 and all("memory_reply" not in e for e in early)  # one for each ask


def test_a_feed_that_fails_on_a_record_ends_the_case_as_memory(tmp_path):
    """Contract (`codex_system._carry`): a sensor feed that dies while a record is written is journaled as
    `memory_failed` (message `record`) and the runner ends the case as `TransportError("memory")` at its next check,
    caused by the `MemoryFailure`, before any item is served (the pull is closed: the model is told nothing of it); the
    frame after the failed record is not served. Regression caught: a store that stopped learning with the case going
    on, or a failure in the feed's thread that the runner never hears of."""
    # the feed's messages: 1 start, 2 task, 3 the first frame (fed as observe of the next serves it), 4 the first record
    held = memory(tmp_path / "spec", carry=True, fault="exit:4")
    with Run(tmp_path, {0: CARRY}, memory=held) as run:
        run.system.task(0, FIRST, 0.0, BRIEF)
        run.stream([0])
        with pytest.raises(TransportError, match="^memory$") as ended:
            run.system.observe(run.batch([1]))
    assert isinstance(ended.value.__cause__, MemoryFailure)
    (failed,) = [e for e in read_log(run.log) if e.get("event") == "memory_failed"]
    assert failed["message"] == "record" and "ended (exit 3) on 'record'" in failed["error"]


# -- the fake codex ---------------------------------------------------------------------------------------------------
def test_the_fake_codex_sends_next_memory_after_frames_and_logs_what_each_server_said_on_initialize(tmp_path):
    """Contract (`codex_fake`): the plan key `next_memory` is a `memory` argument, `{k}` in its string values replaced
    by the number of the frame just shown, on every `next_frame` that follows a frame item and on no other; and
    `execs.jsonl` has `server_instructions`, the `instructions` of each server's `initialize` reply by name (null for
    one that sent none), `{}` for an exec that failed before it launched them. Regression caught: a fake that does not
    exercise the carry (every test above would pass on nothing), or a core that cannot be checked at the host."""
    done = case(tmp_path / "a", CARRY, memory(tmp_path / "a" / "spec", instructions="the core"))
    (first,) = done.execs
    photos = [c["arguments"].get("memory", {}).get("photo") for c in pulled(first)]
    assert photos == [None, "k0", "k1", "k2", "k3", None, "k4", "k5"]  # none before a frame or after a task
    assert pulled(first)[1]["arguments"] == {"memory": {
        "photo": "k0", "record": {"label": "mug", "note": "frame 0"}, "request_id": "r0"}}
    assert first["server_instructions"] == {"stbench": None, "mem": "the core"}
    (tmp_path / "b").mkdir()
    fake = Fake(tmp_path / "b", {0: {"fail": "limit"}})
    with pytest.raises(CodexError, match="usage limit"):
        run_exec(fake)
    assert fake.execs()[0]["server_instructions"] == {}
