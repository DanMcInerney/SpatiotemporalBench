"""The harness end of the pull (`codex_feed`): what the model's `next_frame` and `submit_answer` calls reach, driven in
process by threads that play the model (no `codex`, no process, no model), and over the real connection once.

Expected values are written by hand. The model is a thread that calls `Feed.call` as the tool server would; the runner
is the test's own thread, which supplies items and waits as `CodexSystem` does.
"""

from __future__ import annotations

import base64
import hashlib
import threading
import time

import pytest

from stbench.agent_tools import NO_QUESTION, unanswered
from stbench.codex_feed import BUSY, CLOSED, MAX_CALLS, Feed, Item
from stbench.codex_mcp import Relay

SOON = 20.0  # seconds a wait may take before a test calls it a hang


def deadline(seconds: float = SOON) -> float:
    return time.monotonic() + seconds


class Model:
    """A thread that plays the model: it makes `calls` (tool, arguments) in order and keeps each reply."""

    def __init__(self, feed: Feed, calls: list[tuple[str, dict]], first: bool = False):
        self.replies: list[dict] = []
        self.feed, self.calls = feed, calls
        # the first task goes in the prompt, before any exec starts: serve it before the model can pull
        self.ticket = feed.serve_in_prompt(Item("task", "T0")) if first else None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def open(self) -> None:
        """Wait until the model's first pull is open, as the runner does after launching the exec."""
        assert self.feed.wait(self.ticket, deadline(), lambda: True) == "pulled"

    def run(self) -> None:
        for name, arguments in self.calls:
            self.replies.append(self.feed.call(name, arguments))

    def texts(self) -> list[str]:
        return [reply["content"][0]["text"] for reply in self.replies]

    def join(self) -> None:
        self.thread.join(SOON)
        assert not self.thread.is_alive(), "the model is still waiting"


def pulls(n: int) -> list[tuple[str, dict]]:
    return [("next_frame", {})] * n


def test_the_model_is_shown_one_item_at_a_time_and_never_the_next_before_it_pulls_again():
    """Contract (FD-frame-pull §3.3): the runner supplies an item and waits for the model's next pull; the model has
    then been shown that item and no later one, even though more are ready, and its next pull stays open until the
    runner supplies another. Items are numbered in the order served, the first (in the prompt) as 0. Regression caught:
    a feed that runs ahead (the model shown frame k+1 with frame k), or a wait that returns before the model is done."""
    events: list[dict] = []
    feed = Feed(events.append)
    model = Model(feed, pulls(4), first=True)
    try:
        model.open()
        assert model.texts() == []  # its first pull is open: nothing to show yet
        for n, text in enumerate(["F1", "F2", "F3"], 1):
            ticket = feed.supply(Item("frame", text, k=n))
            assert ticket == n and feed.wait(ticket, deadline(), lambda: True) == "pulled"
            assert model.texts() == ["F1", "F2", "F3"][:n]  # shown this and the earlier ones, and no more
        assert len(model.replies) == 3
        feed.supply(Item("end", "over"))  # the pull that is open takes it
        model.join()
        assert model.texts() == ["F1", "F2", "F3", "over"]
    finally:
        feed.close()
    served = [(e["item"], e["kind"], e.get("via")) for e in events if e["event"] == "served"]
    assert served == [(0, "task", "prompt"), (1, "frame", None), (2, "frame", None), (3, "frame", None),
                      (4, "end", None)]


def test_a_wait_ends_when_the_exec_ends_or_the_deadline_passes_and_a_pull_wins_over_both():
    """Contract: `wait` returns `exit` once the exec is no longer alive and `timeout` at the deadline, with the item
    left for the next pull (it is taken then, not lost); an item taken and followed by a pull is `pulled` even if the
    exec has ended too. Regression caught: a runner that hangs on an exec that died, a deadline that is not kept, or an
    item dropped when its exec died."""
    feed = Feed()
    try:
        ticket = feed.serve_in_prompt(Item("task", "T0"))
        assert feed.wait(ticket, deadline(), lambda: False) == "exit"
        began = time.monotonic()
        assert feed.wait(ticket, deadline(0.3), lambda: True) == "timeout" and time.monotonic() - began < SOON
        later = feed.supply(Item("frame", "F1", k=1))  # nobody has pulled: it waits
        assert feed.wait(later, deadline(0.2), lambda: True) == "timeout"
        feed.begin_exec()  # a new exec starts, and its first pull takes the item
        model = Model(feed, pulls(2))
        assert feed.wait(later, deadline(), lambda: True) == "pulled"
        assert model.texts() == ["F1"]
    finally:
        feed.close()


def test_a_premature_pull_is_refused_once_and_the_second_fails_the_labels_and_goes_through():
    """Contract (FD-frame-pull §3.3): during an ask the first `next_frame` with labels unsubmitted answers an error
    naming them, and the second fails them `no_submit` and returns the next item; the ask is over for the runner at
    that second pull and not before; both calls are journaled as premature and the nudged labels are kept. Regression
    caught: a stream that skips an ask whose questions were never answered, a nudge that never ends, or a runner
    released by the refused call."""
    events: list[dict] = []
    feed = Feed(events.append)
    ask = Item("ask", "Q1: ...", k=3, ask={"when": "during", "now_k": 3, "labels": {"Q1": "qa", "Q2": "qb"}},
               types={"Q1": "count", "Q2": "count"})
    late = ("submit_answer", {"label": "Q1", "answer": {"type": "count", "value": 1}})
    model = Model(feed, [*pulls(3), late], first=True)
    try:
        model.open()
        ticket = feed.supply(ask)
        assert feed.wait(ticket, deadline(), lambda: True) == "pulled"  # the second premature pull went through
        table = feed.close_ask()
        assert table.nudged == ["Q1", "Q2"] and table.premature == 2
        assert {label: (v["status"], v["reason"]) for label, v in table.table.items()} == {
            "Q1": ("failed", "no_submit"), "Q2": ("failed", "no_submit")}
        feed.supply(Item("frame", "F4", k=4))  # the open pull gets the next item
        model.join()
    finally:
        feed.close()
    assert model.texts() == ["Q1: ...", unanswered(["Q1", "Q2"]), "F4", NO_QUESTION]
    pulled = [e for e in events if e["event"] == "call" and e["tool"] == "next_frame"]
    assert [bool(e.get("premature")) for e in pulled] == [False, True, True] and pulled[1]["error"]
    assert "item" in pulled[2] and "result" in pulled[1] and "result" not in pulled[2]


def test_an_answered_ask_ends_at_the_next_pull_and_a_late_submission_is_no_question():
    """Contract: with every label answered the next `next_frame` is not premature and ends the ask; the answers are the
    table `close_ask` returns, and a `submit_answer` after the ask has ended is `NO_QUESTION` and changes nothing.
    Regression caught: a nudge for a complete ask, or an answer that reaches a table the runner has already read."""
    feed = Feed()
    ask = Item("ask", "Q1: ...", k=1, ask={"when": "end", "now_k": 1, "labels": {"Q1": "qa"}}, types={"Q1": "count"})
    answer = {"label": "Q1", "answer": {"type": "count", "value": 5}}
    model = Model(feed, [("next_frame", {}), ("submit_answer", answer), ("next_frame", {}),
                         ("submit_answer", {"label": "Q1", "answer": {"type": "count", "value": 9}})], first=True)
    try:
        model.open()
        assert feed.wait(feed.supply(ask), deadline(), lambda: True) == "pulled"
        table = feed.close_ask()
        assert table.table == {"Q1": {"status": "submitted", "answer": {"type": "count", "value": 5}, "bad": 0}}
        assert table.nudged == [] and table.premature == 0
        feed.supply(Item("end", "over"))
        model.join()
    finally:
        feed.close()
    assert model.texts() == ["Q1: ...", "ok", "over", NO_QUESTION]


def test_a_second_next_frame_while_one_is_open_is_refused():
    """Contract (FD-frame-pull §3.3): one `next_frame` at a time; a second call while one waits answers an error, and
    the one that waits is not disturbed. Regression caught: two pulls that both take items, or a second call that
    blocks behind the first and reorders the stream."""
    feed = Feed()
    model = Model(feed, pulls(1), first=True)
    try:
        model.open()
        second = feed.call("next_frame", {})
        assert second == {"content": [{"type": "text", "text": BUSY}], "isError": True}
        feed.supply(Item("frame", "F1", k=1))
        model.join()
        assert model.texts() == ["F1"]
    finally:
        feed.close()


def test_calls_past_the_cap_between_two_items_are_refused_except_the_pull_and_the_count_restarts_at_an_item():
    """Contract (FD-frame-pull §3.3): at most `max_calls` tool calls between two items; past it a call answers the cap's
    error and does nothing (here a submission that would have been accepted), while `next_frame` is still served (a
    model told to pull must be able to), and the count starts again when an item is taken. Regression caught: a model
    that loops on its tool for the rest of an item with no bound, or a cap that traps it so that it can never pull."""
    feed = Feed(max_calls=2)
    ask = Item("ask", "Q1: ...", k=1, ask={"when": "end", "now_k": 1, "labels": {"Q1": "qa"}}, types={"Q1": "count"})
    bad = {"label": "Q1", "answer": {"type": "nonsense"}}
    good = {"label": "Q1", "answer": {"type": "count", "value": 2}}
    taken = Model(feed, pulls(1), first=True)
    try:
        taken.open()
        feed.supply(ask)
        taken.join()  # the ask is taken: the count of calls starts at 0
        first = feed.call("submit_answer", bad)
        second = feed.call("submit_answer", {"label": "Q9", "answer": {}})
        capped = feed.call("submit_answer", good)  # the third call
        assert first["content"][0]["text"].startswith("error: ") and second["content"][0]["text"].startswith(
            "error: label must be")
        assert capped == {"content": [{"type": "text", "text": MAX_CALLS}], "isError": True}
        nudge = feed.call("next_frame", {})  # a pull is served past the cap, here with the nudge: `good` did not count
        assert nudge["content"][0]["text"] == unanswered(["Q1"])
        pull = Model(feed, pulls(1))  # the second premature pull goes through and waits
        assert feed.wait(1, deadline(), lambda: True) == "pulled"
        feed.supply(Item("frame", "F2", k=2))  # taken: the count starts again
        pull.join()
        assert feed.call("submit_answer", good)["content"][0]["text"] == NO_QUESTION  # not the cap's error
    finally:
        feed.close()


def test_after_the_end_the_pull_answers_the_end_again_and_an_unknown_tool_is_an_error():
    """Contract: once the end item has been served, a further `next_frame` answers the end at once (a model that pulls
    after the job, or an exec resumed to be asked something, is not left waiting); a tool that does not exist is an
    error naming it. Regression caught: a pull after the end that blocks the exec for ever, or a tool name echoed
    without a limit."""
    feed = Feed()
    try:
        model = Model(feed, pulls(1), first=True)
        model.open()
        feed.supply(Item("end", "over"))
        model.join()
        again = feed.call("next_frame", {})
        assert again == {"content": [{"type": "text", "text": "over"}], "isError": False}
        unknown = feed.call("memory_read", {"path": "a"})
        assert unknown == {"content": [{"type": "text", "text": "error: unknown tool 'memory_read'"}],
                           "isError": True}
    finally:
        feed.close()


def test_a_pull_left_open_by_an_exec_that_ended_is_dropped_when_the_next_exec_begins():
    """Contract: `begin_exec` drops the `next_frame` the old exec left open, so the new exec's first pull is not refused
    as a second one and takes the next item, while the old one's connection gets an error and no item. Regression
    caught: a killed exec's open pull that swallows the item meant for its successor, or that makes the successor's
    first call `BUSY` for ever."""
    feed = Feed()
    old = Model(feed, pulls(1), first=True)
    try:
        old.open()
        feed.begin_exec()
        old.join()
        assert old.texts() == [CLOSED]
        new = Model(feed, pulls(1))
        feed.supply(Item("frame", "F1", k=1))
        new.join()
        assert new.texts() == ["F1"]
    finally:
        feed.close()


def test_closing_the_feed_lets_an_open_pull_go():
    """Contract: `close` releases a `next_frame` that waits (an error, no item), so no thread outlives the case.
    Regression caught: a pull blocked for ever when a case ends with the model mid-call."""
    feed = Feed()
    model = Model(feed, pulls(1), first=True)
    model.open()
    feed.close()
    model.join()
    assert model.texts() == [CLOSED]


def test_the_journal_gets_every_item_and_call_with_images_as_their_hash_and_never_as_base64():
    """Contract (FD-frame-pull §3.3, "never base64 in the journal"): each served item is recorded with its kind, `k`,
    text and image as `<jpeg N bytes sha256 HEX16>`, each call with its tool, arguments, error flag and the monotonic
    and wall time of its arrival, and nothing in either holds the image's bytes or their base64. The reply to the model
    does hold the image, as an MCP image block with its type. Regression caught: a journal that grows by an image a
    frame, or a reply without its image block."""
    events: list[dict] = []
    feed = Feed(events.append)
    image = b"\xff\xd8\xff\xe0frame-7\xff\xd9"
    model = Model(feed, pulls(1), first=True)
    try:
        model.open()
        feed.supply(Item("frame", "[k7 c] header", image, k=7))
        model.join()
    finally:
        feed.close()
    assert model.replies[0]["content"] == [{"type": "text", "text": "[k7 c] header"},
                                           {"type": "image", "data": base64.b64encode(image).decode(),
                                            "mimeType": "image/jpeg"}]
    served = [e for e in events if e["event"] == "served"][1]
    assert (served["kind"], served["k"], served["text"], served["images"]) == (
        "frame", 7, "[k7 c] header", [f"<jpeg 13 bytes sha256 {hashlib.sha256(image).hexdigest()[:16]}>"])
    call = next(e for e in events if e["event"] == "call")
    assert (call["tool"], call["arguments"], call["error"], call["item"]) == ("next_frame", {}, False, 1)
    assert all(isinstance(e[key], float) for e in (served, call) for key in ("at", "time"))
    assert base64.b64encode(image).decode() not in repr(events) and "frame-7" not in repr(events)


def test_a_frame_that_comes_with_questions_is_one_reply_frame_first_and_opens_its_ask():
    """Contract (FD-questions-0.10 §1.5): `Item.with_ask` joins an ask to the frame at its step as one item: the reply
    to the pull is three MCP blocks in order, the frame's text, its image, then the questions' text; taking it opens
    the ask (a `next_frame` with a label unsubmitted gets the nudge, a submission is taken, and the table is what
    `close_ask` returns); the journal has one `served` record, a `frame` with the frame's `k`, text and image hash and
    the ask's `when`, `now_k` and labels and `asked`, its text. It refuses an ask at another step and what is not a
    frame and an ask. Regression caught: questions before the image, a joined item that opens no ask (every answer
    refused as "no question is pending"), a record that loses the questions' text or the frame's image hash, or a
    join of an ask to a frame it is not about."""
    events: list[dict] = []
    feed = Feed(events.append)
    image = b"\xff\xd8\xff\xe0frame-3\xff\xd9"
    frame = Item("frame", "[k3 c] header", image, k=3)
    ask = Item("ask", "Questions during the job.\nQ1: ?", k=3,
               ask={"when": "during", "now_k": 3, "labels": {"Q1": "qa"}}, types={"Q1": "count"})
    joined = frame.with_ask(ask)
    answer = ("submit_answer", {"label": "Q1", "answer": {"type": "count", "value": 5}})
    model = Model(feed, [("next_frame", {}), ("next_frame", {}), answer, ("next_frame", {})], first=True)
    try:
        model.open()
        assert feed.wait(feed.supply(joined), deadline(), lambda: True) == "pulled"
        table = feed.close_ask()
        assert table.nudged == ["Q1"] and table.table["Q1"]["status"] == "submitted"
        feed.supply(Item("end", "over"))
        model.join()
    finally:
        feed.close()
    assert model.replies[0]["content"] == [
        {"type": "text", "text": "[k3 c] header"},
        {"type": "image", "data": base64.b64encode(image).decode(), "mimeType": "image/jpeg"},
        {"type": "text", "text": "Questions during the job.\nQ1: ?"}]
    assert model.texts()[1:] == [unanswered(["Q1"]), "ok", "over"]
    (served,) = [e for e in events if e["event"] == "served" and e["item"] == 1]
    assert (served["kind"], served["k"], served["text"], served["asked"]) == (
        "frame", 3, "[k3 c] header", "Questions during the job.\nQ1: ?")
    assert served["ask"] == {"when": "during", "now_k": 3, "labels": {"Q1": "qa"}}
    assert served["images"] == [f"<jpeg 13 bytes sha256 {hashlib.sha256(image).hexdigest()[:16]}>"]
    for wrong_frame, wrong_ask in ((frame, Item("ask", "Q1: ?", k=4)), (ask, ask), (frame, frame)):
        with pytest.raises(ValueError, match="only an ask at a frame's own step joins it"):
            wrong_frame.with_ask(wrong_ask)


def test_a_tool_server_reaches_the_feed_over_the_connection_with_the_key_and_only_with_it():
    """Contract (FD-frame-pull §3.3: the channel to the tool server): a relay given the feed's address and key sends a
    call and gets the reply's content and error flag back as they are, an image block included; a relay with the wrong
    key, one with no address and one that was never given a feed answer an error text and raise nothing. Regression
    caught: a feed that other local processes can use, a relay that raises into the tool loop, or image bytes mangled
    on the way."""
    feed = Feed()
    image = bytes(range(256))
    relay = Relay(feed.server_config())
    box: list[dict] = []
    puller = threading.Thread(target=lambda: box.append(relay.call("next_frame", {})), daemon=True)
    ticket = feed.serve_in_prompt(Item("task", "T0"))
    try:
        assert relay.call("submit_answer", {"label": "Q1", "answer": {"type": "unknown"}}) == {
            "content": [{"type": "text", "text": NO_QUESTION}], "isError": True}
        puller.start()  # a pull from the relay's connection is open once the runner sees it arrive
        assert feed.wait(ticket, deadline(), lambda: True) == "pulled"
        feed.supply(Item("frame", "F", image, k=0))
        puller.join(SOON)
        assert box == [{"content": [{"type": "text", "text": "F"},
                                    {"type": "image", "data": base64.b64encode(image).decode(),
                                     "mimeType": "image/jpeg"}], "isError": False}]
        wrong = Relay({**feed.server_config(), "key": "00" * 32}).call("next_frame", {})
        assert wrong["isError"] and wrong["content"][0]["text"].startswith("error: the harness cannot be reached")
        for config in ({}, {"address": None, "key": ""}):
            assert Relay(config).call("next_frame", {})["isError"] is True
    finally:
        feed.close()
