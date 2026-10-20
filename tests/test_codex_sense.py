"""The memory's sensor feed pipe (`codex_memory.Sense`): the line protocol and the one-request-at-a-time rule.
Model-free: a fake feed process (`memory_fake_sense`), no `codex`, no network, no real memory system."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

from stbench.codex_memory import Sense

SOON = 20.0  # seconds a wait may take before a test calls it a hang


def test_the_feed_speaks_json_lines_whole_and_ends_when_closed(tmp_path):
    """Contract (`codex_memory.Sense`): each message is one ASCII line (non-ASCII text escaped, so the encoding of its
    stdin is no matter), the reply is the one JSON object line it printed, replies pair with messages in order, and
    closing ends its input and waits for it to end. Regression caught: replies read one message late, a message split
    over two lines, or a feed that outlives its case."""
    store = tmp_path / "store"
    feed = Sense([sys.executable, str(Path(__file__).with_name("memory_fake_sense.py")), str(store)])
    try:
        replies = [feed.send({"task": {"brief": f"ünï {n} — \n two lines"}}) for n in range(3)]
    finally:
        feed.close()
    assert replies == [{"ok": True, "n": n} for n in (1, 2, 3)]
    written = [json.loads(line) for line in (store / "sense.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [m["message"]["task"]["brief"] for m in written[:3]] == [f"ünï {n} — \n two lines" for n in range(3)]
    assert written[-1] == {"pid": written[0]["pid"], "closed": True}


def test_two_threads_sending_at_once_each_get_their_own_reply(tmp_path):
    """Contract (`codex_memory.Sense.send`): a request and its reply are one turn, so a reply belongs to its request
    whatever the order in which threads wait for it (a carried record being written when the runner sends `observe`).
    The test makes a second reader that arrives while the first waits read first; that can happen only if the pipe
    has no lock. Regression caught: replies paired with requests by the order of the waiting threads."""
    feed = Sense([sys.executable, str(Path(__file__).with_name("memory_fake_sense.py")), str(tmp_path / "store")])
    cond, readers, second_read = threading.Condition(), [], threading.Event()

    class Crossed:
        """The reply queue: a first reader waits (briefly) for a second, which then reads before it."""

        def __init__(self, inner):
            self.inner = inner

        def put(self, line):
            self.inner.put(line)

        def get(self, timeout):
            with cond:
                readers.append(threading.current_thread())
                cond.notify_all()
                first = len(readers) == 1
                crossed = first and cond.wait_for(lambda: len(readers) > 1, timeout=0.5)
            if crossed:
                assert second_read.wait(SOON)
            line = self.inner.get(timeout=timeout)
            if not first:
                second_read.set()
            return line

    feed._lines = Crossed(feed._lines)
    replies: dict[str, dict] = {}
    first = threading.Thread(target=lambda: replies.update(a=feed.send({"task": {"brief": "a"}})))
    second = threading.Thread(target=lambda: replies.update(b=feed.send({"task": {"brief": "b"}})))
    try:
        first.start()
        with cond:
            assert cond.wait_for(lambda: readers, timeout=SOON)  # the first has written and waits for its reply
        second.start()
        first.join(SOON)
        second.join(SOON)
    finally:
        feed.close()
    assert replies == {"a": {"ok": True, "n": 1}, "b": {"ok": True, "n": 2}}
