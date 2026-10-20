"""A tiny sensor feed standing in for a memory's (`codex_memory`'s `sense`), for the tests of the memory hook.

`python memory_fake_sense.py STORE [FAULT]`: reads one JSON object a line from stdin (UTF-8) until it ends, and for each
writes one JSON line to stdout, `{"ok": true, "n": N}`; for a `record` (a memory that carries, `codex_memory`) it writes
`{"ok": true, "receipt": {"n": N}}`, as a real feed may. For each message it appends one line to `STORE/sense.jsonl`:
the process id, the count, the message and, for an `observe`, whether the image its step names exists at that moment, is
an absolute path and what it holds (the first 16 hex characters of its SHA-256); and when its input ends it appends
`{"closed": true}`. So a test can tell how many processes there were, in what order they were sent what, what the
inbox held while the message was handled and that the process was closed.

FAULT is `KIND:N`, which makes the N-th message (counted from 1) fail as KIND says: `exit` (exit 3 with a line on
stderr, before replying), `eof` (close stdout and exit 0), `text` (reply with a line that is not JSON), `array` (reply
with a JSON array), `hang` (never reply).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

STORE = Path(sys.argv[1])
FAULT = sys.argv[2].partition(":") if len(sys.argv) > 2 else ("", "", "")


def log(entry: dict) -> None:
    STORE.mkdir(parents=True, exist_ok=True)
    with (STORE / "sense.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"pid": os.getpid(), **entry}) + "\n")


def seen(message: dict) -> dict:
    if "observe" not in message:
        return {}
    path = Path(message["observe"]["steps"][0]["rgb"])
    return {"rgb": {"absolute": path.is_absolute(), "exists": path.is_file(),
                    "sha": hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else None}}


def main() -> None:
    n = 0
    for raw in sys.stdin.buffer:
        message = json.loads(raw.decode("utf-8"))
        n += 1
        log({"n": n, "message": message, **seen(message)})
        if FAULT[0] and int(FAULT[2]) == n:
            if FAULT[0] == "exit":
                print("the feed broke", file=sys.stderr, flush=True)
                sys.exit(3)
            if FAULT[0] == "eof":
                sys.stdout.close()
                sys.exit(0)
            if FAULT[0] == "hang":
                time.sleep(600)
            print("not json" if FAULT[0] == "text" else "[1, 2]", flush=True)
            continue
        reply = {"ok": True, "receipt": {"n": n}} if "record" in message else {"ok": True, "n": n}
        print(json.dumps(reply), flush=True)
    log({"closed": True})


if __name__ == "__main__":
    main()
