"""A tiny stdio MCP server standing in for a memory's (`codex_memory`'s `serve`), for the tests of the memory hook.

`python memory_fake_server.py STORE [INSTRUCTIONS]`: JSON-RPC 2.0, one message a line (`initialize`, `tools/list`,
`tools/call`), as `codex_mcp` speaks it. It keeps a store in the folder STORE, as a real memory would: it makes the
folder and its `cache/` when it starts (a store's cache is rebuilt, so it is not kept beside a case), sends INSTRUCTIONS
(when given) as the `instructions` of its `initialize` reply, as a real memory server sends its core, and offers three
tools:

- `claim(text)`: appends the text to `STORE/claims.jsonl` and answers `ok`;
- `view()`: answers the claims so far, one line each, then the frames the sensor feed wrote
  (`memory_fake_sense.py`'s `STORE/sense.jsonl`: `frames k0 k1 ...`);
- `snapshot()`: answers a text block and an image block (a JPEG-like byte string in base64), as a memory that shows a
  picture back would.

Everything it is called with is appended to `STORE/server.jsonl`, with the process id, so a test can tell how many
servers were started (one for each exec).
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

STORE = Path(sys.argv[1])
PICTURE = b"\xff\xd8\xff\xe0" + b"memory-picture" + b"\xff\xd9"
TOOLS = [{"name": name, "description": f"the fake memory's {name}", "inputSchema": {"type": "object"}}
         for name in ("claim", "view", "snapshot")]


def lines(name: str) -> list[dict]:
    path = STORE / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def note(entry: dict) -> None:
    with (STORE / "server.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"pid": os.getpid(), **entry}) + "\n")


def call(name: str, arguments: dict) -> dict:
    note({"tool": name, "arguments": arguments})
    if name == "claim":
        with (STORE / "claims.jsonl").open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(arguments) + "\n")
        return {"content": [{"type": "text", "text": "ok"}], "isError": False}
    if name == "view":
        seen = [m["message"]["observe"]["steps"][0]["k"] for m in lines("sense.jsonl")
                if "observe" in m.get("message", {})]
        text = "\n".join([*(c["text"] for c in lines("claims.jsonl")), "frames " + " ".join(map(str, seen))])
        return {"content": [{"type": "text", "text": text}], "isError": False}
    return {"content": [{"type": "text", "text": "a picture"},
                        {"type": "image", "data": base64.b64encode(PICTURE).decode("ascii"),
                         "mimeType": "image/jpeg"}], "isError": False}


def handle(message: dict) -> dict | None:
    if "id" not in message:
        return None
    ident, params = message["id"], message.get("params") or {}
    if message["method"] == "initialize":
        result = {"protocolVersion": params.get("protocolVersion"), "capabilities": {"tools": {}},
                  "serverInfo": {"name": "fake-memory", "version": "0"},
                  **({"instructions": sys.argv[2]} if len(sys.argv) > 2 else {})}
    elif message["method"] == "tools/list":
        result = {"tools": TOOLS}
    else:
        result = call(params["name"], params.get("arguments") or {})
    return {"jsonrpc": "2.0", "id": ident, "result": result}


def main() -> None:
    (STORE / "cache").mkdir(parents=True, exist_ok=True)
    (STORE / "cache" / "index.bin").write_bytes(b"rebuilt on demand")
    for line in sys.stdin:
        if line.strip():
            reply = handle(json.loads(line))
            if reply is not None:
                sys.stdout.write(json.dumps(reply) + "\n")
                sys.stdout.flush()


if __name__ == "__main__":
    main()
