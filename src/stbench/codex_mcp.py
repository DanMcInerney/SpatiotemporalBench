"""The tool server of the Codex route: a relay from Codex to the harness (`codex_feed`), as a stdio MCP server.

`codex exec` is launched with this module as its only tool source (`python -m stbench.codex_mcp CONFIG.json`). It speaks
JSON-RPC 2.0 over stdin and stdout, one message a line (`initialize`, `tools/list`, `tools/call`, `ping`), and offers
exactly `agent_tools.TOOLS`: `next_frame` and `submit_answer`, on every channel. It decides nothing. Each `tools/call`
is sent over `multiprocessing.connection` to the harness, which answers with the `content` (text blocks and image
blocks) and `isError` of the reply; the relay returns them as they are, so the harness's image bytes come over the
connection and the relay reaches no file. The agent has no memory tool and no pointing tool (it remembers with its own
context). Nothing else is reachable through it. Codex puts the tools in a namespace (the model sees
`mcp__stbench.next_frame`).

A call of a tool that is not in the list is answered here with an error and never leaves the process. A relay that
cannot reach the harness answers every call with an error text.

CONFIG.json: `address` (host and port of the harness's listener) and `key` (hex), nothing else of the case. It is the
one file the relay reads.
"""

from __future__ import annotations

import json
import sys
from multiprocessing.connection import Client
from pathlib import Path

from .agent_tools import TOOLS

SERVER_NAME = "stbench"
PROTOCOL = "2025-06-18"


def tool_specs() -> list[dict]:
    """The tools as MCP lists them (`name`, `description`, `inputSchema`)."""
    return [{"name": t["function"]["name"], "description": t["function"]["description"],
             "inputSchema": t["function"]["parameters"]} for t in TOOLS]


def _error(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": True}


class Relay:
    """The link to the harness: one connection, made at the first call, kept while it works."""

    def __init__(self, config: dict):
        self.address, self.key = config.get("address"), config.get("key", "")
        self.conn = None

    def call(self, name: str, arguments) -> dict:
        """The harness's reply to a call, `{"content", "isError"}`; an error text when it cannot be reached."""
        try:
            if self.conn is None:
                self.conn = Client(tuple(self.address), authkey=bytes.fromhex(self.key))
            self.conn.send_bytes(json.dumps({"tool": name, "arguments": arguments}).encode("utf-8"))
            reply = json.loads(self.conn.recv_bytes())
            return {"content": reply["content"], "isError": bool(reply["isError"])}
        except Exception as error:  # a request never stops the server: a refusal is an error text
            self.conn = None
            return _error(f"error: the harness cannot be reached ({type(error).__name__})")


def handle(message: dict, relay: Relay) -> dict | None:
    """The reply to one JSON-RPC message, or None for a notification."""
    if not isinstance(message, dict) or "id" not in message:
        return None
    ident, method, params = message["id"], message.get("method"), message.get("params") or {}

    def done(result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": ident, "result": result}

    if method == "initialize":
        return done({"protocolVersion": params.get("protocolVersion") or PROTOCOL, "capabilities": {"tools": {}},
                     "serverInfo": {"name": SERVER_NAME, "version": "1"}})
    if method == "ping":
        return done({})
    if method == "tools/list":
        return done({"tools": tool_specs()})
    if method == "tools/call":
        name = params.get("name")
        if name not in {s["name"] for s in tool_specs()}:
            return done(_error(f"error: unknown tool {str(name)[:60]!r}"))
        return done(relay.call(name, params.get("arguments") or {}))
    return {"jsonrpc": "2.0", "id": ident, "error": {"code": -32601, "message": f"method not found: {method}"}}


def serve(config: dict, stdin=None, stdout=None) -> None:
    """Answer messages from `stdin` until it ends. A line that is not JSON is skipped."""
    stdin, stdout, relay = stdin or sys.stdin, stdout or sys.stdout, Relay(config)
    for line in stdin:
        if not line.strip():
            continue
        try:
            reply = handle(json.loads(line), relay)
        except ValueError:
            continue
        if reply is not None:
            stdout.write(json.dumps(reply) + "\n")
            stdout.flush()


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m stbench.codex_mcp CONFIG.json", file=sys.stderr)
        return 2
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    serve(json.loads(Path(argv[0]).read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
