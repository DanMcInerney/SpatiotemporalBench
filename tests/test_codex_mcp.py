"""The tool server of the Codex route (`codex_mcp`): a relay from Codex to the harness (`codex_feed`) over stdio
JSON-RPC, and nothing else. No model, no `codex`, no network (a connection to a `Feed` on the loopback address).

The server decides nothing: what a call answers is the feed's (`test_codex_feed.py`, `test_codex_asks.py`). What is
pinned here is the tool list, the protocol, that a call is forwarded as it is and answered as it is, and that the server
reaches the harness and its own configuration and nothing else.
"""

from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
import threading

from stbench import agent_tools, codex_mcp
from stbench.agent_tools import NO_QUESTION
from stbench.codex_feed import Feed, Item

RETIRED = ["memory_list", "memory_read", "memory_write", "memory_append", "memory_search", "locate"]


def talk(config: dict, *messages: dict) -> list[dict]:
    """The server's replies to JSON-RPC messages, run in-process over text streams."""
    out = io.StringIO()
    codex_mcp.serve(config, io.StringIO("".join(json.dumps(m) + "\n" for m in messages)), out)
    return [json.loads(line) for line in out.getvalue().splitlines()]


def call(name: str, arguments: dict, ident: int = 1) -> dict:
    return {"jsonrpc": "2.0", "id": ident, "method": "tools/call", "params": {"name": name, "arguments": arguments}}


def text_of(reply: dict) -> str:
    return reply["result"]["content"][0]["text"]


class Unreachable:
    """A relay that fails the test when a call is sent to the harness."""

    def call(self, name, arguments):
        raise AssertionError(f"{name} reached the harness")


def test_the_tools_listed_are_next_frame_and_submit_answer_with_their_schemas():
    """Contract (plan: reuse the tool schema, do not fork it; DECISIONS 2026-10-10; FD-frame-pull §3.3): `tools/list`
    offers exactly `agent_tools.TOOLS`, `next_frame` first and then `submit_answer`, each with its description and its
    parameter schema as `inputSchema`; `next_frame` takes no arguments. No memory tool or `locate` is listed. Regression
    caught: a memory tool listed to the headline agent again, a schema copied and drifting from `agent_tools`', or a
    `next_frame` that asks for arguments."""
    (reply,) = talk({}, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    tools = reply["result"]["tools"]
    assert [t["name"] for t in tools] == ["next_frame", "submit_answer"]
    assert tools == [{"name": t["function"]["name"], "description": t["function"]["description"],
                      "inputSchema": t["function"]["parameters"]} for t in agent_tools.TOOLS]
    assert tools[0]["inputSchema"] == {"type": "object", "properties": {}}
    assert tools[1]["inputSchema"]["required"] == ["label", "answer"]
    assert "memory" not in json.dumps(tools).lower()


def test_the_server_speaks_json_rpc_as_codex_expects():
    """Contract (MCP stdio): `initialize` is answered with the client's protocol version and the tools capability,
    `ping` with an empty result, a notification (no id) with nothing at all, an unknown method with error -32601, an
    unknown tool with an `isError` result naming the tool, and a line that is not JSON is skipped. Regression caught: a
    reply to a notification (Codex would read it as a broken response), a crash on a bad line, or no `isError` flag on a
    refusal (the model would read the error as a result)."""
    hello = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}}
    replies = talk({}, hello, {"jsonrpc": "2.0", "method": "notifications/initialized"},
                   {"jsonrpc": "2.0", "id": 2, "method": "ping"},
                   {"jsonrpc": "2.0", "id": 3, "method": "resources/list"}, call("rm_rf", {}, 4))
    assert [r["id"] for r in replies] == [1, 2, 3, 4]
    assert replies[0]["result"]["protocolVersion"] == "2025-03-26"
    assert replies[0]["result"]["capabilities"] == {"tools": {}}
    assert replies[0]["result"]["serverInfo"]["name"] == "stbench"
    assert replies[1]["result"] == {} and replies[2]["error"]["code"] == -32601
    assert replies[3]["result"]["isError"] is True and "rm_rf" in text_of(replies[3])
    out = io.StringIO()
    ping = json.dumps({"jsonrpc": "2.0", "id": 9, "method": "ping"})
    codex_mcp.serve({}, io.StringIO(f"not json\n\n{ping}\n"), out)
    assert [json.loads(line)["id"] for line in out.getvalue().splitlines()] == [9]


def test_the_retired_memory_and_pointing_tools_are_unknown_to_the_server_and_never_reach_the_harness():
    """Contract (DECISIONS 2026-10-10): a call of a tool the agent is not offered (the five memory tools and `locate`)
    is the server's unknown-tool error, flagged `isError`, whatever its arguments, and is not sent to the harness at
    all. Regression caught: a memory tool still answering a model that calls it from habit (the agent would keep a
    folder the benchmark no longer gives it), or an unknown name forwarded for the harness to deal with."""
    message = {"jsonrpc": "2.0", "id": 1, "method": "tools/call"}
    for n, name in enumerate(RETIRED, 1):
        reply = codex_mcp.handle({**message, "id": n, "params": {"name": name, "arguments": {"path": "a.md", "k": 7}}},
                                 Unreachable())
        assert reply["result"]["isError"] is True and text_of(reply) == f"error: unknown tool {name!r}"


def test_a_call_is_forwarded_to_the_harness_and_answered_as_the_harness_answers_it():
    """Contract (FD-frame-pull §3.3): the server sends each call (name and arguments) over the connection and returns
    the reply's content and `isError` as they are: here an error from the feed (a submission outside an ask), and a
    frame with its image block, byte for byte. Regression caught: a server that decides for itself, drops an image
    block, or turns an error into a result."""
    feed = Feed()
    image = bytes(range(200))
    feed.serve_in_prompt(Item("task", "T0"))
    try:
        config = feed.server_config()
        refused, = talk(config, call("submit_answer", {"label": "Q1", "answer": {"type": "unknown"}}))
        assert text_of(refused) == NO_QUESTION and refused["result"]["isError"] is True
        box: list[dict] = []
        puller = threading.Thread(target=lambda: box.extend(talk(config, call("next_frame", {}, 5))), daemon=True)
        puller.start()
        feed.supply(Item("frame", "[k0 c] h", image, k=0))
        puller.join(20)
        (frame,) = box
        assert frame["id"] == 5 and frame["result"]["isError"] is False
        blocks = frame["result"]["content"]
        assert [b["type"] for b in blocks] == ["text", "image"] and blocks[0]["text"] == "[k0 c] h"
        assert blocks[1]["mimeType"] == "image/jpeg" and base64.b64decode(blocks[1]["data"]) == image
    finally:
        feed.close()


def test_a_server_that_cannot_reach_the_harness_answers_every_call_with_an_error():
    """Contract: with no address, a wrong key or a harness that is gone, each `tools/call` is answered with an error
    text and `isError`, `tools/list` still works, and nothing is raised. Regression caught: a traceback on stdout (Codex
    would take it for a protocol message), or a model left waiting on a call that never answers."""
    feed = Feed()
    config = feed.server_config()
    feed.close()
    for cfg in ({}, {**config, "key": "00" * 32}, config):  # none, wrong, and a harness whose listener is closed
        listed, answered = talk(cfg, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, call("next_frame", {}, 2))
        assert len(listed["result"]["tools"]) == 2
        assert answered["result"]["isError"] is True
        assert text_of(answered).startswith("error: the harness cannot be reached")


AUDITED = '''
import sys
from stbench import codex_mcp

opened = []
sys.addaudithook(lambda event, args: opened.append(str(args[0])) if event == "open" else None)
code = codex_mcp.main([sys.argv[1]])
print("\\n".join(opened), file=sys.stderr)
sys.exit(code)
'''


def test_the_server_reads_no_file_but_its_configuration(tmp_path):
    """Contract (FD-frame-pull §3.3): the relay reads no file but its configuration: run as a process, over a real call
    to a feed (a refused submission and a tools list), every file it opens, bar the modules it imports, is the
    configuration file. Regression caught: a server that reaches the disk (a trace, an answer table, a path the model
    names), which the old server did and a model-facing process must not."""
    feed = Feed()
    try:
        config = tmp_path / "server.json"
        config.write_text(json.dumps(feed.server_config()), encoding="utf-8")
        driver = tmp_path / "audited.py"
        driver.write_text(AUDITED, encoding="utf-8")
        lines = [{"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                 call("submit_answer", {"label": "Q1", "answer": {"type": "unknown"}}, 2),
                 call("memory_read", {"path": str(tmp_path / "secret.txt")}, 3)]
        done = subprocess.run([sys.executable, str(driver), str(config)], input="".join(json.dumps(m) + "\n"
                                                                                       for m in lines),
                              capture_output=True, text=True, encoding="utf-8", timeout=60)
    finally:
        feed.close()
    assert done.returncode == 0, done.stderr
    assert [json.loads(line)["id"] for line in done.stdout.splitlines()] == [1, 2, 3]
    opened = [line for line in done.stderr.splitlines() if line and not line.endswith((".py", ".pyc", ".pyd", ".dll"))]
    assert opened == [str(config)], opened


def test_the_server_runs_as_a_process_the_way_codex_starts_it(tmp_path):
    """Contract (the launch line of `codex_exec`): `python -m stbench.codex_mcp CONFIG.json` reads its configuration,
    answers JSON lines on stdout as they come (flushed, so Codex is not left waiting) and exits when stdin ends.
    Regression caught: a server that buffers its replies or prints anything but JSON-RPC on stdout, which Codex
    would read as a protocol error, or one that cannot be run as a module."""
    feed = Feed()
    try:
        config = tmp_path / "server.json"
        config.write_text(json.dumps(feed.server_config()), encoding="utf-8")
        proc = subprocess.Popen([sys.executable, "-m", "stbench.codex_mcp", str(config)], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, text=True, encoding="utf-8")
        proc.stdin.write(json.dumps(call("submit_answer", {"label": "Q1", "answer": {"type": "unknown"}})) + "\n")
        proc.stdin.flush()
        reply = json.loads(proc.stdout.readline())
        assert text_of(reply) == NO_QUESTION
        proc.stdin.close()
        assert proc.wait(timeout=30) == 0 and proc.stdout.read() == ""
    finally:
        feed.close()


def test_the_server_refuses_a_missing_configuration(capsys):
    """Contract: started without its configuration file it says how to run it and exits 2, writing nothing to stdout.
    Regression caught: a traceback on stdout, which Codex would take for a protocol message."""
    assert codex_mcp.main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "usage: python -m stbench.codex_mcp CONFIG.json" in captured.err
