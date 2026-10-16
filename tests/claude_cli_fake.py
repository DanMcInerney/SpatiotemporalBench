"""A stand-in for the `claude` command, for the tests of the Claude Code agent (`claude_kit` builds the `Claude` that
runs it; the example agent's own fake is `claude_fake.py`, which answers one message a process).

It is run as `python claude_cli_fake.py <the arguments claude would get>` and does what `claude -p --input-format
stream-json --output-format stream-json --verbose` does that the route depends on, with no network and no model: it
reads one user message a line from stdin and runs a turn for each, in ONE process (it exits when stdin ends); launches
the tool server named in the `--mcp-config` file and talks JSON-RPC 2.0 to it as Claude Code does; prints the events of
a turn (`system`/`init` once, an `assistant` event with a `tool_use` block for each call of the server's tools and a
`user` event with the `tool_result`, the image as the API writes it, `{"type": "image", "source": {"type": "base64",
...}}`, and last the `result`); and answers `--version`. A command line that lacks `-p`, `--verbose` or either
stream-json format exits 2 as `claude` does.

What the "model" does is `codex_fake`'s: it pulls (`next_frame`) until the reply says the job is over, answers each
label of an ask with `submit_answer` from how many frames (`[k<n> ` header lines) it has been shown in the whole
process, and ends the turn with one line. Usage, by hand: the t-th turn of the process reports the session's running
totals on its `result`, 10 input tokens, 60 cache reads, 30 cache writes and 10 output ones (4 of them thinking) for
each turn so far, and a `total_cost_usd` of 0.25 for each.

`FAKE_CLAUDE_PLAN` (a JSON file) changes the n-th turn (counted from 0 over the whole run, in the order they began), as
in `{"3": {"fail": "limit"}}`. A step may set `fail` (`limit`, `auth`, `context`, `overloaded`, `boom`: an error
`result` as Claude Code prints it, an assistant message carrying the API error beside it), `fail_late` (the same, after
the model made its calls), `crash` (exit 1 with the login message on stderr and no `result`), `noise` (a line that is
not JSON before the turn's events), `escape` (a `tool_use` of that built-in tool, such as `Bash`), `permission` (a
`control_request`), `sleep` (seconds, before anything) and for the pull `pulls` (end the turn after this many pulls in
this turn) and `hang` (after this many pulls sleep `hang_s` seconds, default 60, instead of pulling again).

Everything it was given and did is appended to `FAKE_CLAUDE_DIR`: `processes.jsonl` (one line a process, at its start:
the arguments, the working folder and what it holds, the environment variables that matter, the system prompt),
`messages-N.jsonl` (the user messages of process N, in order) and `calls-N.jsonl` (one line a tool call as it is made,
with the reply's text, whether it was an error, and the hash of each image block).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

HOME = Path(os.environ["FAKE_CLAUDE_DIR"]) if "FAKE_CLAUDE_DIR" in os.environ else None
KEYFRAME = re.compile(r"^\[k(\d+) ", re.M)
ASKED = re.compile(r"^(Q\d+): .*\nAnswer type: (\w+)\.", re.M)
END = "The job is over: there are no more frames."  # `llm_render.END_MESSAGE` (a test pins that they agree)
ENV_NAMES = ("CLAUDE_CODE_DISABLE_CLAUDE_MDS", "CLAUDE_CODE_DISABLE_AUTO_MEMORY", "MCP_TIMEOUT", "MCP_TOOL_TIMEOUT")
FAILURES = {"limit": ("rate_limit", "You've hit your session limit · resets 3pm"),
            "auth": ("authentication_failed", "Invalid API key · Please run /login"),
            "context": ("invalid_request", "Prompt is too long"),
            "overloaded": ("server_error", "API Error: 529 Overloaded"),
            "boom": ("unknown", "the model exploded")}
_OUT = threading.Lock()
_IDS = iter(range(1, 10**9))


def emit(event: dict) -> None:
    with _OUT:
        sys.stdout.write(json.dumps(event) + "\n")
        sys.stdout.flush()


class Server:
    """The tool server, driven as Claude Code drives an MCP server."""

    def __init__(self, spec: dict):
        self.proc = subprocess.Popen([spec["command"], *spec["args"]], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, encoding="utf-8")
        self.n = 0
        self.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                "clientInfo": {"name": "claude-fake", "version": "0"}})
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        self.tools = [t["name"] for t in self.rpc("tools/list", {})["tools"]]

    def rpc(self, method: str, params: dict) -> dict:
        self.n += 1
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.n, "method": method, "params": params}) + "\n")
        self.proc.stdin.flush()
        reply = json.loads(self.proc.stdout.readline())
        assert reply["id"] == self.n, reply
        return reply.get("result", reply.get("error"))

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(timeout=30)


def claim(prefix: str) -> int:
    """The next number of `prefix` among all of them, counted from 0: a number is taken by creating its file."""
    n = 0
    while True:
        try:
            os.close(os.open(HOME / f"{prefix}-{n}", os.O_CREAT | os.O_EXCL))
            return n
        except FileExistsError:
            n += 1


def step_of(n: int) -> dict:
    plan = os.environ.get("FAKE_CLAUDE_PLAN")
    return json.loads(Path(plan).read_text(encoding="utf-8")).get(str(n), {}) if plan else {}


class Process:
    """The one process of a case: the tool server, the frames shown and the turns so far."""

    def __init__(self, n: int, session: str, spec: dict):
        self.n, self.session, self.turns, self.shown = n, session, 0, []
        self.step: dict = {}
        self.pulls, self.nudged = 0, False
        self.server = Server(spec)

    def log(self, entry: dict) -> None:
        with (HOME / f"calls-{self.n}.jsonl").open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(entry) + "\n")

    def message(self, content: list[dict], usage: dict | None = None, **more) -> None:
        emit({"type": "assistant", "session_id": self.session, **more,
              "message": {"id": f"msg_{next(_IDS)}", "role": "assistant", "content": content,
                          "usage": usage or {"input_tokens": 1, "output_tokens": 1}}})

    def call(self, name: str, arguments: dict) -> dict:
        """One tool call, as Claude Code prints it: the `tool_use`, then the `tool_result` with the images as the API
        writes them. A record: the text, whether the reply was an error and the hash of each image block."""
        ident = f"toolu_{next(_IDS)}"
        self.message([{"type": "tool_use", "id": ident, "name": f"mcp__stbench__{name}", "input": arguments}])
        result = self.server.rpc("tools/call", {"name": name, "arguments": arguments})
        blocks = [b if b["type"] == "text" else
                  {"type": "image", "source": {"type": "base64", "media_type": b["mimeType"], "data": b["data"]}}
                  for b in result["content"]]
        emit({"type": "user", "session_id": self.session, "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": ident, "is_error": result["isError"], "content": blocks}]}})
        images = [hashlib.sha256(base64.b64decode(b["data"])).hexdigest()[:16]
                  for b in result["content"] if b["type"] == "image"]
        reply = {"text": "\n".join(b["text"] for b in result["content"] if b["type"] == "text"),
                 "error": result["isError"], "images": images, "blocks": [b["type"] for b in result["content"]]}
        self.log({"tool": name, "arguments": arguments, **reply})
        return reply

    def pull(self) -> dict | None:
        """The reply of the next `next_frame`, or None when the plan ends the turn (or hangs and then ends it)."""
        step = self.step
        if step.get("hang") is not None and self.pulls >= step["hang"]:
            time.sleep(step.get("hang_s", 60))
            return None
        if step.get("pulls") is not None and self.pulls >= step["pulls"]:
            return None
        self.pulls += 1
        reply = self.call("next_frame", {})
        self.shown.append(reply["text"])
        return reply

    def submit(self, label: str, kind: str) -> None:
        count = len(KEYFRAME.findall("\n".join(self.shown)))
        good = {"count": {"type": "count", "value": count}, "number": {"type": "number", "value": count},
                "position": {"type": "position", "xy": [count, 0]}}.get(kind, {"type": "unknown"})
        self.call("submit_answer", {"label": label, "answer": good})

    def answer(self, reply: dict) -> dict | None:
        for label, kind in dict(ASKED.findall(reply["text"])).items():
            self.submit(label, kind)
        return self.pull()

    def result(self, text: str, failure: str | None = None) -> None:
        """The end of a turn: a closing message and the `result`, with the session's running totals."""
        t = self.turns
        usage = {"input_tokens": 10 * t, "cache_read_input_tokens": 60 * t, "cache_creation_input_tokens": 30 * t,
                 "output_tokens": 10 * t, "output_tokens_details": {"thinking_tokens": 4 * t}}
        error, said = FAILURES[failure] if failure else (None, text)
        self.message([{"type": "text", "text": said}], **({"error": error} if error else {}))
        emit({"type": "result", "subtype": "success", "is_error": failure is not None, "num_turns": t,
              "result": said, "session_id": self.session, "total_cost_usd": 0.25 * t, "usage": usage,
              "permission_denials": []})

    def run_turn(self, text: str) -> int:
        self.turns += 1
        step, self.pulls = self.step, 0
        self.shown.append(text)
        if step.get("noise"):
            with _OUT:
                sys.stdout.write('[claude-code:unrecognized_model] {"model":"claude-fake"}\n')
                sys.stdout.flush()
        if step.get("sleep"):
            time.sleep(step["sleep"])
        if step.get("permission"):
            emit({"type": "control_request", "request_id": "r1", "request": {"subtype": "can_use_tool"}})
        if step.get("escape"):
            self.message([{"type": "tool_use", "id": "toolu_x", "name": step["escape"], "input": {"command": "ls"}}])
        if step.get("crash"):
            return 1
        if failure := step.get("fail"):
            self.result("", failure)
            return 0
        reply = self.pull()
        while reply is not None and reply["text"] != END:
            reply = self.answer(reply) if ASKED.search(reply["text"]) else self.pull()
        if failure := step.get("fail_late"):
            self.result("", failure)
        else:
            self.result("noted")
        return 0


def main(argv: list[str]) -> int:
    sys.stderr.reconfigure(encoding="utf-8")
    if argv[:1] == ["--version"]:
        print("9.9.9 (Claude Code fake)")
        return 0
    for needed in ("-p", "--verbose"):
        if needed not in argv:
            print(f"Error: {needed} is required here", file=sys.stderr)
            return 2
    formats = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a in ("--input-format", "--output-format")]
    if formats != ["stream-json"] * 2:
        print("Error: stream-json is required for both formats", file=sys.stderr)
        return 2
    spec = json.loads(Path(argv[argv.index("--mcp-config") + 1]).read_text(encoding="utf-8"))["mcpServers"]["stbench"]
    n = claim("process")
    system = argv[argv.index("--system-prompt") + 1] if "--system-prompt" in argv else None
    with (HOME / "processes.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"n": n, "argv": argv, "cwd": os.getcwd(), "cwd_files": sorted(os.listdir(".")),
                            "env": {k: os.environ.get(k) for k in ENV_NAMES}, "system": system, "mcp": spec}) + "\n")
    process = Process(n, f"session-{n:04d}", spec)
    code = 0
    for line in sys.stdin:
        if not line.strip():
            continue
        text = json.loads(line)["message"]["content"]
        with (HOME / f"messages-{n}.jsonl").open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(text) + "\n")
        process.step = step_of(claim("turn"))
        if process.turns == 0:
            emit({"type": "system", "subtype": "init", "session_id": process.session,
                  "tools": [f"mcp__stbench__{name}" for name in process.server.tools],
                  "mcp_servers": [{"name": "stbench", "status": "connected"}], "permissionMode": "dontAsk"})
        code = process.run_turn(text)
        if code:
            print("Invalid API key · Please run /login", file=sys.stderr)
            return code
    process.server.close()
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
