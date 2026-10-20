"""A stand-in for the `claude` command, for the tests of the Claude Code agent (`claude_kit` builds the `Claude` that
runs it; the example agent's own fake is `claude_fake.py`, which answers one message a process).

It is run as `python claude_cli_fake.py <the arguments claude would get>` and does what `claude -p --input-format
stream-json --output-format stream-json --verbose` does that the route depends on, with no network and no model: it
reads one user message a line from stdin and runs a turn for each, in ONE process (it exits when stdin ends); launches
every MCP server named in the `--mcp-config` file (the tool server, `stbench`, and a memory's, if the run has one) and
talks JSON-RPC 2.0 to each as Claude Code does; prints the events of a turn (`system`/`init` once, an `assistant` event
with a `tool_use` block for each call of a server's tools, named `mcp__SERVER__TOOL`, and a `user` event with the
`tool_result`, the image as the API writes it, `{"type": "image", "source": {"type": "base64", ...}}`, and last the
`result`); and answers `--version`. A command line that lacks `-p`, `--verbose` or either stream-json format exits 2
as `claude` does. It holds the "model" to `--allowedTools` as `--permission-mode dontAsk` does: a call of a tool that no
pattern allows (the exact name, `mcp__SERVER` for a whole server or a `*` ending) is not made, and the turn's `result`
lists it in `permission_denials`.

What the "model" does is `codex_fake`'s: it pulls (`next_frame`) until the reply says the job is over, answers each
label of an ask with `submit_answer` from how many frames (`[k<n> ` header lines) it has been shown in the whole
process, and ends the turn with one line. Usage, by hand: the t-th turn of the process reports the session's running
totals on its `result`, 10 input tokens, 60 cache reads, 30 cache writes and 10 output ones (4 of them thinking) for
each turn so far, and a `total_cost_usd` of 0.25 for each.

What the "model" does with a memory is `codex_fake`'s: `memory` (calls of a memory's server, `{"server": name, "tool":
..., "arguments": {...}}`, made after every frame item the model is shown, `{k}` in a string argument replaced by the
frame's number), `memory_ask` (the same, when it is given questions) and `next_memory` (a `memory` argument on every
`next_frame` that follows a frame item, the record of a memory that carries). Their replies are not shown to the
"model", and the calls are logged apart, with their `server`. A step may also set `server` and `tool` (a `tool_use` of
that other server, at the start of the turn) and `compact` (a `compact_boundary` system event, with that many tokens
before, as Claude Code prints one when it compacts).

`FAKE_CLAUDE_PLAN` (a JSON file) changes the n-th turn (counted from 0 over the whole run, in the order they began), as
in `{"3": {"fail": "limit"}}`. A step may set `fail` (`limit`, `auth`, `context`, `overloaded`, `boom`: an error
`result` as Claude Code prints it, an assistant message carrying the API error beside it), `fail_late` (the same, after
the model made its calls), `crash` (exit 1 with the login message on stderr and no `result`), `noise` (a line that is
not JSON before the turn's events), `escape` (a `tool_use` of that built-in tool, such as `Bash`), `permission` (a
`control_request`), `sleep` (seconds, before anything) and for the pull `pulls` (end the turn after this many pulls in
this turn), `hang` (after this many pulls sleep `hang_s` seconds, default 60, instead of pulling again) and
`premature` (on an ask, pull this many times before answering anything).

Everything it was given and did is appended to `FAKE_CLAUDE_DIR`: `processes.jsonl` (one line a process, at its start:
the arguments, the working folder and what it holds, the environment variables that matter and, apart, the ones that
set compaction (`compaction`), the system prompt, the tool server's `config` file, the servers it launched (`mcp`, by
name, and their names, `servers`) and the `instructions` each one's `initialize` reply held, by name, null for none),
`messages-N.jsonl` (the user messages of process N, in order) and `calls-N.jsonl` (one line a tool call as it is made,
with the `server`, the reply's text, whether it was an error, and the hash of each image block).
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
COMPACT_NAMES = ("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", "DISABLE_AUTO_COMPACT")
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
    """An MCP server (the tool server, or a memory's), driven as Claude Code drives one."""

    def __init__(self, spec: dict):
        self.proc = subprocess.Popen([spec["command"], *spec["args"]], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, encoding="utf-8")
        self.n = 0
        hello = self.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                        "clientInfo": {"name": "claude-fake", "version": "0"}})
        self.instructions = hello.get("instructions")
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


def filled(found, k: str):
    """`found` (JSON) with `{k}` in every string value replaced by the frame number `k`."""
    if isinstance(found, str):
        return found.replace("{k}", k)
    if isinstance(found, list):
        return [filled(v, k) for v in found]
    if isinstance(found, dict):
        return {key: filled(v, k) for key, v in found.items()}
    return found


def allowed(patterns: list[str], name: str) -> bool:
    """Whether `--allowedTools` allows the tool `name`: its exact name, a whole server (`mcp__SERVER`) or a `*` end."""
    return any(p == name or (p.startswith("mcp__") and p.count("__") == 1 and name.startswith(p + "__"))
               or (p.endswith("*") and name.startswith(p[:-1])) for p in patterns)


def step_of(n: int) -> dict:
    plan = os.environ.get("FAKE_CLAUDE_PLAN")
    return json.loads(Path(plan).read_text(encoding="utf-8")).get(str(n), {}) if plan else {}


class Process:
    """The one process of a case: the tool server, the frames shown and the turns so far."""

    def __init__(self, n: int, session: str, specs: dict, patterns: list[str]):
        self.n, self.session, self.turns, self.shown = n, session, 0, []
        self.step: dict = {}
        self.pulls, self.nudged = 0, False
        self.patterns, self.denied = patterns, []
        self.frame: str | None = None   # the number of the frame item last shown, until another item comes
        self.servers = {name: Server(spec) for name, spec in specs.items()}
        self.server = self.servers["stbench"]

    def log(self, entry: dict) -> None:
        with (HOME / f"calls-{self.n}.jsonl").open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(entry) + "\n")

    def message(self, content: list[dict], usage: dict | None = None, **more) -> None:
        emit({"type": "assistant", "session_id": self.session, **more,
              "message": {"id": f"msg_{next(_IDS)}", "role": "assistant", "content": content,
                          "usage": usage or {"input_tokens": 1, "output_tokens": 1}}})

    def call(self, name: str, arguments: dict, server: str = "stbench") -> dict:
        """One tool call of `server`, as Claude Code prints it: the `tool_use`, then the `tool_result` with the images
        as the API writes them. A record: the text, whether the reply was an error and the hash of each image block. A
        call no `--allowedTools` pattern allows is not made: it is denied (the turn's `result` lists it) and answers an
        error."""
        ident = f"toolu_{next(_IDS)}"
        full = f"mcp__{server}__{name}"
        self.message([{"type": "tool_use", "id": ident, "name": full, "input": arguments}])
        if not allowed(self.patterns, full):
            self.denied.append(full)
            result = {"content": [{"type": "text", "text": f"Permission to use {full} has been denied."}],
                      "isError": True}
        else:
            result = self.servers[server].rpc("tools/call", {"name": name, "arguments": arguments})
        blocks = [b if b["type"] == "text" else
                  {"type": "image", "source": {"type": "base64", "media_type": b["mimeType"], "data": b["data"]}}
                  for b in result["content"]]
        emit({"type": "user", "session_id": self.session, "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": ident, "is_error": result["isError"], "content": blocks}]}})
        images = [hashlib.sha256(base64.b64decode(b["data"])).hexdigest()[:16]
                  for b in result["content"] if b["type"] == "image"]
        reply = {"text": "\n".join(b["text"] for b in result["content"] if b["type"] == "text"),
                 "error": result["isError"], "images": images, "blocks": [b["type"] for b in result["content"]]}
        self.log({"tool": name, "server": server, "arguments": arguments, **reply})
        return reply

    def arguments(self) -> dict:
        """The arguments of the next `next_frame` call: the plan's `next_memory` after a frame item, else none."""
        if self.step.get("next_memory") is None or self.frame is None:
            return {}
        return {"memory": filled(self.step["next_memory"], self.frame)}

    def remember(self, calls: list[dict] | None, reply: dict) -> None:
        """The plan's calls of a memory's server, made after the item `reply` (`{k}`: the frame's number)."""
        found = KEYFRAME.match(reply["text"])
        for call in calls or ():
            self.call(call["tool"], filled(call.get("arguments", {}), found.group(1) if found else "?"),
                      call["server"])

    def pull(self) -> dict | None:
        """The reply of the next `next_frame`, or None when the plan ends the turn (or hangs and then ends it)."""
        step = self.step
        if step.get("hang") is not None and self.pulls >= step["hang"]:
            time.sleep(step.get("hang_s", 60))
            return None
        if step.get("pulls") is not None and self.pulls >= step["pulls"]:
            return None
        self.pulls += 1
        reply = self.call("next_frame", self.arguments())
        self.shown.append(reply["text"])
        if not reply["error"]:   # an item came: a frame is what the next pulls follow
            found = KEYFRAME.match(reply["text"])
            self.frame = found.group(1) if found else None
        return reply

    def submit(self, label: str, kind: str) -> None:
        count = len(KEYFRAME.findall("\n".join(self.shown)))
        good = {"count": {"type": "count", "value": count}, "number": {"type": "number", "value": count},
                "position": {"type": "position", "xy": [count, 0]}}.get(kind, {"type": "unknown"})
        self.call("submit_answer", {"label": label, "answer": good})

    def answer(self, reply: dict) -> dict | None:
        self.remember(self.step.get("memory_ask"), reply)
        for _ in range(self.step.get("premature", 0)):   # pull before answering: the first is refused, the second goes
            early = self.pull()
            if early is None or not early["error"]:
                return early   # the ask was closed by the pull that went through: this is the next item
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
              "permission_denials": [{"tool_name": name, "tool_use_id": "x", "tool_input": {}}
                                     for name in self.denied]})

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
        if step.get("server"):   # a call of some other MCP server, which Claude Code would have no tool for
            other = f"mcp__{step['server']}__{step.get('tool', 'x')}"
            self.message([{"type": "tool_use", "id": "toolu_y", "name": other, "input": {}}])
        if step.get("compact"):
            emit({"type": "system", "subtype": "compact_boundary", "session_id": self.session,
                  "compact_metadata": {"trigger": "auto", "pre_tokens": step["compact"]}})
        if step.get("crash"):
            return 1
        if failure := step.get("fail"):
            self.result("", failure)
            return 0
        reply = self.pull()
        while reply is not None and not reply["text"].startswith(END):   # a carried reply may follow the end's text
            if KEYFRAME.match(reply["text"]):   # a frame item: the plan's memory calls, then its questions
                self.remember(step.get("memory"), reply)
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
    specs = json.loads(Path(argv[argv.index("--mcp-config") + 1]).read_text(encoding="utf-8"))["mcpServers"]
    patterns = argv[argv.index("--allowedTools") + 1:]
    patterns = patterns[:next((i for i, a in enumerate(patterns) if a.startswith("--")), len(patterns))]
    n = claim("process")
    system = argv[argv.index("--system-prompt") + 1] if "--system-prompt" in argv else None
    process = Process(n, f"session-{n:04d}", specs, patterns)   # it launches the servers, whose `initialize` we record
    with (HOME / "processes.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"n": n, "argv": argv, "cwd": os.getcwd(), "cwd_files": sorted(os.listdir(".")),
                            "env": {k: os.environ.get(k) for k in ENV_NAMES}, "system": system, "mcp": specs,
                            "servers": sorted(specs),
                            "config": json.loads(Path(specs["stbench"]["args"][-1]).read_text(encoding="utf-8")),
                            "compaction": {k: os.environ.get(k) for k in COMPACT_NAMES},
                            "server_instructions": {k: v.instructions for k, v in process.servers.items()}}) + "\n")
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
                  "tools": [f"mcp__{k}__{name}" for k, v in process.servers.items() for name in v.tools],
                  "mcp_servers": [{"name": k, "status": "connected"} for k in process.servers],
                  "permissionMode": "dontAsk"})
        code = process.run_turn(text)
        if code:
            print("Invalid API key · Please run /login", file=sys.stderr)
            return code
    for server in process.servers.values():
        server.close()
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
