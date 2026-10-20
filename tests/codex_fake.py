"""A stand-in for the `codex` command, for the tests of the Codex route (`codex_kit` builds the `Codex` that runs it).

It is run as `python codex_fake.py <the arguments Codex would get>` and does what `codex exec --json` does that the
route depends on, with no network and no model: it keeps threads (`exec` starts one and `resume` extends it), reads the
prompt from stdin, launches every MCP server named in the `-c mcp_servers.<name>.*` arguments (the tool server,
`stbench`, and a memory's, if the run has one) and talks JSON-RPC 2.0 to each as Codex does (`initialize`,
`tools/list`, `tools/call`), and prints the JSON lines of an exec: `thread.started`, the calls of the servers' tools
(with the reply's content, an image block and its base64 included, as a Codex event may carry it, and the `server` it
was made to), an agent message, and `turn.completed` with usage as Codex prints it: the thread's running total, 100
tokens in, 40 of them cached, and 10 out for each exec of the thread. It reads the file named by
`model_instructions_file` when the exec starts, as Codex does, and records its text. `--version` and `debug models`
answer too.

What the "model" does: it pulls. After the prompt (the system prompt and the first task) it calls `next_frame` through
the tool server until the reply says the job is over, and then replies with one line and completes the turn. The frames
it counts are the keyframes (a line `[k<n> ` starts one) it has been shown, in the thread's earlier messages and in this
exec's replies. On an ask (a reply with lines `Qn: text` and `Answer type: kind.`) it answers each label with one
`submit_answer(label, answer)` from that count (the answer from the type), then pulls again. A reply carries questions
when its text has those lines, whether it is an ask item or a frame item that comes with its questions (the frame's
header line first, `[k<n> ...`, then the ask's text, as `codex_feed.Item.with_ask` makes it): the frame is counted with
the others and the questions are answered from that count, before the next pull. It remembers with its own context, as
the reference agent does: it keeps no file.

`FAKE_CODEX_PLAN` (a JSON file) changes the n-th exec (counted from 0 over the whole run, in the order they were
started), as in `{"3": {"fail": "limit"}}`. A step may set `fail` (`limit`, `auth`, `context`, `boom`, or `flaky`, a 503
that the retry, which is the next exec, does not meet), `fail_late` (the same failures, after the model made its
calls), `sleep` (seconds, before anything), `incomplete` (exit 0 with no `turn.completed`), `exit` (the exit code,
default 0), and for the pull:

- `pulls`: end the turn (reply, complete) after this many pulls in this exec; `hang`: after this many pulls sleep
  `hang_s` seconds (default 60) instead of pulling again (`hang` 0 never pulls);
- `premature`: on an ask pull this many times before answering anything (the second is the one that goes through);
- `submit`: how each label is answered, `none` (never), `late` (only after the nudge), `bad` (one invalid submission and
  then a valid one), `bad2` (two invalid ones) or a dict of those by label, a label left out answering as usual;
- `double`: make the first pull twice at once, through a second tool server (one of the two is refused);
- `linger`: after the last call, wait this many seconds before completing the turn;
- `escape` (an item of that type, such as `command_execution`) and `server` and `tool` (an item that calls that tool of
  that other server), emitted once this many pulls (`escape_at`, default 0) have been made;
- `stray` (a list of `submit_answer` arguments called before the first pull, outside any ask) and `call` (a tool name,
  called with no arguments, such as a retired `memory_read`: the server answers an unknown-tool error);
- `memory`: tool calls of a memory's server, each `{"server": name, "tool": ..., "arguments": {...}}`, made after every
  frame item the model is shown (a reply whose text starts `[k<n> `, a frame that comes with questions included), the
  string `{k}` in a string argument replaced by the frame's number; `memory_ask`: the same, made when the model is given
  questions (an ask item, or a frame item that carries them), before it answers. The replies are not shown to the
  "model" (they take no part in the count of frames it has seen), and the calls are not the tool server's: they are
  logged apart, with their `server`;
- `next_memory`: a memory that rides on the pull (an arm whose SPEC says `carry`): a `memory` argument (any JSON, in
  which the string `{k}` in every string value is replaced by the number of the frame just shown) on every `next_frame`
  call that follows a frame item, until the next item comes, so a pull the feed refuses (a premature one) carries it
  too. The reply, whose last block is then the memory's, is the "model's" next item as any other.

Everything it was given and did is appended to `execs.jsonl` (one line an exec, at its start: its argv, its prompt, the
text of its instructions file, the names of the MCP servers it launched and `server_instructions`, the `instructions`
each server's `initialize` reply held, by name, null for a server that sent none; `{}` for an exec that fails or ends
before it launches its servers) and `calls-N.jsonl` (one line a tool call,
as it is made, with the `server` it went to, and a last `finished` line) in `FAKE_CODEX_DIR`.
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
import tomllib
from pathlib import Path

HOME = Path(os.environ["FAKE_CODEX_DIR"]) if "FAKE_CODEX_DIR" in os.environ else None
KEYFRAME = re.compile(r"^\[k(\d+) ", re.M)
ASKED = re.compile(r"^(Q\d+): .*\nAnswer type: (\w+)\.", re.M)
END = "The job is over: there are no more frames."  # `llm_render.END_MESSAGE` (a test pins that they agree)
USAGE = {"input_tokens": 100, "cached_input_tokens": 40, "cache_write_input_tokens": 0, "output_tokens": 10,
         "reasoning_output_tokens": 0}
FAILURES = {"limit": "You've hit your usage limit. Upgrade to Pro or try again later.",
            "auth": "Your access token could not be refreshed. Please log in again with `codex login`.",
            "context": ("Codex ran out of room in the model's context window. Start a new thread or clear earlier "
                        "history before retrying."),  # Codex 0.159.3's own wording, read from its binary
            "flaky": "stream error: 503 Service Unavailable", "boom": "the model exploded"}
MODELS = {"models": [{"slug": "fake-model", "tool_mode": "code_mode_only", "apply_patch_tool_type": "freeform",
                      "experimental_supported_tools": ["clock"], "multi_agent_version": "v2",
                      "supports_search_tool": True, "base_instructions": "Codex coding agent",
                      # as the real catalog lists them: no `none`, and `max` beyond `xhigh` (no stbench flag takes it)
                      "supported_reasoning_levels": [{"effort": e, "description": f"{e} reasoning"}
                                                     for e in ("low", "medium", "high", "xhigh", "max")]},
                     {"slug": "other-model", "tool_mode": None},
                     {"slug": "limited-model", "tool_mode": None,  # a model whose entry stops at high
                      "supported_reasoning_levels": [{"effort": e, "description": e}
                                                     for e in ("low", "medium", "high")]}]}
_OUT = threading.Lock()
_ITEMS = iter(range(1, 10**9))


def emit(event: dict) -> None:
    with _OUT:
        sys.stdout.write(json.dumps(event) + "\n")
        sys.stdout.flush()


class Server:
    """The tool server, driven as Codex drives an MCP server."""

    def __init__(self, command: str, args: list[str], log, name: str = "stbench"):
        self.proc = subprocess.Popen([command, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                     encoding="utf-8")
        self.n, self.log, self.name = 0, log, name
        hello = self.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                        "clientInfo": {"name": "codex-fake", "version": "0"}})
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

    def call(self, name: str, arguments: dict) -> dict:
        """One tool call: its events as Codex prints them, and a record: the text, whether the reply was an error and
        the hash of each image block."""
        item = {"id": f"item_{next(_ITEMS)}", "type": "mcp_tool_call", "server": self.name, "tool": name,
                "arguments": arguments, "result": None, "error": None}
        emit({"type": "item.started", "item": {**item, "status": "in_progress"}})
        result = self.rpc("tools/call", {"name": name, "arguments": arguments})
        emit({"type": "item.completed", "item": {**item, "result": {"content": result["content"],
                                                                    "structured_content": None},
                                                 "status": "completed"}})
        blocks = result["content"]
        images = [hashlib.sha256(base64.b64decode(b["data"])).hexdigest()[:16]
                  for b in blocks if b["type"] == "image"]
        reply = {"text": "\n".join(b["text"] for b in blocks if b["type"] == "text"), "error": result["isError"],
                 "images": images, "blocks": [b["type"] for b in blocks]}
        self.log({"tool": name, "server": self.name, "arguments": arguments, **reply})
        return reply

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(timeout=30)


def filled(found, k: str):
    """`found` (JSON) with `{k}` in every string value replaced by the frame number `k`."""
    if isinstance(found, str):
        return found.replace("{k}", k)
    if isinstance(found, list):
        return [filled(v, k) for v in found]
    if isinstance(found, dict):
        return {key: filled(v, k) for key, v in found.items()}
    return found


def value(setting: str):
    return tomllib.loads("v = " + setting.split("=", 1)[1])["v"]


def others(args: dict) -> dict[str, tuple[str, list[str]]]:
    """The MCP servers other than the tool server that the `-c` settings name: name -> (command, arguments)."""
    names = sorted({key.split(".")[1] for key in args["c"] if key.startswith("mcp_servers.")} - {"stbench"})
    return {name: (value(f"x={args['c'][f'mcp_servers.{name}.command']}"),
                   value(f"x={args['c'][f'mcp_servers.{name}.args']}")) for name in names}


def parse(args: list[str]) -> dict:
    """The arguments of an exec: the sub-command, the `-c` settings (key -> TOML value), the features disabled, the
    flags, with what follows `--` as the positional words."""
    found = {"sub": None, "c": {}, "disable": [], "flags": [], "cd": None, "model": None, "words": []}
    rest = args[args.index("exec") + 1:]
    if rest and rest[0] == "resume":
        found["sub"], rest = rest[0], rest[1:]
    i = 0
    while i < len(rest):
        arg = rest[i]
        if arg == "--":
            found["words"] = rest[i + 1:]
            break
        if arg in ("-c", "--disable", "-C", "-m"):
            i += 1
            if arg == "-c":
                key, _, text = rest[i].partition("=")
                found["c"][key] = text
            elif arg == "--disable":
                found["disable"].append(rest[i])
            elif arg == "-C":
                found["cd"] = rest[i]
            else:
                found["model"] = rest[i]
        else:
            found["flags"].append(arg)
        i += 1
    return found


def load(thread: str) -> dict:
    path = HOME / "threads" / f"{thread}.json"
    if not path.exists():
        print(f"Error: thread/resume failed: no rollout found for thread id {thread} (code -32600)", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def claim() -> int:
    """The number of this exec among all of them, counted from 0. Execs may run at once (`eval --jobs`), so a number is
    taken by creating its file, which only one of them can."""
    n = 0
    while True:
        try:
            os.close(os.open(HOME / f"exec-{n}", os.O_CREAT | os.O_EXCL))
            return n
        except FileExistsError:
            n += 1


def step_of(n: int) -> dict:
    plan = os.environ.get("FAKE_CODEX_PLAN")
    return json.loads(Path(plan).read_text(encoding="utf-8")).get(str(n), {}) if plan else {}


class Model:
    """What the "model" does in one exec (the module docstring): pull until the job is over, answer the asks."""

    def __init__(self, n: int, step: dict, history: list[str], prompt: str, command: str, server_args: list[str],
                 others: dict[str, tuple[str, list[str]]]):
        self.n, self.step, self.shown = n, step, [*history, prompt]
        self.pulls, self.foreign, self.nudged = 0, False, False
        self.frame: str | None = None  # the number of the frame item last shown, until another item comes
        self.make = lambda: Server(command, server_args, self.log)  # a tool server, as Codex starts one
        self.server = self.make()
        self.others = {name: Server(cmd, cmd_args, self.log, name) for name, (cmd, cmd_args) in others.items()}

    def log(self, entry: dict) -> None:
        with (HOME / f"calls-{self.n}.jsonl").open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(entry) + "\n")

    def frames(self) -> int:
        return len(KEYFRAME.findall("\n".join(self.shown)))

    def arguments(self) -> dict:
        """The arguments of the next `next_frame` call: the plan's `next_memory` after a frame item, else none."""
        if self.step.get("next_memory") is None or self.frame is None:
            return {}
        return {"memory": filled(self.step["next_memory"], self.frame)}

    def pull(self) -> dict | None:
        """The reply of the next `next_frame`, or None when the plan ends the turn (or hangs and then ends it)."""
        step = self.step
        if (step.get("escape") or step.get("server")) and not self.foreign and self.pulls >= step.get("escape_at", 0):
            self.foreign = True
            if step.get("escape"):
                emit({"type": "item.completed", "item": {"id": "x", "type": step["escape"], "command": "cat secret"}})
            if step.get("server"):
                emit({"type": "item.completed", "item": {"id": "y", "type": "mcp_tool_call", "server": step["server"],
                                                         "tool": step.get("tool", "anything"), "arguments": {}}})
        if step.get("hang") is not None and self.pulls >= step["hang"]:
            time.sleep(step.get("hang_s", 60))
            return None
        if step.get("pulls") is not None and self.pulls >= step["pulls"]:
            return None
        self.pulls += 1
        if step.get("double") and self.pulls == 1:  # two calls at once, from two tool servers
            other, box = self.make(), []
            second = threading.Thread(target=lambda: box.append(other.call("next_frame", self.arguments())))
            second.start()
            first = self.server.call("next_frame", self.arguments())
            second.join()
            other.close()
            reply = next(r for r in (first, box[0]) if not r["error"])
        else:
            reply = self.server.call("next_frame", self.arguments())
        self.shown.append(reply["text"])
        if not reply["error"]:  # an item came: a frame is what the next pulls follow
            found = KEYFRAME.match(reply["text"])
            self.frame = found.group(1) if found else None
        return reply

    def submit(self, label: str, kind: str, mode: str | None) -> None:
        count = self.frames()
        good = {"count": {"type": "count", "value": count}, "number": {"type": "number", "value": count},
                "position": {"type": "position", "xy": [count, 0]}}.get(kind, {"type": "unknown"})
        for _ in range({"bad": 1, "bad2": 2}.get(mode, 0)):
            self.server.call("submit_answer", {"label": label, "answer": {"type": "nonsense"}})
        if mode != "bad2":
            self.server.call("submit_answer", {"label": label, "answer": good})
            self.server.call("submit_answer", {"label": label, "answer": good})  # it is over: the server says so

    def remember(self, calls: list[dict] | None, reply: dict) -> None:
        """The plan's calls of a memory's server, made after the item `reply` (`{k}`: the frame's number)."""
        found = KEYFRAME.match(reply["text"])
        for call in calls or ():
            arguments = filled(call.get("arguments", {}), found.group(1) if found else "?")
            self.others[call["server"]].call(call["tool"], arguments)

    def answer(self, reply: dict) -> dict | None:
        """An ask: pull early as the plan says, answer each label, pull again; the reply to the pull that follows."""
        self.remember(self.step.get("memory_ask"), reply)
        labels = dict(ASKED.findall(reply["text"]))
        plan = self.step.get("submit")
        for _ in range(self.step.get("premature", 0)):
            early = self.pull()
            if early is None or not early["error"]:
                return early  # the ask was closed by the pull that went through: this is the next item
            self.nudged = True
        late = []
        for label, kind in labels.items():
            mode = plan.get(label) if isinstance(plan, dict) else plan
            if mode == "none":
                continue
            if mode == "late" and not self.nudged:
                late.append((label, kind))
                continue
            self.submit(label, kind, mode)
        after = self.pull()
        if after is not None and after["error"] and late:  # the nudge
            for label, kind in late:
                self.submit(label, kind, None)
            after = self.pull()
        return after

    def run(self) -> None:
        for arguments in self.step.get("stray", ()):
            self.server.call("submit_answer", arguments)
        if self.step.get("call"):
            self.server.call(self.step["call"], {})
        reply = self.pull()
        while reply is not None and not reply["text"].startswith(END):  # a carried reply may follow the end's text
            if KEYFRAME.match(reply["text"]):  # a frame item: the plan's memory calls, then the questions it comes with
                self.remember(self.step.get("memory"), reply)
            # the questions of an ask item, or of a frame item that comes with them: either way `reply["text"]` has them
            reply = self.answer(reply) if ASKED.search(reply["text"]) else self.pull()
        self.server.close()
        for other in self.others.values():
            other.close()


def main(argv: list[str]) -> int:
    if argv[:1] == ["--version"]:
        print("codex-cli 9.9.9-fake")
        return 0
    if argv[:2] == ["debug", "models"]:
        print(json.dumps(MODELS))
        return 0
    args = parse(argv)
    prompt = sys.stdin.buffer.read().decode("utf-8")
    n = claim()
    step = step_of(n)
    sub = args["sub"]
    base = args["words"][0] if len(args["words"]) > 1 else None
    held = load(base) if base else {"history": [], "execs": 0}
    history, execs = held["history"], held["execs"] + 1
    thread = base if sub == "resume" else f"thread-{n:04d}"
    (HOME / "threads").mkdir(exist_ok=True)
    named = value(f"x={args['c']['model_instructions_file']}")
    reaches = not (step.get("incomplete") or step.get("fail"))  # the exec that ends or fails first launches no server
    model = None
    if reaches:
        server_args = value(f"x={args['c']['mcp_servers.stbench.args']}")
        command = value(f"x={args['c']['mcp_servers.stbench.command']}")
        model = Model(n, step, history, prompt, command, server_args, others(args))
        model.log({"tools": model.server.tools, "config": json.loads(Path(server_args[-1]).read_text(encoding="utf-8")),
                   "tool_timeout": args["c"].get("mcp_servers.stbench.tool_timeout_sec"),
                   "others": {name: server.tools for name, server in model.others.items()}})
    started = {"n": n, "argv": argv, "sub": sub, "base": base, "thread": thread, "prompt": prompt,
               "instructions": Path(named).read_text(encoding="utf-8"), "servers": sorted(["stbench", *others(args)]),
               "server_instructions": {} if model is None else {
                   "stbench": model.server.instructions, **{name: o.instructions for name, o in model.others.items()}},
               "cwd": os.getcwd(), "cwd_files": sorted(os.listdir(".")),
               "cd_files": sorted(os.listdir(args["cd"])) if args["cd"] else None, "history": history, "step": step}
    with (HOME / "execs.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(started) + "\n")
    emit({"type": "thread.started", "thread_id": thread})
    emit({"type": "turn.started"})
    if step.get("sleep"):
        time.sleep(step["sleep"])
    if step.get("incomplete"):
        return 0  # exits 0 without finishing the turn, as an exec does after something was blocked
    if failure := step.get("fail"):
        emit({"type": "error", "message": FAILURES[failure]})
        emit({"type": "turn.failed", "error": {"message": FAILURES[failure]}})
        return 1
    try:
        model.run()
        if step.get("linger"):  # the turn does not complete for a while after the model's last call
            time.sleep(step["linger"])
    finally:
        (HOME / "threads" / f"{thread}.json").write_text(
            json.dumps({"history": model.shown, "execs": execs}), encoding="utf-8")
    model.log({"finished": True})
    if failure := step.get("fail_late"):  # the model has acted (its tool calls are above) when the turn fails
        emit({"type": "turn.failed", "error": {"message": FAILURES[failure]}})
        return 1
    emit({"type": "item.completed", "item": {"id": "z", "type": "agent_message", "text": "noted"}})
    emit({"type": "turn.completed", "usage": {k: v * execs for k, v in USAGE.items()}})  # the thread's running total
    return step.get("exit", 0)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
