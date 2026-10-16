"""A stand-in for the `claude` command, for the tests of the Claude Code example agent (`examples/claude_code_agent`).

Run as `python claude_fake.py <the arguments Claude Code would get>`; standard library only, no network, no model.
`--version` prints `9.9.9 (Claude Code fake)` and `--help` a usage text that lists the flags the agent passes.
Otherwise it needs `-p`, `--verbose` and both stream-json formats (as `claude -p` does), reads one user message from
stdin and prints three events: `system`/`init`, one `assistant` and the `result`. A message that fails the checks (a
flag missing, stdin not one user message of image and text blocks) exits 2.

- **Sessions** live in `$CLAUDE_CONFIG_DIR/fake-sessions/<id>.json` (a new id `s-<hex>` for a new session);
  `--resume ID` of a session that is not there exits 1 with `No conversation found with session ID: ID` on stderr.
- **Auto memory** is `$CLAUDE_CONFIG_DIR/projects/fake/memory/MEMORY.md`: a new session notes whether it existed
  (`memory_at_start`), and every message appends the frame keys (`[k12 ...` header lines) its text names.
- **Usage, by hand**: the n-th invocation of a session (n from 1) reports on its `assistant` event 100 input tokens,
  600 n cache reads, 200 cache writes and 50 output tokens, for the `--model` model; the first one also uses
  `claude-haiku-4-5` (10 in, 5 out). The result's `modelUsage` is the session's cumulative sums in camelCase and its
  `total_cost_usd` is 0.01 n (a figure no agent should report).
- **Replies**: a message with `Q<n>:` lines is answered by one JSON object giving each label `{"type": "unknown"}`,
  after a sentence of prose; any other message by `ok`. `FAKE_CLAUDE_PLAN` (a JSON file keyed by the invocation's
  number over the whole run, from 0: the number of lines `FAKE_CLAUDE_LOG` holds so far) may set `reply` (a text to
  give instead), `fail` (`auth`: exit 1 with `Invalid API key · Please run /login` on stderr and an error result) or
  `shrink` (the `assistant` event holds only 100 cache reads, as after a compaction; the result's sums are unchanged).
- **Record**: `FAKE_CLAUDE_LOG` gets one JSON line an invocation: the arguments, the working folder, the variables
  `ENV_NAMES` (null when unset), the session, whether it resumed, n, the images (the first 16 hex characters of each
  one's SHA-256, in order), the text, `memory_at_start` and `memory_now` (whether the auto memory existed when this
  message arrived).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import uuid
from pathlib import Path

ENV_NAMES = ("HOME", "USERPROFILE", "CLAUDE_CONFIG_DIR", "STBENCH_CASE_NONCE",
             "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "DISABLE_TELEMETRY")
KEYFRAME = re.compile(r"^\[(k\d+) ", re.M)
LABEL = re.compile(r"^(Q\d+): ", re.M)
AUTH = "Invalid API key · Please run /login"
HELP = """Usage: claude [options] [prompt]   (the fake: only the flags the example agent passes)

Options:
  -p, --print                    Print response and exit
  --input-format <format>        Input format (stream-json)
  --output-format <format>       Output format (stream-json)
  --verbose                      Override verbose mode setting
  --model <model>                Model for the current session
  --tools <tools...>             Specify the list of available tools
  --permission-mode <mode>       Permission mode to use for the session
  --max-turns <turns>            Maximum number of agentic turns
  --resume [value]               Resume a conversation by session ID"""


def die(code: int, message: str) -> None:
    sys.stderr.buffer.write((message + "\n").encode("utf-8"))
    sys.stderr.flush()
    raise SystemExit(code)


def flag(argv: list[str], name: str) -> str | None:
    """The value after `name` in `argv`, or None."""
    i = argv.index(name) + 1 if name in argv else len(argv)
    return argv[i] if i < len(argv) else None


def read_message() -> tuple[str, list[str]]:
    """(the text, the images' fingerprints) of the one user message on stdin, checked for the shape `claude` reads."""
    lines = [line for line in sys.stdin.buffer.read().decode("utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        die(2, f"fake claude: expected one user message on stdin, got {len(lines)} lines")
    msg = json.loads(lines[0])
    if msg.get("type") != "user" or msg.get("message", {}).get("role") != "user":
        die(2, "fake claude: stdin is not a user message")
    text, images = [], []
    for block in msg["message"]["content"]:
        if block.get("type") == "image":
            source = block["source"]
            if source.get("type") != "base64" or source.get("media_type") != "image/jpeg":
                die(2, f"fake claude: an image source this fake does not take: {sorted(source)}")
            images.append(hashlib.sha256(base64.b64decode(source["data"])).hexdigest()[:16])
        elif block.get("type") == "text":
            text.append(block["text"])
        else:
            die(2, f"fake claude: a content block of type {block.get('type')!r}")
    return "\n".join(text), images


def main(argv: list[str]) -> None:
    if argv == ["--version"]:
        print("9.9.9 (Claude Code fake)")
        return
    if argv == ["--help"]:
        print(HELP)
        return
    formats = (flag(argv, "--input-format"), flag(argv, "--output-format"))
    if "-p" not in argv or formats != ("stream-json", "stream-json"):
        die(2, "fake claude: needs -p with --input-format stream-json and --output-format stream-json")
    if "--verbose" not in argv:
        die(2, "Error: When using --print, --output-format=stream-json requires --verbose")
    model, resume = flag(argv, "--model") or "unknown-model", flag(argv, "--resume")
    config = Path(os.environ["CLAUDE_CONFIG_DIR"])
    text, images = read_message()
    log = Path(os.environ["FAKE_CLAUDE_LOG"]) if "FAKE_CLAUDE_LOG" in os.environ else None
    run_n = len(log.read_text(encoding="utf-8").splitlines()) if log is not None and log.exists() else 0
    plan = {}
    if "FAKE_CLAUDE_PLAN" in os.environ:
        plan = json.loads(Path(os.environ["FAKE_CLAUDE_PLAN"]).read_text(encoding="utf-8")).get(str(run_n), {})

    sessions = config / "fake-sessions"
    memory = config / "projects" / "fake" / "memory" / "MEMORY.md"
    if resume is not None:
        if not (sessions / f"{resume}.json").is_file():
            die(1, f"No conversation found with session ID: {resume}")
        state = json.loads((sessions / f"{resume}.json").read_text(encoding="utf-8"))
        sid = resume
    else:
        sid = f"s-{uuid.uuid4().hex[:12]}"
        state = {"n": 0, "memory_at_start": memory.exists(), "usage": {}}
    n = state["n"] + 1

    memory_now = memory.exists()  # what the auto memory held when this message arrived
    keys = KEYFRAME.findall(text)
    if keys and not plan.get("fail"):
        memory.parent.mkdir(parents=True, exist_ok=True)
        with memory.open("a", encoding="utf-8") as f:
            f.write(" ".join(keys) + "\n")
    labels = LABEL.findall(text)
    reply = plan.get("reply")
    if reply is None:
        answers = json.dumps({label: {"type": "unknown"} for label in labels})
        reply = f"Here are my answers.\n{answers}" if labels else "ok"

    if log is not None:
        record = {"args": argv, "cwd": os.getcwd(), "env": {name: os.environ.get(name) for name in ENV_NAMES},
                  "session": sid, "resumed": resume is not None, "n": n, "images": images, "text": text,
                  "memory_at_start": state["memory_at_start"], "memory_now": memory_now}
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    print(json.dumps({"type": "system", "subtype": "init", "session_id": sid, "model": model}), flush=True)
    if plan.get("fail") == "auth":
        print(json.dumps({"type": "result", "subtype": "error_during_execution", "is_error": True, "result": AUTH,
                          "session_id": resume or "", "num_turns": 0, "total_cost_usd": 0, "usage": {},
                          "modelUsage": {}}), flush=True)
        die(1, AUTH)

    usage = {"input_tokens": 100, "cache_read_input_tokens": 600 * n, "cache_creation_input_tokens": 200,
             "output_tokens": 50}
    totals = state["usage"]
    for name, (fresh, read, written, out) in {model: (100, 600 * n, 200, 50),
                                              **({"claude-haiku-4-5": (10, 0, 0, 5)} if n == 1 else {})}.items():
        t = totals.setdefault(name, {"inputTokens": 0, "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0,
                                     "outputTokens": 0, "costUSD": 0.0})
        t["inputTokens"] += fresh
        t["cacheReadInputTokens"] += read
        t["cacheCreationInputTokens"] += written
        t["outputTokens"] += out
        t["costUSD"] += 0.001
    state["n"] = n
    sessions.mkdir(parents=True, exist_ok=True)
    (sessions / f"{sid}.json").write_text(json.dumps(state), encoding="utf-8")

    held = {**usage, "cache_read_input_tokens": 100} if plan.get("shrink") else usage  # a compacted conversation
    print(json.dumps({"type": "assistant", "session_id": sid,
                      "message": {"role": "assistant", "model": model, "content": [{"type": "text", "text": reply}],
                                  "usage": held}}), flush=True)
    print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": reply, "session_id": sid,
                      "num_turns": 1, "total_cost_usd": round(0.01 * n, 2), "usage": usage, "modelUsage": totals}),
          flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
