"""The `claude` command, held to the reference agent's tools: its argv, its environment and the rules for what comes
back (the Claude Code twin of `codex_exec`; docs/claude.md).

`Claude` builds the command line of `claude -p` for one model and one effort and starts it (`popen`);
`claude_live.ClaudeLive` runs it and reads it. A case is ONE process: stream-json in and stream-json out, the first
user message on stdin, and a later one (the fixed resume line, when the model ended its turn early) written to the same
stdin, so the conversation never leaves the process and nothing is saved to disk. The model reaches nothing but the tool
server `codex_mcp` (`next_frame` and `submit_answer`, which Claude Code names `mcp__stbench__next_frame` and
`mcp__stbench__submit_answer`), because the process is launched with (`Claude.argv`):

- `--tools ""`: none of Claude Code's built-in tools (no shell, files, web, subagents or tool search);
- `--strict-mcp-config --mcp-config FILE`: the tool server as the only MCP server, and `--allowedTools` the two tools
  by name with `--permission-mode dontAsk`, so those two run without a prompt and anything else is denied rather than
  asked; `--disable-slash-commands` turns off skills and commands;
- `--setting-sources ""`: no user, project or local settings (hooks, plugins, permissions, memory settings);
  `--no-session-persistence`: no transcript on disk and nothing to resume from there;
- `--system-prompt`: the reference agent's system prompt (`llm_render`) in place of Claude Code's own, not appended to
  it; the first user message is the first task;
- the environment (`ENV`): `CLAUDE_CODE_DISABLE_CLAUDE_MDS` and `CLAUDE_CODE_DISABLE_AUTO_MEMORY` (no CLAUDE.md of the
  user, a parent folder or the project is read, and no auto memory), and the tool server's start-up and call timeouts;
- an empty working directory, and the credential of the user's own login (`--bare` is not used: it ignores OAuth).

What the model is left with is those two tools. Every event is checked as it arrives (`check_event`, the one owner of
the check): a call of any tool but the server's (Claude Code's two MCP resource helpers aside, `HELPERS`) is
`ToolEscaped`, and a permission request (`control_request`: a call held for approval) is a `CodexError`. The error
classes and `failure_reason` are `codex_exec`'s: the two agents share the vocabulary of how a run fails.

A turn is over when its `result` event comes. It failed when that event says `is_error` or is not a `success`, when it
lists `permission_denials`, or when the process ended without one; `failure` names the class by the message
(`classify`: a usage or rate limit, a 429 or an overloaded API; a login problem; a full context window; else a plain
error). Claude Code retries a transient API failure itself (its `api_retry` events are journaled as warnings), so the
harness never retries one. `usage` and `total_cost_usd` of the `result` are the session's running totals (like Codex's
thread totals: `codex_log.Journal` takes each turn's own share); the cost is the API-equivalent dollars, which a
subscription does not pay.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .agent_tools import NEXT_NAME, SUBMIT_NAME, ContextTooLong
from .codex_exec import (
    AUTH,
    CONTEXT,
    LIMIT,
    SERVER,
    CodexAuth,
    CodexError,
    CodexLimit,
    CodexMissing,
    ToolEscaped,
)

ALIASES = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}
EFFORTS = ("low", "medium", "high", "xhigh", "max")  # `claude --effort`
ALLOWED = (f"mcp__{SERVER}__{NEXT_NAME}", f"mcp__{SERVER}__{SUBMIT_NAME}")
HELPERS = ("ListMcpResourcesTool", "ReadMcpResourceTool")  # Claude Code's own MCP resource tools: no resources exist
ENV = {"CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1", "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
       "MCP_TIMEOUT": "60000", "MCP_TOOL_TIMEOUT": "600000"}  # ms: the tool server's start (60 s) and a call (600 s)
CMD_LIMIT = 30000   # characters: Windows cuts a command line at 32,767 (the longest system prompt is under 10,000)
LIMIT_MORE = r"|hit your .{0,24}limit|limit reached|resets? (?:at )?\d|overloaded|\b529\b|credit balance|billing"
AUTH_MORE = r"|invalid api key|authentication|\boauth\b|token has expired"
CONTEXT_MORE = r"|prompt is too long|context window|context_length|too many images"
CLAUDE_LIMIT = re.compile(LIMIT.pattern + LIMIT_MORE, re.I)
CLAUDE_AUTH = re.compile(AUTH.pattern + AUTH_MORE, re.I)
CLAUDE_CONTEXT = re.compile(CONTEXT.pattern + CONTEXT_MORE, re.I)
CMD_SHIM = (".cmd", ".bat")
_VERSIONS: dict[tuple, str] = {}   # `claude --version` by command: a case starts a process, and a dataset has many


def model_id(model: str) -> str:
    """`haiku`, `sonnet` and `opus` as the ids they stand for, and a full `claude-...` id as it is. ValueError for any
    other name (the id is what the journal, the summary and the price table are keyed by)."""
    if model in ALIASES:
        return ALIASES[model]
    if re.fullmatch(r"claude-[A-Za-z0-9._-]+", model):
        return model
    raise ValueError(f"the model must be {', '.join(ALIASES)} or a full id such as {ALIASES['sonnet']}, not {model!r}")


def classify(message: str) -> type[CodexError] | type[ContextTooLong]:
    """The error class a failure message belongs to (`codex_exec.classify`'s order, with Claude Code's wordings)."""
    if CLAUDE_CONTEXT.search(message):
        return ContextTooLong
    if CLAUDE_LIMIT.search(message):
        return CodexLimit
    if CLAUDE_AUTH.search(message):
        return CodexAuth
    return CodexError


def check_event(event: dict) -> None:
    """Raise `ToolEscaped` for a call of a tool that is not one of the tool server's (Claude Code's own MCP resource
    helpers aside), and `CodexError` for a call held for approval or a tool server that did not start. Anything else is
    fine. The one owner of the check: every event of every turn is run through it as it arrives."""
    kind = event.get("type")
    if kind == "control_request":
        raise CodexError("a tool call was held for approval: the allow-list did not apply")
    if kind == "system" and event.get("subtype") == "init":
        for server in event.get("mcp_servers") or ():
            if isinstance(server, dict) and server.get("status") in ("failed", "needs-auth"):
                raise CodexError(f"the tool server {server.get('name')!r} is {server['status']}: it did not start")
    if kind != "assistant":
        return
    for block in (event.get("message") or {}).get("content") or ():
        if isinstance(block, dict) and block.get("type") == "tool_use" and \
                block.get("name") not in (*ALLOWED, *HELPERS):
            raise ToolEscaped(f"claude called the tool {str(block.get('name'))[:60]!r}, which is not one of the tool "
                              "server's: the restriction failed")


def _count(value) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def usage_of(raw: dict, cost) -> dict:
    """`llm_client`'s usage names for a `result` event's `usage` (the session's running totals) and `total_cost_usd`:
    the prompt tokens include the cached and the cache-written ones, as there, and the thinking tokens are counted
    inside the output ones."""
    cached, written = _count(raw.get("cache_read_input_tokens")), _count(raw.get("cache_creation_input_tokens"))
    thinking = (raw.get("output_tokens_details") or {}).get("thinking_tokens")
    return {"prompt_tokens": _count(raw.get("input_tokens")) + cached + written,
            "completion_tokens": _count(raw.get("output_tokens")), "cached_tokens": cached,
            "cache_write_tokens": written, "reasoning_tokens": _count(thinking),
            "cost": float(cost) if isinstance(cost, int | float) and not isinstance(cost, bool) else 0.0}


def digest(events: list[dict], noise: list[str]) -> dict:
    """What the events of a turn say: `session` (the id they carry, "" if none), `messages` (the model's text),
    `warnings` (the lines that were not events, the API retries, tools offered beyond the server's), `usage` (the
    `result`'s, in `llm_client`'s names, zeros without one) and `completed` (a `result` came)."""
    session, messages, warnings, result = "", [], list(noise), None
    for e in events:
        session = str(e.get("session_id") or session)
        kind = e.get("type")
        if kind == "assistant":
            messages += [str(b.get("text") or "") for b in (e.get("message") or {}).get("content") or ()
                         if isinstance(b, dict) and b.get("type") == "text"]
        elif kind == "result":
            result = e
        elif kind == "system" and e.get("subtype") == "api_retry":
            warnings.append(f"api_retry {e.get('attempt')}/{e.get('max_retries')}, status {e.get('error_status')}: "
                            f"{e.get('error')}")
        elif kind == "system" and e.get("subtype") == "init":
            beyond = [str(t) for t in e.get("tools") or () if t not in (*ALLOWED, *HELPERS)]
            warnings += [f"tools offered beyond the server's: {', '.join(beyond)}"] if beyond else []
    usage = usage_of((result or {}).get("usage") or {}, (result or {}).get("total_cost_usd"))
    return {"thread": session, "messages": messages, "warnings": warnings, "usage": usage,
            "completed": result is not None}


def failure(events: list[dict], code: int | None, err: str) -> tuple[type, str] | None:
    """(error class, message) when the turn did not complete, else None: its `result` is missing, an error or lists
    `permission_denials`. `code` is the process's exit code (None while it runs). The message is the result's text, else
    the text of an assistant message that carries an API error, else the end of stderr."""
    result = next((e for e in reversed(events) if e.get("type") == "result"), None)
    if result is not None and not result.get("is_error") and result.get("subtype", "success") == "success":
        if denied := [str(d.get("tool_name")) for d in result.get("permission_denials") or () if isinstance(d, dict)]:
            return CodexError, (f"a tool call was denied by the permission mode ({', '.join(denied)}): the allow-list "
                                "did not apply")
        return None
    text = ""
    if result is not None:
        errors = "; ".join(str(x) for x in result.get("errors") or ())
        text = str(result.get("result") or "") or errors or str(result.get("subtype") or "")
    for e in reversed(events):   # an API error is an assistant message with the kind of error beside it
        if not text and e.get("type") == "assistant" and e.get("error"):
            body = " ".join(str(b.get("text") or "") for b in (e.get("message") or {}).get("content") or ()
                            if isinstance(b, dict))
            text = f"{e['error']}: {body}".strip()
    text = text or err.strip()[-500:] or f"claude exited {code} without finishing the turn"
    return classify(text), text


class Claude:
    """The `claude` command for one model. `command` is the executable and its leading arguments (default: `claude` on
    PATH); `env` is added to the environment. `model` is the full model id (`model_id`). `prepare` checks the command;
    `argv` is the command line of the one process of a case and `popen` starts it. Safe to share between threads."""

    def __init__(self, model: str, *, reasoning: str = "low", command: list[str] | None = None,
                 env: dict[str, str] | None = None):
        if reasoning not in EFFORTS:
            raise ValueError(f"reasoning must be one of {list(EFFORTS)}, got {reasoning!r}")
        self.model, self.reasoning, self.env = model_id(model), reasoning, env or {}
        self._command = command

    @property
    def command(self) -> list[str]:
        if self._command is None:
            found = shutil.which("claude")
            if not found:
                raise CodexMissing("the claude command is not on PATH: install Claude Code and log in with `claude`")
            self._command = [found]
        return self._command

    @property
    def version(self) -> str:
        """`claude --version`, e.g. `2.1.284 (Claude Code)`; read once a process for each command."""
        if (key := tuple(self.command)) not in _VERSIONS:
            try:
                done = subprocess.run([*self.command, "--version"], capture_output=True, text=True, encoding="utf-8",
                                      errors="replace", stdin=subprocess.DEVNULL, timeout=60.0,
                                      env={**os.environ, **self.env})
            except (OSError, subprocess.TimeoutExpired) as error:
                raise CodexMissing(f"cannot run `{self.command[0]} --version`: {error}") from None
            if done.returncode:
                raise CodexMissing(f"`claude --version` exited {done.returncode}: {done.stderr.strip()[-300:]}")
            _VERSIONS[key] = done.stdout.strip()
        return _VERSIONS[key]

    def prepare(self) -> None:
        """Check that the command runs. Raises CodexMissing (no command, one that does not run, or a `.cmd` shim, which
        cmd.exe would misread the system prompt for); idempotent. `run` and `eval` call it before any case."""
        if os.name == "nt" and self.command[0].lower().endswith(CMD_SHIM):
            raise CodexMissing(f"{self.command[0]} is a .cmd shim, which cmd.exe would misread the system prompt for: "
                               "use the native claude.exe (`claude install`)")
        _ = self.version

    def close(self) -> None:
        """Nothing to remove (the codex transport's catalog has no twin here); the runner calls it for both."""

    def argv(self, mcp_config: Path, system_prompt: str) -> list[str]:
        """The command line of the case's process: the first user message goes on stdin, and so does each later one."""
        return [*self.command, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
                "--model", self.model, "--effort", self.reasoning, "--setting-sources", "", "--tools", "",
                "--disable-slash-commands", "--strict-mcp-config", "--mcp-config", str(mcp_config),
                "--allowedTools", *ALLOWED, "--permission-mode", "dontAsk", "--no-session-persistence",
                "--system-prompt", system_prompt]

    def mcp_config(self, server_config: Path) -> dict:
        """The `--mcp-config` file: the tool server (`codex_mcp`, given its own configuration file) and no other."""
        return {"mcpServers": {SERVER: {"type": "stdio", "command": sys.executable,
                                        "args": ["-m", "stbench.codex_mcp", str(server_config)]}}}

    def popen(self, argv: list[str], cwd: Path, stderr) -> subprocess.Popen:
        """Start the process: stdin and stdout are pipes, stderr is `stderr` (an open file), and on POSIX it leads a
        session of its own so that `kill_tree` can end it with what it started. CodexMissing when it cannot be run,
        CodexError for a command line Windows would cut."""
        if os.name == "nt" and (size := len(subprocess.list2cmdline(argv))) > CMD_LIMIT:
            raise CodexError(f"the command line is {size} characters, over the {CMD_LIMIT} Windows passes to a program")
        try:
            return subprocess.Popen(argv, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                    env={**os.environ, **ENV, **self.env}, start_new_session=os.name != "nt")
        except OSError as error:
            raise CodexMissing(f"cannot run {argv[0]}: {error}") from None
