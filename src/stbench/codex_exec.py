"""The `codex exec` command, stripped to the reference agent's tools (`codex_system`, `codex_mcp`): its argv and the
rules for what comes back.

`Codex` builds the command line of `codex exec [resume ID] --json` for one model, one prompt (on stdin), and starts it
(`popen`); `codex_live.Live` runs it and reads it. A case is one thread: the first exec starts it (`start`) and every
later one continues it (`resume`); there is no fork and no ephemeral exec. The model reaches nothing but the tool server
`codex_mcp` (`next_frame` and `submit_answer`: it remembers with its own context, which Codex compacts itself), because
every exec is launched the same way (`Codex.argv`):

- `sandbox_mode=read-only`, `approval_policy=never`, `--ignore-user-config`, `--ignore-rules` and `--strict-config` (a
  key this Codex does not know is an error, never a tool left switched on), in an empty working directory (`-C`; a
  resume has no such flag and runs in an empty folder);
- `--disable` for the shell and exec tools, the browser, computer use, image generation, apps, plugins, multi-agent,
  image viewing, skills, goals, hooks and the other features that reach files or the web, and `web_search=disabled`;
- the instruction messages Codex adds (skills, permissions, environment with the working directory, collaboration
  mode, apps) switched off, and `model_instructions_file`, a one-line base instruction in place of Codex's
  coding-agent prompt (the reference agent's prompt is its first message, as its system prompt; a memory's guide is the
  base instructions' second paragraph, never part of a message);
- Codex's own automatic compaction left at its default: the agent manages its own context, as it would in use. A run
  that asks for it (`compact_at`, `--compact-at TOKENS`) adds `model_auto_compact_token_limit` to every exec, start
  and resume, so Codex compacts the thread earlier (a lever of the experiment, never the reference agent's);
- a model catalog (the installed Codex's own, `codex debug models`, one entry patched, in a temp file) in which the
  model has `tool_mode` `direct` (the gpt-6 models are `code_mode_only`, which reaches MCP tools only through a
  JavaScript `exec` tool), no apply_patch, no experimental tools (the clock), no multi-agent and no tool search;
- the tool server as the only MCP server, `required` (an exec that cannot start it fails), its calls pre-approved and a
  tool timeout of `TOOL_TIMEOUT_S` (a pull is answered at once, so the timeout is never near); a run with a memory
  (`codex_memory`, `--memory SPEC`) adds that memory's server, configured the same way, and appends the memory's guide
  to the base instructions, and nothing else of the exec changes;
- no personal instructions: Codex adds `AGENTS.md` (or `AGENTS.override.md`) of its home folder to every exec as a
  message, whatever the flags, so `prepare` refuses a home folder that holds a non-empty one (`PERSONAL`).

What the model is left with is the server's tools, the helpers Codex always adds for MCP servers (`list_mcp_resources`,
`list_mcp_resource_templates` and `read_mcp_resource`, which can only ask that server, whose resources are none) and
`request_user_input`, which answers that no one is there. Every event is checked as it arrives (`check_event`, the one
owner of the check): an item that is not a message, a thought or a call of the server's tools (a command, a file
change, a web search) is `ToolEscaped`, and the exec is an error, whatever its exit code says (`codex exec` exits 0
after blocking a call). Codex's own two resource listers, which it reports as a server called `codex`, are the one
exception (`HELPERS`). The live proof (a prompt that asks for a file by path, a directory listing, a shell command and a
web page) is in FINDINGS; `tests/test_codex_*.py` pin the argv. With a memory its server is a second one whose calls the
check lets through (`check_event`, `memory`); a call of any other server is still `ToolEscaped`.

A failure is named by its message: `CodexAuth` (log in again), `CodexLimit` (the plan's usage or rate limit),
`ContextTooLong` (the window is full despite Codex's own compaction: the case ends and the reason is counted),
`CodexTimeout`, else `CodexError`; `failure_reason` is the one word that counts each. One failure before the model did
anything that looks transient (a 5xx, a dropped connection) is tried once more after `RETRY_WAIT` seconds (`transient`);
nothing is retried in a loop.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .agent_tools import ContextTooLong
from .safecopy import remove_tree

KINDS = ("start", "resume")  # the exec kinds of a case's one thread
SERVER = "stbench"
BASE_INSTRUCTIONS = "You are a helpful assistant. Use the tools you are given when they help."
DISABLED = ("shell_tool", "unified_exec", "browser_use", "browser_use_external", "browser_use_full_cdp_access",
            "computer_use", "image_generation", "apps", "plugins", "multi_agent", "multi_agent_v2", "view_image",
            "skill_search", "tool_suggest", "goals", "hooks", "workspace_dependencies", "remote_plugin", "sleep_tool")
CONFIG = ("sandbox_mode=read-only", "approval_policy=never", "web_search=disabled",
          "skills.include_instructions=false", "include_environment_context=false",
          "include_permissions_instructions=false", "include_collaboration_mode_instructions=false",
          "include_apps_instructions=false")
SERVER_CONFIG = ("required=true", "default_tools_approval_mode=approve", "startup_timeout_sec=60")
TOOL_TIMEOUT_S = 600.0  # seconds Codex waits for one tool call (`tool_timeout_sec`); a pull is answered at once
CATALOG_PATCH = {"apply_patch_tool_type": None, "experimental_supported_tools": [], "multi_agent_version": None,
                 "tool_mode": "direct", "supports_search_tool": False}
EFFORTS = ("none", "low", "medium", "high", "xhigh")  # `model_reasoning_effort`: a value, never a change to the argv
ITEMS = ("agent_message", "reasoning", "error", "mcp_tool_call")  # what the events of an exec may hold
HELPERS = {"codex": ("list_mcp_resources", "list_mcp_resource_templates")}  # Codex's own listers: no resources exist
PERSONAL = ("AGENTS.md", "AGENTS.override.md")  # what Codex reads from its home folder into every exec's messages
RETRY_WAIT = 10.0  # seconds before the one retry of a transient failure
UNSAFE = re.compile(r'[&|<>^%"\r\n]')  # what cmd.exe reads as syntax, in an argument given to a .cmd shim
CMD_LIMIT = 8000  # characters: cmd.exe cuts a command line at 8191, and a .cmd shim is run by it

LIMIT = re.compile(r"usage limit|usage_limit|rate limit|rate_limit|quota|usage not included|\b429\b", re.I)
AUTH = re.compile(r"not logged in|not signed in|codex login|log in again|login|unauthori[sz]ed|\b401\b|credentials|"
                  r"refresh token|sign in", re.I)
CONTEXT = re.compile(r"context window exceeded|context_length_exceeded|maximum context length|exceeds the context|"
                     r"ran out of room in the model's context window", re.I)  # the last is Codex 0.159.3's own wording
TRANSIENT = re.compile(r"\b5\d\d\b|overloaded|temporar|disconnect|connection|timed out|reset by peer", re.I)


class CodexError(RuntimeError):
    """An exec that did not complete; the message says why. The episode stops and a rerun starts it again."""


class CodexMissing(CodexError):
    """No usable `codex` command, a model it does not list, or personal instructions in its home: nothing can run."""


class CodexAuth(CodexError):
    """Codex is not logged in (or its login expired): nothing can run until `codex login`."""


class CodexLimit(CodexError):
    """The plan's usage or rate limit was reached."""


class CodexTimeout(CodexError):
    """The exec ran past its timeout and was killed with everything it started."""


class ToolEscaped(CodexError):
    """An exec used something other than the tool server: the restriction failed, so the run is invalid."""


@dataclass(frozen=True)
class Result:
    """A finished exec. `usage` has `llm_client`'s names (prompt, completion, cached, cache write and reasoning
    tokens) and a cost of 0; it is the running total of the thread, as Codex prints it (`codex_log` takes the exec's
    own share), and zeros when the turn did not complete (`completed` false: the harness killed it, and a killed turn
    prints no usage)."""

    thread: str
    messages: list[str]
    usage: dict
    events: list[dict]
    argv: list[str]
    latency_s: float
    attempts: int
    warnings: list[str]
    stderr: str
    completed: bool = True
    killed: bool = False


def toml(value: str | os.PathLike) -> str:
    """`value` as a TOML literal string with forward slashes (no escapes to get wrong in an argument)."""
    text = str(value).replace("\\", "/")
    if "'" in text or "\n" in text:
        raise CodexError(f"cannot pass {text!r} to codex: it holds a quote or a newline")
    return f"'{text}'"


def server_settings(name: str, command: list[str], tool_timeout: float) -> list[str]:
    """The `-c` settings of one MCP server: its command and arguments, required, its calls pre-approved, the tool
    timeout."""
    return [f"mcp_servers.{name}.command={toml(command[0])}",
            f"mcp_servers.{name}.args=[{','.join(toml(a) for a in command[1:])}]",
            *(f"mcp_servers.{name}.{c}" for c in SERVER_CONFIG),
            f"mcp_servers.{name}.tool_timeout_sec={tool_timeout:g}"]


def classify(message: str) -> type[CodexError] | type[ContextTooLong]:
    """The error class a failure message belongs to."""
    if CONTEXT.search(message):
        return ContextTooLong
    if LIMIT.search(message):
        return CodexLimit
    if AUTH.search(message):
        return CodexAuth
    return CodexError


def failure_reason(error: BaseException) -> str:
    """The word that counts a failed exec: `timeout`, `limit`, `auth`, `context`, `escaped` or `missing`, else
    `error`. The journal and the case's failure counts use the same words."""
    for kind, word in ((CodexTimeout, "timeout"), (CodexLimit, "limit"), (CodexAuth, "auth"),
                       (ContextTooLong, "context"), (ToolEscaped, "escaped"), (CodexMissing, "missing")):
        if isinstance(error, kind):
            return word
    return "error"


def _message(text: str) -> str:
    """The reason in an error event's message: the inner message when it is a JSON error body, else the text."""
    try:
        found = json.loads(text)
    except ValueError:
        return text
    inner = found.get("error") if isinstance(found, dict) else None
    return str(inner.get("message") or text) if isinstance(inner, dict) else text


def json_line(line: str):
    """The JSON value of a line of an exec's output, None when it is not JSON."""
    try:
        return json.loads(line)
    except ValueError:
        return None


def _usage(raw: dict) -> dict:
    """`llm_client`'s usage names for Codex's: the prompt tokens include the cached ones, as there."""
    return {"prompt_tokens": raw.get("input_tokens", 0), "completion_tokens": raw.get("output_tokens", 0),
            "cached_tokens": raw.get("cached_input_tokens", 0),
            "cache_write_tokens": raw.get("cache_write_input_tokens", 0),
            "reasoning_tokens": raw.get("reasoning_output_tokens", 0), "cost": 0.0}


def check_event(event: dict, memory: str | None = None) -> None:
    """Raise `ToolEscaped` for an event that holds an item that is not a message, a thought or a call of the tool
    server or of the run's memory (`memory`, the name of its server; Codex's own resource listers aside), and
    `CodexError` for a call held for approval (the pre-approval did not apply). Anything else is fine. The one owner of
    the check: every event of every exec is run through it as it arrives."""
    if event.get("type") not in ("item.started", "item.completed", "item.updated"):
        return
    item = event.get("item") or {}
    what = item.get("type")
    if what not in ITEMS:
        raise ToolEscaped(f"codex used {what!r}, which is not one of the tool server's tools: the restriction failed")
    if what != "mcp_tool_call" or item.get("tool") in HELPERS.get(item.get("server"), ()):
        return
    if item.get("server") not in (SERVER, *([memory] if memory else [])):
        raise ToolEscaped(f"codex called the tool {item.get('tool')!r} of the server {item.get('server')!r}, not one "
                          f"of {SERVER!r}" + (f" or the memory's {memory!r}" if memory else ""))
    if event["type"] == "item.completed" and "requires approval" in str((item.get("error") or {}).get("message", "")):
        raise CodexError("a tool call was held for approval: the pre-approval did not apply")


def digest(events: list[dict]) -> dict:
    """What the events of an exec say: `thread` (the id of `thread.started`, "" if none), `messages` (the agent's
    replies), `warnings` (error items), `usage` (`turn.completed`'s running total in `llm_client`'s names, zeros without
    one) and `completed` (a `turn.completed` came)."""
    thread, messages, warnings, usage, completed = "", [], [], {}, False
    for e in events:
        kind = e.get("type")
        if kind == "thread.started":
            thread = str(e.get("thread_id") or "")
        elif kind == "turn.completed":
            usage, completed = {k: v for k, v in (e.get("usage") or {}).items() if isinstance(v, int)}, True
        elif kind == "item.completed":
            item = e.get("item") or {}
            if item.get("type") == "agent_message":
                messages.append(str(item.get("text") or ""))
            elif item.get("type") == "error":
                warnings.append(str(item.get("message") or ""))
    return {"thread": thread, "messages": messages, "warnings": warnings, "usage": _usage(usage),
            "completed": completed}


def failure(code: int, events: list[dict], err: str) -> tuple[type, str] | None:
    """(error class, message) when the exec did not complete, else None. Exit code 0 is not enough: the turn must have
    completed and not failed (an `error` event alone, such as a reconnect notice, is a warning)."""
    done = any(e.get("type") == "turn.completed" for e in events)
    if code == 0 and done and not any(e.get("type") == "turn.failed" for e in events):
        return None
    text = ""
    for e in [e for e in events if e.get("type") in ("turn.failed", "error")]:
        found = e.get("error", {}).get("message") if isinstance(e.get("error"), dict) else e.get("message")
        text = _message(str(found or ""))
    text = text or err.strip()[-500:] or f"codex exec exited {code} without finishing the turn"
    return classify(text), text


def acted(events: list[dict]) -> bool:
    """True when the model did something in the exec: a reply or a tool call (a retry would repeat it)."""
    return any(e.get("type") == "item.completed" and e["item"].get("type") in ("agent_message", "mcp_tool_call")
               for e in events)


def transient(error: BaseException, events: list[dict]) -> bool:
    """True for a failure worth the one retry: a plain `CodexError` whose message looks transient (a 5xx, a dropped
    connection) in an exec in which the model did nothing yet."""
    return type(error) is CodexError and TRANSIENT.search(str(error)) is not None and not acted(events)


_MODELS: dict[tuple, dict] = {}  # the installed Codex's catalog, by command: it is read once a process
_LOCK = threading.Lock()


class Codex:
    """The `codex` command for one model. `command` is the executable and its leading arguments (default: `codex` on
    PATH, a `.cmd` shim resolved); `env` is added to the environment. `prepare` checks the command and writes the
    patched model catalog; `argv` is the command line of an exec and `popen` starts it; `close` removes the catalog (so
    does dropping the object). Safe to share between threads. `guide` is a memory's guide (`codex_memory`), the text
    that follows the one-line base instruction in the file Codex reads it from; None, the reference agent, leaves the
    file as that line alone. `compact_at` is a number of tokens (`model_auto_compact_token_limit`) past which Codex
    compacts the thread, passed to every exec; None, the reference agent, leaves Codex's own limit."""

    def __init__(self, model: str, *, reasoning: str = "low", command: list[str] | None = None,
                 env: dict[str, str] | None = None, sleep: Callable[[float], None] = time.sleep,
                 guide: str | None = None, compact_at: int | None = None):
        if reasoning not in EFFORTS:
            raise ValueError(f"reasoning must be one of {list(EFFORTS)}, got {reasoning!r}")
        if compact_at is not None and not (type(compact_at) is int and compact_at >= 1):   # a bool is no count
            raise ValueError(f"compact_at must be a whole number of tokens, at least 1, got {compact_at!r}")
        self.model, self.reasoning, self.guide, self.compact_at = model, reasoning, guide, compact_at
        self.env, self._sleep, self._command, self._version = env or {}, sleep, command, None
        self._home: Path | None = None
        self._remove: Callable[[], object] | None = None
        self.catalog: Path | None = None  # the patched model catalog, and the file of the one-line base instruction
        self.instructions: Path | None = None

    @property
    def command(self) -> list[str]:
        if self._command is None:
            found = shutil.which("codex")
            if not found:
                raise CodexMissing("the codex command is not on PATH: install the Codex CLI and run `codex login`")
            self._command = [found]
        return self._command

    def _call(self, *args: str, timeout: float = 60.0) -> str:
        try:
            done = subprocess.run([*self.command, *args], capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", stdin=subprocess.DEVNULL, timeout=timeout,
                                  env={**os.environ, **self.env})
        except (OSError, subprocess.TimeoutExpired) as error:
            raise CodexMissing(f"cannot run `{self.command[0]} {' '.join(args)}`: {error}") from None
        if done.returncode:
            raise CodexMissing(f"`codex {' '.join(args)}` exited {done.returncode}: {done.stderr.strip()[-300:]}")
        return done.stdout

    def home(self) -> Path:
        """The folder Codex reads its personal files from: `CODEX_HOME` (this object's `env` first), else `~/.codex`."""
        found = self.env.get("CODEX_HOME") or os.environ.get("CODEX_HOME")
        return Path(found) if found else Path.home() / ".codex"

    def personal(self) -> list[Path]:
        """The non-empty personal instruction files of the Codex home (`PERSONAL`); only their existence and size are
        looked at, never their content."""
        found = [self.home() / name for name in PERSONAL]
        return [p for p in found if p.is_file() and p.stat().st_size > 0]

    @property
    def version(self) -> str:
        """`codex --version`, e.g. `codex-cli 0.159.3`."""
        if self._version is None:
            self._version = self._call("--version").strip()
        return self._version

    def prepare(self) -> None:
        """Check the command and write the patched model catalog. Raises CodexMissing (no command, an unlisted model, an
        effort the model's catalog entry does not list, or personal instructions in the Codex home); idempotent. `run`
        and `eval` call it before any exec, so a refused effort costs nothing."""
        with _LOCK:
            if self.catalog is not None:
                return
            if found := self.personal():
                raise CodexMissing(
                    f"{found[0]} holds your personal Codex instructions, which Codex adds to every exec as a message "
                    "the reference agent is not sent, so the run would not be the reference agent: move it aside for "
                    "the run (and put it back after)")
            _ = self.version
            key = tuple(self.command)
            if key not in _MODELS:
                try:
                    _MODELS[key] = json.loads(self._call("debug", "models"))
                except ValueError:
                    raise CodexMissing("`codex debug models` did not print JSON") from None
            models = [dict(m) for m in _MODELS[key].get("models", [])]
            slugs = [m.get("slug") for m in models]
            if self.model not in slugs:
                raise CodexMissing(f"codex does not list the model {self.model!r}; it lists "
                                   f"{', '.join(map(str, slugs))}")
            for m in models:
                if m["slug"] == self.model:
                    listed = [level.get("effort") if isinstance(level, dict) else level
                              for level in m.get("supported_reasoning_levels") or ()]
                    if listed and self.reasoning not in listed:  # an entry that lists none leaves the choice to Codex
                        raise CodexMissing(f"codex lists the model {self.model!r} at the reasoning efforts "
                                           f"{', '.join(map(str, listed))}, not {self.reasoning!r}")
                    m.update(CATALOG_PATCH)
            self._home = Path(tempfile.mkdtemp(prefix="stbench-codex-catalog-"))
            self._remove = weakref.finalize(self, remove_tree, self._home)  # also when `close` is never called
            self.catalog, self.instructions = self._home / "models.json", self._home / "instructions.md"
            self.catalog.write_text(json.dumps({"models": models}), encoding="utf-8", newline="\n")
            text = BASE_INSTRUCTIONS if not self.guide else f"{BASE_INSTRUCTIONS}\n\n{self.guide}"
            self.instructions.write_text(text, encoding="utf-8", newline="\n")

    def close(self) -> None:
        if self._home is not None:
            self._remove()
            self._home = self.catalog = self.instructions = None

    def backoff(self) -> None:
        """Wait `RETRY_WAIT` seconds, the pause before the one retry of a transient failure."""
        self._sleep(RETRY_WAIT)

    # -- one exec ---------------------------------------------------------------------------------------------------
    def argv(self, kind: str, thread: str | None, server_config: Path, empty: Path, *,
             tool_timeout: float = TOOL_TIMEOUT_S, memory: tuple[str, list[str]] | None = None) -> list[str]:
        """The command line of an exec: `kind` is `start` or `resume` (which takes `thread`), `server_config` the tool
        server's configuration file and `empty` an empty folder (a start's `-C`). The prompt is on stdin. `memory` is a
        memory's MCP server, `(name, argv)` (`codex_memory.Memory.server`), configured as the tool server is and with
        the same timeouts; None (the reference agent) adds nothing."""
        if kind not in KINDS:
            raise ValueError(f"an exec is one of {list(KINDS)}, got {kind!r}")
        if (kind == "start") != (thread is None):
            raise ValueError(f"a {kind!r} exec {'has no' if kind == 'start' else 'needs a'} thread, got {thread!r}")
        if self.catalog is None:
            raise CodexMissing("Codex.prepare() has not been called")
        args = [*self.command, "exec", *([] if kind == "start" else [kind]), "--json", "--skip-git-repo-check",
                "--ignore-user-config", "--ignore-rules", "--strict-config", "-m", self.model]
        server = [sys.executable, "-m", "stbench.codex_mcp", str(server_config)]
        settings = [*CONFIG, f"model_reasoning_effort={self.reasoning}",
                    *([] if self.compact_at is None else [f"model_auto_compact_token_limit={self.compact_at}"]),
                    f"model_catalog_json={toml(self.catalog)}",
                    f"model_instructions_file={toml(self.instructions)}",
                    *server_settings(SERVER, server, tool_timeout),
                    *(server_settings(memory[0], memory[1], tool_timeout) if memory is not None else [])]
        for setting in settings:
            args += ["-c", setting]
        for feature in DISABLED:
            args += ["--disable", feature]
        args += ["-C", str(empty)] if kind == "start" else []
        return [*args, "--", *([thread] if thread else []), "-"]

    def popen(self, argv: list[str], cwd: Path, stderr) -> subprocess.Popen:
        """Start an exec: stdin and stdout are pipes, stderr is `stderr` (an open file), and on POSIX it leads a session
        of its own so that `kill_tree` can end it with what it started. A `.cmd` shim on Windows is run by cmd.exe, so
        an argument it would misread and a command line it would cut are refused first (CodexError); CodexMissing when
        it cannot be run at all."""
        if os.name == "nt" and argv[0].lower().endswith((".cmd", ".bat")):
            for arg in argv[len(self.command):]:
                if UNSAFE.search(arg):
                    raise CodexError(f"an argument holds a character that cmd.exe reads as syntax: {arg[:80]!r}")
            if (size := len(subprocess.list2cmdline(argv))) > CMD_LIMIT:
                raise CodexError(f"the command line is {size} characters, over the {CMD_LIMIT} that cmd.exe passes to "
                                 "a .cmd shim")
        try:
            return subprocess.Popen(argv, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                    env={**os.environ, **self.env}, start_new_session=os.name != "nt")
        except OSError as error:
            raise CodexMissing(f"cannot run {argv[0]}: {error}") from None
