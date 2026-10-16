"""One `codex exec` (`codex_exec`, `codex_live`): the way every exec is launched, what is made of what comes back, and
how a failure is named, over the fake command (`codex_fake`). No `codex`, no model, no network.

The restriction is the contract that matters most: the model must reach nothing but the tool server. The live proof
that Codex itself then offers nothing else (a prompt that asks for a file by path, a directory, a shell command and a
web page) is a FINDINGS entry; these tests keep the line that makes it true on every exec.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path

import pytest

from codex_kit import Fake, run_exec
from stbench import codex_exec
from stbench.agent_tools import ContextTooLong
from stbench.codex_exec import (
    BASE_INSTRUCTIONS,
    Codex,
    CodexAuth,
    CodexError,
    CodexLimit,
    CodexMissing,
    CodexTimeout,
    ToolEscaped,
    check_event,
    classify,
    digest,
    failure_reason,
)
from stbench.codex_feed import Feed, Item
from stbench.codex_live import Live
from stbench.llm_render import END_MESSAGE

# the features the plan names as reaching files or the web, written out by hand
MUST_DISABLE = {"shell_tool", "unified_exec", "browser_use", "computer_use", "image_generation", "apps", "plugins",
                "multi_agent", "view_image"}
NEVER_PASSED = {"--dangerously-bypass-approvals-and-sandbox", "--full-auto", "--yolo", "--enable", "--add-dir", "--oss",
                "--output-schema", "--profile", "-p", "-s", "--sandbox", "--search", "--ask-for-approval", "-a", "-i"}


def settings(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == "-c"]


def test_every_exec_is_launched_with_the_restrictions_and_the_tool_server_only(tmp_path):
    """Contract (plan 4): a start and a resume each run with `--ignore-user-config --ignore-rules
    --skip-git-repo-check --strict-config`, `sandbox_mode=read-only` and `approval_policy=never`, web search off, the
    shell, exec, browser, computer use, image generation, apps, plugins, multi-agent and image viewing disabled, the
    instruction messages switched off, the tool server as the one required MCP server with its calls pre-approved and a
    tool timeout of 600 s (the pull's), and none of the flags that widen what Codex may do (no image is attached: no
    `-i`). A start also gets `-C` to an empty folder; the prompt is read from stdin and the argv ends `-- [thread] -`. A
    case is one thread: no other kind of exec exists, and no argv holds `--ephemeral`. The tool server is given the
    feed's address and key and nothing else of the case. Regression caught: any exec (the resume, say, which no one
    thinks to check) launched without the restriction, so a model could read a file or the web, the retired fork route
    returning (design §12), or a tool timeout of 120 s on an exec that pulls."""
    fake = Fake(tmp_path)
    first = run_exec(fake)
    second = run_exec(fake, kind="resume", thread=first.thread)
    for result, sub in ((first, None), (second, "resume")):
        argv = result.argv[len(fake.codex.command):]
        assert argv[:2] == ["exec", sub or "--json"]
        for flag in ("--json", "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules", "--strict-config"):
            assert argv.count(flag) == 1
        assert not NEVER_PASSED & set(argv) and "--ephemeral" not in argv
        cfg = settings(argv)
        for needed in ("sandbox_mode=read-only", "approval_policy=never", "web_search=disabled",
                       "skills.include_instructions=false", "include_environment_context=false",
                       "include_permissions_instructions=false", "include_collaboration_mode_instructions=false",
                       "include_apps_instructions=false", "model_reasoning_effort=low",
                       "mcp_servers.stbench.required=true", "mcp_servers.stbench.default_tools_approval_mode=approve",
                       "mcp_servers.stbench.tool_timeout_sec=600"):
            assert needed in cfg, needed
        assert not [c for c in cfg if c.startswith("mcp_servers.stbench.tool_timeout_sec=") and c.endswith("=120")]
        assert any(c.startswith("model_catalog_json='") for c in cfg)
        assert any(c.startswith("model_instructions_file='") for c in cfg)
        assert {c.split(".")[1] for c in cfg if c.startswith("mcp_servers.")} == {"stbench"}  # the one MCP server
        assert {argv[i + 1] for i, a in enumerate(argv) if a == "--disable"} >= MUST_DISABLE
        assert argv[-2:] == ["--", "-"] if sub is None else argv[-3:] == ["--", first.thread, "-"]
    execs = fake.execs()
    assert [e["sub"] for e in execs] == [None, "resume"]
    assert ["-C" in e["argv"] for e in execs] == [True, False] and execs[0]["cd_files"] == []
    assert [e["base"] for e in execs] == [None, first.thread]
    assert [set(e["config"]) for e in execs] == [{"address", "key"}] * 2 and execs[0]["tool_timeout"] == "600"
    for kind, thread in (("fork", first.thread), ("start", first.thread), ("resume", None)):
        with pytest.raises(ValueError, match="exec is one of|thread"):
            fake.codex.argv(kind, thread, tmp_path / "server.json", tmp_path / "empty")


def test_compaction_is_codex_default(tmp_path):
    """Contract (plan §6.5 N7, recommendations §2.4): Codex's own automatic compaction is left on, so no exec is
    launched with a compaction limit of any size: neither the start nor the resume holds
    `model_auto_compact_token_limit`, and `CONFIG`, the settings every exec shares, names none. Regression caught: the
    override that once switched the agent's context management off (a limit of 100,000,000 tokens) coming back, so a
    long case would overflow the window instead of being compacted as it would be in use."""
    fake = Fake(tmp_path)
    fake.codex.prepare()
    for kind, thread in (("start", None), ("resume", "thread-0000")):
        argv = fake.codex.argv(kind, thread, tmp_path / "server.json", tmp_path / "empty")
        assert not [c for c in settings(argv) if "compact" in c.split("=")[0]]  # a `-c key=value` setting's key
    assert not [c for c in codex_exec.CONFIG if "compact" in c]


def test_the_working_folder_of_an_exec_is_empty_and_removed_after(tmp_path):
    """Contract (plan 4: `-C` an empty temp dir): the folder Codex starts in holds nothing (no image, not the server's
    configuration, not a trace), the `-C` folder of a start is empty, and the exec's temp folder is gone once
    `Live.close` has run, successful or not. Regression caught: a model shown a file of the harness's, or temp folders
    left behind by every exec of a run."""
    fake = Fake(tmp_path)
    run_exec(fake)
    (record,) = fake.execs()
    assert record["cwd_files"] == [] and record["cd_files"] == []
    assert not Path(record["cwd"]).exists() and not Path(record["cwd"]).parent.exists()
    fake.plan({1: {"fail": "boom"}})
    with pytest.raises(CodexError):
        run_exec(fake)
    assert not Path(fake.execs()[-1]["cwd"]).parent.exists()


def test_a_finished_exec_is_read_into_its_thread_replies_and_usage(tmp_path):
    """Contract: the thread is the id of `thread.started` (the same one for a resume), the replies are the agent
    messages, and the usage has `llm_client`'s names with the prompt tokens including the cached ones and a cost of 0,
    and is the thread's running total as Codex prints it (the fake reports 100 tokens in, 40 cached and 10 out for each
    exec of a thread, so a thread's second exec reports 200, 80 and 20). The model's tool calls are the feed's to
    journal, not the result's. Regression caught: tokens read from the wrong keys (a summary of zeros), or a running
    total taken for one exec's own tokens."""
    fake = Fake(tmp_path)
    first = run_exec(fake, prompt="[k0 2026-03-02T08:00:00] robot at (2, 1)")
    assert first.thread == "thread-0000" and first.messages == ["noted"] and first.attempts == 1
    assert first.completed and not first.killed
    assert first.usage == {"prompt_tokens": 100, "completion_tokens": 10, "cached_tokens": 40, "cache_write_tokens": 0,
                           "reasoning_tokens": 0, "cost": 0.0}
    assert first.latency_s > 0 and first.warnings == []
    assert [e["type"] for e in first.events][:2] == ["thread.started", "turn.started"]
    again = run_exec(fake, kind="resume", thread=first.thread)
    assert again.thread == first.thread
    assert (again.usage["prompt_tokens"], again.usage["cached_tokens"], again.usage["completion_tokens"]) == (
        200, 80, 20)


def test_the_events_of_an_exec_are_digested_by_hand_written_rules():
    """Contract: `digest` reads the thread from `thread.started`, the replies from completed agent messages, the
    warnings from error items and the usage from `turn.completed` (in `llm_client`'s names; zeros and `completed` false
    when the turn never completed). Regression caught: usage taken from an event that is not the turn's end, or a reply
    counted twice from its started and completed events."""
    events = [{"type": "thread.started", "thread_id": "T"},
              {"type": "item.started", "item": {"type": "agent_message", "text": "half"}},
              {"type": "item.completed", "item": {"type": "agent_message", "text": "done"}},
              {"type": "item.completed", "item": {"type": "error", "message": "reconnecting"}},
              {"type": "turn.completed", "usage": {"input_tokens": 7, "cached_input_tokens": 3, "output_tokens": 2,
                                                   "reasoning_output_tokens": 1}}]
    assert digest(events) == {"thread": "T", "messages": ["done"], "warnings": ["reconnecting"], "completed": True,
                              "usage": {"prompt_tokens": 7, "completion_tokens": 2, "cached_tokens": 3,
                                        "cache_write_tokens": 0, "reasoning_tokens": 1, "cost": 0.0}}
    cut = digest(events[:3])
    assert cut["completed"] is False and cut["usage"]["prompt_tokens"] == 0 and cut["messages"] == ["done"]


@pytest.mark.parametrize(("fail", "error", "words"), [
    ("limit", CodexLimit, "usage limit"), ("auth", CodexAuth, "codex login"),
    ("context", ContextTooLong, "context window"), ("boom", CodexError, "exploded")])
def test_a_failed_exec_is_named_by_its_message(tmp_path, fail, error, words):
    """Contract (plan 6): a usage or rate limit, an auth failure and a full context window are told apart from any
    other failure, each with Codex's own message; the exec ran once. Regression caught: a limit reported as a generic
    failure (the run says nothing useful)."""
    fake = Fake(tmp_path, {0: {"fail": fail}})
    with pytest.raises(error, match=words) as raised:
        run_exec(fake)
    assert type(raised.value) is error and len(fake.execs()) == 1


def test_exit_code_zero_is_not_success_and_a_foreign_tool_invalidates_the_run(tmp_path):
    """Contract (plan: `codex exec` can exit 0 when something was blocked): an exec that exits 0 without a completed
    turn is an error; one whose events hold an item that is not a message or a call of the tool server (a command, a
    file change, a web search) or a call of another MCP server is `ToolEscaped`, whatever else it did. Regression
    caught: a blocked exec read as a reply, or a restriction that failed and went unseen."""
    fake = Fake(tmp_path, {0: {"incomplete": True}, 1: {"escape": "command_execution"}, 2: {"escape": "web_search"},
                           3: {"server": "other"}, 4: {"server": "codex", "tool": "read_file"},
                           5: {"server": "codex", "tool": "list_mcp_resources"}})
    with pytest.raises(CodexError, match="exited 0 without finishing the turn"):
        run_exec(fake)
    for what in ("'command_execution'", "'web_search'", "of the server 'other'", "'read_file' of the server 'codex'"):
        with pytest.raises(ToolEscaped, match=what):
            run_exec(fake)
    assert run_exec(fake).messages == ["noted"]  # Codex's own resource lister (it finds none) is not an escape


@pytest.mark.parametrize(("event", "outcome"), [
    ({"type": "thread.started", "thread_id": "T"}, None),
    ({"type": "item.completed", "item": {"type": "agent_message", "text": "x"}}, None),
    ({"type": "item.started", "item": {"type": "reasoning"}}, None),
    ({"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "stbench", "tool": "next_frame"}}, None),
    ({"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "codex",
                                         "tool": "list_mcp_resource_templates"}}, None),
    ({"type": "item.started", "item": {"type": "command_execution", "command": "ls"}}, ToolEscaped),
    ({"type": "item.updated", "item": {"type": "file_change"}}, ToolEscaped),
    ({"type": "item.started", "item": {"type": "mcp_tool_call", "server": "web", "tool": "fetch"}}, ToolEscaped),
    ({"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "codex", "tool": "read_mcp_resource"}},
     ToolEscaped),
    ({"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "stbench", "tool": "submit_answer",
                                         "error": {"message": "tool call requires approval"}}}, CodexError)])
def test_each_event_is_checked_by_the_one_rule(event, outcome):
    """Contract: `check_event` is the one owner of the per-event rule, used on every event as it arrives: messages,
    thoughts, a call of the server's tools and Codex's two resource listers pass; any other item, a call of another
    server (or of Codex's own reader of resources) and a call held for approval stop the exec. Regression caught: a rule
    that is applied to the events of a finished exec only, or a helper that opens the door to a resource reader."""
    if outcome is None:
        check_event(event)
    else:
        with pytest.raises(outcome):
            check_event(event)


def test_an_exec_that_escapes_is_killed_when_the_event_arrives_not_at_its_end(tmp_path):
    """Contract (FD-frame-pull §3.3): every event is checked as it arrives, and an escape kills the exec at once with
    everything it started: the exec here would sleep a minute after a command item, and it is over within seconds,
    unfinished, and its result is `ToolEscaped`. Regression caught: an escape found only when the exec ends, which with
    one exec for a whole case would let a model use a tool for the rest of the case."""
    fake = Fake(tmp_path, {0: {"escape": "command_execution", "hang": 0, "hang_s": 60}})
    feed = Feed()
    began = time.monotonic()
    live = Live(fake.codex, kind="start", thread=None, prompt="hello", server=feed.server_config())
    try:
        assert live.wait(45)
        with pytest.raises(ToolEscaped, match="command_execution"):
            live.result()
    finally:
        live.close()
        feed.close()
    assert time.monotonic() - began < 45 and not fake.execs()[0]["finished"]


def test_the_events_of_a_live_exec_hold_an_image_as_its_hash_and_never_as_base64(tmp_path):
    """Contract (FD-frame-pull §3.3, "never base64 in the journal"; a case is one long exec that may carry hundreds of
    frames): the events `Live` keeps hold each image block of a tool result as its hash, so neither the exec's result
    nor the journal nor the harness's memory grows by a frame an event. The fake prints the reply's content with the
    image's base64, as a Codex event may. Regression caught: base64 of every frame kept in the events of a long exec."""
    fake = Fake(tmp_path)
    feed = Feed()
    image = b"\xff\xd8\xff\xe0frame-7\xff\xd9"
    ticket = feed.supply(Item("frame", "[k7 c] h", image, k=7))
    live = Live(fake.codex, kind="start", thread=None, prompt="hello", server=feed.server_config())
    try:
        assert feed.wait(ticket, time.monotonic() + 60, live.alive) == "pulled"
        feed.supply(Item("end", END_MESSAGE))
        assert live.wait(60)
        result = live.result()
    finally:
        live.close()
        feed.close()
    text = json.dumps(result.events)
    assert base64.b64encode(image).decode() not in text and "<jpeg 13 bytes sha256 c0eb4e1e112bf53f>" in text
    done = [e["item"]["result"]["content"] for e in result.events
            if e["type"] == "item.completed" and e["item"].get("tool") == "next_frame"]
    assert [[block["type"] for block in content] for content in done] == [["text", "image"], ["text"]]


def test_an_exec_the_harness_kills_has_a_result_with_no_usage(tmp_path):
    """Contract: `Live.kill` ends the exec and what it started; its result is a `Result` with `killed` true, no
    completed turn and zero usage (a killed turn prints none), and not an error. Regression caught: a killed exec read
    as a failed one (the case would end on a message limit it ended itself), or its zeros read as the thread's total
    (the next exec's share would be negative)."""
    fake = Fake(tmp_path, {0: {"sleep": 60}})
    feed = Feed()
    live = Live(fake.codex, kind="start", thread=None, prompt="hello", server=feed.server_config())
    try:
        assert live.alive()
        began = time.monotonic()
        live.kill()
        result = live.result()
    finally:
        live.close()
        feed.close()
    assert not live.alive() and time.monotonic() - began < 30
    assert result.killed and not result.completed and result.usage["prompt_tokens"] == 0


@pytest.mark.skipif(os.name != "nt", reason="a .cmd shim is run by cmd.exe on Windows only")
def test_a_cmd_shim_is_not_given_what_cmd_exe_would_misread(tmp_path):
    """Contract (the Windows install of Codex is a `.cmd` shim, run by cmd.exe): an argument with a character cmd.exe
    reads as syntax (`&`, `|`, `%`, ...) and a command line past what cmd.exe passes on are refused with the reason,
    before anything is run. Regression caught: a path with `&` that cmd.exe splits into a second command, or a command
    line that is silently cut."""
    shim = tmp_path / "codex.cmd"
    codex = Codex("fake-model", command=[str(shim)])
    with pytest.raises(CodexError, match="cmd.exe reads as syntax: 'a&b'"):
        codex.popen([str(shim), "exec", "a&b"], tmp_path, None)
    many = [str(shim), "exec", *(f"-c{n}.jpg" for n in range(1500))]
    with pytest.raises(CodexError, match="over the 8000 that cmd.exe passes"):
        codex.popen(many, tmp_path, None)


def test_a_missing_command_or_model_stops_before_any_exec(tmp_path, monkeypatch):
    """Contract (plan 6): no `codex` on PATH, a command that cannot run, and a model the installed Codex does not list
    are `CodexMissing` from `prepare`, with a message that says what to do, before any exec. Regression caught: a
    traceback from `subprocess`, or a first exec that fails with Codex's own, less helpful message."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(CodexMissing, match="not on PATH.*codex login"):
        Codex("fake-model").prepare()
    with pytest.raises(CodexMissing, match="cannot run `.*no-such-codex --version`"):
        Codex("fake-model", command=[str(tmp_path / "no-such-codex")]).prepare()
    fake = Fake(tmp_path, model="gpt-nine")
    with pytest.raises(CodexMissing, match="does not list the model 'gpt-nine'; it lists fake-model, other-model"):
        fake.codex.prepare()
    assert fake.execs() == []


@pytest.mark.parametrize("name", codex_exec.PERSONAL)
def test_personal_instructions_in_the_codex_home_refuse_the_run(tmp_path, monkeypatch, name):
    """Contract (`codex_exec` docstring): Codex adds the `AGENTS.md` or `AGENTS.override.md` of its home folder
    (`CODEX_HOME`, else `~/.codex`) to every exec as a message whatever the flags, so `prepare` refuses a home that
    holds a non-empty one, naming the file and before any exec, and accepts an absent or empty one; the transport's own
    `env` wins over the process environment. Regression caught: a user with personal instructions whose run is labelled
    `codex:<model>`, though the model was shown text that the reference agent never sends."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(home))
    fake = Fake(tmp_path)
    (home / name).write_text("", encoding="utf-8")
    fake.codex.prepare()  # an empty file adds no message
    fake.codex.close()
    (home / name).write_text("Always answer in French.\n", encoding="utf-8")
    refused = fake.new()
    with pytest.raises(CodexMissing, match=re.escape(f"{home / name} holds your personal Codex instructions")):
        refused.prepare()
    assert refused.catalog is None and fake.execs() == []
    other = tmp_path / "other"
    other.mkdir()
    refused.env = {**refused.env, "CODEX_HOME": str(other)}
    refused.prepare()  # a home that holds none
    refused.close()


@pytest.fixture(scope="module")
def home_at_module_setup():
    """The Codex home as a module-scoped fixture sees it, which pytest sets up before any function-scoped fixture."""
    return Codex("fake-model").home()


def test_a_module_scoped_fixture_never_sees_the_developers_codex_home(home_at_module_setup, tmp_path_factory):
    """Contract (`conftest.no_codex_home`): the Codex home is a folder of the test run for every fixture, whatever its
    scope, not the developer's `~/.codex`; `test_codex_system` runs a whole case in a module-scoped fixture, and
    `prepare` refuses a home that holds a personal `AGENTS.md`. Regression caught: the isolation made function-scoped
    again, so the case errors at setup, six tests, on a machine with a personal `AGENTS.md` (WSL) while passing on one
    without (Windows)."""
    assert home_at_module_setup.is_relative_to(tmp_path_factory.getbasetemp())
    assert home_at_module_setup != Path.home() / ".codex"


def test_the_catalog_is_codex_own_with_only_the_chosen_model_patched(tmp_path):
    """Contract (`codex_exec` docstring): the model catalog handed to Codex is the installed Codex's own with the chosen
    model's entry set to direct tool mode, no apply_patch, no experimental tools, no multi-agent and no tool search, and
    nothing else touched (another model's entry and the entry's other keys are as Codex printed them); the base
    instruction file holds the one line. `close` removes both. Regression caught: a patch that rewrites the whole
    catalog, or one that leaves the model on `code_mode_only`, where it would reach tools through JavaScript."""
    fake = Fake(tmp_path)
    fake.codex.prepare()
    models = {m["slug"]: m for m in json.loads(fake.codex.catalog.read_text(encoding="utf-8"))["models"]}
    assert models["other-model"] == {"slug": "other-model", "tool_mode": None}
    mine = models["fake-model"]
    assert (mine["tool_mode"], mine["apply_patch_tool_type"], mine["experimental_supported_tools"],
            mine["multi_agent_version"], mine["supports_search_tool"]) == ("direct", None, [], None, False)
    assert mine["base_instructions"] == "Codex coding agent"  # replaced by model_instructions_file, not the catalog
    assert fake.codex.instructions.read_text(encoding="utf-8") == BASE_INSTRUCTIONS
    folder = fake.codex.catalog.parent
    fake.codex.close()
    assert not folder.exists() and fake.codex.catalog is None


def test_the_error_classes_are_chosen_by_the_words_codex_uses():
    """Contract: `classify` picks the class from the wording of Codex's own messages (limits, login, context), limits
    first. Regression caught: an auth failure whose text mentions a limit read as a limit, or a model-not-supported
    error read as an auth failure because it says 'ChatGPT account'."""
    assert classify("You've hit your usage limit. Upgrade to Pro") is CodexLimit
    assert classify("429 Too Many Requests") is CodexLimit
    assert classify("Your access token could not be refreshed. Please log in again") is CodexAuth
    assert classify("not logged in") is CodexAuth
    assert classify("The 'x' model is not supported when using Codex with a ChatGPT account.") is CodexError
    assert classify("context window exceeded") is ContextTooLong
    assert classify("Codex ran out of room in the model's context window. Start a new thread or clear earlier "
                    "history before retrying.") is ContextTooLong  # the message Codex 0.159.3 prints at an overflow


def test_each_failure_has_one_word_that_counts_it():
    """Contract: `failure_reason` gives the word a failed exec is counted under in the journal and the case's failures:
    `timeout`, `limit`, `auth`, `context`, `escaped`, `missing`, and `error` for any other failure. Regression caught:
    the journal and the case's failure counts naming one failure two ways, or a context overflow counted as a generic
    error (it is the failure the plan says a long case may end on)."""
    errors = [CodexTimeout("t"), CodexLimit("l"), CodexAuth("a"), ContextTooLong("c"), ToolEscaped("e"),
              CodexMissing("m"), CodexError("x"), RuntimeError("y")]
    assert [failure_reason(e) for e in errors] == ["timeout", "limit", "auth", "context", "escaped", "missing",
                                                   "error", "error"]
