"""The `claude` command the Claude Code agent runs (`claude_exec`): the argv that holds the model to the two tools of
the tool server, the model ids, how a failure is named from the events of a turn, how usage and cost are read, and what
is checked on every event. No process, no model, no network: every expected value is written out by hand.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from stbench import agent_tools, codex_mcp
from stbench.agent_tools import ContextTooLong
from stbench.claude_exec import ALLOWED, Claude, check_event, classify, digest, failure, model_id, usage_of
from stbench.codex_exec import CodexAuth, CodexError, CodexLimit, ToolEscaped
from stbench.codex_log import image_hash, scrubbed


def test_the_argv_holds_the_model_to_the_two_tools_of_the_server():
    """Contract (docs/claude.md, the isolation flags): the process is launched `-p` with stream-json both ways (and
    `--verbose`, which stream-json output needs), the model by id and the effort, no setting source (`""`), no built-in
    tool (`--tools ""`), no skill or slash command, only the tool server of the `--mcp-config` file, the two tools
    pre-approved by name under `dontAsk` (anything else is denied, never asked), nothing saved to disk and the
    reference agent's system prompt in place of Claude Code's own. Written out whole, by hand. Regression caught: a
    tool left on (a missing or non-empty `--tools`), the user's settings, hooks, plugins or CLAUDE.md reaching the
    model, a system prompt appended to Claude Code's (`--append-system-prompt`) instead of replacing it, `--bare`
    (which ignores the OAuth login a subscription runs on), a permission mode that asks or allows more (`default`,
    `bypassPermissions`), or a transcript written to the user's disk."""
    claude = Claude("opus", reasoning="xhigh", command=["claude"])
    assert claude.argv(Path("mcp.json"), "THE PROMPT") == [
        "claude", "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
        "--model", "claude-opus-5-5", "--effort", "xhigh", "--setting-sources", "", "--tools", "",
        "--disable-slash-commands", "--strict-mcp-config", "--mcp-config", "mcp.json",
        "--allowedTools", "mcp__stbench__next_frame", "mcp__stbench__submit_answer",
        "--permission-mode", "dontAsk", "--no-session-persistence", "--system-prompt", "THE PROMPT"]


def test_the_allow_list_is_exactly_the_tools_the_server_offers():
    """Contract: `--allowedTools` names `mcp__stbench__<tool>` for each tool `codex_mcp` serves and no other, and the
    `--mcp-config` file names that server (run as `python -m stbench.codex_mcp CONFIG`) and no other. Regression
    caught: a tool added to the server but not to the allow-list (denied in `dontAsk`, so the model could not use it)
    or an allowed name the server does not offer, or a second server in the config."""
    offered = [f"mcp__stbench__{spec['name']}" for spec in codex_mcp.tool_specs()]
    assert list(ALLOWED) == offered == [f"mcp__stbench__{t['function']['name']}" for t in agent_tools.TOOLS]
    config = Claude("sonnet", command=["claude"]).mcp_config(Path("server.json"))
    ((name, spec),) = config["mcpServers"].items()
    assert name == "stbench" and spec["args"] == ["-m", "stbench.codex_mcp", "server.json"]
    assert spec["type"] == "stdio"


@pytest.mark.parametrize(("given", "wanted"), [
    ("haiku", "claude-haiku-5-5"), ("sonnet", "claude-sonnet-5-5"), ("opus", "claude-opus-5-5"),
    ("claude-sonnet-4-5", "claude-sonnet-4-5"), ("claude-opus-5-5", "claude-opus-5-5")])
def test_the_short_names_are_the_ids_the_journal_and_the_price_table_carry(given, wanted):
    """Contract: `haiku`, `sonnet` and `opus` stand for `claude-haiku-5-5`, `claude-sonnet-5-5` and
    `claude-opus-5-5`, a full `claude-...` id is used as given, and the transport carries the id. Regression caught:
    an alias recorded as the model (a summary and a usage report that no price row matches) or an id rewritten."""
    assert model_id(given) == wanted == Claude(given, command=["claude"]).model


@pytest.mark.parametrize("bad", ["gpt-6-luna", "fable", "", "claude-", "claude sonnet", "Sonnet"])
def test_a_name_that_is_no_claude_model_is_refused(bad):
    """Contract: anything but the three short names and a `claude-...` id is a ValueError that lists the choices (the
    driver turns it into a refusal before anything is sent). Regression caught: a typo or a Codex model id passed to
    `--model` and spent on."""
    with pytest.raises(ValueError, match="haiku, sonnet, opus"):
        model_id(bad)


def test_the_reasoning_is_the_clis_effort_and_only_its_levels_are_taken():
    """Contract: `--reasoning` maps to `--effort` for low, medium, high, xhigh and max, and nothing else (Codex's
    `none` among them) is a ValueError. Regression caught: an effort the CLI would reject reaching the first call."""
    for effort in ("low", "medium", "high", "xhigh", "max"):
        argv = Claude("haiku", reasoning=effort, command=["claude"]).argv(Path("m.json"), "p")
        assert argv[argv.index("--effort") + 1] == effort
    with pytest.raises(ValueError, match="reasoning must be one of"):
        Claude("haiku", reasoning="none", command=["claude"])


@pytest.mark.parametrize(("message", "kind"), [
    ("You've hit your session limit · resets 3pm", CodexLimit),
    ("Claude AI usage limit reached|1760000000", CodexLimit),
    ("API Error: 429 {\"type\":\"error\",\"error\":{\"type\":\"rate_limit_error\"}}", CodexLimit),
    ("API Error: 529 Overloaded", CodexLimit),
    ("overloaded_error: Overloaded", CodexLimit),
    ("Credit balance is too low", CodexLimit),
    ("Invalid API key · Please run /login", CodexAuth),
    ("OAuth token has expired. Please obtain a new token or refresh your existing token.", CodexAuth),
    ("authentication_failed: 401", CodexAuth),
    ("Prompt is too long", ContextTooLong),
    ("API Error: 400 Too many images in the request: 101 > 100", ContextTooLong),
    ("API Error: 400 prompt is too long: 250000 tokens > 200000 maximum", ContextTooLong),
    ("API Error: 500 Internal server error", CodexError),
    ("the model exploded", CodexError)])
def test_a_failure_is_named_by_its_message(message, kind):
    """Contract (the vocabulary of how a run fails, shared with Codex): a usage or rate limit, a 429 and an overloaded
    API are a limit; a login problem is auth; a full context window is `ContextTooLong`; anything else is a plain
    error (exactly `CodexError`, which counts as `error`). Regression caught: an overloaded API read as a model error
    (no reason in the counts), a limit read as auth, or a full window that ends the case as an unnamed error."""
    assert classify(message) is kind


def _assistant(*blocks, **more) -> dict:
    return {"type": "assistant", "session_id": "s1", "message": {"id": "m1", "content": list(blocks)}, **more}


def _result(**more) -> dict:
    return {"type": "result", "subtype": "success", "is_error": False, "result": "noted", "session_id": "s1", **more}


def test_a_turn_fails_by_what_its_events_say():
    """Contract (`claude_exec.failure`): a `success` result that is not an error and lists no denial is no failure; an
    error result is named by its text, or its `errors`, or its subtype; an API error carried by an assistant message
    names a turn that has no result; a turn with no result at all by the end of stderr, or by the exit code; a tool
    the permission mode denied is a plain error that says so. Regression caught: an error result taken as a good turn
    (the answers would be read as made), a failed process that reports nothing, or a denied allowed tool that stalls
    the case with no reason."""
    assert failure([_result()], None, "") is None
    overloaded = _result(is_error=True, result="API Error: 529 Overloaded")
    assert failure([_assistant({"type": "text", "text": "x"}), overloaded], None, "") == (
        CodexLimit, "API Error: 529 Overloaded")
    broken = _result(subtype="error_during_execution", is_error=True, result="", errors=["boom", "bang"])
    assert failure([broken], 0, "") == (CodexError, "boom; bang")
    assert failure([_result(subtype="error_max_turns", is_error=True, result="")], 0, "") == (
        CodexError, "error_max_turns")
    carried = _assistant({"type": "text", "text": "You've hit your limit · resets 8pm"}, error="rate_limit")
    assert failure([carried], 1, "") == (CodexLimit, "rate_limit: You've hit your limit · resets 8pm")
    assert failure([{"type": "system", "subtype": "init"}], 1, "Invalid API key · Please run /login\n") == (
        CodexAuth, "Invalid API key · Please run /login")
    assert failure([], 3, "") == (CodexError, "claude exited 3 without finishing the turn")
    denied = _result(permission_denials=[{"tool_name": "mcp__stbench__next_frame", "tool_use_id": "t1"}])
    kind, text = failure([denied], None, "")
    assert kind is CodexError and "denied" in text and "mcp__stbench__next_frame" in text


def test_the_usage_and_cost_are_the_results_in_the_names_the_journal_keeps():
    """Contract (`claude_exec.usage_of`, `digest`): the prompt tokens are the fresh, the cache-read and the
    cache-write ones together (8 + 60 + 30 = 98), the cached ones the cache reads, the completion tokens the output
    ones (the thinking ones inside them, 4), and the cost the result's `total_cost_usd`; a turn with no result has
    zeros and is not complete; the lines that were not events, the API retries and tools beyond the server's are its
    warnings. Regression caught: prompt tokens that leave out the cache (a price that is far too low), thinking
    tokens added to the output ones twice, or a missing field that raises."""
    usage = {"input_tokens": 8, "cache_read_input_tokens": 60, "cache_creation_input_tokens": 30, "output_tokens": 12,
             "output_tokens_details": {"thinking_tokens": 4}}
    assert usage_of(usage, 0.5) == {"prompt_tokens": 98, "completion_tokens": 12, "cached_tokens": 60,
                                    "cache_write_tokens": 30, "reasoning_tokens": 4, "cost": 0.5}
    assert usage_of({}, None) == {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
                                  "cache_write_tokens": 0, "reasoning_tokens": 0, "cost": 0.0}
    events = [{"type": "system", "subtype": "init", "session_id": "s1", "tools": [*ALLOWED, "Bash"]},
              {"type": "system", "subtype": "api_retry", "attempt": 1, "max_retries": 10, "error_status": 529,
               "error": "overloaded"},
              _assistant({"type": "text", "text": "looking"}, {"type": "tool_use", "name": ALLOWED[0]}),
              _result(usage=usage, total_cost_usd=0.5)]
    read = digest(events, ["[claude-code:unrecognized_model] {}"])
    assert read == {"thread": "s1", "messages": ["looking"], "completed": True, "usage": usage_of(usage, 0.5),
                    "warnings": ["[claude-code:unrecognized_model] {}", "tools offered beyond the server's: Bash",
                                 "api_retry 1/10, status 529: overloaded"]}
    cut = digest(events[:3], [])
    assert cut["completed"] is False and cut["usage"]["prompt_tokens"] == 0


def test_every_event_is_checked_for_a_tool_outside_the_server():
    """Contract (`check_event`, the one owner of the check): a `tool_use` of either tool of the server passes, as do
    Claude Code's two MCP resource helpers (which can only ask a server with no resources); a built-in tool, a tool of
    another server and a tool that does not exist are `ToolEscaped`; a permission request (`control_request`) and a
    tool server that failed to start are plain errors; text, results and other events pass. Regression caught: a
    model that reaches a shell or the web unnoticed, a case that goes on with no tool server, or a held approval that
    waits forever."""
    for name in (*ALLOWED, "ListMcpResourcesTool", "ReadMcpResourceTool"):
        check_event(_assistant({"type": "tool_use", "id": "t", "name": name, "input": {}}))
    check_event(_assistant({"type": "text", "text": "hello"}))
    check_event({"type": "user", "message": {"content": [{"type": "tool_result", "content": "Bash"}]}})
    check_event(_result())
    for name in ("Bash", "Read", "WebFetch", "mcp__other__next_frame", "next_frame", None):
        with pytest.raises(ToolEscaped):
            check_event(_assistant({"type": "text", "text": "x"}, {"type": "tool_use", "id": "t", "name": name}))
    with pytest.raises(CodexError, match="held for approval") as held:
        check_event({"type": "control_request", "request_id": "r", "request": {"subtype": "can_use_tool"}})
    assert type(held.value) is CodexError
    with pytest.raises(CodexError, match="'stbench' is failed"):
        check_event({"type": "system", "subtype": "init", "mcp_servers": [{"name": "stbench", "status": "failed"}]})
    check_event({"type": "system", "subtype": "init", "mcp_servers": [{"name": "stbench", "status": "connected"}]})


def test_an_image_block_in_the_apis_form_is_written_as_its_hash():
    """Contract (`codex_log.scrubbed`, which the journal runs every event through): the image block Claude Code
    prints for a tool result, `{"type": "image", "source": {"type": "base64", "data": ...}}`, keeps its other keys
    and gets `hash` in place of `data` (the MCP form already did); the text around it is untouched. Regression
    caught: a journal that holds every frame of a long case as base64 (hundreds of megabytes) when the events of a
    turn are the whole case."""
    jpeg = b"\xff\xd8\xff\xe0 a frame \xff\xd9"
    block = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.b64encode(jpeg).decode("ascii")}}
    event = {"type": "user", "message": {"content": [{"type": "tool_result", "content": [
        {"type": "text", "text": "[k3 ..."}, block]}]}}
    shown = scrubbed(event)["message"]["content"][0]["content"]
    assert shown == [{"type": "text", "text": "[k3 ..."},
                     {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                  "hash": image_hash(jpeg)}}]
    assert "data" not in str(shown) and image_hash(jpeg).startswith("<jpeg 15 bytes sha256 ")
