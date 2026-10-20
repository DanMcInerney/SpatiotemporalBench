# The Claude Code reference agent (`claude:<model>`)

The same reference agent as [the Codex one](codex.md), reached through the Claude Code CLI instead of the Codex CLI, so
that a Claude model (`haiku`, `sonnet` or `opus`) runs exactly the way a Codex model does and switching between the two is
the prefix of the system name. The prompt, the two tools (`next_frame`, `submit_answer`), the items the model is sent,
the held frame, the resume line, the timeouts, the journal, the summary, the leak scan, the rows and every option of
`stbench run` and `stbench eval` are the Codex agent's (`claude_system.ClaudeSystem` is a `CodexSystem`; `codex_feed`,
`codex_mcp`, `codex_asks` and `llm_render` are shared and unchanged). It needs `claude` on PATH (tested against
`2.1.284`; a native `claude.exe`, not a `.cmd` shim) and the login Claude Code already has: the subscription's, or an
API key. It spends that login's quota, or dollars on a key.

```sh
uv run stbench eval codex:gpt-6.1-sol --data data --select length=short --reasoning medium --workers 9 --out runs/sol
uv run stbench eval claude:sonnet     --data data --select length=short --reasoning high   --workers 3 --out runs/cc-sonnet
uv run stbench run  claude:opus DATASET/private/<case id> --reasoning xhigh --out runs/opus-<case id>
uv run stbench eval compare runs/sol runs/cc-sonnet --cells-from DEVROOT
```

| Option | Meaning |
| --- | --- |
| `<model>` | `haiku`, `sonnet` or `opus` (`claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`), or a full `claude-...` id as it is. The id is what `--model` gets and what the journal, the summaries and the price table carry. Anything else is refused (exit 2). |
| `--reasoning EFFORT` | `low` (the default, as for Codex), `medium`, `high`, `xhigh` or `max`: Claude Code's `--effort`, recorded in the summary. |
| `--channel`, `--brief`, `--during`, `--timeout`, `--text-only`, `--out` | As for the Codex agent ([codex.md](codex.md), "Running it"): `rgb_pose` (the default) or `detections`, the brief variant, whether the during-job questions are asked, the seconds from serving an item to the next pull, no image at all, and the folder. |
| `--memory SPEC` | A memory store beside the agent's own context, the Codex agent's option and SPEC ([a memory](#a-memory-beside-the-agent---memory-spec)). Refused on `detections`, with `--text-only` and for other agents. |
| `--compact-at TOKENS` | Make Claude Code compact the conversation once its context passes about TOKENS, earlier than its own limit: a whole number from 1 to 66,400, rounded up to a multiple of 800 ([compaction](#compaction---compact-at-tokens)). Refused for other agents. |

A folder that already holds a `claude.calls.jsonl` refuses the run (exit 2), and `stbench eval` removes such a case's
folder and runs the case again from its start: a Claude Code run cannot be replayed. The quota is the login's. A
subscription has a rolling usage limit; a case that hits it ends as `limit` (below) and reruns from its start with the
same `eval` command. Start at 3 workers (`--workers 3`) and raise it while no case ends as `limit`.

## One process a case

A case is ONE `claude -p` process in stream-json mode (`claude_live.ClaudeLive`). The first task is its first user
message; the model pulls frames with `next_frame` inside that turn, exactly as the Codex model does inside its exec. If
the model ends its turn without pulling, the same process is sent the fixed line (`llm_render.RESUME_MESSAGE`) as the
next user message on stdin, at most three times in a row, and then the case ends as `stalled`. Nothing is saved to disk
(`--no-session-persistence`): a process that dies has lost the conversation, so a failed turn is never retried (Claude
Code retries a transient API error itself, and its `api_retry` events are journaled as warnings). The model remembers
with its own context, which Claude Code compacts itself; the benchmark adds no memory tool and no instruction.

## The argv, and what each part is for

```
claude -p --input-format stream-json --output-format stream-json --verbose
       --model <id> --effort <level>
       --setting-sources "" --tools "" --disable-slash-commands
       --strict-mcp-config --mcp-config <file naming the stbench relay only>
       --allowedTools mcp__stbench__next_frame mcp__stbench__submit_answer
       --permission-mode dontAsk --no-session-persistence
       --system-prompt <the reference agent's system prompt>
```

With `--memory SPEC` the same command line gains two things and nothing else: the memory's server name after the two
tools in `--allowedTools` (`mcp__NAME`, every tool of that server) and its guide at the end of `--system-prompt`, after
a blank line; the `--mcp-config` file then names the memory's server beside the relay.

| Flag | Holds the model to the two tools by |
| --- | --- |
| `--tools ""` | no built-in tool: no shell, files, web, subagents or tool search. |
| `--strict-mcp-config --mcp-config` | the tool server (`codex_mcp`, a thin relay to the harness) as the only MCP server. |
| `--allowedTools` + `--permission-mode dontAsk` | the two tools by name run without a prompt; anything else is denied, never asked. |
| `--setting-sources ""` | no user, project or local settings: no hooks, plugins or permission rules of the user's. |
| `--disable-slash-commands` | no skills or slash commands. |
| `--system-prompt` | the reference agent's prompt replaces Claude Code's own; it is not appended to it. The first user message is the task. |
| `--no-session-persistence` | no transcript on the user's disk. |
| environment | `CLAUDE_CODE_DISABLE_CLAUDE_MDS=1` and `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1` (no CLAUDE.md of the user, a parent folder or the project, no auto memory), `MCP_TIMEOUT` 60 s and `MCP_TOOL_TIMEOUT` 600 s. Nothing else of the user's environment is changed: their login, `ANTHROPIC_API_KEY` and `CLAUDE_CONFIG_DIR` pass through. |
| working folder | an empty temp folder. |

`--bare` is not used: it ignores the OAuth login a subscription runs on. `--safe-mode` is not used: it turns MCP servers
off. Every event is checked as it arrives (`claude_exec.check_event`): a `tool_use` of anything but the two tools
(Claude Code's two MCP resource helpers aside) kills the process at once and the run is **invalid** (as for Codex); a
permission request (`control_request`) or a tool server that failed to start fails the case. The init event's tool list
is checked too: tools beyond the server's are a warning in the journal.

## A memory beside the agent (`--memory SPEC`)

`stbench run|eval claude:<model> --memory SPEC` is the Codex agent's memory arm ([codex.md](codex.md), "A memory beside
the reference agent"; `codex_memory` loads the one SPEC, so the file that runs under `codex:<model>` runs under
`claude:<model>` unchanged) on Claude Code: the same transport, prompt, images, model, effort and tools, with one memory
system beside the agent's own context. Its baseline is the reference row of the same model and effort. The SPEC's
`name`, `serve`, `sense`, `prompt` and `carry`, the sensor feed, the carried record, the store kept beside the case as
`<name>-store`, the `memory` failure, the marks in `results.json` and the reports, and `eval compare` are as there. What
Claude Code changes is where each thing goes:

| Thing | Where it is on Claude Code |
| --- | --- |
| the memory's MCP server (`serve`, `{store}` and `{model}` filled in; `{model}` is the full `claude-...` id) | a second entry of the `--mcp-config` file, started once with the process: the whole case, a resume included, talks to one server. Every tool of it is allowed by the server's name (`mcp__NAME`) under `dontAsk`. |
| the guide (`prompt`) | the end of `--system-prompt`, after a blank line. The system prompt survives Claude Code's compaction, as the base instructions do Codex's. |
| the sensor feed | one process for the case, written each `start`, `task` and frame, a frame as it is served to the model and never before (the memory holds exactly the frames the model has been served, no frame ahead: [codex.md](codex.md), "A memory beside the reference agent"), and each `record` of a memory that carries, as for Codex. |
| a memory call in the journal | a `memory_call` event, built from the `tool_use` of an `assistant` event and the `tool_result` of the `user` event that answers it (`claude_exec.memory_results`). The leak scan reads its result as a tool message, and `claude.summary.json` counts its tools by name. |

**What never changes.** Every message and item the model is sent is the reference agent's, byte for byte: the first user
message (the task), each later task, frame, ask, the resume line and the end item. The prompt says nothing of memory
beyond the guide the SPEC supplies, and the journal's first prompt stays the reference agent's (the guide is in the
`run` event). `tests/test_claude_memory.py` pins this against a run without a memory, resume included. A call of any
server but the relay and the memory's is a `ToolEscaped` and the run is invalid. The memory's server must offer no
tool that reads the disk outside its store (the held-out truth is on disk), as for Codex.

**Two limits of the command line.** The guide is part of the argv, and Windows passes a program at most 32,767
characters, so the system prompt (under 10,000) and the guide together must stay under 30,000 (`CMD_LIMIT`; a longer
pair is refused before the process starts). A SPEC's `serve` argv holds no quote or line break, as for Codex, so one SPEC
serves both agents.

## Compaction (`--compact-at TOKENS`)

Claude Code compacts its own conversation, and the reference agent leaves that at its default. Claude Code has no flag
that sets the limit, so `--compact-at` uses the two variables that were measured to work on `claude -p`
(`CLAUDE_CODE_AUTO_COMPACT_WINDOW=100000` and `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=P`: a window of 100,000 at 30 percent
compacted mid-turn at about 25,000 tokens on Claude Code 2.1.284, with `--disable-slash-commands`). The threshold is
(100,000 - 20,000) x P / 100, so TOKENS is rounded **up** to the next multiple of 800 (`--compact-at 50000` is P = 63,
50,400 tokens) and the largest limit is 66,400 (P = 83: a threshold past the window less 33,000 tokens is not used, and
that bound is kept, not tested here). A larger count is refused with exit 2, before anything is made. The process is
started with those two variables set and with the user's own `DISABLE_AUTO_COMPACT` taken out of its environment; the
argv and every message are unchanged, and the agent is told nothing of it. A reference run leaves the user's environment
as it is.

The lever is recorded as for Codex: `compact_at` (null without a limit) in the journal's `run` event and in
`claude.summary.json`, `"compact_at"` in `results.json`, `claude:<model>, compacts at 50k tokens` in the reports, and the
agent lever of `eval compare`. The recorded number is the requested one, not the rounded one. When Claude Code prints a
`compact_boundary` event the turn's warnings say `compaction (auto, N tokens before)`, so a run with a limit shows
whether it took; the summary counts it among the warnings.

Both arms of a comparison must share the limit. Claude Code compacts the whole conversation with its own summary, which
keeps the system prompt (the guide among it), the memory server's `instructions` (cut at 2,048 characters) and the user
messages, and replaces the tool results with a summary written for coding: the model holds less of
what it saw, which is the point of the experiment, and what the memory server holds is the only full record of the
frames.

## Files, tokens and cost

A case writes the Codex agent's files under its own name, so the reports, `eval compare`, the leak check and
`--select`/`--cases` work unchanged: `claude.calls.jsonl` (the journal: a `run` event, the events of the case, and one
record per turn with Claude Code's JSON events, any image as a hash) and `claude.summary.json` (written even when the
run fails). The summary has `provider: "claude"`, the CLI's version under `claude`, the model id, the effort, and
`llm_summary`'s counts. A line the CLI prints that is not JSON (it prints a notice for a model id it does not list yet)
is a warning of the turn, never a failure.

- **Tokens** come from the `result` event of each turn, which prints the session's running totals, so the journal takes
  each turn's own share (as for a Codex thread): prompt tokens are the fresh, cache-read and cache-write ones together,
  `cached` the cache reads, `reasoning` the thinking tokens inside the output ones. A turn the harness kills prints none:
  the count is then a lower bound and the summary says so. The harness's `summary.json` keeps the usage by model id.
- **Cost.** `claude.summary.json` has `cost_usd`, the sum of the CLI's own `total_cost_usd`, with a `cost_note`: it is
  the API-equivalent dollars, which a subscription does not pay. The harness's `summary.json` and `results.json` still
  price the tokens by `prices.json`, as for every system (`cost_usd` None with a note for a model the table lacks, such
  as a full `claude-...` id that is not one of its entries; `--max-usd` then reserves `--case-usd` for each case).

## How a failure is named

`claude_exec.classify` reads the message of the `result` event that says `is_error` (or the assistant message that
carries an API error, or the end of stderr for a process that died without a result) and uses Codex's words: `limit`
(a usage or rate limit, a 429, an overloaded API), `auth` (a login problem: `claude` again, or `claude setup-token`),
`context` (the window is full despite Claude Code's own compaction), `timeout`, `escaped`, `stalled`, else `error`.
They are counted in `exec_failures`, and the last failed turn's message is kept in `exec_failure_message`.

## Unverified without a live call

This page was first written from the CLI's `--help` and its binary. The 0.11.0 reference rows (42 short and medium
cases each for `haiku`, `sonnet` and `opus`, every case complete) settle part of it: `--allowedTools` with
`--permission-mode dontAsk` lets the two MCP tools run with no prompt, and a case of more than 100 frames runs to its end
(22 of the 42 hold more, up to 295) with no `context` failure. Still not checked directly: that an MCP image block in a
tool result reaches the model as an image (the rows' `now` scores, far above the floor on questions only the latest
frame answers, say it does; a memory's image is not tested); whether Claude Code drops the oldest pictures of a long case
(the API caps the images of one request, documented as 100 for a 200k-token window) and how many it keeps; that
`--setting-sources ""` loads nothing; what else the request carries beside the system prompt (a current-date line,
say); and whether the `result` event's `usage` and `total_cost_usd` are the session's running totals (they are read as
such; `thread_usage` in each journal record is the figure as printed).

For the memory and the limit, also unchecked against the real CLI: that `--allowedTools mcp__NAME` lets every tool of
the memory's server run under `dontAsk` (a tool denied would fail the case as `error`, "denied by the permission mode");
that the memory server's `initialize` `instructions` reach the model when `--system-prompt` replaces Claude Code's own
prompt, which Codex does with its base instructions and which is not known for Claude Code; that the init event lists a
slow memory server as connected before the first turn (one that reports `failed` or `needs-auth` fails the case); the
shape of the `compact_boundary` event (read as `compact_metadata.trigger` and `.pre_tokens`; a different shape prints
`None` in the warning and nothing else is lost); and that the two variables still take effect on a Claude Code newer
than 2.1.284. The tests run all of it against `tests/claude_cli_fake.py`, which holds the model to `--allowedTools` as
`dontAsk` does.
