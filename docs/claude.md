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
frame answers, say it does); whether Claude Code drops the oldest pictures of a long case
(the API caps the images of one request, documented as 100 for a 200k-token window) and how many it keeps; that
`--setting-sources ""` loads nothing; what else the request carries beside the system prompt (a current-date line,
say); and whether the `result` event's `usage` and `total_cost_usd` are the session's running totals (they are read as
such; `thread_usage` in each journal record is the figure as printed).
