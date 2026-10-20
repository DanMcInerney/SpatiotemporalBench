"""The reference agent on the stream protocol, over the Claude Code CLI: one process a case, frames pulled.

`ClaudeSystem` is `CodexSystem` with `claude -p` for its transport, so everything the model is sent and every rule of
the pull is the Codex agent's (`codex_system`; docs/codex.md): the prompt and the messages are `llm_render`'s, the
tools are `next_frame` and `submit_answer` (`codex_mcp` relays them to `codex_feed`), a frame that is asked about
joins its questions in one item, the answers map back to qids, and the tokens of the case are counted under the
model's id. What differs is the unit of work. A Codex exec is one turn of a thread, started again to resume; a Claude
Code process holds the whole conversation (stream-json on stdin, nothing saved to disk), so:

- the first task starts the process (`claude_live.ClaudeLive`): the system prompt goes in `--system-prompt`, in place of
  Claude Code's own, and the first user message is the task message. The journal's first prompt is still the system
  prompt and the task together (`CodexSystem._first`), which is what the model was shown, so the leak scan reads the
  same bytes for both agents;
- when the model ends its turn without pulling (a `result` event with the process still up), the resume line
  (`llm_render.RESUME_MESSAGE`) is written to the same stdin as the next user message, at most `max_resumes` times in a
  row, and then the case ends as `stalled`. A turn is what the journal calls an exec: `kind` is `start` for the first
  and `resume` for each later one;
- a failed turn ends the case as the transport failure it is (`limit`, `auth`, `context`, `timeout`, `escaped`,
  `error`), and nothing is retried here: Claude Code retries a transient API error itself, and a process that died
  has lost the conversation, which was never saved. A usage or rate limit of the subscription arrives as the
  `result` of the turn.

A memory (`codex_memory`, `--memory SPEC`) is added as it is to the Codex agent and changes no message the model is
sent: the process has the memory's MCP server beside the tool server (every tool of it allowed) for the whole case, the
memory's guide follows the system prompt in `--system-prompt` (the journal's first prompt stays the reference agent's:
the guide is in the `run` event), and the sensor feed, the carried record, the store kept beside the case and the
`memory` failure are `CodexSystem`'s, unchanged. The memory's calls are journaled as `memory_call` events
(`claude_live`).

The tokens and the cost of a turn are its own share of the session's running totals the `result` event prints
(`codex_log.Journal`); a killed turn prints none, so then the count is a lower bound (the summary says so).
"""

from __future__ import annotations

from .agent_tools import ContextTooLong
from .claude_live import ClaudeLive
from .codex_exec import CodexError, Result
from .codex_system import CodexSystem


class ClaudeSystem(CodexSystem):
    """A `CodexSystem` whose `codex` is a `claude_exec.Claude` and whose execs are the turns of one process."""

    def _first(self, text: str) -> str:
        self._opening = text   # the first user message; the system prompt goes to --system-prompt
        return super()._first(text)

    def _launch(self, kind: str, prompt: str, attempt: int = 1) -> None:
        """Begin a turn: the first one starts the process, a later one writes `prompt` (the resume line) to it. Journals
        the prompt."""
        self.feed.begin_exec()
        self._sent = (kind, prompt)
        self.journal.note({"event": "exec_started", "kind": kind, "base": self.thread, "prompt": prompt,
                           "attempt": attempt})
        try:
            if kind == "start":
                memory = None if self.memory is None else self.memory.server(self.state_dir, self.codex.model)
                self.live = ClaudeLive(self.codex, system=self._system_prompt(), prompt=self._opening,
                                       server=self.feed.server_config(), wake=self.feed.wake, memory=memory,
                                       note=self.journal.note)
            else:
                self.live.send(prompt)
        except CodexError as error:
            self._fail(error)
        self._reaped = False
        self.execs += 1

    def _reap(self) -> Result | None:
        """Read the turn that ended and journal it; the process stays up for the next one. Raises `TransportError` when
        the turn failed."""
        live = self.live
        try:
            result = live.result()
        except (CodexError, ContextTooLong) as error:
            live.close()
            self._fail(error)
        self._account(result)
        return result
