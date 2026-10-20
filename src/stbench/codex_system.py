"""The reference agent on the stream protocol, over the Codex CLI: one thread a case, one long exec, frames pulled.

`CodexSystem` is an in-process `System`. What the model is sent and the tools it has are the reference agent's: the
prompt and the messages are `llm_render`'s, the tools are `next_frame` and `submit_answer` (`codex_mcp` relays them to
`codex_feed`), and Codex is held to those tools and nothing else (`codex_exec`). The agent remembers with its own
context: it has no memory tool, no pointing tool and no instruction about what to keep, and the state directory stays
empty (but for a run with a memory, below). It runs on `rgb_pose`, the default and the headline, reading each frame's
camera image alone and judging distance from the picture (the robot has a camera and no depth sensor); and on
`detections`, the arm that measures what the pixels cost, reading each frame's detections as text lines and no image.
With `text_only` it reads no image on `rgb_pose` either, the header lines alone: the zero-shot check of what the
questions give away. It is an option and not a channel, and `cmd_codex` stamps the run with it.

The harness pushes and the model pulls. The case is one `codex exec` (`codex_live.Live`) on one thread; the runner's
messages become the replies to the model's `next_frame` calls, in stream order, and each of them returns once the model
has pulled again (it is done with the item) or has been found not to be going to:

- `task`: the first task starts the exec, and its message is the system prompt and the task message; every later task
  is the reply to a pull;
- `observe`: one frame (the stream sends one step an `observe`; a longer list is a ValueError): its header line and, on
  `rgb_pose`, its camera JPEG as an image block (read when the frame arrives, since the inbox holds it only now); on
  `detections` its detection lines; with `text_only` the header alone. The frame is held, not served, and `observe`
  returns at once: what the harness sends next shows what goes with it (a held frame);
- `ask`: the questions, labelled Q1..Qn, never by qid. The model answers each with `submit_answer(label, answer)`; a
  `next_frame` with labels unsubmitted gets one error naming them, and the second fails them `no_submit` (a second bad
  submission for a label fails it `malformed`). The answers map back to qids;
- `finish`: the end item, "The job is over", then up to `finish_wait_s` for the turn to complete, then a kill. The
  tokens of the case under the model's id (input, cached, cache writes and output) are each exec's own share of its
  thread's running total; a turn that was killed prints none, so then the count is a lower bound (the summary says so).

A memory (`codex_memory`, `stbench run|eval --memory SPEC`) is one thing added beside the agent's own context, and
nothing the model is sent changes with it: every message and item (the first prompt, each task, frame, ask, resume line
and the end item) is the reference agent's, byte for byte. What differs is the base instructions of the exec (the
memory's guide follows the one line), a second MCP server in its argv (the memory's own, for the store in the state
directory; the model calls it and `codex_live` journals the calls), and the memory's sensor feed, one process for the
case that `start` opens before anything is sent to the model. Each `start`, `task` and frame is written to it, one JSON
line, and its one-line reply awaited before the system goes on. A frame is written as it is served to the model
(`_feed_held`, the one place a held frame leaves: alone, or with its questions), never when the runner reads ahead
to the next message: the store holds exactly the frames the model has been served, so frame k is in it when the model is
shown frame k and frame k+1 is not. It is never sent an ask, a question or the end. A feed that fails ends the case as
`TransportError("memory")` (journaled as `memory_failed`). `close` ends the
feed and copies the store to `keep`.

A memory that carries (`Memory.carry`, SPEC key `carry`) changes two more things, both only for its own arm: the tool
server offers `next_frame` an optional `memory` argument (`agent_tools.tools(True)`), and the model's record rides on
its pull. `_carry` writes `{"record": memory}` to the sensor feed from the feed's thread, before the pull is counted
(`codex_feed`), and the feed's reply is the last text block of the item that pull gets; the runner learns of the pull
only then, so it never uses the one sense pipe at the same time. A feed that fails on a record is journaled
(`memory_failed`, message `record`) and the case ends as `TransportError("memory")` at the runner's next check
(`_settle`), which raises before any item is served: the pull that carried the record is closed with the rest of the
feed, so the model is told nothing of the failure.

A held frame (FD-questions-0.10 §1.5). A separate ask item costs the model one pull more for every question, and the
controller's real moment is looking and deciding together, so an ask that follows a frame joins it: the model gets one
item, the frame (header, image) and then the ask's text, and answers before it pulls again. The rule, by the harness's
next message: an `ask` at the held frame's own step joins it (one item; the frame is served together with the ask and is
timed by the ask's timeout, `ask_during` or `ask_end`, from the moment it is served); an `observe`, a `task` or `finish`
serves the held frame alone first, as it was served before, then handles the message. So a frame is never held past the
next message, and the model never has frame k+1 before it has pulled after frame k and its questions. Each `ask` after a
held frame joins that frame if the frame is still held, else it is its own item: the stream sends a point's asks one
after another (each returns when the model has answered), so the first of two asks at one step joins the frame and the
second is served alone. An ask at another step than the held frame's serves the frame alone first. A frame's deadline
(`observe`) starts when it is served and not when it was held, and a held frame is dropped when the system closes. No
message asks between a `task` and the first frame, so nothing joins a task. Because `observe` returns at once, the
harness's `observe` wall time of a frame is near zero and the time it took the model is in the message that served it.

The wait after an item ends in one of three ways. The model pulls: on. The exec ends: when it ended cleanly (the model
ended its turn early) the thread is resumed with one fixed line, `llm_render.RESUME_MESSAGE`, and after `max_resumes`
resumes in a row with no pull between the next early end is `TransportError("stalled")`; a failed exec is
`TransportError(reason)` (one transient failure before the model did anything is tried once more). The message's
deadline (`stream_protocol.TIMEOUTS`) passes: the exec is killed and the case ends as `TransportError("timeout")`. A
`CodexError` (a limit, a timeout, an auth failure, a failed exec, an escaped tool) or `ContextTooLong` is noted in the
journal (`exec_failed`) and raised as `stream_protocol.TransportError(reason)`, chained to it, which ends the case
(`stream.run_case`): a Codex failure is never retried by the harness, and the questions not yet answered fail. Nothing
here reads a step's `identity` or `detections`, a question's `template` or `slots`, or anything of the dataset
(`llm_render` reads a detection's named fields, never its key `d`); no branch names a model.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from . import llm_render
from .agent_tools import ContextTooLong
from .codex_exec import Codex, CodexError, CodexTimeout, Result, failure_reason, transient
from .codex_feed import Feed, Item
from .codex_live import Live
from .codex_log import Journal
from .codex_memory import Memory, MemoryFailure, Sense, feed_step
from .safecopy import remove_tree
from .stream_protocol import TIMEOUTS, System, TransportError, failed, timeout_key

TOKENS = {"tokens_in": "prompt_tokens", "tokens_cached_in": "cached_tokens",
          "tokens_cache_write_in": "cache_write_tokens", "tokens_out": "completion_tokens"}


@dataclass(frozen=True)
class Settings:
    max_calls_item: int = 100  # tool calls the feed serves between two items
    max_resumes: int = 3  # resumes in a row, with no pull between, before the next early end is "stalled"
    finish_wait_s: float = 60.0  # seconds the last turn has to complete after the end item, then a kill


DEFAULTS = Settings()


class CodexSystem(System):
    """`thread` is the id of the Codex thread of the case (None before its first exec); `execs` counts the execs
    started. `asks_during` says the job asks during the job, so the prompt may speak of decision points: the harness
    derives it from the job's `decisions` and `looks`, never from the run's `during` arm (the prompt is the same bytes
    on both arms of the leak check); `asks_end` says it asks after the job (its `ends`). `match_m`, `clearance_m` and
    `reacquire_m` are the job's matching radius, clearance and re-find radius (`JobSpec.match_m`, `clearance_m`,
    `reacquire_m`) for the prompt's credit section; None for all three leaves the section out. `lists` and
    `headings` say the job asks a list (`task_things`) or a heading (`open_heading`), so the section carries those
    bullets (`brief.asks_lists`, `asks_headings`), and `hazards` the list families it asks during the job, whose
    omissions the section says score -3 (`brief.list_hazards`). `kinds` are the answer types the job asks
    (`brief.answer_types`, from the job and never from the run's `during` arm): the prompt shows the shape of those and
    of `unknown` only, and an ask holding another type is refused before it is sent (None: every type is shown, for a
    caller that knows no job). `families` are the families the job asks (`brief.asked_families`), which with `kinds`
    decide the credit section's bullets about one kind of answer or one family (None: all of them). `text_only`
    withholds every image (see the module docstring); `start` raises ValueError on the `detections` channel, which sends
    none. `memory` is a memory the agent keeps beside its context (`codex_memory`; None, the reference agent, keeps
    none), and `keep` the folder its store is copied to when the case closes (without its `cache/`, which the store
    rebuilds), so what the agent wrote can be read after the case's state folder is gone."""

    channels = ("rgb_pose", "detections")

    def __init__(self, state_dir, workdir, *, codex: Codex, journal: Journal, settings: Settings = DEFAULTS,
                 timeouts: dict[str, float] = TIMEOUTS, asks_during: bool = False, asks_end: bool = True,
                 match_m: float | None = None, clearance_m: float | None = None, reacquire_m: float | None = None,
                 lists: bool = False, headings: bool = False, hazards: tuple[str, ...] = (),
                 kinds: tuple[str, ...] | None = None, families: tuple[str, ...] | None = None,
                 turns: bool = True, text_only: bool = False, memory: Memory | None = None, keep: Path | None = None,
                 **opts):
        super().__init__(state_dir, workdir, **opts)
        self.codex, self.journal, self.settings, self.timeouts = codex, journal, settings, timeouts
        self.asks_during, self.asks_end, self.match_m, self.text_only = asks_during, asks_end, match_m, text_only
        self.clearance_m, self.reacquire_m, self.lists, self.headings = clearance_m, reacquire_m, lists, headings
        self.hazards, self.kinds, self.families, self.turns = hazards, kinds, families, turns
        self.memory, self.keep = memory, keep
        self.sense: Sense | None = None  # the memory's sensor feed, once `start` opened it
        self._lost: MemoryFailure | None = None  # the sensor feed failed on a carried record, which the runner ends on
        self._camera: dict | None = None  # public/camera.json, which the feed is given with the frames
        self.thread: str | None = None
        self.execs = 0
        self.tokens = dict.fromkeys(TOKENS, 0)
        self.feed: Feed | None = None
        self.live: Live | None = None
        self.stalls = 0  # resumes in a row with no pull between
        self._reaped = True  # `live` has been read and journaled
        self._sent: tuple[str, str] = ("start", "")  # the kind and prompt of the latest launch
        self._msg: tuple[str, int] = ("task", 0)  # the purpose of the message being served, and its questions
        self._held: Item | None = None  # the latest frame, not yet served: the next message says what goes with it
        self._held_step: dict | None = None  # its step for the memory's feed, written only when the frame is served
        self._sense_dir: Path | None = None  # where the held frame's image is put for the feed (`_feed_held`)

    # -- the stream ---------------------------------------------------------------------------------------------
    def start(self, session, robot, channel) -> None:
        if self.text_only and channel != "rgb_pose":
            raise ValueError(f"text_only withholds the camera images of rgb_pose, not those of {channel!r}")
        if self.memory is not None and channel != "rgb_pose":
            raise ValueError(f"a memory is fed the camera images of rgb_pose, not {channel!r}")
        super().start(session, robot, channel)
        carries = self.memory is not None and self.memory.carry
        self.feed = Feed(self.journal.note, max_calls=self.settings.max_calls_item,
                         **({"carry": self._carry, "carry_name": self.memory.name} if carries else {}))
        if self.memory is not None:  # before anything is sent to the model
            self._camera = json.loads((self.public / "camera.json").read_text(encoding="utf-8"))
            try:
                self.sense = self.memory.open_sense(self.state_dir, self.codex.model)
            except MemoryFailure as error:
                self._unfed("start", error)
            self._sense("start", {"camera": self._camera, "robot": robot})

    def task(self, index: int, clock: str, gap_s: float, brief: str) -> None:
        self._flush()   # the frame before the task is served first (and fed), so the memory hears of the task after it
        self._sense("task", {"index": index, "clock": clock, "gap_s": gap_s, "brief": brief})
        text = llm_render.task_message(index, clock, gap_s, brief)
        item = Item("task", text)
        if self.live is None:  # the case's first message: it starts the thread, and the item is in the prompt
            self._msg = ("task", 0)
            ticket = self.feed.serve_in_prompt(item)
            self._launch("start", self._first(text))
            self._settle(ticket, timeout_key("task"))
        else:
            self._deliver(item, "task", timeout_key("task"))

    def observe(self, steps: list[dict]) -> None:
        if len(steps) != 1:
            raise ValueError(f"the reference agent is sent one frame at a time, not {len(steps)}")
        step = steps[0]
        text, image = llm_render.frame_message(step, self.channel, text_only=self.text_only)
        data = (self.workdir / image).read_bytes() if image else None
        self._flush()  # the frame before is not asked about: it goes alone (and is fed to the memory as it goes)
        self._held = Item("frame", text, data, k=step["k"])
        self._held_step = feed_step(step, self.workdir) if self.sense is not None else None   # fed when served

    def ask(self, when: str, now: dict, questions: list[dict]) -> dict[str, dict]:
        labelled = llm_render.labelled(questions)
        text = llm_render.ask_message(when, now, labelled,   # only `text` and `answer_type` are read; a bad type stops
                                      shown=None if self.kinds is None else llm_render.listed(self.kinds))
        qids = {label: q["qid"] for label, q in labelled.items()}
        item = Item("ask", text, k=now["k"], ask={"when": when, "now_k": now["k"], "labels": qids},
                    types={label: q["answer_type"] for label, q in labelled.items()})
        if self._held is not None and self._held.k == now["k"]:  # the questions are about the frame held: one item
            held, self._held = self._held, None
            self._feed_held(held)
            item = held.with_ask(item)
        try:
            self._deliver(item, "ask", timeout_key("ask", when), len(qids))
        finally:
            asked = self.feed.close_ask()
        table, nudged = (asked.table, asked.nudged) if asked is not None else ({}, [])
        return {qids[label]: self._answer(qids[label], label, table.get(label, {}), label in nudged)
                for label in labelled}

    def finish(self) -> dict:
        self._end()
        return self._usage()

    def close(self) -> None:
        try:
            if self.live is not None:
                self.live.kill()
                if not self._reaped:
                    self._record("close")
                self.live.close()
            if self.feed is not None:
                self.feed.close()
        finally:
            if self.sense is not None:
                self.sense.close()
            if self._sense_dir is not None:
                remove_tree(self._sense_dir)
            self._keep_store()

    # -- parts --------------------------------------------------------------------------------------------------
    def _usage(self) -> dict:
        return {"models": {self.codex.model: dict(self.tokens)}, "peak_context": None, "remote_store_bytes": None}

    def _sense(self, kind: str, body: dict) -> None:
        """Send one message to the memory's sensor feed and wait for its reply (nothing without a feed). A failure is
        journaled and ends the case as `TransportError("memory")`."""
        if self.sense is None:
            return
        try:
            self.sense.send({kind: body})
        except MemoryFailure as error:
            self._unfed(kind, error)

    def _feed_held(self, held: Item) -> None:
        """Write the frame `held` to the memory's feed, as it is about to be served to the model and not before: the
        memory holds exactly the frames the model has been served, never the next one the runner has read ahead to (the
        message after a frame says whether questions join it). The inbox no longer holds the image, so the frame's
        bytes are put in a file of the system's own for the length of the message."""
        step, self._held_step = self._held_step, None
        if step is None or self.sense is None:
            return
        if self._sense_dir is None:
            self._sense_dir = Path(tempfile.mkdtemp(prefix="stbench-sense-"))
        image = self._sense_dir / Path(step["rgb"]).name
        image.write_bytes(held.image)
        try:
            self._sense("observe", {"camera": self._camera, "steps": [{**step, "rgb": str(image)}]})
        finally:
            image.unlink(missing_ok=True)

    def _carry(self, memory: dict) -> dict:
        """The model's record, written to the sensor feed from the feed's thread (`codex_feed`), and the feed's reply. A
        failure is journaled and kept for the runner (`_settle`), which ends the case before an item is served, so the
        error reply is never read by the model."""
        try:
            return self.sense.send({"record": memory})
        except MemoryFailure as error:
            self._lost = error
            self._journal_unfed("record", error)
            return {"ok": False, "error": "memory_failed", "message": "the memory could not be reached"}

    def _journal_unfed(self, kind: str, error: MemoryFailure) -> None:
        self.journal.note({"event": "memory_failed", "message": kind, "error": str(error)[:500]})

    def _unfed(self, kind: str, error: MemoryFailure) -> None:
        """The memory's sensor feed failed on the message `kind`: journal it and end the case."""
        self._journal_unfed(kind, error)
        raise TransportError("memory") from error

    def _keep_store(self) -> None:
        """Copy the memory's store to `keep`, without its `cache/` (replacing what an earlier run of the case left
        there). A copy that fails is journaled: the answers stand."""
        store = None if self.memory is None or self.keep is None else self.memory.store(self.state_dir)
        if store is None or not store.is_dir():
            return
        try:
            remove_tree(self.keep)
            shutil.copytree(store, self.keep, ignore=lambda folder, names: {"cache"} & set(names)
                            if Path(folder) == store else set())
        except OSError as error:
            self.journal.note({"event": "store_not_kept", "error": str(error)[:500]})

    def _answer(self, qid: str, label: str, entry: dict, nudged: bool) -> dict:
        """The answer of one label from the ask's table, noted in the journal: submitted (`repaired` when a bad try or
        a nudge came first), or the failed marker with its reason."""
        if entry.get("status") == "submitted":
            outcome = "repaired" if entry.get("bad") or nudged else "submitted"
            self.journal.note({"event": "answer_outcome", "qid": qid, "label": label, "outcome": outcome})
            return entry["answer"]
        reason = entry.get("reason") or "no_submit"
        self.journal.note({"event": "answer_outcome", "qid": qid, "label": label, "outcome": "failed",
                           "reason": reason})
        return failed(reason)

    def _system_prompt(self) -> str:
        """The reference agent's system prompt for this case (`llm_render.system_prompt`)."""
        return llm_render.system_prompt(self.public, self.channel, asks_during=self.asks_during,
                                        asks_end=self.asks_end, match_m=self.match_m, clearance_m=self.clearance_m,
                                        reacquire_m=self.reacquire_m, lists=self.lists, headings=self.headings,
                                        hazards=self.hazards, kinds=self.kinds, families=self.families,
                                        turns=self.turns, text_only=self.text_only)

    def _first(self, text: str) -> str:
        """The first message of the thread: the system prompt, then `text`."""
        return "\n\n".join([self._system_prompt(), text])

    def _deliver(self, item: Item, purpose: str, key: str, questions: int = 0) -> None:
        """Hand the model an item as the reply to its pull and wait until it pulls again (`_settle`). A frame still held
        is served first, alone."""
        self._flush()
        self._msg = (purpose, questions)
        self._settle(self.feed.supply(item), key)

    def _flush(self) -> None:
        """Serve the frame that is held, alone, and wait for the pull after it, within the `observe` timeout counted
        from now: the message that came next asks nothing about it. Nothing when no frame is held."""
        held, self._held = self._held, None
        if held is not None:
            self._feed_held(held)
            self._msg = ("observe", 0)
            self._settle(self.feed.supply(held), timeout_key("observe"))

    def _settle(self, ticket: int, key: str) -> None:
        """Wait for the pull after the item `ticket`, within the message's timeout: a resume for a turn that ended
        early, a retry for a transient failure, and `TransportError` for the rest (see the module docstring)."""
        deadline = time.monotonic() + self.timeouts[key]
        while True:
            outcome = self.feed.wait(ticket, deadline, self.live.alive)
            if self._lost is not None:  # a carried record failed (journaled when it did)
                raise TransportError("memory") from self._lost
            if outcome == "pulled":
                self.stalls = 0
                return
            if outcome == "timeout":
                self.live.kill()
                self._record(self._msg[0])
                self._fail(CodexTimeout(f"no pull within {self.timeouts[key]:g} s of the {self._msg[0]}: codex exec "
                                        "was killed"))
            if self._reap() is None:  # relaunched for a transient failure
                continue
            if self.stalls >= self.settings.max_resumes:
                self._fail(CodexError(f"the model ended its turn {self.stalls + 1} times in a row without calling "
                                      "next_frame"), "stalled")
            self.stalls += 1
            self._launch("resume", llm_render.RESUME_MESSAGE)

    def _launch(self, kind: str, prompt: str, attempt: int = 1) -> None:
        """Start an exec on the case's thread (`start` when there is none, else `resume`) and journal its prompt."""
        self.feed.begin_exec()
        self._sent = (kind, prompt)
        self.journal.note({"event": "exec_started", "kind": kind, "base": self.thread, "prompt": prompt,
                           "attempt": attempt})
        try:
            memory = None if self.memory is None else self.memory.server(self.state_dir, self.codex.model)
            self.live = Live(self.codex, kind=kind, thread=self.thread, prompt=prompt, server=self.feed.server_config(),
                             wake=self.feed.wake, attempt=attempt, memory=memory, note=self.journal.note)
        except CodexError as error:
            self._fail(error)
        self._reaped = False
        self.execs += attempt == 1

    def _reap(self) -> Result | None:
        """Read the exec that ended and journal it. None when it failed in a way worth one retry and was started again;
        raises `TransportError` when it failed."""
        live = self.live
        try:
            result = live.result()
        except (CodexError, ContextTooLong) as error:
            retry = live.attempt == 1 and transient(error, live.events)
            live.close()
            if retry:
                self.codex.backoff()
                self._launch(*self._sent, attempt=2)
                return None
            self._fail(error)
        live.close()
        self._account(result)
        return result

    def _record(self, purpose: str) -> None:
        """Read and journal the exec as it ended when the harness ended it (killed, or `close`); a failure is journaled
        and not raised."""
        try:
            self._account(self.live.result())
        except (CodexError, ContextTooLong) as error:
            self._noted(error, purpose)

    def _account(self, result: Result) -> None:
        """Journal an ended exec with its own share of the thread's tokens and carry its thread on."""
        used = self.journal.exec(session=self.session, kind=self.live.kind, base=self.thread, result=result)
        for name, key in TOKENS.items():
            self.tokens[name] += used[key]
        self.thread = result.thread or self.thread
        self._reaped = True

    def _noted(self, error: BaseException, purpose: str, reason: str | None = None) -> None:
        """Journal a failed exec: the word that counts it, its message and, for an ask, how many questions it took."""
        questions = self._msg[1] if purpose == "ask" else 0
        self.journal.note({"event": "exec_failed", "purpose": purpose, "kind": self._sent[0],
                           "reason": reason or failure_reason(error), "message": str(error)[:500],
                           **({"questions": questions} if questions else {})})
        self._reaped = True

    def _fail(self, error: BaseException, reason: str | None = None) -> None:
        """End the case: journal the failure and raise `TransportError(reason)`, chained to `error`."""
        self._noted(error, self._msg[0], reason)
        raise TransportError(reason or failure_reason(error)) from error

    def _end(self) -> None:
        """The job is over: tell the model, give the last turn `finish_wait_s` to complete, kill it otherwise, and read
        the exec. The frame still held is served first, alone (a failure of that is raised, as any message's); a
        failure after it is journaled, not raised: the answers stand."""
        if self.live is None or self._reaped:
            self._held, self._held_step = None, None  # no exec to serve it to: the memory is not fed it either
            return
        self._flush()
        live = self.live  # after the flush: serving the frame may have resumed the thread in a new exec
        self._msg = ("finish", 0)
        if live.alive():
            ticket = self.feed.supply(Item("end", llm_render.END_MESSAGE))
            deadline = time.monotonic() + self.settings.finish_wait_s
            if self.feed.wait(ticket, deadline, live.alive, pull=False) == "pulled":  # it took the end item
                live.wait(max(0.0, deadline - time.monotonic()))
            live.kill()
        self._record("finish")
        live.close()

    def _more(self, text: str, timeout: float) -> list[str]:
        """One more message on the thread, after the job is over: `text` as a resume, run to its end within `timeout`
        seconds. The model's replies. For a probe of the thread (`tools/codex_canary.py`); `TransportError` as for any
        exec."""
        self._msg = ("finish", 0)
        self._launch("resume", text)
        while True:
            if not self.live.wait(timeout):
                self.live.kill()
                self._record("finish")
                self._fail(CodexTimeout(f"the exec was killed after {timeout:g} s"))
            if (result := self._reap()) is not None:
                return result.messages
