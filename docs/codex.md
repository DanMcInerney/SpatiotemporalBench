# The Codex reference agent (`codex:<model id>`)

The benchmark's own agent: a model reached through the Codex CLI and a ChatGPT login, on the plan's quota. It needs no
API key and spends no dollars, and it is the kind of agent that held-out numbers may come from, because its tools cannot
read the disk. It is one agent, fixed in the code: the argv of the exec (`codex_exec`), the tools
(`agent_tools`) and their replies, the prompt and the items it is sent (`llm_render`). It reads the camera image
of each frame on `rgb_pose`, the default and the headline channel. The robot has a camera and no depth sensor (0.6.0):
no channel sends it a depth image, and it judges distance from the picture. On `detections` it reads each frame's detections as text, the diagnostic arm
that measures what the pixels cost ([the detections arm](#the-detections-arm)). It is a system like any other
(`docs/agent.md`): it gets the same messages (one frame in each `observe`), in process, and answers through the same
interface, so a run of it is scored by the same code as a run of your agent. It needs `codex` on PATH (tested with `codex-cli 0.159.3`) and a working
`codex login`. The same agent on the Claude Code CLI, for `haiku`, `sonnet` and `opus`, is [claude.md](claude.md): the
options, the files and the rows are these, and switching is the prefix of the system name. `--memory SPEC` and
`--compact-at TOKENS` are the Claude agent's too (claude.md, "A memory beside the agent" and "Compaction"): one SPEC runs
under both.

**It remembers with its own context only** (a memory beside it is an option that measures a store,
[below](#a-memory-beside-the-reference-agent---memory-spec)). The benchmark measures an agent's own memory, so the
reference agent is given no memory tool, no instruction on what or when to remember, no per-frame cue, no statement that
its conversation may be shortened and no pointing tool (`locate`): its tools are `next_frame`, the sensor feed, and
`submit_answer`, and Codex manages and compacts its context itself. Its native file and shell tools stay disabled ([held
to the agent's tools](#held-to-the-agents-tools)) because they could read the private truth on disk, so it has no file
to write to either. It is handed the robot's sensors, time-ordered and timestamped (the benchmark offers the agent no
tool to help it in its task): each frame's camera image with the robot's pose, clock and status, and the subgoal where
the job states one, one frame at a time, as it asks for the next with `next_frame`, and `submit_answer` to give its
answers. `next_frame` is the robot's sensor feed, as `submit_answer` is the answer channel; it helps the agent with
nothing. The system prompt tells it the camera as the robot knows it (the image size, the horizontal field of view and
where the camera is mounted on the base, and that the camera is its only eye on the world), the frame of an offset
answer, and how an answer is credited (against the job's clearance and re-find radius). **The 0.4.0 and 0.4.1 reference
rows, and 0.5.0's pilot, were made with the earlier scaffold** (five memory tools, a prompt that told the agent to
update its memory after every batch and what to keep, an "Update your memory now." cue on every message, `locate`, and
no depth image): they measure the model with our scaffold, not its own memory.

## Running it

```sh
uv run stbench run codex:gpt-6-luna DATASET/private/<case id>                                  # the default effort, low
uv run stbench run codex:gpt-6.1-sol DATASET/private/<case id> --reasoning medium --out runs/sol-<case id>
```

`stbench run` resolves `<agent>:<model>` through its registry (`cmd_run.AGENTS`) to `cmd_codex.run`, the run driver. The
model id comes from the command line and goes nowhere else: nothing in the agent branches on a model or a provider.

| Option | Meaning |
| --- | --- |
| `--channel rgb_pose\|detections` | The channel the agent reads (`rgb_pose` is the default): on `rgb_pose` each frame's camera image alone; on `detections` each frame's detections as text and no image ([the detections arm](#the-detections-arm)). A name that is none of these is refused (exit 2). |
| `--reasoning EFFORT` | `none`, `low` (the default), `medium`, `high` or `xhigh`: Codex's `model_reasoning_effort`, recorded in the summary. An effort the model's entry in Codex's catalog does not list is refused before any exec (exit 2). |
| `--brief`, `--during` | As for every system (`docs/agent.md`, "The brief"): the brief variant, and whether the during-job questions are asked. |
| `--timeout SECONDS` | Seconds the model may take, from being served an item to its next `next_frame` call (default: each message's own, `stream_protocol.TIMEOUTS`: 600 s for a task, a frame and a during ask, 3,600 s for the end ask). Past it the exec is killed with what it started and the case ends as `timeout`. |
| `--memory SPEC` | A memory store beside the agent's own context ([a memory](#a-memory-beside-the-reference-agent---memory-spec)). Refused on `detections`, with `--text-only` and for agents other than `codex:` and `claude:`. |
| `--compact-at TOKENS` | Force Codex to compact the thread once its context passes TOKENS (a whole number, at least 1), earlier than Codex's own limit: `-c model_auto_compact_token_limit=TOKENS` on every exec, the resume included ([compaction](#how-a-run-goes)). For experiments on memory beyond the context; refused for agents other than `codex:` and `claude:` (Claude Code takes the same option through its own variables, claude.md). |
| `--out DIR` | Where the files below go (default `DATASET/runs/codex_<model>-<case id>`; never under the dataset's `scripts/` or `private/`). |

A folder that already holds a `codex.calls.jsonl` refuses the run (exit 2): a Codex call is not recorded for reuse, so a
run cannot be replayed, and the journal of one run must not be mixed with another's. Remove the folder, or choose another
`--out`. A run in which no exec finished (no login, say) leaves no journal behind, so it can be run again in place; an
invalid run (below) keeps its journal whatever finished.

**Concurrency and quota.** Every Codex run on the machine shares one login, and the login is rate-limited. A case holds
one `codex exec` at a time, so `stbench eval --workers N` holds N; keep the sum over every row running at once at about
20 or fewer (18 is the working cap). On a Pro login 19-28 execs at once ran clean and 36-44 got
`429 Too Many Requests` ("CodexLimit: exceeded retry limit"), which stopped their cases (2026-10-04); a stopped case
reruns from its start with the same `eval` command (`docs/eval.md`, "Resume"). That ceiling was measured with one exec
for each message of a case. A pulled case makes a model request for every frame inside its one exec, which is
estimated at about three times the requests of a case on that pattern, and the ceiling has not been measured for the
pull: start below the cap and watch for 429s. The login's weekly use is in its session
logs, `~/.codex/sessions/YYYY/MM/DD/*.jsonl`, as `rate_limits.primary.used_percent` with its `resets_at`: read it
before each set of runs and stop by 99%, since past 100% the account spends credits and blocks its owner's own Codex use
until the reset. The quota a case costs is its tokens: about 690 M tokens a point for `gpt-6-luna` and 62 M for
`gpt-6.1-sol` (Release 0.4.1). Each frame's request reads the whole context again, almost all of it from the cache, so
a case's prompt tokens rise with the pull: expect about three to four times
sol's tokens a case of the one-exec-for-each-message pattern. Never read or copy `~/.codex/auth.json`.

The run writes, beside the `answers.jsonl`, `rows.jsonl` and `summary.json` that `stbench run` writes for every system:
- `codex.calls.jsonl`, the journal: a `run` event first (the model, the effort, `compact_at` (null unless `--compact-at`
  was given), the case, the channel, `text_only`, the brief, the per-item call bound, the resume limit, the timeouts,
  Codex's version and the tools), then the events of the case as they happen: `exec_started` (an exec's kind and
  prompt), `served` (one for each item the model was shown, in order: its kind, `k`, its text and its image as
  `<jpeg N bytes sha256 HEX16>`, never base64, with the harness's clock), `call` (each tool call the model made: its arguments, the reply, whether it
  was an error or premature, and when it arrived), `answer_outcome` and `exec_failed`; and one record per
  `codex exec`, written when it ends (the model's replies, Codex's JSON events with images as hashes, the tokens it
  used, and `killed` when the harness ended it). It holds the questions' text, so it never goes under the dataset or to
  a system;
- `codex.summary.json`, written even when the run fails, counted from the journal by `llm_summary`: the execs by kind
  (`start` and `resume`), the frames served and, among them, the frames that came with questions (`frames_asked`), the
  `next_frame` pulls and the premature ones among them (made while questions were unanswered), the resumes (an exec
  started to continue a turn that ended early), the per-frame latency (the time from serving a plain frame to the
  model's next pull: median, p90 and the count; `frame_latency`) and, apart, the latency of the frames that came with
  questions (`asked_latency`, timed to the pull after the answers: looking, deciding and answering), the tokens (the
  cached ones inside the prompt tokens: the cost of a run is its tokens, not dollars; a turn the harness kills prints no
  usage, so its tokens are missing and the summary says the count is a lower bound), the tool calls by name and the
  errors among them, the retries and the warnings, the answers by outcome (submitted, repaired, failed as `malformed` or
  `no_submit`, timed out), the execs that failed by reason, the last failed exec's message (`exec_failure_message`);
  and, beside them, the model, the effort, `text_only`, `compact_at` (null unless `--compact-at` was given), `memory`
  (the name, for a run with one), Codex's version, the case's status, its failures by reason and the wall time of every
  message (`CaseRun`), the leak scan's hits and `invalid`.

## How a run goes

A case is one Codex **thread**, held in one long `codex exec`. The harness pushes the stream's messages in order and the
model pulls them: it calls `next_frame` when it is ready for the next thing, and nothing advances until it does.
There is no fork, no ephemeral exec
and no reset; the state folder stays empty (the agent writes no file). One exec is held for the whole case because a
Codex turn waits about 12 to 13 s for its first token whatever it carries, while a request that follows a tool result
inside a turn took about 3 to 5 s: an exec for each frame would pay the first wait for every
frame.

The tools are `next_frame()`, with no arguments, and `submit_answer(label, answer)`. A `next_frame` call returns the next
item of the stream, in time order. The runner supplies it, so the model sets its own pace, and the robot's clock never
moves while it thinks (a frame's clock is the case's start clock plus the script's own time at that frame).

| Message | What the model is given |
| --- | --- |
| `task` | The first one starts the exec, and its message is the system prompt and the task message (the first brief, composed by `brief.py`): the exec's first prompt. Every later one is the reply to a `next_frame` call, with its own task message: `Task 1 begins at <clock>.`, the unobserved gap, the assignment. |
| `observe` | The reply to a `next_frame` call: one frame, its header line, `[k12 2026-03-02T08:00:12.0] base at (5.60, 5.05, 0.00), heading -90.0 deg, velocity (0.000, -0.250) m/s, camera pitch 5 deg; status: phase cleaning, task 0, battery_pct 97, brush true; subgoal: cleaning the kitchen (room 2 of 3)` (the velocity is the robot's own, the controller's odometry: the base's displacement from the previous frame of its task over the true time between them, in the map frame, to the millimetre a second, `(0.000, 0.000)` at a task's first frame; the subgoal, "what the robot is doing now", is there for a job that states one and is withheld under `--brief neutral`; it is on the one header line, so nothing else starts a line with `[k`), and, on `rgb_pose`, the camera JPEG as an MCP image block after it. Nothing follows the frame but the questions about it, when it has any (below): no cue. On `detections` no image comes: the same header line is followed by the frame's detections as text. In the [text-only option](#the-text-only-option) no image comes either: a frame is its header line alone. |
| `ask` | The reply to a `next_frame` call: the questions, labelled Q1..Qn in order and shown with their text, their answer type and that type's shape. The model answers each with `submit_answer(label, answer)`, then calls `next_frame`. A during ask comes with its frame, in the same reply (below). |
| `finish` | The reply to the `next_frame` the model has open: "The job is over: there are no more frames." The model answers with one short line; the harness gives that turn up to 60 s to complete and then ends the exec. No later exec: the usage the agent reports is the tokens of the case per model (here the one model), `tokens_in`, `tokens_cached_in`, `tokens_cache_write_in` and `tokens_out`, each summed over each exec's own share of the thread's running total: the `finish` report of `docs/agent.md`. **Codex prints no cache writes** (`cache_write_input_tokens` is 0 in every exec of codex-cli 0.159.3, while OpenAI bills GPT-6 cache writes at 1.25 times input), so `tokens_cache_write_in` is always 0 and the cost of a Codex run is a lower bound, short by at most 0.25 × its uncached input tokens × `in`. A turn the harness had to kill prints no usage, so its tokens are missing from the count, which is then also a lower bound. Read a Codex cost against another agent's knowing that; the count is re-read from the journals on each new Codex version. |

- **The pull** (`codex_feed`). The runner gives the feed one item at a time (a task message, a frame, an ask or the end),
  and the model's next `next_frame` call means it is done with that item. That call stays open until the runner's next
  message supplies the next item, which, the frames being on disk, is at once. So the harness's `task`, `observe` and
  `ask` each return when the model pulls again, and a frame's image is read from the inbox only while that frame is the
  current one. The server serves one call at a time: a `next_frame` while another is waiting gets an error. Between two
  items the model may make at most 100 tool calls (`max_calls_item`); `next_frame` is exempt, since a model told to pull
  must be able to, and any other call past the cap gets an error.
- **An ask is complete at the model's next accepted pull.** A `next_frame` while labels are unsubmitted gets an error
  naming them (the ask's one nudge); a second such call fails those labels as `no_submit` and returns the next item.
  The summary counts both as premature pulls. No frame follows an ask until every label is answered or the second premature
  pull. A bad submission is told why, with the type's shape, and gets one repair; a second is `malformed`.
- **A frame and its questions are one item** (0.10.0; `codex_system`, `codex_feed.Item.with_ask`). The controller's real
  moment is to look and to decide, and a separate ask item costs the model one pull more for every question. So the
  harness holds each frame, not serving it, until the runner's next message shows whether questions come with it. An `ask`
  at the held frame's own step **joins** it: the model's reply to one `next_frame` is the frame's header line, its image,
  then the questions, and it answers with `submit_answer` before it pulls again. An `observe`, a `task` or `finish` serves
  the held frame alone first, as before, then handles the message, and so does an ask at another step, so a frame is never
  held past the next message and the model never has frame k+1 before it has pulled after frame k and its questions. The
  frame's deadline starts when it is served, and an ask that joins it is timed by its own timeout (`ask_during`; the end
  ask, `ask_end`, joins the last frame when that frame is still held). Several asks at one point stay one question each:
  the first joins the frame and the next is served alone after the first is answered, so a later question's wording
  cannot inform an earlier answer. A held frame is dropped when the system closes. The protocol for any other system is
  unchanged: `observe`, then `ask`.
- **Deadlines.** The time from serving an item to the model's next pull is bounded by that message's timeout
  (`stream_protocol.TIMEOUTS`). Past it the exec is killed with everything it started and the case ends as
  `TransportError("timeout")`. The exec as a whole is bounded by the case's 12 h. Codex's own tool timeout is never
  near, since a pull is answered at once.
- **An early end.** If the model ends its turn before the job is over, the thread is resumed with one fixed line,
  "Call next_frame to continue." (`llm_render.RESUME_MESSAGE`); the journal counts the resumes. Three resumes in a row,
  with no pull between, are allowed, and the next early end ends the case as `stalled`. A resume does not renew the
  item's deadline, so a model cannot buy time by ending its turn.
- **The channel to the tool server.** The tool server (`codex_mcp`) is a child of Codex, so the harness cannot call it
  in process. It is a thin relay: the harness listens on the loopback interface with
  `multiprocessing.connection.Listener(("127.0.0.1", 0), authkey=...)` and a random 32-byte key, and the server's
  temporary configuration holds that address and the key and nothing else of the case. The server connects with
  `multiprocessing.connection.Client` and sends each call over the connection, and a frame's image bytes come back over
  it, so the server reads no file. The key keeps other local processes off the connection. **No module imports `socket`**:
  `tests/test_network_boundary.py` allows only `asset_fetch` to import a network module, and
  `multiprocessing.connection` opens the same connection without one.
- **The prompt** (`llm_render.system_prompt`) says only the role and the protocol, neutrally. It is built from
  `public/camera.json`, `public/robot.json`, the channel and one fact of the job, whether it asks anything during the
  job: the role (the brain of a robot doing a job, told the job, shown what the robot senses and asked questions), the
  frame of reference (metres, x east, y north, z up; headings counter-clockwise from +x; a camera pitch positive up),
  the camera as the robot knows it (the image size and horizontal field of view from `camera.json`, and its mount from
  `robot.json`'s `camera.mount_m`: one generic sentence built from the numbers, such as "The camera is mounted 0.38 m
  ahead of the base and 0.50 m above it", or "0.10 m below the base" for a camera that hangs under a flier, whose
  base z in each header is its altitude; the pitch is in each header; and the robot's body from
  `robot.json`'s `flies` and `body` (`llm_render.body_sentence`): "The robot's body is a disc 34 cm across and 9 cm
  tall, centred on the base" or "a box 80 cm long and 58 cm wide", and for a ground robot "It drives over anything up to
  2 cm high and passes under anything 9.5 cm or more above the floor" (the climb and the top of the band
  `grid.band` reads obstacles in), a flier's sentence stating neither; every number comes from `robot.json`, so
  "how far could your body go" has an answer), how things come (by calling `next_frame`, which returns the next thing in
  time order: a task message, one frame with the robot's pose, its own velocity, clock and status, the questions of an ask, or word that
  the job is over; nothing advances until the model calls it again), how to answer (`submit_answer` once for each
  question, by its label, using only what the robot has sensed so far, then `next_frame` again), that it replies with
  one short line when the job is over, the shape of each answer type the job asks (`answers.PROMPT_SHAPES`, in its
  order, and `unknown`: `brief.answer_types` finds the types from the `answer_type` of the families the job asks, its
  `looks`, its decision families and its end families or the questionnaire's default one, and `cmd_codex` passes them
  through `codex_system` as `kinds`, so a job is shown no shape it never asks, and the same bytes go to both arms of
  the leak check; an ask holding a type the prompt did not show is refused, and so is a case whose questions have one),
  the unknown rule and, last, how an answer is credited (`credit_sentence`, a section titled "How an answer is
  credited:" of plain bullets, one idea each in short sentences, with the same rules and numbers as the single
  paragraph it replaced, each part told only to a job whose own questions need it: by its answer types, its families
  and, for the turned corridor, its `clear_directions` (`brief.answer_types`, `asked_families`, `asks_turns`): first a definition of the point an offset names (of a thing or a person): of what the camera showed of it, the point
  nearest the floor point below the camera, measured along the floor (the nearest point of the convex hull the visible
  parts make on the floor), from what the camera shows now for a thing in view and from everything it showed clearly
  where the thing now stands for a thing seen earlier; a side it never showed does not count, and with the camera over
  the thing the answer is 0 ahead and 0 left; then the point a place names (`llm_credit.PLACE_TARGET`: the middle of the thing where it stood, its x and y in the map frame, not the point nearest the camera); then the corridor (`llm_render.CORRIDOR_TARGET`): how far the robot's body
  could move straight ahead, back or turned in place to its left or right and then driven is how far it can go along the
  heading before any part of it touches anything at its own height, counted as the distance its front edge (ahead, or
  after the turn) or rear edge (back) moves; a box sweeps its rectangle and a disc a disc, from `robot.json`'s shape; then
  the heading (`llm_render.HEADING_TARGET`): degrees to the left of the way the robot faces, from -180 to 180, the body
  turned in place and the run measured to the first thing it would touch. Then the credit itself: an answer earns four
  quarters, one for the right direction, one for being within twice the margin the job keeps (the clearance C, or the
  re-find radius R for a place in the map), one for being within that margin, and the last for being as exact as one pixel
  of the camera; overstating how far the body could move, or placing a thing or a person farther from the camera than it
  is, counts against the answer twice as fast (a worked case: a distance half the margin too long scores what one a
  whole margin too short does, and a distance a margin too long scores 0); a list earns each thing it holds the same way,
  and a thing left out costs a whole thing and a place that matches nothing half of one; a heading earns credit by how
  much room it gives up against the best heading; an answer with credit c scores c minus (1 - c), and the score of
  `unknown`. C, R and the matching radius are the job's `clearance_m`, `reacquire_m` and `match_m`, which `cmd_codex`
  reads from the job as it reads the fact below and passes through `codex_system`; the section is built from them and
  from `credit_kernel`'s constants, so it cannot disagree with the scorer, and it names a heading's credit and a list's
  only for a job that asks them; a caller that gives none gets no such section, and
  one that gives some but not all is refused; see `docs/agent.md`, "Credit"). Right
  after the map frame the prompt states the offset frame: an offset answer is metres from the camera, along the floor,
  ahead along the robot's heading (the one in each frame header, whichever way the camera is pitched) and left to
  its left, and a thing's height is no part of it (the shape line of
  `PROMPT_SHAPES["offset"]` says the same); and after the mount sentence it says the camera is the robot's only eye
  on the world, so any distance it gives is its own estimate. It says nothing of memory: not what or when to keep, not that the conversation may be shortened, no memory
  folder, no "locate things as you see them", no coaching of what to attend to. It names no mission, no floor plan and
  no room. The brief arrives in the task message, never in the prompt. Questions are said to come "at some points
  during the job and after it" only for a job that asks during the job, by a family at a decision point or at look
  points (`brief.asks_during`, from the job's `decisions` and `looks`), else "after the job", so
  the prompt never promises a question that does not come. It is read from the job and not from the run's `--during`
  arm, so both arms of the leak check send the same bytes. (The brief itself is the benchmark's, the same
  for every system; it too says nothing of what to keep and lists no kinds or colours.)
- **No depth image, on any channel.** The robot has a camera and no depth sensor (0.6.0; before it a
  depth image beside each camera image and `rgbd_pose` offered it as an option; both are gone, with `llm_depth`). A
  frame on `rgb_pose` carries its one camera image, and the `detections` arm and the text-only option carry none.
- **The tools** are `next_frame` and `submit_answer`, on every channel: no memory tool (`memory_list`, `memory_read`,
  `memory_write`, `memory_append`, `memory_search` were the retired five) and no `locate` (a pixel of a frame to a map
  point through the depth image; the agent estimates positions from what it is sent). The modules `llm_memory` and
  `llm_locate` stay in the package with their tests and nothing offers them. Outside an ask `submit_answer` answers that
  no question is pending. (One option adds an argument, not a tool: a memory that carries its record, below, is
  offered `next_frame` with an optional `memory`.)
- **Compaction is Codex's own, at its default.** The agent manages its own context, as it would in use, and is told
  nothing about it. The context grew about 480 tokens a frame in the per-message runs, so by that measure a medium case
  of 280 frames (the 90th percentile) holds about 140 k tokens and fits the 272 k window of luna and astra without
  compacting; a long case is longer and may reach the window. Whether Codex compacts in the middle of one turn, as a held
  exec needs, has not been exercised yet: if it cannot, a window that is full ends the case (`context`) and
  the case's failures count it. Nothing is retried to make room.
  `--compact-at TOKENS` forces an earlier limit, for experiments on memory beyond the context: every exec of the case,
  the resume included, is launched with `-c model_auto_compact_token_limit=TOKENS`, so Codex compacts the thread once
  its context passes that many tokens. Nothing else of the exec changes, and the agent is still told nothing of it. It
  is a lever of its own and never the reference agent's setting: `results.json` has `"compact_at"` (absent for a
  reference row), the reports name the system `codex:<model>, compacts at 50k tokens`, a resume into a folder made with
  another limit is refused, and `eval compare` reads it with the system, the effort and the memory as the one agent
  lever, so both arms of a comparison must share it.
- **What has run.** The pull is tested against a fake Codex. Whether an MCP image block in a tool result reaches the
  model, how long a frame takes, a long turn and compaction in the middle of a turn are measured against the real Codex
  by a live probe before any row is made.
- **An ask is read per label by the code every agent shares** (`agent_tools.read_submission`; the table of one ask is
  `codex_asks`, on the harness side). Questions are known to the model by label only: no qid is ever shown, and the
  answers map back. A failed answer is `{"type": "failed", "reason": ...}`, scored as malformed.

## The detections arm

`--channel detections` runs the same agent on the detections channel (methodology §1): the same thread, tools,
prompt (but for what the frames are made of), brief and questions, with each frame's **detections as text lines** in
place of its camera image. The arm measures **what the pixels cost**: what the pictures leave to the agent and the
detections hand it, namely detecting a thing, naming it in the question's words, locating it and deciding to keep it.
Run on the same cases as `rgb_pose`, the difference per family is that cost. It is not a pure perception loss, and
what is left on `detections` is not memory alone: each detection's text carries the thing's map centre (the coordinates
a gold is made of) and its kind and colour (the question's own vocabulary), so memory over it is a lookup, and an agent
that keeps the lines scores high (the 0.5.0 pilot, which ran the earlier scaffold with its memory tools:
sol +0.99). Read the arm as a ceiling for a
record kept from perfect detections, not as the memory score of an agent that must see.

What the agent sees on it. The system prompt says the images are not sent and what a detection line holds; each frame
is the `rgb_pose` frame with no image, and under its unchanged header line its detections:

```
[k12 2026-03-02T08:00:12.0] base at (5.60, 5.05, 0.00), heading -90.0 deg, velocity (0.000, -0.250) m/s, camera pitch 5 deg; status: ...
Detections in frame k12 (2):
  - mug (color red, size small): map (2.00, 1.00, 0.90), camera (0.10, -0.30, 1.50), size 0.10 x 0.10 x 0.12 m, box (120, 200) to (180.5, 260) px
  - door: map (3.10, 1.20, 1.00), camera (0.50, 0.00, 2.50), size 0.90 x 0.05 x 2.00 m, box (300, 10) to (420, 470) px, open true
```

- **Every field the channel gives a scripted system, and no more**: kind, attributes (colour, size), the centre in the
  map frame and in the camera frame (x right, y down, z forward), the size, the box in pixels, a door's `open` and
  (0.7.0) the **floor hull** of a recognisable thing, `floor hull [(1.000, 2.000), (1.500, 2.000), ...]`: the convex
  hull, counter-clockwise and to the millimetre, of the thing's visible pixels dropped onto the floor (map-frame x, y),
  which the system prompt explains. A detection without one (older perception) is rendered without the field. A frame
  with none says `Detections in frame k13: none`.
- **No key `d`, no identity.** `d` is fresh for every detection of every frame and names nothing
  (`docs/agent.md`), so it is not shown, and nothing says that two detections of two frames are one thing: that is
  the agent's memory work, as on pictures. A step's `identity` (the oracle's, privileged) is never read, and the
  renderer reads only the fields above, so no entity id, truth field or path can reach the model through a row.
- **The same tools, and no depth.** The tool server of this channel lists `next_frame` and `submit_answer`, as on
  `rgb_pose`. The prompt has no word of depth.
- The messages of the two channels are the same but for what the frames are made of: the arm adds a channel and
  changes nothing of what the headline agent is sent.

```sh
uv run stbench run codex:gpt-6.1-sol DATASET/private/<case id> --channel detections --reasoning medium
uv run stbench eval codex:gpt-6.1-sol --data DATASET --channel detections --reasoning medium     # a dataset, resumable
uv run stbench eval compare RUNDIR_RGB RUNDIR_DETECTIONS                                           # what the pixels cost
```

Tokens are counted per arm as for any run; a frame as text and a frame as an image cost differently, so the two arms'
costs are not a like-for-like price of perceiving.

## The text-only option

A zero-shot check of what the questions give away without the pictures: the same run as on `rgb_pose` with every image
withheld. It is an option and not a channel, since the stream the agent is sent is `rgb_pose`'s and only the Codex
agent's messages change:

- **No image is sent to the model**, though the stream still delivers the JPEGs. A frame is its header line (pose,
  heading, velocity, camera pitch, status, subgoal) and nothing else, with no image block after it.
- **The prompt says so**: after the camera's size, field of view and mount it adds "In this run you are not sent the
  images: a frame is its header line alone", and its frame bullet says a frame is its header line alone, with no image.
  Everything else (the offset frame, the answer shapes, the credit section, the tool) is the camera run's, so the two
  differ in the pictures only.
- **It is stamped on the run**: `"text_only": true` in the journal's `run` event and in `codex.summary.json` (false
  on a camera run), and the run prints "text-only run: no image was sent to the model". `cmd_codex.run` reads
  `args.text_only`; `stbench run codex:<model> ... --text-only` sets it. Compare a text-only run only with another
  text-only run or as a floor under the camera rows, never as a camera row.
- **It is refused on `detections`** (exit 2): that channel sends no image, so there is nothing to withhold.

A score at or under the scripted floor per ability says the questions are not answerable from the headers alone. It is
weaker than a text-only model trained on the questions; the scripted learners cross-validated over
scripts stand in for that.

## A memory beside the reference agent (`--memory SPEC`)

`--memory SPEC` (on `stbench run` and `stbench eval`, for `codex:` and [`claude:`](claude.md#a-memory-beside-the-agent---memory-spec)
agents) measures a memory store under the reference agent: the same transport, prompt, images, model, effort and tools, with one memory system beside the agent's
own context. Its baseline is the reference row of the same model and effort, and `eval compare` sees one lever, the agent
(below). SPEC is a JSON file (`codex_memory`):

```json
{"name": "mymemory",
 "serve": ["/path/to/python", "-m", "mymemory.server", "--model", "{model}", "{store}"],
 "sense": ["/path/to/python", "-m", "mymemory.feed", "{store}"],
 "prompt": "guide.md"}
```

`name`, `serve` and `prompt` are required; `sense` and `carry` are optional (`carry` needs `sense`), and any other key
is refused. `prompt` names a non-empty UTF-8 file beside SPEC. `name` is the
MCP server's name: a lower-case letter, then up to 31 lower-case letters, digits, `_` or `-`, and never `stbench` or
`codex`. `serve` must contain `{store}`.

**What never changes.** Every message and item the model is sent is the reference agent's, byte for byte: the first
prompt (the system prompt and the first task), every task, frame, ask, resume line and the end item. The prompt still
says nothing of memory; the guide is not in it. The tools `next_frame` and `submit_answer`, the pull, the held frame, the
questions, the credit and the scoring are unchanged, and so is the exec's whole argv but for the two things below.

**What changes** is three things, and only these:
- **The guide** (`prompt`, a UTF-8 text file beside SPEC) follows the one-line base instruction in the file Codex reads
  as its base instructions (`model_instructions_file`), after a blank line. It is how this agent uses this memory; it is
  never part of a message, because the first message is the start of one long thread that Codex may compact.
- **The server** (`serve`) is a second MCP server in every exec of the case (the start and each resume), configured as the
  tool server is: `required`, its calls pre-approved, the same startup and tool timeouts. `{store}` is the case's store,
  `<state_dir>/<name>`, absent at the start of each case, and `{model}` the model id. Its calls are the agent's own: they
  are not `escaped` (a call of any other server still is), they do not go through the feed and they do not count toward
  `max_calls_item`. The server must offer no tool that reads the disk outside its store (the held-out truth is on
  disk), such as one that adds a file from any path. Its argv holds no quote or line break.
- **The sensor feed** (`sense`, optional) is the robot's own sensors into the store: one process for the case, started
  before anything is sent to the model, that the harness writes one JSON object a line to (ASCII, flushed) and reads one
  JSON object line back from for each message, within 120 s (`codex_memory.SENSE_TIMEOUT`):

  ```
  {"start": {"camera": <public/camera.json>, "robot": <robot.json>}}
  {"task": {"index": 0, "clock": "...", "gap_s": 0.0, "brief": "..."}}
  {"observe": {"camera": <public/camera.json>, "steps": [STEP]}}        exactly one step
  {"record": <the model's memory argument>}                             only with "carry": true (below)
  ```

  A STEP is a whitelist, not the step as streamed: `k`, `clock`, `pose` (`pos`, `yaw_deg`), `camera` (`pos`, `rot`,
  `pitch_deg`), `velocity` and `subgoal` (when the step has them), `status`, and `rgb`, the absolute path of a file of the
  harness's own that holds the frame's image, which exists only while the message is handled. Never `identity`,
  `detections` or depth. The feed is never sent an ask, a question or the end, so the store learns nothing of any
  question. **A frame is fed as it is served to the model, and not before** (a change after 0.11.0: until then the harness fed a frame
  when it read it in, one message ahead of serving it, to learn whether questions join it, so the store held the next
  frame while the model looked at the one before). The store holds exactly the frames the model has been served: frame k
  is in it when the model is shown frame k, and frame k+1 is not until the model is served it. A frame served alone is
  fed just before it goes out, by the message that follows it (the next frame, a task, or the end); one served with its
  questions is fed just before it goes out with them; a task is fed to the memory after the frame before it. A frame the
  model is never served is never fed. A feed that cannot start, exits, closes its output, prints a line that is no JSON object or is silent past the timeout
  ends the case as the transport failure `memory`; it is journaled (`memory_failed`) and the summary keeps the reason.
  The feed is closed (its input ended, killed after 5 s) when the case closes.

**A memory that carries its record** (`"carry": true` in SPEC, optional, a bool, needs `sense`). The model then writes its
record in the call it was going to make anyway: `next_frame` takes an optional `memory` object, the record about the frame
it was just shown, "as the memory's claim takes it ({photo, record, request_id}), instead of a claim call", and the
memory's reply comes back with the next item. The description of the argument, the only text of the case the arm adds to
what the model is sent, is:

> Your memory record about the frame you were just shown, as the memory's claim takes it ({photo, record, request_id}),
> instead of a claim call; the memory's reply comes at the end of the next item.

The harness writes `{"record": <memory as given>}` to the sensor feed as one more line, and reads one JSON object line
back, **before it counts the pull**: the runner learns of a pull only by its arrival, so it never uses the one pipe to the
feed while a record is on it, and a second `next_frame` meanwhile is `BUSY`. The reply is appended as the **last text
block of the item the pull returns**, `<name>: <compact JSON>` (`mem: {"ok":true,"receipt":{"n":5}}`), after the item's
own blocks, which are unchanged and in order. A pull that is refused (`BUSY`, or made while questions are unanswered) carries
nothing, and after the end item nothing is carried. A feed that cannot be reached when a record is written is journaled
and ends the case as `memory` at the runner's next check, before any item is served: the pull is closed and the model
is told nothing of it. Without `carry` the tools, the
tool server's configuration and every byte the model is sent are as before, and a `memory` argument sends nothing.

**The store** is the agent's state, in the case's state folder: `state_bytes` counts it, and nothing carries over from one
case to the next. When the case closes it is copied beside the case's other files as `<name>-store`, without its
`cache/` (which the store rebuilds), replacing an earlier copy, so what the agent wrote can be read after the run.

**The marks.** The journal's `run` event records the memory as loaded: its `name`, the `serve` and `sense` argvs and the
guide's text itself (no hash), and `"carry": true` when it is on; each completed call of the memory's tools is a `memory_call` event (server, tool,
arguments, the reply's text, any image as `<jpeg N bytes sha256 HEX16>`, never base64) and the leak scan reads its
result as a tool message, as it reads an item the feed served. `codex.summary.json` has `memory` (the name) and counts
tool calls by name with the memory's among them (`claim` once for each frame is `tool_calls["claim"]` over `frames`). With
`carry` the record is not a tool call: the `call` event of the `next_frame` that carried it holds `arguments.memory` and
`memory_reply` (the reply object; absent for a pull that carried nothing), the
`served` event of the item it returned keeps `text` as the item's own and has `memory_reply` (the text the model read,
last), and the leak scan reads it as part of that item. A record that is not carried is a `memory_dropped` event with a
`reason`: `end` (a pull after the end item, which is also a `call`, without `memory_reply`) or `abandoned` (the pull's exec
ended while its record was carried: no `call`, and the event has the `memory_reply` nobody read). A feed that failed on a
record is `memory_failed` with message `record`.
`results.json` has `"memory": "<name>"` (absent for a reference row) and `results.md` and the reports name the system
`codex:<model> with memory <name>`. A default folder gains `-memory-<SPEC's stem>`; a resume into a folder of another
memory, or of none, is refused. `eval compare` and `pilot report` take the memory as part of the agent lever (with the
system and the reasoning effort), so a memory row and the reference row of its model and effort differ in one lever.

**It runs on `rgb_pose` with the images** (the feed is given the camera's frames): `--memory` is refused (exit 2) on the
`detections` channel, with `--text-only`, for an agent that is neither `codex:` nor `claude:` and for a SPEC that cannot
be loaded, before anything is sent.

```sh
uv run stbench run codex:gpt-6-luna DATASET/private/<case id> --reasoning xhigh --memory SPEC.json --out OUT
uv run stbench eval codex:gpt-6-luna --data DATASET --reasoning xhigh --memory SPEC.json
uv run stbench eval compare RUNDIR_REFERENCE RUNDIR_MEMORY
```

To measure the memory beyond the agent's context, force the compaction in both arms: the same `--compact-at 50000` on the
reference run and on the memory run. `eval compare` reads the limit with the memory in the one agent lever, so it would
not refuse a pair that differs in both; the arms must share the limit.

## Held to the agent's tools

Every exec, the case's first and each resume, is launched with `sandbox_mode=read-only`, `approval_policy=never`,
`--ignore-user-config`, `--ignore-rules` and `--strict-config` (an unknown key or feature is an error, never a tool left
on), `web_search=disabled`, an empty working folder, `--disable` for the shell and exec tools, the browser, computer
use, image generation, apps, plugins, multi-agent, image viewing, skills and the rest of the features that reach files
or the web, Codex's own instruction messages switched off (skills, permissions, environment, collaboration mode, apps)
and its coding-agent prompt replaced by one line. A patched copy of Codex's model catalog gives the chosen model direct
tool calls (the gpt-6 models otherwise reach tools only through a JavaScript `exec` tool), no apply_patch, no clock, no
multi-agent and no tool search. The tool server (`codex_mcp`, a stdio MCP server Codex starts for each exec, a thin
relay to the harness) is the only MCP server, and with a memory that memory's server is the second ([a
memory](#a-memory-beside-the-reference-agent---memory-spec)). What the model sees is its prompt, the server's two tools
(shown as `mcp__stbench.next_frame` and `mcp__stbench.submit_answer`) and the helpers Codex always adds
(`list_mcp_resources`, `list_mcp_resource_templates`, `read_mcp_resource`, which can only ask the tool server, and
`request_user_input`). Every exec's events are checked as they arrive (`codex_live`): an item that is not a message, a
thought or a call of the tool server or the memory's (a command, a file change, a web search) is `ToolEscaped`, the exec
is killed at once, not at its end, and the run is invalid, whatever the exit code says. The one thing no flag removes is
the `AGENTS.md` (or `AGENTS.override.md`) of the Codex home (`$CODEX_HOME`, else `~/.codex`): Codex adds it to every
exec as a message, so `Codex.prepare` refuses a home that holds a non-empty one (exit 2: move it aside for the run). The
restriction was tried live with a prompt that asks for a file by path, a directory listing, a shell command and a web
page.

**Why its file and shell tools stay off, and why nothing replaces them.** Codex's native file and shell tools could read
the private truth on disk (`DATASET/private/`), so they are disabled in every exec. The earlier scaffold gave the agent
a memory folder through the tool server in their place. The benchmark measures the agent's own memory, so the tool
server offers no such folder now: the agent remembers with its own context, which Codex compacts itself.

## Between cases

The agent carries nothing from one case to the next:
- Every case is a new thread, which no later case resumes. The agent has no memory folder and writes no file; the
  case's state folder stays empty.
- Codex keeps the user's login (`CODEX_HOME`, else `~/.codex`; the harness copies nothing of it) and its own session
  files there, as it does for every run. `--ignore-user-config` keeps the user's `config.toml` out of every exec, and
  Codex's `memories` feature is off by default (codex-cli 0.159.3) but no flag disables it (`codex_exec.DISABLED` does
  not name it); the one thing of the home that no flag removes is the personal `AGENTS.md`, which `prepare` refuses
  (above).
- **State sizes are not comparable across agents.** The reference agent's thread lives in the user's
  `~/.codex/sessions`, outside the case's state folder, so `state_bytes` is 0 for it, whatever it remembers. The
  Claude Code example's session (every frame in base64 among it) lives under its `CLAUDE_CONFIG_DIR`, which is
  `state/home/.claude` inside the state folder, and is counted (`docs/agent.md`, "Memory").
- What no flag can show is whether Codex carries anything between threads on its own side (its session files, its
  `memories` feature, its daemon). `tools/codex_canary.py` asks the real Codex: it plants a code word in one
  case's thread and asks that case, a second case and a bare fresh thread whether they were given one. A pass is that
  the planted case recalls it and neither of the others does. On codex-cli 0.159.3 it passed on `gpt-6.1-sol` (medium)
  and `gpt-6-luna` (xhigh), 2026-10-07, when a case was one exec for each message; the canary
  now runs the pull path (a case is one long exec, and the probe is a resume of it) and has not been run live since. It is repeated on every new Codex version (`uv run
  python tools/codex_canary.py --data DATASET --cases CID_A,CID_B --model MODEL --reasoning EFFORT --out OUT`,
  two cases on two scripts and a bare thread, and `--dry-run` shows what it would send). A failure is fixed before the next measurement, by
  disabling the feature in `codex_exec.DISABLED`.
- **What the canary does not reach.** It runs one case at a time, B straight after A, while a measured row runs about 20
  cases at once: a carry-over that needs concurrency is not tested. `memories` is off by default but disabled by no
  flag, and the user's Codex home may already hold a memories store: a Codex version that turned the feature on by
  default would let the user's own memories into every exec, which a planted token does not test. So the canary also
  writes `codex features list` (run under an empty temp `CODEX_HOME`, free, no model call) into `canary.json` as
  `features`: read the `memories` line there on every new Codex version.

## The leak scan, failures and exits

After the case, `llm_scan` reads the conversations rebuilt from the journal (`codex_log.transcripts`: the first prompt,
every item served as a tool message and the replies to the model's other calls, in order, as the model held them at
each ask and at the end): the leak scan's markers and the map's name in what the harness wrote, including each item
served; no question id, case id or dataset path in any message shown to the model; **no question's text before its
ask** (a question counts as asked only in its ask's own item, the last message of that ask's transcript: a plain ask,
or the frame item that carries it, where it is allowed in that item and nowhere before; every earlier message,
the frame served just before the ask included, is read against the asks before it, and what is served after the last
ask against all of them); and **no frame header with `k` above an ask's step in that ask's transcript**. The feed makes the last two
impossible by construction, since it serves one item at a time in stream order; the scan stays as the check. A hit, or an escaped tool, makes the run invalid: `codex.summary.json`
says `"invalid": true`, the run raises, `stbench run` exits 1 and writes none of `answers.jsonl`, `rows.jsonl` and
`summary.json`. The journal stays beside `codex.summary.json`, even when no exec finished: an escape on the first exec
leaves no other record of which tool it used (the journal's `exec_failed` event names it), and the folder then refuses
a rerun like any folder with a journal.

A Codex failure (a usage or rate limit, an auth failure, a timeout, a full window, a failed exec, a model that stopped
pulling) ends the case: it
is raised as a transport failure, the status says `ended: <message>: TransportError: <reason>` (`limit`, `auth`,
`timeout`, `context`, `escaped`, `missing`, `stalled`, `error`), the questions not yet answered are failed, and `stbench run`
exits 1; its `answers.jsonl` (the failed questions), `rows.jsonl`, `summary.json` and Codex files are written (an
`escaped` failure makes the run invalid, above), except that a run in which no exec finished leaves no journal. A
failure before the model did anything that looks transient (a 5xx, a dropped connection) is tried once more after 10 s;
nothing is retried in a loop. Exit 2 means the run was refused before anything was sent: no `codex`, a model it does
not list, an effort it does not list, a channel it does not read, or a folder with a journal.
