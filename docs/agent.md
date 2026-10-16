# Plugging in your own agent

An agent is one Python class in a file of its own. The harness runs it as the robot's brain for one case: it tells the
agent its job, feeds it what the robot's sensors saw one frame at a time, asks it questions as the job goes and at the
end, and scores the answers. The agent keeps whatever memory it wants, in any way it wants, for the whole case. Two
examples, standard library only, are in `examples/`: `minimal_agent` (it logs what it is sent and answers `unknown`) and
`template_agent` (every message and every answer shape, with TODOs). The commands below are written for a checkout
(`uv run stbench ...`); from a pip install, drop `uv run`. The protocol's definition, with the frame
rule and the failure rules, is the docstring of `src/stbench/stream_protocol.py`.

## What a run is

A **case** is a window of one script: a robot doing a job on a map for one to six tasks, delivered from its first step.
Everything the agent is sent is relative to the case: the first step is `k` 0 in task 0 at the case's own start clock,
whatever came before it in the script. The harness starts your class once, sends the messages below in order, and never
rewinds, forks or resets it: one process, one memory, one pass.

1. `start`: who you are for this run, the robot's spec and the channel.
2. For each task of the case, `task` (its brief, its clock, the gap since the last task), then one `observe` for each
   of the task's frames, in time order, with an `ask` right after the frame at any point the case asks about (a look
   point, which can be any frame from the fifth of a task on; the four jobs ask at no decision point since 0.7.0).
3. After the last frame, the end `ask` (all the end questions together), then `finish`.

**One frame per `observe`** (0.10.0). Every system gets exactly one frame in each `observe`, and its reply is the
request for the next: the harness sends nothing more until the reply has come, so the system sets the pace. The message
keeps its shape, a list of steps, now always a list of one, so an agent written for batches runs unchanged. Nothing in
the stream marks where a decision point falls. (Until 0.9.0 a message carried up to 8 frames, cut after every decision
point and every task's last step. That rule is now the question compiler's **look window**, and a system never sees one:
[questions.md](questions.md), "Look points".) An agent that pays a start-up cost for each message pays it for each
frame now: the Claude Code example runs one `claude -p` per message, so one per frame, and the reference Codex agent
avoids that by holding one session and pulling frames inside it ([codex.md](codex.md), "How a run goes"; the same agent
on Claude Code is [claude.md](claude.md)).

A case asks questions at two times. **During the job**, right after the frame it is about, which can be any frame of
the case (a look point; the robot's decision frames come first in the draw, so a question can show a frame is one). A
look point has up to three slots (0.10.0), each drawn from the job's table (`JobSpec.look_shares`) and each its own `ask`,
the first slot's first, so a look point with several questions is that many `ask` messages in a row with the same
`now`, each sent when the one before it is answered. Slot 1 is the job's harm questions (`clear_path`: how far its body
could move, ahead, back or turned in place to either side and driven; `closest_approach`: how close the person you can
see nearest it would come in the next 5 s; `tangle_ahead`: what on the floor its brush could catch; `drop_zone_people`
is written but no job asks it), slot 2 what it saw earlier and does not see in the last 8 frames (`ego_mem`), slot 3
the rest (`ego_now`: what the robot sees in its latest frame; `open_heading`: which way gives the body the most room). **After the job**, the end questions come together in one
`ask`. Which families a job asks is [questions.md](questions.md), and why they are these,
[question-design.md](question-design.md). An `ask` may come in the middle of a case, after any frame: write the agent for
both; the protocol is the same.

Nothing from the future is readable: the working folder's `inbox/` holds only the current frame's files, during its
`observe` and during the ask that follows, and it is emptied before the next frame. No question is ever asked before the step it is about, and no answer is ever
checked, so there is no feedback.

## The interface

```python
class MyAgent:
    def __init__(self, state_dir, workdir, **opts): ...   # state_dir: read-write; workdir: read-only
    def start(self, session, robot, channel): ...         # optional: the run's token, robot.json's dict, the channel
    def task(self, index, clock, gap_s, brief): ...       # optional: a task begins
    def observe(self, steps): ...                         # one frame: a list of one step; returning asks for the next
    def ask(self, when, now, questions): ...              # optional: {qid: answer} for a set of questions
    def answer(self, question): ...                       # one question's answer, if you have no `ask`
    def finish(self): ...                                 # optional: the usage you report, a dict (or {})
    def close(self): ...                                  # optional
```

`from system_shim import System` gives an optional base class (`self.state_dir`, `self.workdir`, `self.opts`,
`self.public`, and defaults: `task` and `close` do nothing, `ask` calls `answer` for each question, `finish` returns
`{}`). The file's folder is put first on `sys.path`, so its other modules import. It talks to the harness through a
small shim over JSON lines on stdin and stdout; your `print` goes to stderr, which is quoted when the agent dies.

**`start(session, robot, channel)`**: `session` is a fresh 16-hex token for this run (never the case id), `robot` is
`public/robot.json` as a dict (the robot's name, body, camera, the phases and status keys it reports),
`channel` is `rgb_pose` or `detections` (below). The same token is in your environment as `STBENCH_CASE_NONCE`: use it
to namespace any remote memory.

**`task(index, clock, gap_s, brief)`**: `index` counts from 0 in the case, `clock` is the local ISO time of the task's
first step (to the tenth of a second, as every clock), `gap_s` is the unobserved seconds since the last task ended (0 for the first), and `brief` is text: the
first brief says who you are, what the robot is doing, what you will be sent and the penalty for a wrong answer; a
later brief is that task's assignment alone.

**A step** is `{k, clock, ...}` and always case-relative:
- `k`: the step number from 0; `clock`: its local ISO time to the tenth of a second (`2026-12-06T09:39:02.4`, always one
  decimal), the form questions quote. The date and the time of day are drawn per case; the differences between clocks are
  true, and the agent's thinking never advances the clock ([question-design.md](question-design.md), "Timestamps").
- `pose`: the robot's base, `{pos: [x, y, z], rot: [w, x, y, z], yaw_deg}`; `camera`: the camera,
  `{pos, rot, pitch_deg}` (metres, x east, y north, z up). `status`: the robot's own readings, `{phase, task, ...}`.
- `velocity`: the base's own velocity `[vx, vy]`, metres a second in the map frame (0.10.0; the robot's controller knows
  how it is moving, so a system need not difference two poses). It is the displacement of the delivered base position
  from the previous delivered frame of the same task to this one, over the true time between the two (not the
  tenth-rounded clocks), to the millimetre a second, and `[0.0, 0.0]` at a task's first frame, which has no frame before
  it in its task. It comes with `pose`, on every channel that sends it. `closest_approach`'s gold reads these same
  numbers ([families/closest_approach.md](families/closest_approach.md)); the reference agent's header shows them
  ([codex.md](codex.md)).
- `subgoal`: a sentence on what the robot is doing now as part of its task ("carrying the red mug from the coffee table
  to the hall table"), present on every channel for a job that states one. It is the robot's own knowledge, written by
  the job from its plan, and it never tells a question's answer. A run under the `neutral` brief (below) is sent
  steps without it.
- `rgb`: path of a JPEG relative to `workdir` (`inbox/rgb/000012.jpg`), present on `rgb_pose`. `depth`: path of a
  16-bit PNG (`inbox/depth/000012.png`), present on `detections` only, the gate channel (a perfect detector, no
  sensor). `rgb_pose` sends none, and the reference agent is never sent depth on any channel: the robot has a camera and
  no depth sensor (0.6.0). The public files `robot.json` and `camera.json` still describe the scorer's private depth
  sensor (`depth_unit`, `near_m`, `far_m`); an agent on `rgb_pose` has no use for them.
- `detections`: `[{d, kind, attributes, bbox_px, center_cam, center_map, extent_m, floor_hull}]` (`floor_hull`: the outline the
  thing's visible pixels make on the floor, null when it is not recognisable in the frame; a person's is set
  by the person evidence rule instead), and `state`
  (`{"open": true}`) on
  a door: what a perfect detector saw, `d` a key that is fresh for every detection of every frame (it never names the
  same thing twice).
- `workdir/public/` holds `robot.json` and `camera.json`; `workdir/inbox/` the current frame's picture, emptied
  before the next frame.

**Channels.** `rgb_pose` (the default, the agents' channel and the headline) sends `rgb`, `pose`/`camera` and `status`
and no depth: the agent judges distance from the picture ([methodology.md](methodology.md), rule 9). `rgbd_pose` is
retired. `detections` (the gate channel, a perfect detector and no sensor) sends `depth`, `pose`/`camera`, `status`
and `detections`, and no pictures; the Codex reference agent is not sent that depth. A case is run on a channel that can answer at least one of its questions, and a
question whose `needs` name a sensor the channel does not send is not asked. The Codex reference agent reads `rgb_pose`
as each frame's camera image alone, with the camera's size, field of view and mount, the offset frame and how an
answer is credited in its system prompt, with the robot's body, the corridor, the heading and the credit by
consequence ([codex.md](codex.md), "How a run goes"); a run may also withhold
every image (the text-only option, a zero-shot check, not a channel: [codex.md](codex.md)); and it
reads `detections` as text lines, one for each detection with no key
`d` and no image: the arm that measures what the pixels cost (detecting, naming in the question's words, locating,
deciding to keep), whose difference per family from `rgb_pose` is that cost and not a pure perception loss, since its
text carries each detection's map centre and the question's vocabulary and memory over it is a lookup
([codex.md](codex.md), "The detections arm").

**`ask(when, now, questions)`**: `when` is `"during"` (right after the step `now = {k, clock}`, one question: a look
point with several questions is that many asks with the same `now`, and the first is answered before the next is sent) or
`"end"` (after the last step, the whole set). Each question is `{qid, text, answer_type}` and nothing else. Return
`{qid: answer}`; a question you leave out, answer twice or answer with something that is not an object counts as a
malformed answer, which has credit 0 and is penalised: it scores -1 at the default penalty, below `unknown` (0 where
the question has an answer, +1 where the thing was never seen). The robot holds still while you think: nothing
advances until you reply, and the same holds for a frame: the next one is not sent until `observe` has returned.

**An answer** is one of these objects (the strict parser is `answers.py`; an extra key or a wrong type has credit 0 and
is penalised the same):

| `type`      | fields                                                                                  |
| ----------- | --------------------------------------------------------------------------------------- |
| `position`  | `xy: [x, y]` or `xyz: [x, y, z]`, map metres                                            |
| `offset`    | `ahead_m`, `left_m`: metres from the camera along the floor, ahead of it and to its left |
| `heading`   | `deg_left`: a finite number of degrees from -180 to 180, to the left of the way the robot faces (to its right is negative) |
| `time`      | `clock`: local ISO time as the frames give it, to the tenth (`YYYY-MM-DDTHH:MM:SS.s`), no zone |
| `yesno`     | `value`: true or false                                                                  |
| `count`     | `value`: a non-negative integer                                                         |
| `number`    | `value`: a finite number (metres for a distance)                                        |
| `set`       | `items: [{xy: [x, y], change: "added" or "removed" or "moved"}]`, at most 100           |
| `positions` | `items: [{xy: [x, y]} or {xyz: [x, y, z]}]`, at most 100: places, no `change`           |
| `cells`     | `items: [[i, j], ...]`, integer cells of the answer grid (`grid.py`), at most 1,000     |
| `route`     | `points: [[x, y], ...]`, 1 to 50 map points in the order they are passed                |
| `unknown`   | nothing else; always allowed, and a wrong answer scores below it                        |

**`finish()`** returns the usage you report:

```json
{"models": {"<model id>": {"tokens_in": 1200, "tokens_cached_in": 800, "tokens_cache_write_in": 0, "tokens_out": 300}},
 "peak_context": 4096, "remote_store_bytes": null}
```

`models` holds one entry for each model you called, under the id you called it by; `{}` says you called none. The
tokens are case totals, never per message. `tokens_in` counts every input token, the cached and the cache-written ones
included; `tokens_cached_in` the ones read from a cache; `tokens_cache_write_in` the ones written to one; `tokens_out`
every output token, reasoning included. `peak_context` is the most tokens one model request held.
`remote_store_bytes` is the size of a remote store you keep, null if you keep none. Any value may be null (you do not
know), and `{}` reports nothing. It is self-reported and never scored. The cost beside a score is your tokens times the
benchmark's price table (`prices.py`), whatever you paid and never a figure of yours, and a model the table lacks
has no cost. A reported remote store (any number, 0 included) stamps the run `remote_memory` and makes it unverified.
These are the rules; the stamp and the cost beside a score are written by `stbench run`'s summary (`remote_memory`,
`cost_usd`, `cost_note`: `docs/format.md`) and carried into `stbench eval`'s results. The harness itself
measures the wall time of every message and the size of `state_dir` at the end of each task and at `finish`.

## Credit

Every metric answer is credited by the consequence of its error for the job, in four quarters
([questions.md](questions.md), "Credit"): one for the right direction (the side, the order, blocked or clear), one for
being within twice the margin the job keeps, one for being within that margin, and the last only for being as exact as
one pixel of the camera at the geometry the answer rests on. Between those points the credit falls smoothly, so no
answer coarser than a pixel earns the whole, and an answer scores `2s - 1` for a credit `s`. The margin is the job's
**clearance C** (`JobSpec.clearance_m`) for where a thing or a person is from the camera, how far the robot's body could
move and how close a person came, and its **re-find radius R** (`JobSpec.reacquire_m`, its matching radius unless it
declares one; 1.0 m in a home) for a place in the map. Since 0.11.0 the margin of a distance, an offset or a heading
widens to a tenth of the distance judged where that is more, never finer than two pixels of the camera there
(`N' = max(C, 0.1 g, 2 rho)`, [questions.md](questions.md), "Credit"): within ten margins of the robot it is C, and a
place keeps R.

An offset (where a thing is from the camera) names a point of the thing: of what the camera showed of it, the point
nearest the floor point below the camera, measured along the floor (the nearest point of the outline, the convex hull,
that the visible parts of the thing make on the floor). For a thing in view that is what the camera shows of it now; for
a thing seen earlier it is everything the camera showed clearly of it where it now stands, measured from the camera now.
A side the camera never showed does not count, and with the camera over the thing the answer is 0 ahead and 0 left. It
is credited on its two-dimensional error: the direction quarter is a bearing within 45 degrees of the true one (or an
error within C), and a thing or a person placed farther from the camera than it is counts against the answer twice as fast
(the margin is halved on that side).

The same C credits a distance in metres: how far the robot's body could move before it touched anything (`clear_path`),
and how close a person came (`closest_person`). The prompt defines the corridor (the body moves straight, and the
distance is how far it can go along the heading before any part of it touches anything at its own height; back is from
the back of the body, and a turn is the body turned in place and then driven from its front edge) and states the body
(its shape and size, its height, and for a ground robot the climb and the band's top) from `robot.json`, which a system
already receives. The gold sweeps the body's own shape: a box its rectangle, a disc a disc. Overstating how far the body
could move counts against the answer twice as fast, and the direction quarter is being on the right side of "is there
room to move at all" (the prompt says that of the corridor only; for how close a person came the same side is the
body's length plus C), and within a factor of two of the true distance when there is.

A place in the map is credited against R, to the middle of the thing where it stood (its x and y in the map frame,
which the prompt says in so many words: it is not the point nearest the camera that an offset names). For a thing longer
than 0.5 m the camera's resolution at the place grows by the thing's half-length beyond 0.25 m only when its top was
above the camera every time the robot saw it clearly there (the camera saw its sides, not its middle). The direction
quarter is the answer within R of the thing; the room the thing is in earns nothing, since it is a fact of the map and
the kind that needs no camera. The margins do not depend on how far the thing is. The Codex agent's system prompt states
them in metres and centimetres, from the same numbers as the gold rows.

A heading (`open_heading`) is credited by its regret: how much room it gives up against the best heading, in metres, on
the same four quarters with C as the margin (since 0.11.0, on the need rule above, with the best run for the distance
judged). The direction quarter is room to move: a heading whose run is at least the body's length plus C, or the best run
less a pixel where no heading has that much; a heading whose run is under the body's length plus C is on the harmful
side. The way to a remembered thing (`way_to`, the helper's) is credited by how much nearer to it one R of driving along
the heading brings the robot, against the best heading's, by the shortest way its body fits through the floor it has
seen. Its margin is the largest of C, a tenth of R and twice the field's cell (20 cm on the helper: "Its margin is 20 cm: the
largest of 15 cm, a tenth of 1 m and twice the 10 cm cell of the floor that the way is worked out on"), its quarter 4 is
whole within one cell, 10 cm, of the best heading's progress, and any heading with the room the body needs (the body's
length plus C: "the robot body's length plus 15 cm") has the right direction, even one that leads away from the thing; a
heading with less room than the body needs, or into floor never seen, scores as a wrong answer.

A list of places is credited per item: each thing in the list earns the same four quarters at the job's matching radius,
grown by the largest listed thing's half-length beyond 0.25 m whatever the camera saw, the places paired one to one
with the things in the pairing whose summed credit is largest. A list item is a place (0.11.0): a pair earns its
direction quarter within the matching radius of the thing, as a single place does within R, and nowhere else; before,
any pair within four matching radii earned it, which paid a prior that knew the layout. A thing left out costs a whole
thing and a place that matches no thing half of one, unless it is at a thing the question counts neither way. When
there were none, the empty list is the one thing: the credit is 1 over 1 plus half the number of false places (0.11.0;
before, one false place scored as ten did). An answer with credit c scores c minus (1 - c), so a right yes or no
scores 1 and a wrong one -1; `unknown` scores 0, and 1 where nothing the robot sensed answers the question.

An answer whose action would end in harm scores worse still (0.10.0). A harmful error larger than the job's clearance C
(or twice what one pixel of the camera resolves there, if more) is a **catastrophe** and scores minus 2 if the robot would
go into something it had seen (a corridor said longer than it is, where the true corridor is shorter than the body's
length plus C and the answer is not: an overstated corridor with room to spare is a wrong answer, not a collision), and
minus 3 if it would go into a person or onto a hazard of its own kind (a person said to be clear who came inside the
body, a person placed farther away than they are, a person or a cable-like thing left out of a list), whatever credit it
earned. `unknown` is never one. The weights are stated in the prompt, with the list omission that scores minus 3 for a
job that asks such a list (the vacuum's cord within 0.3 m of its front edge); the gates and the validity checks never
read them, and the reports give the share of harm answers that are catastrophes
([question-design.md](question-design.md), "How answers are graded").

| Job | Its place | Clearance C (m) | Matching and re-find radius (m) |
| --- | --- | --- | --- |
| `vacuum/1`, `vacuum/2` | a home: rooms of 3 to 5 m, things seen from 1 to 3 m | 0.10 | 0.5 (re-find 1.0) |
| `helper/1` | the same, with a reaching arm | 0.15 | 0.5 (re-find 1.0) |
| `amr/1` | a warehouse: sight lines of 5 to 10 m | 0.45 | 1 |
| `drone/1` | a downtown: seen from 7 to 15 m up | 2.4 | 2 |

C is `JobSpec.clearance`, declared with its source in the job's file ([questions.md](questions.md), "Credit"); the radius
is `JobSpec.match_m`, twice the job's unit (the first rung of its `ladder`), read from no gold.

## Memory

`state_dir` is yours: write anything there, and the harness reads only its size. The process may also keep memory of
its own, since it lives for the whole case. Nothing carries over to another case: each case starts with an empty
`state_dir`, a new process and a home of its own. Before your class is built the shim points `HOME`, `USERPROFILE`,
`APPDATA`, `LOCALAPPDATA`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_STATE_HOME`, `XDG_CACHE_HOME`, `CLAUDE_CONFIG_DIR`
and `CODEX_HOME` at empty folders under `state_dir/home`. A tool you run that keeps memory of its own there starts
empty in every case, and what it writes counts in the state size. That makes state sizes comparable only between agents
that keep their memory in the same place: the Claude Code example's session (every frame in base64) lives under
`CLAUDE_CONFIG_DIR`, inside the state folder, and is counted, while the reference Codex agent's thread lives in the
user's `~/.codex/sessions`, outside `state_dir` (`docs/codex.md`). What is worth keeping is for the agent to judge: the
briefs say what the job is and, where the job asks during questions, their kinds (below), and nothing of what to keep
or what the things are called.

The Codex reference agent keeps no `state_dir` at all: it has no memory tool, no locate tool and no instruction about
what to keep, and it remembers with its own context, which Codex compacts itself; its native file and shell tools stay
disabled because they could read the private truth on disk (`docs/codex.md`). Its 0.4.x rows
used an earlier scaffold with memory tools, memory coaching and `locate`; they measure the model with that scaffold,
not its own memory.

**Credentials.** A `--local` run inherits the environment of the shell that runs `stbench run`, so give a key as a
variable (`ANTHROPIC_API_KEY`, `CLAUDE_CODE_OAUTH_TOKEN`, `OPENAI_API_KEY`, ...). No login file of yours is in the
case's home, so a tool that needs a login takes it from a variable. The harness writes no credential anywhere. The
Docker sandbox passes none of the host's variables, and has no network.

**Not isolated** (the threat model is an honest or buggy agent): with `--local`, the temp folder (the user's, shared by
every case); memory kept at an absolute path or under a variable not listed; on Windows, a tool that asks the OS for
its folders (the Known Folder API, as Rust's `dirs` crate does) and so ignores the variables; a process that outlives
the case; and a remote store you do not report. `STBENCH_MAPS` (the folders of private, held-out maps) set in the
shell that runs `stbench run --local` reaches the agent like any variable and hands it the held-out maps' path: set it
per command, never in the shell of a `--local` run.

## The brief

A system is sent its brief as text with `task` (`brief.py`). The harness composes it at run time; it is never
a file. The first brief of a case is a paragraph: who the robot is and where it works, the case's first assignment, the
sensors the channel delivers, the during questions' kinds where the job asks any, and the penalty for a wrong answer.
Each later task's brief is that task's assignment alone. It describes the job and the place, never the exam: it names
no object, layout or answer. The brief gives the agent no help with its task, so it carries two things less than the
first drafts did:

- **No keep line.** It does not say what to keep ("Keep whatever information you judge useful ..."): what is worth
  remembering is the agent's to judge, like the memory itself.
- **No vocabulary.** It does not list the kinds and colours a question may use. Naming what it sees is part of
  perception; the words a question uses are the question's own, and they are what the agent gets. (The 0.5.0 pilot's
  vocabulary sentence, "Movable things are named by kind (...) and by colour (...).", is gone with its
  `brief.vocabulary` and `Catalog.light_kinds`.)

The during sentence is the one thing a brief says of the exam:

- **The during sentence** names the kinds of during question the job asks, and only those (`brief.during_sentence`,
  from the job's `decisions` and `looks`). Since 0.8.0 `clear_path` has the phrase "how far the robot could move" and
  `person_mem` (which no job asks now) reuses `ego_mem`'s, so the brief does not announce people; since 0.9.0 `open_heading` has "which way the
  robot has the most room", said once for a job that asks it; since 0.10.0 `closest_approach` has "how close something
  would come to the robot", `drop_zone_people` (no job asks it) "what stands near the point below the robot" and
  `tangle_ahead` "what lies on the floor near the robot" (none names a person or a thing). The AMR's reads "At some points during the job
  you will be asked where a thing you can see is from your camera, where a thing you saw before is from your camera, and
  how close something would come to the robot."; the drone's "... where a thing you saw before is from your camera, how
  far the robot could move, and what stands near the point below the robot."; the vacuum's "... where a thing you saw
  before is from your camera, how far the robot could move, which way the robot has the most room, and what lies on the
  floor near the robot."; the helper's names five kinds. The Codex system prompt says only that questions come "at some points
  during the job and after it". A job that asks none has no such sentence, so its brief promises no question that never
  comes. The sentence is the same bytes under every brief variant and in both arms of the leak check (`--during
  on|off`), which differ in the asks alone.

The variants (`--brief`) change the assignment text and, for `neutral`, the subgoal: `correct` is each task's own
assignment, and `neutral` replaces every task's by "It is carrying on with its usual work." and sends the steps
without their `subgoal`, since a subgoal that names the task's own things would state the assignment again. There is no
`swapped` variant any more: the subgoal says the true task in every frame, so a swapped brief would contradict it;
`stbench eval` refuses it, and so does a run of a job that states subgoals.

## When something goes wrong

A failed `start`, `task` or `observe` (an error reply, a message not answered in time, an exit) ends the case, and so
does an `ask` that times out, exits or breaks the protocol; every question not yet answered then scores as malformed.
The timeouts are `start` 120 s; `task`, `observe`, a during `ask` and `finish` 600 s; the end `ask` 3,600 s; and a
whole case may last 12 hours. An `ask` that answers with an error, or in the wrong shape, fails only its own questions
and the case goes on. A `finish` that fails leaves the answers standing. The summary beside the answers
(`summary.json`) says how the case ended and counts the failed questions by reason.

## Running it

```sh
# 1. Locally, no Docker: a plain subprocess through the same shim. Fast to try; NOT sandboxed.
uv run stbench run examples/minimal_agent/agent.py:MinimalAgent DATASET/private/<case id> --local
# 2. In the Docker sandbox: the verified mode. Same command without --local.
uv run stbench run my_agent/agent.py:MyAgent DATASET/private/<case id>
# options: --channel rgb_pose|detections  --brief correct|neutral  --during on|off  --timeout S  --out DIR
```

It writes `answers.jsonl` (`{"qid", "answer"}` in the order of the case's questions), `rows.jsonl` (one scored row for
each gold of the case) and `summary.json` under `DATASET/runs/<system>-<case id>/`, or `--out`, and prints the headline.
The summary says how the case ended, where it ran and what you reported spending, and holds a `scores` block: the
headline (the mean of the `now` and `memory` columns, each the mean reported score over its own questions, weighted
equally; a question is `now` when the latest frame alone gives its gold, which is what you hold at the ask, and `memory`
when an earlier frame is needed), the same split by scope, family, own-path reference and the other dimensions, and the counts of correct,
partial, wrong and abstained answers (`docs/format.md` has the keys). A run scores what it ran, so a case with no
`gold.jsonl` (one `stbench compile` has not reached) is refused before anything is sent. The exit code is 0 for a
complete case; 1 for one that ended early, for a run that could not be scored (its answers and summary are kept) and for
an invalid run (a Codex run whose leak scan hit, `docs/codex.md`); and 2 for a run that was refused before anything was
sent, a case with no gold included. A baseline (`all_unknown`, ...) runs the same way by its name, and a reference agent
as `<agent>:<model>` (`docs/codex.md`; `claude:<model>` is the same agent on Claude Code, `docs/claude.md`).

To run a system over many cases, `stbench eval SYSTEM --data DATASET` takes the same three kinds of SYSTEM and the same
options (`--channel`, `--brief`, `--during`, `--local`, `--timeout`, `--reasoning`) over the dataset's compiled cases:
`--workers N` at a time, resumable (a finished case is skipped, any other is run again from its start), under a spend
cap with `--max-usd TOTAL --case-usd X`, and `--arms LEVER=V1,V2` runs one lever's arms (the leak check's during on and
off, the correct and neutral briefs) on every case. Each case is written as above under `RUNDIR/cases/<case id>/`, beside a
`results.json` and a `results.md` with one aggregate a layout, and `stbench eval compare` lays results
side by side with paired differences. `docs/eval.md` has the options, the resume rule and the layout.

**`--local`** runs `python -I -B <shim> FILE CLASS` in a fresh temporary folder. It is the interpreter that runs
`stbench`, so numpy and Pillow are importable (install anything else it needs into that environment); the agent has the
network, your environment variables (an API key for a model works) and your permissions. Nothing enforces what the
sandbox does, namely read-only `workdir`, state only in `state_dir` and no peeking at the dataset (its `private/` folder
holds the gold, and a local agent can read it), so the summary is stamped `"sandbox": "none"`, `stbench run` says "run
without the Docker sandbox: not a verified score", and the result is an unverified one. The agent's folder must not
contain or sit inside the dataset. A crash fails the run with the end of the agent's stderr.

**Docker** is the mode a verified score comes from: no network, a read-only root, 4 GiB, your folder copied in. An agent
that calls a model API therefore cannot run there until a verified mode with an egress allowlist exists, and runs
`--local` meanwhile. "Verified" means the agent cannot reach the dataset; the sandbox refuses only truth-file names in
the agent's folder, so it does not rule out an agent that carries the public gold in its own code, and a result should
come with the agent's code. It needs Docker and the sandbox image, `python:3.12-slim`, and the harness never pulls it.
Pulling is your own step:

```sh
docker pull python:3.12-slim
```

The sandbox has only the standard library plus what your folder vendors, and it refuses an agent folder inside the
repository checkout or the dataset: copy `examples/minimal_agent/` out before running it there. An agent that runs with
`--local` using only the standard library runs in Docker unchanged.
