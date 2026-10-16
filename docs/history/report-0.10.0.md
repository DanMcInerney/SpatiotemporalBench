# SpatiotemporalBench 0.10.0: technical report

Version 0.10.0 · code: [DanMcInerney/SpatiotemporalBench](https://github.com/DanMcInerney/SpatiotemporalBench) · data:
not uploaded on its own (`DanMcInerney/spatiotemporalbench` 0.11.0 holds the same 64 shifts, compiled again)

*This is the technical report of 0.10.0, kept for comparison. It describes 0.10.0's one-frame pull, its
task-decision questions and its rows, which are Codex models (astra, sol and luna) on 42 cases. The current report is
[../report.md](../report.md); what changed is in [the changelog](../../CHANGELOG.md).*

This report goes deeper than the [README](../../README.md). It covers what 0.10.0 measures and, as plainly, what it does
not; how it scores; which questions it asks and why; what three reference models and two scripted references scored;
whether the design did what it set out to do; how validity is checked; and where the benchmark stops. 0.10.0 keeps the
renders of 0.7.0 to 0.9.0 and compiles them again: the agent pulls one frame at a time, and the questions are the
decisions each robot's brain faces, asked far more often. The report of 0.9.0, whose questions and rows this release
replaces, is kept in [history/report-0.9.0.md](report-0.9.0.md), and that of 0.7.0 in
[history/report-0.7.0.md](report-0.7.0.md). Every number comes from the release's compiled data and result
files, or from a review or a measurement of the data named where the number stands. The tables of results and gates are
pasted from the outputs of `stbench eval` and `stbench check` by `tools/fill_report.py`; the few figures no tool
prints (the age bands, the catastrophes, the full-credit shares and the family slices) were counted from each run's
`results.json`.

**In short.** A simulated robot does a real job in a photoreal place. Its camera frames reach the agent one at a time,
each with the robot's pose, its own velocity, the clock and its status, and no depth; the agent asks for the next frame
when it is done with the last. It is asked the decisions the robot's job depends on, harm first: how far its body could
move before it touched anything, how close the person nearest it would come in the next 5 seconds, what lies on the
floor that its brush could catch. Then come the questions about what it saw before: where things are from it, in view and
out of view, and where on the map it last saw them. Code checks each answer against the simulator's record and credits it
in four quarters by what its error would do to the job, and an answer whose action would have ended in harm is charged a
flat penalty in the reported score. 0.10.0 measures perception and memory. It does not test change over time: no job asks
what moved, appeared or went since an earlier pass, or where a moving person is going (Section 7). On 42 of the 124 short
and medium cases that ask a question (Section 4 says why 42), `codex:gpt-6-luna` (xhigh) scores -0.091 [-0.253, +0.046] on `apartment_a`, +0.008 [-0.132, +0.125] on `downtown_a`, +0.308 [+0.137, +0.459] on `warehouse_a`; `codex:gpt-6.1-sol` (medium) scores +0.461 [+0.325, +0.595] on `apartment_a`, +0.593 [+0.433, +0.742] on `downtown_a`, +0.554 [+0.431, +0.646] on `warehouse_a`; `codex:gpt-6-astra` (xhigh) scores +0.638 [+0.524, +0.717] on `apartment_a`, +0.768 [+0.630, +0.849] on `downtown_a`, +0.598 [+0.448, +0.777] on `warehouse_a`; the floor and the map reader score
`apartment_a`: `all_unknown` (rgb_pose) +0.032 [+0.016, +0.050], `map_reader` (rgb_pose) +0.220 [+0.134, +0.309]; `downtown_a`: `all_unknown` (rgb_pose) +0.132 [+0.078, +0.219], `map_reader` (rgb_pose) +0.647 [+0.543, +0.763]; `warehouse_a`: `all_unknown` (rgb_pose) +0.083 [+0.000, +0.167], `map_reader` (rgb_pose) +0.083 [+0.000, +0.167]. These are dev rows: the questions were
refined on the shifts they are scored on, and nothing is held out.

## 1. What is measured, and the setup

The agent acts as the robot's onboard brain and is asked what a controller asks itself. The questions are decisions, and
they fall into four kinds (the vision's, [vision.md](../vision.md)): **harm**, what an action now would hit, tangle on or
endanger; **help**, which action now best moves the task forward; **situation**, what is around the robot, in view and out
of view; and **reuse**, what is worth remembering to do the job here again. Each question is also one of the abilities
below, which decide the headline's cells.

| Ability | A controller asks itself | Column |
| --- | --- | --- |
| Distance | "Where is the nearest part of that drum I can see, from me, now?" | `now` |
| Memory | "Where, as I last saw it, is the red bin from where I am?" "Where did I last see the laundry basket?" | `memory` |
| Task relevance | "What stood within 2 m of the way I drove?" | `memory` |
| Clearance | "If I drove straight ahead, backed up or turned and drove, how far could my body go?" "How close would the person nearest me come in the next 5 s?" "How close did a person come?" "What on the floor could my brush catch?" | `now` or `memory`, by what answers it |
| Heading (diagnostic) | "Which way gives my body the most room?" | counted in no mean |
| Knowing what it does not know | `unknown` when the sensors never showed the answer | in every question |

**What is not measured.** Change over time is not tested. 0.10.0 measures where things are now, and where things the
robot saw earlier are, from seconds to hours before. The change families are built (`same_place`, `object_was` and
`changed`) and no job asks one: the world changes only between tasks, and the data hold too few changes that the robot
saw on both sides to ask a fair question about (Section 7). Anticipation is untested for the same reason. Control, free
text and the inside of a memory system are not measured either ([methodology.md](../methodology.md), section 1), and help is
not counted: the help kind holds `open_heading` alone, a diagnostic (Section 7).

| Job | Robot | Camera height and pitch | Place | One task |
| --- | --- | --- | --- | --- |
| `vacuum/2` | robot vacuum, a 34 cm disc | 0.08 m, fixed, 5° up | `apartment_a` | cleans one or two rooms |
| `helper/1` | Stretch-class home helper | 1.30 m, tilts from 60° down to 20° up | `apartment_a` | carries a thing from a surface in one room to a surface in another |
| `amr/1` | MiR250-class tote robot | 0.50 m, 10° down | `warehouse_a` | docks at tote shelves and packing tables, or runs a picking loop |
| `drone/1` | Wing-class tether delivery drone | under the body, 20° to 90° down; cruises at 15 m, lowers from 7 m | `downtown_a` | flies to a drop point, inspects it, lowers a package |

Every camera gives a 640 × 480 JPEG with an 80° horizontal field of view, a focal length of 381 pixels. Nothing is
labelled and no depth is sent: the agent estimates every distance from the pixels. Each frame has one header line with
the step and clock, the base's position and heading in map metres, the robot's own velocity, the camera's pitch, the
robot's status and its subgoal. The reference agent's prompt also states the robot's body, so that "how far could your
body go" has an answer.

![A frame from the home helper's camera: a dark TV console with a speaker, a picture frame, a router, a remote and books, over a wooden floor](../figures/helper-frame.jpg)

*A frame as the agent receives it: the helper's camera, 1.30 m up and pitched 45° down.*

**One frame at a time, pulled by the agent** ([agent.md](../agent.md); [codex.md](../codex.md), "How a run goes"). 0.9.0 sent
up to 8 frames in a message, cut after every decision point. 0.10.0 sends exactly one frame in each `observe`, and the
reply to it is the request for the next: the harness sends nothing until the reply has come, so the agent sets the pace.
- `now` is the latest frame. A question is `now` only when the frame the agent holds when asked gives its gold alone;
  what needs an earlier frame is `memory` ([methodology.md](../methodology.md), rule 4).
- The robot's own velocity is in every step (`velocity`, `[vx, vy]` in metres a second, the controller's odometry), so a
  system need not difference two poses.
- The reference agent holds one Codex thread for the whole case and pulls its frames inside it with `next_frame`; it
  answers with `submit_answer`. One long exec is kept because a Codex turn waits about 12 to 13 s for its first token
  however small the message, while a request after a tool result inside a turn took about 3 to 5 s.
- A frame and its questions arrive together, so the model looks and decides in one round trip. Several questions at one
  point are asked one at a time, so a later question's wording cannot inform an earlier answer.
- Nothing in the stream marks where a decision point falls. The old batch rule is the question compiler's look window,
  and a system never sees one.

**Shifts and cases.** The renders are those of 0.7.0 to 0.9.0: no frame was rendered for 0.10.0, and the scripts are
compiled again into cases. A shift is 2 to 6 tasks of one job. Between tasks the robot is parked and unobserved for 10
minutes to 3 hours, and the place changes: things move, doors open and shut, people come and go. A case is a window of a
shift, streamed from step 0 on a fresh clock: `short` (1 task), `medium` (2 or 3) or `long` (4 to 6). The reference rows
run the short and medium cases. **Counts** in the compiled data, from the release's `release.json`:

| Job | Place | Scripts | Cases (short / medium / long) | Questions | Frames |
| --- | --- | --- | --- | --- | --- |
| `vacuum/2` | `apartment_a` | 16 | 16 / 16 / 4 | 689 | 6,460 |
| `helper/1` | `apartment_a` | 16 | 16 / 16 / 4 | 513 | 3,157 |
| `amr/1` | `warehouse_a` | 16 | 16 / 16 / 4 | 235 | 4,679 |
| `drone/1` | `downtown_a` | 16 | 16 / 16 / 4 | 264 | 2,534 |

In all, 64 shifts were rendered once and compiled into 144 cases of
16,830 frames, with 1,701 questions.

| Kind | Family | Ability (column) | When | It asks | Answer |
| --- | --- | --- | --- | --- | --- |
| Harm | `clear_path` | clearance (`now` or `memory`) | during | how far the body could move before it touched anything: ahead, back, or after turning in place left or right | number |
| Harm | `closest_approach` | clearance (`now`; `memory` when the person walks) | during | how close the person it can see nearest would come in the next 5 seconds, if both kept their velocity | number |
| Harm | `tangle_ahead` | clearance (`memory`, or `now` when one frame shows every item) | during | what lies on the floor within 1 m, ahead or to the sides, that the brush could catch | a list of map points |
| Harm | `closest_person` | clearance (`memory`) | after the job | how close a person came during a task, as the robot saw them | number |
| Help | `open_heading` | heading (diagnostic) | during | which heading, turned in place, would let the body go farthest | heading |
| Situation | `ego_now` | distance (`now`) | during | where the nearest visible point of a thing in view is, from the camera, along the floor | offset |
| Situation | `ego_mem` | memory (`memory`) | during | the same for a thing seen before the last 8 frames of the task and not in them, from where the camera is now | offset |
| Reuse | `object_where` | memory (`memory`) | after the job | where on the map it last saw a thing | a map point |
| Reuse | `task_things` | relevance (`memory`) | after the job | where the things near its task were, by the job's rule: on the floor within 1.2 m of its way (helper), within 2 m of its drive (AMR) | a list of map points |

Two more families are built and asked by no job: `drop_zone_people` (who stands within 4 m of the point below the drone), for
want of supply, and `person_mem` (where a person last seen is, once out of view), which failed a gate on the helper
([families/drop_zone_people.md](../families/drop_zone_people.md), [families/person_mem.md](../families/person_mem.md)).

**Look points** ([question-design.md](../question-design.md), section 5; [questions.md](../questions.md), "Look points"). A look
family is asked right after a frame, and the frame can be any frame of the case:
- the candidates are the frames at least 4 steps into their task, except its last frame;
- the frames of each job's priority decisions come first in the draw (before a crossing, a station, a grasp, a doorway, a
  lowering, a room done), then the rest, each in a hash order of the case id and the frame;
- a frame is kept when it is at least 4 frames from every frame kept, until the case holds one point for every 6 frames.

A point asks one question for each of up to three slots, filled harm first. Each slot draws its family from a table the
job declares, as a percent of points. The vacuum and the AMR are asked only about things on the floor (`World.on_floor`):
a mug on a table is no obstacle to a vacuum.

| Job | Slot 1 (harm) | Slot 2 (what it saw before) | Slot 3 (the rest) | After the job |
| --- | --- | --- | --- | --- |
| `vacuum/2` | `clear_path` 50, `tangle_ahead` 50 | `ego_mem` 100 | `open_heading` 20 | `object_where` |
| `helper/1` | `clear_path` 50, `closest_approach` 50 | `ego_mem` 65 | `ego_now` 20, `open_heading` 15 | `object_where`, `task_things` |
| `amr/1` | `closest_approach` 100 | `ego_mem` 100 | `ego_now` 100 | `task_things`, `closest_person` |
| `drone/1` | `clear_path` 60 | `ego_mem` 70 | none | none |

The draw reads the case id, the plan and the table, never the world, a gold or an answer, so which question is asked
reveals nothing about what is in view. The `clear_path` direction is drawn the same way (ahead, back and left for the
vacuum; ahead, back, left and right for the helper; ahead and back for the drone). The end questions come together in one
ask after the last frame. Some questions have `unknown` for answer, in the same words as the rest: those about a thing, a
corridor or a patch of floor the robot never saw.

Questions by job and family, over all lengths with the short and medium cases in brackets, and the share of each
family's golds that is `unknown`, counted from the release's `gold.jsonl`:

| Job | Family | Questions | `unknown` golds |
| --- | --- | --- | --- |
| `vacuum/2` | `clear_path` | 315 (180) | 2% |
| `vacuum/2` | `tangle_ahead` | 133 (81) | 0% |
| `vacuum/2` | `open_heading` | 134 (85) | 0% |
| `vacuum/2` | `ego_mem` | 83 (70) | 20% |
| `vacuum/2` | `object_where` | 24 (21) | 17% |
| `helper/1` | `clear_path` | 96 (66) | 12% |
| `helper/1` | `closest_approach` | 22 (19) | 0% |
| `helper/1` | `open_heading` | 54 (40) | 0% |
| `helper/1` | `ego_now` | 52 (43) | 0% |
| `helper/1` | `ego_mem` | 168 (144) | 18% |
| `helper/1` | `object_where` | 74 (67) | 26% |
| `helper/1` | `task_things` | 47 (31) | 0% |
| `amr/1` | `closest_approach` | 24 (19) | 0% |
| `amr/1` | `closest_person` | 34 (30) | 0% |
| `amr/1` | `ego_now` | 105 (86) | 0% |
| `amr/1` | `ego_mem` | 27 (22) | 37% |
| `amr/1` | `task_things` | 45 (36) | 0% |
| `drone/1` | `clear_path` | 196 (144) | 9% |
| `drone/1` | `ego_mem` | 68 (55) | 21% |

In all, 1,701 questions, 1,239 of them on the short and medium cases.

## 2. Scoring

**The need is the job's.** What an error costs is set by what the job needs to act on the answer. Where a thing or a
person is from the camera, how far the body could move, how close a person came or would come, and a heading are credited
against the job's clearance C, the margin the robot's own operating spec keeps beside its path to the things it meets. A
place on the map is credited against the re-find radius R, within which the robot, driving back, finds the thing again
with it in view. A list is credited against its matching radius M.

| Job | C | R | M | Source of C |
| --- | --- | --- | --- | --- |
| `vacuum/2` | 0.10 m | 1.0 m | 0.5 m | the loose end of what camera vacuums aim to pass beside an obstacle |
| `helper/1` | 0.15 m | 1.0 m | 0.5 m | OWMM-Agent counts a pick when the end-effector is within 0.15 m of the target |
| `amr/1` | 0.45 m | 1.0 m | 1.0 m | the MiR250's specification: a 580 mm robot needs a 1,450 mm corridor, 0.435 m a side |
| `drone/1` | 2.4 m | 2.0 m | none | Wing's patent US12399508B2: a delivery point at least 8 ft from trees and 10 ft from buildings |

C is never a detection size, a docking or placement precision, or a facility-layout rule. The sources of R and the
sources rejected for C are in the report of 0.9.0, section 2 ([history/report-0.9.0.md](report-0.9.0.md)). The
values were fixed before 0.7.0's first run, have not moved since, and are stated in the prompt in metres and centimetres.

**Four quarters.** An answer that is `e` off the gold earns a credit `s` that is the mean of four quarters, one for each
layer of the vision, against the need N and the gold's resolution ρ, the error of one pixel of the camera at the
geometry the gold rests on, at least 1 cm (written into every gold row, `tolerances.resolution_m`). Where 2ρ is more
than N, N is widened to N' = 2ρ. The arithmetic is `credit_kernel` ([questions.md](../questions.md), "Credit").

| Quarter | It is earned by | Whole | Falls to nothing |
| --- | --- | --- | --- |
| Direction | the right side, the right order, blocked or clear | on the right side | a step, 0 or 1 |
| The right thing or action | being near the gold | within 2N' | at 4N', linearly |
| The right amount | being within what the job needs | within N' | at 2N', linearly |
| Precision | being as exact as the camera can resolve | within ρ | at N', log-linearly in e / ρ |

The direction is decided by the answer's type: for an offset, a bearing within 45° of the gold's, or an error within N;
for a distance in metres, the same side of T, the body's length plus C (for a corridor, whether there is room to move at
all) and, where the gold is at least T, within a factor of two of it; for a place, within R; for a heading, a run at
least half the best run; for a list item, being matched. The curve is continuous and strictly falling between ρ and 4N',
and its one step is the direction quarter, so no answer coarser than a pixel earns the whole. An answer with credit `s`
scores `2s - 1`:

| The answer | Credit | Score |
| --- | --- | --- |
| within one pixel of the gold (ρ) | 1 | +1 |
| at N | 3/4 | +0.5 |
| at 2N | 1/2 | 0, as good as `unknown` |
| at 4N or beyond, the direction right | 1/4 | -0.5 |
| beyond, the direction wrong | 0 | -1 |
| `unknown`, where the frames hold the answer | | 0 |
| `unknown`, where nothing the robot sensed answers | | +1 |
| anything else, where nothing the robot sensed answers | | -1 |
| a catastrophe, in the reported score only (below) | | minus its weight, 2 or 3 |

An answer beats `unknown` when it is within twice the need, and the prompt says to answer only when more likely right
than wrong: knowing one's own precision is part of the ability.

**Harm costs more than caution.** On the harmful side the need is halved, in all three of its quarters, and ρ is not: a
clearance answered longer than the gold (more room than there was), a closest approach answered wider (a person who came
closer than said), an offset whose error points away from the camera along the gold's bearing (a thing or a person placed
farther than it is), and a heading whose run is under T. A cautious error of 2N and a harmful one of N leave the robot
the same margin. A place to re-find a thing has no harmful side.

**A catastrophe is charged in the reported score only** ([question-design.md](../question-design.md), section 6;
[methodology.md](../methodology.md), rule 7). It is an answer whose action would end in harm, by a harmful error larger than
`max(C, 2ρ)`. Whatever credit it earned, it scores minus a flat weight, declared once in `credit_kernel` with its reason
and never tuned to a score:

| Family | An answer is a catastrophe when | Weight |
| --- | --- | --- |
| `clear_path` | the true corridor is under the body's length plus C (there was no room to move) and the answer is at or over it | 2: contact with a thing the robot saw |
| `closest_approach`, `closest_person` | a person came inside the body's length plus C and the answer says they stayed outside it | 3: contact with a person |
| `tangle_ahead` | the list leaves out a thing within 0.3 m of the body's front edge | 3: a hazard of the robot's own kind, a tangle |

An overstated corridor with room to spare is a wrong answer and no collision, so it is not a catastrophe, and `unknown`
never is. The reference agent's prompt states the weights, built from the same constants.

**Two scores.** The **gate score** is the penalised score `2s - 1` with no catastrophe term: every gate, the `knowable`
check, the evidence verdict and the leak check read it (a result row's `score_gate`). The **reported score** is the same
except for a catastrophe, and it is the one the headline, the reports and the reference rows read (the row's `score`). A
weight inside a gate would lower every shortcut that guesses, so a family could pass a gate because of the weight and not
because the sensors are needed; the design review measured it ([question-design.md](../question-design.md), section 6).

**Worked examples,** worked with the scorer from made-up golds to show the arithmetic; they are not rows of any run. The
helper (C = 0.15 m, a body 0.34 m long, so T = 0.49 m) is asked how far its body could move straight ahead. The gold is
0.30 m, under T: there is no room to move. One pixel there is ρ = 5 cm.
- The answer 0.45 m is on the same side of T (direction 1) and 15 cm too long, the harmful side, so N = 7.5 cm widens to
  2ρ = 10 cm: the thing quarter is 1, the amount quarter 0.5 and precision 0. The credit is 0.625 and the score +0.25.
- The answer 0.20 m is 10 cm short, the cautious side: N stays 15 cm, the credit is 0.84 and the score +0.68.
- The answer 1.0 m says there is room to move, the wrong side of T: credit 0 and a gate score of -1, where `unknown`
  scores 0. It is also a catastrophe, an answer over T where the gold is under it by more than max(C, 2ρ) = 15 cm, so its
  reported score is -2. Asked how close the nearest person would come, with a gold of 0.40 m inside T, the answer 2.0 m
  is the same error with weight 3: a gate score of -1 and a reported score of -3.

**The headline** is the mean of two columns that count equally, whatever their numbers of questions: `now`, the cells the
latest frame answers, and `memory`, the cells that need what the robot saw before. Inside a column each cell, an ability
in that column, counts equally and is pooled over its own questions (a ratio of sums). Clearance has a cell in each
column, by the evidence that answers each row. Each place's mix differs, so its columns hold:
- the **apartment** (the vacuum and the helper): `now` is distance (the helper's `ego_now`) and clearance (corridors and
  cords that one frame answers); `memory` is memory (`ego_mem` and `object_where`, pooled), relevance (`task_things`, the
  helper's) and clearance (what needs memory);
- the **warehouse**: `now` is distance (`ego_now`) and clearance (`closest_approach` on a person who stands, or on no
  one); `memory` is memory (`ego_mem`), relevance (`task_things`) and clearance (`closest_person`, and a person who
  walks);
- **downtown**: `now` is clearance, the drone's corridors in view; `memory` is memory (`ego_mem`) and clearance, the
  corridors that need memory.

A (job, ability, column) cell with fewer than 20 questions in the whole dataset is a diagnostic, shown and counted in no
mean, and so is the ability `heading`. There is no own-path column: the pose and status are given, so the robot's own
path is a log lookup and is never asked. Intervals are 95% percentile bootstraps over shifts (2,000 resamples), each
resample recomputing the abilities, the columns and the headline. No number pools places. Beside each headline the reports
give the unpenalised mean credit, the share of answers in each credit band, the abstention bound, and three things that
are not cells of the headline: the split by kind (harm, help, situation and reuse), the catastrophe rate (the share of the
harm rows answered that are catastrophes, for each job), and the score by the age of the evidence (0.10.0). Each gold row
records `age_s`, the clock seconds from its oldest evidence frame to the ask, and `age_band`: `now` (the latest frame
alone gives the gold), `recent` (memory, up to 10 s), `task` (older, in the task the question falls in) or `earlier` (in
an earlier task); both are null on an `unknown` row. The score by band is a memory curve: credit against how long ago the
robot saw the answer ([question-design.md](../question-design.md), section 1).

**Why the target is the nearest visible point.** A ground camera sees a thing's near face, and the middle sits behind it,
so an offset names the point of the thing's floor hull nearest the floor point under the camera, which a pixel-perfect
reader reproduces exactly. A place (`object_where`, an item of `task_things` or `tangle_ahead`) names the middle of the
thing where it stood, a place to drive back to; the prompt says which point each answer names. The 0.7.0 report measured
the choice ([history/report-0.7.0.md](report-0.7.0.md), section 2).

## 3. Which questions are asked, and why

**Every question is a decision.** The robot's brain faces four kinds of decision in its job, and each robot's real job
decides what is asked of it ([question-design.md](../question-design.md), section 2, has the table for each):
- **The vacuum** steers round floor obstacles, backs out and turns when cornered, and stops until a person frees it when a
  cord winds round its brush. *Harm:* how far it can drive on, back out or turn away (`clear_path`), and what lies within
  a metre that its brush could catch (`tangle_ahead`). *Help:* which way gives it room (`open_heading`). *Situation:*
  where the things on the floor it passed are from it now. *Reuse:* where it last saw a thing.
- **The home helper** drives to a thing, carries it, backs away from counters and turns in tight rooms, and shares the
  home with a resident. *Harm:* how far its body can go before it touches a table edge, a chair or a console, and how
  close the resident would come in the next few seconds (`closest_approach`). *Situation:* where things are, in view and
  out of view. *Reuse:* where things are kept, and what stood on the floor of its routes.
- **The AMR** carries totes between stations in aisles it shares with pickers, and must stop before a person in its path.
  *Harm:* how close the person nearest it would come (`closest_approach`), and how close one came during a task
  (`closest_person`). *Situation* and *reuse:* where the things on the floor were, and what stood within 2 m of its route.
  Its corridors are mostly the 5 m cap in open aisles, so it has no corridor or help question its sensors decide.
- **The drone** flies between its nest and a drop zone at 7 to 15 m and must not meet a building. *Harm:* how far it can
  fly ahead or back at its height. *Situation:* where the ground things it saw are from it now. It asks nothing about the
  people below it, for want of supply, and nothing after the job.

The strongest questions need what is in view now, what was seen before and whether it has changed since. 0.10.0 asks the
first two and not the third (Sections 1 and 7). Some ask about actions toward space the robot cannot see now, which only
memory answers: the corridor behind the robot or round a corner it has turned, the cord beside it that it passed frames ago.

**More questions, asked at decisions.** The frames are a case's fixed cost, and a question costs far more than a frame:
the reference agent takes about 1 to 3 s to look at a frame and tens of seconds to answer a question. 0.9.0 asked its
short and medium cases one question in about 30 frames, so a case streamed all its frames to ask a handful of questions
([question-design.md](../question-design.md), section 5). The vision's "Speed" section is amended accordingly: a case may ask
as many fair questions as its frames support within its time ([vision.md](../vision.md)). A case now asks about one question
in every 9 to 25 frames, by job (the same section has each job's density, and the supply that limits the AMR). The slot
tables of Section 1 put harm first at every point, and the frames of each job's priority decisions come first in the draw,
so the questions are asked where a controller would ask them. Asking more, and asking at decisions, has a cost the design
measures and does not hide:
- **Cueing.** A question that names a thing makes it salient. So no subject is asked twice in a case, wording comes from
  fixed templates, and the leak check of Section 6 measures how much asking during the job changes later answers.
- **A decision frame shows itself.** A question being asked can show that a frame is one of the robot's decision moments.
  The robot's status and subgoal already show that, and the plan alone decides where they fall, so for a family whose
  answer depends on the world at that moment it is harmless. For a structure family it was a channel: the final review of
  the branch found that a decision frame puts a corridor question at the same plan pose, by the same furniture, in every
  script, and a prior of other scripts' answers learned the helper's corridors. The helper's `before_place` frames left the
  decision frames, and the evidence verdict passes (Section 6).

**New in 0.10.0,** with the pages that hold the gold and the reasoning:
- **`closest_approach`** (AMR, helper; [families/closest_approach.md](../families/closest_approach.md)): the least floor
  distance over the next 5 s between the point below the camera and the nearest person, if each kept the velocity they
  have, capped at 10 m. It is the decision a planner makes about a person: the AMR's slows or yields when a person would
  come within its protective field. A person's motion counts only where the sensors pair their sightings (exactly one
  evidence person in the ask's frame and in the earlier frame used), and otherwise the point is a dead band.
- **`tangle_ahead`** (vacuum; [families/tangle_ahead.md](../families/tangle_ahead.md)): the cords and cloths on the floor
  within a metre ahead or to the sides, as a list of places. Bath mats, pedestal mats and clothes piles recur at the same
  spots from script to script, so they are don't-care circles and never items.
- **The robot's own velocity** in every step, which makes a `closest_approach` row whose person stands a `now` row, and
  **floor things only** for the floor robots, so that a thing's kind does not predict the slice.

**Dropped by rule.** A family that fails a gate is repaired as a broken instrument or dropped from its job, never forced or
tuned ([question-design.md](../question-design.md), sections 3 and 9):
- **`person_mem` on the helper.** On the first 0.10.0 compile 7 of its 19 questions were answerable and 12 had an
  `unknown` gold, so a perfect reader led a blind `unknown` by exactly the answerable share, +0.368, against the gate's
  0.4. It was dropped and not re-tuned. Where a person who has left the view is now goes unasked: an open gap.
- **`drop_zone_people` on the drone.** It drew 2 questions over the 36 development drone cases, under the 20 a counted
  cell needs, with no per-family diagnostic, so the final review took it off the drone.
- **0.9.0's drops** (the vacuum's `task_things`, a place's room as direction credit) stand
  ([history/report-0.9.0.md](report-0.9.0.md), section 3).

**Not asked, and why** ([question-design.md](../question-design.md), section 3, has the full list):

| Not asked | Why |
| --- | --- |
| What was moved, added or removed since an earlier pass | `changed` failed on layout priors: most of the vacuum's golds were the empty set, which the current view and the question's form guess. The supply of observed change is too small for any other change question (Section 7) |
| Questions at the start of a task | Only the vacuum has a task that carries on earlier work, and the blockers on a route to the dock are a layout prior |
| The task's own target, a blocked bay's blocker | The brief, the subgoal and the robot's response tell it (`World.told`) |
| The drone's `closest_approach` | A floor distance from a camera 15 m up is not the drone's harm decision |
| The AMR's corridors; `open_heading` for the AMR and the drone | Most of the AMR's golds are the 5 m cap, its best run is always the cap, and the open high street is the longest run from every point of the drone's square |
| Answers with uncertainty; weighting harm in the headline | `unknown` stays the one way to say the sensors did not show the answer, and harm costs more row by row, through the harmful side and the catastrophe term |

**Two structure families.** How far the body could move and which way has the most room depend partly on the place's fixed
structure, and the maps are public. The agent is never given a map, so it perceives and remembers the structure, and the
fixed structure around the robot is asked ([methodology.md](../methodology.md), rule 5). Every case starts a fresh agent that
has never seen the place. The **map reader** answers each corridor and heading from the public floor plan, and it is
published for one reason: to show which questions ask about fixed structure. It is a reference and not a competitor, and
it is not a gate. A high map-reader score on a question says that the question measures the perception of fixed geometry
and nothing that changes, so a score on it is a claim about a system that holds no copy of the map, and a held-out place
would make it a claim about every system. Only `clear_path` and `open_heading` are structure families; making another
needs the rule amended.

**The screens** of 0.7.0 and 0.9.0 stand: a question is asked only where a careful reader of the frames could answer it
within the job's accuracy, and never where the words or the place give it away (a thing is recognisable, has no look-alike
sighted in its colour, ties to a surface a reader with no depth can read, and does not move after it was seen; a
corridor's stopper is tied to a floor contact the camera can read to within C/2). Each screen reads the timeline and the
cameras, never a gold or a score, and removes questions without adding any. They are in
[history/report-0.9.0.md](report-0.9.0.md), section 3, and [questions.md](../questions.md), "How a gold is computed".
A person is evidence at 1,024 pixels with the feet tied to the floor, and a frame that shows someone it cannot place is a
dead band for `closest_approach`.

**The questions over the releases,** on the same cases: 0.6.0 asked 1,009, 0.7.0 785 and 0.9.0 668. 0.10.0's compile
holds 1,701, asked at far more points of each case.

## 4. Results per place

The reference agent is the Codex CLI on a ChatGPT login ([codex.md](../codex.md)). A case is one long thread. Its two tools
are `next_frame`, the robot's sensor feed, and `submit_answer`, and it has no memory tool: its context is its memory, and
Codex compacts it itself. Its prompt states the camera model, the robot's body, C, R, the four quarters of the credit, the
halved margin on the harmful side, the weights of a catastrophe, how a list and a heading are credited and what point each
answer names. The rows use the camera channel (`rgb_pose`), the correct brief and the during questions. Sol (`gpt-6.1-sol`) ran at medium reasoning effort, luna (`gpt-6-luna`) and astra
(`gpt-6-astra`; the installed Codex catalog lists no 6.1 astra) at xhigh, once each. The floor, `all_unknown`, answers
`unknown` to everything; the map reader answers each corridor and heading with the public map's raycast and `unknown` to
the rest.

**The rows cover 42 of the 124 short and medium cases that ask a question** (379 of their 1,239 questions: 20 apartment
cases, 11 warehouse and 11 downtown; 21 short and 21 medium). Every system ran the same 42. They are the cases all three
models had finished when the runs used up the Codex plan's weekly quota (cost, below). Which cases finished first was
set by the order the runs took them, not by any outcome: they are the first 45 cases by id, less 3 that astra had not
started. A first start on 14 October ended on a Codex-side failure: the cases in flight timed out and every later one was refused at its first message. Those cases were run again from the start, and in the runs that make the rows every case that started ended complete. The intervals are wide for it, and rows on every case are left for a later revision.
Coverage: `all_unknown`: 42 of 42 cases; `map_reader`: 42 of 42 cases; `codex:gpt-6-luna` (xhigh): 42 of 42 cases; `codex:gpt-6.1-sol` (medium): 42 of 42 cases; `codex:gpt-6-astra` (xhigh): 42 of 42 cases.

| Layout | System | Headline (95% interval) | `now` | `memory` | Cases scored |
| --- | --- | --- | --- | --- | --- |
| `apartment_a` | `codex:gpt-6.1-sol` (medium) | +0.461 [+0.325, +0.595] | +0.512 [+0.266, +0.696] over 34 | +0.410 [+0.266, +0.568] over 172 | 20 |
| `apartment_a` | `codex:gpt-6-astra` (xhigh) | +0.638 [+0.524, +0.717] | +0.629 [+0.436, +0.751] over 34 | +0.646 [+0.577, +0.720] over 172 | 20 |
| `apartment_a` | `codex:gpt-6-luna` (xhigh) | -0.091 [-0.253, +0.046] | -0.045 [-0.275, +0.169] over 34 | -0.137 [-0.332, +0.005] over 172 | 20 |
| `apartment_a` | `map_reader` | +0.220 [+0.134, +0.309] | +0.282 [+0.155, +0.420] over 34 | +0.157 [+0.086, +0.224] over 172 | 20 |
| `apartment_a` | `all_unknown` | +0.032 [+0.016, +0.050] | +0.000 [+0.000, +0.000] over 34 | +0.065 [+0.033, +0.100] over 172 | 20 |
| `warehouse_a` | `codex:gpt-6.1-sol` (medium) | +0.554 [+0.431, +0.646] | +0.618 [+0.367, +0.785] over 38 | +0.491 [+0.392, +0.605] over 29 | 11 |
| `warehouse_a` | `codex:gpt-6-astra` (xhigh) | +0.598 [+0.448, +0.777] | +0.692 [+0.464, +0.841] over 38 | +0.505 [+0.286, +0.773] over 29 | 11 |
| `warehouse_a` | `codex:gpt-6-luna` (xhigh) | +0.308 [+0.137, +0.459] | +0.480 [+0.274, +0.663] over 38 | +0.136 [-0.120, +0.402] over 29 | 11 |
| `warehouse_a` | `map_reader` | +0.083 [+0.000, +0.167] | +0.000 [+0.000, +0.000] over 38 | +0.167 [+0.000, +0.333] over 29 | 11 |
| `warehouse_a` | `all_unknown` | +0.083 [+0.000, +0.167] | +0.000 [+0.000, +0.000] over 38 | +0.167 [+0.000, +0.333] over 29 | 11 |
| `downtown_a` | `codex:gpt-6.1-sol` (medium) | +0.593 [+0.433, +0.742] | +0.499 [+0.246, +0.751] over 22 | +0.687 [+0.439, +0.799] over 46 | 11 |
| `downtown_a` | `codex:gpt-6-astra` (xhigh) | +0.768 [+0.630, +0.849] | +0.769 [+0.631, +0.901] over 22 | +0.766 [+0.509, +0.876] over 46 | 11 |
| `downtown_a` | `codex:gpt-6-luna` (xhigh) | +0.008 [-0.132, +0.125] | +0.080 [-0.075, +0.231] over 22 | -0.064 [-0.230, +0.096] over 46 | 11 |
| `downtown_a` | `map_reader` | +0.647 [+0.543, +0.763] | +1.000 [+1.000, +1.000] over 22 | +0.295 [+0.085, +0.525] over 46 | 11 |
| `downtown_a` | `all_unknown` | +0.132 [+0.078, +0.219] | +0.000 [+0.000, +0.000] over 22 | +0.263 [+0.156, +0.438] over 46 | 11 |

How to read the rows:
- **The floor's `memory`** is the mean, over the memory cells, of the share of each cell's questions that have `unknown`
  for answer, which scores +1 for it. Its `now` is 0 where every `now` question has an answer in view.
- **Each column is a mean of cells.** The cells of each place's columns are in Section 2. A cell with its questions and an
  interval is in the comparison below, one line for each (ability, column) and for each family. A cell marked
  `diagnostic` holds under 20 questions and counts in no mean.
- **The map reader** holds the map. Its rows are the reference for the structure families, and a place where its row is
  high is a place where many of the counted questions ask about fixed structure (Sections 3 and 7).
- **`open_heading` is reported and not counted.** The apartment's headline leaves its rows out; its line is in the
  comparison, flagged as a diagnostic, and the help kind in each run's `results.md`.
- **The rows are not comparable with 0.9.0's.** The questions, the gold values and the scope tags differ, and a compile
  draws fresh question ids.

**The rows in full.** The comparison below is `stbench eval compare` over the five runs, the floor first. Each line is a
mean of the reported score with its 95% interval, and each later column's paired difference from the floor, over the
questions both answered.

### rgb_pose channel, correct brief, during on
#### Results

- A = all_unknown, rgb_pose channel, correct brief, during on, sandbox in-process (the benchmark's own code), 42 of 42 cases finished
- B = map_reader, rgb_pose channel, correct brief, during on, sandbox in-process (the benchmark's own code), 42 of 42 cases finished
- C = codex:gpt-6-luna, rgb_pose channel, correct brief, during on, sandbox in-process (the benchmark's own code), 42 of 42 cases finished
- D = codex:gpt-6.1-sol, rgb_pose channel, correct brief, during on, sandbox in-process (the benchmark's own code), 42 of 42 cases finished
- E = codex:gpt-6-astra, rgb_pose channel, correct brief, during on, sandbox in-process (the benchmark's own code), 42 of 42 cases finished

Differences are each column minus A, over the questions both answered, paired by (case, question), with a 95% CI from a bootstrap over scripts and the number of pairs after it. Means are the reported score.

Each layout is scored apart from the others (its own map, kinds and jobs), and no number pools layouts. The mean is the reported score: a wrong answer costs more than `unknown`, which scores 0, so it can be negative, and an answer that would have ended in harm (a catastrophe) scores minus its weight.

The headline is the mean of the `now` and `memory` columns, which count equally whatever their numbers of questions (VISION rule 4), and inside a column its cells count equally, a cell being an ability in that column (an ability asked in both has a cell in each), each pooled over its own questions; so is the score above the floor, and every other line pools its own questions. A diagnostic ability, and a cell with fewer than 20 questions on a job, are shown and counted in no mean.

#### apartment_a

| family | A | B | B - A | C | C - A | D | D - A | E | E - A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **headline** | +0.032 [+0.016, +0.050] | +0.220 [+0.134, +0.309] | +0.187 [+0.097, +0.277] over 244 | -0.091 [-0.253, +0.046] | -0.123 [-0.288, +0.017] over 244 | +0.461 [+0.325, +0.595] | +0.429 [+0.289, +0.569] over 244 | +0.638 [+0.524, +0.717] | +0.606 [+0.487, +0.692] over 244 |
| `now` | +0.000 [+0.000, +0.000] | +0.282 [+0.155, +0.420] | +0.282 [+0.160, +0.423] over 41 | -0.045 [-0.275, +0.169] | -0.045 [-0.285, +0.163] over 41 | +0.512 [+0.266, +0.696] | +0.512 [+0.272, +0.692] over 41 | +0.629 [+0.436, +0.751] | +0.629 [+0.446, +0.750] over 41 |
| `memory` | +0.065 [+0.033, +0.100] | +0.157 [+0.086, +0.224] | +0.092 [+0.022, +0.152] over 203 | -0.137 [-0.332, +0.005] | -0.202 [-0.412, -0.045] over 203 | +0.410 [+0.266, +0.568] | +0.345 [+0.200, +0.514] over 203 | +0.646 [+0.577, +0.720] | +0.582 [+0.488, +0.679] over 203 |
| clearance, now (n 22) | +0.000 [+0.000, +0.000] | +0.565 [+0.310, +0.840] | +0.565 [+0.299, +0.848] over 22 | +0.012 [-0.385, +0.374] | +0.012 [-0.378, +0.349] over 22 | +0.554 [+0.179, +0.813] | +0.554 [+0.198, +0.792] over 22 | +0.672 [+0.311, +0.882] | +0.672 [+0.313, +0.879] over 22 |
| clearance, memory (n 84) | +0.036 [+0.000, +0.088] | +0.312 [+0.118, +0.491] | +0.276 [+0.077, +0.452] over 84 | -0.330 [-0.439, -0.234] | -0.366 [-0.499, -0.256] over 84 | +0.319 [+0.095, +0.544] | +0.284 [+0.043, +0.519] over 84 | +0.566 [+0.397, +0.708] | +0.530 [+0.331, +0.685] over 84 |
| distance (n 12) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 12 | -0.101 [-0.326, +0.110] | -0.101 [-0.299, +0.091] over 12 | +0.470 [+0.130, +0.658] | +0.470 [+0.164, +0.650] over 12 | +0.586 [+0.478, +0.639] | +0.586 [+0.485, +0.638] over 12 |
| heading, now (n 7, diagnostic) | +0.000 [+0.000, +0.000] | +0.243 [-0.167, +1.000] | +0.243 [-0.167, +1.000] over 7 | -0.043 [-0.833, +1.000] | -0.043 [-0.833, +1.000] over 7 | -0.071 [-0.333, +1.000] | -0.071 [-0.333, +1.000] over 7 | +0.000 [-0.167, +1.000] | +0.000 [-0.167, +1.000] over 7 |
| heading, memory (n 31, diagnostic) | +0.000 [+0.000, +0.000] | -0.038 [-0.257, +0.237] | -0.038 [-0.257, +0.220] over 31 | -0.721 [-0.880, -0.568] | -0.721 [-0.868, -0.566] over 31 | -0.143 [-0.330, +0.183] | -0.143 [-0.332, +0.173] over 31 | +0.065 [-0.219, +0.412] | +0.065 [-0.207, +0.411] over 31 |
| memory (n 82) | +0.159 [+0.079, +0.233] | +0.159 [+0.079, +0.233] | +0.000 [+0.000, +0.000] over 82 | -0.162 [-0.320, +0.006] | -0.320 [-0.529, -0.092] over 82 | +0.370 [+0.303, +0.439] | +0.211 [+0.113, +0.322] over 82 | +0.556 [+0.439, +0.662] | +0.397 [+0.228, +0.577] over 82 |
| relevance (n 6) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 6 | +0.080 [-0.501, +0.448] | +0.080 [-0.445, +0.431] over 6 | +0.540 [+0.148, +0.914] | +0.540 [+0.196, +0.899] over 6 | +0.818 [+0.751, +0.928] | +0.818 [+0.761, +0.918] over 6 |
| clear_path | +0.041 [+0.000, +0.092] | +0.529 [+0.263, +0.742] | +0.488 [+0.201, +0.707] over 73 | -0.448 [-0.648, -0.251] | -0.489 [-0.683, -0.290] over 73 | +0.332 [+0.170, +0.498] | +0.291 [+0.104, +0.478] over 73 | +0.663 [+0.400, +0.828] | +0.622 [+0.341, +0.805] over 73 |
| closest_approach | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 3 | +0.517 [+0.420, +0.710] | +0.517 [+0.420, +0.710] over 3 | +0.704 [+0.215, +0.949] | +0.704 [+0.215, +0.949] over 3 | +0.870 [+0.710, +0.950] | +0.870 [+0.710, +0.950] over 3 |
| ego_mem | +0.140 [+0.073, +0.220] | +0.140 [+0.073, +0.220] | +0.000 [+0.000, +0.000] over 57 | -0.339 [-0.475, -0.193] | -0.479 [-0.665, -0.306] over 57 | +0.215 [+0.128, +0.300] | +0.075 [-0.062, +0.197] over 57 | +0.486 [+0.373, +0.603] | +0.346 [+0.173, +0.517] over 57 |
| ego_now | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 12 | -0.101 [-0.326, +0.110] | -0.101 [-0.299, +0.091] over 12 | +0.470 [+0.130, +0.658] | +0.470 [+0.164, +0.650] over 12 | +0.586 [+0.478, +0.639] | +0.586 [+0.485, +0.638] over 12 |
| object_where | +0.200 [+0.000, +0.348] | +0.200 [+0.000, +0.348] | +0.000 [+0.000, +0.000] over 25 | +0.241 [-0.020, +0.563] | +0.041 [-0.343, +0.511] over 25 | +0.722 [+0.588, +0.830] | +0.522 [+0.281, +0.758] over 25 | +0.715 [+0.486, +0.878] | +0.515 [+0.164, +0.839] over 25 |
| open_heading | +0.000 [+0.000, +0.000] | +0.014 [-0.177, +0.269] | +0.014 [-0.171, +0.261] over 38 | -0.596 [-0.833, -0.307] | -0.596 [-0.836, -0.321] over 38 | -0.130 [-0.317, +0.147] | -0.130 [-0.314, +0.144] over 38 | +0.053 [-0.175, +0.333] | +0.053 [-0.156, +0.339] over 38 |
| tangle_ahead | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 30 | +0.122 [-0.174, +0.409] | +0.122 [-0.159, +0.395] over 30 | +0.422 [+0.140, +0.813] | +0.422 [+0.130, +0.791] over 30 | +0.377 [+0.128, +0.637] | +0.377 [+0.144, +0.614] over 30 |
| task_things | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 6 | +0.080 [-0.501, +0.448] | +0.080 [-0.445, +0.431] over 6 | +0.540 [+0.148, +0.914] | +0.540 [+0.196, +0.899] over 6 | +0.818 [+0.751, +0.928] | +0.818 [+0.761, +0.918] over 6 |

| relevance | A | B | B - A | C | C - A | D | D - A | E | E - A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| answerable, critical | +0.000 [+0.000, +0.000] | +0.185 [+0.069, +0.322] | +0.185 [+0.069, +0.322] over 228 | -0.246 [-0.319, -0.160] | -0.246 [-0.319, -0.160] over 228 | +0.286 [+0.184, +0.416] | +0.286 [+0.184, +0.416] over 228 | +0.508 [+0.416, +0.624] | +0.508 [+0.416, +0.624] over 228 |
| answerable, situational | - | - | - | - | - | - | - | - | - |
| unknown | +1.000 [+1.000, +1.000] | +0.625 [+0.285, +1.000] | -0.375 [-0.714, +0.000] over 16 | -0.500 [-0.846, -0.077] | -1.500 [-1.826, -1.091] over 16 | +0.500 [+0.000, +1.000] | -0.500 [-0.933, -0.105] over 16 | +0.375 [+0.000, +0.800] | -0.625 [-0.947, -0.235] over 16 |

- B - A on the critical answerable rows: shown (lower bound +0.069, over 228 pairs)
- C - A on the critical answerable rows: not shown (lower bound -0.319, over 228 pairs)
- D - A on the critical answerable rows: shown (lower bound +0.184, over 228 pairs)
- E - A on the critical answerable rows: shown (lower bound +0.416, over 228 pairs)

#### downtown_a

| family | A | B | B - A | C | C - A | D | D - A | E | E - A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **headline** | +0.132 [+0.078, +0.219] | +0.647 [+0.543, +0.763] | +0.516 [+0.375, +0.635] over 68 | +0.008 [-0.132, +0.125] | -0.124 [-0.334, +0.002] over 68 | +0.593 [+0.433, +0.742] | +0.461 [+0.242, +0.624] over 68 | +0.768 [+0.630, +0.849] | +0.636 [+0.437, +0.735] over 68 |
| `now` | +0.000 [+0.000, +0.000] | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] over 22 | +0.080 [-0.075, +0.231] | +0.080 [-0.075, +0.231] over 22 | +0.499 [+0.246, +0.751] | +0.499 [+0.246, +0.751] over 22 | +0.769 [+0.631, +0.901] | +0.769 [+0.631, +0.901] over 22 |
| `memory` | +0.263 [+0.156, +0.438] | +0.295 [+0.085, +0.525] | +0.031 [-0.250, +0.269] over 46 | -0.064 [-0.230, +0.096] | -0.328 [-0.650, -0.185] over 46 | +0.687 [+0.439, +0.799] | +0.423 [+0.038, +0.618] over 46 | +0.766 [+0.509, +0.876] | +0.503 [+0.110, +0.679] over 46 |
| clearance, now (n 22) | +0.000 [+0.000, +0.000] | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] over 22 | +0.080 [-0.075, +0.231] | +0.080 [-0.075, +0.231] over 22 | +0.499 [+0.246, +0.751] | +0.499 [+0.246, +0.751] over 22 | +0.769 [+0.631, +0.901] | +0.769 [+0.631, +0.901] over 22 |
| clearance, memory (n 32) | +0.312 [+0.154, +0.500] | +0.375 [+0.000, +0.692] | +0.062 [-0.500, +0.539] over 32 | +0.105 [-0.139, +0.428] | -0.208 [-0.411, +0.033] over 32 | +0.808 [+0.682, +0.924] | +0.496 [+0.269, +0.712] over 32 | +0.859 [+0.814, +0.914] | +0.547 [+0.366, +0.728] over 32 |
| memory (n 14) | +0.214 [+0.067, +0.501] | +0.214 [+0.067, +0.501] | +0.000 [+0.000, +0.000] over 14 | -0.233 [-0.635, -0.086] | -0.447 [-1.095, -0.144] over 14 | +0.565 [+0.145, +0.745] | +0.351 [-0.304, +0.666] over 14 | +0.673 [+0.187, +0.843] | +0.459 [-0.220, +0.734] over 14 |
| clear_path | +0.185 [+0.091, +0.270] | +0.630 [+0.460, +0.818] | +0.444 [+0.190, +0.727] over 54 | +0.095 [-0.082, +0.273] | -0.091 [-0.281, +0.082] over 54 | +0.682 [+0.554, +0.824] | +0.497 [+0.357, +0.670] over 54 | +0.823 [+0.761, +0.879] | +0.637 [+0.557, +0.739] over 54 |
| ego_mem | +0.214 [+0.067, +0.501] | +0.214 [+0.067, +0.501] | +0.000 [+0.000, +0.000] over 14 | -0.233 [-0.635, -0.086] | -0.447 [-1.095, -0.144] over 14 | +0.565 [+0.145, +0.745] | +0.351 [-0.304, +0.666] over 14 | +0.673 [+0.187, +0.843] | +0.459 [-0.220, +0.734] over 14 |

| relevance | A | B | B - A | C | C - A | D | D - A | E | E - A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| answerable, critical | +0.000 [+0.000, +0.000] | +0.800 [+0.655, +0.947] | +0.800 [+0.655, +0.947] over 55 | -0.021 [-0.126, +0.080] | -0.021 [-0.126, +0.080] over 55 | +0.614 [+0.468, +0.733] | +0.614 [+0.468, +0.733] over 55 | +0.779 [+0.715, +0.838] | +0.779 [+0.715, +0.838] over 55 |
| answerable, situational | - | - | - | - | - | - | - | - | - |
| unknown | +1.000 [+1.000, +1.000] | -0.538 [-0.875, +0.000] | -1.538 [-1.875, -1.000] over 13 | +0.231 [-0.286, +0.667] | -0.769 [-1.273, -0.308] over 13 | +0.846 [+0.500, +1.000] | -0.154 [-0.545, +0.000] over 13 | +0.846 [+0.500, +1.000] | -0.154 [-0.545, +0.000] over 13 |

- B - A on the critical answerable rows: shown (lower bound +0.655, over 55 pairs)
- C - A on the critical answerable rows: not shown (lower bound -0.126, over 55 pairs)
- D - A on the critical answerable rows: shown (lower bound +0.468, over 55 pairs)
- E - A on the critical answerable rows: shown (lower bound +0.715, over 55 pairs)

#### warehouse_a

| family | A | B | B - A | C | C - A | D | D - A | E | E - A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **headline** | +0.083 [+0.000, +0.167] | +0.083 [+0.000, +0.167] | +0.000 [+0.000, +0.000] over 67 | +0.308 [+0.137, +0.459] | +0.225 [+0.086, +0.347] over 67 | +0.554 [+0.431, +0.646] | +0.471 [+0.312, +0.601] over 67 | +0.598 [+0.448, +0.777] | +0.515 [+0.340, +0.723] over 67 |
| `now` | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 38 | +0.480 [+0.274, +0.663] | +0.480 [+0.275, +0.657] over 38 | +0.618 [+0.367, +0.785] | +0.618 [+0.383, +0.780] over 38 | +0.692 [+0.464, +0.841] | +0.692 [+0.460, +0.840] over 38 |
| `memory` | +0.167 [+0.000, +0.333] | +0.167 [+0.000, +0.333] | +0.000 [+0.000, +0.000] over 29 | +0.136 [-0.120, +0.402] | -0.031 [-0.184, +0.105] over 29 | +0.491 [+0.392, +0.605] | +0.324 [+0.216, +0.499] over 29 | +0.505 [+0.286, +0.773] | +0.338 [+0.124, +0.656] over 29 |
| clearance, now (n 4) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 4 | +0.802 [+0.574, +1.000] | +0.802 [+0.603, +1.000] over 4 | +0.645 [+0.128, +0.978] | +0.645 [+0.228, +0.955] over 4 | +0.672 [+0.155, +0.976] | +0.672 [+0.226, +0.952] over 4 |
| clearance, memory (n 11) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 11 | +0.443 [+0.029, +0.810] | +0.443 [+0.052, +0.798] over 11 | +0.783 [+0.641, +0.893] | +0.783 [+0.644, +0.892] over 11 | +0.875 [+0.797, +0.954] | +0.875 [+0.797, +0.953] over 11 |
| distance (n 34) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 34 | +0.159 [-0.201, +0.435] | +0.159 [-0.194, +0.434] over 34 | +0.591 [+0.463, +0.710] | +0.591 [+0.465, +0.705] over 34 | +0.712 [+0.640, +0.776] | +0.712 [+0.639, +0.775] over 34 |
| memory (n 6) | +0.500 [+0.000, +1.000] | +0.500 [+0.000, +1.000] | +0.000 [+0.000, +0.000] over 6 | +0.455 [+0.007, +1.000] | -0.045 [-0.122, +0.003] over 6 | +0.792 [+0.613, +1.000] | +0.292 [+0.101, +0.484] over 6 | +0.551 [+0.091, +1.000] | +0.051 [-0.408, +0.719] over 6 |
| relevance (n 12) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 12 | -0.491 [-0.776, -0.254] | -0.491 [-0.767, -0.273] over 12 | -0.104 [-0.423, +0.315] | -0.104 [-0.427, +0.347] over 12 | +0.089 [-0.235, +0.526] | +0.089 [-0.241, +0.560] over 12 |
| closest_approach | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 5 | +0.441 [-0.100, +1.000] | +0.441 [-0.052, +1.000] over 5 | +0.716 [+0.128, +0.987] | +0.716 [+0.228, +0.970] over 5 | +0.737 [+0.155, +0.986] | +0.737 [+0.226, +0.969] over 5 |
| closest_person | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 10 | +0.587 [+0.350, +0.820] | +0.587 [+0.351, +0.800] over 10 | +0.761 [+0.634, +0.867] | +0.761 [+0.634, +0.864] over 10 | +0.863 [+0.791, +0.947] | +0.863 [+0.794, +0.943] over 10 |
| ego_mem | +0.500 [+0.000, +1.000] | +0.500 [+0.000, +1.000] | +0.000 [+0.000, +0.000] over 6 | +0.455 [+0.007, +1.000] | -0.045 [-0.122, +0.003] over 6 | +0.792 [+0.613, +1.000] | +0.292 [+0.101, +0.484] over 6 | +0.551 [+0.091, +1.000] | +0.051 [-0.408, +0.719] over 6 |
| ego_now | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 34 | +0.159 [-0.201, +0.435] | +0.159 [-0.194, +0.434] over 34 | +0.591 [+0.463, +0.710] | +0.591 [+0.465, +0.705] over 34 | +0.712 [+0.640, +0.776] | +0.712 [+0.639, +0.775] over 34 |
| task_things | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 12 | -0.491 [-0.776, -0.254] | -0.491 [-0.767, -0.273] over 12 | -0.104 [-0.423, +0.315] | -0.104 [-0.427, +0.347] over 12 | +0.089 [-0.235, +0.526] | +0.089 [-0.241, +0.560] over 12 |

| relevance | A | B | B - A | C | C - A | D | D - A | E | E - A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| answerable, critical | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 64 | +0.114 [-0.058, +0.254] | +0.114 [-0.058, +0.254] over 64 | +0.497 [+0.400, +0.601] | +0.497 [+0.400, +0.601] over 64 | +0.623 [+0.529, +0.733] | +0.623 [+0.529, +0.733] over 64 |
| answerable, situational | - | - | - | - | - | - | - | - | - |
| unknown | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] over 3 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] over 3 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] over 3 | +0.333 [-1.000, +1.000] | -0.667 [-2.000, +0.000] over 3 |

- B - A on the critical answerable rows: not shown (lower bound +0.000, over 64 pairs)
- C - A on the critical answerable rows: not shown (lower bound -0.058, over 64 pairs)
- D - A on the critical answerable rows: shown (lower bound +0.400, over 64 pairs)
- E - A on the critical answerable rows: shown (lower bound +0.529, over 64 pairs)

#### Per job

##### apartment_a, helper/1

| family | A | B | B - A | C | C - A | D | D - A | E | E - A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **headline** | +0.067 [+0.020, +0.103] | +0.070 [-0.026, +0.276] | +0.004 [-0.083, +0.220] over 87 | -0.053 [-0.247, +0.147] | -0.120 [-0.308, +0.108] over 87 | +0.398 [+0.309, +0.494] | +0.331 [+0.223, +0.453] over 87 | +0.573 [+0.527, +0.647] | +0.507 [+0.441, +0.601] over 87 |
| `now` | +0.000 [+0.000, +0.000] | +0.141 [+0.034, +0.393] | +0.141 [+0.038, +0.396] over 22 | +0.083 [-0.136, +0.358] | +0.083 [-0.126, +0.358] over 22 | +0.561 [+0.346, +0.791] | +0.561 [+0.335, +0.790] over 22 | +0.661 [+0.565, +0.767] | +0.661 [+0.561, +0.765] over 22 |
| `memory` | +0.133 [+0.040, +0.205] | -0.001 [-0.114, +0.150] | -0.134 [-0.217, +0.065] over 65 | -0.190 [-0.398, +0.065] | -0.323 [-0.543, -0.014] over 65 | +0.235 [+0.090, +0.422] | +0.102 [-0.058, +0.360] over 65 | +0.486 [+0.345, +0.666] | +0.353 [+0.191, +0.568] over 65 |
| clearance, now (n 10) | +0.000 [+0.000, +0.000] | +0.283 [+0.070, +0.786] | +0.283 [+0.082, +0.750] over 10 | +0.266 [+0.029, +0.723] | +0.266 [+0.040, +0.687] over 10 | +0.652 [+0.359, +0.948] | +0.652 [+0.361, +0.941] over 10 | +0.736 [+0.555, +0.902] | +0.736 [+0.563, +0.881] over 10 |
| clearance, memory (n 9) | +0.222 [+0.000, +0.444] | -0.180 [-0.536, +0.324] | -0.402 [-0.650, +0.098] over 9 | -0.443 [-0.801, +0.055] | -0.665 [-1.125, -0.138] over 9 | -0.185 [-0.661, +0.375] | -0.407 [-0.848, +0.077] over 9 | +0.075 [-0.368, +0.575] | -0.147 [-0.556, +0.312] over 9 |
| distance (n 12) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 12 | -0.101 [-0.321, +0.091] | -0.101 [-0.299, +0.091] over 12 | +0.470 [+0.160, +0.654] | +0.470 [+0.164, +0.650] over 12 | +0.586 [+0.480, +0.638] | +0.586 [+0.485, +0.638] over 12 |
| heading (n 5, diagnostic) | +0.000 [+0.000, +0.000] | +0.411 [+0.018, +1.000] | +0.411 [+0.018, +1.000] over 5 | -0.800 [-1.000, -0.500] | -0.800 [-1.000, -0.500] over 5 | +0.281 [-0.167, +1.000] | +0.281 [-0.167, +1.000] over 5 | +0.300 [-0.500, +1.000] | +0.300 [-0.500, +1.000] over 5 |
| memory (n 45) | +0.178 [+0.000, +0.250] | +0.178 [+0.000, +0.250] | +0.000 [+0.000, +0.000] over 45 | -0.208 [-0.336, -0.080] | -0.385 [-0.562, -0.148] over 45 | +0.350 [+0.251, +0.442] | +0.172 [+0.066, +0.343] over 45 | +0.564 [+0.469, +0.686] | +0.387 [+0.234, +0.645] over 45 |
| relevance (n 6) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 6 | +0.080 [-0.466, +0.441] | +0.080 [-0.445, +0.431] over 6 | +0.540 [+0.208, +0.893] | +0.540 [+0.196, +0.899] over 6 | +0.818 [+0.756, +0.923] | +0.818 [+0.761, +0.918] over 6 |
| clear_path | +0.125 [+0.000, +0.308] | +0.076 [-0.192, +0.586] | -0.049 [-0.319, +0.419] over 16 | -0.179 [-0.523, +0.265] | -0.304 [-0.784, +0.163] over 16 | +0.171 [-0.079, +0.467] | +0.046 [-0.332, +0.364] over 16 | +0.339 [+0.106, +0.576] | +0.214 [-0.128, +0.458] over 16 |
| closest_approach | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 3 | +0.517 [+0.420, +0.710] | +0.517 [+0.420, +0.710] over 3 | +0.704 [+0.215, +0.949] | +0.704 [+0.215, +0.949] over 3 | +0.870 [+0.710, +0.950] | +0.870 [+0.710, +0.950] over 3 |
| ego_mem | +0.143 [+0.000, +0.250] | +0.143 [+0.000, +0.250] | +0.000 [+0.000, +0.000] over 28 | -0.399 [-0.554, -0.291] | -0.542 [-0.706, -0.375] over 28 | +0.162 [+0.060, +0.261] | +0.019 [-0.136, +0.162] over 28 | +0.459 [+0.305, +0.636] | +0.316 [+0.093, +0.583] over 28 |
| ego_now | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 12 | -0.101 [-0.321, +0.091] | -0.101 [-0.299, +0.091] over 12 | +0.470 [+0.160, +0.654] | +0.470 [+0.164, +0.650] over 12 | +0.586 [+0.480, +0.638] | +0.586 [+0.485, +0.638] over 12 |
| object_where | +0.235 [+0.000, +0.381] | +0.235 [+0.000, +0.381] | +0.000 [+0.000, +0.000] over 17 | +0.109 [-0.065, +0.444] | -0.127 [-0.436, +0.431] over 17 | +0.660 [+0.489, +0.788] | +0.424 [+0.148, +0.721] over 17 | +0.739 [+0.552, +0.874] | +0.504 [+0.208, +0.824] over 17 |
| open_heading | +0.000 [+0.000, +0.000] | +0.411 [+0.018, +1.000] | +0.411 [+0.018, +1.000] over 5 | -0.800 [-1.000, -0.500] | -0.800 [-1.000, -0.500] over 5 | +0.281 [-0.167, +1.000] | +0.281 [-0.167, +1.000] over 5 | +0.300 [-0.500, +1.000] | +0.300 [-0.500, +1.000] over 5 |
| task_things | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 6 | +0.080 [-0.466, +0.441] | +0.080 [-0.445, +0.431] over 6 | +0.540 [+0.208, +0.893] | +0.540 [+0.196, +0.899] over 6 | +0.818 [+0.756, +0.923] | +0.818 [+0.761, +0.918] over 6 |

| relevance | A | B | B - A | C | C - A | D | D - A | E | E - A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| answerable, critical | +0.000 [+0.000, +0.000] | +0.068 [+0.001, +0.183] | +0.068 [+0.001, +0.183] over 77 | -0.122 [-0.217, +0.041] | -0.122 [-0.217, +0.041] over 77 | +0.349 [+0.267, +0.476] | +0.349 [+0.267, +0.476] over 77 | +0.557 [+0.471, +0.686] | +0.557 [+0.471, +0.686] over 77 |
| answerable, situational | - | - | - | - | - | - | - | - | - |
| unknown | +1.000 [+1.000, +1.000] | +0.600 [-0.533, +1.000] | -0.400 [-1.143, +0.000] over 10 | -0.600 [-1.000, -0.111] | -1.600 [-2.000, -1.143] over 10 | +0.400 [+0.000, +1.000] | -0.600 [-1.000, +0.000] over 10 | +0.400 [+0.000, +1.000] | -0.600 [-1.000, +0.000] over 10 |

- B - A on the critical answerable rows: shown (lower bound +0.001, over 77 pairs)
- C - A on the critical answerable rows: not shown (lower bound -0.217, over 77 pairs)
- D - A on the critical answerable rows: shown (lower bound +0.267, over 77 pairs)
- E - A on the critical answerable rows: shown (lower bound +0.471, over 77 pairs)

| length | A | B | C | D | E |
| --- | --- | --- | --- | --- | --- |
| short | +0.167 [+0.000, +0.308] | +0.192 [+0.095, +0.282] | -0.166 [-0.366, +0.102] | +0.367 [+0.220, +0.574] | +0.519 [+0.372, +0.742] |
| medium | +0.078 [+0.000, +0.125] | +0.086 [+0.036, +0.106] | -0.184 [-0.280, -0.037] | +0.347 [+0.243, +0.465] | +0.553 [+0.490, +0.714] |

##### apartment_a, vacuum/2

| family | A | B | B - A | C | C - A | D | D - A | E | E - A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **headline** | +0.037 [+0.010, +0.076] | +0.527 [+0.388, +0.640] | +0.490 [+0.349, +0.604] over 157 | -0.206 [-0.525, +0.060] | -0.243 [-0.561, +0.025] over 157 | +0.430 [+0.109, +0.647] | +0.393 [+0.066, +0.623] over 157 | +0.602 [+0.237, +0.810] | +0.565 [+0.191, +0.789] over 157 |
| `now` | +0.000 [+0.000, +0.000] | +0.800 [+0.589, +0.970] | +0.800 [+0.589, +0.958] over 19 | -0.201 [-0.881, +0.386] | -0.201 [-0.856, +0.349] over 19 | +0.473 [-0.169, +0.851] | +0.473 [-0.140, +0.845] over 19 | +0.620 [-0.056, +0.947] | +0.620 [-0.064, +0.947] over 19 |
| `memory` | +0.074 [+0.021, +0.152] | +0.253 [+0.140, +0.395] | +0.179 [+0.089, +0.267] over 138 | -0.212 [-0.397, -0.060] | -0.286 [-0.538, -0.102] over 138 | +0.387 [+0.272, +0.524] | +0.313 [+0.162, +0.484] over 138 | +0.585 [+0.437, +0.730] | +0.511 [+0.300, +0.688] over 138 |
| clearance, now (n 12) | +0.000 [+0.000, +0.000] | +0.800 [+0.589, +0.970] | +0.800 [+0.589, +0.958] over 12 | -0.201 [-0.881, +0.386] | -0.201 [-0.856, +0.349] over 12 | +0.473 [-0.169, +0.851] | +0.473 [-0.140, +0.845] over 12 | +0.620 [-0.056, +0.947] | +0.620 [-0.064, +0.947] over 12 |
| clearance, memory (n 75) | +0.013 [+0.000, +0.041] | +0.371 [+0.181, +0.556] | +0.358 [+0.177, +0.535] over 75 | -0.317 [-0.426, -0.211] | -0.330 [-0.451, -0.214] over 75 | +0.380 [+0.174, +0.628] | +0.367 [+0.145, +0.627] over 75 | +0.625 [+0.489, +0.777] | +0.611 [+0.472, +0.768] over 75 |
| heading, now (n 7, diagnostic) | +0.000 [+0.000, +0.000] | +0.243 [-0.167, +1.000] | +0.243 [-0.167, +1.000] over 7 | -0.043 [-0.833, +1.000] | -0.043 [-0.833, +1.000] over 7 | -0.071 [-0.333, +1.000] | -0.071 [-0.333, +1.000] over 7 | +0.000 [-0.167, +1.000] | +0.000 [-0.167, +1.000] over 7 |
| heading, memory (n 26, diagnostic) | +0.000 [+0.000, +0.000] | -0.124 [-0.370, +0.117] | -0.124 [-0.370, +0.093] over 26 | -0.705 [-0.886, -0.512] | -0.705 [-0.881, -0.514] over 26 | -0.225 [-0.420, +0.107] | -0.225 [-0.412, +0.100] over 26 | +0.019 [-0.290, +0.444] | +0.019 [-0.274, +0.397] over 26 |
| memory (n 37) | +0.135 [+0.042, +0.280] | +0.135 [+0.042, +0.280] | +0.000 [+0.000, +0.000] over 37 | -0.106 [-0.425, +0.162] | -0.242 [-0.677, +0.097] over 37 | +0.394 [+0.284, +0.502] | +0.259 [+0.078, +0.410] over 37 | +0.545 [+0.306, +0.726] | +0.410 [+0.060, +0.658] over 37 |
| clear_path | +0.018 [+0.000, +0.048] | +0.657 [+0.390, +0.861] | +0.639 [+0.383, +0.853] over 57 | -0.523 [-0.785, -0.310] | -0.541 [-0.786, -0.335] over 57 | +0.377 [+0.200, +0.562] | +0.360 [+0.170, +0.557] over 57 | +0.754 [+0.471, +0.916] | +0.736 [+0.452, +0.914] over 57 |
| ego_mem | +0.138 [+0.042, +0.286] | +0.138 [+0.042, +0.286] | +0.000 [+0.000, +0.000] over 29 | -0.280 [-0.510, -0.047] | -0.418 [-0.758, -0.119] over 29 | +0.267 [+0.099, +0.402] | +0.129 [-0.143, +0.302] over 29 | +0.513 [+0.325, +0.667] | +0.375 [+0.072, +0.587] over 29 |
| object_where | +0.125 [+0.000, +0.500] | +0.125 [+0.000, +0.500] | +0.000 [+0.000, +0.000] over 8 | +0.523 [-0.333, +0.919] | +0.398 [-0.667, +0.910] over 8 | +0.854 [+0.759, +0.980] | +0.729 [+0.433, +0.923] over 8 | +0.663 [-0.056, +0.943] | +0.538 [-0.550, +0.931] over 8 |
| open_heading | +0.000 [+0.000, +0.000] | -0.047 [-0.251, +0.194] | -0.047 [-0.243, +0.188] over 33 | -0.565 [-0.837, -0.218] | -0.565 [-0.828, -0.233] over 33 | -0.192 [-0.388, +0.089] | -0.192 [-0.391, +0.069] over 33 | +0.015 [-0.220, +0.317] | +0.015 [-0.202, +0.313] over 33 |
| tangle_ahead | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 30 | +0.122 [-0.157, +0.390] | +0.122 [-0.159, +0.395] over 30 | +0.422 [+0.147, +0.794] | +0.422 [+0.130, +0.791] over 30 | +0.377 [+0.123, +0.613] | +0.377 [+0.144, +0.614] over 30 |

| relevance | A | B | B - A | C | C - A | D | D - A | E | E - A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| answerable, critical | +0.000 [+0.000, +0.000] | +0.244 [+0.078, +0.439] | +0.244 [+0.078, +0.439] over 151 | -0.310 [-0.395, -0.235] | -0.310 [-0.395, -0.235] over 151 | +0.254 [+0.120, +0.446] | +0.254 [+0.120, +0.446] over 151 | +0.483 [+0.354, +0.655] | +0.483 [+0.354, +0.655] over 151 |
| answerable, situational | - | - | - | - | - | - | - | - | - |
| unknown | +1.000 [+1.000, +1.000] | +0.667 [+0.385, +1.000] | -0.333 [-0.600, +0.000] over 6 | -0.333 [-1.000, +0.600] | -1.333 [-2.000, -0.500] over 6 | +0.667 [-0.500, +1.000] | -0.333 [-1.500, +0.000] over 6 | +0.333 [-0.500, +1.000] | -0.667 [-1.500, +0.000] over 6 |

- B - A on the critical answerable rows: shown (lower bound +0.078, over 151 pairs)
- C - A on the critical answerable rows: not shown (lower bound -0.395, over 151 pairs)
- D - A on the critical answerable rows: shown (lower bound +0.120, over 151 pairs)
- E - A on the critical answerable rows: shown (lower bound +0.354, over 151 pairs)

| length | A | B | C | D | E |
| --- | --- | --- | --- | --- | --- |
| short | +0.070 [+0.000, +0.130] | +0.378 [+0.003, +0.558] | -0.310 [-0.453, -0.212] | +0.232 [-0.088, +0.474] | +0.361 [+0.016, +0.614] |
| medium | +0.020 [+0.000, +0.051] | +0.194 [+0.033, +0.390] | -0.311 [-0.455, -0.219] | +0.292 [+0.151, +0.528] | +0.544 [+0.398, +0.744] |

##### downtown_a, drone/1

| family | A | B | B - A | C | C - A | D | D - A | E | E - A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **headline** | +0.132 [+0.078, +0.219] | +0.647 [+0.543, +0.763] | +0.516 [+0.375, +0.635] over 68 | +0.008 [-0.132, +0.125] | -0.124 [-0.334, +0.002] over 68 | +0.593 [+0.433, +0.742] | +0.461 [+0.242, +0.624] over 68 | +0.768 [+0.630, +0.849] | +0.636 [+0.437, +0.735] over 68 |
| `now` | +0.000 [+0.000, +0.000] | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] over 22 | +0.080 [-0.075, +0.231] | +0.080 [-0.075, +0.231] over 22 | +0.499 [+0.246, +0.751] | +0.499 [+0.246, +0.751] over 22 | +0.769 [+0.631, +0.901] | +0.769 [+0.631, +0.901] over 22 |
| `memory` | +0.263 [+0.156, +0.438] | +0.295 [+0.085, +0.525] | +0.031 [-0.250, +0.269] over 46 | -0.064 [-0.230, +0.096] | -0.328 [-0.650, -0.185] over 46 | +0.687 [+0.439, +0.799] | +0.423 [+0.038, +0.618] over 46 | +0.766 [+0.509, +0.876] | +0.503 [+0.110, +0.679] over 46 |
| clearance, now (n 22) | +0.000 [+0.000, +0.000] | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] over 22 | +0.080 [-0.075, +0.231] | +0.080 [-0.075, +0.231] over 22 | +0.499 [+0.246, +0.751] | +0.499 [+0.246, +0.751] over 22 | +0.769 [+0.631, +0.901] | +0.769 [+0.631, +0.901] over 22 |
| clearance, memory (n 32) | +0.312 [+0.154, +0.500] | +0.375 [+0.000, +0.692] | +0.062 [-0.500, +0.539] over 32 | +0.105 [-0.139, +0.428] | -0.208 [-0.411, +0.033] over 32 | +0.808 [+0.682, +0.924] | +0.496 [+0.269, +0.712] over 32 | +0.859 [+0.814, +0.914] | +0.547 [+0.366, +0.728] over 32 |
| memory (n 14) | +0.214 [+0.067, +0.501] | +0.214 [+0.067, +0.501] | +0.000 [+0.000, +0.000] over 14 | -0.233 [-0.635, -0.086] | -0.447 [-1.095, -0.144] over 14 | +0.565 [+0.145, +0.745] | +0.351 [-0.304, +0.666] over 14 | +0.673 [+0.187, +0.843] | +0.459 [-0.220, +0.734] over 14 |
| clear_path | +0.185 [+0.091, +0.270] | +0.630 [+0.460, +0.818] | +0.444 [+0.190, +0.727] over 54 | +0.095 [-0.082, +0.273] | -0.091 [-0.281, +0.082] over 54 | +0.682 [+0.554, +0.824] | +0.497 [+0.357, +0.670] over 54 | +0.823 [+0.761, +0.879] | +0.637 [+0.557, +0.739] over 54 |
| ego_mem | +0.214 [+0.067, +0.501] | +0.214 [+0.067, +0.501] | +0.000 [+0.000, +0.000] over 14 | -0.233 [-0.635, -0.086] | -0.447 [-1.095, -0.144] over 14 | +0.565 [+0.145, +0.745] | +0.351 [-0.304, +0.666] over 14 | +0.673 [+0.187, +0.843] | +0.459 [-0.220, +0.734] over 14 |

| relevance | A | B | B - A | C | C - A | D | D - A | E | E - A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| answerable, critical | +0.000 [+0.000, +0.000] | +0.800 [+0.655, +0.947] | +0.800 [+0.655, +0.947] over 55 | -0.021 [-0.126, +0.080] | -0.021 [-0.126, +0.080] over 55 | +0.614 [+0.468, +0.733] | +0.614 [+0.468, +0.733] over 55 | +0.779 [+0.715, +0.838] | +0.779 [+0.715, +0.838] over 55 |
| answerable, situational | - | - | - | - | - | - | - | - | - |
| unknown | +1.000 [+1.000, +1.000] | -0.538 [-0.875, +0.000] | -1.538 [-1.875, -1.000] over 13 | +0.231 [-0.286, +0.667] | -0.769 [-1.273, -0.308] over 13 | +0.846 [+0.500, +1.000] | -0.154 [-0.545, +0.000] over 13 | +0.846 [+0.500, +1.000] | -0.154 [-0.545, +0.000] over 13 |

- B - A on the critical answerable rows: shown (lower bound +0.655, over 55 pairs)
- C - A on the critical answerable rows: not shown (lower bound -0.126, over 55 pairs)
- D - A on the critical answerable rows: shown (lower bound +0.468, over 55 pairs)
- E - A on the critical answerable rows: shown (lower bound +0.715, over 55 pairs)

| length | A | B | C | D | E |
| --- | --- | --- | --- | --- | --- |
| short | +0.231 [+0.000, +0.556] | +0.385 [-0.133, +0.727] | -0.376 [-0.853, +0.128] | +0.744 [+0.667, +0.830] | +0.708 [+0.666, +0.803] |
| medium | +0.182 [+0.109, +0.264] | +0.582 [+0.367, +0.766] | +0.122 [+0.032, +0.248] | +0.638 [+0.517, +0.770] | +0.812 [+0.736, +0.887] |

##### warehouse_a, amr/1

| family | A | B | B - A | C | C - A | D | D - A | E | E - A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **headline** | +0.083 [+0.000, +0.167] | +0.083 [+0.000, +0.167] | +0.000 [+0.000, +0.000] over 67 | +0.308 [+0.137, +0.459] | +0.225 [+0.086, +0.347] over 67 | +0.554 [+0.431, +0.646] | +0.471 [+0.312, +0.601] over 67 | +0.598 [+0.448, +0.777] | +0.515 [+0.340, +0.723] over 67 |
| `now` | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 38 | +0.480 [+0.274, +0.663] | +0.480 [+0.275, +0.657] over 38 | +0.618 [+0.367, +0.785] | +0.618 [+0.383, +0.780] over 38 | +0.692 [+0.464, +0.841] | +0.692 [+0.460, +0.840] over 38 |
| `memory` | +0.167 [+0.000, +0.333] | +0.167 [+0.000, +0.333] | +0.000 [+0.000, +0.000] over 29 | +0.136 [-0.120, +0.402] | -0.031 [-0.184, +0.105] over 29 | +0.491 [+0.392, +0.605] | +0.324 [+0.216, +0.499] over 29 | +0.505 [+0.286, +0.773] | +0.338 [+0.124, +0.656] over 29 |
| clearance, now (n 4) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 4 | +0.802 [+0.574, +1.000] | +0.802 [+0.603, +1.000] over 4 | +0.645 [+0.128, +0.978] | +0.645 [+0.228, +0.955] over 4 | +0.672 [+0.155, +0.976] | +0.672 [+0.226, +0.952] over 4 |
| clearance, memory (n 11) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 11 | +0.443 [+0.029, +0.810] | +0.443 [+0.052, +0.798] over 11 | +0.783 [+0.641, +0.893] | +0.783 [+0.644, +0.892] over 11 | +0.875 [+0.797, +0.954] | +0.875 [+0.797, +0.953] over 11 |
| distance (n 34) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 34 | +0.159 [-0.201, +0.435] | +0.159 [-0.194, +0.434] over 34 | +0.591 [+0.463, +0.710] | +0.591 [+0.465, +0.705] over 34 | +0.712 [+0.640, +0.776] | +0.712 [+0.639, +0.775] over 34 |
| memory (n 6) | +0.500 [+0.000, +1.000] | +0.500 [+0.000, +1.000] | +0.000 [+0.000, +0.000] over 6 | +0.455 [+0.007, +1.000] | -0.045 [-0.122, +0.003] over 6 | +0.792 [+0.613, +1.000] | +0.292 [+0.101, +0.484] over 6 | +0.551 [+0.091, +1.000] | +0.051 [-0.408, +0.719] over 6 |
| relevance (n 12) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 12 | -0.491 [-0.776, -0.254] | -0.491 [-0.767, -0.273] over 12 | -0.104 [-0.423, +0.315] | -0.104 [-0.427, +0.347] over 12 | +0.089 [-0.235, +0.526] | +0.089 [-0.241, +0.560] over 12 |
| closest_approach | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 5 | +0.441 [-0.100, +1.000] | +0.441 [-0.052, +1.000] over 5 | +0.716 [+0.128, +0.987] | +0.716 [+0.228, +0.970] over 5 | +0.737 [+0.155, +0.986] | +0.737 [+0.226, +0.969] over 5 |
| closest_person | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 10 | +0.587 [+0.350, +0.820] | +0.587 [+0.351, +0.800] over 10 | +0.761 [+0.634, +0.867] | +0.761 [+0.634, +0.864] over 10 | +0.863 [+0.791, +0.947] | +0.863 [+0.794, +0.943] over 10 |
| ego_mem | +0.500 [+0.000, +1.000] | +0.500 [+0.000, +1.000] | +0.000 [+0.000, +0.000] over 6 | +0.455 [+0.007, +1.000] | -0.045 [-0.122, +0.003] over 6 | +0.792 [+0.613, +1.000] | +0.292 [+0.101, +0.484] over 6 | +0.551 [+0.091, +1.000] | +0.051 [-0.408, +0.719] over 6 |
| ego_now | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 34 | +0.159 [-0.201, +0.435] | +0.159 [-0.194, +0.434] over 34 | +0.591 [+0.463, +0.710] | +0.591 [+0.465, +0.705] over 34 | +0.712 [+0.640, +0.776] | +0.712 [+0.639, +0.775] over 34 |
| task_things | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 12 | -0.491 [-0.776, -0.254] | -0.491 [-0.767, -0.273] over 12 | -0.104 [-0.423, +0.315] | -0.104 [-0.427, +0.347] over 12 | +0.089 [-0.235, +0.526] | +0.089 [-0.241, +0.560] over 12 |

| relevance | A | B | B - A | C | C - A | D | D - A | E | E - A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| answerable, critical | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] over 64 | +0.114 [-0.058, +0.254] | +0.114 [-0.058, +0.254] over 64 | +0.497 [+0.400, +0.601] | +0.497 [+0.400, +0.601] over 64 | +0.623 [+0.529, +0.733] | +0.623 [+0.529, +0.733] over 64 |
| answerable, situational | - | - | - | - | - | - | - | - | - |
| unknown | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] over 3 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] over 3 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] over 3 | +0.333 [-1.000, +1.000] | -0.667 [-2.000, +0.000] over 3 |

- B - A on the critical answerable rows: not shown (lower bound +0.000, over 64 pairs)
- C - A on the critical answerable rows: not shown (lower bound -0.058, over 64 pairs)
- D - A on the critical answerable rows: shown (lower bound +0.400, over 64 pairs)
- E - A on the critical answerable rows: shown (lower bound +0.529, over 64 pairs)

| length | A | B | C | D | E |
| --- | --- | --- | --- | --- | --- |
| short | +0.030 [+0.000, +0.086] | +0.030 [+0.000, +0.086] | +0.114 [-0.131, +0.340] | +0.524 [+0.448, +0.623] | +0.643 [+0.521, +0.775] |
| medium | +0.059 [+0.000, +0.106] | +0.059 [+0.000, +0.106] | +0.193 [+0.074, +0.316] | +0.514 [+0.336, +0.709] | +0.579 [+0.436, +0.793] |

**By the age of the evidence, and the catastrophes.** Each run's `results.md` gives the reported score by the age band of
its evidence ("Per age of the evidence"), overall and for each job, and the catastrophe rate for each job
("Catastrophes").

The reported score by the age of the evidence, over every place (the `unknown` rows have no band):

| System | `now` (101) | `recent` (64) | `task` (154) | `earlier` (28) |
| --- | --- | --- | --- | --- |
| `codex:gpt-6-astra` (xhigh) | +0.65 | +0.64 | +0.49 | +0.62 |
| `codex:gpt-6.1-sol` (medium) | +0.51 | +0.37 | +0.29 | +0.39 |
| `codex:gpt-6-luna` (xhigh) | +0.09 | -0.54 | -0.16 | 0.00 |

The catastrophe rate, the share of the harm rows answered that are catastrophes, from each run's `results.md`:

| System | All jobs | `vacuum/2` | `helper/1` | `amr/1` | `drone/1` |
| --- | --- | --- | --- | --- | --- |
| `codex:gpt-6-astra` (xhigh) | 3 of 155 | 2 of 83 | 1 of 17 | 0 of 14 | 0 of 41 |
| `codex:gpt-6.1-sol` (medium) | 4 of 132 | 3 of 70 | 1 of 12 | 0 of 14 | 0 of 36 |
| `codex:gpt-6-luna` (xhigh) | 13 of 130 | 12 of 78 | 1 of 18 | 0 of 15 | 0 of 19 |

The score falls with the age of the evidence: sol goes from +0.51 on what the latest frame shows to +0.29 on what it saw
earlier in the task, and astra holds +0.64 over the last 10 s and falls to +0.49. Luna is near zero on the frame it holds
and below the floor on what it saw before. The `earlier` band holds 28 questions, too few to read a slope from. Every
catastrophe is a corridor in the apartment that the vacuum or the helper would have driven into: none is a person or a
cord, and the warehouse and downtown have none.

**What the models got right and wrong.**

From the comparison above and each run's family rows:
- **The corridors.** The drone's are the easiest questions of the release: only building walls bound them, against a
  clearance of 2.4 m, and astra lands within what the job needs on 21 of the 22 answerable ones that need memory (+0.86 with
  the `unknown` rows) and on 19 of the 22 in view (+0.77); sol scores +0.81 and +0.50. The vacuum's, read from a camera 8 cm up against 10 cm, split the models
  most: astra +0.79 on the remembered ones, sol +0.37, luna -0.58 with 10 catastrophes. The helper's remembered
  corridors (behind it, or after turning, against 15 cm) are the hardest: sol -0.18, astra +0.08, on 9 questions.
- **`closest_approach`** has few rows on these cases (5 on the AMR, 3 on the helper), nearly all a person standing; sol and
  astra score +0.65 to +0.87 on them.
- **`tangle_ahead`** does not separate astra from sol (+0.36 and +0.40 on the 29 that need memory); luna scores +0.11. A cord seen
  from 8 cm up is hard for every model.
- **What it saw before** (`ego_mem`) separates the models in the apartment: astra +0.46 and +0.51 on the helper and the
  vacuum, sol +0.16 and +0.27, luna -0.40 and -0.28. Where on the map it last saw a thing (`object_where`) is easier: sol
  +0.66 and +0.85, astra +0.74 and +0.66.
- **The AMR's `task_things`**, the list of things that stood within 2 m of its drive, is the weakest counted cell of the
  release: sol -0.10 and astra +0.09 on 12 questions.
- **The map reader** is high only where the counted questions ask about fixed structure: downtown's corridors (1.00 in
  view, +0.38 remembered) and, less, the apartment's (its clearance cells +0.56 and +0.31, which hold its zeros on the people and
  the cords as well). In the warehouse it scores the floor.

**How the cases ended.** The comparison lists the cases each run finished. A case ends early when the model stops
pulling (`stalled`), a message exceeds its timeout (`timeout`), or the window is full (`context`); the failed questions
count in the score ([codex.md](../codex.md), "The leak scan, failures and exits"). Codex's own compaction, in the middle of
one long turn, had not been exercised before these rows.

Every one of the 42 cases ended complete for all three models, with no premature pull and no resume. The median of the
cases' per-frame latency was 2.0 s for sol, 2.3 s for astra and 1.6 s for luna, with a median p90 of 2.8, 3.4 and 2.2 s.
No case of the 42 was compacted: none is longer than 294 frames. Beyond the 42, Codex's own session logs show it compacted a
luna thread past frame 600 and an astra thread past frame 440, each at about 243,000 input tokens; those cases were unfinished when the runs
stopped and are not in the rows. Compaction is the agent's own memory management, and the benchmark reports it rather
than correcting it.

**Cost and tokens.** The cost is notional, at the price table, and a lower bound: Codex reports no cache writes, and a
turn the harness kills prints no usage. The runs spent no dollars, only the plan's quota. One thread holds a case, and each
pull re-reads what the thread holds, so a case reads far more input than a case of 0.9.0 did. The reference agents' means
per case, by length:
`all_unknown`, a case: no tokens were reported; `map_reader`, a case: no tokens were reported; `codex:gpt-6-luna` (xhigh), a case: short 3,448,466 tokens in (3,312,140 cached), 13,156 out, 386.0 s, $0.0533; medium 12,727,382 tokens in (12,449,000 cached), 21,877 out, 567.1 s, $0.1633; all 8,087,924 tokens in (7,880,570 cached), 17,517 out, 476.5 s, $0.1083; `codex:gpt-6.1-sol` (medium), a case: short 3,064,011 tokens in (2,972,459 cached), 2,457 out, 223.7 s, $0.5049; medium 11,077,868 tokens in (10,888,771 cached), 4,568 out, 516.9 s, $1.5128; all 7,070,940 tokens in (6,930,615 cached), 3,513 out, 370.3 s, $1.0088; `codex:gpt-6-astra` (xhigh), a case: short 3,179,471 tokens in (3,082,478 cached), 5,250 out, 480.1 s, $n/a; medium 11,502,579 tokens in (11,332,309 cached), 8,858 out, 730.9 s, $n/a; all 7,341,025 tokens in (7,207,394 cached), 7,054 out, 605.5 s, $n/a.

A case read about five times 0.9.0's input tokens: sol 3.1 million on a short case and 11.1 million on a medium one,
against 0.6 and 2.3 million in 0.9.0's rows, above the projection of three to four times. Nearly all of it is cached. The
wall time of a medium case halved (sol 517 s against 1,129 s), because a pull inside one turn costs far less than a
new exec; a short case's moved less (sol 224 s against 288 s, luna and astra up by a fifth to a third). The quota is the limit: in 73 minutes, 18 execs at once (6 a model) took a ChatGPT
Pro plan's weekly use from 53% to 96%, most of it astra's at xhigh, and that is why the rows stop at 42 cases. No exec
hit a rate limit.

## 5. Did it work

The goal was realistic difficulty without tricks: questions that are the decisions a robot's job depends on, credit as
tight as the job needs and no tighter, full credit that is hard but always reachable, and no question a careful reader of
the frames could not answer. Three targets were set before any row of 0.9.0 or of 0.10.0 was read and have not changed:
- **A strong model in a realistic band.** Sol at about 0.25 to 0.5 on realistically hard, task-based questions. A band to
  check realism by, never a target to tune to.
- **The strongest model does not saturate.** Astra, the strongest model run, stays well below the ceiling, and no cell is
  at 1.00.
- **Full credit is rare.** A perfect answer takes a camera read to a pixel, so only a small share of answers earn it. The
  design expected 15 to 25% of astra's answerable answers and 10 to 15% of sol's.

| Layout | System | Headline (95% interval) | Target |
| --- | --- | --- | --- |
| `apartment_a` | `codex:gpt-6.1-sol` (medium) | +0.461 [+0.325, +0.595] | about 0.25 to 0.5 |
| `apartment_a` | `codex:gpt-6-astra` (xhigh) | +0.638 [+0.524, +0.717] | well below 1.00 |
| `warehouse_a` | `codex:gpt-6.1-sol` (medium) | +0.554 [+0.431, +0.646] | about 0.25 to 0.5 |
| `warehouse_a` | `codex:gpt-6-astra` (xhigh) | +0.598 [+0.448, +0.777] | well below 1.00 |
| `downtown_a` | `codex:gpt-6.1-sol` (medium) | +0.593 [+0.433, +0.742] | about 0.25 to 0.5 |
| `downtown_a` | `codex:gpt-6-astra` (xhigh) | +0.768 [+0.630, +0.849] | well below 1.00 |

- **Sol's band: met in the apartment, a little above it elsewhere.** Sol lands at +0.46 in the apartment, inside the
  band, and at +0.55 in the warehouse and +0.59 downtown, a little above it; every interval overlaps the band.
- **Astra does not saturate: met.** It scores +0.60 to +0.77, and none of its counted cells reaches 1.00; its highest are
  the warehouse's remembered clearance (+0.88, 11 questions) and downtown's (+0.86).
- **Full credit is rare: not met.** Astra earned full credit on 33% of the answerable questions, sol on 21% and luna on
  9%, above the 15 to 25% and 10 to 15% expected. Most of the excess is on the vacuum's and the drone's corridors, where
  one pixel at a camera 8 cm up or down a long street is a wide margin: astra earned full credit on 68% of the vacuum's
  answerable corridors and on 50% of the drone's.

**What changed from 0.9.0.** 0.9.0 asked at the end of a batch of up to 8 frames, one question in about 30 frames. 0.10.0
asks at any frame, harm first, one in every 9 to 25, and holds the agent to the one frame it has, so a corridor that
needed a frame of the batch is `memory` now. The questions reach what the camera cannot see now (the corridor behind, the
cord beside the robot that it passed frames ago) and what is dangerous for the job (the person nearest, the cord in the
brush's path).

Against 0.9.0's rows, which asked other questions over more cases, only the direction can be read. Sol's headlines
moved little (apartment +0.46 against +0.44, warehouse +0.55 against +0.49, downtown +0.59 against +0.56), astra's
moved down in the warehouse (+0.60 against +0.70), and the order of the models held on every place. The models differ
most on what they saw before and on the vacuum's corridors (above). Every catastrophe is a corridor in the apartment;
the new harm families, `closest_approach` and `tangle_ahead`, drew none. The rows were run after the last change to the questions, the credit and the prompt: no C,
R, rule, share or tolerance moved after they were read.

**The review.** An independent review of the branch read the compiled data, the code and the gates (Section 6). It was a
read of the instrument and not of a model's answers.

No answer-by-answer review of these rows' misses was made, as 0.9.0's calibration had (it found no wrong gold). The
misses are not yet split between the instrument and the models.

## 6. Validity

Every gold is computed by code from the world timeline, the visibility log and the scorer's depth and masks, and names the
frames it rests on. Nothing the robot did not see changes a gold. No person or model writes or judges one.

**The gates.** `stbench check` runs scripted systems over every case on the oracle channel, which sends each recognisable
detection's floor hull: what a perfect perceiver would know, and nothing about what is asked. Since the memory table
reproduces the gold from a channel a system could receive, the ceiling is a real test. The thresholds were fixed before
data and read the gate score, never the catastrophe term (Section 2).

| Gate | Rule |
| --- | --- |
| ceiling | the oracle scores at least 0.95 |
| knowable | the oracle less `memory_table`: upper 95% bound at most 0.02 |
| memory needed | on a memory family, the current view, with or without learned priors, scores at most `all_unknown` + 0.05 |
| shortcuts | each scores at most `all_unknown` + 0.10: answers learned from other shifts, the robot's own path log, another case's stream, the questions' wording, a form adversary |
| separation | `memory_table` beats the best shortcut (for `ego_now`, also the kind's typical range) by at least 0.40, lower 95% bound above 0.30 |
| balance | no constant answer, learned on other shifts, beats `all_unknown` |

On the two structure families the map reader and `other_case` are references and are left out of the shortcut and
separation gates, and `path_log` answers from the robot's own log alone: its detour and its longest swept run stand only
where the lower end of the 95% interval of their gain over `unknown`, over other scripts' records, is above 0. On their
`memory` rows the current view is gated without a pose key. The evidence verdict, which ships, is stricter than a point
reading: a shortcut fails when its gain is over the margin with a 95% interval above 0, in the whole group, in any length
or in either slice ([eval.md](../eval.md); [methodology.md](../methodology.md), rule 5). The check of this release:

| Layout | Job | Family | Scripts | n | Unknown share | ceiling | knowable | memory_needed | shortcuts | separation | time_needed | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `apartment_a` | `helper/1` | `clear_path` | 16 | 96 | 0.125 | ok +1.000 vs 0.95 | n/a | ok +0.000 vs 0.05 | ok +0.000 vs 0.1 | ok +0.875 vs 0.4 [+0.772, +0.940] | n/a | pass |
| `apartment_a` | `helper/1` | `closest_approach` | 10 | 22 | 0.000 | ok +1.000 vs 0.95 | ok +0.000 vs 0.02 [+0.000, +0.000] | n/a | ok +0.000 vs 0.1 | ok +1.000 vs 0.4 [+1.000, +1.000] | n/a | pass |
| `apartment_a` | `helper/1` | `ego_mem` | 16 | 168 | 0.185 | ok +1.000 vs 0.95 | ok +0.000 vs 0.02 [+0.000, +0.000] | ok +0.000 vs 0.05 | ok +0.000 vs 0.1 | ok +0.815 vs 0.4 [+0.752, +0.877] | n/a | pass |
| `apartment_a` | `helper/1` | `ego_now` | 14 | 52 | 0.000 | ok +1.000 vs 0.95 | ok +0.000 vs 0.02 [+0.000, +0.000] | n/a | ok +0.001 vs 0.1 | ok +0.999 vs 0.4 [+0.996, +1.000] | n/a | pass |
| `apartment_a` | `helper/1` | `object_where` | 16 | 74 | 0.257 | ok +1.000 vs 0.95 | ok +0.000 vs 0.02 [+0.000, +0.000] | ok +0.000 vs 0.05 | ok +0.014 vs 0.1 | ok +0.743 vs 0.4 [+0.618, +0.853] | n/a | pass |
| `apartment_a` | `helper/1` | `open_heading` | 14 | 54 | 0.000 | ok +1.000 vs 0.95 | n/a | ok +0.000 vs 0.05 | ok +0.000 vs 0.1 | ok +1.000 vs 0.4 [+1.000, +1.000] | n/a | pass |
| `apartment_a` | `helper/1` | `task_things` | 13 | 47 | 0.000 | ok +1.000 vs 0.95 | ok +0.000 vs 0.02 [+0.000, +0.000] | ok +0.030 vs 0.05 | ok +0.030 vs 0.1 | ok +1.000 vs 0.4 [+1.000, +1.000] | n/a | pass |
| `apartment_a` | `vacuum/2` | `clear_path` | 14 | 315 | 0.016 | ok +1.000 vs 0.95 | n/a | ok +0.013 vs 0.05 | ok +0.000 vs 0.1 | ok +0.984 vs 0.4 [+0.962, +0.997] | n/a | pass |
| `apartment_a` | `vacuum/2` | `ego_mem` | 14 | 83 | 0.205 | ok +1.000 vs 0.95 | ok +0.000 vs 0.02 [+0.000, +0.000] | ok +0.000 vs 0.05 | ok +0.000 vs 0.1 | ok +0.795 vs 0.4 [+0.717, +0.873] | n/a | pass |
| `apartment_a` | `vacuum/2` | `object_where` | 10 | 24 | 0.167 | ok +1.000 vs 0.95 | ok +0.000 vs 0.02 [+0.000, +0.000] | ok +0.000 vs 0.05 | ok +0.000 vs 0.1 | ok +0.833 vs 0.4 [+0.692, +0.950] | n/a | pass |
| `apartment_a` | `vacuum/2` | `open_heading` | 14 | 134 | 0.000 | ok +1.000 vs 0.95 | n/a | ok +0.000 vs 0.05 | ok +0.000 vs 0.1 | ok +1.000 vs 0.4 [+1.000, +1.000] | n/a | pass |
| `apartment_a` | `vacuum/2` | `tangle_ahead` | 14 | 133 | 0.000 | ok +1.000 vs 0.95 | ok +0.000 vs 0.02 [+0.000, +0.000] | ok +0.033 vs 0.05 | ok +0.000 vs 0.1 | ok +1.000 vs 0.4 [+1.000, +1.000] | n/a | pass |
| `downtown_a` | `drone/1` | `clear_path` | 16 | 196 | 0.092 | ok +1.000 vs 0.95 | n/a | ok +0.000 vs 0.05 | ok +0.000 vs 0.1 | ok +0.908 vs 0.4 [+0.865, +0.948] | n/a | pass |
| `downtown_a` | `drone/1` | `ego_mem` | 14 | 68 | 0.206 | ok +1.000 vs 0.95 | ok +0.000 vs 0.02 [+0.000, +0.000] | ok +0.000 vs 0.05 | ok +0.000 vs 0.1 | ok +0.794 vs 0.4 [+0.695, +0.879] | n/a | pass |
| `warehouse_a` | `amr/1` | `closest_approach` | 14 | 24 | 0.000 | ok +1.000 vs 0.95 | ok +0.000 vs 0.02 [+0.000, +0.000] | ok +0.000 vs 0.05 | ok +0.000 vs 0.1 | ok +1.000 vs 0.4 [+1.000, +1.000] | n/a | pass |
| `warehouse_a` | `amr/1` | `closest_person` | 16 | 34 | 0.000 | ok +1.000 vs 0.95 | ok +0.000 vs 0.02 [+0.000, +0.000] | ok +0.000 vs 0.05 | ok +0.000 vs 0.1 | ok +1.000 vs 0.4 [+1.000, +1.000] | n/a | pass |
| `warehouse_a` | `amr/1` | `ego_mem` | 11 | 27 | 0.370 | ok +1.000 vs 0.95 | ok +0.000 vs 0.02 [+0.000, +0.000] | ok +0.000 vs 0.05 | ok +0.000 vs 0.1 | ok +0.630 vs 0.4 [+0.429, +0.794] | n/a | pass |
| `warehouse_a` | `amr/1` | `ego_now` | 16 | 105 | 0.000 | ok +1.000 vs 0.95 | ok +0.000 vs 0.02 [+0.000, +0.000] | n/a | ok +0.032 vs 0.1 | ok +0.876 vs 0.4 [+0.741, +1.011] | n/a | pass |
| `warehouse_a` | `amr/1` | `task_things` | 16 | 45 | 0.000 | ok +1.000 vs 0.95 | ok +0.000 vs 0.02 [+0.000, +0.000] | ok +0.000 vs 0.05 | ok +0.000 vs 0.1 | ok +1.000 vs 0.4 [+1.000, +1.000] | n/a | pass |

Each gate shows its verdict, its value against its bound and the 95% interval where it has one (`separation` also wants the interval's lower bound above 0.3). `ok` passes on the interval, `point` on the point estimate alone, `FAIL` fails, `n/a` does not apply or has no rows.

Point reading: pass. Interval reading: pass.

No instrument is broken.
- apartment_a vacuum/2: 4 of 36 compiled cases ask nothing and are not checked

On the release's compile `stbench check` passes in every reading: every gate on its point value and on its 95% interval,
and the evidence reading the export requires, with no instrument broken. No family fails, so none is disclosed as
failing. The balance rule holds everywhere: the best constant learned on other shifts scores at most +0.030 over
`all_unknown` held out (the helper's `task_things`, interval -0.164 to +0.194). The only thin cell is the helper's
heading `now`, a diagnostic already.

**The review and its repairs.** After the questions were built, an independent review of the branch read the code and the
compiled data. It found one blocker and several faults, and every one was repaired without moving a threshold to a score
(the findings are cited by number in [question-design.md](../question-design.md) and [CHANGELOG.md](../../CHANGELOG.md)):
- **B1, the decision-frame channel.** The helper's `before_place` frames came first in the look draw, and a decision frame
  puts a corridor question at the same plan pose, by the same furniture, in every script. A prior of other scripts'
  answers learned the helper's corridors and the evidence verdict failed on the helper's `clear_path` `memory_needed`. The
  frames left the decision frames and the cell reads +0.000. No gate, threshold or share changed.
- **S2,** `drop_zone_people` left the drone for want of supply (Section 3). **S3,** the robot's own velocity is delivered
  with every step, so a `closest_approach` row whose person stands is `now` and a system need not hold the frame before
  the ask. **S5,** a corridor's catastrophe needs the true corridor under the body's length plus C, so a 30 m answer to a
  24 m corridor is a wrong answer and not a collision. **S6,** a `now` row's age is 0. **S7,** the prompt states what
  omitting a cord from a list costs, for the job that asks the list.

**Pull and prompt checks.** The pull makes two of the leak scan's checks impossible by construction, since it serves one
item at a time in stream order: no question's text reaches the model before its ask, and no frame past an ask's step is
served before it. The scan stays as a check ([codex.md](../codex.md), "The leak scan, failures and exits"). The tools of the
reference agent cannot read the disk, so it cannot reach the truth. The canary that tests whether Codex carries
anything between cases (`tools/codex_canary.py`, which runs the pull path) was not run live before these rows, so whether
Codex carries anything from one case to the next on the pull path is
not checked live. Each case runs as a fresh exec that ignores the user's Codex configuration, in an empty working
folder.

**The leak check was not rerun.** Does asking during the job change the end answers? The check runs `gpt-6.1-sol`
(medium) over every short and medium case of the jobs that ask at their end (the helper, the vacuum and the AMR; the drone
asks nothing at its end) with the during questions off (`stbench eval --during off`), against the same model's row with
them on. Both arms stream the same
frames in the same order, and nothing in a frame marks where a question falls: the arms differ in the during asks alone.
`stbench pilot report --leak` reads d, the mean of (on - off) of the gate score over the end questions paired by case and
question, with a bootstrap interval over scripts, against the margin 0.05 declared before the arms were read. It was
not run for 0.10.0: the quota that would have paid for it is the one the rows used up. Its last reading is 0.9.0's, on
0.9.0's questions: pooled d = +0.043, interval [-0.030, +0.118], inconclusive, with the vacuum's own reading above the
margin (+0.178, interval [-0.037, +0.534]) ([history/report-0.9.0.md](report-0.9.0.md), section 6). 0.10.0 asks one and a half to five times as many
questions during the job, so that reading does not carry over, and a leak is not excluded.

**The given-name check.** Do a question's words find its target? On 0.7.0's questions a reviewer was shown 180 targets, with
only the question's words and the frame, and found 179, against a bar of 90% a place declared first
([history/report-0.7.0.md](report-0.7.0.md), section 6). The new families name no entity ("a person",
"anything"), and the others' targets are named as in 0.7.0; the check was not rerun.

## 7. Limits

- **Change over time is not tested.** 0.10.0 measures perception and memory: where things are now, and where things the
  robot saw earlier are, from seconds to hours before. No job asks what moved, appeared or went since an earlier pass, or
  where a moving person is going. The vision lists change and anticipation among the abilities; this release does not
  measure them, and the reason is in the data, measured on this release's compiled cases by a read-only count that calls
  no model:
  - The world changes only between tasks, so the 64 short cases, one task each, hold no change.
  - In the 16 medium cases of each job the world moves many things between tasks, but the robot's later tasks mostly go
    elsewhere. It sees both sides of a few, and only some of those are things a question could name, the only one of
    their description:

    | Job | Things moved between tasks | Seen on both sides | In how many of the 16 cases | A question could name |
    | --- | --- | --- | --- | --- |
    | `amr/1` | 133 | 27 | 14 | 7 |
    | `drone/1` | 90 | 11 | 4 | 3 |
    | `helper/1` | 146 | 28 | 14 | 6 |
    | `vacuum/2` | 165 | 20 | 11 | 1 |

  - Whether a thing moved is largely given by its kind. Downtown vans moved in 10 of 32 passes and skips in 0 of 39; the
    vacuum's shoes in 15 of 35 and chairs never; the warehouse's drums in 54 of 109 and roll cages in 0 of 43. A fair
    question has to balance moved and unmoved within a kind, or the kind answers it, and that leaves the built
    `same_place` family 4 questions in the whole release. No job reaches the 20 questions a counted cell needs.
  - Anticipation is untested too. Of the 46 `closest_approach` questions only 2 have a person walking (both the AMR's);
    the helper's resident never walks at a frame tried. A score on the family is mostly about ranging a still person from
    the picture and reading the robot's own velocity.

  The fix is in the data and not in the questions: scripts that bring the robot back to places after the world has
  changed; what changes drawn per thing by a coin and not by its kind, as the drone's drop-zone coin already is; and more
  people walking in view. That needs new renders, so it is left for a later version.
- **Help is uncounted.** The help kind holds `open_heading` alone, which stays a diagnostic: asked, gated and reported,
  and counted in no headline mean. Its golds are right and the convention is followed, but its credit is a cliff: a run
  changes by metres within a few degrees at a doorway, a cord's end or a furniture edge, while the regret is read against
  C, 10 to 15 cm, so a heading a few degrees off the best opening scores -0.4 to -1, as does one that leaves 1 to 2 m of
  room. That breaks the vision's "roughly right ones still score". Any tolerance chosen now would be chosen after reading
  scores, so the credit is redesigned before any of its scores are counted (in the next version). The
  numbers are on [families/open_heading.md](../families/open_heading.md) and in
  [the report of 0.9.0](report-0.9.0.md), section 7. Only the apartment asks it, so the help kind is the
  apartment's alone and is a split of the reports. The help questions the vision names, which route and which approach,
  are not built.
- **The map reader is a reference, not a competitor.** It is published only to show which questions ask about fixed
  structure, because the agent is never given a map and every case starts a fresh agent that has never seen the place.
  Where it scores high, as on the drone's in-view corridors, which only building walls bound because every movable thing
  downtown is under the drone's flight height, those questions measure the perception of fixed geometry and not anything
  that changes. That is declared behaviour of a structure family on a public place (methodology rule 5). A held-out place
  would make the score a claim about every system, and there is none.

  On these rows the map reader scores +0.65 downtown, between sol and astra, because its in-view corridors score 1.00
  there; +0.22 in the apartment, from its corridors; and the floor's +0.08 in the warehouse, which asks no corridor.
- **Dev rows, no held-out set.** The questions, the screens, the credit, the slot tables and the repairs were settled on
  these 64 shifts, and the repairs came from the gates, the golds and a review of the compile, not from a score. No fresh
  shifts were rendered to confirm the rows. The gold is public and each place is one layout, so a result can be overfit.
  Publish your agent's code with its result.
- **The person out of view goes unasked.** `person_mem` is dropped from the helper (Section 3), so where a person who has
  left the view is has no question; the person in view is `closest_approach`'s. It returns when the answerable supply lets
  `separation` pass. The drone's people are unasked too.
- **Thin cells, and supply.** A cell with fewer than 20 questions is a diagnostic and counts in no mean; small cells have
  wide intervals, and the 20-question rule was not loosened to count a cell. The AMR asks one question in about 25
  frames, above the target of one in 15: every one of its slots is already whole, a person is in range at few draws, and
  the floor holds few things to name twice ([question-design.md](../question-design.md), section 5).

  On the release's compile the only thin cell is the helper's heading `now`, a diagnostic already.
- **Memory is tested at 1 to 3 tasks.** The data reach hours, and the long cases (4 to 6 tasks) have no row here; real
  shifts last hours. At the lengths of the rows, memory is recall over a long context more than over compaction or notes.
  The `earlier` age band needs a second task, so no short case holds one.
- **The memory is the agent's own context.** The reference agent has no memory tool, so how well it remembers depends on
  how Codex manages and compacts its thread, which the benchmark neither sees nor sets. Token counts and costs are lower
  bounds (no cache writes are reported, and a killed turn prints no usage). The rows ran 18 execs at once with no
  rate-limit error; the plan's weekly quota, not the rate limit, is what stopped them.
- **One place per domain:** one apartment, one warehouse, one downtown block. Nothing here says how an agent does in a
  place it has never seen. Downtown is the easiest place: nothing at the drone's height but walls, a clearance of 2.4 m,
  and corridors a map answers.
- **`unknown` on a glimpse.** Some wrong answers to an `unknown` question land on a thing seen only in glimpses too small to
  show what it is, because the prompt never says that a question may name a thing never seen or not there. The fix is a
  prompt sentence, a brief change for a later version.
- **A known floor-texture mark.** The apartment's floor has a cable-like mark that a model twice took for a cable bundle in
  0.7.0's pilot, and the vacuum is now asked about cables. Removing the mark needs a re-render.
- **The score and control.** Whether the score predicts control has no closed-loop check yet, and none is claimed.
- **Single runs.** Each model ran once, at one reasoning effort. Run-to-run variance is not measured, and the rows say
  nothing about other models, efforts or memory designs.
- **42 of the 124 cases.** The rows cover the 42 short and medium cases all three models had finished when the runs used
  up the plan's weekly quota (Section 4), about 11 to 20 a place, so their intervals are wide. A full row of astra at
  xhigh costs, by the quota these runs used, about a week of a Pro plan's quota on the pull.
- **The leak check was not rerun** (Section 6). 0.9.0's inconclusive reading was on other questions, and 0.10.0 asks one
  and a half to five times as many during the job, so a leak is not excluded.

## Reproducing the numbers

From a downloaded dataset (the README's Quickstart). The rows ran over 42 cases, whose ids are the `cases[].id` of each
`results/<name>.json` in the dataset: `--cases ID,...` in place of the two `--select` options reproduces them, and the
commands as written run every short and medium case.

```sh
stbench eval all_unknown --data data --select length=short --select length=medium --out runs/all_unknown
stbench eval map_reader --data data --select length=short --select length=medium --out runs/map_reader
stbench eval codex:gpt-6.1-sol --data data --select length=short --select length=medium --reasoning medium --workers 6 --out runs/sol
stbench eval codex:gpt-6-luna --data data --select length=short --select length=medium --reasoning xhigh --workers 6 --out runs/luna
stbench eval codex:gpt-6-astra --data data --select length=short --select length=medium --reasoning xhigh --workers 6 --out runs/astra
stbench eval compare runs/all_unknown runs/map_reader runs/luna runs/sol runs/astra
stbench eval distance runs/sol runs/all_unknown --data data
stbench check --data data --verdict evidence --out check.json
```

The leak check, not run for 0.10.0, is one more run of sol, with the during questions off, over the cases of the jobs that ask at their end,
and the report that reads it:

```sh
stbench eval codex:gpt-6.1-sol --data data --select length=short --select length=medium --select job=helper/1 --select job=vacuum/2 --select job=amr/1 --reasoning medium --during off --workers 6 --out runs/sol-off
stbench pilot report --leak runs/sol runs/sol-off
```

Each run's `results.md` gives the headline, abilities, kinds, families, the age of the evidence, the catastrophes, credit
bands, unpenalised credit, abstention bound and resources; `results.json` holds every scored row. `eval distance` gives the
offset errors against C and the credit sensitivity. A model's answers differ from run to run.
