# SpatiotemporalBench 0.7.0: technical report

Version 0.7.0, not released: an internal checkpoint, never tagged or uploaded (0.9.0 ships its data) · code:
[DanMcInerney/SpatiotemporalBench](https://github.com/DanMcInerney/SpatiotemporalBench)

*This is the technical report of 0.7.0, kept for comparison. It describes the three-rung credit, the four question
families of that release and its rows. The current report is [../report.md](../report.md); what changed is in
[the changelog](../../CHANGELOG.md).*

This report goes deeper than the [README](../../README.md). It covers what 0.7.0 measures, how it scores, which questions
it asks and why, what two reference models scored, whether the design did what it set out to do, how validity is
checked, and where the benchmark stops. Every number comes from the release's compiled data and result files, or from
a review named where the number stands.

**In short.** A simulated robot does a real job in a photoreal place. Its camera frames reach the agent a few at a time,
with the robot's pose, clock and status, and no depth. The agent is asked where things are: in view now, seen earlier,
on the map, and near its route. Code checks each answer against the simulator's record, and credit is set by what the
job needs. On the 120 short and medium cases that ask a question, `gpt-6.1-sol` at medium effort (sol) scores +0.55
in the apartment, +0.42 in the warehouse and +0.77 downtown; `gpt-6-luna` at xhigh (luna) scores −0.18, −0.16 and
−0.12; answering `unknown` to everything scores +0.05, +0.06 and +0.13. These are dev rows: the questions were refined
on the shifts they are scored on, and nothing is held out.

## 1. What is measured, and the setup

The agent acts as the robot's onboard brain and is asked what a controller asks itself. **Distance:** where is that
thing I can see, from me, now? **Memory:** where, as I last saw it, is that thing from me now, and where on the map did
I last see it? **Relevance:** where were the things near the way I went?

| Job | Robot | Camera height and pitch | Place | One task |
| --- | --- | --- | --- | --- |
| `vacuum/2` | robot vacuum, a 34 cm disc | 0.08 m, fixed, 5° up | `apartment_a` | cleans one or two rooms |
| `helper/1` | Stretch-class home helper | 1.30 m, tilts from 60° down to 20° up | `apartment_a` | carries a thing from a surface in one room to a surface in another |
| `amr/1` | MiR250-class tote robot | 0.50 m, 10° down | `warehouse_a` | docks at tote shelves and packing tables, or runs a picking loop |
| `drone/1` | Wing-class tether delivery drone | under the body, 20° to 90° down; cruises at 15 m, lowers from 7 m | `downtown_a` | flies to a drop point, inspects it, lowers a package |

Every camera gives a 640 × 480 JPEG with an 80° horizontal field of view, a focal length of 381 pixels. Nothing is
labelled and no depth is sent: the agent estimates every distance from the pixels. Each frame has one header line with
the step and clock, the base's position and heading in map metres, the camera's pitch, the robot's status and its
subgoal.

![A frame from the home helper's camera: a dark TV console with a speaker, a picture frame, a router, a remote and books, over a wooden floor](../figures/helper-frame.jpg)

*A frame as the agent receives it: the helper's camera, 1.30 m up and pitched 45° down.*

**Shifts and cases.** A shift is 2 to 6 tasks of one job. Between tasks the robot is parked and unobserved for 10
minutes to 3 hours, and the place changes: things move, doors open and shut, people come and go. Each job has 16
shifts, 12 of 2 or 3 tasks and 4 of 4 to 6, rendered once in Blender. A case is a window of a shift, streamed from
step 0 on a fresh clock: `short` (1 task), `medium` (2 or 3) or `long` (4 to 6), 16, 16 and 4 a job. By the medians
per job, a short case is 46 to 98 frames and 1 to 3 minutes of work, a medium one 100 to 267 frames over 41 to 98
minutes, and a long one 217 to 832 frames over 2.7 to 4.3 hours, unobserved gaps included.

| Family | Ability (column) | When | It asks | Answer |
| --- | --- | --- | --- | --- |
| `ego_now` | distance (`now`) | during, at a look point | where the nearest visible point of a thing in view is, from the camera, along the floor | offset: metres ahead, metres left |
| `ego_mem` | memory (`memory`) | during, at a look point | the same for a thing seen before the last batch and not in it, from where the camera is now | offset |
| `object_where` | memory (`memory`) | after the job | where on the map it last saw a thing | a map point |
| `task_things` | relevance (`memory`) | after the job | where the things near its task were, by the job's rule: on the floor of the rooms it cleaned (vacuum), on the floor within 1.2 m of its way (helper), within 2 m of its drive (AMR) | a list of map points |

Look points are batch ends at least 4 steps into a task and 8 steps apart, at most 6, 10 and 14 in a short, medium and
long case. They are drawn from the case id and the stream's cuts, never from the world, and each asks at most one
`ego_now` and one `ego_mem` question. The end questions come together after the last frame. The drone asks no
`task_things`: its things stand in parking lanes and loading zones that the public map gives away. About a fifth of
the `ego_mem` and `object_where` questions have `unknown` for answer, in the same words as the rest: a thing no frame
showed, or a description that matches nothing in the place.

**Counts** in the compiled data: all 144 cases, with the 120 scored short and medium cases in brackets.

| Job | Cases that ask (short / medium / long) | `ego_now` | `ego_mem` | `object_where` | `task_things` | All | Frames |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `vacuum/2` | 13 / 14 / 4 | 20 (16) | 66 (51) | 24 (21) | 47 (34) | 157 (122) | 6,460 |
| `helper/1` | 16 / 16 / 4 | 74 (64) | 143 (119) | 76 (68) | 48 (32) | 341 (283) | 3,157 |
| `amr/1` | 16 / 16 / 4 | 48 (37) | 34 (28) | 46 (43) | 46 (37) | 174 (145) | 4,679 |
| `drone/1` | 14 / 15 / 4 | 42 (37) | 45 (34) | 26 (22) | – | 113 (93) | 2,534 |
| All | 59 / 61 / 16 | 184 (154) | 288 (232) | 172 (154) | 141 (103) | 785 (643) | 16,830 |

Unknown golds are 67 of 288 `ego_mem` questions (23%) and 38 of 172 `object_where` questions (22%); 34 of 141
`task_things` golds (24%) are empty lists. Eight short and medium cases ask nothing, five of the vacuum's and three of
the drone's.

## 2. Scoring

**Credit by the job.** An offset is credited against the job's clearance C, the margin the robot's own operating spec
keeps beside its path. A map point is credited against the re-find radius R, within which the robot, driving back,
finds the thing again with it in view.

| Job | C | Offset rungs | Source of C | R | Source of R |
| --- | --- | --- | --- | --- | --- |
| `vacuum/2` | 0.10 m | 5, 10, 20 cm | the loose end of what camera vacuums aim to pass beside an obstacle: Roborock says the S7 MaxV gets close to shoes, and a buyer's guide calls a 15 cm berth wasteful | 1.0 m | HomeRobot's real-world find rule (in view within 1 m); Habitat ObjectNav's 1.0 m success radius |
| `helper/1` | 0.15 m | 7.5, 15, 30 cm | OWMM-Agent counts a pick when the end-effector is within 0.15 m of the target | 1.0 m | as the vacuum |
| `amr/1` | 0.45 m | 0.225, 0.45, 0.9 m | the MiR250's specification: a 580 mm robot needs a 1,450 mm corridor, 0.435 m a side | 1.0 m | a convention: no warehouse source gives one |
| `drone/1` | 2.4 m | 1.2, 2.4, 4.8 m | Wing's patent US12399508B2: a delivery point at least 8 ft from trees and 10 ft from buildings | 2.0 m | Wing's patent US12221211B2: never more than about 2 m from the intended point |

C is never a detection size, a docking or placement precision, or a facility-layout rule. So a robot vacuum's 15 mm
detection limit, the MiR's ±3 mm docking, ISO 3691-4's 0.5 m beside a path and Zipline's "within two feet" placement
were each rejected as a source. The values were fixed before 0.7.0's first run and have not moved since. The review
that fixed them re-scored 0.6.0's answers under them, and it raised the drone's C from a proposed 0.6 m to 2.4 m, which
made the drone easier. The prompt states C, R and the rungs in metres and centimetres.

**Three rungs and a penalty.** An offset's error is the distance along the floor between the answered and the true
offset, and the rungs do not widen with range. A map point is measured to the middle of the thing; its rungs grow by
the kind's half-length beyond 0.25 m only where the thing's top was above the camera at every sighting. Each rung met
is a third of the credit c, and an answer scores c − (1 − c):

| The answer | Credit c | Score |
| --- | --- | --- |
| within C/2 (R/2) | 1 | +1 |
| within C (R) | 2/3 | +1/3 |
| within 2C (2R) | 1/3 | −1/3 |
| beyond 2C (2R) | 0 | −1 |
| `unknown`, where the frames hold the answer | | 0 |
| `unknown`, where nothing the robot sensed answers | | +1 |
| anything else, where nothing the robot sensed answers | | −1 |

An answer beats `unknown` only within C, and the prompt says to answer only when more likely right than wrong: knowing
one's own precision is part of the ability. A list is credited by its F1 against the gold's things, a listed point
matching a thing within 0.5 m in the home and 1 m in the warehouse (more for a large thing). A point in a don't-care
circle counts neither way: a thing at the edge of the rule, a thing another question names, or a thing too thin to
matter, such as a remote. An empty list is right only when the gold is empty.

**The headline** is (`now` + `memory`) / 2. `now` is distance. `memory` is the mean of memory (`ego_mem` and
`object_where`, pooled) and relevance; on the drone it is memory alone. The columns count equally whatever their
numbers of questions: for sol in the apartment, (+0.542 + (+0.476 + 0.638) / 2) / 2 = +0.549. Intervals are 95%
percentile bootstraps over shifts (2,000 resamples), each resample recomputing the abilities, the columns and the
headline. No number pools places. An ability with fewer than 20 questions on a job would be a diagnostic, counted in
no mean; none is in 0.7.0 (the vacuum's `ego_now` has exactly 20).

**Why the target is the nearest visible point.** A ground camera sees a thing's near face, and the middle sits behind
it. On these shifts' 0.6.0 questions, a reader given the scorer's depth and no size priors answered the centroid of
each thing's visible pixels. It scored +0.90 to +1.00 under 0.6.0's range-relative credit. Against the middle at the
job's C it scored:

| Job | Middle behind the nearest visible point (median) | Depth-perfect reader against the middle at C: `ego_now`, `ego_mem` |
| --- | --- | --- |
| `vacuum/2` | 0.19 m | **+0.25**, +0.47 |
| `helper/1` | 0.18 m | +0.75, +0.76 |
| `amr/1` | 0.41 m | +0.99, +0.98 |
| `drone/1` | 1.72 m | +0.45, +0.55 |

A question that a perfect perceiver fails without knowing each kind's size is partly a trick. So the gold is what the
frames show: the point of the thing's floor hull (its visible pixels, dropped onto the floor through the scorer's
depth) nearest the floor point under the camera. For `ego_mem` the hull is the union of every recognisable sighting at
the thing's current place. A pixel-perfect reader reproduces this gold exactly; had the gold been the nearest corner
of a footprint box, that reader would score only +0.76 to +0.92. The words say "along the floor" because on the drone
the 3-D nearest point and the floor-nearest point differ by more than 0.3 m in half the golds. They say "as you last
saw it", "you saw" and "now" because a draft reading, "the side that faced you", differs from the gold by more than
C/2 in 49-77% of `ego_mem` questions. In the dev pilot, 85% of sol's `ego_mem` misses were estimation errors on the
gold's own point. `object_where` keeps the middle: a place to drive back to, credited on R.

## 3. Which questions are asked

A question is asked only where a careful reader of the frames could answer it within the job's accuracy, and never
where the words or the place give it away. Each screen below is one rule for every job. It reads the timeline and the
cameras, never a gold or a score, and it removes questions without adding any.

**Recognisable.** A thing is asked about only from frames where a person could name it: at least 1,024 pixels, a mean
luminance of at least 0.01, a box at least 2 pixels clear of every edge, and a depth reading. Its colour word must be
the colour most of those frames show. This keeps out specks, slivers cut by the edge, things in the dark, and labels
the render does not bear out. It removes most sightings: at the candidate look points of these shifts, of the movable
things in a frame, 9% (vacuum), 17% (helper), 18% (AMR) and 37% (drone) were recognisable. On an earlier dev render,
103 of 602 carried things (17%) did not show their colour word. A blind crop review tested a 512-pixel floor against a
90% bar declared first; it named 60 of 60 crops downtown, 34 of 56 in the warehouse and 39 of 60 in the apartment, so
the floor stays at 1,024 pixels everywhere.

**No look-alikes.** A thing is not named while something it could be taken for is sighted in its colour, with black,
grey and white counted as one colour. The groups include tote, bin, box and bag; car, van, pickup and box truck;
dumpster and skip; mug, glass and bowl; and, new in 0.7.0, {folded clothes, towel, book, book stack}, {jacket, hanging
coat, clothes pile, folded clothes} and {cable bundle, extension cord}. A decoy, a description that matches nothing in
the place, is asked only when its kind was sighted and no look-alike in its colour was, and at most once a case. The
screen keeps out questions whose words fit two things. The dev pilot caught two the old groups missed: asked about a
cable bundle, sol answered at an extension cord, and asked about a red book it never saw, at a red towel. On an earlier
dev render the screen removed 4% (AMR), 11% (drone), 16% (helper) and 28% (vacuum) of `object_where`'s answerable
candidates. On these shifts, merging towel with book and adding the cable pair took the helper's `object_where` from
85 questions to 76.

**The surface tie.** A reader with no depth places a point where its pixel ray meets a surface whose height it knows.
So an `ego_now` or `ego_mem` thing is asked only where an evidence frame ties its gold point to a surface. Either a
visible pixel shows the thing meeting the floor, a shelf or a table top within C/2 of the gold point, and one pixel's
error there, e = (d² + H²) / (H · f), is within C/2 (d the floor distance, H the camera's height above that surface, f
the focal length in pixels). Or the thing stands at the near edge of a support whose top is at or above the camera,
and the edge is the tie. Things hanging on a face (coats on a wardrobe's side) and overhangs over a hidden base (a
floor lamp's shade) are never tied. From the vacuum's 8 cm camera one pixel reaches C/2 out to about 1.2 m; from the
AMR's and the helper's, about 6 m. The rule replaced a first screen that read each pixel at its true height, which a
reader with no depth cannot know; under it, seven floor lamps and five hanging coats were still asked, and the surface
tie drops all twelve. On all 144 cases it took `ego_now` / `ego_mem` from 33 / 78 to 20 / 66 on the vacuum, 52 / 51 to
42 / 45 on the drone, 78 / 141 to 74 / 143 on the helper and 37 / 39 to 48 / 34 on the AMR, where a contact read at a
rack's top admits things the first screen dropped. A floor-pixel-only rule was rejected: it cut the AMR's `ego_mem`
from 39 to 17 and the drone's `ego_now` from 52 to 16, dropping rack and table things the log reviews judged fair.

**The set-back screen.** A thing resting on a surface at or above the camera is not asked when its gold point lies
more than C/2 behind the surface's near edge, along the sight line. From below, the camera sees the edge, not how far
back the thing stands: a box 0.23 m back on a 0.9 m packing table seen from the AMR's 0.5 m camera, or a microwave on a
counter seen from the vacuum's 8 cm. With the merged look-alike groups, it removed 34 of the vacuum's 112 `ego_mem`
questions and 13 of its 46 `ego_now` questions, mostly things on counters and desks. It drops all 19 set-back things
among sol's abstentions in the dev pilot. Most of the AMR's raised things stand at a rack or table edge, and stay.

**Unchanged placement.** An `ego_mem` thing is asked only when every sighting of it, from its last recognisable one to
the look point, lies within 1 cm of that place. 0.6.0 allowed 0.15 m, more than C/2 for the vacuum. The rule keeps a
faithful memory and the gold from disagreeing because a thing was nudged after it was seen. It removed no question on
the earlier dev render where it was measured: it is a guard. `object_where` keeps 0.15 m, against an R of 1 m.

**Nothing given away.** Light furniture is never asked, since its place is drawn within 1 m of where the public map
puts it. Nothing the robot's own log tells is asked: the thing it carries, or a thing its log says blocked it. No thing
is asked twice in a case. In all, on the same 144 cases:

| Compile | Questions |
| --- | --- |
| 0.6.0 | 1,009 |
| 0.7.0's dev pilot: the nearest visible point, a first resolvability screen, the soft-goods groups | 883 |
| after the log reviews' repairs: the merged look-alike groups and the set-back screen | 823 |
| 0.7.0: the surface tie | 785 |

## 4. Results per place

The reference agent is the Codex CLI (`codex/9`) on a ChatGPT login. Its one tool is `submit_answer`, and it has no
memory tool: its context is its memory. In the dev pilot every frame stayed in sol's context, with no compaction. Its
prompt states the camera model, C, R, the rungs and the target point. The rows use the camera channel (`rgb_pose`), the
correct brief and the during questions, over the 120 short and medium cases that ask a question. Sol ran at medium
reasoning effort and luna at xhigh, once each. Nine of sol's cases and three of luna's ended on a Codex capacity error
and were rerun from their start, so both rows cover all 120 cases. The floor, `all_unknown`, answers `unknown` to
everything.

Beside each penalised headline stand two diagnostics. The **unpenalised credit** is the headline's statistic over each
answer's credit c, with no penalty; it runs from 0 to 1, and a right `unknown` earns 1. The **abstention bound** is the
headline had the system said `unknown` exactly where an answerable question's answer scored below 0: the most that
knowing its own precision could add.

| Place (cases, questions) | System | Headline [95%] | Distance (`now`) | Memory | Relevance | `memory` column | Credit | Bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Apartment (59, 405) | sol | **+0.55** [+0.48, +0.61] | +0.54 | +0.48 | +0.64 | +0.56 | +0.72 | +0.62 |
| | luna | −0.18 [−0.28, −0.09] | −0.35 | 0.00 | −0.02 | −0.01 | +0.31 | +0.19 |
| | floor | +0.05 [+0.04, +0.07] | 0.00 | +0.22 | 0.00 | +0.11 | +0.05 | +0.05 |
| Warehouse (32, 145) | sol | **+0.42** [+0.31, +0.53] | +0.41 | +0.68 | +0.20 | +0.44 | +0.60 | +0.48 |
| | luna | −0.16 [−0.29, −0.03] | −0.26 | −0.02 | −0.09 | −0.06 | +0.28 | +0.16 |
| | floor | +0.06 [+0.03, +0.08] | 0.00 | +0.23 | 0.00 | +0.11 | +0.06 | +0.06 |
| Downtown (29, 93) | sol | **+0.77** [+0.62, +0.90] | +0.80 | +0.74 | – | +0.74 | +0.88 | +0.81 |
| | luna | −0.12 [−0.27, +0.04] | −0.31 | +0.06 | – | +0.06 | +0.35 | +0.19 |
| | floor | +0.13 [+0.06, +0.19] | 0.00 | +0.25 | – | +0.25 | +0.13 | +0.13 |

- **The floor's memory** is +0.22 to +0.25 because about a fifth of the memory questions have `unknown` for answer.
  Its distance is 0, since every `ego_now` question has an answer in view.
- **Within memory, the map is easy and the offset is not:** sol scores +0.84, +0.85 and +0.91 on `object_where`, but
  +0.29, +0.42 and +0.63 on `ego_mem`. Its weakest cell is the warehouse's relevance, where it answered `unknown` to
  78% of the whole-task lists.
- **Sol beats luna** in every cell, by 0.58 to 0.89 in the headline, and luna is below the floor on every place.
  Wrong answers with no credit were 7%, 5% and 5% of sol's questions and 32%, 29% and 27% of luna's.
- **The bounds.** Sol's is 0.04 to 0.08 above its headline. Luna's is above the floor on every place: its answers
  carry information, but it answers where it should decline. The penalty decides luna against the floor; it never
  decides sol against luna.

**Credit bands**, in percent of each ability's questions. `unknown` counts every `unknown` answer, right or wrong, so it
includes the right answers to unknown-gold memory questions. For a list, the band is of its F1.

| Place | Ability | n | sol: full / C / 2C / miss / `unknown` | luna: full / C / 2C / miss / `unknown` |
| --- | --- | --- | --- | --- |
| Apartment | distance | 80 | 56 / 23 / 14 / 5 / 3 | 11 / 18 / 23 / 45 / 4 |
| | memory | 259 | 37 / 13 / 8 / 8 / 35 | 18 / 8 / 6 / 31 / 37 |
| | relevance | 66 | 64 / 8 / 5 / 5 / 20 | 15 / 8 / 11 / 23 / 44 |
| Warehouse | distance | 37 | 38 / 38 / 14 / 5 / 5 | 14 / 22 / 19 / 41 / 5 |
| | memory | 71 | 56 / 14 / 6 / 7 / 17 | 17 / 14 / 9 / 31 / 30 |
| | relevance | 37 | 19 / 3 / 0 / 0 / 78 | 5 / 0 / 0 / 16 / 78 |
| Downtown | distance | 37 | 78 / 14 / 8 / 0 / 0 | 8 / 8 / 43 / 27 / 14 |
| | memory | 56 | 63 / 5 / 4 / 9 / 20 | 13 / 14 / 7 / 27 / 39 |

**The apartment's two jobs**, by the scorer's own statistic over the rows of one job:

| Job (cases, questions) | System | Headline [95%] | Distance (n) | Memory (n) | Relevance (n) | Credit | Bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `vacuum/2` (27, 122) | sol | +0.56 [+0.42, +0.69] | +0.48 (16) | +0.50 (72) | +0.78 (34) | +0.73 | +0.63 |
| | luna | −0.23 [−0.43, −0.03] | −0.35 | −0.07 | −0.12 | +0.28 | +0.16 |
| | floor | +0.03 [+0.01, +0.05] | 0.00 | +0.13 | 0.00 | +0.03 | +0.03 |
| `helper/1` (32, 283) | sol | +0.52 [+0.42, +0.60] | +0.56 (64) | +0.47 (187) | +0.48 (32) | +0.69 | +0.59 |
| | luna | −0.15 [−0.25, −0.03] | −0.35 | +0.02 | +0.09 | +0.33 | +0.21 |
| | floor | +0.06 [+0.05, +0.08] | 0.00 | +0.25 | 0.00 | +0.06 | +0.06 |

The two jobs' headlines sit inside each other's intervals. The vacuum's distance rests on 16 questions, where sol
earned full credit on 50%, missed none and answered `unknown` to 6%. The vacuum's floor is lower because only 9 of its
122 questions have `unknown` for answer. Sol's `ego_mem` is +0.37 on the vacuum and +0.25 on the helper; its
`object_where` is +0.81 and +0.85.

**Cost and tokens.** The cost is notional, at the pinned price table (`prices/1`), and a lower bound, since Codex
reports no cache writes. The runs spent no dollars: both rows together used about 4% of a ChatGPT plan's weekly quota.

| Place | System | Input tokens a case (cached) | Output tokens a case | Notional cost a case | Wall time a case |
| --- | --- | --- | --- | --- | --- |
| Apartment | sol | 2.02 M (95%) | 1,944 | $0.42 | 575 s |
| | luna | 2.08 M (94%) | 13,893 | $0.039 | 566 s |
| Warehouse | sol | 1.48 M (93%) | 1,474 | $0.35 | 531 s |
| | luna | 1.56 M (93%) | 11,536 | $0.031 | 508 s |
| Downtown | sol | 0.54 M (88%) | 780 | $0.18 | 269 s |
| | luna | 0.59 M (88%) | 6,191 | $0.015 | 272 s |

In all, sol cost $41.05 and 182 M input tokens over the 120 cases, luna $3.77 and 190 M. Each row took about two
hours at 9 cases at a time. In the apartment a vacuum case, which streams more frames, took sol 2.77 M input tokens,
$0.53 and 666 s; a helper case took 1.39 M, $0.32 and 498 s.

## 5. Did it work

The goal was realistic difficulty without tricks: credit as tight as the job needs, questions about what the job meets,
and no question a careful reader of the frames could not answer. A strong model somewhere around 0.25 to 0.5 was a
sanity check on realism, never a target. No C, R or rule moved after a 0.7.0 score was read.

**Three steps on the same 64 shifts.** 0.6.0 asked 821 questions over 124 cases, with credit relative to range and the
thing's middle as the target. 0.7.0's dev pilot asked 721 over 123 cases, before the log reviews' repairs. 0.7.0 asks
643 over 120.

| Place | Row | 0.6.0 | 0.7.0, dev pilot | 0.7.0 |
| --- | --- | --- | --- | --- |
| Apartment | sol headline | +0.69 | +0.42 | **+0.55** |
| | sol distance, memory, relevance | +0.66, +0.69, +0.75 | +0.33, +0.42, +0.63 | +0.54, +0.48, +0.64 |
| | luna, floor | +0.04, +0.05 | −0.16, +0.05 | −0.18, +0.05 |
| Warehouse | sol headline | +0.63 | +0.48 | **+0.42** |
| | sol distance, memory, relevance | +0.87, +0.68, +0.10 | +0.57, +0.51, +0.29 | +0.41, +0.68, +0.20 |
| | luna, floor | −0.05, +0.05 | −0.09, +0.06 | −0.16, +0.06 |
| Downtown | sol headline | +0.96 | +0.78 | **+0.77** |
| | sol distance, memory | +1.00, +0.91 | +0.83, +0.72 | +0.80, +0.74 |
| | luna, floor | +0.26, +0.11 | −0.01, +0.11 | −0.12, +0.13 |

Credit by the job did most of the work, from 0.6.0 to the dev pilot. The repairs then moved each place its own way.
The apartment rose as questions a camera-only reader could not place left: sol's apartment distance went from 43% full
credit, 14% missed and 14% `unknown` to 56%, 5% and 3%. The warehouse fell, because the surface tie admitted rack and
table things the AMR's camera can place and sol finds harder. The drone held.

**What the log reviews found.** Two reviews judged sol's lost credit in the dev pilot one question at a time, reading
the frames sol had, the target's mask, the gold, sol's point and the geometry from above. "Instrument" meant a wrong
gold, ambiguous words, a target the frames do not show, credit stricter than the frames allow, or a look-alike the
screens missed; anything else was the model's. No sampled gold was wrong.

| Job | Answers that lost credit: the instrument's share of it | `unknown` on answerable questions: the instrument's share |
| --- | --- | --- |
| `amr/1` | 1% (1 of 44 questions) | 4% (1 of 25) |
| `drone/1` | 0% (0 of 20) | none: sol never abstained on an answerable question |
| `helper/1` | 2% (1 of 33 sampled) | 4% (1 of 25 sampled) |
| `vacuum/2` | 20% (8 of 40; 95% interval 5-37%) | 52% (31 of 58; 39-62%) |

The model's losses were these. On the AMR, a thing's middle answered instead of its near face, mostly drums (24% of
lost credit), then bearing and decoys. On the drone, decoys (53%), each answered with a thing of another kind in the
named colour, then bearing and memory. On the helper, memory (32%) and range (30%); on the vacuum, memory (27%) and
range (22%). Asking about a thing the robot never saw is realistic ("where did you see my green bag?"), and the prompt
says `unknown` is then right. Sol's abstentions were its own on the AMR (`unknown` to 22 of 34 knowable whole-task
lists, against full credit on 11 of the 12 it answered) and on the helper (things seen in one or two frames, recalled
later).

The vacuum's instrument faults had seven causes. Four were repaired, each by a rule for every job: look-alikes outside
the groups (the merged groups above); thin things such as remotes, which the `task_things` gold counted as nothing
while the prompt counts them (now don't-care); 18 things set back on furniture above the 8 cm camera (the set-back
screen); and 12 floor lamps and hanging coats with no readable floor contact (the surface tie). Three were not. A
cable-like mark in the apartment's floor texture, which sol twice answered when asked where a cable bundle was, needs a
re-render. One question at exactly one pixel's error stands under the rule declared before data. One toilet brush
rendered too dark to name, and one case cannot set a threshold. No review was run on the final rows.

**Why the drone stays high.** Its clearance is large because its job's is: Wing keeps a delivery point 8 ft from trees
and 10 ft from buildings. And what it meets is easy for a strong model: 27 movable things, all vehicle-sized (skips,
cars, pickups, vans, food carts, portable toilets, dumpsters and box trucks), on flat ground, seen from about 15 m with
the pose given, so a thing's range follows from where its base sits in the picture. A reviewer named all 60 of its
crops at 512-1,023 pixels blind. Sol's median offset error is a third of C on the drone, against 0.55 to 0.61 of C on
the ground jobs:

| Job | C | Sol's median offset error | Within C/2 | Within C | Within 2C |
| --- | --- | --- | --- | --- | --- |
| `vacuum/2` | 0.10 m | 0.061 m | 43% | 79% | 98% |
| `helper/1` | 0.15 m | 0.082 m | 48% | 72% | 89% |
| `amr/1` | 0.45 m | 0.269 m | 41% | 81% | 96% |
| `drone/1` | 2.4 m | 0.766 m | 76% | 89% | 97% |

These are shares of the offset questions sol answered, `ego_now` and `ego_mem` together. Re-scored by `stbench eval
distance` with every C halved, the headlines would be +0.28 (apartment), +0.12 (warehouse) and +0.46 (downtown); with
C doubled, +0.72, +0.62 and +0.87. Scaling R by 0.5 or 2 moves no headline by more than 0.04. Even at half its sourced
clearance the drone would score +0.46. What would make it realistically harder is what it meets, small street things
and people, which 0.7.0 does not build.

## 6. Validity

Every gold is computed by code from the world timeline, the visibility log and the scorer's depth and masks, and names
the frames it rests on. Nothing the robot did not see changes a gold.

**The gates.** `stbench check` runs scripted systems over every case on the oracle channel, which sends each
recognisable detection's floor hull: what a perfect perceiver would know, and nothing about what is asked. Since the
memory table reproduces the gold from a channel a system could receive, the ceiling is a real test. The thresholds
were fixed before data.

| Gate | Rule |
| --- | --- |
| ceiling | the oracle scores at least 0.95 |
| knowable | the oracle less `memory_table`: upper 95% bound at most 0.02 |
| memory needed | on a memory family, the current view, with or without learned priors, scores at most `all_unknown` + 0.05 |
| shortcuts | each scores at most `all_unknown` + 0.10: answers learned from other shifts, the robot's path log with the public map, another case's stream, the questions' wording, a form adversary |
| separation | `memory_table` beats the best shortcut (for `ego_now`, also the kind's typical range) by at least 0.40, lower 95% bound above 0.30 |
| balance | no constant answer, learned on other shifts, beats `all_unknown` |

The evidence verdict is stricter: a shortcut fails when its gain is over the margin with a 95% interval above 0, in the
whole group, in any length or in either slice. Over all 785 questions of the 144 cases (the 136 that ask one), the
check gives ALL GATES PASS and EVIDENCE VERDICT PASSES, with no broken line:

| Job | Families (questions) | Oracle | Worst shortcut over the floor | Lowest separation [95%] |
| --- | --- | --- | --- | --- |
| `vacuum/2` | `ego_now` 20, `ego_mem` 66, `object_where` 24, `task_things` 47 | 1.00 | +0.07 | +0.83 [+0.69, +0.95] |
| `helper/1` | `ego_now` 74, `ego_mem` 143, `object_where` 76, `task_things` 48 | 1.00 | +0.01 | +0.75 [+0.63, +0.86] |
| `amr/1` | `ego_now` 48, `ego_mem` 34, `object_where` 46, `task_things` 46 | 1.00 | +0.00 | +0.71 [+0.54, +0.86] |
| `drone/1` | `ego_now` 42, `ego_mem` 45, `object_where` 26 | 1.00 | +0.02 | +0.71 [+0.53, +0.85] |

The memory table matched the oracle on all 15 groups, and the current view sat at the floor (+0.00) on every memory
family. The worst shortcut, +0.07 against the 0.10 bound, was the robot's path log with the public map on the vacuum's
`task_things`. Separation ran from +0.71 to +1.00. The best constant learned on other shifts was `unknown` itself on
13 groups; on the other two, a fixed list of two places scored below `all_unknown` (−0.07 on the helper's
`task_things`, −0.36 on the AMR's). An earlier check, on the dev pilot's 883 questions, also passed every gate.

**The given-name check.** Do a question's words find its target? A reviewer was shown 180 targets of these shifts, 60 a
place, across `ego_now`, `ego_mem` and `object_where`, with only the question's words and the frame, and marked a point
before any mask was opened. Against a bar of 90% a place, declared first, 179 of 180 were found: 60 of 60 in the
apartment and the warehouse, 59 of 60 downtown, where the miss was the reviewer's own slip. Counting confident calls
alone, the apartment falls to 42 of 60: the other 18, often a dark electronic box or a soft garment, were found as the
only thing the words fit. The same check found folded clothes, towels, books and jackets side by side with no
look-alike group, which led to the soft-goods groups.

## 7. Limits

- **Dev rows, no held-out set.** The questions, the screens and the repairs were settled on these 64 shifts, after
  sol's dev-pilot answers there were read. Each repair is a rule for every job, made because the instrument was wrong,
  but no fresh shifts were rendered to confirm the rows. The gold is public and each place is one layout, so a result
  can be overfit. Publish your agent's code with its result.
- **Memory is tested at 1 to 3 tasks.** At these lengths every frame of a case stayed in sol's context, so memory here
  is recall over a long context, not over compaction or notes. The 16 long cases (4 to 6 tasks) have no 0.7.0 row, and
  real shifts last hours. On 0.6.0's long cases sol showed no fall: +0.77 (apartment), +0.71 (warehouse) and +1.00
  (drone, 22 questions), against +0.69, +0.63 and +0.96 on short and medium, though the warehouse's memory fell from
  +0.68 to +0.52. Re-scored under 0.7.0's C and R as a proxy (against the middle, without 0.7.0's screens), the long
  cases give +0.37, +0.46 and +0.74, against +0.33, +0.48 and +0.94 on short and medium.
- **Change and prediction are not measured.** At sourced change rates the change families asked under 0.5 questions a
  medium case and left every job, and predicting where walking people go was measured as a no-go on the ground jobs.
  Whether the score predicts control has no closed-loop check yet.
- **One place per domain:** one apartment, one warehouse, one downtown block. Nothing here says how an agent does in a
  place it has never seen.
- **A known floor-texture mark.** The apartment's floor has a cable-like mark that sol twice took for a cable bundle
  in the dev pilot. Removing it needs a re-render. One question sits at exactly one pixel's error, and two dark renders
  were noted, too few to set a rule.
- **The drone sits near its ceiling** at its sourced 2.4 m clearance. This is reported, not tightened.
- **Single runs.** Each model ran once, at one reasoning effort. Run-to-run variance is not measured, and the rows say
  nothing about other models, efforts or memory designs. Small cells, such as the vacuum's 16 scored `ego_now`
  questions and the drone's 22 `object_where` questions, have wide intervals.

## Reproducing the numbers

From a downloaded dataset (the README's Quickstart):

```sh
stbench eval all_unknown --data data --select length=short --select length=medium --out runs/all_unknown
stbench eval codex:gpt-6.1-sol --data data --select length=short --select length=medium --reasoning medium --workers 9 --out runs/sol
stbench eval codex:gpt-6-luna --data data --select length=short --select length=medium --reasoning xhigh --workers 9 --out runs/luna
stbench eval compare runs/all_unknown runs/luna runs/sol
stbench eval distance runs/sol runs/all_unknown --data data
stbench check --data data --verdict evidence --out check.json
```

Each run's `results.md` gives the headline, abilities, families, credit bands, unpenalised credit, abstention bound and
resources; `results.json` holds every scored row, and section 4's per-job split is the scorer's aggregate over one
job's rows. `eval distance` gives section 5's error table and sensitivity. A model's answers differ from run to run.
