# SpatiotemporalBench

SpatiotemporalBench tests whether an LLM or agent, acting as a robot's onboard brain, knows the space around the robot
well enough to act in it: what its camera shows now, what it showed earlier in the job, and what either means for the
robot's next move.

A simulated robot does a real job in a photoreal place: a robot vacuum and a home helper in an apartment, a tote robot
in a warehouse, a delivery drone over a downtown block. Its camera recording is streamed to the agent one frame at a
time, with the robot's pose, its own velocity, the clock and its status; the agent asks for the next frame when it is
done with the last. During the job and after it, the agent is asked what a controller would ask itself: how far could my
body go before it touched anything, ahead, behind or after turning in place; how close would the person nearest me come;
where is that thing from me now, and the one I passed a minute ago; where on the map did I last see it. Code checks
every answer against the simulator's record of the world and credits it by what its error would do to the job. No
person or model writes or grades a gold answer.

**Why.** Robots are getting LLM and agent brains. A brain that drives a robot through a job has to keep track of the
space it works in, from its own sensors, over the job, and know when it does not know. This benchmark measures that,
perception and memory both, and nothing else: not control, not free text, not multiple choice.

**What it measures, and what it does not.** It measures perception and memory: where things are now, and where the
things the robot saw earlier are, from seconds to hours before. It does not test change over time. No question asks
what moved, appeared or went since an earlier pass, or where a moving person is going; [Limits](#limits) says why.

> **Version 0.11.0.** Four jobs on three places: 64 rendered shifts, 144
> cases, 1,761 questions. One public split, with the gold public and no held-out set. The
> reference rows are dev rows, scored on the shifts the questions were refined on (see [Limits](#limits)); the
> reference models are Claude models, where 0.10.0's were Codex models. To run it, go to the
> [Quickstart](#quickstart).

![A frame from the home helper's camera: a dark TV console with a speaker, a picture frame, a router, a remote and books, over a wooden floor](https://raw.githubusercontent.com/DanMcInerney/SpatiotemporalBench/main/docs/figures/helper-frame.jpg)

*Frame k18 of the case below, exactly as the agent receives it: the home helper's head camera, 1.30 m up and pitched
45° down, 640 × 480 pixels with an 80° horizontal field of view. Nothing in the image is labelled, and no depth comes
with it. Right after this frame the agent is asked how far the robot's body could move straight ahead before it touched
anything.*

## How questions are made and scored

**Every question is a decision the robot's brain faces in its job.** It asks about an action and its consequence, never
about a thing for its own sake. The questions cover four kinds of knowledge:
- **Harm:** what an action now would hit, tangle on or endanger.
- **Help:** which action now best moves the task forward, such as the best heading, or which way to set off to reach a
  thing seen before. Its two questions, `open_heading` and `way_to`, count in the headline.
- **Situation:** what is around the robot now, in view and out of view.
- **Reuse:** what is worth remembering to do the job here again, such as where things are kept.

The strongest questions need both what is in view now and what was seen before. Some ask about actions toward space the
robot cannot see now, which only memory answers. A question asks about the world and actions, never about a sensor's raw
output, so the same question fits any set of sensors. Answers are open: a distance, a heading, a place or a set, never
a multiple choice.

**Credit follows the consequence for the task.** The right direction earns some credit, the right thing or action more,
and the right amount within what the job needs more still. Full credit is kept for precision at the limit of what the
camera can resolve. What the job needs grows with distance: within ten of the job's own margin it is that margin, and
beyond it a tenth of the distance, the same for every robot, so a door 5 m away answered 4 m scores as good as `unknown`
on every ground robot, whatever its camera. Credit falls off smoothly, so roughly right answers still score and perfect
ones are rare. An error that would cause harm costs more than a cautious one. A set scores per item. `unknown` is right
when the sensors never showed the answer, and a confident wrong answer costs more than `unknown`. Full credit is hard
but always reachable: an ideal reader of the same frames, with perfect memory, scores 1. [Scoring](#scoring) gives the
arithmetic.

**What is asked, when, and why.** Questions are drawn at any frame of a case, the robot's decision frames first, up to
three at a frame and harm first, so a case asks about one question in every 9 to 25 frames. Each job has questions for
the hazards it meets: how far the body could move, the person nearest the robot and how close they would come, the cable
a brush could catch; and the helper is asked which way it would set off to fetch a thing it saw. Each answer records the
age of its evidence (now, within 10 s, earlier in the task, or before an unobserved gap), and the reports score by it.
An answer that would have ended in harm is charged a flat penalty in the reported score and in no validity check.
[docs/question-design.md](docs/question-design.md) says what each job's brain needs, which questions are asked of it,
which are deliberately not, when they are asked and how they are graded, with the measurements behind each choice.

## A case

Case `0b29b8dc4f4d5002` is one of the home helper's short cases: one task, carrying the grey remote from the TV console
to the shoe rack, in 71 frames. It asks ten questions: one of harm, four of situation, two of help and three of reuse.
Everything below is the case's own data and the benchmark's own words (long lines wrapped, `[...]` marks a cut), except
the answers in the last table, which are examples scored by the scorer itself. No model wrote them.

**1. The first call: the system prompt, then the task message.**

```text
You are the brain of a robot that is doing a job. You are told the job, then shown what the robot senses as it works,
and you are asked questions about it.
[...] The camera image is 640 x 480 pixels with a horizontal field of view of 80 degrees. The camera is mounted 1.30 m
above the base. The robot's body is a box 34 cm long and 33 cm wide and 1.41 m tall, centred on the base. It drives over
anything up to 1 cm high and passes under anything 1.435 m or more above the floor. The camera is the robot's only eye
on the world: nothing measures a distance for you, so any distance you give is your own estimate.

How this works:
- You receive what the robot senses by calling next_frame. Each call returns the next thing, in time order: a task
  message, one frame, the questions of an ask, or word that the job is over. Call it again when you are ready for the
  next; nothing advances until you do.
[...]
Answer only if you are more likely right than wrong; otherwise answer {"type": "unknown"}. A wrong answer costs more than
unknown.

How an answer is credited:
- How far the body could move is how far it can go before any part of it touches anything at its own height.
- Straight ahead, along the heading, count the distance its front edge moves.
[...]
- Credit is four quarters that add up.
- Quarter 1 is the right direction (set out below for each kind of answer).
- Quarter 2 is whole within twice the margin, falling steadily to nothing at four times the margin.
- Quarter 3 is whole within the margin, falling steadily to nothing at twice the margin.
- Quarter 4 is whole when the answer is as exact as one pixel of your camera at that point (never finer than 1 cm),
  falling smoothly to nothing at the margin.
- The margin is 1 m for a place.
- For an offset, a distance in metres or a heading, the margin is 15 cm or a tenth of the right distance (from your
  camera to the right point, the right number of metres or the best heading's run), whichever is larger.
[...]
- Some errors count against you twice as fast: on the harmful side the margin is half as large in quarters 2, 3 and 4.
[...]
- Going into something the robot had seen scores -2. That is a distance in metres longer than the right one, when the
  right one is under the room the body needs and yours is not.
[...]
- A question may name a thing your camera never showed clearly enough to tell what it is, or one that is not in this
  place at all; then `unknown` is right.

Task 0 begins at 2026-12-22T07:12:00.0.
You are the onboard assistant of a home helper robot working on one floor of a furnished home. Its current assignment is
to carry something from the living room to the hall. You will receive timestamped camera images and the robot's own pose
and status as it works.

At some points during the job you will be asked where a thing you can see is from your camera, where a thing you saw
before is from your camera, which way to set off to reach a thing you saw before, how far the robot could move, which way
the robot has the most room, and how close something would come to the robot. After the job you will answer structured
questions about the environment and its history. Use only evidence available so far. Answer unknown when it is
insufficient: a wrong answer costs more than unknown.
```

**2. A frame.** Frames arrive one at a time, each one header line and one attached image, and the agent calls for the
next when it is done with this one. This is frame k18, the one in the picture above (the header line, then its camera
image):

```text
[k18 2026-12-22T07:12:19.8] base at (0.94, 3.91, 0.00), heading -180.0 deg, velocity (0.000, 0.000) m/s, camera pitch
-45 deg; status: phase navigating, task 0, battery_pct 81; subgoal: going to the tv console to pick up the grey remote
```

**3. Harm (`clear_path`), asked with that frame:**

```text
Q1: If you drove straight ahead now, how far could your body go before it touched anything? Answer from what you have
seen, in metres from the front of your body; answer 5 if nothing is within 5 m.
Answer type: number. answer = {"type": "number", "value": X} (a number; metres for a distance)
```

The gold is 0.307 m: how far the body could go before it touched the first thing the robot has seen in its way, from
the cells the robot has seen free. One pixel of the camera, at the contact the stop rests on, is 4.7 cm, so full credit
is within 4.7 cm. The helper's clearance C is 0.15 m. [Scoring](#scoring) works this question through.

**4. Situation (`ego_now`), at frame k22.** The robot is still at the console, picking up the remote.

```text
Q1: Where is the point of the picture frame you can see now that is nearest to your camera, measured along the floor?
Answer in metres ahead of the camera and to its left.
```

The gold is 0.760 m ahead and 0.083 m to the left.

**5. Help (`way_to`), at frame k38.** The robot is carrying the remote to the shoe rack, at (4.10, 3.84) heading 0°. The
router it saw on the console, around frames k15 to k22, is behind it and out of view.

```text
Q1: If you had to go now to the router you saw, which way would you set off to get there by the shortest way your body
fits through, of the ways you have seen? Answer the heading in degrees to the left of the way you face now, from -180 to
180 (to your right is negative).
```

The gold is −162°, nearly an about-face to the right. For each heading the gold holds how much nearer to the router one
metre of driving along it brings the robot, by the shortest way its body fits through the floor it has seen or driven:
1.08 m for −162°, 0.84 m for 174°, 0.39 m for 150°, −1.0 m for straight ahead (away from it), and none for the headings
to the robot's right, which have no room to move or run into floor it never saw.

**6. Situation (`ego_mem`), at frame k56.** The robot has put the remote on the shoe rack and is going back to its spot.
It is at (5.46, 0.96), heading 147°, and the green plate it saw in frame k46 has not been in view for the last eight
frames.

```text
Q1: As you last saw it, where is the point of the green plate you saw that is nearest to your camera now, measured along
the floor? Answer in metres ahead of the camera and to its left.
```

The gold is 0.316 m behind the camera and 0.257 m to its left.

**7. Reuse (`object_where` and `task_things`), after the job:**

```text
Q1: Where did you last see the brown bag? Answer in map coordinates.
Q2: Where did you last see the blue bowl? Answer in map coordinates.
Q3: Where were the things on the floor within 1.2 m of the way you went to fetch and put down the thing in your first
task, which you would have to steer around next time? Answer in map coordinates. [...]
```

The golds are the bag at (0.88, 4.42), the bowl at (3.18, 4.16), and for the list one item, a beige clothes pile at
(2.70, 3.79), with 31 don't-care circles: things the question counts neither way, such as the bag and the bowl that the
other questions name.

**The whole case,** with example answers and what the scorer gives them. A score is `2s - 1` for a credit `s`, or minus
the weight of a catastrophe whatever the credit; `unknown` scores 0.

| Frame | Kind | Family | Asked about | Gold | Example answers: score |
| --- | --- | --- | --- | --- | --- |
| k18 | harm | `clear_path` | straight ahead | 0.307 m | 0.35: +1.00; 0.45: +0.24; 0.164: +0.52; 1.0: −2.00 |
| k22 | situation | `ego_now` | picture frame | (0.76, 0.08) | (0.77, 0.10): +0.82; (0.85, 0.10): +0.39; (1.00, 0.10): −0.30 |
| k32 | situation | `ego_mem` | speaker | (−2.32, 1.59) | (−2.35, 1.60): +0.81; (−2.20, 1.70): +0.58; (−2.90, 1.60): −0.50 |
| k38 | help | `way_to` | router | −162° | −162°: +1.00; 174°: +0.40; 150°: −0.37; 0°: −0.50; −90°: −1.00 |
| k50 | situation | `ego_mem` | floor lamp | (0.25, −4.82) | (0.25, −4.80): +0.96; (0.50, −4.80): +0.59; (0.25, −4.00): +0.15 |
| k56 | situation | `ego_mem` | green plate | (−0.32, 0.26) | (−0.32, 0.26): +1.00; (−0.45, 0.26): +0.10; (−0.60, 0.26): −0.45; (0.30, 0.26): −1.00 |
| k68 | help | `way_to` | a green box | `unknown` | `unknown`: +1.00; −90°: −1.00 |
| end | reuse | `object_where` | brown bag | (0.88, 4.42) | 0.08 m off: +0.78; 0.9 m off: +0.51; 1.2 m off: −0.10 |
| end | reuse | `object_where` | blue bowl | (3.18, 4.16) | 0.08 m off: +0.77; 0.9 m off: +0.51; 1.2 m off: −0.10 |
| end | reuse | `task_things` | things near the route | 1 item | [(2.8, 3.8)]: +0.72; [(3.1, 3.8)]: +0.53; []: −1.00 |

Offsets are (ahead, left) in metres from the camera; places are (x, y) on the map; headings are degrees to the left of
the way the robot faces. The 1.0 m answer to `clear_path` scores −2.00 because it is a catastrophe
([Scoring](#scoring)). The speaker and the floor lamp are far enough away that the margin is a tenth of their distance
(0.28 m and 0.48 m), not C = 0.15 m, so the same errors score more than they did in 0.10.0. The `way_to` at k68 names a
green box that nothing in the frames shows, so `unknown` is right and any heading is wrong. The case's headline is the
mean of its columns, as [Scoring](#scoring) says: `now` holds the distance and clearance rows the latest frame answers
(k18 and k22), and `memory` the rest.

## Jobs and places

| Job | Robot | Camera | Place | One task |
| --- | --- | --- | --- | --- |
| `vacuum/2` | robot vacuum, a 34 cm disc | 8 cm up, fixed, 5° up | `apartment_a`, a furnished apartment | cleans one or two rooms, or finishes a room it was sent home from |
| `helper/1` | Stretch-class home helper | 1.30 m up, tilts from 60° down to 20° up | `apartment_a` | carries a thing from a surface in one room to a surface in another, or brings one to its spot |
| `amr/1` | MiR250-class tote robot | 0.50 m up, 10° down | `warehouse_a`, pallet racks, shelving and packing tables | docks at tote shelves and packing tables, or runs an order-picking loop |
| `drone/1` | Wing-class tether delivery drone | under the body, tilts from 20° to 90° down | `downtown_a`, a downtown block | flies the streets at 15 m to a drop point, inspects it, descends to 7 m and lowers a package |

A **shift** is 2 to 6 tasks of one job in one world. Between tasks the robot is docked or parked and unobserved for 10
minutes to 3 hours, and the place changes: things are moved, doors open and shut, people come and go. No question asks
about that change ([Limits](#limits)). Each job has 16 shifts, 12 of 2-3 tasks and 4 of 4-6, rendered once in Blender.

A **case** is a window of a shift: `short` (1 task), `medium` (2 or 3) or `long` (4 to 6). It is delivered from step 0
on a fresh clock, so nothing tells the agent where the window sits in the shift. Each job has 36 cases (16 short, 16
medium, 4 long):

| Job | Place | Scripts | Cases (short / medium / long) | Questions | Frames |
| --- | --- | --- | --- | --- | --- |
| `vacuum/2` | `apartment_a` | 16 | 16 / 16 / 4 | 711 | 6,460 |
| `helper/1` | `apartment_a` | 16 | 16 / 16 / 4 | 541 | 3,157 |
| `amr/1` | `warehouse_a` | 16 | 16 / 16 / 4 | 245 | 4,679 |
| `drone/1` | `downtown_a` | 16 | 16 / 16 / 4 | 264 | 2,534 |

## How a case runs

**What the agent gets:**
- each frame's camera image, a 640 × 480 JPEG with an 80° horizontal field of view;
- one header line per frame: the step and clock, the base's position and heading in map metres, its own velocity (the
  controller's odometry), the camera's pitch, the robot's status (phase, task, battery and the like) and its subgoal, a
  sentence on what it is doing now;
- a message at the start of each task: the clock, the unobserved gap since the last task, and the assignment;
- the camera's size, field of view and mount, and the robot's body (`robot.json` and `camera.json`; the reference agent
  gets them in its system prompt, with the answer shapes and how answers are credited).

**What it never gets:** depth, labels, boxes, ids, a map of the place, a question before it is asked, any feedback on
an answer, or a frame from the future. It estimates every distance itself from the pixels.

**The order.**
1. Each task opens with its task message, then its frames one at a time, in time order. The agent asks for the next
   frame when it is done with the current one, and nothing advances until it does.
2. **During the job**, the agent is asked right after some frames. These look points are drawn from the case id and
   the plan's decision frames (which come first), never from the world, at least 4 steps into a task and 4 steps apart:
   about one point in 6 frames of a case. A point has up to three slots, harm first, and each slot draws one family from
   the job's table (below), so which question is asked reveals nothing about what is in view. A frame and its questions
   reach the reference agent together.
3. **After the job**, all the end questions come together.

The robot holds still while the agent thinks. Nothing carries over between cases.

**The questions.**

| Kind | Ability (column) | Family | When | It asks | Answer |
| --- | --- | --- | --- | --- | --- |
| Harm | Clearance (`now` or `memory`, by what answers it) | `clear_path` | during | how far the body could move before it touched anything: straight ahead, straight back, or after turning in place to the left or right (the drone: ahead and back) | `number` |
| Harm | Clearance (`memory`) | `closest_person` | end | how close a person came to the robot during a task, as it saw them | `number` |
| Harm | Clearance (`now` or `memory`, by what answers it) | `closest_approach` | during | how close the person the robot can see nearest it would come to it in the next 5 seconds if both kept moving as they are (AMR, helper) | `number` |
| Harm | Clearance (`memory`) | `approach_mem` | during | how close the person the robot saw last, now out of view, would come to it in the next 5 seconds if it kept moving as it is (AMR, helper; 0.11.0) | `number` |
| Harm | Clearance (`now` or `memory`, by what answers it) | `tangle_ahead` | during | the things on the floor within 1 m, ahead or to the sides, that the brush could catch (vacuum) | `positions` |
| Harm | Clearance (`now`) | `drop_zone_people` | during | the people within 4 m of the point below the drone (written; asked by no job, for too little supply) | `positions` |
| Help | Clearance (`now` or `memory`, by what answers it) | `open_heading` | during | which heading, turned in place, would let the body go farthest (counted since 0.11.0) | `heading` |
| Help | Memory (`memory`) | `way_to` | during | which way to set off to reach a thing seen earlier, by the shortest way the body fits through the floor it has seen (helper; 0.11.0) | `heading` |
| Situation | Distance (`now`) | `ego_now` | during | where the nearest visible point of a thing in view is, from the camera, along the floor | `offset` |
| Situation | Memory (`memory`) | `ego_mem` | during | the same for a thing seen earlier and not in view, measured from where the camera is now | `offset` |
| Reuse | Memory (`memory`) | `object_where` | end | where on the map it last saw a thing | `position` |
| Reuse | Relevance (`memory`) | `task_things` | end | where the things near its task were: on the floor within 1.2 m of the way it went (helper), within 2 m of the way it drove (AMR), the things on the floor of the rooms it cleaned that its brush could catch (vacuum; 0.11.0) | `positions` |

Each job asks the families its robot's decisions need:

| Job | During the job | After the job |
| --- | --- | --- |
| `vacuum/2` | `clear_path` (ahead, back, left), `tangle_ahead`, `ego_mem`, `open_heading` | `object_where`, `task_things` |
| `helper/1` | `clear_path` (ahead, back, left, right), `closest_approach`, `approach_mem`, `ego_mem`, `way_to`, `ego_now`, `open_heading` | `object_where`, `task_things` |
| `amr/1` | `closest_approach`, `approach_mem`, `ego_mem`, `ego_now` | `task_things`, `closest_person` |
| `drone/1` | `clear_path` (ahead, back), `ego_mem` | none |

The vacuum's `task_things` was dropped by rule in 0.9.0, when it failed a validity gate (see [Validity](#validity)), and
returns in 0.11.0 as the list of what its brush could catch, with the kinds that recur at one spot from shift to shift
(bath mats, clothes piles) counted neither way. The helper does not ask `person_mem`, where the person it last saw is: it
failed a validity gate (`separation`) on the first compile of 0.10.0, and a family that fails a gate is dropped, never
re-tuned. A look point
has up to three slots, harm first, and a case asks one question in about 9 to 25 frames. An answer is one JSON object:

```text
{"type": "number", "value": 0.31}                              metres: how far the body could move, how close a person came
{"type": "heading", "deg_left": 23}                            degrees to the left of the way the robot faces, -180 to 180
{"type": "offset", "ahead_m": 0.52, "left_m": 0.24}          metres ahead of the camera along the heading, and to its left
{"type": "position", "xy": [0.9, 4.5]}                         map metres
{"type": "positions", "items": [{"xy": [2.8, 3.8]}, ...]}      one item per thing; an empty list means none
{"type": "unknown"}                                            always allowed
```

A negative offset is behind the camera or to its right, and a thing's height is no part of it. The target of an offset
is what the frames show: the point of the outline that the thing's visible pixels make on the floor (their convex hull)
nearest the floor point under the camera. A place names the middle of the thing where it stood. About a fifth of the
`ego_mem` and `object_where` questions name a thing no frame showed, or a description that matches nothing in the
place. Their right answer is `unknown`, and their words are the same as the others'. A corridor through space the robot
never looked at is `unknown` too, and a guess costs the full penalty. A heading into space it never saw has a run of 0,
and a heading toward a remembered thing that has no room to move, or runs into floor never seen, is a wrong answer.

## Scoring

**Credit is set by what the job needs.** A distance from the camera, how far the body could move, how close a person
came and a heading are credited against the job's **clearance C**, the margin the robot's own operating spec keeps
beside its path. A place on the map is credited against the **re-find radius R**, within which the robot, driving back,
finds the thing again. A list is credited against its matching radius. Each value traces to a source of its kind, is
stated in the prompt, and has not moved since it was declared. The margin for a distance, an offset or a heading grows
with the distance judged: it is C, or a tenth of the right distance if that is more (0.11.0).

| Job | C | Source of C | R | Source of R |
| --- | --- | --- | --- | --- |
| `vacuum/2` | 0.10 m | the loose end of camera vacuums' stated passing distances | 1 m | HomeRobot's real-world find rule; Habitat ObjectNav's success radius |
| `helper/1` | 0.15 m | OWMM-Agent counts a pick within 0.15 m of the target | 1 m | as the vacuum |
| `amr/1` | 0.45 m | MiR250: a 1,450 mm corridor for a 580 mm robot | 1 m | a convention: no warehouse source gives one |
| `drone/1` | 2.4 m | Wing patent US12399508B2: a delivery point 8 ft from trees, 10 ft from buildings | 2 m | Wing patent US12221211B2: no more than about 2 m from the intended point |

**Four quarters.** An answer that is `e` off the gold, against the need N and the gold's resolution ρ (one pixel of the
camera at the geometry the gold rests on, at least 1 cm), earns a credit `s` that is the sum of four quarters:

| Quarter | It is earned by | Whole | Falls to nothing |
| --- | --- | --- | --- |
| Direction | the right side, the right order, blocked or clear | on the right side | a step: 0 or 1 |
| The right thing or action | being near | within 2N | at 4N |
| The right amount | being within what the job needs | within N | at 2N |
| Precision | being as exact as the camera can resolve | within ρ | at N, log-linearly |

Where one pixel is half the need or more, N is widened to 2ρ, and for a distance, an offset or a heading it is widened
to a tenth of the distance judged where that is larger: **N' = max(N, 0.1 g, 2ρ)**, with g the right distance (an
offset's range from the camera, a distance's gold value, a heading's best run). A place and a list item keep R and the
matching radius. The direction of an offset is a bearing within 45° of the gold's, or an error within N; of a distance
in metres, the same side of the body's length plus C (for a corridor, is there room to move at all) and, above it,
within a factor of two; of a place, within R, and of a list item, within the matching radius; of a heading, room to
move: a run of the body's length plus C or longer, or the best run less a pixel where no heading has that much. The
curve is continuous and strictly falling between ρ and 4N, and its one step is the direction quarter, so no answer
coarser than a pixel earns the whole. An answer with credit `s` scores `2s - 1`:

| The answer | Credit | Score |
| --- | --- | --- |
| within one pixel of the gold | 1 | +1 |
| at N | 3/4 | +0.5 |
| at 2N | 1/2 | 0, as good as `unknown` |
| at 4N or beyond, the direction right | 1/4 | −0.5 |
| beyond, the direction wrong | 0 | −1 |
| `unknown`, where the frames hold the answer | | 0 |
| `unknown`, where nothing the robot sensed answers | | +1 |
| anything else, where nothing the robot sensed answers | | −1 |

**Harm costs more than caution.** On the harmful side the need is halved (the tenth of the distance too, not the
two-pixel floor): a clearance answered longer than it is, a closest approach answered wider, a thing or a person placed
farther from the camera than it is, a heading toward too little room. A cautious error of 2N and a harmful one of N
leave the robot the same margin, so an over-estimate costs more than an under-estimate of the same size. A place to
re-find a thing has no harmful side.

**A catastrophe is charged on top, in the reported score only.** An answer whose action would end in harm, by a harmful
error larger than the whole margin (C, or two pixels if more), scores a flat negative whatever credit it earned. A
corridor said to be longer than it is, where the true one is shorter than the body plus C and the answer is not, would
drive the robot into something it saw: −2. A closest approach said to be clear when the person came inside that room,
or a cord the brush could catch within 30 cm of its front edge left out of a list, is a person or a hazard of the
robot's own kind: −3. `unknown` is never one. Every validity check reads the score without this term
([docs/question-design.md](docs/question-design.md), section 6).

**A set scores per item.** The answered places are paired one to one with the things by the largest summed credit, each
earning its own four quarters at its own resolution. A missed thing costs one and a false place half, and a place inside
a don't-care circle counts neither way. Each item is a place, so it earns its direction quarter within the matching
radius of the thing and nowhere else (0.11.0). Where there was nothing to list, the empty list is the one thing, and
each false place dilutes it: its credit is 1 over 1 plus half the false places. The list is perfect only when it holds
every thing and nothing else.

**A decision scores by its regret,** the run it gives up against the best heading, in metres, on the same rule with g the
best run. The way to a remembered thing (`way_to`) is scored the same way on the progress each heading makes: how much
nearer to the thing one metre of driving along it brings the robot, by the shortest way the body fits through the floor
it has seen, against the best heading's. A heading with no room to move, or into floor never seen, has no progress and
scores −1, so a wrong doorway costs its whole regret.

**Worked example.** In the case above the helper (C = 0.15 m) is asked how far its body could move straight ahead. The
gold is 0.307 m, and one pixel there is ρ = 4.7 cm. The body is 0.34 m long, so T = 0.34 + 0.15 = 0.49 m: the gold is
under T, which means there is no room to move at all.
- The answer 0.45 m is on the same side of T, so the direction quarter is 1. It is 14 cm too long, which is the harmful
  side, so N = 7.5 cm widens to 2ρ = 9.4 cm: the thing quarter is 1 (within 18.9 cm), the amount quarter 0.48 (between
  9.4 and 18.9 cm) and precision 0 (beyond 9.4 cm). The credit is (1 + 1 + 0.48 + 0) / 4 = 0.62 and the score +0.24.
- The answer 0.164 m is the same 14 cm off on the cautious side. N stays 15 cm, the amount quarter is 1 and precision
  0.04: the credit is 0.76 and the score +0.52.
- The answer 1.0 m says there is room to move: the wrong side of T, 0.69 m off. The credit is 0 and the gate score −1,
  and the answer would drive the robot into something it saw by more than the margin, so it is a catastrophe and the
  reported score is −2, where `unknown` would have scored 0.

**The far field.** The same helper is asked how far its body could move toward a door 5 m away. One pixel there is ρ =
13.7 cm, and the margin is not C = 15 cm but a tenth of the distance, 0.50 m.
- The answer 4.5 m is 0.5 m short: the thing and amount quarters are whole, precision is 0, the credit is 3/4 and the
  score +0.50.
- The answer 4.0 m is 1.0 m short, twice the margin: the amount quarter is gone, the credit is 1/2 and the score 0, as
  good as `unknown`. (In 0.10.0 the margin was two pixels, 27 cm, and this scored −0.42 on the helper, −0.06 on the AMR
  and −0.03 on the vacuum, by their cameras; it scores 0 on all three now.)
- The answer 6.0 m is 1.0 m too long, on the harmful side, where the margin is half as large (0.25 m, which two pixels,
  27 cm, widen): the credit is 0.29 and the score −0.42. An over-estimate costs more than an under-estimate.

**The headline** is the mean of two columns that count equally, whatever their numbers of questions. Every cell is
the mean of the reported score:
- **`now`**: the cells answered from the latest frame, distance (`ego_now`) and the clearance rows that frame
  answers by itself;
- **`memory`**: the cells that need what the robot saw before, memory (`ego_mem`, `object_where` and `way_to`, pooled),
  relevance (`task_things`) and the clearance rows that need memory (`closest_person`, `approach_mem`,
  `closest_approach` rows with a walking person, and the corridors and headings the latest frame does not answer).

Inside a column the cells, each an ability in that column pooled over its own questions, are averaged by their weights:
a cell with 20 questions or more on a job in the dataset weighs 1, so they count equally when every cell is that full,
and one with fewer weighs `n / 20` of a full cell (0.10.0 left it out of the mean). A column weighs the largest weight
of its cells, and the cells are the release's own for every score of a release.
`open_heading` is a clearance question and counts. Only the prediction ability, which no job asks, is a diagnostic.
There is no own-path column: the pose and status are given, so the robot's own path is a log lookup and is never asked.
Each place is scored on its own, and no number pools places. Intervals are 95% bootstraps over shifts. The reports also
split the questions by kind (harm, help, situation, reuse) and by the age of the evidence, and give the catastrophe
rate, the unpenalised mean credit, the share of answers in each credit band and the abstention bound: the score had the
agent said `unknown` exactly where its answer scored below 0.

## What makes a question fair

A question is asked only where a careful reader of the frames could answer it within the job's accuracy, and never
where the words or the place give it away. Every screen below removes questions; none adds one.

- **Recognisable.** A thing is asked about only from frames where a person could name it: at least 1,024 pixels, lit,
  clear of the image's edge. Its colour word must be the colour most of its frames show. A person may touch the top and
  the sides of the frame, because their feet place them.
- **No look-alikes.** A thing is not named while something it could be taken for, in its colour, is in sight: a bag
  and a box, a towel and folded clothes, a cable bundle and an extension cord, a bin and a drum, among others.
- **A glimpse is not a sighting.** A question about a thing the robot never saw (right answer `unknown`) is not asked
  when its frames glimpsed a look-alike kind in the asked colour, too small to tell: a person asked for the blue skip
  could not tell the glimpse from it.
- **The surface tie.** A reader with no depth places a point where its pixel ray meets a surface whose height it knows.
  So a thing is asked only where a frame shows it meeting the floor, a shelf or a table top, read to within C/2 at one
  pixel's error, or where it stands at the near edge of a support at or above the camera.
- **A corridor's stopper is tied and visible.** A corridor is asked only where the first thing the body would touch is
  tied to a floor contact a camera can read to within C/2, and is a wall, a fixed piece or a thing a person could name;
  not where an obstacle within 1 cm of the body's side, or a height within 5 mm of the body's band, decides the gold;
  and only while the robot drives.
- **Nothing given away.** Light furniture, whose place a learned layout gives, is never asked, nor is anything the
  robot's own log tells about, such as the thing it carries. No thing is asked twice in a case.
- **Gates.** A family ships on a job only when a perfect reference reaches the ceiling and every shortcut stays at the
  floor ([Validity](#validity)). One that fails is repaired as a broken instrument or dropped, never tuned to a score.

**Fixed structure is asked.** How far the body could move, and which way has the most room, depend partly on the
place's fixed structure: its walls, racks, buildings and furniture. The agent is never given a map, so it perceives and
remembers that structure like anything else. But the maps are public, and a system that holds a place's layout can answer
these questions without the sensors. The **map reader**, which answers each corridor and heading with the public map's
raycast from the delivered pose, is an internal design diagnostic in `stbench check` and appears in no results table:
how a system that holds a map scores is not what is tested.

## Results

The reference agent is the Claude Code CLI on a Claude Code login ([docs/claude.md](docs/claude.md); it is the agent of
[docs/codex.md](docs/codex.md) reached through another CLI) with two tools, `next_frame`, the robot's sensor feed, and
`submit_answer`, and no memory tool: its context is its memory. A case is one long session in which it pulls its frames
one at a time. Its prompt states the camera, the robot's body, C, R and how an answer is credited. The rows use the
camera channel (`rgb_pose`), the correct brief and the questions during the job. Three models ran, each once at medium
effort: `claude:haiku`, `claude:sonnet` and `claude:opus`. The floor, `all_unknown`, answers `unknown` to everything.
Each cell is the reported score (`2s - 1`, less a flat charge for a catastrophe); the headline has its 95% interval and
is the mean of the `now` and `memory` columns, which show the number of questions their counted cells rest on.

**The reference models changed from Codex to Claude in 0.11.0.** 0.10.0's rows were three Codex models (astra, sol and
luna). The harness runs both agents the same way, so 0.11.0's rows are three Claude models. The models, the questions,
the prompt and the scorer all differ, so the rows do not compare with 0.10.0's: those are in
[the report of 0.10.0](docs/history/report-0.10.0.md), and its answers scored again by 0.11.0's scorer are in the
[CHANGELOG](CHANGELOG.md).

**The rows cover 42 of the 124 short and medium cases that ask a question** (20 apartment cases, 11 warehouse, 11
downtown). Every system ran the same 42, the same cases as 0.10.0's rows, chosen then by the order the runs took them
and not by any outcome. The intervals are wide for it, and rows on every case are left for a later revision.
Coverage: `all_unknown`: 42 of 42 cases; `claude:haiku` (medium): 42 of 42 cases; `claude:sonnet` (medium): 42 of 42 cases; `claude:opus` (medium): 42 of 42 cases.

**These are dev rows, not a held-out confirmation.** They are scored on the 64 shifts the questions and screens were
refined on. Every repair made there fixed a broken instrument and applies to every job, but it was made on this data.

| Layout | System | Headline (95% interval) | `now` | `memory` | Cases scored |
| --- | --- | --- | --- | --- | --- |
| `apartment_a` | `claude:haiku` (medium) | +0.152 [+0.076, +0.250] | +0.120 [+0.012, +0.276] over 41 | +0.184 [+0.102, +0.275] over 215 | 20 |
| `apartment_a` | `claude:sonnet` (medium) | +0.286 [+0.159, +0.404] | +0.254 [+0.055, +0.435] over 41 | +0.317 [+0.210, +0.407] over 215 | 20 |
| `apartment_a` | `claude:opus` (medium) | +0.567 [+0.518, +0.642] | +0.589 [+0.521, +0.705] over 41 | +0.544 [+0.430, +0.663] over 215 | 20 |
| `apartment_a` | `all_unknown` | +0.034 [+0.019, +0.048] | +0.000 [+0.000, +0.000] over 41 | +0.067 [+0.037, +0.097] over 215 | 20 |
| `warehouse_a` | `claude:haiku` (medium) | +0.273 [+0.112, +0.440] | +0.420 [+0.235, +0.593] over 24 | +0.127 [-0.062, +0.325] over 40 | 11 |
| `warehouse_a` | `claude:sonnet` (medium) | +0.307 [+0.077, +0.499] | +0.360 [+0.028, +0.652] over 24 | +0.254 [-0.008, +0.486] over 40 | 11 |
| `warehouse_a` | `claude:opus` (medium) | +0.419 [+0.188, +0.638] | +0.501 [+0.276, +0.699] over 24 | +0.338 [-0.002, +0.686] over 40 | 11 |
| `warehouse_a` | `all_unknown` | +0.056 [+0.000, +0.125] | +0.000 [+0.000, +0.000] over 24 | +0.111 [+0.000, +0.250] over 40 | 11 |
| `downtown_a` | `claude:haiku` (medium) | +0.185 [+0.019, +0.325] | +0.109 [-0.079, +0.318] over 22 | +0.261 [+0.008, +0.467] over 46 | 11 |
| `downtown_a` | `claude:sonnet` (medium) | +0.366 [+0.213, +0.505] | +0.380 [+0.201, +0.586] over 22 | +0.351 [+0.111, +0.534] over 46 | 11 |
| `downtown_a` | `claude:opus` (medium) | +0.614 [+0.517, +0.717] | +0.596 [+0.463, +0.788] over 22 | +0.632 [+0.506, +0.781] over 46 | 11 |
| `downtown_a` | `all_unknown` | +0.132 [+0.078, +0.219] | +0.000 [+0.000, +0.000] over 22 | +0.263 [+0.156, +0.438] over 46 | 11 |

- The floor's `memory` is the mean, over the memory cells, of the share of each cell's questions whose right answer is
  `unknown`, which scores +1 for it. Its `now` is 0: every `now` question has an answer in view.
- The order is the same in every place: opus above sonnet above haiku above the floor. Sonnet lands at +0.29 to +0.37,
  inside the band of about 0.25 to 0.5 that a strong model should meet on realistically hard questions; opus at +0.42 to
  +0.61, with no counted cell at 1.00 (its highest is +0.71).
- Full credit is rare, as designed: opus earned it on 22% of the answerable questions, sonnet on 13% and haiku on 6%.
- The hardest questions are the helper's remembered corridors (every model below zero), the heading with room to
  turn into, and the AMR's list of things it passed. Downtown's corridors are bounded only by building walls, and opus
  scores +0.70 on the remembered ones; making the drone harder needs new data (people at the drop, obstacles at flight
  height), which this release does not render.

The [technical report](docs/report.md) gives each place's cells, the families, the case lengths, the kinds and the
cost. Each run's `results.md` adds the per-family rows, the credit bands, the age bands, the catastrophes, the
abstention bound, tokens, wall time and a notional cost at the price table.

## Validity

The rules are in [docs/vision.md](docs/vision.md) and [docs/methodology.md](docs/methodology.md); the gates and reports
are in [docs/eval.md](docs/eval.md) and [docs/questions.md](docs/questions.md).

On the 0.10.0 compile (1,701 questions), `stbench check` passed in every reading, with no instrument broken. On the
0.11.0 compile (1,761 questions) the free check of the release passes in every reading on every job: `ALL GATES PASS`
and `EVIDENCE VERDICT PASSES`, per job and in full. The largest lead a shortcut has over `all_unknown` is +0.04 on
`open_heading` for the helper and +0.00 for the vacuum, +0.03 on `way_to`, +0.01 on the helper's `task_things` and on
`tangle_ahead`, and +0.00 on `approach_mem`; the gate's line is +0.10. The checks are:

- **Every gold is computed by code** from the world timeline and what the camera recorded, and names the frames it
  rests on. Nothing the robot did not see changes a gold.
- **A perfect reference scores 1.00.** A table built from the oracle channel, which gives each detection's floor
  outline (`floor_hull`), reproduces every gold: 1.00 on every family and job.
- **The shortcuts sit at the floor.** Answers learned from other shifts, the robot's own path log, another case's
  stream, the questions' wording, and for memory the current view alone: none beats `all_unknown` by more than its gate
  allows, and `stbench check` prints each gate's value.
- **No constant beats `unknown`.** The compiler draws answer classes so that no fixed answer scores above
  `all_unknown` on a family, and `stbench check` confirms it ([docs/balance.md](docs/balance.md)).
- **The stricter readings pass too.** Every gate passes on its point value and on its 95% interval, as well as on the
  evidence reading the export requires.

`stbench check --data data --verdict evidence --out check.json` reproduces the gate table on a downloaded dataset.

## Limits

- **Dev rows, no held-out split.** The rows are scored on the shifts the questions were refined on, and no fresh shifts
  were rendered to confirm them. The gold is public and each place is one layout, so a result can be overfit. Publish
  your agent's code with its result.
- **Change over time is not tested.** No question asks what moved, appeared or went since an earlier pass, or where a
  moving person is going. The world changes only between tasks, so the short cases hold no change; in the medium cases
  the robot sees both sides of only 11 to 28 of the 90 to 165 changes a job's 16 medium cases hold, whether a thing
  moved is largely given by its kind, and balancing that leaves a fair change question 4 questions in the whole release.
  The fix is new data (scripts that revisit places after the world changes, changes drawn per thing by a coin rather
  than by kind, more people walking), which needs new renders, so it is left for a later version.
  [docs/report.md](docs/report.md) has the measurement.
- **Help is new and thin.** `open_heading` counts since 0.11.0, on the one need rule: its direction is room to move, and
  the robot's own log alone scores +0.04 on the helper and −0.14 on the vacuum under it (over 0.10.0's rows). Through
  0.10.0 its credit was a cliff and it counted in no mean. `way_to` is the helper's alone: 29 questions over its 36
  cases, every answerable one a turn of at least 30 degrees. The rows' 42 cases hold few of either, so their intervals are
  wide.
- **The structure questions are public.** The maps are public, so a system that holds a place's layout can answer the
  corridor and heading questions without the sensors. The map reader that does so is an internal design
  diagnostic in `stbench check`, and appears in no results table.
- **Thin cells.** A cell with fewer than 20 questions on a job weighs `n / 20` of a full cell. On the 0.11.0 compile no
  cell is thin: the least, the AMR's clearance `now` and the vacuum's relevance cell, hold 22 questions each.
- **The rows cover 42 of the 124 short and medium cases** (above), so their intervals are wide, and each model ran once.
  The one-frame pull re-reads the whole session at every frame, so a case reads millions of input tokens; the report's
  cost table has the means, with the cached share.
- **The leak check was not rerun at full size.** Whether asking during the job changes the end answers was last read
  at full size on 0.9.0's questions (inconclusive: d +0.043, interval −0.030 to +0.118, with the vacuum's own reading
  above the margin). A subset reading on 0.11.0's compile (`claude:sonnet`, during off against on, the 15 fast-tier
  cases with an end question) gives d +0.123 [-0.018, +0.347] over 33 pairs of 9 scripts: too thin to read, and a leak
  is not excluded. The report reads the questions with an answerable gold for its verdict and the ones whose right
  answer is `unknown` as an abstention shift, apart.
- **`unknown` on a glimpse.** Some wrong answers to an `unknown` question land on a thing seen only in glimpses too
  small to show what it is. The prompt now says that a question may name a thing never seen or not there, and a screen
  drops a question whose frames glimpsed a look-alike in the asked colour; no row was run without the sentence, so its
  effect alone is not measured, and each run's `results.md` carries the abstention report (recall, precision, and
  `unknown` on answerable questions).
- **Memory is tested at 1-3 tasks, not full shifts.** The rows cover short and medium cases, minutes of driving. The 16
  long cases (4-6 tasks) have no row, and real shifts last hours.
- **One place of each kind:** one apartment, one warehouse, one downtown. Nothing here says how an agent does in a place
  it has never seen.
- **A floor-texture mark.** The apartment's floor texture has a cable-like mark, which a Codex model twice took for a cable bundle
  in the 0.7.0 pilot. Removing it needs a re-render.
- **Not measured:** whether the score predicts control (no closed-loop check yet).

## Quickstart

**Install** (Python 3.12 or later). The package runs without Blender, a GPU or any further download.

```sh
pip install "spatiotemporalbench @ git+https://github.com/DanMcInerney/SpatiotemporalBench" huggingface_hub
```

The package is not on PyPI yet; once it is, `pip install "spatiotemporalbench>=0.11"` installs it by name. From a
checkout, run `uv sync --extra dev` and write `uv run stbench ...` for `stbench ...` below.

**Get the data**, about 2 GB: the frames, the cases, and the questions and gold.

```sh
hf download DanMcInerney/spatiotemporalbench --repo-type dataset --revision v0.11.0 --local-dir data
```

**Run the free reference** over the short and medium cases, in about a minute: the floor.

```sh
stbench eval all_unknown --data data --select length=short --select length=medium --out runs/all_unknown
```

**Run the reference agent.** It needs the `codex` CLI on your PATH and `codex login` with a ChatGPT plan: no API key and
no dollars, only the plan's quota. Keep about 20 or fewer Codex cases running at once on one login. A run is resumable:
the same command skips finished cases.

```sh
stbench eval codex:gpt-6.1-sol --data data --select length=short --select length=medium --reasoning medium --workers 9 --out runs/sol
```

**Or run it on Claude Code**, as 0.11.0's reference rows were (`--reasoning medium`). The same agent, the same options
and the same files, through the `claude` CLI on your own Claude Code login (the subscription's quota, or an API key):
`haiku`, `sonnet`, `opus` or a full `claude-...` id, with `--reasoning` as `--effort`. Switching between the two is the
prefix of the system name ([docs/claude.md](docs/claude.md)).

```sh
stbench eval claude:sonnet --data data --select length=short --select length=medium --reasoning medium --workers 3 --out runs/cc-sonnet
```

**Plug in your own agent.** An agent is one Python class; the harness feeds it the case and asks it the questions.
`examples/minimal_agent` (in the repository) logs what it is sent and answers `unknown`; `examples/template_agent` shows
every message and answer shape; `examples/claude_code_agent` drives the Claude Code CLI. The protocol, in short:

```python
import json

from system_shim import System  # the harness provides it when it runs your file


class MyAgent(System):
    def observe(self, steps):
        """One frame: a list of one step. It has k, clock, pose {pos, rot, yaw_deg}, camera {pos, rot, pitch_deg},
        velocity [vx, vy] (the base's own, m/s), status, subgoal, and rgb, a JPEG path under self.workdir that is gone
        after the next frame. Returning asks for the next frame."""
        with (self.state_dir / "steps.jsonl").open("a", encoding="utf-8") as f:
            for step in steps:
                f.write(json.dumps({"k": step["k"], "base": step["pose"]["pos"]}) + "\n")

    def answer(self, question):
        """{qid, text, answer_type} -> one answer of that type, or unknown."""
        return {"type": "unknown"}
```

```sh
stbench run examples/minimal_agent/agent.py:MinimalAgent data/private/0b29b8dc4f4d5002 --local
stbench eval my_agent/agent.py:MyAgent --data data --local --select length=short --workers 4
```

`--local` runs your agent as a plain subprocess, stamped `sandbox: none` and scored as unverified. Without it the agent
runs in the Docker sandbox, which is the verified mode. Your agent may keep any memory it likes: only its answers count,
and what it spent is reported beside them.

**Look at the results.** Each run folder holds `results.md` (one section per place) and `results.json`.

```sh
stbench eval compare runs/all_unknown runs/sol
stbench view runs/sol --data data
```

`eval compare` lays runs side by side with paired differences and refuses runs that differ in more than one lever.
`view` writes one HTML page with every case: the questions, the answers, the gold, the frames as they were streamed,
and the agent's transcript. `eval distance` breaks the offset answers down by error against C and by range.

## Documents

- [docs/vision.md](docs/vision.md): the goals and the rules. [docs/methodology.md](docs/methodology.md): how the rules
  become questions, scores and checks.
- [docs/question-design.md](docs/question-design.md): what is asked, when, how it is graded and why.
  [docs/questions.md](docs/questions.md), [docs/families/](docs/families/): the questions and how each gold is
  computed. [docs/balance.md](docs/balance.md): the balance rule.
  [docs/categories.md](docs/categories.md): the six question categories, and how a new job or place is authored from them.
- [docs/agent.md](docs/agent.md): your own agent. [docs/codex.md](docs/codex.md): the reference agent.
  [docs/claude.md](docs/claude.md): the same agent on Claude Code.
  [docs/eval.md](docs/eval.md): `eval`, `compare`, `distance`, `view`. [docs/format.md](docs/format.md): the files.
- [docs/jobs.md](docs/jobs.md), [job-helper](docs/job-helper.md), [job-amr](docs/job-amr.md),
  [job-drone](docs/job-drone.md): the jobs. [docs/world.md](docs/world.md), [docs/people.md](docs/people.md): what
  changes between tasks, and the people.
- [docs/report.md](docs/report.md): the technical report. [docs/history/report-0.10.0.md](docs/history/report-0.10.0.md),
  [docs/history/report-0.9.0.md](docs/history/report-0.9.0.md) and
  [docs/history/report-0.7.0.md](docs/history/report-0.7.0.md): the reports of 0.10.0, 0.9.0 and 0.7.0, kept for
  comparison. [CHANGELOG.md](CHANGELOG.md): each version's changes.

## Licences

- **Code:** MIT ([LICENSE](LICENSE)). `render/` runs inside Blender's Python and is GPL-3.0-or-later
  ([render/LICENSE](render/LICENSE)).
- **Data:** CC BY 4.0. Third-party 3D assets are CC0 or CC BY only, listed with their licences in
  `src/stbench/assets/manifest.json` and credited in the dataset's `CREDITS.md`. The frames made with the GPL-3.0
  renderer are part of the CC BY 4.0 dataset, and so is the frame in `docs/figures/`.

## Citation

Cite the version you used from [CITATION.cff](CITATION.cff); GitHub's "Cite this repository" button turns it into
BibTeX or APA.

## Contributing

Report a problem, a wrong gold or a result through [GitHub
issues](https://github.com/DanMcInerney/SpatiotemporalBench/issues); for a result, attach its `results.json` and your
agent's code. Before a pull request, run the free checks (model-free, no downloads, under 3 minutes) and the linter:

```sh
uv run pytest -q -n auto
uv run ruff check src tests render examples
```

A new robot, job or place is a profile and a module, not a special case: see [docs/jobs.md](docs/jobs.md).
