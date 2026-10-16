# Question design

What the benchmark asks, when it asks, how it grades an answer, and why. It is for a researcher or engineer who will run
the benchmark or add a question. The rules are in [methodology.md](methodology.md), the compiler's mechanics and the
record shapes in [questions.md](questions.md), and each family has a page in [families/](families/). This page is the
reasoning behind them.

The questions of 0.10.0 were redesigned on 2026-10-14, to build out the questions and to document them and the reasoning
behind them. The design briefs and the measurements behind this page are the maintainers' development records and are
not in this repository; the page carries what they decided and the numbers that decided it. The review (an independent
read of the first version of the design) is cited by the labels of its findings: B1 to B4 are the blockers and S1 to S7
the should-fixes. Every measured number names the compile it was measured on. The redesign was followed by lane J2's
validity fixes to `closest_approach` and the decision that drops `person_mem` from the helper. Constants and rules are
named so that each claim can be checked in the code.

0.11.0 follows the requests of 2026-10-15 and 2026-10-16: score on a graded curve, and ask questions that are a little
harder but still the robot's own decisions, without a new render. It changes how every distance is credited (one need
rule, section 6), counts the heading question (`open_heading`), asks the helper a route question (`way_to`) and the
vacuum a list again, and adds one sentence to the prompt about things never seen. The changes are in sections 1, 2, 3,
5, 6 and 8. A second pass (2026-10-16) scores a list item as a place and the empty list on a curve (section 6), asks the
AMR and the helper `approach_mem`, the closest approach to the person last seen (sections 2 and 5), and records why the
drone gets no new family on this data (section 3).

## 1. What a question is

A question is a decision the robot's brain faces. The agent is asked at the latest frame it holds, or later from what it
has built up since. It is never asked about a thing for its own sake: "how close would the person nearest you come in
the next 5 seconds?" is a question, "what colour is the mug?" is not.

**Four kinds** say what the answer is for (`abilities.FAMILY_KIND`; [vision.md](vision.md)):

| Kind | The answer is for | Families |
| --- | --- | --- |
| Harm | what an action now would hit, tangle on or endanger | `clear_path`, `closest_person`, `closest_approach`, `drop_zone_people`, `tangle_ahead` (`person_mem` is filed here too; no job asks it) |
| Help | which action moves the task forward | `open_heading` (counted in clearance since 0.11.0), `way_to` (the helper's; counted in memory; 0.11.0) |
| Situation | what is around the robot, in view and out of view | `ego_now`, `ego_mem` |
| Reuse | what is worth keeping to do the job here again | `object_where`, `task_things` |

Harm is asked first at every look point (section 5), and it is the one kind whose worst errors carry a catastrophe
weight (section 6). The kind is a split of the reports and no cell of the headline (methodology rule 4): the two help
families sit in two abilities, `open_heading` in clearance and `way_to` in memory, and each ability keeps its own cell.

**Four evidence ages** say how old the evidence is. The scope (`now` or `memory`) says only whether the latest frame
answers the question. Each gold row also records `age_s`, the clock seconds from the gold's oldest evidence keyframe to
the ask, and `age_band` (`questions_age`; `records.AGE_BANDS`):

| Band | When | Example |
| --- | --- | --- |
| `now` | the scope is `now`: the latest frame alone gives the gold | the nearest person in view, and where they are |
| `recent` | `memory`, and `age_s` is at most 10 s (`questions_age.RECENT_S`) | a cable the robot drove past 6 s ago, now beside or behind it |
| `task` | `memory`, older than 10 s, with the evidence in the task the question falls in | the person seen in the hall 40 s ago |
| `earlier` | `memory`, with the evidence in an earlier task | a thing last seen before an unobserved gap |

Both fields are null on an `unknown` row, which names no evidence. The reports give the reported score by band, overall
and for each job (`by.age_band`, `job_age_band`): a memory curve, credit against how long ago the robot saw the answer.
The headline is unchanged, the mean of the `now` and `memory` columns, so the bands refine the `memory` column and do
not reweigh it. The rows by band on the dev data (all gold rows of all lengths, on the compile after the final review's
repair pass):

| Job | `now` | `recent` | `task` | `earlier` | null |
| --- | --- | --- | --- | --- | --- |
| AMR | 127 | 13 | 58 | 27 | 10 |
| Drone | 96 | 44 | 58 | 34 | 32 |
| Helper | 105 | 53 | 217 | 76 | 62 |
| Vacuum | 135 | 187 | 287 | 54 | 26 |

The repair moves the AMR's `now` count up by 16 and its `recent` count down by 16: a `closest_approach` row whose person
stands is `now` again (the robot's own velocity is delivered with every step, so frame k alone gives its gold), and only
the two rows with a walking person (the AMR's) stay `memory` (section 5 and the family page). A `now` row's `age_s` is
0: before the repair a `now` row aged from the evidence frame it named, which gave 35 `open_heading` and 9 `clear_path`
`now` rows an age above 0 (final review S6). The helper's counts move with its draw, which no longer puts
`before_place` first.

**Now and memory questions are mixed in one stream.** A controller needs both all the time, and the realistic memory
question comes at a decision: "before you reverse, what is behind you?". A memory question asked while the job goes on
tests memory kept online; held to the end, it would let an agent review everything at once. With one frame per message
`now` is exact: it is the frame the agent holds. The price of mixing is cueing, since a question that names a thing
makes it salient. So no subject is asked twice in a case (the recent view is the subject of the harm families), wording
comes from fixed templates, and the `--during off` arm of a run measures how much asking during the job changes later
answers ([eval.md](eval.md), "Arms").

## 2. The jobs: what the brain needs, and what is asked

For each robot the table says how it can do harm, what it needs to make progress, and what it should remember to do the
job again, and then what the benchmark asks of each. The families are in the order of a look point's slots (section 5).
The answer shapes, credits and golds are on the family pages.

### Vacuum (`vacuum/2`, the apartment, camera 8 cm up)

| | The brain needs | Asked as |
| --- | --- | --- |
| Harm | not to drive into walls or legs; not to wind a cable, cloth or towel round the brush; not to wedge under low furniture | the corridor ahead, back and left (`clear_path`; the body is swept at its own height, so low furniture that touches it stops the run); what the brush could catch within 1 m (`tangle_ahead`) |
| Progress | which way is open; what is not cleaned yet; the way back to the dock | which heading gives the body the most room (`open_heading`, the help kind) |
| Reuse | where clutter collects; which doors were shut; what it left unfinished | where a thing it passed lies now (`ego_mem`, things on the floor only); at the end, where it last saw a named thing (`object_where`) and, since 0.11.0, where the things its brush could catch lay in the rooms it cleaned (`task_things`) |

| Family (kind) | An example, as the agent reads it | Why the vacuum is asked |
| --- | --- | --- |
| `clear_path` (harm) | "If you reversed straight back now, how far could your body go before it touched anything? Answer from what you have seen, in metres from the back of your body; answer 3 if nothing is within 3 m." | It backs out of corners and turns toward walls it cannot see. Ahead is often in view; back and left are remembered space |
| `tangle_ahead` (harm) | "List anything lying on the floor within 1 m of you, ahead or to your sides, that your brush could catch (a cable bundle, an extension cord, a power strip or a towel; a bath mat, a clothes pile or a pedestal mat counts neither way). Answer the middle of each in map coordinates, or an empty list if there is nothing." | A cord wound round the brush stops the robot until a person frees it. The sides were seen frames ago, so the answer is usually memory |
| `ego_mem` (situation) | "As you last saw it, where is the point of the floor fan you saw that is nearest to your camera now, measured along the floor? Answer in metres ahead of the camera and to its left." | A thing it drove past is a thing to steer round next time. Only things on the floor are asked: a mug on a table is no obstacle to it |
| `open_heading` (help) | "If you had to turn in place and drive straight away from here, which way, of those you have seen, would let your body go farthest before it touched anything? Answer the heading in degrees to the left of the way you face now, from -180 to 180 (to your right is negative)." | The way out of a tight spot. Counted since 0.11.0, credited as room to move (section 6) |
| `object_where` (reuse, at the end) | "Where did you last see the red bag? Answer in map coordinates." | Where things are kept, to find them again |
| `task_things` (reuse, at the end; 0.11.0) | "Where were the things on the floor of the rooms you cleaned that your brush could catch (a cable bundle, an extension cord, a power strip or a towel; a bath mat, a clothes pile or a pedestal mat counts neither way) in your first task, which you would have to steer around next time? Answer in map coordinates. [...]" | A cord left on the floor stops the next clean, so the list is what to steer around next time. Cords and cloth are the items; the mats and the clothes pile that recur at one spot from script to script are don't-care, and the question says both in the same words as `tangle_ahead` ([families/task_things.md](families/task_things.md)) |

### Helper (`helper/1`, the apartment, a person at home)

| | The brain needs | Asked as |
| --- | --- | --- |
| Harm | not to bump the person; not to knock things off surfaces | the person nearest it, in the next 5 s (`closest_approach`) and, since 0.11.0, the person it saw last, once out of view (`approach_mem`); how far the body can go ahead, back and turned in place (`clear_path`; the directions asked are on [families/clear_path.md](families/clear_path.md)). Where a person it last saw is, once out of view (`person_mem`), is not asked: section 3; how close that person would come to it is (`approach_mem`) |
| Progress | where the thing to fetch is and how to reach it; whether the place to put it down is free | which way to set off for a thing it saw before (`way_to`, 0.11.0) and which heading gives the body the most room (`open_heading`) |
| Reuse | where things are kept; what the person moves; which surfaces are usually clear | where things it passed lie (`ego_mem`, `ego_now`); at the end, where it last saw a thing (`object_where`) and the things near its way (`task_things`) |

| Family (kind) | An example, as the agent reads it | Why the helper is asked |
| --- | --- | --- |
| `clear_path` (harm) | "If you turned in place to face your left and then drove straight, how far could your body go before it touched anything? [...]" | It backs away from a counter and turns beside furniture, in space the head camera has left behind |
| `closest_approach` (harm) | "If you kept moving as you are now, how close would the person you can see nearest you come to you in the next 5 seconds? Answer the smallest distance in metres along the floor from the point below your camera to the person; answer 10 if no one would come within 10 m." | A helper beside a resident must stop an arm swing or a reverse before it reaches them. Details on [families/closest_approach.md](families/closest_approach.md) |
| `approach_mem` (harm, 0.11.0) | "If you kept moving as you are now, how close would you come in the next 5 seconds to the person you saw last, where you last saw them? Answer the smallest distance in metres along the floor from the point below your camera to that person; answer 10 if you would not come within 10 m." | The resident sits out of view behind a helper that reverses or turns. It keeps the last-seen person as an obstacle and must stop an arm swing or a reverse before it reaches them, whether or not they are in view. It needs memory of a person, a thing in no layout, composed with its own motion. Details on [families/approach_mem.md](families/approach_mem.md) |
| `ego_now`, `ego_mem` (situation) | "Where is the point of the grey remote you can see now that is nearest to your camera, measured along the floor? [...]" and the same "As you last saw it, [...]" | Where things are around it, in view and out of it |
| `way_to` (help, 0.11.0) | "If you had to go now to the router you saw, which way would you set off to get there by the shortest way your body fits through, of the ways you have seen? Answer the heading in degrees to the left of the way you face now, from -180 to 180 (to your right is negative)." | A person says "bring me my glasses": the helper must recall where it saw them and which doorway to take. It is the help kind's core decision, the way to the goal, and a wrong doorway costs the whole regret ([families/way_to.md](families/way_to.md)) |
| `open_heading` (help) | as the vacuum's | Which way leaves room to turn and leave |
| `object_where` (reuse, at the end) | "Where did you last see the grey remote? Answer in map coordinates." | Where things are kept |
| `task_things` (reuse, at the end) | "Where were the things on the floor within 1.2 m of the way you went to fetch and put down the thing in your first task, which you would have to steer around next time? Answer in map coordinates. [...]" | What lay along its route, as a list to avoid next time |

### AMR (`amr/1`, the warehouse)

| | The brain needs | Asked as |
| --- | --- | --- |
| Harm | not to hit people, above all at blind aisle ends; not to hit pallets that jut into an aisle | the person nearest it, in the next 5 s (`closest_approach`) and, since 0.11.0, the person it saw last, once out of view (`approach_mem`); at the end, how close a person came in a task (`closest_person`) |
| Progress | which aisle is clear; where the pick shelf and the packing table are; whether the table is free | nothing is asked as help (below) |
| Reuse | which aisles get blocked; where pallets are staged; where people work | where floor things it passed lie (`ego_now`, `ego_mem`); at the end, the things near its way (`task_things`) |

| Family (kind) | An example, as the agent reads it | Why the AMR is asked |
| --- | --- | --- |
| `closest_approach` (harm) | as the helper's, above | Its planner slows or yields when a person would come within its protective field in the next few seconds (ISO 3691-4 asks it to stop before contact) |
| `approach_mem` (harm, 0.11.0) | as the helper's, above | At a blind aisle end the AMR saw a picker 20 s ago, now hidden by the rack. Its planner keeps the last-seen person as an obstacle and slows if its course passes within its protective field (ISO 3691-4 asks it to stop before contact). `closest_approach` finds someone in view at only about 10% of the AMR's keyframes, while at 41% a person seen in the last 15 frames, within 10 m, is out of view now; this asks about them |
| `closest_person` (harm, at the end) | "During your first task, how close did a person come to you, as you saw them? Answer the smallest distance in metres along the floor from the point below your camera to the nearest point of a person; answer 10 if you saw no one within 10 m." | The near-miss record a fleet keeps. One a case |
| `ego_now`, `ego_mem` (situation) | "Where is the point of the blue tote you can see now that is nearest to your camera, measured along the floor? [...]" | Only things on the floor are asked: stock held off the floor on racks is no obstacle to it |
| `task_things` (reuse, at the end) | "Where were the things within 2 m of the way you drove in your first task, which you would have to steer around next time? Answer in map coordinates. [...]" | What lay along its route |

### Drone (`drone/1`, downtown, 7 to 15 m up)

| | The brain needs | Asked as |
| --- | --- | --- |
| Harm | not to hit a wall or a building edge at its altitude; not to lower onto a person | the corridor ahead and back (`clear_path`, asked only where the base is at least `JobSpec.fly_floor_m` = 8.35 m up); who stands within 4 m of the point below it is `drop_zone_people`, which no job asks since the final review's S2 |
| Progress | which heading leads to the drop; whether the drop zone is clear | unasked: whether people stand in the drop zone (`drop_zone_people`, below) |
| Reuse | where the obstacles on its routes are; which zones are usually occupied | where a thing it saw before is, from the camera now (`ego_mem`); no end question |

| Family (kind) | An example, as the agent reads it | Why the drone is asked |
| --- | --- | --- |
| `clear_path` (harm) | "If you flew straight ahead now at this height, how far could your body go before it touched anything you have seen? Answer in metres from the edge of your body." | A wall is what it can hit. Downtown's corridors are bounded only by static building walls, which the camera sees (every movable thing there is under the flight height), so this question measures seeing big static structure, and a score here is a claim only about a system with no copy of the map (methodology rule 5) |
| `drop_zone_people` (harm; written, asked by no job) | "Before you lower the package: where are the people within 4 m of the point below you? Answer their positions in map coordinates, or an empty list if there are none." | It must not lower a package onto a person. It would be asked in the phases `inspect`, `descend` and `lower`, and only when the 4 m disc is wholly in the frame; the drone asked it until the final review (2 rows over 36 dev cases, under `abilities.MIN_ABILITY_N`), and its code and page stay for a dataset with the supply |
| `ego_mem` (situation) | "As you last saw it, where is the point of the white van you saw that is nearest to your camera now, measured along the floor? [...]" | What it saw on the way down is what it works from on the way back |

## 3. What is deliberately not asked

Each item is a question the design considered and left out, with the reason and the rule that decided it (valid, simple,
realistic, fast: methodology §3). Items are from the 0.10.0 design's list of what it left out, its review (findings B1
to B4 and S1 to S7), the brainstorm's list and, for the last block of rows, the 0.11.0 design, and for the drone's rows
and the people questions of the AMR and the helper, its second pass.

| Not asked | Why |
| --- | --- |
| Questions at the start of a task, before its first frame | They would ask the robot to plan from memory, but only the vacuum has a task kind that carries on earlier work (`JobSpec.continues`) and there are too few such tasks; the blockers on a route to the docking line are a layout prior (valid) |
| The task's own target ("where is the tote you were sent for?") | The brief and the subgoal name it, so `World.told` excludes it from every question (review S3) |
| A blocked bay's or doorway's blocker | The robot's own log tells it: the response (skip, abort) and the subgoal follow the coin, so `World.told` keeps every question from naming it ([questions.md](questions.md), "How a gold is computed") |
| Change sets: "what was moved, added or removed here since last time" | `changed` failed on layout priors: 77% of the vacuum's golds were the empty set, which the current view and the question's form guess, and it left every job (valid) |
| A free spot to put a thing down | The subgoal and the robot's response tell it |
| Answers with uncertainty (a plus-or-minus radius, scored by an interval score) | Left out of this build. `unknown` stays the one way to say that the sensors did not show the answer. An interval answer would add an answer type and a second credit, and no question needs it yet (simple) |
| Weighting harm in the headline | The kind is a split of the reports and no cell of the headline (methodology rule 4); a harm weight would make it one. Harm costs more row by row instead, through the harmful side and the catastrophe term (section 6) |
| The drone's `closest_approach` | A floor distance from a camera 15 m up is not the drone's harm decision (review B1) |
| `drop_zone_people` at all, for now | Its supply is too low: 2 rows over the 36 dev drone cases, inside the drone's counted clearance `now` cell with no per-family diagnostic and no readable gate, so the final review's S2 takes it off the drone (section 8; review B1 and B2). The family, its gold and its page stay for a dataset with more supply |
| Where a person last seen is, once out of view (`person_mem`, the helper's) | It failed `separation` on the helper's compile after lane J2: 7 of its 19 questions were answerable and 12 had an `unknown` gold, which a blind `unknown` scores 1 on, so a perfect reader led by +0.368 against the gate's 0.4. It is dropped by the evidence rule and not re-tuned (valid). A resident in view is `closest_approach`'s. An open gap for when the answerable supply allows ([families/person_mem.md](families/person_mem.md), "What would bring it back") |
| A person's motion as a separate question | People are mostly still (the residents stand about 97% of the time), so such a question would mostly be answered "none". Motion is part of `closest_approach`, whose page has the numbers |
| The AMR's corridors | 62% of the golds the budget allows are the 5 m cap, and a held-out constant scores +0.33 ([questions.md](questions.md), "What a case asks") |
| The vacuum's people, and the vacuum's right turn | No person is placeable from the 8 cm camera; the vacuum follows edges with the wall on its right, so a held-out constant leads by +0.28 |
| What is left to clean (`uncovered`) | The 8 cm camera never certifies a 0.5 m cell wholly free and unswept |
| Which way to go, to a place by name (the layout's topology) | The help decisions that rest on the sensors are asked: the way with the most room (`open_heading`) and, since 0.11.0, the way to a thing the robot saw (`way_to`, helper only). A route between fixed named places is the same in every case of a place, so `blind` learns it; methodology §1 and rule 5 keep it for a held-out place (valid) |
| `open_heading` for the AMR and the drone | Every best run of the AMR is the cap, and the open high street is the longest run from every point of the drone's square, a layout fact ([questions.md](questions.md), "What a case asks") |
| Decision frames of the vacuum's `before_backout` and the AMR's `before_undock` | There is no supply: the vacuum has no back-out site (`job_vacuum2_backout.SITE_KINDS = ()`) and the AMR's `before_undock` is a dead band (it declares no `clear_phases`; review S2) |
| Retargeting by `JobSpec.relevance` | It emptied cells (the vacuum's `ego_mem` fell from 36 rows to 2, the helper's `ego_now` from 31 to 6; review S3). The floor robots are filtered by `World.on_floor` instead |
| Bath mats, pedestal mats and clothes piles as tangle items | They recur at the same spots from script to script, a layout prior (review S4). They are don't-care circles ([families/tangle_ahead.md](families/tangle_ahead.md)), and since 0.11.0 the same in the vacuum's `task_things` list, which is what brought that list back |
| Blind corners, unfinished work, what stays and what moves | In the brainstorm and not built. The order proposed was: motion and closest approach, the robots' own hazards, target and progress questions, blind corners, then change. (The way to the goal was built in 0.11.0 as `way_to`.) |
| `way_to` for the vacuum, the AMR and the drone | The vacuum has no fetch decision, and its way home is a layout fact its own log often gives. The AMR is dispatched by a fleet manager that holds a map, its floor holds few nameable things (its `ego_mem` fill is 0.026), and it drives the shortest way to its current station, so its own motion answers; its way back to a station it left is "turn back", which its own log gives. The drone flies over the streets, so the way to a ground thing is mostly its bearing, which `ego_mem` asks (realistic; valid) |
| "Which doorway leads toward X" as its own family | Folded into `way_to`. A room's doorways are a small discrete set, so a separate question would be a multiple choice in disguise, which the vision rules out. `way_to`'s credit already makes the doorway the decision: a wrong doorway costs the whole regret (simple; valid) |
| The offset between two remembered things | Not built. It is no decision (the vision: never a thing for its own sake). Its evidence is two `object_where` answers subtracted, so its only added hardness is error compounding, which is stingier scoring in disguise (valid) |
| A tighter tolerance or a smaller need, to make questions harder | Ruled out (2026-10-15: questions are not made arbitrarily hard). The last quarter is already precision the job does not need, and "credit an answer more strictly than the job needs" is a "do not" (methodology §4) |
| Longer gaps, to make memory matter | Evidence age past 10 s made no row harder on 0.10.0's answers (astra, reported score on the 42 rows: `now` +0.65, `recent` +0.64, `task` +0.49, `earlier` +0.62), and "make cases longer or stranger than real jobs" is a "do not" (methodology §4) |
| `approach_mem` for the drone and the vacuum | At altitude a person is not the drone's contact hazard, and at its drop the people stand at layout spots (below). The vacuum's 8 cm camera places no person. The AMR and the helper ask it (section 2) |
| `person_mem` (an offset to the last-seen person) for the AMR | It is `ego_mem` for people, which strong models already score highly (the AMR's `ego_mem`: astra +0.55, sol +0.79). `approach_mem` adds the own-motion step that makes it a decision |
| A list of where people worked, for the AMR or the drone | People's dwell places recur from script to script (the warehouse's 574 within 2 m of another script's, 472 within 0.5 m; downtown's 639 of 643 within 2 m), so such a list is a layout prior, the helper's list failure built in |
| The drone: `approach_mem` at the drop ("how close to the point below you is the person you saw last") | The drop's phases (inspect, descend, lower) hold 741 keyframes, a person within 10 m at 371 and seen at 276, but every such person is near 3 of the drop zones and dwell places recur. The proxy has 61 rows, and the constant 6 m scores +0.417 and a pose-keyed pool +0.567. C = 2.4 m against golds of 0 to 10 m leaves about 4 needs of range. It fails `blind` and balance by construction (valid) |
| The drone: the drop clearance ("how far from the point below you is the nearest thing you have seen", at `before_lower`) | 51 decisions on 16 scripts. All 23 aborts have golds of 0 to 0.3 m (the coin's blocker) and the go golds run 0 to 7.8 m, so the constant 0.5 m scores +0.341. The blocker is what the response then tells (`World.told`) (valid) |
| The drone: a people list at the drop (`drop_zone_people` at 10 m, with memory) | The same dwell recurrence; under the place rule (section 6) a blind dense place within 2 m earns the direction quarter. A layout prior (valid) |
| The drone: `way_to` or `open_heading` | The drone's way depends on the map alone (`job_drone.py`): flights to a drop repeat poses across scripts, so `path_log`'s pose-keyed levels hold the same answer, and the open high street is the longest run from the square. A layout fact; `ego_mem` already asks the bearing (valid) |
| The drone: vehicles near the drop (`task_things`) | 0.6.0: the floor-surface centres scored +0.25 to +0.40, because vehicles stand at parking and loading slots, a slot prior (valid) |
| The drone: change between flights | 12 of 27 carried things move between flights, but in 16 medium cases the drone sees both sides of 11 and can name 3: no supply |
| Choosing or dropping questions by a model's pass rate | Never (methodology §4: tune nothing to what one system fails). A family leaves a job only for failing a gate |

**Why the drone gets no new family on this data.** The agent is never given a map, and downtown's corridors are bounded
only by static building walls, because every movable thing is under the flight height, so `clear_path` measures seeing
big static structure. `ego_mem` is the drone's memory of ground things. A new drone family must be about something that
changes or is remembered over time, and each candidate above fails a validity test on this data: people near the drop
points stand at layout spots, a remembered person at the drop scores +0.42 with a constant, a drop clearance +0.34, and
flight routes are fixed by the map, so a pose-keyed pool learns any route or heading answer. Downtown stays near its
ceiling, and its score is reported under methodology rule 10, not tuned. What would change it is a data round with
renders: people placed by a coin around each drop point rather than at fixed dwell places, vehicles drawn per flight
rather than per slot, and more walkers on its streets. `approach_mem` and `drop_zone_people` would then be the drone's
families as written.

## 4. How a question is worded

The rules for wording come from the brainstorm and are kept in the family templates:
1. State the decision in a few words ("If you reversed straight back now, [...]"). Never hint at the answer.
2. Ask for the quantity the controller acts on: how far, which way, where, when, which.
3. Name only what the task names: "a person", "anything". The question names no entity and no id.
4. Answer from the camera for the `now` and recent questions, and in map coordinates for questions about places and
   lists.
5. Put the units, the frame and the cap ("answer 10 if no one would come within 10 m") in the question.
6. Make a question hard through the world and the precision asked, never through the wording. The world supplies the low
   camera, clutter, occlusion, motion, gaps and change.

`text` is the family's render of its slots and nothing else (`FAMILIES[template].render`), so the wording cannot carry a
gold ([questions.md](questions.md), "The families").

## 5. When questions are asked

**Look points.** A look family (`JobSpec.looks`) is asked right after a frame, and the frame can be any frame of the
case (`questions_looks`):
- the candidates are the frames at least `LOOK_MIN_STEPS` = 4 steps into their task, except the task's last frame;
- the frames of the decisions the job's task turns on come first in the draw (`JobSpec.look_decisions`, below), each
  group in `stable_hash(case id, k, "ask")` order;
- a frame is kept when it is at least `LOOK_GAP_STEPS` = 4 frames from every frame kept, until the case holds
  `ceil(frames / LOOK_PER_FRAMES)` points, with `LOOK_PER_FRAMES` = 6;
- the draw reads the case id and the plan, never the world, a gold or an answer.

| Job | Decision frames drawn first |
| --- | --- |
| AMR | `before_crossing` (0.5 m before the way crosses into an aisle), `before_station` (at the station, facing it) |
| Helper | `before_grasp`, `before_doorway` (`before_place` is not: below) |
| Vacuum | `before_doorway`, `room_done` (the last cleaning frame of a room) |
| Drone | `before_lower` (the last frame of the descent), `before_descent` (the last frame of the inspect hover) |

A question being asked can therefore show that a frame is one of the robot's decision moments. The robot's own status
and subgoal already show that, and the plan decides where those moments fall. For a family whose answer depends on the
world at that moment that is harmless; for a **structure** family (`clear_path`, `open_heading`) it was a channel: the
final review found that a decision frame puts a structure question at the same plan pose, by the same furniture, in
every script. 44 of the helper's 106 `clear_path` rows sat at decision frames (9 of the vacuum's 315), the carrier gold
(`right` 0.56 m, 10 rows in 5 scripts) had 8 of them at `before_place`, and the helper's evidence gate split by the
review read +0.211 (answerable +0.405 [0.181, 0.672]) on decision frames alone, +0.000 on the other frames and passed
without `before_place` (+0.010 [-0.072, 0.111]). So `before_place` leaves the helper's `look_decisions` (repair R1). The
second form of the repair, no decision frame drawn first for a structure family, is kept for a structure family that still
fails. No look family is mapped to a decision
(`questionnaire.check_decisions`): a decision frame only changes where points fall.

**Three slots a point, harm first.** A point asks one question a slot of the job's slot table (`JobSpec.look_shares`).
Slot `i` draws a family by `stable_hash(case id, k, f"slot{i}") % 100` against the slot's cumulative percents; a point
whose drawn family finds no candidate is a dead band, and nothing replaces it. Slot 1 is the job's harm questions, slot
2 is what the robot saw before, slot 3 is the rest. No score enters a share: each job's file states the share and the
measured fill it rests on.

| Job | Slot 1 | Slot 2 | Slot 3 |
| --- | --- | --- | --- |
| AMR | `closest_approach` 100 | `ego_mem` 100 | `ego_now` 50, `approach_mem` 50 |
| Drone | `clear_path` 60 | `ego_mem` 70 | (none: `drop_zone_people` left, below) |
| Helper | `clear_path` 50, `closest_approach` 50 | `ego_mem` 65, `way_to` 35 | `ego_now` 20, `open_heading` 15, `approach_mem` 30 |
| Vacuum | `clear_path` 50, `tangle_ahead` 50 | `ego_mem` 100 | `open_heading` 20 |

The drone's `drop_zone_people` held slot 3 whole because its fill was so low that any part of a slot would give a
handful of rows; it drew 2 rows over the 36 dev cases, under `abilities.MIN_ABILITY_N`, in the drone's counted clearance
`now` cell with no per-family diagnostic, so the final review's S2 took it off the drone, and the drone has two slots.
Nothing else was rebalanced. The helper's slot 1 splits evenly because half the slot is the least that gives
`closest_approach`, the family of the lowest fill, its 20 rows over the 36 dev cases (800 points at 0.5 x 0.063 = 25);
the corridor has the rows to spare at the same half. Its slot 2 was `ego_mem` at 25 and `person_mem` at 40 until
`person_mem` was dropped (section 3), and `ego_mem` took the 40 (its fill, 0.512, is high, and capped by the distinct
things a case holds). In 0.11.0 the 35 that slot 2 then drew nothing on goes to `way_to`, the route to a thing the
helper saw, so `ego_mem` keeps its 65 and its supply; `way_to` is asked only in the helper's driving phases
(`JobSpec.look_phases`: `navigating`, `carrying`, as the corridor), and a point that draws it where the robot is not
driving, or where no remembered thing has a route through floor seen, is a dead band. Slot 3 is sized so that the three
slots give about 0.5 questions a point. In 0.11.0 it also carries `approach_mem` (`closest_approach` for the person out of
view): the helper's slot 3 drew nothing at 65% of its points and gives it 30 of them, and the AMR's `ego_now` gives it
half of the slot, which moves no cell's mean because `ego_now` is its most-asked family and keeps rows to spare. The hazard families are asked only in the status phases the job declares for
them (`JobSpec.look_phases`): the vacuum's `cleaning` and `transit` (the drone's `inspect`, `descend` and `lower` left
with `drop_zone_people`). The corridor is asked only in the phases the job declares (`JobSpec.clear_phases`: a ground
robot's driving phases, every phase of the drone), since elsewhere the status answers it.

**Why there are more questions.** The request of 2026-10-14 was for more questions per case, and the measured reason is
that frames are the fixed cost of a case. 0.9.0 asked 550 questions over 16,678 frames of its short and medium cases,
one in about 30. A short case streams a median 71 frames (10th and 90th percentiles 43 and 119)
and a medium case 168 (94 and 280), and 0.9.0 asked a median of 4 questions a case (10th and 90th percentiles 2 and 9).
A pulled frame costs the reference agent about 1 to 3 s: 2 s a frame for sol on 12 frames, a median 3.1 s over a
120-frame turn, and a median 2.25 s (90th percentile 3.0 s) on a real case. A question costs more, since the model
thinks about an answer: a 0.9.0 ask with one question took a median 30.1 s for sol at medium effort, 34.9 s for luna and
51.6 s for astra at xhigh, of which about 12 to 13 s is a Codex turn's floor that pulling removes, leaving about 18, 23
and 39 s. At one question in 30 frames a case paid for its whole stream to ask a handful of questions.

[vision.md](vision.md) "Speed" used to say that harder questions "never add questions or time to a case". That sentence
is amended: a case may ask more questions, because the frames are paid for anyway. The target was at least twice 0.9.0's
density, one question in 15 frames or fewer, aiming at one in 10.

**Measured density.** Short and medium cases, on the compile after the final review's repair pass (repair R1;
every job's row is from that compile): the look questions only, then all questions with the end ones. The 0.9.0
column was measured on the 0.9.0 compile. The harm share is of all the job's
questions of every length. The repair moved two rows: the drone lost `drop_zone_people` (1 in 12.6 became 1 in 12.7) and
the helper lost `before_place` from its decision frames and `person_mem` from its slot 2 (1 in 12.5 and 9.3 became 1 in
11.5 and 8.7):

| Job | 0.9.0, all questions | Look questions | All questions | Harm share of all |
| --- | --- | --- | --- | --- |
| AMR | 1 in 35.8 frames | 1 in 37.5 | 1 in 24.7 | 25% |
| Drone | 1 in 29.8 | 1 in 12.7 | 1 in 12.7 | 74% |
| Helper | 1 in 14.1 | 1 in 11.5 | 1 in 8.7 | 23% |
| Vacuum | 1 in 69.2 | 1 in 14.0 | 1 in 13.3 | 65% |

Points run one in 5.9 frames at `LOOK_PER_FRAMES` 6, one in 7.8 at 8; the 4-frame gap caps them near one in 5.1. Three
jobs meet the one-in-15 target. Harm holding half of the questions (a goal of the brainstorm) is out of reach by supply
on the AMR and the helper (review B3).

**Fill per drawn slot** is the questions asked over the slots that drew the family, all lengths, on the same compile:

| Job | Family: fill |
| --- | --- |
| AMR | `closest_approach` 0.023, `ego_mem` 0.026, `ego_now` 0.101 (0.10.0's; 0.11.0's `approach_mem`: 0.094, 47 rows over 502 drawn slots) |
| Drone | `clear_path` 0.566, `ego_mem` 0.161 (`drop_zone_people` 0.003, since dropped) |
| Helper | `clear_path` 0.258, `closest_approach` 0.051, `ego_mem` 0.313, `ego_now` 0.338, `open_heading` 0.491 (0.10.0's; 0.11.0's `approach_mem`: 0.078, 19 rows over 244 drawn slots) |
| Vacuum | `clear_path` 0.429, `tangle_ahead` 0.169, `ego_mem` 0.055, `open_heading` 0.457 |

(These are 0.10.0's. In 0.11.0 the helper's `way_to` asked 25 questions in the short and medium cases from 208 slots that
drew it, a fill of 0.12 against the 0.25 the design expected: the end questions reserve their things first, a thing is
named once a case, and a thing needs a route through floor the robot has seen. It asked 29 in all lengths.)

**The AMR is supply-limited.** Every one of its slots is whole (100), so a share cannot lift it. At whole slots most
draws of `closest_approach` find no person in range, a few more are lost to the subject `closest_person` reserves for
its task ([families/closest_approach.md](families/closest_approach.md) has the counts), and `ego_mem` found only things
an earlier look had already named in 401 of its 1,043 draws, since the floor holds few things. Lane J2's repair of
`closest_approach` took the density down a little more (one question in 24.4 frames became one in 24.7): a frame with two
people in it, or with a person it shows and does not place, is now a dead band, because the sensors do not pair their
sightings across frames.

**A frame and its questions arrive together.** For the reference agent a frame and its questions are one item, the reply
to one `next_frame` call: the frame's header line, its image, then the questions about it ([codex.md](codex.md), "How a
run goes"). The model looks and decides in one round trip, where a separate item costs it one pull more for every
question, and it is the controller's real moment. The harness holds frame `k` until its next message shows whether
questions come with it, and the frame's deadline starts when it is served. The protocol for any other agent keeps
`observe` then `ask`. A real AMR short case on this path, on the 0.10.0 draft: 5 of 5 answered, a median 3.1 s a plain frame, a
median 10.4 s (90th percentile 42 s, the end list) for a frame with questions, 417 s in all.

**One question for each ask.** Two questions at one point are two asks in turn. The first rides with its frame and the
next is sent after the first is answered, so a later question's wording cannot inform an earlier answer. It costs one
pull, about 2 to 3 s.

**End questions** come together in one ask after the last frame (`questionnaire.end`): `object_where`, `task_things` and
`closest_person`, up to 12, 18 or 20 for a short, medium or long case. The vacuum's ends are `object_where` and
`task_things` (0.11.0), the latter once for each cleaning task of the window. A thing in the case's last `LOOK_WINDOW`
frames (`World.end_window`) is not asked about at the end: the recent view would answer it.

## 6. How answers are graded

**Four quarters.** An answer with error `e` against the need `N'` (below) and the gold's resolution `rho` earns `s =
(direction + thing + amount + full) / 4`. The right direction earns a quarter, the right thing a quarter that is whole
within `2N'` and gone at `4N'`, the right amount one whole within `N'` and gone at `2N'`, and precision one whole within
one pixel and gone at `N'`. The arithmetic is `credit_kernel` and [questions.md](questions.md), "Credit". An answer
scores `2s - 1`, so a right-direction answer `4N'` away scores -0.5.

**One need for every distance (0.11.0).** The job's need `N` is the clearance C, the re-find radius R or a list's match
radius M. The need an answer is scored on is

    N' = max(N, 0.1 g, 2 rho)

where `g` is the distance the answer is judged over: a distance answer's gold value (`clear_path`, `closest_person`,
`closest_approach`), an offset's ground range from the camera (`ego_now`, `ego_mem`) and a heading's best run
(`open_heading`; the re-find radius for `way_to`). A place keeps R and a list item M, with no distance term. The first
two terms are halved on the harmful side (below); `2 rho` is not. The constant is `credit_kernel.KAPPA`.

- Within ten needs, the near field where harm happens, the tenth is under N and nothing changes: 1.0 m on the vacuum,
  1.5 m on the helper, 4.5 m on the AMR and 24 m on the drone. Beyond that the margin is a tenth of the distance, the
  same for every robot.
- Before 0.11.0 the far-field margin was `2 rho`, two pixels of the robot's own camera at that range, so the same 20%
  error was worth -0.42 on one robot and +0.60 on another. Now the ground robots agree.
- A tenth is the largest relative slack that keeps every constant answer at the floor and the models apart; a fifth
  starts to lift the AMR's constants. The form, an absolute floor and a relative term, is `numpy.isclose`'s
  `max(atol, rtol |b|)`. The field's own distance metrics are looser (SpatialRGPT counts an answer within 25% a
  success). Full credit is still only at the camera's pixel.
- **It keeps the catastrophe term as built** (below), and adds no scoring part: the harmful side and that term already
  make an over-estimate worse.

**The door, scored.** A door 5 m away, answered 4 m, scores 0.00 on every ground robot, as good as `unknown`. These
are the cases for each job, a gold of 5 m. For the vacuum, helper and drone it is a corridor (`clear_path`, a number);
the AMR asks no corridor, so for it the gold is a thing 5 m straight ahead (`ego_now`, an offset). One pixel at 5 m is
each job's median `rho / range` times 5: 0.24 m (vacuum), 0.14 m (helper), 0.11 m (AMR) and 0.03 m (drone). The
answers above 5 m are the harmful side.

| Answer | Vacuum 0.10.0 | Vacuum 0.11 | Helper 0.10.0 | Helper 0.11 | AMR 0.10.0 | AMR 0.11 | Drone 0.10.0 | Drone 0.11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.0 (20% over) | -0.03 | -0.03 | -0.42 | -0.42 | -0.50 | -0.50 | +0.53 | +0.53 |
| 5.5 (10% over) | +0.47 | +0.47 | +0.08 | +0.08 | -0.06 | +0.00 | +0.62 | +0.62 |
| 5.0 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 |
| 4.75 | +0.96 | +0.96 | +0.56 | +0.77 | +0.71 | +0.73 | +0.77 | +0.77 |
| 4.5 (10% under) | +0.47 | +0.50 | +0.08 | +0.50 | +0.44 | +0.50 | +0.68 | +0.68 |
| 4.0 (20% under) | -0.03 | **+0.00** | -0.42 | **+0.00** | -0.06 | **+0.00** | +0.60 | +0.60 |
| 3.0 | -0.50 | -0.50 | -0.50 | -0.50 | -0.50 | -0.50 | +0.02 | +0.02 |
| 2.5 | -0.50 | -0.50 | -0.50 | -0.50 | -0.50 | -0.50 | -0.02 | -0.02 |

How to read it:
- **On the cautious side the three ground robots now agree:** +0.50 at 10% under, 0.00 at 20% under and -0.50, the
  direction quarter alone, from 40% under. The helper's 4.0 m was -0.42 under 0.10.0, because the margin was its camera's
  two pixels (0.27 m) and not a tenth of the distance (0.50 m).
- **Over-estimates cost more,** because the harmful side halves the distance term: 6.0 m scores -0.42 on the helper and
  -0.50 on the AMR, against 0.00 for 4.0 m.
- **The vacuum barely changes.** Its 8 cm camera resolves 0.24 m at 5 m, so two pixels (0.47 m) already decide its
  need on both sides.
- **The drone keeps its 2.4 m clearance,** which dominates to 24 m; a 1 m error at 5 m is inside what the job keeps
  anyway. At 3.0 m the direction quarter is lost: its room to move is T = 1.3 + 2.4 = 3.7 m, so 3.0 m says there is no
  room where there was.
- `tests/test_credit_far_field.py` pins the helper's 4.0, 4.5 and 6.0 and an AMR offset 5 m away by hand.

The drone case, "overestimating the distance to the next building is worse than underestimating":

| Drone corridor | Answer | 0.10.0 | 0.11 |
| --- | --- | --- | --- |
| a building 30 m away | 27 (10% under) | +0.38 | +0.50 |
| | 33 (10% over) | -0.12 | +0.00 |
| | 36 (20% over) | -0.50 | -0.50 |
| a building 3 m away (under T) | 2.5 | +0.66 | +0.66 |
| | 4.0 | +0.02 | +0.02 |
| | 5.5 | -2 (catastrophe) | -2 (catastrophe) |

**What the rule does to the stored answers** (0.10.0's answers to the 42 cases the reference rows cover, scored again
under 0.11's scorer with no model call; the headline of each place, the mean of its `now` and `memory` columns):

| System | `apartment_a` 0.10.0 | 0.11 scorer | `warehouse_a` 0.10.0 | 0.11 scorer | `downtown_a` 0.10.0 | 0.11 scorer |
| --- | --- | --- | --- | --- | --- | --- |
| astra | +0.638 | +0.615 | +0.598 | +0.602 | +0.768 | +0.790 |
| sol | +0.461 | +0.449 | +0.554 | +0.558 | +0.593 | +0.615 |
| luna | -0.091 | -0.045 | +0.308 | +0.321 | +0.008 | +0.050 |
| `all_unknown` | +0.032 | +0.031 | +0.083 | +0.083 | +0.132 | +0.132 |

The order astra > sol > luna holds in every place and the gap from astra to sol is about where it was. The apartment
moves by two changes in opposite directions: the tenth lifts sol and astra by about 0.03, and counting `open_heading`
lowers them by 0.04 to 0.05. These are the 0.10.0 questions and prompt. 0.11.0's own reference rows, on its questions and prompt, are Claude models
([report.md](report.md), Section 4).

**Considered and not taken.**

| Alternative | Why not |
| --- | --- |
| A mean relative accuracy over thresholds (VSI-Bench) | It breaks "full credit only at the pixel": astra's full-credit share would go from 0.34 to 0.61, and it saturates `ego_now` and `closest_person` |
| A smooth curve (exponential, Cauchy) | Not anchored at the need, it scores an answer at the need at or below abstaining; anchored there, it raises every model by 0.07 to 0.14 and lets a constant answer through on the AMR |
| A fifth instead of a tenth | It lifts the AMR's constants toward the floor and narrows the gap between sol and luna |
| Keeping the camera-dependent need | The same 20% error would be worth -0.42 or +0.60 by robot |
| An asymmetric loss (LINEX, a pinball loss) | An extra parameter with units; the halved harmful side is already asymmetric |

**What could prove it wrong:** a gate that fails under the rule in the full free check (the design measured the constant
answers and the map reader, not the learned shortcuts), or a family whose best constant beats `all_unknown` on fresh
data. The response would be `KAPPA` = 0.05, which changes no headline measurably; it is never a constant per
family.

**The harmful side.** Where the error would put the robot in harm (a corridor said longer than it is, a person said to
be farther than they are, a thing placed farther from the camera than it is, a heading without room to move) the need's
first two terms are halved (`credit_kernel.HARM_FACTOR`), and the two-pixel floor is not. A cautious error of 2N and a
harmful one of N leave the robot the same margin, and an over-estimate costs more than an under-estimate of the same
size.

**A heading is credited as room to move (0.11.0).** `open_heading` asks which way, of those seen, gives the body the
most room. Its credit is the same kernel on the regret `e`, the best run less the answered heading's run, with `g` the
best run:
- **the direction is room to move:** a run of T (the body's length plus C) or longer, or the best run less a pixel where
  no heading has that much; the harmful side is the rest, a run under T;
- through 0.10.0 the direction was a run of half the best, and the credit was a cliff (a heading 2 to 10 degrees off a
  doorway scored -0.4 to -1; [families/open_heading.md](families/open_heading.md) keeps that account as history).

The design measured the shortcut this has to avoid: scored over the 54 helper and 134 vacuum heading rows of 0.10.0, the
robot's own log (the heading of the longest straight run it drove) scores +0.04 on the helper and -0.14 on the vacuum
here, and +0.38 on the helper under a credit that paid every heading with room linearly in that room, which is why that
credit was not used. The three models score -0.33 (luna), +0.03 (sol) and +0.16 (astra) on the 38 heading rows among
the 42 reference cases, and the best constant heading scores -0.40 on the helper and -0.52 on the vacuum.

**The route credit (`way_to`, 0.11.0).** The answered heading has an outcome in the row's `profile_m`: how much nearer
one R of driving along it brings the robot to the thing, through floor it has seen. The credit is the kernel on the
regret against the best heading's outcome, with N = C and `rho` the field's 0.1 m cell. A heading with no room to move
(a run under T) or into floor never seen has no outcome; it has the best outcome plus R as its error, no direction and
the harmful side, which scores -1. On open floor full credit is within about 26 degrees of the best way, +0.5 at about
37 degrees, 0 at about 53 degrees and -0.5 past about 78 degrees (the design's estimate from the credit's shape, not a
measurement); a wrong doorway costs its whole regret ([families/way_to.md](families/way_to.md)).

**A list item is a place, and the empty list is graded (0.11.0).** A list scores per item (methodology rule 7): the
answered places are matched one to one to the gold's within four match radii, and each pair earns the four quarters at
its gold item's resolution. Two rules changed, and the pairing reach (four radii), the other three quarters and the
don't-care circles did not:
- **A list item is a place.** A pair earns its direction quarter within the match radius M of the gold item and nowhere
  else, as a place does within R (the "place's direction is its radius" of rule 7). Through 0.10.0 every pair earned it
  at any distance up to four radii, the one place answer outside that rule. A pair 1.0 m off with M 0.5 m and a 2 cm
  resolution now earns 0.25 (the thing quarter alone) where it earned 0.5. Why it matters: the free quarter paid a
  sensorless prior for the layout. Every item of the helper's `task_things` lies in one 2.5 m by 1.9 m patch of the living
  room that the helper's way always crosses, and the two densest places of that patch have 47 of the list's 78 item rows
  within M. Under the graded empty rule, a blind list of those two places scores +0.226 over the 47 golds with the free
  quarter and +0.083 without it; on the item golds alone, +0.153 and -0.019.
- **The empty gold's one thing is the empty list:** `s = 1 / (1 + FP_WEIGHT |false|)`. One false place earns 2/3, three
  0.4 and `unknown` 0; through 0.10.0 any false place earned 0, as ten did. A cord placed 0.7 m and 0.9 m from a thing the
  question counts neither way scored -1 twice for Opus on the apartment. That cliff had also hidden the free quarter:
  the helper's blind places on its empty golds cost one or two false places, about 0 under the cliff and +0.58 under the
  graded rule.

The two go in together because the graded empty rule alone fails the helper's `task_things` gates once it stops masking
the prior: on the design's offline replay of the check's own pools, `blind_in_pool` reads +0.143 (the line is 0.10),
`current_view_prior` +0.075 (the line is 0.05) and `form_in_pool` +0.141. With the place rule they read -0.124, 0.000 and
0.000. The other list families (the vacuum's and the AMR's `task_things`, `tangle_ahead`) stay under their gate lines in the
same replay. Scored again with no model call, astra, sol and luna move by -0.02 to +0.04 on the 42 reference cases (astra
+0.615, +0.602, +0.790 becomes +0.621, +0.581, +0.790 on the apartment, warehouse and downtown; sol +0.449, +0.558,
+0.615 becomes +0.450, +0.537, +0.615), and the order holds everywhere. The scoped checks on the 0.11.0 compile give the list families' shortcuts as
`task_things` +0.01 on the helper and +0.00 on the vacuum and the AMR, `tangle_ahead` +0.01, against the line of 0.10, with `memory_needed` +0.00, ceiling 1.00 and separation 1.00 on each. If a list family fails a gate after the change, that family is dropped from that
job by the evidence rule and nothing else changes: no M, no threshold, no weight.

Alternatives not taken: a per-kind spot test generalised to every list marks no helper kind (none of 22 items recurs at
0.1 m, 2 of 22 at 0.5 m), a per-instance test marks 19 of 22 and makes one case's gold depend on which other scripts are
in the dataset, a false place weighing 1 on an empty gold still fails (`current_view_prior` +0.058 against 0.05) and
breaks the one `FP_WEIGHT`, and changing the helper's predicate would change the question when the defect is in the
credit.

**`unknown`.** It scores 0 on a question that has an answer, 1 where the sensors never showed it and -1 for any answer
there. A confident wrong answer costs more than `unknown`. The prompt says so in a sentence added in 0.11.0: a question may name a thing the camera never showed clearly enough to tell what it is, or one that is not
in this place at all, and then `unknown` is right. The sentence is true by construction (an answerable named thing has a
recognisable sighting, an `unknown` one has none in the window or does not exist) and names no question. One screen goes
with it (S4, [families/ego_mem.md](families/ego_mem.md)): an `unknown` candidate is dropped when the window glimpsed a
look-alike kind in the asked colour. It costs 6 of the 95 `unknown` rows. The results report gives abstention recall and
precision and the rate of `unknown` on answerable questions, per model and family, with Wilson intervals, so that
over-abstention is measured and not assumed away. On 0.10.0's answers (42 cases; recall is the share of `unknown`-gold
questions answered `unknown`, precision the share of `unknown` answers on an `unknown` gold) sol's recall was 0.84 and
precision 0.31, astra's 0.78 and 0.48 and luna's 0.47 and 0.28, and the three said `unknown` on 17%, 8% and 11% of the
answerable questions.

**Two scores.** The gates and the reports read different scores (`score`, review B4):
- `score.penalised` is `2s - 1`, with `unknown` as above. It is the **gate score**: every gate, `knowable` and the
  evidence verdict read it, as the row's `score_gate`.
- `score.reported` is `penalised`, except for a catastrophe, which scores minus its weight whatever credit it earned. It
  is the row's `score`, and the reports, the headline and the reference rows read it.

No weight ever reaches a validity verdict. A gate asks whether a shortcut beats abstaining; a catastrophe weight inside
that score lowers every shortcut that guesses, so a family could pass its gates because of the weight and not because
the sensors are needed. The review measured it: with the weight inside, the helper's `clear_path` map reader would have
gone from +0.279 to +0.049. `stbench check`'s verdicts are identical with the term in the reported score.

**A catastrophe** is an answer whose action would end in harm. It needs a harmful error larger than `margin_m` = `max(C,
2 rho)`: an error inside the clearance the job keeps anyway, or inside what two pixels of the camera resolve, is
imprecision and never one. `unknown` is never a catastrophe, nor is an answer of another type than the gold's. The rule
is on the gold row, `tolerances["catastrophe"]`, and the scorer reads it (`score_catastrophe`); the gold never judges an
answer. Rules and weights by family:

| Family | Rule | Weight | An answer is a catastrophe when |
| --- | --- | --- | --- |
| `clear_path` | `over` with `threshold_m` = body length + C | 2 | the gold is under the threshold (there was no room to move) and the answer is at or over it, by more than the margin: the robot would drive or fly into something it had seen. An overstatement of a corridor with room to spare is a wrong answer and no collision (a drone corridor's median gold is 24 m against a 2.4 m margin: final review S5; it was a flat -2 for any overstatement past the margin). A gold at the question's cap is exempt: nothing nearer was seen |
| `closest_person`, `closest_approach`, `approach_mem` (0.11.0) | `over` with `threshold_m` = body length + C | 3 | the gold is under the threshold (a person came inside the body's room) and the answer is at or over it, by more than the margin |
| `person_mem` (no job asks it) | `radial` | 3 | it places the person farther from the camera than they are, along the gold's bearing, by more than the margin |
| `drop_zone_people` (no job asks it) | `miss` | 3 | it leaves out a person whose hull lies within C of the point below the drone |
| `tangle_ahead` | `miss` | 3 | it leaves out a thing whose hull lies within `hazard_rule.FRONT_M` = 0.3 m of the body's front edge |

**The trigger stays body length plus C.** A stopping-distance trigger would need a reaction time and a braking
deceleration for each robot, and none has a primary source. For the ground robots at their job speeds the stopping
distance is about T anyway (inferred), and on the drone it would fire rarely. Every constant states its source (section
9), so it stays as built; a source for each robot's braking that put the stopping distance well above T would make it a
later candidate.

**The weights** are declared once, in `credit_kernel`, with their reason, and are never tuned to a score. 2
(`CATASTROPHE_THING`) is contact with a thing the robot saw: it damages or stops the job, and a person clears it. 3
(`CATASTROPHE_PERSON`) is contact with a person or a hazard of the robot's own kind, a tangle: it endangers someone or
ends the job until a person intervenes. `FRONT_M` is 1 s of travel at the vacuum's 0.3 m/s. The reference agent's prompt
states both weights, built from the same constants (`llm_credit`), the threshold of the corridor's (an overstatement
counts when the true corridor is under the body's length plus C, which the prompt calls "the room the body needs" and
words "the robot body's length plus 15 cm" on the helper, never "plus the margin": the far field's tenth does not widen
it; and the answer is not) and, for a job that asks
a list family during the job, the omission that costs 3: a list with no place within a matching radius of a cord within
`FRONT_M` of the vacuum's front edge (final review S7; a job that asks `drop_zone_people` would be told the same of a
person within C of the point below it, and the others, whose list is `task_things`, are told nothing of it).

**The catastrophe rate** is the share of the harm rows answered (`unknown` aside) that are catastrophes, overall and for
each job (`catastrophe_rate`; `### Catastrophes` in `results.md`). It shows a reckless agent that a mean score can hide.

**An illustration, not a result.** 0.9.0's answers (sol, luna and astra; 121 short and medium cases) were replayed with
the new rules. With the term off the replay
reproduces the published headlines exactly. With it on, on the apartment (the helper and the vacuum), sol's headline
goes from +0.443 to +0.364, luna's from -0.133 to -0.340 and astra's from +0.663 to +0.615; the warehouse is unchanged,
since no catastrophe fired there. The catastrophe rate by job:

| Model | Helper | Vacuum | Drone | AMR |
| --- | --- | --- | --- | --- |
| sol | 17% | 18% | 8% | 0 |
| luna | 33% | 53% | 18% | 0 |
| astra | 3% | 20% | 3% | 0 |

The review replayed the same answers by cell (the headline cells of the reported score, before and after; 0.9.0 asked
`person_mem`, and 0.10.0 does not):

| Model | Cell | Before | After |
| --- | --- | --- | --- |
| sol | vacuum `clear_path` | +0.40 | +0.21 |
| sol | helper `clear_path` | +0.35 | +0.20 |
| sol | `person_mem` | +0.39 | +0.16 |
| luna | vacuum `clear_path` | -0.24 | -0.75 |
| luna | `person_mem` | +0.09 | -0.68 |

Astra barely moves, except the vacuum's `clear_path` (+0.47 to +0.22). No catastrophe fired on an answer with a credit
of 0.75 or more, and answering 0 every time cannot win. A few rows weigh a lot: 3 of 21 rows move sol's vacuum cell by
-0.19. These are 0.9.0's answers to 0.9.0's questions with the new rules applied afterward; the rows on the redesigned
questions are in [report.md](report.md), section 4.

## 7. Timestamps

Every clock a system is sent is `YYYY-MM-DDTHH:MM:SS.s`, with one decimal of a second always (`case.format_clock`,
`case.case_clock`). In the header line of a frame it reads `[k12 2026-03-02T08:00:18.0] [...]`.
- **Fake:** the date and the time of day. A case's start clock is drawn from the case's own random id, a 2026 date
  between 07:00 and 10:00, never from the script's inputs or the window, so a clock names neither the seed nor the
  window's place in the script.
- **Accurate:** the time between clocks. A frame's clock is the start clock plus the script's own time since the case's
  first frame, rounded to a tenth of a second, and an unobserved gap between tasks is its true length. Frames are at
  least a second apart, so no two share a stamp. The agent's thinking never advances the clock.
- **Tenths, not seconds.** Until this build a clock was rounded to the second. A walking person covers a metre in under
  a second, and `closest_approach` divides displacements by the clock difference, so the second-rounded clock moved its
  gold by more than C on 6% of person rows (measured on the dev root).
- **Metadata, not pixels.** The clock is in the header line and the step's `clock` key. It is not painted on the image,
  which would change the pixels the benchmark measures.

## 8. What is still open

- **No Codex model has read the 0.11 questions.** 0.11.0's reference rows are Claude models ([report.md](report.md),
  Section 4). A Codex fast tier (sol and luna on 24 dev cases, [eval.md](eval.md)) can follow. A rise of more than 0.10 in sol's `unknown` rate on answerable questions there would
  be an instrument fault to look for in the wording, not a property of the model.
- **The free check of the 0.11 dev root passes** in every reading on every job: `ALL GATES PASS` and `EVIDENCE VERDICT
  PASSES`, one job at a time. The largest lead a shortcut has over `all_unknown` is +0.04 on `open_heading` for the
  helper and +0.00 for the vacuum, +0.03 on `way_to`, and +0.00 on the vacuum's `task_things`, against a gate line of
  +0.10. The family pages record the scoped checks that decided the two new asks: `way_to` on the helper's 36 cases and
  the vacuum's `task_things` on its 36 each pass every gate ([families/way_to.md](families/way_to.md),
  [families/task_things.md](families/task_things.md)).
- **No held-out set.** No render happens near the release, so every row is a dev row, scored on the scripts the
  questions were refined on. A set of fresh scripts, compiled once at a frozen commit and reported apart from the dev
  rows, is designed and deferred.
- **Change over time is still untested.** The world changes only between tasks, the short cases hold no change and the
  medium cases show too few observed changes ([README](../README.md), "Limits"); it needs new data.
- **`way_to`'s supply is below the design's estimate.** It asked 25 questions on the helper's short and medium cases (29
  in all lengths) where about 45 were expected: the end questions reserve their things first, a thing is named once a
  case and a thing needs a route through floor the robot has seen. All 19 answerable golds in the short and medium cases
  ask for a turn of at least 30 degrees, 14 of them for 120 degrees or more. It pools into the memory cell, so it
  creates no thin cell.
- **The vacuum's list is thin.** About 0.6 questions a case (22 over its 36 cases, 17 with items, `empty_keep_percent`
  20), so a reading of the family alone is wide. It carries no catastrophe: it is asked after the job.
- **The leak check has two readings and has not been rerun on 0.11.** The verdict reads the pairs with an answerable
  gold; the pairs with an `unknown` gold are an abstention shift, reported apart ([eval.md](eval.md), "D2's verdict").
  The prompt's new sentence is meant to shrink that shift. Both need a model run.
- **The own-log screen is held in reserve.** It is off in the specs; a free check that shows `path_log` or its in-pool
  twin over the shortcut line on a heading family turns it on for that family (methodology rule 5), and a family that
  still fails leaves that job.
- **The gates, after the repairs (0.10.0).** On the compile after the final review's repair pass (repair R1)
  every gate passes on every family and job, in the point verdict and in the evidence verdict (`ALL GATES PASS`,
  `EVIDENCE VERDICT PASSES`); the balance rule holds everywhere. Against the compile before it: the helper's
  `clear_path` `memory_needed` read +0.033 (evidence cells +0.136 on the answerable rows, +0.177 on the long cases,
  failing) and reads +0.000 in every cell; the helper's `closest_approach` has no `memory` row, so that gate has nothing
  to read; the drone no longer has `drop_zone_people`; the other gates' values moved only by the helper's new draw
  (`task_things`' `shortcuts` +0.030 against 0.10). `person_mem` failed `separation` earlier and was dropped from the
  helper (below).
- **The helper's `clear_path` memory cells were the decision-frames-first draw, and the repair removes it.** The first
  diagnosis, recurring routes accepted as fixed structure, was wrong. 44 of the helper's 106 corridor rows sat at decision frames, 8 of the carrier gold's 10 at `before_place`, and
  a decision frame puts a structure question at the same plan pose in every script; the decision-frame rows alone read
  +0.211, the others +0.000, and the rows without `before_place` passed. `before_place` left the helper's
  `look_decisions`; the second form (no decision frame first for a structure family) was not needed. Section 5 has the
  draw, and [families/clear_path.md](families/clear_path.md) the numbers.
- **A person out of view goes unasked.** The helper's `person_mem` is dropped (section 3), so where a resident who has
  left the view is has no question; the resident in view is `closest_approach`'s. It returns when the family's
  answerable supply lets `separation` pass ([families/person_mem.md](families/person_mem.md)).
- **The AMR's clearance `now` cell counts again.** It held 6 questions on the compile after lane J2 (all of them
  `closest_approach`'s "no one" rows: a row with a person was `memory`), so the AMR's `now` column was distance alone.
  The robot's own velocity is now delivered with every step, so the rows whose person stands are `now` again, and the
  cell holds 22 (6 "no one" and 16 still-person rows); on the 0.10.0 compile its `memory` cell held 36 (the 2 rows of a
  walking person and `closest_person`'s 34), and 0.11.0's `approach_mem` takes it to 83. Lane J's first wiring also tagged the person rows `now` and the cell held 26, which the check
  showed false because the robot's velocity then needed the frame before; delivering it removes that need, and the check
  reads the new tag as true (`current_view` reproduces every `now` row,
  [families/closest_approach.md](families/closest_approach.md), "Scope").
- **`closest_approach`'s motion half has two rows.** The dev people mostly stand, so of the family's 46 questions 32 are
  a person who stands (`now`), 12 are "no one" (`now`) and 2 are a person who walks (`memory`, the AMR's): a score on
  the family is mostly about ranging a still person from the picture and reading the robot's own velocity. The final
  review counted 0 walking rows among 33, by the evidence set, which a walker sighted at the frame before the ask does
  not lengthen (the same two rows were there). A dataset with walking people would exercise the rest.
- **`drop_zone_people` is asked by no job.** On the 0.10.0 draft it asked 2 questions over the 36 drone cases: 593 slots drew
  it, 413 fell outside its three phases and 174 more found nobody within 4 m, and 1% of the empty golds are kept. Even
  if every frame were asked, the dev root holds 34 `items` rows. The counted-cell rule has no per-family diagnostic, so
  its rows sat inside the drone's clearance cells at about 2%; the final review's S2 took it off the drone. The code,
  the gold and the page stay for a dataset with the supply
  ([families/drop_zone_people.md](families/drop_zone_people.md)).
- **The help kind counts since 0.11.0.** Through 0.10.0 it held `open_heading` alone, a diagnostic ability counted in no
  mean, because its credit was a cliff: a heading 2 to 10 degrees off the right opening scored -0.4 to -1
  ([families/open_heading.md](families/open_heading.md) keeps that account). The credit is now room to move on the one
  need rule (section 6), and `open_heading` is a clearance question; `way_to` joins it in memory. On the 0.10.0 compile
  the first asked 58 questions on the helper and 134 on the vacuum (all lengths), and the second 29 on the helper. The
  three models' heading scores are far apart (luna -0.33, sol +0.03, astra +0.16 on 38 rows), a hard family and no
  saturation.
- **The AMR's density** is 1 question in 24.7 frames, above the one-in-15 target, for the supply reasons of section 5.
- **Reference rows ran on 42 of the 124 short and medium cases** ([report.md](report.md), section 4): the cases all
  three models had finished when those runs stopped.
- **Counted cells.** Since 0.11.0 a cell weighs `min(1, n / 20)` and none is left out (methodology rule 4);
  `open_heading` is in clearance, so the helper's two heading `now` questions sit in its clearance `now` cell and not in
  a thin cell of their own. In 0.10.0, on the compile after the final review's repair pass (all lengths;
  `abilities.MIN_ABILITY_N` = 20) the only thin cell was the helper's heading `now` (2 questions, a diagnostic).
  Counted: AMR clearance `now` 22 and `memory` 36; drone clearance `now` 96 and `memory` 100; helper clearance `now` 51
  and `memory` 67; vacuum 102 and 346. Before the redesign three clearance cells sat under the line after `now` became
  the latest frame; lane J's wiring had left the helper's heading `now` alone thin, with the
  AMR's clearance `now` at 26 (50 and 94 on the helper); the compile after lane J2 had the AMR's clearance `now` at 6
  (thin) and the helper's `memory` cell at 109 with `person_mem`'s 19.

## 9. Adding a question

A question family is a module, a gold and a page. The checklist, in the order the registries read it:
1. `questions_<x>.py` defines `SPECS`, `GOLDS` and, for a family asked at look points, `LOOKS`
   ([questions.md](questions.md), "The family modules"); name it in `questions_text.FAMILY_MODULES`. Its gold is code
   over the timeline and the visibility log, names its evidence, and returns a dead band (`None`) where the sensors do
   not decide it. Every constant states its source in a comment.
2. Its table answers go in `table_<x>.py` (`table_answers.ANSWER_MODULES`) and its `path_log` answers in
   `baselines_path_<x>.py` (`baselines_path.PATH_MODULES`), so the gates have a perfect reader and the shortcuts to
   compare.
3. `abilities.FAMILY_ABILITY` and `FAMILY_KIND` file it, and `categories.FAMILY_CATEGORY` names its category
   ([categories.md](categories.md)); `records.BALANCE_POOLS` names its classes when a draw balances
   them; `brief.DURING_PHRASES` words the brief's during sentence, which names no person or thing.
4. A job asks it by naming it in `JobSpec.looks` and a slot of `JobSpec.look_shares`, with `look_phases` where the
   status decides whether it means anything. A harm family adds its `tolerances["catastrophe"]` rule (`questions_harm`,
   `rules.CATASTROPHES`) with a weight from `credit_kernel`.
5. The family page says what it asks, how the gold is computed, what is don't-care, the constants with their sources,
   the measured supply and the baselines. A family that fails a gate on a job is dropped from that job, not re-tuned.
6. No special case: nothing in a family or a job names one robot, one memory system or one model.
