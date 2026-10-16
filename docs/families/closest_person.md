# `closest_person`

How close a person came to the robot during a task, as the robot saw them. End, `memory`. Code: `stbench.questions_people`
(the gold), `stbench.person_seen` (the person evidence rule, shared with the oracle channel), `stbench.table_people` (the
answerer). The record and questionnaire rules are in [questions.md](../questions.md). New in 0.8.0. It is
the near-miss record a fleet keeps: ISO 3691-4 requires an AMR to stop before contact with a person in its path, and a
controller that has to slow down where people come close needs to know how close they came.

| | |
| --- | --- |
| Category | C2 Separation |
| Asked | end, one question a case, about one task of the window. The AMR's second end family (`JobSpec.ends`), beside `task_things`; no other job asks it |
| Text (slots) | "During your {first} task, how close did a person come to you, as you saw them? Answer the smallest distance in metres along the floor from the point below your camera to the nearest point of a person; answer 10 if you saw no one within 10 m." (`task`, the case index, worded as an ordinal) |
| Answer, credit | `number`, `graded` against the job's clearance C (0.45 m for the AMR), or a tenth of the gold if that is more ([methodology.md](../methodology.md), rule 7; 0.11.0): the resolution is one pixel at the nearest evidence sighting, the harmful side is a closest approach answered wider than it was (`harm` `over`: the need is halved), and the direction quarter is the same side of `T` (the body's length plus C) and, above it, within a factor of two |
| Scope, `needs` | `memory`; see, pose |
| Gold function | the `closest_person` entry of `questions_people.GOLDS`, called with the world: at most one `Moment`; subjects `("w:{k0}-{last}",)`, from the task's first keyframe to the last keyframe of the task in the window, so the task named by its ordinal is the subject and no person is |
| Class | `person` or `no_one` (`records.BALANCE_POOLS`); reference None, `mapper_gold` False |

## Why it is hard

A task is 50-200 frames and a picker shows in a handful of them, often cut by the frame's top as the robot passes. The
agent must find every sighting, range each one, and keep the smallest. Missing a close pass, or misranging a clipped
body, costs the full credit.

## The gold

One question a case, about the task its hash puts first. The window's tasks are taken in `stable_hash(case id,
"closest_person", str(task))` order and the first that passes its screens is asked (one question a task would give the
AMR's medium cases 3.5 end questions a case against 2.88 and break the end budget).

- **With a person.** The gold is the least, over the task's keyframes and its evidence sightings (below), of the floor
  distance from that keyframe's camera floor point to the nearest point of the sighting's floor hull
  (`nearest_part`), each from its own frame's camera, capped at 10 m. The task is dropped when a row of a person who is
  **not** evidence lies nearer than the gold by more than C/2 (the truth's footprint): a person the agent may see
  closer, whose distance no frame resolves.
- **"No one"** (gold 10). Asked only when no person row of 256 px or more lies within 10 m of the task, and then kept for half of
  the golds that pass, by `stable_hash(case id, "closest_person", "empty") % 100 < 50`, so that 10 is not the common answer.

The gold reads only the task's frames, all before the end ask, so it is knowable from what was sent. Its median is about
2.4 m.

## The person evidence rule (`person_seen.evidence`)

A sighting of a person is evidence when it has at least `RECOG_PX` (1,024) pixels, `RECOG_LUM`, a depth at every pixel
(no gap), the contact tie at C (`nearest_part.surface_tie` on `ego_support.contact_bases`: a pixel of the feet within
max(2 cm, a pixel's height) of the floor **and within C/2 of the hull's nearest point to the camera, along the
floor**, read within C/2 at one pixel), and a box that does not touch the **bottom** edge of the frame. The second
clause matters at the helper's C/2 = 7.5 cm: a person's belly or back projects about 10 cm beyond their heels, so it
drops 46% of the helper's clear sightings (not the AMR's, whose C/2 is 22.5 cm). The top and the sides may cut the person.

`recognise`'s edge rule (a box 2 pixels from every edge) exists because a thing cut by the frame may be a sliver of
itself and misnamed. A person cut at the chest by the top of the frame is plainly a person, and their feet place them.
Under the strict rule 41 of the AMR's 46 closest-approach golds, the close passes, would be dropped (the median gold
would be 3.5-3.7 m against 2.3-2.5 m), because a person passing close is the clipped one. The visible floor hull of a
sighting cut at the top lies a median 0.07 m (q90 0.10 m) beyond the truth's footprint, well inside C/2. The rule is one
neutral function: the gold and the oracle channel's `floor_hull` both call it ([format.md](../format.md)).

## Supply, baselines and difficulty

- **Supply.** 30 short and medium rows (29 with a person, 1 "no one") and 4 long on the dev root (a dev-data read, the
  0.11.0 release's 34 rows in 16 scripts); no other job asks it (in 0.8.0's read the helper had 10 rows, the drone's
  constant scored +0.64 and the vacuum never places a person).
- **Gates.** All of them; a held-out constant scores +0.00. `current_view` holds the latest frame only, and `path_log` has
  no people to read: the walks never depend on the robot. On the 0.11.0 release the family passes every gate on the AMR:
  ceiling +1.00, `knowable` +0.00, `memory_needed` +0.00 (`current_view` scores -0.91), shortcuts +0.00 and `separation`
  +1.00.
- **Answerers.** `table_people.ANSWERERS`: from the detections of kind `person` that carry a `floor_hull`.
- **Difficulty.** The precision bound is +0.80 for sol (its `ego_mem` range errors at 2-4 m) and -0.21 for luna. It
  assumes the right sightings are found. Results: the reference models' rows are in [report.md](../report.md), section
  4; the family has 10 questions on `warehouse_a` among the scored short and medium cases, so its row is a diagnostic.
- **Asking `person_now` during a task** may help this question; methodology rule 6 measures such effects, it does not
  assume them. `person_now` is not asked, in 0.8.0 or since (its 17 AMR and 11 helper rows add relevance, not difficulty: its
  precision bound equals `ego_now`'s).
