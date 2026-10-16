# `person_mem`

"As you last saw them, where is the point of the person you saw that is nearest to your camera now, measured along the
floor? Answer in metres ahead of the camera and to its left." The offset, from the camera as it is at the look point, of
the nearest part of the person the robot saw before the recent view at the look point (the last 8 frames of the task up
to it) and does not see in it. During, `memory`. Code: `stbench.questions_people` (the gold, with
[`closest_person`](closest_person.md)), `stbench.person_seen` (the evidence rule), `stbench.table_people`. The record and
questionnaire rules are in [questions.md](../questions.md). New in 0.8.0. **No job asks it** (the helper did through
0.9.0 and in the first 0.10.0 draft; below, "Why it is not asked"). It is [`ego_mem`](ego_mem.md) for a person, and it
asks where the resident is from the robot now: a helper about to swing its arm or reverse must know where the person
stands. It is a different question from [`person_where`](person_where.md), which asks where a person was at an earlier
event and failed `time_needed` because the resident rarely moves; that gate does not apply here.

| | |
| --- | --- |
| Category | C3 Locate |
| Asked | by no job. It was the helper's, in the second slot of its look table (45% beside `ego_mem` 30% and `open_heading` 20% through 0.9.0; 40% beside `ego_mem` 25% in the first 0.10.0 draft) until the check dropped it. The family stays in the code, with its gold, its table answers and its catastrophe rule, for a job that names it in `JobSpec.looks`; it is asked only for a job with `JobSpec.one_person`: "the person" must name one person, and the helper's home holds one resident in 14 shifts and none in 2 (a generator invariant, which the helper's checks hold: a world with two fails validation, naming the script). The helper is the one job that declares it |
| Text (slots) | "As you last saw them, where is the point of the person you saw that is nearest to your camera now, measured along the floor? Answer in metres ahead of the camera and to its left." The during sentence of the brief is `ego_mem`'s, so the brief does not announce people |
| Answer, credit | `offset`, `graded` on the 2-D error against the job's clearance C (0.15 m for the helper): the resolution is one pixel at the camera of the evidence sighting, and the harmful side is a person placed farther from the camera than they are (`harm` `radial`) |
| Scope, `needs` | `memory`; see, pose |
| Gold function | the `person_mem` entry of `questions_people.GOLDS`, called with the world and the point: a `Moment` or None |
| Class | `answerable` or `unknown` (`records.BALANCE_POOLS`). **Not sliced** (below); the helper kept 7% of its `unknown` golds (`JobSpec.look_unknown_keep`) |

## Why it is not asked

On the helper's compile after lane J2 (the dev root), the check's `separation` gate failed for this family and
for no other: the table (`memory_table`) beat `blind` by +0.368 (95% interval +0.130 to +0.706), where the gate wants at
least 0.4 with the interval's lower bound above 0.3. Every other gate passed (ceiling +1.00, `knowable` +0.00,
`memory_needed` +0.00: the current view cannot answer it, shortcuts +0.00). The compile had 19 questions over the 36 helper
cases, all in the short and medium cases, at a fill of 0.061 a drawn slot (19 of 312 draws): 7 answerable and 12 with an
`unknown` gold.

The failure is arithmetic, not noise. A blind `unknown` scores 1 on an `unknown` gold and 0 on an answerable one, and a
perfect reader scores 1 on both, so the lead of a perfect reader over blind is exactly the answerable share, here 7 of 19
or +0.368. With 7 of 19 rows answerable the gate cannot pass whatever the reader does: the family would not tell an agent that
perceives and remembers from one that answers `unknown`. The resident is in view in 4.5% of the helper's frames,
so an answerable gold (an evidence sighting before the recent view and no row of the person since) is rare, and a point
whose window holds no row of any person is common. 0.9.0's compile of the same family passed with 16 questions (separation
+0.750; [the 0.9.0 report](../history/report-0.9.0.md), the gate table): the draw changed with the points, and the family sits near the line on
a thin supply.

The rule that decided it is the evidence rule with the vision's first success criterion, that the score ranks agents "by
that ability and nothing else. Not by ... guessing" ([vision.md](../vision.md)): a family that fails a gate on a job is
dropped from that job, not re-tuned ([question-design.md](../question-design.md), section 9). The helper's 40 of slot 2
went to `ego_mem`, which now has 65. The person in view stays covered by [`closest_approach`](closest_approach.md); where
a resident who is out of view is goes unasked, an open gap ([question-design.md](../question-design.md), section 8), and
[`approach_mem`](approach_mem.md) (0.11.0) asks how near the helper would come to that resident, not where they are.

**What would bring it back.** A compile that gives the family enough answerable rows for `separation`: more than 40% of its
questions answerable, with the interval's lower bound above 0.3, at a count that gives that interval (a cell counts from 20
questions). That takes more answerable supply, which means more points at which the resident was seen and has since left
the view (a resident who moves between rooms along the helper's route, or more cases), not a change to the draw, the credit
or the gate. The job file names the family again in a slot beside `ego_mem` when it does.

## Answerable, unknown, or not asked

- **Answerable** when the window's latest evidence sighting ([`closest_person`](closest_person.md), "The person
  evidence rule") of the person lies before the recent view at `k` (the last 8 frames of the task up to it), and **no row of the person, at any pixel count**,
  falls after it up to `k`. The person is steady (people shift on the spot), and the current view would not answer. The
  gold is that **one sighting's** floor hull, its point nearest the camera's floor point at `k`, in the camera's frame at
  `k`: "as you last saw them", whatever has happened since (truth rule 2). A union over sightings would blur the place.
- **`unknown`** when the window up to `k` holds **no row of any person, at any pixel count**: the robot never saw the
  person, who may not have been home. The prompt scores `unknown` only "when nothing the robot sensed answers the
  question", so a person the robot plainly saw is never `unknown`, whether or not the sighting passes the evidence
  rule (at the helper's C/2 = 7.5 cm the rule rejects 46% of clear sightings, a person's belly or back projecting
  about 10 cm beyond their heels; final review, B1).
- Else None: a point whose rows all fail the evidence rule, a glimpse after the sighting, or a sighting in the
  recent view, leaves the point unasked.

**No slice.** `ego_mem` draws its `unknown` share by a per-point slice, which needs an allowed candidate in both pools at
the point. With one resident a point holds an answerable candidate or an unknown one, never both, so a sliced
`person_mem` is never asked. The class is the world's (seen, or never seen in the window), and the `unknown` golds are
thinned by a declared keep share, `stable_hash(case id, family, str(k), "unknown") % 100 < 7`. The thinning reads the
class only. **One ask a sighting.** The answerable candidate's subject is the sighting it reads, `w:<j>` for the
evidence sighting's keyframe `j` (the one person is implicit), so the case's subject rule asks each sighting at most
once however many look points follow it (before, a case asked one sighting up to six times, and 27 answerable rows
came from 14 sightings; final review, S4). The `unknown` candidate names none.

## Baselines, supply and difficulty

- **Answerers.** `table_people.ANSWERERS`: from the detections of kind `person` that carry a `floor_hull` (one person in
  the helper's world). The oracle and `memory_table` read the same hulls the gold takes and reproduce it. `current_view`
  holds the latest frame, where the person is not, so it is `unknown`: the `memory_needed` gate holds it under the floor.
- **Gates.** All of them, as for `ego_mem`; the held-out constant offset scores +0.00. Only `separation` failed in the
  0.10.0 draft (above).
- **Supply.** On the dev root (a dev-data read, 0.8.0 as fixed by the final review) 16 short and medium rows (11
  answerable, 5 `unknown`) and 3 long answerable: 14 answerable rows from 14 distinct sightings (the same 14 the
  unfixed compile asked 27 times, with 7 `unknown`, 31 short and medium rows). The helper's clearance · memory cell
  (this family alone) was then 19 questions over all lengths, under the 20-question line, so it was a diagnostic by rule
  ([eval.md](../eval.md), "Diagnostics"), not part of a mean. The count moves with the hash range the family holds. The resident moved by more than C after the sighting at 2 of the 14 answerable rows
  (1.3 m and 0.36 m) and was gone at 0 (the unfixed compile asked one sighting four more times, at the end of which the
  resident had left: 4 of its 27).
- **Catastrophe (0.10.0).** A person placed farther from the camera than they are, along the gold's bearing, by more than
  `max(C, 2 rho)` is a catastrophe of weight 3 (`rules.CATASTROPHES`, rule `radial`: a helper that believes the person is
  clear moves into them). It is charged in the reported score only ([questions.md](../questions.md), "Credit"). The rule
  stays with the family, but no job's rows carry it now.
- **In the 0.10.0 draft** (the dev root, on lane J's wiring and again after lane J2) the family asked 19 questions over all
  lengths, all in the short and medium cases, at a fill of 0.061 a drawn slot: 7 answerable and 12 `unknown` (above). With
  `clear_path` and `closest_approach` the helper's clearance `memory` cell was counted (94 questions on lane J's wiring); without
  this family the cell loses its 19.
- **Difficulty.** The resident was last seen a median 22-29 s earlier, 3.9-4.3 m away, and the robot has turned since.
  The precision bound is +0.14 for sol and -0.40 for luna (sol's `ego_mem` bound on the helper is +0.25). 0.9.0's reference
  rows asked it ([the 0.9.0 report](../history/report-0.9.0.md)); no 0.10.0 or 0.11.0 row does.
