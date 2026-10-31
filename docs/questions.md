# Questions and gold

What a case asks and how each answer is checked: the question and gold records, the families, how the compiler computes
a gold, and the rules that keep a question from teaching its answer. Code: `stbench.records`, `stbench.questions*`,
`stbench.questionnaire`, `stbench.grid`, `stbench.freespace`. Rules and reasons: [methodology.md](methodology.md) §3 (the truth
rules). How a system is streamed a case and asked
is [agent.md](agent.md); the reference agent is [codex.md](codex.md); the file layout is
[format.md](format.md). The commands are for a checkout (`uv run stbench ...`); from a pip install, drop `uv run`.
A question is authored from a category, a decision with a gold template and the hooks a job fills in:
[categories.md](categories.md) is the authoring unit, and each family page names its category.

**0.10.0** redesigns what a case asks and when ([question-design.md](question-design.md) is the reasoning behind it, and
this page the rules). A case asks one question in about 9 to 25 frames, by job (0.9.0: about one in 30). A look point can
be any frame of the case, the frames of the robot's decisions come first in the draw, and a point has three slots, filled
harm first (below, "Look points"). Three families are new: `closest_approach` (how close the person nearest the robot
would come in the next 5 s; AMR and helper), `tangle_ahead` (what on the floor the vacuum's brush could catch) and
`drop_zone_people` (who stands within 4 m of the point below the drone; asked by no job since the final review's S2). The floor robots (vacuum, AMR) are asked only
about things on the floor. Every gold row records the age of its evidence (`age_s`, `age_band`), and a harm answer whose
action would end in harm is a catastrophe, charged in the reported score only ("Credit", "Catastrophes"). The "Credit"
paragraphs below that say 0.9.0 are still the rule except where they say otherwise; the paragraphs on the 0.8.0 and
0.9.0 budgets and shares are history.

**0.11.0** scores every distance on one need rule, `N' = max(N, 0.1 g, 2 rho)`, and counts the heading question (below,
"Credit"; [question-design.md](question-design.md), section 6). It asks the helper the way to a thing it saw (`way_to`,
in the second slot beside `ego_mem`), asks the vacuum the list of what its brush could catch at the end (`task_things`),
drops an `unknown` candidate whose window glimpsed a look-alike in the asked colour, and adds one sentence to the prompt
(a question may name a thing never shown or not there; then `unknown` is right). It scores a list item as a place, earning its direction
quarter within the match radius, and an empty gold's list on a curve (below, "Credit"). It asks the AMR and the helper a
second harm question about a person, `approach_mem`: the closest approach to the person last seen, now out of view. It
keeps the 0.7.0 renders and compiles them again: no new render.

0.9.0 asks each question as a decision the robot's brain faces and credits it by its consequence for the job (below,
"Credit"; [vision.md](vision.md), "How questions are made and scored"). It asks four abilities from the camera alone,
distance, memory, task relevance and clearance. Clearance now asks the corridor in every direction the robot can move.
It also asks which way has the most room (`open_heading`): in 0.9.0 and 0.10.0 the diagnostic ability `heading`, asked,
gated and reported (as the help kind) and counted in no mean, and since 0.11.0 a clearance question that counts
([families/open_heading.md](families/open_heading.md) says why it did not, and how it does). The next section is what a
case asks. The sections after it that describe `zone_clear`, `out_of_view` or the 0.5.0-control mapping, and the account
of how the 0.4.x families were decided, record 0.4.x and 0.5.0 and are marked as history.

**History.** Release 0.5.0 revised the questions in two steps. 0.5.0 added three families that ask about change over time, so the
set is no longer only "where was it"; it made a gold cite a frame a person could name and let a large thing's footprint
count; it drew the questions that matter to the job first; and it stopped the decoys, and the real things a look-alike
hides, giving themselves away. 0.5.0-control asks each job a question about the place its next manoeuvre uses while it
is still undecided (`zone_clear`, `now`), asks `object_where` as "last seen" of each answerable thing, gives each job
the tolerances of its place and a near rule where the camera sees only so far, keeps out of every end question what the
robot's own log tells about (`World.told`), and names a thing only by a colour its render shows (below, "Real things and
look-alikes"). 0.5.0 also added the warehouse's look-alikes for the questions about a thing never seen, a bin with a
drum and a sack with a pallet (the AMR's log review: both reference models answered "the yellow bin", never seen, with
the yellow drum beside it). 0.6.0 is the five-abilities revision: the `offset` answer, look
points and the families `ego_now`, `ego_mem` and `task_things`. It adds two home look-alike pairs from the look-dev
review, an air purifier with a space heater and a pedal bin with a bath bin (below), takes `object_was` and `same_place`
out of every job's `ends` (above), and makes the AMR's `task_things` path the whole drive, so a final stop that was
blocked and skipped is still driven to (below, `task_things`). 0.7.0 credits an offset by the job's clearance and a
place by its re-find radius (below, "Credit"). 0.8.0 asks the questions a robot's job depends on, not more of them: how
far the body could move (`clear_path`), how close a person came (`closest_person`) and where the person last seen is
(`person_mem`), inside 0.7.0's budget. A look point draws its family from a table each job declares; the corridor is
read at the 2 cm sub-cells with a 3 mm allowance over the climb; and the first structure family, `clear_path`, ships
with a map reader as its reference in `stbench check` (below, "The task-critical questions"). 0.9.0 credits every metric answer
by its consequence in four quarters, asks `clear_path` in every direction a ground robot can move, from what it last saw
of every cell up to the point, and adds `open_heading`, a decision scored by its regret (below, "The decision
questions"). 0.10.0 delivers the frames one at a time (the old batch rule becomes the compiler's look window, and `now` is the
latest frame) and redesigns the questions, as above (below, "Look points" and "How a gold is computed"). The sections
below state each rule. Where a page still records a gate check, it is a 0.4.x record
and says so.

## Commands

```sh
uv run stbench perceive DATASET/scripts/<sid>                # the private visibility log and the oracle channel
uv run stbench compile DATASET [--cases ID,...] [--workers N]  # each case's private questions.jsonl and gold.jsonl
```

`compile` prints each case's questions per family and the total. It refuses a script whose render is missing or covers
only part of it (`render_cli.render_binding_problems`), exit 2, naming the reason; `perceive` refuses it the same way. A
gold compiled over a partial render would be wrong without saying so. Every later command refuses a case that has no
gold, naming `stbench compile`.

## What a case asks

The robot has a camera, its own pose, the clock and its task status,
and no depth: the agent estimates distances. `abilities.FAMILY_ABILITY` knows six abilities, one controller question
each, in two columns of the headline ([eval.md](eval.md) for how they are scored); prediction is a
diagnostic (`abilities.DIAGNOSTIC`), reported and counted in no mean (heading was one through 0.10.0). The four shipped
jobs ask four of them, distance in the `now` column, memory and task relevance in the `memory` column, and clearance in
both (each row in the column of its scope):

| Ability (column) | Family | When | Answer | Asked by | Page |
| --- | --- | --- | --- | --- | --- |
| Distance (`now`) | `ego_now` | during, at look points | `offset` | AMR, helper | [families/ego_now.md](families/ego_now.md) |
| Memory (`memory`) | `ego_mem` | during, at look points | `offset` | all four jobs | [families/ego_mem.md](families/ego_mem.md) |
| | `way_to` (help, 0.11.0) | during, at look points | `heading` | helper | [families/way_to.md](families/way_to.md) |
| | `object_where` | end | `position` | helper, vacuum | [families/object_where.md](families/object_where.md) |
| Change (`memory`) | `same_place` | end | `yesno` | none | [families/same_place.md](families/same_place.md) |
| | `object_was` | end | `position` | none | [families/object_was.md](families/object_was.md) |
| Task relevance (`memory`) | `task_things` | end | `positions` | helper, AMR, vacuum (0.11.0) | [families/task_things.md](families/task_things.md) |
| Clearance (`now` or `memory`, by evidence) | `clear_path` | during, at look points | `number` | drone, helper, vacuum | [families/clear_path.md](families/clear_path.md) |
| Clearance (`memory`) | `closest_person` | end | `number` | AMR | [families/closest_person.md](families/closest_person.md) |
| | `person_mem` | during, at look points | `offset` | none (the helper's through the first 0.10.0 draft; dropped by the check) | [families/person_mem.md](families/person_mem.md) |
| Clearance (`now` or `memory`, by evidence) | `closest_approach` | during, at look points | `number` | AMR, helper | [families/closest_approach.md](families/closest_approach.md) |
| Clearance (`memory`) | `approach_mem` (0.11.0) | during, at look points | `number` | AMR, helper | [families/approach_mem.md](families/approach_mem.md) |
| | `drop_zone_people` | during, at look points | `positions` | none (the drone asked it until the final review's S2) | [families/drop_zone_people.md](families/drop_zone_people.md) |
| | `tangle_ahead` | during, at look points | `positions` | vacuum | [families/tangle_ahead.md](families/tangle_ahead.md) |
| Clearance (`now` or `memory`, by evidence; the ability `heading`, a diagnostic, through 0.10.0) | `open_heading` (help) | during, at look points | `heading` | helper, vacuum | [families/open_heading.md](families/open_heading.md) |
| Prediction (diagnostic) | `mover_next` | not built: only if B0 shows supply | `position` | none | |

`zone_clear`, `out_of_view` and `adjacent` were retired in 0.6.0 (unmapped from every job, modules kept); the coins and
responses at decision points stay as world realism, and no decision kind asks a family. Since 0.8.0 each job declares
the families it asks at look points (`JobSpec.looks`), the table that draws between them (`JobSpec.look_shares`, below,
"Look points") and its end families (`JobSpec.ends`). Since 0.9.0 it also declares the directions of its corridor
(`JobSpec.clear_directions`: ahead, back, left, right for a ground robot, ahead and back for a flier; refused at
registration, naming the job, when it declares another). Since 0.10.0 it also declares the decision kinds whose frames
come first in the draw (`JobSpec.look_decisions`) and the status phases a hazard family is asked in
(`JobSpec.look_phases`). The table is 0.11.0's, from `job_*.py`; the share of a slot is the percent of points at which
its family is drawn, and the reason for each is in the job's file, with the fill it rests on:

| Job | Look slot 1 (percent of points) | Look slot 2 | Look slot 3 | End | Declared beside |
| --- | --- | --- | --- | --- | --- |
| AMR (`amr/1`) | `closest_approach` 100 | `ego_mem` 100 | `ego_now` 50, `approach_mem` 50 | `task_things`, `closest_person` | no `clear_path` (`clear_directions` unused); `look_decisions` `before_crossing`, `before_station` |
| drone (`drone/1`) | `clear_path` 60 | `ego_mem` 70 | (no third slot) | none | `clear_directions` ahead, back; `clear_phases` every phase, `fly_floor_m` 8.35, `look_unknown_keep` `clear_path` 75; no `look_phases`; `look_decisions` `before_lower`, `before_descent` |
| helper (`helper/1`) | `clear_path` 50, `closest_approach` 50 | `ego_mem` 65, `way_to` 35 | `ego_now` 20, `open_heading` 15, `approach_mem` 30 | `object_where`, `task_things` | `clear_directions` ahead, back, left, right; `clear_phases` `navigating`, `carrying`; `look_phases` `way_to`: `navigating`, `carrying`; `one_person` (held by the job's checks; `person_mem` was the family that read it); `look_unknown_keep` `clear_path` 35; `look_decisions` `before_grasp`, `before_doorway` |
| vacuum (`vacuum/2`) | `clear_path` 50, `tangle_ahead` 50 | `ego_mem` 100 | `open_heading` 20 | `object_where`, `task_things` | `clear_directions` ahead, back, left; `clear_phases` `cleaning`; `look_phases` `tangle_ahead`: `cleaning`, `transit`; `look_decisions` `before_doorway`, `room_done` |

Slot 1 is the job's harm questions, slot 2 what the robot saw before, slot 3 the rest; the drone's `drop_zone_people`
had slot 3 to itself because its fill is too low to share a slot, and left the drone with its 2 questions over the dev
root (final review S2), so the drone has two slots. No score enters a share. The helper's slot 2 was `ego_mem` 25 and
`person_mem` 40 in the first 0.10.0 draft: `person_mem` failed `separation` (7 of its 19 questions answerable, a lead of
+0.368 against 0.4), was dropped by the evidence rule, and `ego_mem` took its 40. In 0.11.0 the 35 that slot 2 drew
nothing on goes to `way_to`, so `ego_mem` keeps its 65 and its supply. The helper's `clear_path` `memory_needed`
evidence cells failed until the final review found the cause in the draw (the decision frames come first, and a decision
frame puts a structure question at the same plan pose in every script) and `before_place` left the helper's
`look_decisions`; every gate now passes ([families/clear_path.md](families/clear_path.md), "The helper's `memory`
cells"; [question-design.md](question-design.md), section 8). The reason for each share is in the job's file and in
[jobs.md](jobs.md), [job-helper.md](job-helper.md) and [job-amr.md](job-amr.md).

**`approach_mem` takes half of the AMR's slot 3 and 30 of the helper's 65 empty points** (0.11.0). The AMR's slot 3 drew
`ego_now` at every point; it now draws `ego_now` at 50 and `approach_mem` at 50, so `ego_now`, the warehouse's
most-asked `now` family, is asked half as often, which moves no cell's mean, and the clearance `memory` cell gains a
question that needs a person seen earlier. The helper's slot 3 drew nothing at 65% of its points, and `approach_mem`
takes 30 of them. The drone and the vacuum never draw it: at altitude a person is not the drone's contact hazard and the
people at its drop stand at layout spots ([question-design.md](question-design.md), section 3), and the vacuum's 8 cm
camera places no person. A failing gate on a job removes `approach_mem` from that job's slot, followed by a recompile,
never a re-tune. On the 0.11.0 compile `approach_mem` holds 47 questions on the AMR and 19 on the helper, and the
AMR's `ego_now` falls from 105 to 65.

**History: the budget of 0.8.0 and 0.9.0, and the rule.** Through 0.9.0 a case could not ask more than 0.7.0 had. 0.10.0
drops that rule, to ask more questions per case ([vision.md](vision.md), "Speed"): the frames are the fixed
cost of a case, and the density is measured ([question-design.md](question-design.md), "When questions are asked"). The
next paragraphs record the 0.8.0 and 0.9.0 tables and the rule that chose them.

**Which questions give way, and the rule.** The budget is the time a tier takes: for every job and length the mean
questions and the mean asks a case on release-060 may not rise from 0.7.0's, while the cap on look points, the
two questions a point and `MAX_END` stayed caps. The rule that chose the table, in order, with no model's score in it:
1. A family that fails a validity check or gate on a job, as measured, is not asked there.
2. Within the budget the job's controller question comes first (can I move without touching anything, and where are the
   people), then where things are now and as last seen, then where on the map a thing was last seen.
3. Every family asked keeps at least 20 rows on short and medium cases, or it is not asked on that job.
4. The percents are the largest that hold the means. The review's table declared `ego_now` 90 and `ego_mem` 100 (AMR),
   `ego_mem` 100 (drone), 40 (helper) and 60 (vacuum); on the real compile the families that left a job freed subjects,
   so `ego_mem` found more rows and the means rose, and the jobs lower those percents to the largest on a grid that
   holds every length's mean asks and questions (the AMR's `ego_now` to 70, `ego_mem` to 35; the drone's to 85; the
   helper's to 32; the vacuum's to 36). The controller families and the other jobs' `ego_now` keep their declared
   shares; the AMR's `ego_now` is the one exception, declared 90 and lowered to 70.

**The vacuum asked no `task_things` in 0.9.0 and 0.10.0 (history; 0.11.0 asks it again, below).** The free check of the
0.9.0 compile (the dev root, a dev read) found the family failing the evidence verdict on `vacuum/2`:
`path_log_in_pool`, the robot's own log with the layout prior of other scripts' golds, scored +0.258 over `all_unknown`
on its 14 long rows (95% interval +0.151 to +0.376), +0.135 on the 24 medium and -0.152 on the 11 short, against a point
gate of +0.10. The cause is the dataset and not the credit: 90% of the vacuum's 68 distinct gold items have another
script's item within 0.5 m, and the predicate ("on the floor of the rooms you cleaned") reaches every recurring spot in
those rooms, so the places other scripts held items, within reach of the case's own logged anchor, are an out-of-sample
layout prior that earns without the camera (methodology rule 5: a fact that is the same in every case of a place is
asked only of places never seen). A family that fails a gate under the evidence verdict is dropped from its job by rule
([methodology.md](methodology.md), rule 5); raising `FP_WEIGHT` or
narrowing the match radius would be tuning a credit to pass a gate. The vacuum's end is `object_where` alone, so it
declares no predicate (`JobSpec.task_things` is None, as the drone's) and has no task-relevance cell, and the
apartment's keeps the helper's rows. The way back is a held-out place, or a list restricted to the things off the
layout's recurring spots: a design change for a later release. The mean questions a case below are those before the
drop; it only lowers the vacuum's.

**The vacuum asks it again in 0.11.0, as the list of what its brush could catch**. The predicate
(`TaskThings.item_kinds`, new in 0.11.0; empty means every kind) names the kinds a brush catches as the only items
(`hazard_rule.CATCH_KINDS`) and makes the three that recur at one spot from script to script (`hazard_rule.SPOT_KINDS`:
bath mat, pedestal mat, clothes pile) don't-care, the split `tangle_ahead` makes. A thing of any other kind is no item
and no don't-care circle, so a list that includes it is charged a false place. The gate that dropped it reads the other
way now (`path_log_in_pool` -0.298); the family is asked once for each cleaning task of the window, about 0.6 questions
a case ([families/task_things.md](families/task_things.md)).

So `closest_person` takes the AMR's second end slot (the near-miss record a fleet keeps) and `object_where` leaves the
AMR; the drone's end ask disappears and funds its corridor (a delivery drone re-finds nothing on the ground; walls are
what it can hit); `ego_now` leaves the drone (nothing it names can be hit from 15 m) and the vacuum (16 rows in 0.7.0,
under the 20-question line). Where `object_where` stays (helper, vacuum), no task-critical end family survives and an
end question costs no ask. In 0.8.0, on the dev root (release-060's 64 shifts, a dev read), the mean questions a case, short and
medium cases, were 3.06 and 5.25 (AMR; 0.7.0: 3.56 and 5.50), 2.00 and 3.19 (drone; 2.25 and 3.56), 6.25 and 11.19
(helper; 6.38 and 11.31) and 2.94 and 4.38 (vacuum; 2.94 and 4.69), and 612 questions in all against 643. No job's and
length's mean questions or asks rise. Per case, asks rise in 0-8 of a length's 16 cases: that is the price of a draw
that never reads the world. Every (job, ability, column) cell holds at least 20 short and medium questions but the
drone's clearance cell in the `memory` column (15, then left out of the headline).

**What 0.9.0 changes in the table, and why the budget holds.** Share moves from the easiest questions, the ones in
view, to the ones that need the space the camera cannot see now: on the helper `ego_now` gives 20 points of slot 1 to
`clear_path`, which draws each of its four directions equally, so the turned and reversed corridors three times in
four; in slot 2 `person_mem` (60 to 45) and `ego_mem` (32 to 30) give points to `open_heading`. On the vacuum the
forward corridor gives two thirds of its draws to the turned ones and `ego_mem` gives six points (36 to 30) to
`open_heading`. The drone and the AMR keep their tables. No job's or length's mean questions or asks a case rise from
0.7.0's. The design replayed the questionnaire's own draw over every family's real candidates and declared the helper's
`open_heading` at 25, the vacuum's `clear_path` at 25 and its `open_heading` at 20; on the real compile (the dev root,
release-060's 64 shifts) the long helper cases' mean questions rose from 14.50 to 14.75 and the vacuum's medium mean
asks from 3.19 to 3.56, because the turned corridors have more answerable points than the replay counted (the stopper
takes the nearest contact within C/2). The stop rule (the same share-lowering rule as 0.8.0's, with no score in it) cut
the helper's `open_heading` to 20. Cutting the vacuum's `open_heading` alone left it at 5 and two rows, too few to gate,
so it declares `clear_path` 20 and `open_heading` 15. The mean questions a case on the compile, short, medium and long cases, are 5.56, 10.25 and 14.50
(helper), 2.56, 4.69 and 8.25 (vacuum; 0.7.0: 2.94, 4.69 and 8.75), 2.00, 3.19 and 4.50 (drone) and 3.06, 5.25 and 5.75
(AMR), none above 0.7.0's (before the vacuum's `task_things` was dropped). `open_heading` carries no subject (it names no thing and no time), which is what lets a
point ask it beside a corridor or an `ego_mem` question. The planning rule that each family keep 20 short and medium rows gives way to the
cell line, which is the one the headline reads: every (job, ability, column) cell is counted at 20 questions (0.9.0 to
0.10.0; since 0.11.0 a cell under 20 weighs `n / 20`, [eval.md](eval.md), "The splits"). On the 0.9.0 compile the cells
under 20 are the drone's and the vacuum's clearance cells in the `memory` column (13 and 4 short and medium questions;
the vacuum's was 11 while `open_heading` counted as clearance), left out of the headline then; the helper's, left out in
0.8.0, has 30. `open_heading` was not in clearance through 0.10.0: it was the diagnostic ability `heading`, in no cell
the headline reads (below, "The decision questions"); since 0.11.0 it is a clearance question and counts.

**Not asked in 0.10.0 either**, with the reasons: questions at the start of a task, the task's own target, a bay's blocker,
change sets, a free spot to put a thing down, answers with uncertainty, a harm weight in the headline, the drone's
`closest_approach`, `tangle_ahead` items that are a layout prior, and the decision frames with no supply
([question-design.md](question-design.md), "What is deliberately not asked").

**Not asked.** The AMR's corridors (ahead: 62% of the golds the budget allows are the 5 m cap and a held-out constant
scores +0.33; back +0.24; to the sides the cap or the layout decides and the map reader leads by +0.53 and +0.80), the
vacuum's right turn (a held-out constant scores +0.28: it follows edges with the wall on its right), sideways flight
(a constant +0.15), `open_heading` for the AMR (every best run is the cap) and for the drone (the open high street is the
longest run from every point), and `person_now` (a person's offset at every look-window end: it adds relevance, not
difficulty) are measured and left out; so are people questions for the drone and the vacuum (no person is
placeable from the vacuum's 8 cm camera, and the drone's pedestrians cover 250-340 px seen from above, under the
1,024 px floor), the closest approach for the drone (a floor distance from 15 m up is not its decision; the AMR and the helper
ask it since 0.10.0), counts of people and time to contact. Three reuse
decisions are answered by the pose log the agent is given, so they are not asked: where the dock is from here, a route
between two places the robot has been, and what stays put and what moves (too rare at the sourced rates).

**No job asks `object_was` or `same_place`**: on release-c both fell under the drop line below
(0.42 a medium case on the AMR, 0.33 on the drone, 0.10 on the vacuum, 0 on the helper), and kept they would print BROKEN
reach lines that block the export. Their modules, tests and family pages stay, and the Change row above is the ability
as defined, not as asked: no job has a change column in 0.7.0 or 0.8.0. A blocker a
coin sets is told by the robot's own log, so `World.told` keys on every zone in `JobSpec.zones` and a look question
reads what the log had told by its point (`told(upto=k)`). `JobSpec.task_things` is the job's predicate: "on the floor within 1.2 m of the way you went to fetch and put down the
thing" (helper) and "within 2 m of the way you drove" (AMR, the whole drive, `waiting` included, which drops `floor`), each stating the distance its gold uses. A thing no taller than the predicate's `min_height_m` (a remote, a cable) is a
don't-care place there, not an item and not nothing (0.7.0, the dev pilot's log review).

**Credit (0.9.0; the need rule 0.11.0).** A metric answer is credited by the consequence of its error for the job,
in four quarters, each the vision's layer in its order ([vision.md](vision.md)). 0.7.0's three flat rungs (a third each within C/2, C and 2C) and 0.8.0's clearance rungs are history: a
`ladder` row of an old compile still scores as before, and no 0.9.0 row is one. An answer with error `e` against
the job's need `N` (the clearance C, the re-find radius R or a list's match radius M) and the gold's resolution `rho`
(`tolerances.resolution_m`, below) earns

    s = (direction + thing + amount + full) / 4,    N' = max(N, 0.1 g, 2 rho),    rho at least 1 cm

- **direction**, 0 or 1: the side, the order, blocked or clear (below);
- **thing** `t(e)`: 1 within 2N', falling linearly to 0 at 4N';
- **amount** `a(e)`: 1 within N', falling linearly to 0 at 2N';
- **full** `f(e)`: 1 within `rho`, then `1 - ln(e / rho) / ln(N' / rho)`, to 0 at N'.

**The need grows with the distance (0.11.0).** `g` is the distance the answer is judged over, and 0
for a place and a list item: a number's gold value (`clear_path`, `closest_person`, `closest_approach`), an offset's
ground range from the camera, `hypot(ahead, left)` of the gold (`ego_now`, `ego_mem`), a heading's best run
(`open_heading`) and, for `way_to`, the step the outcome is told over (`step_m`, the re-find radius R). So N' is the
job's own need N while `g` is within ten of it, `0.1 g` beyond that, and never finer than two pixels. On the harmful
side the first two terms are halved and `2 rho` is not (below). The same 20% error therefore costs the same on every
ground robot, where until 0.10.0 the far-field margin was two pixels of the robot's own camera. A place keeps R and a
list item M, with no `range_m` widening. The door 5 m away answered 4 m scores 0.00, 4.5 m +0.50 and 6.0 m -0.42 on the
helper, and 0.00 at 4.0 m on the vacuum and the AMR too ([question-design.md](question-design.md), section 6, has the
table for the four jobs; `tests/test_credit_far_field.py` pins the arithmetic by hand).

So an answer within `rho` scores 1, at N' 3/4, at 2N' 1/2, at 4N' with its direction right 1/4, and beyond that the
direction alone. The curve is continuous and strictly falling between `rho` and 4N', and its one step is the direction
quarter, so no answer coarser than a pixel earns the whole. The penalised score (`2s - 1`) is the same: full credit +1,
within the need +0.5, within twice the need 0 (as good as `unknown`), the direction alone -0.5, wrong -1. `unknown` is
as before: on an answerable gold it scores 0, on an `unknown` gold +1 and any answer -1. Since 0.10.0 this penalised
score is the **gate score** (the row's `score_gate`): every gate, `knowable` and the evidence verdict read it. The score
the reports, the headline and the reference rows read is the **reported score** (the row's `score`), which is the same
except for a catastrophe (below). The kernel is `credit_kernel` (standard library only; the scorer, the reports and the
compile all import it), with `KAPPA` 0.1, `MIN_RHO_M` 0.01, `HARM_FACTOR` 0.5, `FP_WEIGHT` 0.5, `ANGLE_DEG` 45 and
`MATCH_REACH` 4.

**The direction quarter, by answer type.**
- an *offset* (where a thing or a person is from the camera): the answered bearing within 45 degrees of the gold's, or
  `e <= N`;
- a *number* (how far the body could move, how close a person came): the answer and the gold on the same side of
  `T = L + N`, the body's length plus the need (for a corridor, is there room to move at all), and, where the gold is at
  least `T`, the answer within a factor of two of it;
- a *place in the map*: `e <= N` (N is R: going there, the robot finds the thing in view). The room the thing is in
  earns nothing: it is a fact of the map and the kind, which a system without the camera knows too (0.9.0 review);
- a *heading* (`open_heading`, credit `regret`; 0.11.0): room to move. The answered heading's run is `T = L + N` long or
  longer (the body's length plus the need), or at least the best run less a pixel where no heading has that much.
  (Through 0.10.0 it was a run of at least half the best.) A heading of the way to a remembered thing (`way_to`, credit
  `route`) has the direction unless it has no outcome: no room to move, or into floor never seen;
- a *list item* (0.11.0): a place, so `e <= M` (M is the list's match radius). An item matched farther away than M earns
  its other three quarters and no direction, as a place does beyond R (methodology rule 7). Through 0.10.0 a matched
  pair earned the direction at any distance up to `MATCH_REACH` times M, which paid a sensorless prior that put places
  over the densest part of a room.

**Harm costs more than caution.** On the harmful side the need `max(N, 0.1 g)` is halved (N' becomes half of it in all
three anchors; `rho` and the `2 rho` floor stay): a clearance answered longer than the gold, a closest approach answered
wider, an offset whose error points away from the camera along the gold's bearing (a thing or a person placed farther
away than it is), and a heading without room to move (a run under `T`; and for `way_to`, a heading with no outcome). A
place to re-find a thing (`object_where`) has no harmful side. A cautious error of 2N' and a harmful one of N' leave the
robot the same margin, so an over-estimate of the door costs more than an under-estimate of the same size (6.0 m is
-0.42 and 4.0 m 0.00 on the helper). Where one pixel is half the need or more (`rho >= N'/2`) the need is already
widened to `2 rho` on both sides, so the halving does nothing there, and between a quarter and a half of the need the
harmful need is `2 rho` and not N'/2: the kernel's design (on the vacuum's corridor `rho / C` reaches 0.50 at its 90th
percentile). The catastrophe rule is unchanged (below): its trigger is the body's length plus C and its margin
`max(C, 2 rho)`, not N'.

**A catastrophe is charged in the reported score only (0.10.0).** An answer whose action would end in harm scores minus
a weight, whatever credit it earned (`score.reported`; `score_catastrophe.weight` reads the rule on the gold row). It needs
a harmful error larger than `margin_m` = `max(C, 2 rho)`, so an answer that is merely imprecise is never one; `unknown` is
never one, and a gold at the question's cap is exempt (nothing nearer was seen, so nothing was wrongly cleared). The weights
are `credit_kernel`'s, stated and never tuned to a score: `CATASTROPHE_THING` = 2 for contact with a thing the robot saw,
which damages or stops the job and a person clears, and `CATASTROPHE_PERSON` = 3 for contact with a person or a hazard of
the robot's own kind (a tangle), which endangers someone or ends the job until a person intervenes. The rule by family:
- `over` (a number): `clear_path`, weight 2, and `closest_person` and `closest_approach`, weight 3: the answer exceeds the
  gold by more than the margin, and the gold is under `threshold_m` (the body's length plus C) and the answer is at or
  over it. `clear_path` had no threshold until the final review's S5, so any overstatement past the margin was a collision,
  including a 30 m answer to a 24 m drone corridor; now only a corridor shorter than the room the body needs, said to be
  one it can drive, is;
- `radial` (an offset): `person_mem`, weight 3, which no job asks since 0.10.0: the person is placed farther from the camera
  than they are, along the gold's bearing, by more than the margin;
- `miss` (a list): `drop_zone_people` and `tangle_ahead`, weight 3: some point of the row's `critical` list has no answered
  item within the row's `match_m`. A critical person is one whose hull lies within C of the point below the drone; a
  critical tangle item has its hull within `hazard_rule.FRONT_M` = 0.3 m of the body's front edge (1 s of travel at the
  vacuum's 0.3 m/s). The prompt says so (final review S7): a job that asks `tangle_ahead` during the job is told that a
  list with no place within a matching radius of a thing its brush could catch within 30 cm of its front edge scores -3
  (`llm_credit.LIST_CRITICAL`, `brief.list_hazards`); one that asks `drop_zone_people` would be told the same of a person
  within its C of the point below. A job that asks neither (the helper and the AMR, whose list is `task_things`) is told
  nothing of it.

The families that carry a rule from `rules.CATASTROPHES` (`clear_path`, `closest_person`, `person_mem`; the last is
asked by no job) get it from the compile (`rules.with_catastrophe`, called by `questions.records`); the new families set
their own. The weights and the reasons are in [question-design.md](question-design.md), "How answers are graded", with a
replay on 0.9.0's answers. The reports give the catastrophe rate ([eval.md](eval.md), "The splits").

**A list scores per item.** The answer's items outside every don't-care circle are matched first, one to one to the
gold's by maximum summed credit (the Hungarian method, no scipy), each pair within `MATCH_REACH` times M, each matched
item earning its own four quarters at its own resolution (its direction quarter is earned within M, the place rule
above). The items inside a circle are then matched the same way to the gold items left over, so a circle can only add
credit: an item inside it never takes a gold item from one outside it, and adding one never lowers the credit (0.11.1;
before, all the items were matched together, and an item inside a circle could take the gold item of a better one
outside). An answered item outside every circle that matches nothing is false; one inside a circle that matches nothing
is neither true nor false. Then `s` is the matched credit over
`|gold| + FP_WEIGHT |false|`: a missed item costs one and a false item half. An empty gold's one thing is the empty
list, so `s = 1 / (1 + FP_WEIGHT |false|)` (0.11.0): one false place scores 2/3, as it does beside a list of one thing
found, where before 0.11.0 it scored 0, the same as ten.

**A decision scores by its regret.** The answer is an action (a heading); its regret is what it gives up against the
best heading, and the credit is the four quarters on `e` = regret.
- **`open_heading`** (credit `regret`): the regret is the best run less the answered heading's run, in metres, N = C,
  `g` the best run (so N' = max(C, a tenth of the best run, 2 rho)), and the direction is room to move (above). In 0.9.0
  and 0.10.0 the credit was a cliff and was counted in no headline mean: a run changes by metres within a few degrees at
  a doorway, while the regret was read against N = C (10 to 15 cm) and the direction was a run of half the best, so a
  heading 2 to 10 degrees off the right opening scored -0.4 to -1, as did one that left 1 to 2 m of room
  ([families/open_heading.md](families/open_heading.md), "The credit is a cliff", kept as history). Since 0.11.0 it is
  the same kernel on the same need rule as every distance, the family counts in the clearance ability, and the robot's
  own log alone scores +0.04 on the helper and -0.14 on the vacuum under it (over 0.10.0's rows), below the shortcut
  line.
- **`way_to`** (credit `route`, 0.11.0): the gold row holds, for each of the 360 headings at 1 degree, an outcome in
  metres in `profile_m` (null where the heading has no room to move or ends on floor never seen): how much nearer to the
  thing one `step_m` (R, 1.0 m) of driving along it brings the robot, by the shortest way its body fits through the
  floor seen or driven (`route_field`). The regret is the best outcome less the answered heading's; a null heading's is
  the best outcome plus `step_m`, with no direction and the harmful side, which scores -1. N = C, `rho` the field's 0.1
  m cell (`resolution_m`), `g` = `step_m`; for the helper (C 0.15, R 1.0) a tenth of R is under C, so the distance term
  never bites, but 2 `rho` = 0.2 m is over C, so the need the route is credited on is N' = 0.2 m
  (`score_graded.route_credit`): a heading 0.2 m short of the best outcome scores +0.50 and one 0.4 m short 0.00. The
  row's keys are `profile_m`, `step_deg`, `step_m`, `resolution_m` and `need_m`.

**The resolution `rho`** is written into every gold row from the frame the gold rests on, so full credit is what an
ideal reader of the frames reaches:
- a floor point at floor distance `d` from a camera `h` above it: one pixel radially, `(d^2 + h^2) / (h fy)`, combined
  with one pixel laterally, `sqrt(d^2 + h^2) / fx`, at least 1 cm (`rules.resolution_m`; it has none where `h` is at
  most `CONTACT_M`);
- an offset (`ego_now`, `ego_mem`, `person_mem`): at the camera of the evidence sighting that set the surface tie (the
  last evidence frame when no contact did), to the gold point;
- a corridor (`clear_path`, and the best run of `open_heading`): the largest of the grid's 2 cm sub-cell, the stopper's
  floor contact error in the frame the stop rests on, and the tie gap from the stop to that contact; a flier's wall: one
  pixel at the wall's foot from the evidence frame, at least 2 cm, and asked only where that is at most C/2 (0.11.1);
- a route heading (`way_to`): the field's fine cell, 0.1 m (`grid.FINE_M`), the resolution the outcomes are computed at;
- a closest approach: one pixel at the nearest evidence sighting;
- a person in `drop_zone_people`: one pixel at the person's hull centroid, from the ask frame's camera;
- a place (`object_where`): the larger of one pixel at the last recognisable sighting and the hidden widening;
- a list item: one pixel at the item's last recognisable sighting.

On the dev root's compile the median `rho / N` runs from 0.01 (the helper's places) through 0.04 (the drone's
corridor), 0.07 (`closest_person`), 0.08 to 0.09 (the helper's offsets), 0.13 (the helper's corridor and heading) to
0.19 to 0.21 (the AMR's `ego_now`, the vacuum's corridor, heading and `ego_mem`); its 90th percentile reaches 0.50 on the
AMR's `ego_now` and the vacuum's corridor.

**What each family credits** (the gold row's `credit` and its `tolerances`; [format.md](format.md) has the keys):

| Family | Credit | Need | Harmful side | Direction |
| --- | --- | --- | --- | --- |
| `ego_now`, `ego_mem`, `person_mem` (`offset`) | `graded` | C, or a tenth of the range if more | the error points away from the camera along the gold's bearing (`radial`) | bearing within 45 degrees, or `e <= C` |
| `clear_path`, `closest_person`, `closest_approach` (`number`) | `graded` | C, or a tenth of the gold if more | the answer is longer (`over`) | same side of `T = L + C`, within a factor of two |
| `object_where` (`position`) | `graded` | R | none | `e <= R` |
| `task_things`, `drop_zone_people`, `tangle_ahead` (`positions`) | `graded_set` | M (the match radius; widened by the largest item for `task_things` and `tangle_ahead`) | a false item costs half | the item is matched |
| `open_heading` (`heading`) | `regret` | C, or a tenth of the best run if more | the heading's run under `T` | room to move: a run of `T`, or the best less a pixel |
| `way_to` (`heading`) | `route` | C, widened to 2 `rho` = 0.2 m by the field's cell (`g` = R, so the tenth never bites on the helper) | a heading with no outcome (no room, or floor never seen) | the heading has an outcome |

The job's numbers, declared in its file (`JobSpec.clearance`, and `reacquire` where it differs from `match_m`), with their
sources. C is never a detection size, a docking or placement precision, or a
facility-layout rule:

| Job | C | Source of C | R | Source of R |
| --- | --- | --- | --- | --- |
| vacuum (`vacuum/1`, `vacuum/2`) | 0.10 m | the loose end of what camera vacuums aim to pass beside an obstacle: Roborock says the S7 MaxV gets close to shoes, and a secondary guide calls a 15 cm berth wasteful and a pass within about 3 cm of a cable better (the smallest objects they detect, 15 mm and 5 x 3 cm, are no clearance) | 1.0 m | HomeRobot's real-world find rule (the object visible in RGB within 1 m) and Habitat's ObjectNav success radius; R/2 = 0.5 m is the job's `match_m` |
| helper (`helper/1`) | 0.15 m | OWMM-Agent counts a pick when the end-effector is within 0.15 m of the target (its looser option is 0.8 m); a compliant hand absorbs 2 to 5 cm of offset, which fits the C/2 rung | 1.0 m | as the vacuum |
| AMR (`amr/1`) | 0.45 m | the MiR250 is 580 mm wide and needs a 1,450 mm corridor by default, 0.435 m a side (its side protective fields are 210 mm plus a 65 mm tolerance); ISO 3691-4 Annex A.2.1's 0.5 m beside the path is a layout rule and corroborates the order of size only; docking is fiducial-guided at millimetres and is not C | 1.0 m (`match_m`) | a convention: no warehouse source gives a find radius |
| drone (`drone/1`) | 2.4 m | Wing's patent US12399508B2 keeps a delivery point at least 10 ft from a building and 8 ft from trees and counts things over about 3 ft as obstacles; Amazon and Flytrex advise clear areas of the same order; Zipline's two feet is how precisely it places a package, not a clearance | 2.0 m (`match_m`) | Wing's patent US12221211B2: it may be preferable never to deliver more than about 2 m from the intended point |

The home jobs declare `reacquire = 1.0` (their `match_m` is 0.5 m); the AMR's and the drone's R equals their `match_m`.
The row's `range_m` is the slant range from the camera to the thing, kept for the reports; it sets no credit. The prompt
states the credit from the same numbers (`llm_render.credit_sentence`, passed `JobSpec.match_m`, `clearance_m` and
`reacquire_m` by `cmd_codex` through `codex_system`): the four quarters (the right direction; within twice the margin
the job keeps, C, or R for a place; within that margin; and as exact as one pixel of the camera), that for an offset, a
distance in metres or a heading the margin is C or a tenth of the right distance (0.11.0: "from your camera to the right
point, the right number of metres or the best heading's run"), whichever is larger, that overstating how far the robot
could move, or placing a thing or a person farther away than it is, counts against the answer twice as fast, that a list
earns each item it holds the same way (the things and the places paired in the pairing whose summed credit is largest; a
missed item costs a whole one and a false one half; the prompt leaves out that the scorer pairs the items outside every
don't-care circle first, above), that a heading earns credit by how much room it gives up against
the best heading, its right direction being room to move, and, for a job that asks `way_to`, that the way to a thing is
credited by how much nearer to it R of driving along the heading brings the robot, by the shortest way its body fits
through as far as it has seen; that its margin is "the largest of 15 cm, a tenth of 1 m and twice the 10 cm cell of the
floor that the way is worked out on" (20 cm on the helper: `llm_credit.route_margin`, the scorer's own number) and its
quarter 4 whole within one cell (10 cm) of the best heading's progress; that any heading with the room the body needs
("the robot body's length plus 15 cm") has the right direction, even one that leads away from the thing; and that a
heading with less room than the body needs, or one into floor you have not seen, scores as a wrong answer.
The last bullet of the credit section says that a question may name a thing the camera never showed clearly enough to tell what
it is, or one that is not in this place at all, and then `unknown` is right (0.11.0; true of every named thing by
construction, and told to every job). It defines the point an offset names (`llm_credit.OFFSET_TARGET`, "a thing or a
person") and the point a place names (`llm_credit.PLACE_TARGET`: the middle of the thing where it stood, its x and y in
the map frame, which is the point the golds of `object_where` and `task_things` hold), states the robot's body from
`robot.json` (`llm_render.body_sentence`: its shape and size, its height, that the base is its middle, and for a ground
robot the climb and the band's top), and defines the corridor (`llm_credit.CORRIDOR_TARGET`: ahead, back from the rear
edge, or turned in place left or right and driven from the front edge) and the heading (`llm_credit.HEADING_TARGET`).
Each part is told only to a job whose own questions need it (`brief.answer_types`, `asked_families`, `asks_turns`): the
drone, which flies ahead and back only, is told no turned move and no place, and the AMR no corridor. The right
direction it states for a distance in metres is the same side of "the robot body's length plus {C}" ("the room the body
needs": the job's clearance, 10 cm on the vacuum and 15 cm on the helper, and never the margin, which the tenth of a far
distance widens) and, above it, within a factor of two; "under it, there is no room to move" is said
of the corridor only. `rules.relative_ladder` stays for the change families (`object_was`, `same_place`), which no job asks. Reports bin `ego_now` by `ego_frame.band` (near,
mid, far at 6 and 16 of the job's units, `JobSpec.unit_m`).

**The interfaces.** `JobSpec.clearance` is read as `clearance_m`, which raises ValueError naming the job while it is unset
(every registered job declares one); `JobSpec.reacquire` as `reacquire_m`, the job's `match_m` when it declares none.
`rules.resolution_m(cam_pos, xy, base_z, fx, fy)` is the one-pixel error of a floor point and
`rules.graded_tolerances(rho, need, harm, threshold=None)` builds a `graded` row's dict;
`rules.hidden_widening(size, cam_z, top_z)` is `rules.widening(size)` for a thing whose top is above the camera and 0
otherwise. `rules.clearance_ladder(c)` gives `[c/2, c, 2c]` and stays for the rows of a compile before 0.9.0.

**Reference.** `ego_now`'s declared reference is `kind_range`, a scripted baseline that takes the true bearing and puts
the thing at its kind's typical range, the median floor distance of any other script's gold of the kind (`unknown` for
a kind none has a gold of): held out on release-c in 0.6.0, -0.009 (AMR), +0.04 (drone), +0.30 (helper) and -0.129
(vacuum), so not a floor everywhere. It is not a gated shortcut, and distance is reported above the larger of it and
`all_unknown` per job, a reference that abstaining cannot beat (`stbench eval distance RUN KIND_RANGE ALL_UNKNOWN`).

### Look points

A **look point** is where a look family (`JobSpec.looks`) asks during the job. Since 0.10.0 it can be **any frame of the
case**: a system gets one frame at a time and the ask follows the frame it is about, so no frame has to end anything
(`stream._during_at` accepts any step of the window; until 0.10.0 a point had to be the last step of a batch, then of a
look window). `questions_looks` draws the points:

- the candidates are the frames at least `LOOK_MIN_STEPS` = 4 steps into their task, except the task's last frame;
- the frames of the job's priority decisions (`JobSpec.look_decisions`: the moments its task turns on, such as before a
  crossing, a grasp or a lowering) come first in the draw, then every other candidate, each group in `stable_hash(case id,
  k, "ask")` order, as decision points are;
- a frame is kept when it is at least `LOOK_GAP_STEPS` = 4 steps from every frame kept, until the case holds
  `points_cap(frames)` = `ceil(frames / LOOK_PER_FRAMES)` points, with `LOOK_PER_FRAMES` = 6. Six, from the replayed draw
  on the dev root: at 8 a point falls once in 7.8 frames, at 6 once in 5.9, and the gap keeps two points four frames
  apart, so the draw reaches that limit near one in 5.1 and more cap below 6 buys nothing;
- the draw reads the case id and the plan's decision points, never the world or a gold, so which points are asked
  reveals nothing about what is in view. A question being asked can show that a frame is a decision moment; the status
  and subgoal already show that, and the plan decides where they fall;
- a point has as many slots as the job's table declares (`JobSpec.look_shares`; 0.10.0's jobs use three: the harm
  questions, what the robot saw before, the rest), and slot `i` draws its family from the table (per slot a list of
  (family, percent) pairs summing to at most 100): the family whose cumulative range holds `stable_hash(case id, k,
  f"slot{i}") % 100`, or none when the hash falls past the slot's total. The drawn family asks at most one candidate,
  taken through the same turn an end moment uses (a sliced family such as `ego_mem` needs a candidate allowed in both
  its pools); a drawn family with no candidate asks nothing, no other family is tried, and the point's slot is a dead
  band. The draw reads the case id, the point and the table, never the world, a gold or an answer, so which family is
  asked reveals nothing of what is in view. A job that declares no table gets one slot per family of `JobSpec.looks` at
  100; a registered table whose percents pass 100, or that omits a family of `looks`, is refused at registration,
  naming the job;
- a hazard family is asked only in the status phases the job declares for it (`JobSpec.look_phases`, as `clear_phases`
  limits `clear_path`); `drop_zone_people` and `tangle_ahead` raise for a job that asks them without declaring phases, so
  the compiler names no robot's phase;
- a look question whose gold class is `unknown` is dropped unless `stable_hash(case id, family, str(k), "unknown") % 100`
  is under the family's keep share (`JobSpec.look_unknown_keep`; absent is 100, and a sliced family's share is the slice,
  so a keep on it is refused). It reads the class only, as `CAPPED` thins `task_things`' empty golds.

A look question carries the point as its `decision_k`. Each question is its own ask: two questions at one point are two
asks in turn, and the second is sent after the first is answered.

**The recent view slides with the point.** `ego_mem` and `person_mem` (which no job asks now) ask of a thing, or a
person, not seen in the last `LOOK_WINDOW` = 8 frames of the current task up to the point (`look_windows.recent`,
`World.recent`). Before 0.10.0 the view was the look window that ended at the point, a run of up to 8 frames cut at
decisions. The harm families name the same stretch as their subject (`questions_harm.stretch`, `w:<first>-<k>`), so the
subject rule lets only one of two questions fewer than 8 frames apart be asked about the same person or cable. The end
questions read the case's last 8 frames (`World.end_window`) as their recent view.

The decision windows (`look_windows.windows`, `World.windows`: runs of at most `LOOK_WINDOW` steps, cut after every
decision point and every task's last step) are no longer sent to a system. They stay for the unasked families that read
them (`path_ahead`, `where_was_i`, `out_of_view`, `zone_clear`, `clearance_then`) and for `stbench view`.

**The points a case draws, and what they ask,** on the dev root (36 cases a job; short and medium cases, 32 a job), on
the 0.10.0 compile after lane J2 (lane J2 changed the AMR's row alone, and the helper's row predates the removal of
`person_mem` from its slot 2). The 0.11.1 release holds 245 questions on the AMR, 262 on the drone, 541 on the helper
and 711 on the vacuum, in all lengths:

| Job | Frames | Look points | Look questions | End questions | One question in N frames (all questions) |
| --- | --- | --- | --- | --- | --- |
| AMR | 4,760 | 804 | 127 | 66 | 1 in 24.7 |
| drone | 2,530 | 433 | 201 | 0 | 1 in 12.6 |
| helper | 3,577 | 608 | 286 | 98 | 1 in 9.3 |
| vacuum | 5,811 | 981 | 416 | 21 | 1 in 13.3 |

[question-design.md](question-design.md), "When questions are asked", has the fill of each family and why the AMR is
supply-limited.

### The order of the draw

`questionnaire.ask` draws one case in this order:

1. **The end questions**, by the rules of the next section. Their subjects are reserved; `task_things` carries none.
2. **The looks**, over those subjects and the looks so far (the slot draw, then each look family's `unknown` keep): no
   look names a reserved subject, and no entity is named twice. A thing a look names stays a subject for the rest of the case: the during exemption of the old rule ends for
   things. Reserving the end questions' subjects first leaves `ego_now` 36 to 93 questions a job (reserving
   `task_things`' items as well would have left 20 to 29 on three jobs).
3. **`RELAX`**: a thing a look names, and a thing an end question names, becomes don't-care in the end families that
   relax it (`task_things`), and their `answer_class` is recomputed (an items question whose items are all now
   don't-care is empty).
4. **`CAPPED`**: the empty golds of a capped family are thinned by the job: an empty question is kept when
   `stable_hash(case id, family, key, "empty") % 100` is under `JobSpec.task_things.empty_keep_percent` (100 keeps
   every one; a job over a third empty is set to `100 (1 - p) / (2 p)` before the binding read). It reads the class
   after `RELAX`.

The ask order is the during rows (decision points and look points) by keyframe, a decision point's before a look
point's at one keyframe and a point's slots in order, then the end rows by `stable_hash(case id,
candidate key)`.

### Classes and draws

Each gold row carries `answer_class` (null for a family that declares none) so that the balance check can read a
shortcut's winnings per class ([balance.md](balance.md)). The compile draws the classes, by hash, never from a system's
answers:

| Family | Classes and share | The draw |
| --- | --- | --- |
| `ego_mem`, `way_to`, `object_where` | `answerable` 80, `unknown` 20 | per moment (per look point for `ego_mem` and `way_to`, key `str(k)`) by `UNKNOWN_PERCENT`; asked only if both pools hold an allowed candidate |
| `same_place` | `yes` 50, `moved` and `vacated` 50 | a coin per moment (`COINS`, `COIN_PERCENT`) |
| `object_was` | `earlier` 50, `later` 50 | a coin per moment |
| `task_things` | `empty` at most a third, `items` | thins empty golds only (`CAPPED`), to the job's keep share (`empty_keep_percent`: vacuum 20, helper 26, AMR 43) |
| `ego_now` | none | results are reported by range band (`ego_frame.band`) |
| `clear_path` | `answerable`, `unknown` | the world's, not drawn: a ground corridor is `unknown` when under 80% of the counted cells of the window up to the point were seen free, a flier's when no frame of the window saw the wall; the drone keeps 75% of its `unknown` golds by hash and the helper 35% |
| `person_mem` (asked by no job since 0.10.0) | `answerable`, `unknown` | the world's, not sliced: `unknown` only when the window holds no row of any person, at any pixel count (a person seen but never as evidence asks nothing); the helper kept 7% of its `unknown` golds by hash; a sighting is asked once a case (subject `w:<j>`) |
| `open_heading` | `answerable` | the world's: one pool, so the balance gate has one class to read; a point where no heading was seen asks nothing, so there is no `unknown` gold |
| `closest_person` | `person`, `no_one` | "no one" is asked only with no person row of 256 px or more within 10 m, and kept for half of those cases by hash |
| `closest_approach` | `person`, `no_one` | a gold at the cap (10 m) is "no one", asked only with no person row of 256 px or more within 10 m, and kept for 2% of its points by hash (`NO_ONE_KEEP_PERCENT`), so 10 is never the common answer |
| `approach_mem` (0.11.0) | `person`, `no_one` | as `closest_approach`: a gold at the cap (10 m) is "no one", asked only with no person row of 256 px or more in view, kept by hash for `APPROACH_MEM_NO_ONE_KEEP_PERCENT`, declared 100 (the proxy share at the cap is 22% on the AMR and 0% on the helper, both under a third); if the compile measures a share p above a third on a job, it becomes `floor(100 (1 - p) / (2 p))` for the job with the largest p, as `closest_approach`'s does |
| `drop_zone_people`, `tangle_ahead` | `items`, `empty` | the empty list is kept for 1% (`drop_zone_people`) or 13% (`tangle_ahead`) of its points by hash (`EMPTY_KEEP_PERCENT`), the floor of 100 (1 - p) / (2 p) for the measured share p of empties, so the class is at most a third of the rows |

A candidate carries its pool's class, and the draw refuses one whose class does not fit its pool
(`records.BALANCE_POOLS`, `pool_of`: the first pool is a moment's `answerable`, the second its `unknown`).

### Gates per family

| Family | Gates read | Dropped on a job when |
| --- | --- | --- |
| `ego_now` | ceiling, knowable, shortcuts (`kind_range` excluded), separation from the best of the shortcuts and `kind_range`, balance | a shortcut, separation or balance fails |
| `ego_mem` | the same, plus `memory_needed` | `memory_needed`, a shortcut or balance fails |
| `object_where` | as in 0.5.0, balance | as before |
| `same_place`, `object_was` (asked by no job since 0.6.0) | plus `time_needed` | `time_needed`, `form`, `blind` or balance fails, or fewer than 0.5 a medium case |
| `task_things` | `memory_needed`, shortcuts, separation, balance | any of them fails |
| `clear_path` | ceiling, knowable, shortcuts and separation without the references (`map_reader`, `other_case`), `memory_needed` on its `memory` rows with `current_view` and `current_view_prior` carrying no pose key, `path_log` from the robot's own log alone, balance; `stbench check` reports `map_reader` for it as a diagnostic | a gated shortcut, separation, `memory_needed` or balance fails |
| `open_heading` | as `clear_path`: ceiling, knowable, shortcuts and separation without the references, `memory_needed` on its `memory` rows, `path_log` from the robot's own log alone, and a balance rule that reads its one pool (`answerable`) against `all_unknown`; `stbench check` reports `map_reader` for it as a diagnostic | as `clear_path` |
| `closest_person`, `person_mem`, `closest_approach`, `approach_mem` (0.11.0), `drop_zone_people`, `tangle_ahead` | as `ego_mem` (`memory_needed` on the `memory` rows, every shortcut gated, separation, balance) | any of them fails (`person_mem` failed `separation` on the helper and is asked by no job) |
| `mover_next` | CV at the ceiling, `prediction_needed` | stays a diagnostic |

Every gate reads the penalised score `2s - 1` (the row's `score_gate`), never the reported score with the catastrophe term
(methodology rule 7): no harm weight moves a verdict.

A cell (a job's ability in one column) with fewer than 20 questions in the dataset weighs `n / 20` of a full cell in the
headline ([eval.md](eval.md)); it is not a diagnostic (through 0.10.0 it was left out of the headline). Only
`prediction`, which no job asks, is a diagnostic ability. `drop_zone_people` was thin by supply (2 questions on the dev
root) but the weight is per cell and not per family, so its rows sat inside the drone's clearance cells; the
final review's S2 took it off the drone.

### The task-critical questions (0.8.0, extended in 0.9.0)

The families of 0.8.0 ask what a robot's job depends on: can it move without touching anything, who is near, and where
is the person it last saw. 0.9.0 asks the corridor in every direction the robot can move and adds the question of which
way has the most room. Each has a page; what they share:

- **The corridor** (`clear_path`, [page](families/clear_path.md)). `grid.corridor_clear` reads the stop at the 2 cm
  sub-cells (the 0.7.0 rule's 0.1 m cell was a median 4 cm short of the obstacle) and treats a surface at most 3 mm over
  the climb (`grid.DEPTH_QUANTUM_M`) as floor, for every robot. The gold, `table_space`, `table_clear` and every mapper
  call the one function. A ground question is asked only if the stopper is tied to a floor contact a camera can read to
  within C/2 and is visible (a wall, a fixed piece or a recognisable thing); only if no obstacle within 1 cm of the
  body's side, and no height within 0.5 cm of the climb or of the band's top, decides the gold; and only in the phases
  in which the robot drives (`JobSpec.clear_phases`). A flier's corridor is asked at least `JobSpec.fly_floor_m` up,
  against the map's wall boxes, answerable where a window frame saw the wall, and never into open air.
- **Every direction** (0.9.0). The direction is drawn at each look point by `stable_hash(case id, k, "direction")` over
  the job's `clear_directions`, from the case id and the point alone. A ground robot's gold reads what it last saw of
  every cell of the whole window up to the point (truth rule 2), not the latest frame alone; the corridor starts at the
  front edge (ahead), the rear edge with no turn (back), or the front edge after turning in place by 90 degrees (left,
  right). A box body's turn must clear the seen obstacles inside the disc of its half-diagonal. Space the camera did not
  see is `unknown`, so a corridor behind or beside the robot needs memory, or the knowledge that it never looked there.
- **The decision** (`open_heading`, [page](families/open_heading.md)). Which heading, turned in place, gives the body
  the longest straight run before it touches anything it has seen. The answer is the action and it is scored by its
  regret. The gold is the profile of runs at 1 degree steps over the same window cells, so the corridor rule of
  `clear_path` decides it at every heading (`clear_profile`).
- **The people rule** (`person_seen.evidence`, [closest_person](families/closest_person.md)). A sighting of a person is
  evidence with 1,024 px, `RECOG_LUM`, a depth at every pixel, the contact tie at C, and a box off the bottom edge of
  the frame: the top and the sides may cut a person, whose feet place them. The oracle channel's person `floor_hull`
  follows the same function, so the table baselines reproduce the people golds.
- **Two structure families.** `clear_path` and `open_heading` ask how far the body could move or which way has room,
  which is partly the place's fixed structure, and the maps are public. The agent is never given a map, so the structure
  is perceived and remembered; the `map_reader` (the public map's raycast from the delivered pose; on a held-out place,
  which has no map, the pose-keyed pool of other scripts' golds) is the declared reference of both, and `other_case` is
  a reference too. `stbench check` reports both for these families as design diagnostics (the map reader is not a
  published result) and neither is a gate; the other baselines stay gates, `path_log`
  answering from the robot's own log alone and `current_view_prior` without a pose key. Only these two are structure
  families; making another needs methodology rule 5 amended, not a gate's failure. On a public place the score is a
  claim only about a system that holds no copy of the place's map or data.
- **The harm families of 0.10.0** (`closest_approach`, `tangle_ahead` and `drop_zone_people`, which no job asks since the
  final review's S2; each has a page). Each asks about what the robot's next action meets, names no entity, and takes
  the recent view as its subject (`questions_harm.stretch`), so the same person or cable is not asked about twice within
  8 frames. Each keeps a share of its empty or "no one" golds so that the empty class is at most a third of its rows,
  carries a `catastrophe` tolerance (`questions_harm`), and, for the two list families, is asked only in the status
  phases the job declares (`JobSpec.look_phases`). The tangle kinds and the shared constants are `hazard_rule`'s, and
  the arithmetic of `closest_approach` is `approach_rule`'s, which the gold and the table systems both call. The robot's
  own velocity is delivered with every step (`velocity`, `case.base_velocity`: the controller's odometry), so a
  `closest_approach` row whose evidence person stands, or has no earlier sighting to move against, is `now` (frame k and
  the step's velocity give its gold, its age 0) and a row whose person moves is `memory` (it needs their earlier
  sighting). The dev data hold two such walkers, both on the AMR. A person's motion counts only when the sensors pair it
  unambiguously (exactly one evidence person in the ask's frame and in the earlier frame used, and no other person row
  of 256 px or more), since a detection's key names nothing; otherwise the point is a dead band
  ([families/closest_approach.md](families/closest_approach.md)).
- **`approach_mem`** (0.11.0, [page](families/approach_mem.md); AMR and helper) is `closest_approach` for the person
  now out of view: "if you kept moving as you are now, how close would you come in the next 5 seconds to the person you
  saw last, where you last saw them?" Its gold is the robot's own motion composed with the last sighting of exactly one
  person (`approach_rule.last_seen`, `closest_m`), held where they were seen. The gold asks only where that person stood
  still when seen (the `WALKING` screen of `questions_approach`) and has not moved since (a screen by the truth); the
  table systems apply neither and hold the person still. Where the sensors cannot place that person it is a dead band,
  and where someone is in view it is `closest_approach`'s. It is a clearance `memory` question with the `over` harm of
  `closest_approach`, and is asked in any phase. The dead-band counts are on its page.
- **Evidence age.** Every gold row records `age_s` and `age_band` (below, "How a gold is computed").
- **The headline.** Clearance is one ability with a cell in each column (`clear_path`, `closest_approach` and
  `tangle_ahead` by evidence, `drop_zone_people` in `now` (for a job that asks it, which none does), `closest_person` and
  `approach_mem` in `memory`, and `person_mem` in `memory` for a job that asks it, which none does); a column is the mean of its
  counted cells by their weights ([eval.md](eval.md), "The splits"). `open_heading` is a clearance question since 0.11.0
  (the ability `heading`, a diagnostic reported and counted in no mean, through 0.10.0), and a cell weighs `min(1, n / 20)`
  for its `n` questions where it once left out a cell of fewer than 20 (methodology rule 4); a column weighs the largest
  weight of its cells and a cell, in its column, the largest weight of its rows. The reports also split by kind:
  harm, help, situation and reuse (`by.kind`).

## What each job asked in 0.5.0-control (history; the current mapping is above)

A job names what it asks: a during family by the decision kind it is asked at (`JobSpec.decisions`) or at look points
(`JobSpec.looks`), an end family in `JobSpec.ends`. A family no job names is never asked, so adding a family changes no
existing compile. The table below is what the four jobs named in 0.5.0-control, before 0.6.0 retired its during
families and `adjacent` (above); it was read from `job_vacuum2.py`, `job_helper.py`, `job_amr.py` and `job_drone.py`. Each job asks one family during the job, `zone_clear`, about the
place its next manoeuvre uses, and ends with `object_where`; of the three change families only `object_was` is named by
a job, and only by `amr/1` (below).

| Job | Family | When asked | What it asks | Answer type | Scope |
| --- | --- | --- | --- | --- | --- |
| `vacuum/2` | `zone_clear` | during, at `before_room` | whether the doorway ahead is clear of anything taller than the robot climbs, right now | `yesno` | `now` |
| | `object_where` | end, once per answerable thing | where it last saw a thing, when the thing did not move by a metre | `position` | `memory` |
| `helper/1` | `zone_clear` | during, at `before_place` | whether the spot in front where it will put the thing down is clear, right now | `yesno` | `now` |
| | `object_where` | end, once per answerable thing | where it last saw a thing, when the thing did not move by a metre | `position` | `memory` |
| `amr/1` | `zone_clear` | during, at `before_station` | whether the bay ahead is clear, right now | `yesno` | `now` |
| | `out_of_view` | during, at `before_undock` or `before_crossing` | where the movable things on the floor within 5 m that are not in the images just sent were, as it last saw them | `positions` | `memory` |
| | `object_where` | end, once per answerable thing | where it last saw a thing, when the thing did not move by a metre | `position` | `memory` |
| | `object_was` | end, once per event moment of each task | where a thing that moved was at that moment, at its earlier or its later state | `position` | `memory`, temporal |
| `drone/1` | `zone_clear` | during, at `before_lower` | whether the drop zone is clear, right now | `yesno` | `now` |
| | `object_where` | end, once per answerable thing | where it last saw a thing near the drop point (within the 12 m its camera sees), when the thing did not move by a metre | `position` | `memory` |

A job's `zone_clear` is `now`: its gold reads the look window that ends at the decision and no earlier frame (a blocker
in it certifies "no", at least 80% of the zone's cells seen free "yes", else `unknown`), so the recent view answers it,
and a job answers it by a response after the decision (go, abort, skip, elsewhere: `DecisionPoint.response`) that
follows the zone's true state and that the question therefore never sees. `distance_between` is named by no job in
0.5.0-control: its module, page and tests stay, and the evidence verdict, which reads the gates per slice, finds the
helper's constant that pooling hid (+0.375 on the short cases). The briefs say nothing of during questions beyond the
sentence the job's decision kinds compose ([agent.md](agent.md), "The brief"). Where a job names several during families
for one decision kind, the family a point asks is drawn from the case id and the point alone (`stable_hash(case id, k,
"family")`), so which family is asked reveals nothing of the world; no job names two at one kind. `adjacent` is named in
the `ends` of `vacuum/2` and `amr/1` and is asked on held-out layouts only; the first draft has no held-out layout, so
no job asks it ([families/adjacent.md](families/adjacent.md)).

**The change families.** `object_was`, `same_place` and `person_where` ask what only a record of the past can answer
(`FamilySpec.temporal`): where a thing that moved was, at its earlier or its later state
([families/object_was.md](families/object_was.md)), whether it was in one place at two times
([families/same_place.md](families/same_place.md)), where a person was when last seen
([families/person_where.md](families/person_where.md)). They are asked of a thing's last recognisable sighting, as every
gold is; nothing unobserved changes one. `object_where` is the static control beside them: it asks "Where did you last
see the X?" of a thing that did not move by a metre in the window, which is the other half of `object_was`'s things
([families/object_where.md](families/object_where.md)). **`object_was` asks each moved thing at one of its two states**,
the moved things of a case taking them in turn (earlier, later, earlier, ...) in the order of a hash of the case and the
thing's description, so about half the questions have the place before the move for their answer and half the place
after it, in the same words (the anchored form "Where was the X when you ..."); and, as `object_where` does, it asks no
thing seen in the window's last look window and none the robot's log tells about. No notebook that keeps one place for each
thing can answer both halves: that is what `time_needed` tests (below).

**The starting map, the supply rule and what is asked.** The jobs' `ends` began, where M0's supply measures, taken over
the 0.4.1 release before any 0.5.0 output was read, found questions to draw, as `object_was` for `vacuum/2`, `amr/1` and
`drone/1`; `same_place` for `helper/1`, `amr/1` and `drone/1`; `person_where` for `helper/1` and `vacuum/2`. The rule that
decided the rest was declared before the compile was read: **a new family stays in a job's `ends` only when the
whole-root compile asks at least 0.5 of its questions a case in some shipped length, else it leaves the job before any
gate is read**, and the questions are compiled again. It took `same_place` out of the drone and the helper. The dev read
took `person_where` out of the helper and the vacuum (it failed `time_needed` in all three reads: the resident mostly
stays where he was last seen), and the code review took `same_place` out of the AMR, its last job (a notebook that
answers yes to every question scores `2y - 1` on the yes share `y`, and the compile's 9 yes to 5 no was a certain
failure of `time_needed`). So **no job asks `same_place` or `person_where` in 0.5.0**; their modules stay registered,
with their tests. `object_was` was repaired once (it asks a
moved thing at its earlier or its later state) and read again: it passed every gate on the AMR and failed `time_needed`
on the drone and the vacuum, so it left both `ends` and **`amr/1` alone asks it**. The binding gate check reads what stays, as it reads every family (below).

**Change over time is claimed by cell.** A job-and-length cell is said to test change over time only when its temporal
families carry at least 20% of its answerable end questions and at least one a case (`stbench check` prints each cell's
share, `suite_report.temporal_shares`); elsewhere the report says the cell's questions are about where things were
within the window. No cell is claimed before the compile has been read.

**Perception apart.** In 0.5.0-control the `now` column of a result was `zone_clear`'s, asked in each job while the zone
was undecided; since 0.6.0 it is `ego_now`'s (a declared `now` share per job no longer exists: the cells weigh
alike inside each column once they hold 20 questions, so no count above that sets a weight). The headline weighs the `now` and `memory` columns equally, whatever their counts (methodology rule 4), so a job whose
questions are mostly `now` is still half memory in its headline.
Perception is also scored apart from memory by running the reference agent on the `detections` channel
([codex.md](codex.md), "The detections arm"), which measures **what the pixels cost**: detecting a thing, naming it in
the question's words, locating it and deciding to keep it. The difference per family from the same agent on `rgb_pose` is
that cost. It is not a pure perception loss, and what is left is not memory alone: each detection's text carries its map
centre and the question's vocabulary, so memory over it is a lookup.

## What is not asked, and why

- **In 0.10.0:** [question-design.md](question-design.md), "What is deliberately not asked", has each item with its reason:
  questions at the start of a task, the task's own target, a blocked bay's blocker, change sets, a free spot to put a
  thing down, answers with uncertainty, a harm weight in the headline, the drone's `closest_approach`, the decision
  frames with no supply (the vacuum's `before_backout`, the AMR's `before_undock`), and the tangle kinds that recur at one
  spot (bath mats, pedestal mats, clothes piles).
- **In 0.8.0** (the rule above): the AMR's corridor ahead, every ground robot's back
  corridor (`rear_clear`'s history stands), `person_now`, people questions for the drone and the vacuum, and
  `object_where` for the AMR and the drone. `path_ahead` and `rear_clear` are superseded by `clear_path` as a look
  family.
- **The zone families at the end of a window.** `clearance_then` and `uncovered` are not asked: after the response the
  robot's own path and subgoal tell the zone's state (it aborted, skipped or placed elsewhere), so `path_log` answers
  them, and that is a question about the robot's own path, not about the world. `rear_clear` is not mapped (the AMR's
  undock reverses a fixed 1.5 m that is clear by construction) and `slots_empty` is asked by no job. The state of a
  place the robot is about to use was asked while it was still undecided, `zone_clear` in-run, in 0.5.0-control; 0.6.0
  retired it with `out_of_view` and `adjacent`, so no job asks about the state of a zone.
- **`where_was_i`.** Dropped for every job: it is a lookup of the frame header (the pose and the clock are printed in
  every frame), which both reference agents answered perfectly, so it measured neither perception nor memory
  ([families/where_was_i.md](families/where_was_i.md)). A perceptually anchored replacement, "where were you when you
  first saw X", would need its own check first, and its first-glimpse ambiguity is unmeasured, so none is built.
- **`when_seen`, `order`, `changed`.** Retired: no job's `ends` names them (history below). The change families ask
  about change over time in a form the gate `time_needed` can test, and no job asks them either (above).
- **The helper's carried thing, and the blockers.** The thing a helper carries is a real entity in 0.5.0-control (the
  third of its task's `targets`), and a thing that stood in a decision's zone is the blocker the robot's response tells
  about; the subgoal and the response answer for both, so `World.told` keeps every end question from naming them
  ([families/object_where.md](families/object_where.md), "What the log tells").
- **A "no such thing" answer to a decoy.** None: a description that matches nothing in the world is answered `unknown`
  ([families/object_where.md](families/object_where.md), "Absent descriptions").
- **A context budget or a compaction arm.** None: an agent answering from what is still in its context is using its
  memory, and recall over a growing context is part of what is measured.
  Long windows are how a user tests long memory, and results are reported per length.

### How the families were decided in 0.4.x (history)

Every family a 0.4.x job asked passed its job's gate check (`stbench check`, [methodology.md](methodology.md) rule 5). The
vacuum's passed M2's check on its development render (seeds 21-30: 10 scripts, 30 cases). `helper/1`, `amr/1` and
`drone/1` have no development render, so their families were checked on the release data itself: the release gate check on the first six scripts of each (seeds 2001-2006, 18 cases a job, of three
lengths), and for the drone again over all 12 of its scripts (seeds 2001-2012, 36 cases), where
`object_where` and `distance_between` pass. A family that holds there was not chosen out of sample. A family that fails
a gate check is dropped from its job, not re-tuned (one repair round at most, and only for a broken instrument), and the
questions are compiled again. `when_seen` is the one family whose instrument was repaired once, after M3's review found
its `time` answer reachable from the clock alone; it held the repaired gates in `vacuum/2` and `amr/1` and was dropped
from `helper/1` and `drone/1` by them. The 0.4.1
release is 12 scripts a job for the helper, the AMR and the drone and 10 for the vacuum: 46 scripts and 134 cases,
short, medium and long. 0.4.0's binding
gate check, one whole-root check over its 84 short and medium cases, dropped `when_seen` from `amr/1` and `vacuum/2` too,
so no job in 0.4.0 or 0.4.1 asks `when_seen`. 0.4.1's
binding gate check, one whole-root check over all 134 cases, passed every family the jobs ask with no BROKEN line. A
release's `check_report.json` is the record of what each job asks and what held, and `stbench export` refuses a release
whose check fails; 0.5.0's check is read the same way.

0.4.x asked `where_was_i` during the job in `vacuum/2` (at `before_doorway`) and `amr/1` (at `before_station`,
`before_undock` and `before_crossing`), and `out_of_view` in `amr/1` at the same three points. What was not asked then,
and why:

- **`vacuum/2`:** `changed` was dropped at M2's gate check (it failed `memory_needed`, `shortcuts` and `separation`: 77%
  of its golds were the empty set, which the current view and the question's form guessed). `out_of_view` and
  `uncovered` were dropped at M2's fast gate check (their golds were empty after their declared fallbacks), and `order`
  and `distance_between` are out of its `ends` (under one question a case). `path_ahead` and `rear_clear` are not asked,
  because the back-out gate failed and no plan carries a back-out. Its `when_seen` held through the time-shortcut repair
  (its shortcuts sat 0.03-0.04 under the floor) and was dropped by the binding gate check over the
  shipped cases: on its 13 questions it fails `memory_needed` (`current_view_prior_in_pool` +0.269 over the 0.05 bound)
  and `shortcuts` (`blind_in_pool` +0.269 over 0.10), and `blind` and `path_log` are BROKEN (-0.115 under abstaining's
  0.077). So the vacuum asks `where_was_i` during the job and ends with `object_where`.
- **`helper/1`:** M3's fast gate check on look-dev seeds 141-145 took `changed` (no question in 10 cases) and `order`
  (0.1 a case) out of its `ends` and dropped `distance_between` (separation +0.33). M3's review put all three back, to
  be read on the release windows: the `distance_between` failure was the unknown-share fault (8 of 12 questions unknown
  capped separation at 0.33), since fixed, and look-dev windows of 1-3 tasks cannot ask `changed` (it needs three), so a
  drop on those was a drop on a broken instrument. The release gate check (seeds 2001-2006, 18 cases) then decided each
  family. During: `where_was_i` failed shortcuts (`other_case` +0.24: the helper's routes repeat) and `out_of_view`'s
  learned shortcuts were BROKEN after its one repair (the doors), so both were dropped. At the end: `changed` fails
  shortcuts (`blind` +0.33, `current_view` +0.67, n 6) and `order` asks 2 questions in 18 cases (under one a case), so
  both leave `ends`; `when_seen` (36 questions, separation +0.81 before the repair) was dropped after the time-shortcut
  repair, because it fails shortcuts (`blind_in_pool` +0.15); `distance_between` passes (18 questions,
  separation +0.83) and `object_where` passes (79 questions, +0.75). So the helper asks nothing during the job and ends
  with `object_where` and `distance_between`. `zone_clear`, `path_ahead` and `rear_clear` are not asked of the helper
  (it never reverses, its doorways and reach poses face fixed furniture the public map gives, and a zone on a surface
  needs a band rule no measurement needs yet).
- **`drone/1`:** the release gate check (seeds 2001-2006, 18 cases) dropped `zone_clear` (49 questions) and
  `clearance_then` (24): `other_case`, another case's stream, scored +0.27 and +0.21 over `all_unknown` (bound 0.10).
  The drone's golds are balanced by a fair coin (`zone_clear`'s were 26 `yes` and 23 `no`), so those leads could be
  chance at those counts. Both families were mapped back and read once more, by a rule declared before the data: ship
  them only if the final check over all 12 drone scripts (seeds 2001-2012, the six they failed on included) passes them
  on the point verdict. It did not. `zone_clear` (96 questions) fails `shortcuts` (`blind_in_pool` +0.104 against 0.10,
  with `blind` BROKEN at -0.135), and `clearance_then` (49) fails `memory_needed` (`current_view_prior_in_pool` +0.143)
  and `shortcuts` (`form_in_pool` +0.143, with `blind` BROKEN at -0.122). Both are dropped for good.
  `order` asked 5 questions (0.28 a case) and left `ends`. `when_seen` (28
  questions, separation +0.93 before the repair) was dropped after the time-shortcut repair: it fails `memory_needed`
  and `shortcuts` (+0.30, a memoryless learner of the lag behind the window's end answers it). So the drone asks nothing
  during the job and ends with `object_where` and `distance_between`, which pass the final check over its 12 scripts:
  `object_where` 168 questions, separation +0.81 [+0.75, +0.87]; `distance_between` 77, +0.84 [+0.77, +0.94]. Nothing is
  asked at `before_return` (it lies 8 to 13 steps after `before_lower`, so a family there would take half of
  `before_lower`'s draws), `where_was_i` is not asked, and `changed` and `out_of_view` are never asked: both read the
  robot's band of free space and the floor at its base, which is in the air.
- **`amr/1`:** no `path_ahead` (the station's face is 0.12 m ahead of every undock) and no `slots_empty` (a constant
  list of the lanes most often clear beats `all_unknown` by more than a shortcut may). Its release gate check (seeds
  2001-2006, 18 cases): `object_where` (88 questions, separation +0.78), `when_seen` (28, +0.64),
  `distance_between` (26, +0.85), `where_was_i` (32, +1.00) and `out_of_view` (31, +1.00) passed. `zone_clear` (12
  questions) failed shortcuts and separation (`blind` +0.67: after its scope fix its `memory` rows are clear bays), and
  `rear_clear` (12) failed `memory_needed` (a learned shortcut under the pool's own winner scored +0.083, one question's
  margin); both were dropped. `changed` (2 questions in 18 cases; its oracle 0.60, an open instrument question for a
  later release) and `order` (4; `other_case` +0.25) left `ends`. The AMR's two drops of `zone_clear` and `rear_clear`,
  with the drone's of `zone_clear`, leave the first draft with no `now` question (above). 0.4.0's binding gate check
  over the shipped cases then dropped `when_seen` (39 questions): it passes its gates there, but `questionnaire_blind`
  is BROKEN (0.154 against abstaining's 0.256), a line the export refuses; with the long cases in (64 questions) it was
  not, and 0.4.1 changes no definition, so it stays dropped. So the AMR asks `where_was_i` and `out_of_view` during the
  job and ends with `object_where` and `distance_between`.
- **`adjacent`** is named in the `ends` of `vacuum/2` and `amr/1` and is asked on held-out layouts only. The first draft
  has no held-out layout, so no job asks it ([families/adjacent.md](families/adjacent.md)).

## The families

Each family has a page of its own (what it asks, its text and slots, its gold, its unknown rule and don't-cares); a
family lane writes its page with its module. This is every family in the registry (`questions_text.FAMILIES`), with the
jobs of 0.11.0 that ask it. A family no job asks keeps its module, tests and page; the "Asked by" cells say which jobs
asked it in 0.5.0-control or before, as history.

| Family | Asked | Answer, credit | Scope | Asked by | Page |
| --- | --- | --- | --- | --- | --- |
| `ego_now` | during, at a look point: a thing in the latest frame, by the point of it the camera showed nearest the floor point below the camera; a floor robot (the AMR) is asked only about things on the floor | `offset`, `graded` on the job's clearance C | `now` | `amr/1`, `helper/1` (all four through 0.7.0) | [families/ego_now.md](families/ego_now.md) |
| `ego_mem` | during, at a look point: a thing seen before the recent view (the last 8 frames of the task up to the point) and not in it, measured from the camera now; a floor robot (the vacuum, the AMR) is asked only about things on the floor; a fifth of the questions name a thing never seen | `offset`, `graded` on the job's clearance C | `memory` | all four jobs | [families/ego_mem.md](families/ego_mem.md) |
| `task_things` | end, once per task window: the things near the task, by the job's predicate | `positions`, `graded_set` at the job's `match_m` (grown by the largest item's widening; don't-care circles): per item, a false item half a missed one | `memory` | `helper/1`, `amr/1`, `vacuum/2` (since 0.11.0; the vacuum dropped it in 0.9.0, above; the drone asks none) | [families/task_things.md](families/task_things.md) |
| `clear_path` | during, at a look point: how far the body could move before it touched anything, in the direction drawn from the job's `clear_directions` (ground: ahead, back, or turned in place left or right; a flier: ahead or back); a structure family | `number`, `graded` on the job's clearance C | by evidence | `drone/1`, `helper/1`, `vacuum/2` | [families/clear_path.md](families/clear_path.md) |
| `open_heading` | during, at a look point: which heading, turned in place, would let the body go farthest before it touched anything it has seen; no subject; a structure family; a clearance question (0.11.0; the diagnostic ability `heading` through 0.10.0) | `heading`, `regret` on the job's clearance C, room to move | by evidence | `helper/1`, `vacuum/2` | [families/open_heading.md](families/open_heading.md) |
| `way_to` | during, at a look point in the helper's driving phases: which way, of those seen, to set off to reach a thing seen before the recent view, by the shortest way the body fits through; the help kind, in the memory ability (0.11.0) | `heading`, `route`: the progress one R of driving makes, on C widened to 2 `rho` = 0.2 m by the field's cell | `memory` | `helper/1` | [families/way_to.md](families/way_to.md) |
| `closest_person` | end, at most one a case, about one task: the least distance a person came, as seen | `number`, `graded` on the job's clearance C | `memory` | `amr/1` | [families/closest_person.md](families/closest_person.md) |
| `person_mem` | during, at a look point: the nearest point of the person last seen, from the camera now; only where the world holds one person | `offset`, `graded` on the job's clearance C | `memory` | none (`helper/1` through the first 0.10.0 draft, until it failed `separation`) | [families/person_mem.md](families/person_mem.md) |
| `closest_approach` | during, at a look point: how close the person nearest the robot would come to it in the next 5 s if both kept the velocity they have | `number`, `graded` on the job's clearance C | by evidence | `amr/1`, `helper/1` | [families/closest_approach.md](families/closest_approach.md) |
| `approach_mem` | during, at a look point: how close the person last seen, out of view now and standing where it was seen, would come to the robot in the next 5 s if the robot kept the velocity it has (0.11.0; the twin of `closest_approach`) | `number`, `graded` on the job's clearance C | `memory` | `amr/1`, `helper/1` | [families/approach_mem.md](families/approach_mem.md) |
| `drop_zone_people` | during, at a look point in `inspect`, `descend` and `lower`: the people within 4 m of the point below the drone, when that disc is wholly in the frame | `positions`, `graded_set` at the job's `match_m`; a person with pixels and no evidence sighting is a don't-care circle | `now` | no job (the drone, until the final review's S2) | [families/drop_zone_people.md](families/drop_zone_people.md) |
| `tangle_ahead` | during, at a look point in `cleaning` and `transit`: the things on the floor within 1 m, ahead or to the sides, that the brush could catch (cords and cloth; the kinds that recur at one spot are don't-care) | `positions`, `graded_set` at the job's `match_m` grown by the largest item's widening | by evidence | `vacuum/2` | [families/tangle_ahead.md](families/tangle_ahead.md) |
| `path_ahead` | during, at a decision point of a kind that asks it | `number`, the distance ladder | `now` | none (`clear_path` supersedes it as a look family) | [families/path_ahead.md](families/path_ahead.md) |
| `rear_clear` | during, at a decision point of a kind that asks it | `number`, the distance ladder | by evidence | none (`amr/1` dropped it; `clear_path` supersedes it) | [families/rear_clear.md](families/rear_clear.md) |
| `zone_clear` | during, at a decision point of a kind that has a zone rule | `yesno`, exact | `now` (the decision's look window alone) | none (retired in 0.6.0; all four jobs asked it in 0.5.0-control, at one decision kind each; 0.4.x: `amr/1` and `drone/1` asked it by evidence and dropped it, the drone's for good, after a re-read over its 12 scripts) | [families/zone_clear.md](families/zone_clear.md) |
| `where_was_i` | during, at a decision point of a kind that asks it; names a delivered clock at least 60 s back | `position`, the distance ladder | `memory` (own path: `path_log` is its reference) | none (dropped: a frame-header lookup; `vacuum/2` and `amr/1` asked it in 0.4.x) | [families/where_was_i.md](families/where_was_i.md) |
| `out_of_view` | during, at a decision point of a kind that asks it | `positions`, F1 at `MATCH_M` (0.5 m) plus the largest item's widening (bands, twins, things in view, things the log tells about, movers and places seen empty since are don't-care, each with its own widened radius) | `memory` | none (retired in 0.6.0; `amr/1` asked it at `before_undock` and `before_crossing` in 0.5.0-control, `helper/1` dropped it) | [families/out_of_view.md](families/out_of_view.md) |
| `uncovered` | during, when a room's cleaning ends or stops | `cells`, F1 (cells only perception can judge are don't-care) | `memory` (`path_log`'s swath-only answer is a shortcut) | none | [families/uncovered.md](families/uncovered.md) |
| `object_where` | end, once per answerable thing: "Where did you last see the X?", of a critical thing that did not move | `position`, `graded` on the job's re-find radius R, to the middle of the thing (widened for a thing whose top the camera never saw); no harmful side | `memory` | `helper/1`, `vacuum/2` (all four through 0.7.0) | [families/object_where.md](families/object_where.md) |
| `object_was` | end, once per moment of the job's events in each task, of a thing that moved, at its earlier or its later state in turn; the anchored words ("Where was the X when you ...") | `position`, the relative ladder (`rules.relative_ladder`, widened) | `memory`, temporal | none (`amr/1` alone asked it until 0.6.0; the repaired form failed `time_needed` on the drone and the vacuum, which left their `ends`) | [families/object_was.md](families/object_was.md) |
| `same_place` | end, once per pair of event moments in two tasks and kind; a 50% coin between yes and no | `yesno`, exact | `memory`, temporal | none (the supply rule took it from the drone and the helper, the code review from the AMR) | [families/same_place.md](families/same_place.md) |
| `person_where` | end, once per moment of the job's events in each task, at most one a case; only a person unique in the world | `position`, the distance ladder (widened by the person's box) | `memory`, temporal | none (the dev read took it from the helper and the vacuum: it failed `time_needed`) | [families/person_where.md](families/person_where.md) |
| `changed` | end, once per room toured in two tasks at least two apart | `positions`, F1 at 0.5 m (movers, twins and glimpses don't-care) | `memory` | none (`vacuum/2`, `helper/1` and `amr/1` dropped it; `drone/1` never asks it) | [families/changed.md](families/changed.md) |
| `when_seen` | end, once per kind | `time`, a ladder of 15, 30, 60 and 120 s | `memory` | none (`helper/1` and `drone/1` dropped it after the repair of its time shortcut, `amr/1` and `vacuum/2` at the binding gate check) | [families/when_seen.md](families/when_seen.md) |
| `order` | end, once per kind, two things of that kind | `yesno` | `memory` | none (all four jobs took it out of `ends`) | [families/order.md](families/order.md) |
| `distance_between` | end, once per pair of kinds | `number`, the job's distance ladder widened by both things (0.4.x: relative credit) | `memory` | none (`helper/1`, `amr/1` and `drone/1` asked it in 0.5.0; `vacuum/2` took it out before) | [families/distance_between.md](families/distance_between.md) |
| `clearance_then` | end, once per moment that is an event and a zone | `yesno`, exact | `memory` | none (`drone/1` dropped it for good, after the same re-read; the control revision asks the zone in-run instead) | [families/clearance_then.md](families/clearance_then.md) |
| `slots_empty` | end, once per rule of the job's `slot_zones` | `positions`, F1 | `memory` | none | [families/slots_empty.md](families/slots_empty.md) |
| `adjacent` | end, once per pair of unique room kinds joined only through one room, **on a held-out layout only** | `position`, `region` credit (inside the connecting room) | `memory` (`other_case` is its reference, left out of its gates on a held-out layout) | none (held-out layouts only; the release has none) | [families/adjacent.md](families/adjacent.md) |

`text` is the family's render of `slots` and nothing else (`FAMILIES[template].render`): every wording choice is a slot,
and no gold-side value (the decision's keyframe, an event's resolved step, the task's room) is one, so the structured
view tells a baseline nothing the words do not. A thing is named by its kind and describable attributes
(`descriptions.phrase`: "red bag", "pair of black shoes"; a kind named by a plural noun is worded as a unit, so a
question never says "where is the dumbbells": "pair of grey dumbbells", "stack of white folded clothes", since
0.6.0). A window of several tasks names the task an event belongs to
by its case index, `slots["task"]`, written in the text as an ordinal ("in your second task",
`questions_words.task_phrase`): the task is the case's own number, never the script's. A family that names several tasks
carries them as `slots["tasks"]`, a list of case indices; `slots["task"]` None means the window's only task.
`object_where` has `kind` and `attributes` and no moment: "Where did you last see the red bag?". `object_was` keeps the
anchored words and the slots `kind`, `attributes`, `event` and `task` ("Where was the red bag when you began sweeping in
your second task?"): the form tells a moved thing from an unmoved one, not the earlier state it asks of a moved thing from
the later. `same_place` has `kind`, `attributes`, `events` (the two wordings, earlier task first) and `tasks` (the two
case indices). `person_where` has `object_was`'s slots, and a person is named by its kind alone ("the person").

**A list question says what it counts (0.10.0).** The `positions` answer shape in the prompt (`answers.PROMPT_SHAPES`)
gives the form alone: one item for each place the question asks for, each a map position in metres, and an empty list
when there are none. Until 0.10.0 it also said what a "thing" is, which served `task_things` alone; with two more list
families it could not. Each list family now states what it counts in its own text: `task_things` ends with the sentence of
`questions_task.COUNTED` (an object that is not part of the room or its fixed furniture and fittings, so a floor lamp, a
fan or a heater is a thing; a pair or a stack is one thing; people and chairs are not counted either way) and now says why
the things matter ("which you would have to steer around next time"); `tangle_ahead` asks for anything lying on the
floor that "your brush could catch (a cable bundle, an extension cord, a power strip or a towel; a bath mat, a clothes
pile or a pedestal mat counts neither way)", and the vacuum's `task_things` says the same words
(`hazard_rule.catch_words`, built from `CATCH_KINDS` and `SPOT_KINDS`, so a reader is never charged for a thing the words
leave open); `drop_zone_people` asks "the people within 4 m of the point below you". The reference agent's prompt carries
the list bullets of the credit section for a job that asks any list (`brief.asks_lists`).

## What a case asks

A case is a window of a script's tasks (`case.py`), delivered from step 0. Its questions are in `private/<cid>/
questions.jsonl`, in ask order: the **during** questions (look points, and a decision point where a job names a family
for its kind, which no job does in 0.7.0) in keyframe order, then the **end** questions. Each during question is sent
as an ask of its own ([agent.md](agent.md)), so a look point with several (0.10.0: up to three) is that many asks at one
step, each sent when the one before it is answered; the reference agent gets the first with its frame
([codex.md](codex.md)). A system is sent a question as `{qid, text, answer_type}` (`stream_protocol.ask_view`), a during
question after the frame at its decision point and the end questions after the last frame; a scripted baseline
may be sent `template` and `slots` as well (`structured_view`). Nothing else of the row reaches a system.

## The family modules

A family is a module `questions_<x>.py` of the truth side, found through `questions_text.FAMILY_MODULES` (a name with no
module is skipped, as a job not yet written is in `job_registry.JOB_MODULES`). A module defines:

- `SPECS`: its families' `records.FamilySpec`s, each passing `check_family`, with their `render`; `temporal` marks a
  family that asks about change over time (the `time_needed` gate reads those alone);
- `GOLDS`: family -> gold function. A during family's is `(w: World, k: int) -> Cand | None` (None: a dead band at that
  decision point, which is then not asked and no other family drawn for it); an end family's is `(w: World) ->
  list[Moment]`;
- `SLICED`: the end families that draw an unknown slice per moment (default none);
- `COINS`: the end families whose moments hold a yes pool in `answerable` and a no pool in `unknown`, drawn by a 50%
  coin (default none). A family is in at most one of the two sets;
- `LOOKS`, `CAPPED` and `RELAX` (0.6.0, optional): the during families asked at look points, whose gold is `(w, k) ->
  Moment | None` and may be in `SLICED` (`ego_mem`); the end families whose empty golds the draw may thin
  (`task_things`); and the end families whose candidate a look or end question makes don't-care (`task_things`).
  `clear_path` and `person_mem` are `LOOKS` and in no `SLICED` (their `unknown` share is the world's, thinned by
  `JobSpec.look_unknown_keep`).

`questions_text` builds `FAMILIES`, `GOLDS`, `SLICED`, `COINS`, `LOOKS`, `CAPPED` and `RELAX` once, in `FAMILY_MODULES`
order. A family named twice,
a gold with no spec, a spec with no gold, a `SLICED` or `COINS` name that is no end family of its module, or a family in
both sets raises, naming it. A module imports `questions_world`, `questions_select`, `questions_words` (the shared
wording and sizes: `DISTANCE_LADDER`, `ORDINALS`, `task_phrase`, `MAX_CAP_M`, `cap_of`), `records` and neutral modules,
never `questions_text` or `questionnaire`. A `Moment` is what an end family names (a time, a task and a time, a kind, a
pair of moments and a kind): `key`, the candidates `answerable`, and the candidates `unknown` (empty for a family that
draws no slice). `questions_change` holds the three change families; their table answers are `table_change` and
`path_log`'s are `baselines_path_change`. 0.8.0 adds `questions_clear` (`clear_path`: the spec, the words and the
dispatch to `clear_ground.moment` or `clear_fly.moment`) and `questions_people` (`closest_person`, `person_mem`), with
the answerers `table_clear`, `table_people` and `baselines_path_clear`; each registry skips a module that does not
exist. 0.9.0 adds `questions_heading` (`open_heading`) with `clear_profile` (the profile of runs by heading), the
answerers `table_heading` and `baselines_path_heading`, and the map reader's answers by family in
`baselines_map_clear` and `baselines_map_heading` (`baselines_map.MapReader` dispatches through them). The credit's
arithmetic is `credit_kernel`.

## How a gold is computed

Every gold is code over the timeline, the visibility log and the depth the sensors recorded, names its evidence, and no
person or model writes one (methodology, truth rules 1 and 2). The compile reads only what a system is sent, and a window's
own sightings:

- **The window.** Sightings come from the visibility rows of the case's window alone, so a thing seen only before the
  window is evidence of nothing.
- **Recognisable frames.** A sighting is evidence only when its own frame is recognisable (`recognise.recognisable`;
  methodology rule 10, "recognisable in a single frame"): at least `RECOG_PX` (1,024) pixels, a mean luminance of at least
  `RECOG_LUM` (0.01, from 0.5.0-control: where the pixels carry no colour; it was 0.0058), a bounding box at least
  `RECOG_EDGE_PX` (2) pixels from every edge of the image, and a depth. There is no
  total over two frames. A person is the one exception (0.8.0): a sighting of a person is evidence by
  `person_seen.evidence`, whose box may touch the top and the sides of the frame but not the bottom edge, since the
  feet place them ([families/closest_person.md](families/closest_person.md)). **A gold cites the last recognisable frame at or before its moment** (`World.recognised`); a
  clipped or depth-less frame is a sighting (`World.sighted`) and can unsettle a place, but is never the evidence.
- **Places that moved.** `World.stale(e, k, xy)`: some recognisable sighting of the thing after `k` lies at least
  `rules.STALE_M` (1.0 m) plus the kind's `rules.widening` from the place `xy`. `World.arrived`: the mirror, a
  recognisable sighting before `k` that far away. `World.steady(e, k, xy, until)`: every sighting of the thing,
  recognisable or not, from `k` to `until` lies within `rules.MOVED_NO_M` (0.15 m) of `xy`. `object_where` asks neither
  an arrived place nor one that a later sighting, recognisable or not, puts 0.15 m or more elsewhere (not `steady`),
  which covers a stale place; `object_was` asks a **moved** thing (one with a stale place and an arrived one, among
  the places it has at the event moments of the window) at one of its two states, a stale place for the earlier and an
  arrived one for the later, and no thing in the window's last look window; `settled`
  ([families/distance_between.md](families/distance_between.md)), `object_where`, `person_where` and `out_of_view` use
  `steady`.
- **Delivered values.** Every pose a gold uses is the delivered pose row's (`sensors/pose.jsonl`, `yaw_deg` rounded to 3
  places), never the timeline's own floats, so a system's mapper reads exactly what the gold read.
- **The recent view and the decision windows.** The recent view is the last `LOOK_WINDOW` = 8 frames of the current task up
  to a look point (`look_windows.recent`, `World.recent`): `ego_mem` and `person_mem` ask of a thing not seen in it, and
  the harm families name it as their subject. The case's last 8 frames are `World.end_window`, the view the end questions
  are asked from (`task_things`, `object_where`). The decision windows (`look_windows.windows`, `World.windows`: at most 8
  steps, cut after every decision point of the window, asked or not, and after every task's last step) were the stream's
  batches until 0.10.0; a system never sees one now. Only the families no job asks (`path_ahead`, `where_was_i`,
  `out_of_view`, `zone_clear`, `clearance_then`) read them (`World.window_ks`, `World.window_of`), with `stbench view`.
- **Events.** Each wording of the job's event table (`JobSpec.events`) is resolved by `events.resolve` over the
  delivered steps of each task of the window, as a system resolves it: `World.events` maps (case task index, wording) to
  the script keyframe, and a phase a task never has leaves no entry. No compiler module names a phase or a wording. The
  anchored families (`object_was`, `same_place`, `person_where`, `clearance_then`) name a moment; `object_where` no
  longer does.
- **The map and the free-space states.** `World.map` is the layout's public map (what a family reads of a room's polygon
  or a fixed piece, as the table systems do); the free-space cells of a keyframe are computed once per compile,
  whichever families read them.
- **Scope.** A family's scope is its spec's, `now` or `memory`. **`now` is the latest frame**, which is what a system
  holds when it is asked: it gets one frame at a time, so a `now` gold has to be one that frame alone gives. A family
  whose spec says `evidence` gets `now` exactly when its evidence is non-empty and every evidence keyframe is the
  decision's own frame `k` (`questions_world.scope_of`), else `memory`, so no `now` tag claims more than the latest frame
  holds. A flier's `clear_path` keeps that rule. A ground `clear_path` and `open_heading` read the whole window, so their
  scope is by equality: `now` when the gold is a number and the corridor over frame `k`'s own cells
  (`clear_ground.frame_agrees`, `World.latest_cells([k])`) gives the same number within the row's resolution (for
  `open_heading`, when frame `k`'s own profile has a best heading whose regret on the window's profile is within the
  resolution), else `memory`. An answerable `memory` row is then exactly one the latest frame cannot answer, and an
  `unknown` gold is `memory`, as `ego_mem`'s never-seen slice is. Until 0.10.0 these rules read the look window's
  cells, so a corridor that needed an earlier frame of the window was still `now`. On 0.9.0's data the change moves 44 of
  668 rows from `now` to `memory` and none the other way: `clear_path` 78 to 43 `now` (35 rows), `open_heading` 15 to 6 (9
  rows), `ego_now` unchanged (65). No question or gold value changes, only the `scope` tag. It did change which cells
  reached the 20-question line (`abilities.MIN_ABILITY_N`, counted over the whole dataset, [eval.md](eval.md); since
  0.11.0 a cell under it weighs `n / 20` and is not left out): the
  helper's clearance `now` cell falls from 26 golds to 14 and the vacuum's from 20 to 6, so both are thin; the drone's
  clearance `memory` cell rises from 13 to 22 and is counted; the helper's clearance `memory` cell rises from 37 to 49 and
  the vacuum's from 4 to 18 (still thin). The headline's `now` column is then distance plus the drone's clearance cell.
  The rule was not loosened for this. The question redesign of 0.10.0 raised the density instead: on the dev
  root the only thin cell left was the helper's heading `now` (2 questions, left out of the headline), and the clearance
  cells held 26 (`now`) and 36 (`memory`) on the AMR, 98 and 100 on the drone, 50 and 94 on the helper, 102 and 346 on the
  vacuum. Lane J2 then tagged a `closest_approach` row with a person `memory`, which moved the AMR's clearance cells
  to 6 (`now`, thin: the "no one" rows alone, as the AMR asks no `clear_path`) and 52 (`memory`) and the helper's to 35 and
  109. The final review's repair delivers the robot's velocity with every step, so the rows whose person stands
  are `now` again, and drops `drop_zone_people` from the drone: the dev root then holds, in `now` and `memory`, 22 and 36
  on the AMR, 96 and 100 on the drone, 51 and 67 on the helper and 102 and 346 on the vacuum, and the only thin cell is the
  helper's heading `now` (2 questions, left out of the headline in 0.10.0)
  ([families/closest_approach.md](families/closest_approach.md), "Scope").
- **Evidence age (0.10.0).** Each gold row carries `age_s` and `age_band` (`questions_age.age`), null on an `unknown` row.
  `age_s` is the clock time from the gold's oldest evidence keyframe (`min(evidence_k)`) to the ask, read from the
  timeline's keyframe times; the ask is a during question's `decision_k` and, for an end question, the window's last
  frame. `age_band` is `now` exactly when the row's scope is `now`; else `recent` when `age_s` is at most
  `questions_age.RECENT_S` = 10 s, `task` when the evidence is in the task the ask falls in, and `earlier` when it is in an
  earlier task. The reports split the reported score by band (`by.age_band`, `job_age_band`), a memory curve of credit
  against how long ago the robot saw the answer ([question-design.md](question-design.md), section 1).
- **Region and metric.** A gold carries a `region` (a polygon) when its credit is `region`, else null; its metric is
  `xy` for a `position` answer and null otherwise.
- **Critical.** A row is task-critical (`questions_where.critical`) when its thing's kind is one of the task's relevance
  kinds; its place lies in a region whose kind is among the task's `Relevance.regions`; where the job sets
  `Relevance.near` and `Relevance.near_m` the place lies within `near_m` of one of the points (the drone's: the ground
  its camera sees from the drop point, about 12 m from 15 m up; a job that sets none is unchanged); and, where the job
  sets `Relevance.floor` (`vacuum/2`, `amr/1`: robots that work the floor), the thing stood on the robot's floor
  (`World.on_floor`). An unknown row is critical by kind (no place is known). `object_where` joins the window's tasks'
  relevance (a thing is critical when any task of the window makes it so) and asks **only critical things**; a
  `same_place` row is critical when either of its two places is, a `distance_between` row when both things are, an
  `out_of_view` row always. A person is critical only where the job's relevance kinds name people, which no job's does.
  The tag is for selection and for the report's split; it never changes a gold.
- **What the log tells (`World.told`).** A thing the robot's own log tells about is never asked at the end, since the log
  answers it: every entity named by the third of a window task's `targets` (the helper's carried thing) and every thing
  whose footprint overlaps the zone of a decision of the window whose kind has a zone in `JobSpec.zones`, at that
  decision's time (the blockers: the response after the coin and the subgoal that follows say they stood there; since
  0.6.0 the key is the zone and not a `zone_clear` asked there). `World.things` leaves them out, so no end family names
  one, and `task_things` makes them don't-care places. A look question reads `told(upto=k)`: what the log had told by
  its point.
- **Tolerances.** What a metric answer is credited on is the job's, declared with the job and read from no gold
  ("Credit", above; the numbers by job are in its table). An `offset` (`ego_now`, `ego_mem`, `person_mem`) and the
  `number` of `clear_path`, `closest_person` and `closest_approach` are credited on the job's clearance C; a map place (`object_where`) on
  the re-find radius R (`JobSpec.reacquire_m`), to the middle of the thing, with a resolution grown
  by `rules.hidden_widening` only when the thing's top was above the camera at every recognisable sighting
  (`questions_where.tolerances_of`); a list of places (`task_things`, `drop_zone_people`, `tangle_ahead`) at the job's `match_m` (twice the first rung
  of `JobSpec.ladder`: 0.5 m in a home, 1 m in the warehouse, 2 m in the city), each item's radius grown by
  `rules.widening` whatever the camera saw (`tolerances.match_m`, the largest item's), with one resolution an item in the
  gold's item order. Every row carries its need, its resolution and its harmful side (`rules.graded_tolerances`), and for
  `open_heading` the whole profile of runs; the scorer reads the row, and the reference agent's prompt states the same
  numbers. 0.7.0's ladders (`rules.clearance_ladder`) and 0.6.0's range-relative rungs
  (`rules.relative_ladder`) are history: `relative_ladder` and `JobSpec.ladder` stay for the change families and
  `distance_between`, which no job asks.
- **Absent descriptions (decoys).** An unknown row may name a description that matches nothing in the world
  (`questions_where.never_seen`), but only when the world gives no clue that it is absent: a thing of its kind is
  sighted in the window, and no sighted thing of its own kind or of a kind it can be mistaken for
  (`descriptions.CONFUSABLE`) could be taken for its colour (below). `CONFUSABLE` is the kind groups tote/bin/box/bag;
  van/box_truck/pickup/car; bag/laundry_basket; mug/glass/bowl; bowl/plate; dumpster/skip; chair/stool; toy/box (the
  toy's stacked blocks); toy/clothes_pile; book/book_stack/shelved_book (the last three of the home catalog's clutter,
  never asked but sighted like a tracked kind); air_purifier/space_heater and pedal_bin/bath_bin (home clutter that
  render alike, two floor towers and two pedal-bin cylinders of one colour range: `ego_now` and `ego_mem` name them by
  kind alone, so either one hides the other); and, from the 0.7.0 given-name check, the home's soft goods, which
  rendered alike side by side: folded_clothes/towel/book/book_stack and
  jacket/hanging_coat/clothes_pile/folded_clothes (a stack of folded clothes reads as a towel or a book, a towel as a
  book, and the cloth heaps as one another; a question never names one while another of its colour is sighted); and,
  from the 0.7.0 dev pilot's log review, cable_bundle/extension_cord (a cable bundle was answered at an extension
  cord); and a kind's look-alikes
  are the union of the groups that hold it (a bag's are the basket, the tote, the bin and the box; a box's add the
  toy). The decoys, and the world's own things that
  no frame of the window shows, are also screened by `descriptions.CONFUSABLE_UNSEEN`, the groups bin/drum and
  sack/pallet (a drum of a bin's colour in the bay, a pallet's wrapped load for a sack): a bin, a drum, a sack and a
  wrapped pallet are different shapes at the camera's range, so these groups hide nothing that a frame shows (a seen drum
  or pallet is asked beside a seen bin or sack of its colour: a fair test of telling them apart), and count only where
  nothing seen says which is meant (a "yellow bin" never seen beside a sighted yellow drum, a decoy "blue sack" beside
  a sighted blue pallet; the reverse as well). The world's own unseen things are offered before a decoy, and a case asks
  at most one (the questionnaire, below).
- **Real things and look-alikes.** The same screen applies to the real things `object_where`, `object_was` and
  `distance_between` ask (both things of a pair, settled or unseen), and it reads the colour the render shows, not the
  catalog's word (0.5.0-control; review defect 1: a remote labelled black rendered silver, a "grey" laptop was as light
  as the white ones, and "Where was the white laptop?" was asked with it in plain view).
  - **What a render shows.** The visibility log's `rgb_med` is the median sRGB over a thing's mask in a frame.
    `colour_seen.classify` names it with one of the palette's twelve words (black, grey, white, beige, brown, red,
    orange, yellow, green, blue, purple, pink) by its CIELAB lightness, chroma and hue, or, within 3 CIELAB units of a
    boundary, gives the words on both sides ("no confident class"). The thresholds, and why they are what the renders
    show, are in `colour_seen.py`'s docstring. Only a thing's recognisable frames count.
  - **Its picture bears out its word.** The colour that most of a thing's recognisable frames confidently show it in
    (`colour_seen.held`: the one word with the most frames, alone; a frame near a boundary votes for none) must be its
    label (`World.colour_holds`), or it is not asked. So a thing of several colours seen from several sides (a red
    portable toilet under a white roof, from the drone) is held by the colour most of its frames show. A thing with no
    recognisable frame (an unseen one) has none to contradict its label.
  - **No look-alike hides it.** It is not asked when another sighted thing of a kind it can be mistaken for
    (`questions_where.lookalike_sighted`) could be taken for its colour (`World.may_be_taken_for`): any of that thing's
    recognisable frames shows it in the colour or next to a boundary of it, whatever its label. A blue skip is not
    asked beside a blue dumpster, nor a bag beside a box of its colour, since a system that holds the look-alike
    cannot tell which one the question names. It is asked when no look-alike renders in its colour or none is sighted.
    A thing no frame of the window shows is also hidden by `CONFUSABLE_UNSEEN` (a "yellow bin" never seen is not asked
    beside a sighted yellow drum, nor a "blue sack" beside a sighted blue pallet); a seen drum or pallet is asked beside
    a bin or sack of its colour. The screen reads kinds, colours and whether a sighting exists, never a place, a time or
    a gold.
  - **The neutrals are one group.** Black, grey and white are compared as one colour (`colour_seen.could_be`): a person
    cannot reliably tell white, grey, silver and black apart on a thing of several materials under varied light, and a
    median over a mask mixes a body with its screen, lid or roof. So a sighted thing that may be taken for any neutral
    hides every neutral question and decoy, and a neutral thing is hidden by any such look-alike: a "white" laptop is
    not asked, as `unknown` or with a place, beside a "grey" one that renders grey (the review's case: helper case
    `bc5f`), nor a "black" bag beside a grey-rendering one. A chromatic colour keeps the rule above: it is hidden only by
    a look-alike that renders its own word or next to a boundary of it, so a red and an orange thing that render apart
    are both asked. A thing's own picture still has to show its own word (above).
  - **An old root's log.** A log from before 0.5.0-control has no `rgb_med` (and a thing with no recognisable frame has
    nothing measured): the label stands for the render, and a neutral label stands for all three neutrals
    (`colour_seen.label_reads`), because the review found a neutral label unreliable.

  The brief carries no vocabulary ([agent.md](agent.md), "The brief"): the words a question uses are the question's
  own, and naming the things in a frame is the agent's perception.

The golds themselves are on the family pages.

## The questionnaire (`questionnaire.py`)

- **During.** A seeded draw over the window's decision points, independent of the world and every gold: candidates are
  the points whose kind asks a family and that lie far enough into their task (`questionnaire.min_steps`): at least 16
  steps (`GAP_STEPS`) when the kind asks any family that is not `now` (a question that may reach into the past needs
  history behind it; a mixed kind takes this too, and an `evidence` family counts as not `now`), at least 4
  (`NOW_MIN_STEPS`, half a look window: the robot has moved and the view is its own) when it asks only `now` families
  (the answer is the look window the question follows, which needs no history). Candidates are taken in `stable_hash(case id, k,
  "ask")` order, kept when at least 16 steps from every kept point (the spacing is the same for every kind), to 3
  points (short), 5 (medium) or 8 (long); the family is the kind's `stable_hash(case id, k, "family")`-th. The draw
  reads kinds and family specs, never the world or a gold. A drawn point whose evidence rule fails is asked, with
  gold `unknown`; a point whose family finds a dead band (the gold function returns None) is not asked, and nothing else
  is drawn for it. A job that names no kind asks nothing at a decision point, and since 0.6.0 no job names one: a decision kind's only
  role in a draw is to place look points, the job's `look_decisions` coming first. Look points are a second during draw,
  independent of this one ([Look points](#look-points)).
- **End.** The families the job asks (`ends`), in turn: in each round the **temporal families come first**
  (`FamilySpec.temporal`, so a case the cap cuts short keeps its change-over-time questions), each group in
  `stable_hash(case id, family, "family")` order, one question a turn, round robin, to 12 questions (short), 18 (medium)
  or 20 (long) or until every family is out. On its turn a family tries its next moments, in `stable_hash(case id,
  family, moment key, "moment")` order, until one gives a question: a candidate of the moment's pool, taken in
  **selection order**, that the subject rule allows.
- **Slices and coins.** A moment of a **sliced** family (`SLICED`: `object_where`, `ego_mem`, `way_to`, `distance_between`, `when_seen`) is
  kept only when both its pools are non-empty (a moment of any other family, when its answerable pool is), and it gives
  a question only when the subject rule allows a candidate in both its pools (a pool that earlier questions used up
  skips the moment whatever its slice), so the moment's form never predicts its slice. Its pool is `unknown` for 20% of
  moments, drawn per moment (`stable_hash(case id, family, moment key, "slice") % 100 < 20`), so neither the moment's
  position nor the case moves it, and the asked unknown share is the declared one. A **coin** family (`COINS`:
  `same_place` and `object_was`) has its yes candidates as the first pool and its no candidates as the second, and draws the second for
  `COIN_PERCENT` = 50 of its moments the same way, so the yes/no is a fair coin per moment and whether a moment is asked
  never depends on its answer. A family with no moment left drops out of the round. `object_where`'s moments are its
  answerable things, each keyed by its description and each beside the window's one unknown pool; the other sliced
  families' are the job's event moments or its kinds.
- **One a case.** `closest_person` gives at most one `Moment` a case, about the task its hash puts first among those that
  pass its screens (one a task would break the AMR's medium end budget).
- **Selection order.** The candidates of a pool are offered **critical first**, then **the world's own things before an
  absent description** (a `d:` subject, a decoy), then by `stable_hash(case id, candidate key)`. Selection order decides
  only which of the questions a case could ask it does ask, never the ask order below.
- **The subject rule.** No two end questions of a case share an entity, a description or a named time (equal or
  overlapping `w:` subjects), and **at most one asks about an absent description**: the decoys give nothing away only
  while they are few. A time is a subject when the question's text names it; the sightings a gold reads from are not.
  A decision-point question's subjects are recorded and exempt; a look question is drawn after the end questions and
  names none of their subjects. A person is a subject, so a case asks `person_where` at most once.
- **Order.** The end questions are drawn first, then the looks over their subjects, then `RELAX` and `CAPPED` ([The order
  of the draw](#the-order-of-the-draw)). During rows (decision points and look points) by keyframe, then end rows by
  `stable_hash(case id, candidate key)`; `gold.jsonl` follows.

## The records

The validators are `records.check_question` and `check_gold`; the families are `records.FamilySpec` entries passing
`check_family`. Every compiled row passes them or the compile stops.

- **Question row** (`questions.jsonl`, sorted keys): `qid` (16 random hex, fresh each compile), `when` (`during` or
  `end`), `decision_k` (the script keyframe the ask follows; null on an end row), `text`, `template`, `slots`,
  `answer_type`, `needs` (sensor capabilities, never a scope).
- **Gold row** (`gold.jsonl`): `qid`, `family`, `gold` (answer-shaped; `{"type": "unknown"}` exactly when `slice` is
  `unknown`), `slice` (`answerable` or `unknown`; a `same_place` row is `answerable` whether its answer is yes or no),
  `credit`, `region`, `tolerances` (the credit's keys, written in full on every row, unknown ones included:
  [format.md](format.md)), `metric` (`xy` for a position), `scope` (`now` or `memory`), `reference` (null, `path_log`,
  `other_case`, `kind_range` or `map_reader`), `critical` (task-critical against the job's relevance: kind, region of
  the place, the near rule, floor), `subjects` (`e:<id>`, `d:<kind>/<attributes>` for a description absent from the
  world, `r:<region id>`, `a:<k>` for the sighting of a person `approach_mem` asks about, `w:<k>` for a named time),
  `evidence_k` (sorted script keyframes; never empty on an answerable
  gold) and `entities` (the sighted entity, parallel to `evidence_k`; empty when there is none). A row also carries
  `answer_class` (after `slice`: null, or the class a draw balanced) and `range_m` (the slant range its relative rungs
  were built from; null otherwise). Since 0.10.0 a row ends with `age_s` and `age_band` (`records.GOLD_OPTIONAL`: the
  evidence age, null on an `unknown` row; absent from the rows of an earlier compile, which stay readable), and a harm
  family's `tolerances` may hold a `catastrophe` rule (`records.OPTIONAL_TOLERANCES`; [format.md](format.md)).

Keyframes in the gold file (`decision_k`, `evidence_k`, the `w:` subjects) are script numbering, like every private
record; the stream and the question texts are case-relative.

## Checks that guard it

- *Hidden changes never change a gold* (`questions_swap`, `tests/test_compile.py`): exchanging two things' placements
  and sightings, or adding a thing where no frame looked with the depth unchanged, moves no gold; the same thing where a
  frame saw it moves the `path_ahead` gold. A move the robot did not see (after a thing's last sighting) changes no
  candidate or gold of the change families either.
- *The draw ignores the world*: which points are asked depends on the case id, the decision points' kinds and steps and
  the families' specs alone (`tests/test_questionnaire_minimum.py`: the minimum into the task by scope and the spacing).
- *No truth in a question*: a question's delivered view holds no entity id, region id or gold coordinate (`leak.RULES`).
- *Gates* (`stbench check`): perfect knowledge of what was seen scores at the ceiling; the current view, the path log,
  the other cases (a shortcut that learns from them answers a pool only when its best response beats abstaining on the
  scripts left out of it, its gain's bootstrap interval over scripts above 0), the question wording and all-`unknown` sit at the floor, and an own-path family (reference
  `path_log`), a layout-constant one (reference `other_case`, on a held-out layout) and a structure family (reference
  `map_reader`: `map_reader` and `other_case` are left out of its shortcuts and its separation on every layout, and
  `path_log` stays in, answering from the robot's own log alone) are held against their own reference. A family that fails is fixed before data or dropped, never forced. The gates are per layout, job and
  family, so the same family can hold in one job and be dropped from another. The gates and their thresholds are
  `gates.py`'s: `ceiling` (oracle >= 0.95), `knowable`, `memory_needed` (+0.05), `shortcuts` (+0.10), `separation` (>=
  0.40, lower bound > 0.30) and `time_needed`.
- *The two notebooks and `time_needed`.* `latest_table` and `first_table` are scripted systems in the gate suite
  (`baselines.LatestTable`, `baselines.FirstTable`): the oracle's table keeping only each track's latest (first)
  sighting and reading no event table, a notebook of the last (first) place each thing was seen, whatever moment or task
  a question names. `time_needed` holds the larger of their two scores to the shortcut margin on a **temporal** family:
  `max(latest_table, first_table)` <= `all_unknown` + 0.10, and a failure names the notebook; on those families both are
  also rivals `separation` measures the memory table against (they are not in `shortcuts`, which reads other scripts'
  learned pools). On every other family `time_needed` is `n/a` and the notebooks are not read. A family that asks about
  change over time but that a notebook of one place for each thing answers is not one of change over time, and the gate
  says so by failing it; two notebooks because a family asking one state only is answered by one of them (the first form
  of `object_was`, which asked the earlier state alone, was answered by `first_table`). The check report gives each
  gate's value and each group's temporal share by length.
- *Change families, by construction.* `object_was` asks a moved thing at the earlier or the later state in turn, so a
  notebook of first sightings is right on one half and a notebook of latest sightings on the other, and a wrong place
  that is a metre or more from the gold meets at most the last two rungs (credit at most 0.5, penalised score at most
  0; past the last rung, -1); `same_place` is a fair coin, so an always-yes notebook scores near 0 (and no job asks it
  in 0.5.0); `person_where` asks a place at an earlier moment of a thing that moves between tasks (and no job asks it
  in 0.5.0). The pages of the three families have the reasoning.
