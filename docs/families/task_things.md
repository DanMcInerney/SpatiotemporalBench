# `task_things`

"Where were the things the task was about?" The ability is relevance: which things mattered to a task, defined by what
the task was and never named. End, `memory`. Code: `stbench.questions_task` (the gold), `stbench.task_predicate` (the
declaration and the geometry the gold and the tables both call), `stbench.table_task` and `stbench.baselines_path_task`
(the answerers). New in 0.6.0. The record and questionnaire rules are in [questions.md](../questions.md).
Since 0.8.0 the AMR's end is `task_things` and [`closest_person`](closest_person.md); the helper's, and the vacuum's
since 0.11.0, are `object_where` and `task_things`; the drone asks none. The family's golds are unchanged in 0.9.0; its
credit is per item (`graded_set`). **In 0.9.0 it was asked on the helper and the AMR only: the vacuum's `task_things`
was dropped**,
because it failed the binding evidence verdict there. `path_log_in_pool`, answering with the places other scripts' golds
put items within reach of the case's own logged anchor, scored +0.258 over `all_unknown` on the vacuum's long cases
(+0.135 on the medium ones): the home's things recur at the same spots (90% of the vacuum's distinct gold items had
another script's item within 0.5 m) and "on the floor of the rooms you cleaned" reached every such spot, so the list was
a layout prior that no sensor reads. A family that fails a gate on a job is dropped from that job by rule
([question-design.md](../question-design.md), section 9), and methodology rule 5 asks a fact that is the same in every
case of a place only of places never seen. The helper's route predicate keeps its items off the recurring spots (its
`path_log_in_pool` reads -0.221 on the 0.11.1 release data, the AMR's -0.065; it read -0.115 on 0.11.0's, before the
scorer stopped a don't-care circle from costing a list credit).
**In 0.11.0 the vacuum asks it again, as the list of what its brush could catch**, the "list limited to things off the
layout's recurring spots" this page named as the way back: the predicate keeps `rooms` and
`floor`, names the kinds a brush catches as the only items (`TaskThings.item_kinds`) and makes the kinds that recur at one
spot don't-care (`dont_care_kinds`), the split `tangle_ahead` already makes (below). It is a reuse question: a cord left on
the floor stops the next clean, so the list is what to steer around next time, and it is the whole-room list recalled
whole that the evidence names as the hardest row. The gates decide it again: if
`path_log_in_pool` or the balance rule fails, the vacuum drops it by rule, no threshold moved.

| | |
| --- | --- |
| Category | C5 What matters here |
| Asked | end, one question for each task of the window that has an anchor (a job's `ends` names it, and its `JobSpec.task_things` declares the predicate) |
| Text (slots) | "Where were the things {phrase} in your {first} task, which you would have to steer around next time? Answer in map coordinates." and then the sentence that says what a thing is (`questions_task.COUNTED`). The clause after the task says why the things matter (0.10.0, a render change only: the gold, the credit and the slots are unchanged). (`phrase`: the job's predicate in words, which states the distance the gold uses; `task`: the case index) |
| Answer, credit | `positions`, `graded_set` with `tolerances` `match_m` m = `JobSpec.match_m` (twice the job's unit) plus the largest widening of an item, `resolutions_m` (one per gold item, in the gold's item order, one pixel at the item's last recognisable sighting) and `fp_weight` 0.5. The answer's items are matched one to one to the gold's by maximum summed credit within 4m; each matched item earns its own four quarters, its direction quarter within the match radius M alone (a list item is a place, 0.11.0: methodology rule 7); `s` is the matched credit over `|gold| + 0.5 |false|`, a missed item costing one and a false item (outside every don't-care circle) half; an empty gold's one thing is the empty list, `s = 1 / (1 + 0.5 |false|)` (0.11.0) |
| Scope, `needs` | `memory`; see, pose, status (the phases the anchor is made of) |
| Gold function | `questions_task.task_things(w)`: one `Moment` for each task, keyed by its case index, with one candidate; no subjects and no entities, so it reserves nothing; evidence the items' last recognisable sightings and always the task's last keyframe, so an empty gold has its evidence |
| Class | `empty` or `items` (`answer_class`); the questionnaire thins the empty golds by `TaskThings.empty_keep_percent` (`CAPPED`), after `RELAX` has recomputed the class |

## The predicate

`TaskThings` is declared per job and read from what the robot was sent (a task's steps: the base position and
`status.phase`) and the public map, never from the job's plan, so a table can rebuild it.

| Anchor | What the things are near | Built from |
| --- | --- | --- |
| `path` | the way the robot drove | the base positions of the steps whose phase is in `phases`, up to the task's last step of `until` (none: no anchor); a segment joins two matched steps that follow one another |
| `point` | one place | the median x and the median y of the matched steps' base positions |
| `rooms` | the rooms worked in | the regions holding the base of a matched step, and the doorway ways (`zone_shapes`) of their openings |

A footprint's distance is to its nearest point, not its centre. For `path` and `point`, a footprint within `reach_m -
band_m` is inside, within `reach_m + band_m` is in the band, and further is outside. For `rooms` (band 0), wholly inside
a room and off every way is inside, on a way or across a room's edge is in the band, and the rest is outside.

**Item kinds** (0.11.0, `TaskThings.item_kinds`; empty means every kind, as for the helper and the AMR). A predicate may
name the only kinds of a carried thing or clutter that can be an item: the vacuum's are the kinds a brush catches
(`hazard_rule.CATCH_KINDS`: extension cord, power strip, cable bundle, floor towel, and the bath mat, pedestal mat and
clothes pile). A kind it names that `dont_care_kinds` also holds is a don't-care place, not an item (the vacuum's three
that recur at one spot, `hazard_rule.SPOT_KINDS`). A carried thing or clutter of any other kind (a shoe, a bag, a toy) is
**nothing**: no item and no don't-care circle, so a list that includes it is charged a false place, as `tangle_ahead`
charges one for a kind it leaves out; the question names what the brush could catch, and the kinds are the ones
`tangle_ahead`'s question names, in the same words ("a cable bundle, an extension cord, a power strip or a towel; a bath
mat, a clothes pile or a pedestal mat counts neither way": `hazard_rule.catch_words`, built from `TANGLE_KINDS` and
`SPOT_KINDS`, final review S4). Light furniture and people stay don't-care by their mobility,
whatever their kind, as the prompt's "people and chairs are not counted either way" says.

A thing is judged twice, by its footprint as placed and by the one a detection gives (its extents about its centre, no
heading), and is in the band wherever they disagree (`task_predicate.judge`), so the oracle, which sees only the
detection, is not charged for a heading it cannot know.

## The gold

An **item** is a `carried` or `clutter` entity inside the predicate at its last recognisable sighting in the task. It
is a **don't-care place** (a circle of `match_m` plus the thing's widening) when it is not outside the predicate and:
it is in the band; it is light furniture or a person; its kind is in `dont_care_kinds`; the robot's own log tells about
it (`World.told`); it has a twin (another entity of its description that some frame of the window shows); it has a row in
the case's last 8 frames (`World.end_window`); it is not recognisably seen in the task, or moves within it; or it is off the floor by less than
5 cm beyond the robot's climb when the predicate says `floor` (a kerb); or it is no taller than `min_height_m` (a remote,
a cable, a sheet: the prompt's definition of a thing counts it, so naming it costs nothing; 0.7.0, the dev pilot's log
review found sol listing a TV remote and losing credit where the gold had made it nothing; every shipped job declares
`min_height_m` as 0, so in 0.11.0 the rule removes only a thing of no height). An item's other places in
the task, 0.15 m or more from its last, are don't-care too: the question does not say "last". A thing clearly off the
floor (its base more than the climb plus 5 cm from the robot's) wherever it stood in the task is nothing, when the
predicate says `floor`. The racks' floor-level stock (`rack_stock.floor_stock`, scenery that is no entity) is a
don't-care place wherever its pallet is not outside the predicate.

**The swallow closes free don't-care space.** An answer point inside a circle that matches nothing is dropped, not
counted wrong, so a system that knows where a circle is without this case's sensors could put points in it, next to
likely items, at no risk. A circle is *sensor-free* when its place is knowable that way: the racks' stock and light
furniture (the map), a told thing (the log), a thing no frame of the window shows. An item within the circle's radius plus
the question's `match_m` of a sensor-free place is don't-care, so those points reach no item. The circles of what the
sensors show (a seen twin, the band, a person, a thing in the case's last 8 frames, a `RELAX`-named thing) swallow nothing.

`RELAX` makes the things a look question names (`ego_now`, `ego_mem`) don't-care here and recomputes the class.

## The answerers

- **Tables** (`table_task`): the tracks with a sighting in the task, at their last place there, that are no fixed piece of
  the map, are of a kind in `item_kinds` where the predicate names any, are taller than `min_height_m` (the gold makes a thinner one don't-care; a table that never lists it is not charged),
  stand on the floor when `floor` is set, and whose detected footprint reaches
  the predicate (`task_predicate.reaches`). `current_view` holds no task message and lists nothing.
- **`path_log`** (`baselines_path_task`): the best of its candidates: the learned answer over the pool of other scripts' golds,
  with the centres of the public map's floor-level surfaces within the reach of its own logged anchor as an extra
  candidate (it names no lane or clutter anchor), and the recurring-spots rule, which needs no picture: items stand at
  the same spots from script to script (within 0.5 m of another script's item for 21 of 24 helper items, 22 of 28 AMR,
  21 of 36 vacuum on release-c), so it lists the places other scripts' golds put items that lie within the predicate's
  reach of this task's logged anchor (`task_predicate.anchor_of`, `reaches`; places within 0.5 m merged, the empty list
  when none is in reach). Each prior record carries the anchor of its own task (`answer_keys.RECORD_KEYS`), so the rule is
  scored on each script left out, its claim made from the other scripts' golds alone, and answered only when that mean
  beats what the pool's learned answer is held to (abstaining's, its held-out mean, or its in-sample mean under the
  `in_pool` twin). It needs the public map, as the claim does.

## The jobs' predicates

| Job | Predicate (`phrase`) | Anchor (phases, until) | Reach / band | Floor | Other |
| --- | --- | --- | --- | --- | --- |
| `vacuum/2` | on the floor of the rooms you cleaned that your brush could catch (a cable bundle, an extension cord, a power strip or a towel; a bath mat, a clothes pile or a pedestal mat counts neither way) (0.11.0; 0.9.0 and 0.10.0: none, the family was dropped; 0.8.0 and before: "on the floor of the rooms you cleaned", over 0.02 m) | rooms (`cleaning`) | none (the rooms and their doorway ways) | yes | `item_kinds` the seven `hazard_rule.CATCH_KINDS`, `dont_care_kinds` the three `hazard_rule.SPOT_KINDS`; asked once for each cleaning task of the window (an end family beside `object_where`) |
| `helper/1` | on the floor within 1.2 m of the way you went to fetch and put down the thing | path (`navigating`, `grasping`, `carrying`, `placing`, until `placing`) | 1.2 m / 0.4 m | yes | |
| `amr/1` | within 2 m of the way you drove | path (`transit`, `approach`, `at_station`, `undocking`, `waiting`: every step of the task, no `until`, so the drive to a final stop that was blocked and skipped is part of it) | 2.0 m / 0.3 m | no: the rack-held stock stays don't-care | |
| `drone/1` | none: the drone asks no `task_things` | | | | the floor surfaces near the drop point score +0.25 to +0.40 against its gold (vehicles stand in parking lanes and loading zones), and with vehicles made don't-care 0.25 items a task and 80% empty are left ([question-design.md](../question-design.md), section 3) |

Light furniture and people are don't-care by what they are (`mobility`), so no job lists a kind of them. Every job's
`empty_keep_percent` is set from the measured empties by 100 (1 - p) / (2 p): the AMR's is 43 (p 0.537 over release-c's
41 short and medium tasks after `told` and `RELAX`, measured again for the whole-drive path in 0.6.0 lane R3), the
helper's 26 (p 0.659). The vacuum's (0.11.0) is 20: the first compile of its list (keep 100) asked 39 tasks on the 32
short and medium vacuum cases of the dev root, 28 of them empty after `told` and `RELAX` (p 0.718), and
100 x 11 / 56 = 19.6.

## What the agent is told

The question's own text says what a thing is (`questions_task.COUNTED`; until 0.10.0 the `positions` answer shape in the
prompt, `answers.PROMPT_SHAPES`, said it, which served this family alone, and the shape now gives the form only): an object
that is not part of the room or its fixed furniture and fittings (a docking station, a charger, a door, a table, a shelf, a
rack, the room's own floor mat), so a floor lamp, a fan or a heater is a thing; a pair or a stack is one thing; people
and chairs are not counted either way. That is the gold's boundary (`THINGS`, `ITEMS`): fixed pieces are never an item
and never a don't-care place (a don't-care circle there would swallow the items beside it). The credit section says a
list earns each thing it holds the same way as any answer (four quarters, its right direction being within a
matching radius of the thing), a thing left out costs a whole thing and a place that matches nothing half of one,
unless it is at a thing the question counts neither way; a repeated place matches once, and when there were none the
empty list is the one thing, 1 over 1 plus half the false places.

The vacuum's question reads "Where were the things on the floor of the rooms you cleaned that your brush could catch (a
cable bundle, an extension cord, a power strip or a towel; a bath mat, a clothes pile or a pedestal mat counts neither way)
in your first task, which you would have to steer around next time? Answer in map coordinates." and then the same sentence
that says what a thing is. The vacuum's credit section already carries the list paragraph (`brief.asks_lists` is true
for it since 0.10.0, through `tangle_ahead`), so 0.11.0 adds no prompt text for it: it is told how a list is credited
before the first `tangle_ahead` is asked. The list catastrophe (a cord within `FRONT_M` of the front edge, -3) belongs to
`tangle_ahead` alone: an end list is asked after the job and carries none.

## Gates

`memory_needed`, shortcuts, separation and `BALANCE` (the empty share, capped at a third by `empty_keep_percent`). A job
where any fails drops the family.

**The vacuum's list in 0.11.0** (the check of the vacuum's 36 dev cases alone, `--verdict evidence`; 9.3 minutes on 12
workers) passes every gate: ceiling +1.00, knowable +0.00, `memory_needed` +0.00 (`current_view` -0.545), shortcuts +0.00
(`blind`, `path_log`, `questionnaire_blind` and the form adversary all answer `unknown`), separation +1.00 against 0.40, and
the balance rule +0.000 held out; `ALL GATES PASS`, `EVIDENCE VERDICT PASSES`, no `BROKEN` line. The gate that dropped
the family in 0.9.0 reads the other way now: `path_log_in_pool` scores -0.298 (it was +0.258 over `all_unknown` on the
long cases and +0.135 on the medium ones): the kinds that recur at one spot are no item now, and the cords and the floor
towel recur at 0 to 43% within 0.1 m ([`tangle_ahead`](tangle_ahead.md)). No fallback or drop rule applied and no
threshold moved. The release check reads the same way: ceiling, `knowable`, `memory_needed` (`current_view` -0.545),
shortcuts and separation as above, `ALL GATES PASS` and `EVIDENCE VERDICT PASSES` over the four jobs; `path_log_in_pool`
scores -0.458 and `current_view_prior_in_pool` -0.452 (0.11.0's release check read -0.504 and -0.496, before the
scorer stopped a don't-care circle from costing a list credit), and the best held-out constant gains -0.452 over
`all_unknown` (interval -0.720 to -0.105), so none beats abstaining. The helper's and the AMR's lists pass in the same
check, their largest shortcut leads being +0.00 (helper) and +0.00 (AMR) against the line of 0.10 (the helper's read
+0.01 in 0.11.0's).

## Supply

The vacuum's list in the 0.11.0 release (36 cases, 16 scripts), at the declared keep of 20: 4 questions on the 16 short
cases (3 with items, 1 empty), 11 on the 16 medium cases (8 and 3) and 7 on the 4 long ones (6 and 1), 22 in all, 0.61 a
case; on the short and medium cases 15, 11 with items and 4 empty (27%, under the third the keep aims at). On the dev
root's first compile, before the thinning (keep 100), it asked 12, 27 and 20: 1.7 a medium case, 28 of the short and
medium tasks' 39 empty after `told` and `RELAX`. The 17 questions with items hold 19 items (1 or 2 each), and every gold
2 to 12 don't-care circles. The helper's list has 47 questions (39 with items, 8 empty; 31 in the short and medium
cases) and the AMR's 47 (34 with items, 13 empty; 38).
