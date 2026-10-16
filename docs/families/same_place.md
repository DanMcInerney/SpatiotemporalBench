# `same_place`

**Not asked since 0.6.0 (an internal checkpoint, folded into 0.7.0 in the [CHANGELOG](../../CHANGELOG.md)); no job asks
it in 0.11.0.** No job lists `same_place` in its `ends`: at sourced change rates the family asks too few questions to
read (on release-c it fell under the supply line of 0.5 questions a medium case on every job; see "Supply"). The module
and its tests stay; what follows describes the family as built.

Whether a thing was in the same place in two named tasks. End, `memory`, temporal. Code: `stbench.questions_change` (the
gold), with `stbench.questions_where` (`answerable_at`, `askable`) and `stbench.questionnaire` (the coin). The record and
questionnaire rules are in [questions.md](../questions.md). New in 0.5.0; reworded in 0.6.0
with a moved and a vacated "no".

| | |
| --- | --- |
| Category | C6 Change |
| Asked | by no job. End, as many draws a task pair and kind as the pools allow (a job's `ends` would name it) |
| Text (slots) | "Was the {thing} in the same place in your {second} task as in your {first}?" (`kind`, `attributes`, `tasks`: the two case task indices, the earlier first) |
| Answer, credit | `yesno`, exact (`tolerances` `{}`); no range |
| Scope, `needs` | `memory`; see, pose. `FamilySpec.temporal` is true |
| Gold function | `questions_change.same_place(w)`: moments `<i>\|<j>\|<kind>\|<draw>` (i < j), the yes candidates in `answerable`, the no candidates in `unknown` |
| Class | `yes`, `moved` or `vacated`: `answer_class` on the gold row. The yes class is the first pool and the other two the second (`records.BALANCE_POOLS`); moved and vacated are reported apart. The slice is `answerable` for all three |

## Places, and the three classes

A thing has a **place in a task** as [`object_was`](object_was.md) defines it: its last recognisable sighting in the
task, steady to the task's end. Its **first place** in a task is its first recognisable sighting there, when nothing was
sighted elsewhere (recognisable or not) from the task's start to it. For tasks i < j and a thing askable under both
tasks' relevance, with `a` its place in i and `b` its first place in j:

- **yes**: `a` and `b` are under `SAME_M` (0.15 m, `rules.MOVED_NO_M`) apart;
- **moved** (a no): they are at least max(`APART_M` (0.45 m, `rules.MOVED_YES_M`), twice the job's unit 2u) plus the kind's
  `rules.widening` apart, so a small move is not a yes, a no or a question at all;
- **vacated** (a no): the thing has no sighting at all in task j (not even a glimpse) and its place `a` was **seen empty**
  in task j: `grid.place_free` over the free-space cells of task j's keyframes (`World.latest_cells`) and the thing's
  half-width (the shorter horizontal side of its box over two). A place nobody looked at is never empty (methodology, truth
  rule 2); the evidence is the sighting in i and the latest keyframe that observed the place (`questions_change.empty_since`).
  It is asked only where the robot's band can show a thing gone (`questions_change._can_vacate`): the robot does not fly
  (a flier's ground band reads every ground point as floor, so a box seen from the air reads empty), the thing stood on
  the floor (`World.on_floor`) and its top is above the band's floor (`grid.band`; a flatter thing leaves no obstacle
  point). `table_change.can_vacate` mirrors the rule from `ctx["robot"]` and the sighting's base height.

A thing with any visibility row in the window's last look window is not asked at all (the recent view would hold its place and
lift `current_view` above the floor), as for `object_was`.

Between yes and moved, a place pair is not a candidate: a shift too small to be a move and too large to be the same
place. The gold is a yes/no, `answerable` either way: `unknown` is only the abstention (it scores 0). The evidence of a
yes or a moved no is the two sightings; a vacated no's entities are empty. Subjects are `e:<id>` only (a task ordinal is
no subject, methodology rule 6 amended). A row is critical when either of its places is.

## Two pools, a draw each, a fair coin

The yes candidates are the `answerable` pool of a moment and the no candidates its `unknown` pool, the two-pool shape of
`COINS`. A moment is a task pair, a kind and a draw index, for each kind with a yes and a no candidate in the pair and as
many draws as the smaller pool has things. Each moment offers both pools whole and is asked only when both allow a
candidate under the subject rule; the pool it takes is the questionnaire's coin (`stable_hash(case id, family, key,
"slice") % 100 < 50` for the no pool), so neither the moment nor whether it is asked predicts the answer, and the yes
share is half over many cases (`tests/test_questions_same_place.py` holds it within 0.03 of one half over 2,000
synthetic case ids). The kind is in the key so that a kind with only yes things or only no things in a pair is not
asked at all.

## Why it is temporal

It compares a thing's place in two tasks. The current view holds neither, and a notebook of one place for each thing
can only say that a thing is where it is. `latest_table` and `first_table` (no event table, one sighting a track) take
both places as that one sighting and answer yes, never moved or vacated, so each scores `2y - 1` at a yes share `y`:
the fair coin is what holds them at the floor under `time_needed`. `memory_table` (`table_change.same_place`) reads the
latest sighting in the first task and the earliest in the second, says yes below 0.30 m (`table_change.SAME_M`, the
middle of the gap the gold leaves), no above it, and no when the second task never sighted the thing and the cells its
mapper kept of that task show the first place empty. `path_log` reads no picture and answers from the pool of the
template's records.

## Supply

The family needs a gap between two tasks, so it asks in medium and long windows. Natural change is thin, and a vacated
no needs the place seen empty in the later task as well as the thing seen in the earlier one. The supply rule
([questions.md](../questions.md)) keeps the family in a job's `ends` only when the compile asks at least 0.5 of its
questions a case in some shipped length; a generator that changes things by construction would supply it.
