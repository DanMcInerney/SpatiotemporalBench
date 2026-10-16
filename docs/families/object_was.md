# `object_was`

**Not asked since 0.6.0 (an internal checkpoint, folded into 0.7.0 in the [CHANGELOG](../../CHANGELOG.md)); no job asks
it in 0.11.0.** No job lists `object_was` in its `ends`: at sourced change rates the family asks too few questions to
read (on release-c it fell under the supply line of 0.5 questions a medium case on every job; see "Supply"). The module
and its tests stay; what follows describes the family as built.

Where a thing that moved was during a named task, at its earlier or its later state. End, `memory`, temporal. Code:
`stbench.questions_change` (the gold), with `stbench.questions_where` (`answerable_at`, `askable`,
`nearest_slant_at`, shared with [`object_where`](object_where.md); its relative rungs, `relative_tolerances`, are its
own since 0.7.0) and `stbench.questions_world`
(`World.steady`, `World.told`). The record and questionnaire rules are in [questions.md](../questions.md). New in 0.5.0;
reworded in 0.6.0, when its moments stopped being the job's events and became draws.

| | |
| --- | --- |
| Category | C6 Change |
| Asked | by no job. End, as many draws as the window's candidates allow (a job's `ends` would name it; `amr/1` was the last to) |
| Text (slots) | "Where was the {thing} during your {ordinal} task? Answer in map coordinates." (`kind`, `attributes`, `task`: the case task index) |
| Answer, credit | `position` (xy), the **relative ladder** of the slant at which the place was seen (`range_m`), widened for a large thing (`rules.widening`) |
| Scope, `needs` | `memory`; see, pose. `FamilySpec.temporal` is true |
| Gold function | `questions_change.object_was(w)`: moments `draw:<j>`, each holding the window's earlier candidates in `answerable` and its later candidates in `unknown` |
| Class | `earlier` (the first pool) or `later` (the second): `answer_class` on the gold row, balanced by the coin below. The slice is `answerable` for both |

## Time and order are the tasks'

The world changes only in the gaps between tasks (`docs/world.md`), so a task is the finest time step with an answer, and
a question names a task by its ordinal and no event of the job. A window of one task has no change to ask about, so a
short case asks none; a window of two tasks can ask it (it names the first task at the earlier state and the second at
the later), which is why the moments are draws over the window and not tasks.

## A thing's place in a task

A thing has a **place in a task** when its last recognisable sighting lies in the task (`questions_where.answerable_at`
at the task's last keyframe; [`object_where`](object_where.md) has the recognisability rule) and every sighting of it
from there to the task's end, a glimpse or a dark frame too, lies within `rules.MOVED_NO_M` (0.15 m) of that place
(`World.steady`). The gold is that place; its evidence is the sighting. The steadiness rule is what keeps the gold the
memory table's: the table reads the latest sighting in the task, so a thing recognisable at one place and glimpsed at
another later in the task would be held where the gold does not read it.

The things are `questions_where.askable` under the relevance of the task a question names: carried things of the task's
relevance kinds with a description unique in the world, that the log does not tell about (`World.told`), and that no
sighted look-alike hides or whose picture does not show its colour word. A thing with any visibility row in the window's
last look window is not asked (the recent view would answer it, as for `object_where`).

## Candidates: a move beyond the asked state's last rung

A thing is asked at its **earlier** state in task i when a place it has in a later task lies **beyond the last rung of
task i's ladder** from its place in i, and at its **later** state in task j when a place in an earlier task does. The
ladder is the one the question is scored by: 10%, 25% and 50% of the slant `range_m` at which the place was seen (the
nearest of the thing's recognisable sightings in the window that stood within 0.15 m of it, `nearest_slant_at`), at least
5, 10 and 20 cm, plus the kind's widening. The gap is read against the asked state's rung, so a wrong-state answer, which
is the other place, earns no credit: it scores the penalty, not a third of the credit. A move that passes one state's
rung and not the other's gives the thing one candidate; a thing in a task between two such places has both. The distance
is strictly beyond the rung: at the rung an answer earns a third of the credit.

## Two pools, a draw each, a fair coin

The earlier candidates are the `answerable` pool of every moment (class `earlier`) and the later candidates the
`unknown` pool (class `later`), the two-pool shape of `COINS`. A moment is `draw:<j>` for j below the smaller number of
distinct things either pool holds. Every moment offers both pools whole, so it is asked only when both allow a candidate
under the subject rule (no two questions of a case name one thing), and the pool it takes is the questionnaire's coin
(`stable_hash(case id, family, key, "slice") % 100 < 50`, `questionnaire.COIN_PERCENT`). So neither the moment nor
whether it is asked predicts the state, and over many cases the asked states are half and half
(`tests/test_questions_object_was.py` holds the share within 0.03 of one half over 2,000 synthetic case ids). The draws
replace 0.5.0's hash alternation, which left release-c's ten AMR questions tilted (`first_table` +0.60). The subject is
the thing alone (`e:<id>`): a task ordinal is no subject (methodology rule 6, amended).

## Why it is temporal

A moved thing is somewhere else by the end of the window, and the question names a time. The answer is the place the
thing had *then*, which was the earlier place for half of the questions and the later place for the other half. Only a
record that keeps where the thing was seen in each task answers both: a notebook that keeps one place for each thing,
with no time, is right on one state and wrong on the other. There are two such notebooks and `time_needed` reads both:
`latest_table` (right on the later state) and `first_table` (right on the earlier). The gate passes when
`max(latest_table, first_table)` <= `all_unknown` + 0.10, so the coin must hold the states near half and half.

**What the words reveal, openly.** The family asks only about things that moved, so a question is itself a hint that the
thing is somewhere else in another task, and in a window of two tasks the task named says which state is asked: the
first task means the earlier state and the second the later. A memory that keeps one place for each thing *and the task
it was seen in* therefore scores about +0.5 on two-task windows (it answers the later state and abstains on the earlier).
That is not a shortcut: a place with its time is the spatiotemporal memory this family tests, and the notebooks the gate
reads (place without time) stay at 0. In windows of three or more tasks the named task alone does not say which state is
asked, and the memory has to hold each task's place. `memory_table` (`table_change.was_during`) reads the latest sighting inside the named task, the very sightings the
gold reads, so the oracle is held to the ceiling gate.

## Supply

Natural change among the things a home robot sees twice is thin (0.6-1.4% a gap), so release-c's timelines give this
family well under half a question a medium case; a generator that changes things by construction would supply it. The
supply rule ([questions.md](../questions.md)) keeps it in a job's `ends` only when the whole-root compile asks at least
0.5 of its questions a case in some shipped length.
