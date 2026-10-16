# `distance_between`

How far apart two things were, as the robot last saw them. End, `memory`. Code: `stbench.questions_things` (shared with
[`when_seen`](when_seen.md) and [`order`](order.md)), with `stbench.questions_where` (`widened_ladder`). The record and
questionnaire rules are in [questions.md](../questions.md).

**Not asked in 0.11.0.** No job's `ends` names it. It left every job in 0.5.0-control (an internal checkpoint, folded
into 0.7.0 in the [CHANGELOG](../../CHANGELOG.md)): the evidence verdict, which reads the gates per slice, found a
constant that pooling had hidden (+0.375 on the helper's short cases). The module and its tests stay.

| | |
| --- | --- |
| Category | C3 Locate |
| Asked | by no job. End, once per unordered pair of relevance kinds of the window, a kind with itself included (the job's `ends` would name it). Through 0.5.0 `helper/1`, `amr/1` and `drone/1` named it; `vacuum/2` took it out of `ends` earlier (under one question a case). In 0.4.x (credit `relative`) it passed each one's gate check: `helper/1` 18 questions, separation +0.83, and `amr/1` 26, +0.85 (seeds 2001-2006); `drone/1` 77, +0.84 (its final check, seeds 2001-2012). The 0.5.0 definition below is the last any job named; the family left the three jobs in 0.5.0-control (above) |
| Text (slots) | "How far apart were the {thing} and the {thing}, as you last saw them? Answer in metres." (`first`, `second`: each a description `{"kind", "attributes"}`) |
| Answer, credit | `number` (metres, 3 places), `ladder`: the job's rungs (`JobSpec.ladder`, u, 2u, 4u and 8u: 0.25, 0.5, 1 and 2 m for a home, 0.5 to 4 m for the warehouse, 1 to 8 m for the city), each plus the two kinds' `rules.widening` (`questions_where.widened_ladder(ladder, *kinds)`); 0.4.x scored it by `relative` credit |
| Scope, `needs` | `memory`; see, pose |
| Gold function | `questions_things.distance_between(w)`: one `Moment` per pair of kinds, keyed "a\|b" with the kinds sorted |
| Slice | sliced (`SLICED`): the unknown slice is drawn per moment (`questionnaire`) |

## The credit

The gold is a distance between two things' placements, and an agent that places a thing from its depth reads the surface
it saw, not the thing's centre: an error of half a pallet's length is no error of memory. So the credit is the position
ladder, absolute, with each rung widened by both things' extent. `rules.widening` of a kind's catalog box is `max(0,
max(x, y) / 2 - 0.25)` metres (zero for a mug or a bag, 0.35 m for a pallet), and the row's rungs are the four of
the job's ladder (`JobSpec.ladder`, [`object_where`](object_where.md), "The ladder"), each plus the two widenings
(`widened_ladder(ladder, *kinds)`). An answer scores the share of rungs its absolute
error is within (`tolerances` `{"ladder": [...]}`, written on the unknown rows too). 0.4.x scored a relative error
(`relative`): 5% of a 0.73 m gold is 3.6 cm, far tighter than a surface read can hold and a different scale from
`object_where`'s position ladder. `relative` stays in the scorer, but no family uses it.

## Pairs

Both things are *settled*. A thing is settled when:

- it has a recognisable sighting in the window ([`object_where`](object_where.md) has the per-frame rule), and the pair
  is read at its **last** one;
- no visibility row, at any pixel count, shows it in the window's last look window (the recent view would answer: methodology
  rule 4); and
- every sighting of it from that last recognisable one to the window's end, recognisable or not, lies within
  `MOVED_NO_M` (0.15 m) of the place it stood at then (`World.steady`).

The last clause is the relaxed one (0.5.0): a clipped, dark or small sighting at the same place leaves the thing
settled, since the memory table that reads it holds the same place, give or take a few centimetres; only a later
sighting elsewhere unsettles it. 0.4.x refused any later sighting at all, which dropped golds a table would have got
right. [`when_seen`](when_seen.md) keeps the strict form, because its gold is a time and a later sighting is a later
time.

**Neither thing of a pair may be hidden by a look-alike** (0.5.0's pilot fix round), **and each must show its colour
word** (0.5.0-control): a real thing, settled or unseen, that another sighted thing of a confusable kind could be taken
for the colour of (by its render; black, grey and white are one group), or whose own picture does not show its word, is
in no row, answerable or unknown (`questions_where.nameable`; [`object_where`](object_where.md), "Real things and
look-alikes"); the unseen side of an unknown row also takes the unseen-only look-alike groups (`CONFUSABLE_UNSEEN`: a
never-seen bin beside a sighted drum of its colour). The screen is
applied in this family alone of the three that share `questions_things`: no job asks `when_seen` or `order`.

A pair is asked when the two (x, y) positions at their last recognisable sightings lie at least `MIN_APART_M` (0.5 m)
apart. The gold is that distance, rounded to 3 places, from the placements at the two sightings (the delivered ones): a
thing moved after its last sighting keeps the place it was last seen, so nothing unobserved changes a gold (truth rule
2). The evidence is the two last recognisable sightings, sorted by keyframe, and their entities. The row is critical
when both things are, each at its last recognisable place in the task it was seen in (`questions_where.critical`: kind,
region, floor).

**Unknown:** a settled thing with an unseen one of the pair's other kind (the same kind for a kind with itself): the
unseen things are those that exist in the window and show no pixel in it, and the absent descriptions of a relevance
kind ([`object_where`](object_where.md), "Absent descriptions": a kind sighted in the window and no confusable
same-colour kind), the unseen things also screened for look-alikes as above. The gold is `unknown`, the tolerances the widened ladder as on answerable rows. The world's own
unseen things are offered before an absent description, and a case asks at most one absent description over all its end
questions.

**The order the pair is named in** is a coin per pair, `stable_hash(case id, the two subjects in sorted order, "named")
% 2`, which reads no answer (the distance is the same either way) and no slice; a subject is `e:<id>`, or
`d:<kind>/<attributes>` for an absent description. A pair across kinds is asked here, unlike [`order`](order.md): the
answer is a number scored on a ladder of absolute error, not a coin, and how far a prior can get from the kind pair
alone is what the dev gate check measures (the priors' keys loosen the nested `attributes`, then `kind`).
