# `ego_mem`

"As you last saw it, where is the point of the {thing} you saw that is nearest to your camera now, measured along the
floor? Answer in metres ahead of the camera and to its left." (0.7.0, ability *memory*): the offset, from the camera as
it is at the look point, of the nearest part of a named thing the robot saw before the recent view at the look point (the
last `LOOK_WINDOW` = 8 frames of the task up to it, `look_windows.recent`) and does not see in it. During, `memory`. Code: `stbench.questions_ego` (shared with [`ego_now`](ego_now.md)), `stbench.nearest_part`,
`stbench.ego_frame`. It replaces `out_of_view` (one named thing: no empty sets, no rack-stock don't-cares). The offset
frame, the nearest-part target, the resolvability screen, the credit and the naming screens are described in
[ego_now.md](ego_now.md); this page is what differs.

**Who asks it (0.11.0).** It is the second slot of every job's look table (`JobSpec.look_shares`,
[questions.md](../questions.md), "Look points"): the AMR 100% of points, the drone 70%, the vacuum 100% and the helper
65%, beside [`way_to`](way_to.md) at 35% (it had 25% beside [`person_mem`](person_mem.md) at 40% until the check dropped
`person_mem`, and `ego_mem` took its share). A point's slot draw reads the case id and the point alone; a drawn family
with no candidate asks nothing. Through 0.9.0 the AMR drew it at 35%, the drone at 85%, the vacuum at 30% and the helper at 30%.
Two rules changed in 0.10.0: the recent view slides with the point, and a floor robot is asked only about things on the
floor (below).

| | |
| --- | --- |
| Category | C3 Locate |
| Asked | during, at a look point (as `ego_now`); `JobSpec.looks` and `look_shares`, by all four jobs |
| Text (slots) | "As you last saw it, where is the point of the {thing} you saw that is nearest to your camera now, measured along the floor? Answer in metres ahead of the camera and to its left." (`kind`, `attributes`) |
| Answer, credit | `offset`, `graded` on the 2-D error against the job's clearance C (as `ego_now`): the resolution is one pixel at the camera of the evidence sighting that set the surface tie |
| Scope, `needs` | `memory`; see, pose |
| Gold function | `questions_ego.ego_mem(w, k) -> Moment \| None`: the point's `answerable` and `unknown` pools, key `str(k)`; None when nothing is answerable |
| Slice | sliced (`SLICED`): the questionnaire draws the unknown slice per point (`UNKNOWN_PERCENT`, 20%) with the same template. `answer_class` is `answerable` or `unknown` (`records.BALANCE_POOLS`) |

## Answerable

A named thing that
- has a **recognisable sighting before the recent view** (the last 8 frames of the task up to the look point `k`; the last
  one, `j`, is the
  evidence);
- has **no visibility row, at any pixel count, in the recent view** (it would answer);
- is **unchanged**: every sighting of it from `j` to `k`, recognisable or not, lies within `nearest_part.UNCHANGED_M`
  (1 cm) of its place at `j` (`World.steady`, since 0.7.0; 0.6.0 allowed 0.15 m, over C/2 for a vacuum), so the table, which
  unions the hulls of the sightings at one place, holds the thing where the gold reads it;
- has a name that is **unique among the things sighted in the window up to `k`** (`World.sighted`): no other such thing
  of a kind it can be mistaken for (`descriptions.CONFUSABLE`, its own kind included) has its description or could be
  taken for its colour (`colour_seen.could_be`, the neutrals one group), by the screens of [ego_now.md](ego_now.md):
  "a bag beside a box of its colour" is unaskable here as for `object_where`, and a bin and a drum hide each other
  only when the other was never plainly made out by `k`. A twin the robot never saw cannot confuse it, and one first
  seen after `k` does not exist yet; this lifts the supply against naming only world-unique things. A kind-only name
  (clutter) is hidden by any other of its kind or a look-alike of it, light furniture included. Light furniture is never
  asked: its place is a fact of the layout ([ego_now.md](ego_now.md));
- bears its colour word out in the **frames up to `k`**: nothing after the look point decides a question.

The gold is the point nearest the camera's floor point at `k` of the **union of the floor hulls of every recognisable
sighting before the recent view at that place** (the sightings within 1 cm of the place at `j`; `nearest_part.union_hull`,
`nearest_part.nearest_part`), put in the camera frame at `k`: "as you last saw it", whatever has happened since (truth
rule 2). A side that no sighting showed does not count, and a sighting from before a move says nothing of this place.
Three readings of "the nearest part" differ from this gold by more than C/2 on a third to a half of the questions (the
side that faced the robot at the sighting, the thing's true near side now, the middle), which is why
the text says "you saw", "now" and "along the floor", and the prompt says that a side the camera never showed does not
count. The last sighting alone differs from the union by more than C/2 in 0 to 8%, and no frame from the recent view on is read.

**Resolvability** is `ego_now`'s (the surface tie, [ego_now](ego_now.md), "Resolvability"), with each contact
pixel's error read from the camera of its own sighting and the edge tie from the sighting's own camera (a thing seen from
near passes at a tight clearance though it is far from the camera now, and a thing seen from afar does not), and so are
the no-depth rule and the placement rule: the sightings at the place must lie in one placement of the timeline, or a
table that reads centres and the gold could differ.
**The edge tie** is `ego_now`'s too (see
[ego_now](ego_now.md), "Raised things"): a remembered thing resting on a surface whose top is at or above the camera of
a sighting is tied by the surface's near edge, from that sighting's camera, when its gold point lies within C/2 of it along the
sight line. **The credit** is the job's
clearance C, widened to a tenth of the distance beyond ten C (0.11.0), and the point's resolution (`ego_now`, "The
credit"); `range_m` is the slant from the camera now to the point, at the height of the visible point nearest it, for
the report.

## Floor robots (0.10.0)

For a job whose robot works the floor (`Relevance.floor`: the vacuum and the AMR) only a thing that stands on the floor
(`World.on_floor`) at the sighting the gold rests on is asked, and the filter applies to the `unknown` pool too: an unseen
thing is asked only when it stands on the floor at the point. Otherwise a table-top thing could only ever be an `unknown`,
and its kind would predict the slice. The helper and the drone ask of every thing. The filter cut the vacuum's `ego_mem`
by 29% and the AMR's by 44%.

## Unknown

20% of the points' questions, with the same template:
- a **real thing with no pixel anywhere in the window** that exists at `k` and is not hidden by a look-alike sighted in
  the window (`CONFUSABLE` and `CONFUSABLE_UNSEEN`: nothing seen says which is meant, so a yellow bin never seen is not
  asked beside a yellow drum seen). It is never met later in the case, so no frame of the case answers it;
- or an **absent description** (`questions_where.never_seen`, which screens the same groups over the window) of a kind
  already sighted by `k`, so the kind alone gives nothing away. The questionnaire asks at most one a case.
Told things are never named. Both pools draw from the same kinds, so no kind predicts the slice.

**The glimpse screen (S4, 0.11.0).** Either kind of `unknown` candidate is dropped when, between the case's first frame
and the look point `k`, a thing of a look-alike kind (`descriptions.lookalikes`, the unseen groups too) was glimpsed (a
sighting under the recognisability floor, `World.sighted` without `World.recognised`) in a frame whose `rgb_med` may be
taken for the asked colour (`questions_where.glimpse_hides`; `colour_seen.could_be`, the neutrals one group). The screens
above read a glimpse by its label, which is all they have of it; its render can say otherwise, and a person asked for the
blue bag could not tell it from a bluish glimpse. A name with no colour (clutter, named by kind alone) is never screened,
a frame after `k` is never read, and the answerable pool is not screened. On the dev root it removes 6 of the 95
`unknown` rows of `ego_mem` and `object_where` (6%: AMR `ego_mem` 3, helper `object_where` 2, vacuum `ego_mem` 1; the
compile redraws 3 of them from the rest of the pool, so 95 become 92, and one answerable AMR `ego_mem` row goes with an
`unknown` pool that emptied). The wider screens were not applied: any colour of a look-alike glimpse
(S3) would remove 48% of the rows and any glimpse in the colour (S1) 57%, which leaves the AMR's `ego_mem` and the
helper's `object_where` no `unknown` rows at all.

## Answerers

- **Tables** (`table_ego`): the hull of the union of the `floor_hull`s of the sightings, at the place of its latest
  hulled one, of the track the description names (the track with the latest hulled sighting), and its point nearest the
  ask step's camera, in that frame. `current_view`'s store holds the latest frame alone, so what it saw before is
  `unknown` to it: the memory-needed gate holds it under the floor. The oracle and `memory_table` read the same hulls the
  gold takes ([format.md](../format.md)) and reproduce it.
- **`path_log`** (`baselines_path_ego`): it has no picture, but a public layout holds things at fixed spots. The prior's
  `ego_mem` golds of the kind are put back on the map with each record's base, heading and the robot's camera mount, and
  the dense places among them (`places.dense_places`, a place being the golds within the golds' last rung, 2C of the
  job's clearance) that its camera faced at a logged step are candidates. A place is judged on the records that would
  have fired it (those whose own cameras, up to their ask, faced it; the record's `cameras`), by the point of its
  cluster nearest each record's camera with that record's script left out, kept only where the held-out gain over
  abstaining has a bootstrap interval over the scripts above 0. The answer is the point of the winning cluster nearest
  the ask's camera, in the camera frame of the ask. Else the pool's learned constant offset, kept
  only where its held-out gain over abstaining has a bootstrap interval over the scripts above 0 (`blind.learned_answer`, as for
  every learned answer), else `unknown`.

## Gates and risks

`ceiling`, `knowable`, `memory_needed`, `shortcuts` and `separation`, and the balance check
(`answerable` against `unknown`; the best constant offset scores -0.70 to -0.77). The family declares no reference
(`kind_range` answers [`ego_now`](ego_now.md) alone and is `unknown` here). The risk is `path_log` with the map where
things sit at fixed spots (staging lanes).

## Supply

On release-c (24 short and medium cases a job) 0.6.0 asked 46 vacuum, 73 helper, 23 AMR and 70 drone answerable offset
rows, with 14, 24, 8 and 20 unknown ones; with the nearest-part gold, the resolvability screen and the 1 cm unchanged
rule the answerable rows are 34, 69, 22 and 70 (the unchanged rule alone costs none there; the screen drops 26%, 5%, 4%
and 0%). On the dev read (144 cases, 0.7.0) the surface tie leaves 24 of 28 answerable rows on the AMR, 108 of 110 on
the helper, 57 of 69 on the vacuum and 32 of 39 on the drone (the questionnaire redraws among what is left); the
unknown rows are not screened by the surface tie (the S4 glimpse screen of 0.11.0 is the one that reads them).

In 0.10.0 (the dev root) the family asked, in the short and medium cases, 22 questions on the AMR, 55 on the drone, 87
on the helper and 70 on the vacuum, at fills of 0.026, 0.161, 0.512 and 0.055 a drawn slot. The helper's count was from
its old slot 2 (25% of points); at 65% it draws the family at more points, and since a thing an earlier look has named is
not asked again, the gain was for the next compile to say. The 0.11.0 release has, over all lengths (in the short and
medium cases), 152 (130) questions on the helper, 83 (70) on the vacuum, 68 (55) on the drone and 28 (24) on the AMR;
122, 66, 54 and 18 of them are `answerable` and 30, 17, 14 and 10 `unknown`. It is capped by the distinct things a case
holds: the floor holds few, and a thing an earlier look has named is not asked again (401 of the AMR's 1,043 draws found
only such things).
