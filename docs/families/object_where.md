# `object_where`

Where the robot last saw a thing. End, `memory`. Code: `stbench.questions_where`, with `stbench.recognise` (the evidence
rule) and `stbench.questions_world` (`World.recognised`, `stale`, `arrived`, `told`). The record and questionnaire rules
are in [questions.md](../questions.md).

| | |
| --- | --- |
| Category | C3 Locate |
| Asked | end, once per answerable thing of the window (the job's `ends` names it, and a job that declares none asks it alone). It was the end family of all four jobs through 0.7.0; **since 0.8.0 it ends the helper and the vacuum only**. It left the AMR, whose second end family is [`closest_person`](closest_person.md), and the drone, whose first look slot is [`clear_path`](clear_path.md) and which now asks nothing at its end: a delivery drone re-finds nothing on the ground, and walls are what it can hit. Its +0.81 to +0.91 sol bound is a reason to check that it asks what the job needs (it is credited on the job's own re-find radius R, a loose requirement), not to drop it where no task-critical end family survives, and an end question costs no ask. Its first form (through 0.5.0) asked a named moment of the job ("Where was the {thing} {event}?"), and that anchored form is now `object_was`'s alone |
| Text (slots) | "Where did you last see the {thing}? Answer in map coordinates." (`kind`, `attributes`: the thing's description and nothing else; no moment, no event, no task) |
| Answer, credit | `position` (xy), `graded` on the job's **re-find radius R** (the four quarters): the resolution is the larger of one pixel at the last recognisable sighting and `rules.hidden_widening` for a thing whose top was above the camera at the sighting, the direction quarter is the answer within R of the gold (`e <= R`; the room the thing is in earns nothing), and there is no harmful side; an unknown row carries the same keys unwidened (below) |
| Scope, `needs` | `memory`; see, pose |
| Gold function | `questions_where.moments(w)`: one `Moment` for each answerable thing, keyed by its description, its candidate from `pools` |
| Slice | sliced (`SLICED`): the unknown slice is drawn per moment (`questionnaire`), 20% |

## Moments

One moment is one answerable thing of the window, keyed by its description (kind and describable attributes, as JSON).
It holds that thing alone in `answerable` and the window's whole unknown pool in `unknown`; the questionnaire draws the
slice of each moment from a hash of the case, the family and the key (`UNKNOWN_PERCENT`), so neither the moment nor
whether it is asked predicts the slice. A moment of the job's events is no part of this family: nothing moves within a
task (the world changes only in the gaps between tasks), so the question needs no anchor, and "last seen" says plainly
what the gold is. The text is pure: it is the render of `kind` and `attributes`, and no gold-side resolution is a slot.

## Evidence: a recognisable frame

A sighting is evidence only when its own frame is recognisable (`recognise.recognisable`; methodology rule 10, "recognisable
in a single frame"). The thing's visibility record of that one frame has:

- at least `RECOG_PX` (1,024) pixels and a mean luminance of at least `RECOG_LUM` (0.01, from 0.5.0-control: where the
  pixels carry no colour; it was 0.0058);
- a bounding box at least `RECOG_EDGE_PX` (2) pixels from every edge of the image, so a sliver cut by the frame is not
  a thing a person could name; and
- a median depth (`depth_med_m` not null), so a speck beyond the sensor's range or an occluded corner is not either.

There is no pixel total over two frames and no larger count in one. A frame that fails is still a sighting
(`World.sighted`: it can unsettle a place or put a thing in the window), but a gold never cites it: **a gold cites the
last recognisable frame of the window** (`World.recognised`, `questions_where.last_seen`).

## Things

The asked things are carried things of the relevance kinds of the window's tasks (`JobSpec.relevance` of each task of
the window), whose description is unique among the world's describable things, which the robot's own log does not tell
about (below) and which no look-alike hides (below). Light furniture and clutter are never asked. A thing is one of:

- *answerable*: it has a recognisable sighting in the window, no visibility row at any pixel count in the case's last 8
  frames (`World.end_window`; the recent view would answer it: methodology rule 4), has not moved in the window (below), and is critical
  (below); the gold is the placement's (x, y) at its last recognisable sighting, which is the evidence;
- *unknown*: no pixel at all in the window and it exists at the window's last keyframe (nothing the sensors recorded
  shows where it is), or a description of a relevance kind that matches nothing in this world (an absent description,
  below);
- *skipped*: anything else: pixels in the window but no referent a person could name (a glimpse, only dark, clipped by
  the frame or depth-less), a row in the case's last 8 frames, a move, a place that is not critical, a thing the log tells about
  or one a look-alike hides.

The last recognisable sighting may be in any task of the window: the question is where the robot last saw the thing,
nothing unobserved changes a gold (truth rule 2), and a thing that stayed put is where it was seen whichever task that
was.

**A thing that moved is the change families'.** A thing is *stale* at the place it had when a later recognisable
sighting of it lies at least `rules.STALE_M` (1.0 m) plus the kind's widening from that place (`World.stale`), and
*arrived* at a place it came to from at least that far (`World.arrived`: a recognisable sighting before it that far
away). `object_where` asks no thing that arrived at its last place, and no thing that is not *steady* there
(`World.steady`: every sighting of it after its last recognisable one, a glimpse or a dark or clipped frame included,
lies within `rules.MOVED_NO_M`, 0.15 m, of that place), which keeps out every stale thing and more. The second rule is
what keeps the gold the memory table's: the table reads the latest sighting of a track, so a thing that was recognisable
at one place and glimpsed at another after a gap would be held where the gold does not read it (on release-b's data
6% of the answerable things, 12% of the AMR's, moved so, up to 13 m; the first rule, stale and arrived alone, missed
them since a stale place needs a later *recognisable* sighting). A thing that went anywhere by a metre and was
recognised at both places is kept for the change families, [`object_was`](object_was.md), which asks it at its earlier or
its later place (no shipped job asks it in 0.11.0). (From 0.6.0 `object_was` judges a move by the last rung of the state
it asks, which for a near thing is under a metre; a thing that moved less than a metre stays `object_where`'s, and the
subject rule keeps a case from asking both about it.) What stays is a thing that did not go anywhere, so `object_where` is the static control beside the
temporal families: it scores where a thing was last seen. It is not a temporal family, and `time_needed` does not read it.

Both pools hold the same kinds, so no kind predicts the slice. A gold is what was last seen; nothing unobserved changes
it (truth rule 2).

## What the log tells: `World.told`

A thing the robot's own log tells about is never asked at the end, since the log, not memory of the world, answers it
(`World.told`, which `World.things` applies, so every end family leaves it out):

- every entity named by the third of a window task's `targets` (the helper's carried thing: the subgoal names it, where
  it came from and where it went);
- every thing whose footprint overlaps the zone of a decision of the window whose kind has a zone rule
  (`JobSpec.zones`), at that decision's time: the blockers. The zone is the one the coin sets, from the map and the
  delivered pose and heading (`zones.zone_of`: a piece of a kind, a doorway's way or the window of the surface faced).
  The robot's response after the coin ("skip", "abort", "elsewhere") and the subgoal that follows it say a blocker
  stood there. All the window's decisions count, asked or not. Since 0.6.0 no decision asks a family, so the key is the
  zone, not a family asked there.

A question during the job reads `World.told(upto=k)`: the carried things of the tasks that had begun by the look point
`k` and the blockers of the decisions at or before it. A thing only a later task carries, or a later decision's blocker,
may be named at `k`, as the log had not told it yet; an end question, asked after the window, reads all of it.

## Absent descriptions (decoys)

An unknown row names either a thing the world holds and no frame shows (offered first) or an **absent description**: a
kind, colour and size that matches nothing in this world, drawn from other seeds' worlds of the map by the generator's
own sampler (`never_seen`, keyed by `SEED_KEY`). The absent description is offered only when the world gives no clue
that it is absent:

- a thing of its kind is **sighted** in the window, so the kind's presence says nothing (a kind the world lacks would be
  answered by a kind-presence check); and
- no sighted thing of its own kind or of a kind it can be mistaken for (`descriptions.CONFUSABLE`: tote, bin, box and
  bag; van, box truck, pickup and car; bag and laundry basket; mug, drinking glass and bowl; bowl and plate; dumpster and
  skip; chair and stool; toy and box; toy and clothes pile; book, book stack and shelved book; folded clothes and towel;
  folded clothes, book and book stack; jacket, hanging coat, clothes pile and folded clothes, and, for a description
  nothing seen answers, `descriptions.CONFUSABLE_UNSEEN`: bin and drum; sack and pallet) could be taken for its colour
  (`World.may_be_taken_for`, below), so a red box is never asked beside a red tote or a red bag, nor a "white" laptop
  beside a "grey" one that renders white, nor a blue sack beside a sighted blue pallet.

The questionnaire asks **at most one absent description a case** (two `d:` subjects conflict), and the world's own
unseen things before it, so a case cannot earn the unknown share by counting decoys. The answer to a decoy is `unknown`:
there is no "no such thing" answer.

## Real things and look-alikes

The same screen holds for the real things (`questions_where.nameable`: `lookalike_sighted` and `askable`), and from
0.5.0-control it reads the colour the render shows, not the catalog's word (the 0.5.0 log review, defect 1: a remote
labelled black rendered silver, a "grey" laptop was as light as the white ones and "Where was the white laptop?" was
asked with it in plain view). The colour is `rgb_med`, the median sRGB over the thing's mask in a frame
([format.md](../format.md)), named by `colour_seen.classify` (the palette's twelve words, or the words on both sides of
a boundary it is within 3 CIELAB units of).

A real thing is asked in neither pool, answerable or unknown, when:
- **its picture does not show its word** (`World.colour_holds`): the colour most of its recognisable frames confidently
  show it in (`colour_seen.held`: the one word with the most frames, alone) is not its label; a frame near a boundary
  votes for none. A frame that is not recognisable does not vote; a thing with no recognisable frame has none to
  contradict its label; or
- **a look-alike could be taken for it**: another thing of a kind it can be mistaken for (its own kind included: a
  description is unique, but a "grey" and a "white" laptop are not) has a sighting in the window, at any keyframe of
  it, and any of its recognisable frames shows it in the thing's colour or next to a boundary of it, whatever its label
  (`World.may_be_taken_for`, `World.sighted_looks`; the pilot (0.5.0) asked a blue skip beside a blue dumpster, a red
  van beside a red pickup and a bag beside a box of its colour). **Black, grey and white are one group**
  (`colour_seen.could_be`): a person cannot reliably tell them apart on a thing of several materials under varied
  light, and a median over a mask mixes a body with its screen, lid or roof, so a look-alike that may be taken for any
  neutral hides a neutral thing, and the review's "Where was the white laptop?" is not asked beside a grey laptop in
  view. A chromatic colour is hidden only by a look-alike that renders its own word or sits next to a boundary of it.
  A thing no frame of the window shows is also hidden by the `CONFUSABLE_UNSEEN` kinds (a "yellow bin" never seen is not
  asked beside a sighted yellow drum, a "blue sack" never seen not beside a sighted blue pallet; the AMR's log review,
  defect 2); a seen drum, pallet, bin or sack is asked beside one of the other kind. The screen reads kinds, colours and
  whether a sighting exists, never a place, a time or a gold.

**The glimpse screen (S4, 0.11.0)** applies to the `unknown` pool alone, a real unseen thing and an absent description
both: an `unknown` candidate is dropped when a look-alike kind (`descriptions.lookalikes`, the unseen groups too) was
glimpsed in the window (`World.sighted` without any recognisable frame) in a frame whose `rgb_med` may be taken for the
asked colour (`questions_where.glimpse_hides`). The screens above read a glimpse-only thing by its label; its render can
differ, and a person asked for the blue skip could not tell it from a bluish glimpse. A name with no colour is never
screened. It costs 6% of the `unknown` rows on the dev root (6 of the 95 of `object_where` and `ego_mem`, 2 of them
`object_where`'s, both the helper's); S3 (any colour) would cost 48% and S1 (any glimpse in the colour) 57%, and both
leave the helper's `object_where` almost no `unknown` rows. The answerable pool is not screened: the thing it names was
seen plainly.

It is asked when no look-alike renders in its colour (a neutral: renders no neutral), when none is sighted, or when its
kind has no look-alike. A log from before 0.5.0-control (an older root's) has no render: the label stands, a neutral
label for all three neutrals. [`object_was`](object_was.md) and [`distance_between`](distance_between.md) apply it to
their things too.

## Critical, and the near rule

A row is critical against the job's relevance (`questions_where.critical`) when its kind is one of the task's relevance
kinds; its place, for an answerable row, lies in a region whose kind is among the task's `Relevance.regions`; where the
job sets `Relevance.near` and `Relevance.near_m`, the place lies within `near_m` of one of the points; and, where the job
sets `Relevance.floor` (the vacuum and the AMR, which work on the floor), the thing stood on the robot's floor at the
sighting (`World.on_floor`). An unknown row is critical by kind, since no place is known. The near rule is the drone's:
its camera sees about 12 m of ground from the 15 m inspection height, so a thing farther from the drop point than that is
a question about somewhere the robot could not have looked at (a job that sets none is unchanged).

For `object_where`, a window's relevance is joined: a thing is critical when **any** task of the window makes it so, and
**only critical things are answerable**; the questionnaire's selection order (critical first) then matters for the unknown
pool and the other families alone. The tag is on the gold row, and the report splits the answerable rows into critical
and situational (`by.relevance`). A pool offers its critical candidates first ([questions.md](../questions.md),
"Selection order").

## The credit (0.9.0)

The question asks for a place to go back to, so its need is what the job needs to find the thing again: its
**re-find radius R** (`JobSpec.reacquire_m`: the job's own `reacquire` when it declares one, else its `match_m`: 1.0 m
in a home, declared; 1.0 m in the warehouse and 2.0 m in the city, their `match_m`). The row is `graded` with
`need_m` R, absolute: it does not depend on how far away the thing was seen. The target stays the middle of the thing.

The middle of a thing is behind what a camera below its top saw. So the row's `resolution_m` is the larger of one pixel
at the thing's last recognisable sighting (`rules.resolution_m`) and `rules.hidden_widening` (the kind's half-length
beyond `rules.WIDEN_FREE_M`, 0.25 m) **only when the thing's top (its base height plus its height) was above the camera
at every recognisable sighting of it at the place** (`questions_where.place_widening`: the least `hidden_widening` over
them), and by nothing as soon as one view was from at or above the top (it saw the top face, so the middle was in view).
The sightings at the place are the thing's recognisable sightings in the window at which it stood within
`rules.MOVED_NO_M` of the gold place, so a thing that moved a little in the window is not measured at the place it
left. A drone's camera is above most tops, so most of its rows are not widened, and a car seen from nest level as well
as from the air is not. The row holds no room: the direction quarter is `e <= R` alone, which is R's own meaning (going
there, the robot finds the thing in view), and the room a thing is in is a fact of the map and the kind that a system
without the camera knows too (the 0.9.0 review: the room quarter was the whole of `path_log_in_pool`'s lead on the
helper's long cases, +0.126, and -0.062 without it, while the three models moved by 0.025 at most). The gold row also
carries `range_m` (the nearest slant, `World.slant`, over those sightings: the delivered
camera at the keyframe to the middle of the thing's box there, to 4 places, `questions_where.nearest_sighting_at`; kept
for the reports, it sets no credit), and `answer_class` is the slice (`answerable` or `unknown`: the two classes of the
draw's 20% unknown share). On the 0.11.0 release the median `rho / N` is 0.01 for the helper (q90 0.02) and 0.05 for
the vacuum (q90 0.41), and the family asks 73 questions on the helper (66 in the short and medium cases; 55
`answerable` and 18 `unknown`) and 24 on the vacuum (21; 20 and 4).

An unknown row has no range. It carries the same keys of R, unwidened, which no score reads: the gold is `unknown` and
the answer is judged right or wrong.

The job's numbers and their sources are in [questions.md](../questions.md), "Credit". The prompt states R and the four
quarters in words (`llm_render.credit_sentence`, built from `JobSpec`), so it cannot disagree with the scorer, and the
point a place names: the middle of the thing where it stood, its x and y in the map frame, not the nearest visible point
that an offset names (`llm_credit.PLACE_TARGET`; the gold is `World.xy_at`, the placement's base centre, so a reader told
the offset's convention alone would answer the nearest point of a 0.5 m basket and lose a quarter on every place).

## History

0.5.0 asked this family at each moment of the job's events in each task ("Where was the {thing} {event}
in your second task?"), of a thing whose last recognisable sighting lay in that moment's own task. The 0.5.0 log review
found its moments told nothing a "last seen" did not (nothing moves within a task, so the anchor decided no gold) and
that one unmoved thing was asked at several. 0.5.0-control asks each answerable thing once, in the words
above.

0.7.0 credits the row by the job's re-find radius R (R/2, R, 2R), absolute, in place of the relative
ladder of the nearest slant, and widens a rung only where no recognisable sighting at the place had the camera at or
above the thing's top. The questions asked and their golds are the same as in 0.6.0; only the rungs differ.
