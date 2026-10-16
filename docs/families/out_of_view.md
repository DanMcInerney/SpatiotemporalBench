# `out_of_view`

**Retired in 0.6.0 (an internal checkpoint, folded into 0.7.0 in the [CHANGELOG](../../CHANGELOG.md)). No job asks it in
0.11.0.** `JobSpec.decisions` maps no decision kind to a family, and `ego_mem` (a thing the robot saw before, placed
from its camera now) and `task_things` take its place. The module stays. What follows is the family as 0.5.0-control
built it.

"What, that I cannot see now, is within a few metres of me?" The places, as last seen, of the movable
things on the floor near the robot that no row of the images it was just sent shows. During, `memory`. Code:
`stbench.questions_view`. The record and questionnaire rules are in [questions.md](../questions.md).

| | |
| --- | --- |
| Category | C3 Locate |
| Asked | by no job. During, at a decision point of a kind whose `JobSpec.decisions` names it. In 0.5.0-control only `amr/1` asked it, at `before_undock` and `before_crossing` (`before_station` asked `zone_clear`: [job-amr.md](../job-amr.md)). In 0.5.0 it asked it, at `before_station`, `before_undock` and `before_crossing`, and asked nothing else during the job. In 0.4.x it passed the AMR's release gate check (31 questions, separation +1.00). `helper/1` mapped it at `before_doorway`, `before_grasp` and `before_place` and dropped it at the release gate check, where its learned shortcuts (`blind`, `path_log`) scored -0.114 under abstaining, BROKEN lines, after the family's one repair round (the doors). `vacuum/2` dropped it at M2's fast gate check (its golds were empty after the declared fallback: 93% at 0.1 m), and `drone/1` never asked it: it reads the robot band's free space and the floor at the robot's base, which is in the air |
| Text (slots) | "Where, as you last saw them, are the movable things taller than {height} m standing on the floor within {radius} m of you that are not in any of the images you were just sent?" (`height`: 0.1 m as built, `questions_view.HEIGHT`; `radius`: `questions_words.cap_of`, 5 m or a metre inside the depth range) |
| Answer, credit | `positions` (`{"items": [{"xy"}], "dont_care": [{"xy", "radius"}]}`), `f1` with `tolerances {"match_m": m}`, m = `rules.MATCH_M` (0.5 m) plus the largest widening of an item (0.5 m for none); the gold is never `unknown`: nothing seen near is the empty list |
| Scope, `needs` | `memory`; see, pose. The gold reads depth (the "seen gone since" rule below), which the camera does not need: `needs` is what a system needs to answer, so the default `rgb_pose` channel keeps the question |
| Gold function | `questions_view.out_of_view(w, k)`: a `Cand`, never None; `critical` true; evidence the keyframes of the items' last recognisable sightings and the decision's, `entities` empty, subjects the time (`w:<k>`) and each item (`e:<id>`) |
| Slice | none (a during family draws no slice) |

## What changed in 0.5.0-control

Two changes answer the second defect found in a review of the 0.5.0 run's logs, and one follows the camera-only
channel.
- **The racks' floor-level stock is a don't-care place.** Bays 1 and 3 of every `warehouse_a` rack hold pallets of
  cartons on the floor. They are scenery, not entities, and they fit the words ("movable
  things ... standing on the floor"), so a reader that names them read the picture right, yet the gold listed nothing
  there: about 13 of the 31 questions of the 0.5.0 run were affected, and a literal reader averaged +0.67 against +1.00. Each stock pallet is now a don't-care place (rule 6 below), computed from the map by `stbench.rack_stock`.
  With matching first (0.5.0-control), an answer that names one is neither a hit nor a miss.
- **Told things are don't-care.** A thing the robot's own log tells about (`World.told()`: the helper's carried thing, a
  blocker standing in a zone the robot was asked about) is never asked at the end, so it is no item here either, and its
  places are don't-care (rule 5).
- **`needs` drops depth.** The question needs the camera and the pose; the gold may read more.

## What changed in 0.5.0

Three changes answer the findings of a review of the run's logs on this family. The words now say what "now" means: the
images the harness just sent (the decision's batch then, now the look window), where 0.4.x said "cannot see now" and a
reader could take it as the last frame alone. The items are recognisable: an item is a thing with a *recognisable* last
sighting (a frame with enough pixels and light, a box clear of the image's edge and a depth:
[`object_where`](object_where.md)'s rule), so a sliver or a speck is not an item an agent could be asked to name. And
the match is by footprint: a large thing's surface is seen where its centre is not, so an item's matching radius and
every don't-care radius carry the thing's own widening instead of a flat 0.5 m, and an agent that names exactly the
right pallet is not charged for the metre between its surface and its centre.

## The gold

`b0` is the first keyframe of the decision's look window (`World.window_ks(k)`), `here` the delivered base at `k`, `height` and
`radius` the slots. A *thing* is an entity of mobility `light`, `carried` or `clutter` (`MOVABLE`: fixed furniture is
not one, and the layout is public), a *mover* one of mobility `mover` (a person). Everything is read from the window's
own sightings (`World.sight`), the placements' xy and the delivered poses. A place's radius is `MATCH_M` plus the
widening of the entity that stood there (`rules.widening` of its own box), the radius written on a don't-care place and
the one an item matches within.

1. **In view now.** A thing or mover with any visibility row in the look window, at any pixel count, is not an item. Its place
   at `k` is a don't-care place (a system may detect it there), and so are the places a memory would hold it at from
   before the look window (rule 2's; all of them for a twin, rule 3): a thing moved in a gap and only glimpsed in the look window,
   which a table that keeps its old track still lists at the old place, leaves no false positive.
2. **Movers** are never items. Each one's last recognisable place before `b0`, and every sighting after it (recognisable
   or not; every sighting before `b0` when none is recognisable), is don't-care: a memory that tracks a person as a
   thing would report it there.
3. **Twins.** A thing or mover with a twin is never an item (methodology rule 2: identity between identical things is not
   observable, and a table tells its tracks apart by description alone). A twin is another entity of the same kind and
   describable attributes (`descriptions.desc_key`) that some frame **delivered by the end of the look window** shows (a
   sighting at or above the kind's glimpse bound at a keyframe before the window ends); an entity no frame shows is none
   (truth rule 2), and one first shown after the window is none yet, since the ask cannot have been sent that frame.
   Every place either was sighted at before `b0` (recognisable or not) is don't-care, so a table that keeps the old
   track of a twin moved in a gap and re-seen elsewhere, or one track a place, lists nothing a gold calls false.
4. **Every other thing** with a sighting before `b0`: `s` is its last **recognisable** keyframe before `b0`.
   - With none, or with any later sighting before `b0` (recognisable or not) `MOVED_NO_M` (0.15 m) or more from its
     place at `s` (`World.steady`): its places from `s` on (every sighting before `b0` when none is recognisable) are
     don't-care, and it is no item, since a memory that reads its latest detection holds it there.
   - Else `P` is its xy at `s`, kept only when it stood on the floor there (`World.on_floor`: a thing on a table is
     nothing). `d` is the distance from `here` to `P`, `h` its height (`size[2]`):
     - `d` over `radius + EDGE_M` (0.5 m): nothing;
     - `d` within `EDGE_M` of `radius`, or `h` within `BAND_M` (0.05 m) of `height`: don't-care at `P`;
     - **seen gone since**: the keyframes after `s` and before `b0` certify `P` empty (`grid.place_free` over the cells
       the window observed in them, radius half the thing's shorter side): the robot saw it was no longer there, so `P`
       is don't-care and no item (a table cannot be asked to keep a track it saw gone);
     - else `h` above the band: an **item** at `P`; a lower thing: nothing.
5. **Told things.** A thing the robot's log tells about (`World.told()`, see [`object_where`](object_where.md)) is never
   an item. Every place it was sighted at before `b0` is don't-care, as for a twin (rule 3), and its place at `k` too when
   a row of the look window shows it.
6. **Rack stock.** `rack_stock.floor_stock(map)` lists the pallets of stock that stand on the floor of the map's pallet
   racks. A rack row is `BAY_M` (2.7 m) bays by `LEVEL_M` (1.3 m) levels; the map gives a surface to the bay-levels items
   may rest on, and the renderer draws stock in exactly the others (`render/rack_layout.py`; `tests/test_rack_stock.py`
   holds the two to the same bays). For each bay with no surface at level 0, the two pallets stand at its middle plus
   and minus `BAY_M`/4 along the rack, and each is a don't-care place of radius `MATCH_M` plus the widening of a pallet
   as drawn (1.0 x 1.16 m: 0.33 m, so 0.83 m). Stock above the floor is not asked about: the words say "standing on the
   floor". These places are scenery, so they do not depend on what the look window shows, and a map without racks has none.
7. **The swallow.** An item within a don't-care place's radius becomes a don't-care place too, repeated until none
   moves: the scorer before 0.5.0-control dropped answer points inside a don't-care region before matching, so the right
   answer would be dropped there. The scorer now matches first and drops only the answers that matched nothing, so an
   answer inside a region still counts for the gold item it matches; the rule stays so the gold is as it was.

Places are the placements' xy (a stock pallet's, the map's) to 4 decimals, sorted, those within 0.01 m merged (the larger
radius kept). A don't-care place beyond `radius + EDGE_M` of `here` is not written (no item or answer can touch it), a
stock place included. The row's `match_m` is the
largest item radius, or `MATCH_M` for no item.

**What it does not read.** Never where a thing is now: an item is where it stood at its last recognisable sighting, and
a place the robot did not look at again keeps its item whatever has happened there (the swap check and the hidden-change
test pin it). The seen-gone rule reads only the window's free-space cells, the very cells a system's mapper computes
from the same depth.

## Empty golds

An empty gold is honest ("nothing seen near is the empty list" is a fair answer to "what was seen"), and it stays. What
it costs the gates is declared before data (methodology rule 5): a constant empty answer scores `2E - 1` at an empty share
`E` (F1 is 1 when both sides are empty), so `current_view`, which answers the empty list, fails `memory_needed` once `E`
passes 0.525 (the margin is 0.05). The check report gives the share (`groups[].families.out_of_view.empty_set_share`),
and 0.5.0 reports it against that bound. M2 has no pre-gate:

- **The free bound**, before the dev render: the share of `before_doorway` decisions with at least one `MOVABLE` thing
  on the floor taller than `HEIGHT_M + BAND_M` within `radius - EDGE_M` of the base, by true places (an upper bound on
  the non-empty share); under 48% the slot `height` is `FALLBACK_HEIGHT_M` from the start. It did not bind on the
  vacuum's scripts (a tall light piece or clutter stands within 2.5 m of every doorway decision), but it says nothing of
  what the robot saw; at M2's fast gate check on look-dev seeds 141-145 every gold of 15 was empty at 0.3 m, and the
  slot's height is the fallback, `questions_view.HEIGHT` = 0.1 m.
- **At the dev gate check:** over 52% empty golds in a group: the fallback if not taken, else the family is dropped.

## For the answerers

A table system lists the tracks it last saw before the look window that the gold lists, from its own sightings: on the floor
then, not seen gone since (the same `place_free` over the cells of the steps after the sighting), `size[2]` over
`height`, within `radius` of the ask's base, and of no kind that is a fixed piece of the public map (its furniture that
is not light, and its doors: the timeline's doors are fixed entities, which the gold never lists). The gold's don't-care
places (things in view now, at their place now and at the places before the look window a track may hold; people and their
glimpses; every place of a twin; glimpses after a sighting; the height and radius bands; places seen empty since; and
what the swallow adds) absorb what a track cannot tell apart, among them the ghosts a table can keep: a thing moved in a
gap and only glimpsed in the look window (never detected there), whose old track stays at the old place, and identical things
(a moved twin re-seen more than 0.5 m on starts a new track and leaves the old one). A thing with a twin is therefore
never an item, whatever its tracks say. The stock pallets of the racks are in no table, and a reader or a perceiver that
names one is not charged (rule 6). `current_view` answers the empty list.

**Under one-frame delivery** (0.10.0) the table answerer of this family sees a look window of one frame, so it no longer
reproduces the gold, and the question's words ("the images you were just sent") describe the look window's frames, not a
message. The family is retired and no job asks it, so no score is affected; a job that asks it again first reworks it for
one-frame delivery.
