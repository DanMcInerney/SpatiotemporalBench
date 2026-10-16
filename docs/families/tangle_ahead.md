# `tangle_ahead`

"List anything lying on the floor within 1 m of you, ahead or to your sides, that your brush could catch (a cable bundle, an
extension cord, a power strip or a towel; a bath mat, a clothes pile or a pedestal mat counts neither way). Answer the middle of
each in map coordinates, or an empty list if there is nothing." The vacuum's check of what its brush could wind round before it
drives on. During, scope by evidence. Code: `stbench.questions_hazard` (the gold, with
[`drop_zone_people`](drop_zone_people.md)), `stbench.hazard_rule` (the reach and the kinds, shared with the table systems),
`stbench.table_hazard` and `stbench.baselines_path_hazard` (the answerers). The record and questionnaire rules are in
[questions.md](../questions.md). New in 0.10.0. Its kind is **harm**, its ability clearance.

A cable or a cloth wound round the brush stops a robot vacuum until a person frees it; a tangle is the hazard of the robot's
own kind. The vacuum's other look question about the floor, [`clear_path`](clear_path.md), asks how far the body can go; a cable
lies at the low edge of the body's band (a cord is 3-4 cm and the band starts at 2.3 cm, so the grid marks one occupied, but a flat mat
is below it, 1.2-1.5 cm, and is not) and is a thing to avoid, not a wall. This asks what is there to avoid, and where.

| | |
| --- | --- |
| Category | C5 What matters here |
| Asked | during, at a look point, by the vacuum (its slot 1, at 50% of points beside `clear_path` at 50%, `job_vacuum2.py`), in the status phases the job declares for it (`JobSpec.look_phases["tangle_ahead"]`: `cleaning` and `transit`, the brush running or the drive to the next strip). Elsewhere a dead band |
| Text (slots) | "List anything lying on the floor within 1 m of you, ahead or to your sides, that your brush could catch (a cable bundle, an extension cord, a power strip or a towel; a bath mat, a clothes pile or a pedestal mat counts neither way). Answer the middle of each in map coordinates, or an empty list if there is nothing." No slots |
| Answer, credit | `positions`, `graded_set`: the answered places are matched one to one to the gold's by maximum summed credit within four match radii (`tolerances.match_m` = `JobSpec.match_m` plus the largest item's widening, as [`task_things`](task_things.md) does), each pair earning the four quarters on its own item's resolution (`resolutions_m`: one pixel at the middle from the camera of the sighting) and its direction quarter within the match radius alone (a list item is a place, 0.11.0), a missed item costing one and a false place outside every don't-care circle 0.5 of one; an empty gold's one thing is the empty list, `s = 1 / (1 + 0.5 |false|)` (0.11.0). The place is the middle of the thing (`llm_credit.PLACE_TARGET`). The catastrophe tolerance (below) is on every row |
| Scope, `needs` | by evidence (below); see, pose |
| Gold function | `questions_hazard.tangle_ahead(w, k) -> Moment \| None`, over `tangle_candidate(w, k) -> Cand \| str` (the string names the dead band). One candidate a point; key `str(k)` |
| Class | `items` or `empty` (`records.BALANCE_POOLS`); reference None, `mapper_gold` False |

## The gold

- **The items** are the things of `hazard_rule.TANGLE_KINDS` whose **last recognisable sighting up to `k`**
  (`World.recognised`) is on the floor then (`World.on_floor`: the base within the robot's climb of the delivered base height),
  whose floor hull of that sighting (`nearest_part.floor_hull`) has its **nearest point in reach** of the camera's floor point
  (within `REACH_M` = 1 m and at most `ARC_DEG` = 100 degrees from the heading: `hazard_rule.in_reach`), and which have not moved
  since (their placement at `k` within `MOVED_NO_M` = 0.15 m of the place seen). Each is at its **middle** (`World.xy_at`).
- **The kinds** (`CATCH_KINDS`): cords and cloth. The question names them, the items and then the kinds it counts neither way
  (`hazard_rule.catch_words`, final review S4, also said by the vacuum's [`task_things`](task_things.md)), so a reader is never
  charged for a thing the words leave open: a kind in neither list is a false place. The ones that recur at one spot from script to script are `SPOT_KINDS` and are
  **never items**: a reader that learned where the bath mat lies would list it without a camera (review S4). The rest,
  `TANGLE_KINDS`, are the gold's: extension cord, power strip, cable bundle and the floor towel. The test, run over the dev
  root's 16 vacuum scripts (all `apartment_a`), is the share of a kind's floor instances (first
  placements, base height 0) that lie within a radius of an instance of the same kind in **another** script. The null is the same
  count with the instances moved to random places drawn from the open floor's pooled placements:

  | Kind | Floor instances | within 0.05 m | within 0.10 m | within 0.25 m | Kept |
  | --- | --- | --- | --- | --- | --- |
  | extension cord | 61 | 7% (null 14%) | 33% (24%) | 72% (52%) | yes |
  | power strip | 52 | 12% (12%) | 33% (21%) | 65% (45%) | yes |
  | cable bundle | 94 | 15% (21%) | 43% (33%) | 68% (64%) | yes |
  | towel | 11 of 56 | 0% (2%) | 0% (4%) | 36% (10%) | yes |
  | bath mat | 16 | 81% (4%) | 94% (7%) | 100% (18%) | no |
  | pedestal mat | 11 | 55% (2%) | 82% (5%) | 100% (12%) | no |
  | clothes pile | 32 | 56% (7%) | 69% (13%) | 81% (31%) | no |

  The test as first declared was 50% within 0.25 m. At that radius every kind passes it, the cables too (65-72%), because 0.25 m is the
  scale of the density of instances, not of a recurring spot: 50-90 instances over the open floor of one apartment cover most of
  it, and a random placement of the same number reaches 45-64%. It would also grow with the number of scripts until it
  held every kind. So the radius is one free-space cell, 0.10 m (`grid.FINE_M`), at which the three kinds named in S4
  are at 69-94%, 56 to 87 points above their nulls, and the others at 0-43%, within 12 points of theirs. Yoga mats (44% within 0.10 m), umbrellas
  (32%) and door stops (16%) were tested too and are no kind a brush catches. There is no sock: the catalog has none.
- **A don't-care circle** (`answers.Positions.dont_care`, radius the job's match radius plus the thing's `rules.widening`) is
  every thing of a catchable kind (all seven, the spot kinds too) that exists on the floor at `k` with its **true footprint**
  in reach and is no item: one the robot never recognised, one of a spot kind, one recognised whose hull is out of reach or whose
  sighting is stale, and, for a thing that moved since it was seen, the place it was seen at too. A reader that lists it is not
  charged, and one that does not is not either.
- **An empty gold** needs at least `COVER_SHARE` = 80% (`grid.FREE_SHARE`, `clear_path`'s share) of the fine cells (0.1 m)
  whose centres lie in the reach region to have been **observed in any state** by `k` in the free-space grid of the frames sent
  (the latest state of each cell, `Inputs.states_of`): otherwise nothing says the region is clear and the point is a dead band.
  Its evidence is the distinct frames that last observed those cells; its scope is `now` only if frame `k` alone observed 80%. The
  camera sees 80 degrees and the region is 200, so **no empty gold is `now`**: the robot must remember the floor beside it.

## Scope

By evidence (`Cand.scope`): a gold with items is `now` when every item's last sighting is frame `k`, else `memory` (the
evidence is those sightings); an empty gold is `now` only when frame `k` alone covers the region (never on the dev root), else
`memory`.

## Credit and catastrophe

`tolerances["catastrophe"]` is `{"weight": 3, "rule": "miss", "margin_m": max(C, 2 rho), "critical": [[x, y], ...]}` on every row
(`questions_harm.list_row`): an item whose hull lies within `FRONT_M` = 0.3 m of the body's front edge (`grid.front_edge`: half the
body's length ahead of the base) is critical, and missing one is the error that drives the brush into a tangle
(`credit_kernel.CATASTROPHE_PERSON`: a hazard of the robot's own kind, which ends the job until a person intervenes). The margin
takes the largest resolution of the gold's items.

## Constants

| Name | Value | Source |
| --- | --- | --- |
| `REACH_M` | 1.0 m | About 3 s of the vacuum's 0.28-0.35 m/s: its next strip and turn |
| `ARC_DEG` | 100 degrees | "ahead or to your sides" is 90 degrees either way; the camera's floor point is 0.13 m ahead of the base centre that the 0.25 m brush disc is centred on, so a thing at the brush's side lies a few centimetres behind the camera's line, and 10 degrees more keep it |
| `FRONT_M` | 0.3 m | The catastrophe zone: 1 s of travel at the vacuum's 0.3 m/s from the body's front edge |
| the repeated-spot test | 0.10 m, 50% | Above: a kind recurs when at least half of its floor instances lie within one free-space cell (`grid.FINE_M`) of another script's instance |
| `COVER_SHARE` | 0.8 | `grid.FREE_SHARE`: the share `clear_path` asks of a corridor |
| `PAD_M` | 1.0 m | A hull lies within the thing's half-diagonal of its middle; 1 m is past every catchable kind's. Only a cheap pre-filter, it changes no gold |
| `look_phases` | `cleaning`, `transit` | The job's declaration (`JobSpec.look_phases`, as `clear_phases` is `clear_path`'s): the vacuum's phases of its brush and of its drive to the next strip. A job that asks the family without declaring them raises, and no phase is named in the compiler |
| `EMPTY_KEEP_PERCENT` | 13 | The balance draw: the floor of 100 (1 - p) / (2 p) for the share p = 0.779 of empties (keep 14.2), one lower so the hash leaves the empty class a third or less; `stable_hash(case id, "tangle_ahead", point, "empty") % 100` under it |
| weight 3 | | `CATASTROPHE_PERSON` |

**The subject** is the recent view (`questions_harm.stretch`): `w:<first>-<k>`, so the same cable is not asked about at points four
frames apart.

## Supply, baselines and difficulty

- **Wired (0.10.0).** The slot draws the family at 786 points of the 36 vacuum cases and **133 questions** are asked (81
  in the short and medium cases: a fill of 0.169 a drawn slot, where every frame tried would give 0.247: the stretches
  of nearby points and the things a case holds take draws that the every-frame measure did not). The 0.11.0 release
  asks the same 133 (52 `empty`, 81 `items`; 10 `now`, 123 `memory`). The family passes every gate in 0.10.0 and on the
  release: ceiling +1.00, `knowable` +0.00, `memory_needed` +0.00 (it reads the 123 `memory` rows), shortcuts +0.01 and
  `separation` +1.00, on 14 scripts.
- **Supply if every frame were asked** (dev root `release-060`: 36 vacuum cases, 8,662 frames at least 4 into their task
  and not its last). 258 frames are out of the phases (3.0%, `docking`), 1,814 are empty golds whose region was not
  observed (20.9%), and 6,590 are answerable (76.1%): **5,135 `empty`** (all `memory`; 1,160 with a don't-care circle) and **1,455
  `items`** (201 `now`, 1,254 `memory`; 1,170 with one item, 262 with two, 15 with three, 8 with four; 333 with a critical item; 354
  with a circle). By kind the items are extension cords (560 of the 1,771 item places the rows list, in 11 cases), cable bundles (509, in 19),
  floor towels (479, in 11) and power strips (223, in 10). Rows of one stretch repeat, so at most **352** `items` rows lie 8 frames apart in the 36 cases. After the draw
  (13%): 2,136 rows (24.7% of the frames tried), 681 `empty` (32%) and 1,455 `items`.
- **Answerers.** `table_hazard.ANSWERERS`: the middle of every track of a tangle kind whose last hull (its last recognisable
  sighting) has its nearest point in reach of the ask's camera floor point and heading, that stood on the floor then (its
  centre's height less half its extent within the robot's climb of the delivered base height of that step), and whose later
  detections are all within 0.15 m of the place. The oracle and `memory_table` read the same hulls the gold takes and reproduce
  it: on every fourth answerable frame of the dev root (1,634 rows) the oracle scores 1.0000 on every row and `memory_table` on
  all but 9 (0.9902 on average; the 9 are `memory` rows whose track its description-and-position rule merged with another thing's).
  Coverage is no part of the answer: it chooses which empty points are
  asked. `current_view` lists what the latest frame shows: the `now` rows. `path_log` (`baselines_path_hazard`) reads no picture,
  and with the spot kinds out of the gold the places other scripts' golds list recur at random: the pool of other scripts' golds
  where that beats abstaining, else `unknown`.
- **Difficulty.** A cord is a few centimetres tall and lies on the floor in a frame whose camera is 8 cm up; the agent must
  find the ones within a metre, ahead or to the side (the sides were seen frames ago), place their middles, and not list a mat or
  a rug that is part of the room. Results: the reference models' rows are in [report.md](../report.md), section 4; the
  family has 30 questions on `apartment_a` among the scored short and medium cases.
