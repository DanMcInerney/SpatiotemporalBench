# `drop_zone_people`

"Before you lower the package: where are the people within 4 m of the point below you? Answer their positions in map
coordinates, or an empty list if there are none." The drone's check of the ground it is about to lower a package onto,
**asked by no job** (the drone asked it in the 0.10.0 draft, until the final review's S2; below). During, scope `now`.
Code: `stbench.questions_hazard` (the gold, with [`tangle_ahead`](tangle_ahead.md)), `stbench.hazard_rule` (the zone,
shared with the table systems), `stbench.approach_rule` (the hull centroid), `stbench.person_seen` (the person evidence
rule, as [`closest_person`](closest_person.md)), `stbench.table_hazard` and `stbench.baselines_path_hazard` (the
answerers). The record and questionnaire rules are in [questions.md](../questions.md). New in 0.10.0. Its kind is
**harm**, its ability clearance.

A delivery drone that lowers on a tether over a sidewalk or a plaza must not lower onto a person: the operators' own rules keep
a clear area around the drop point (the job's clearance C is 2.4 m, the smaller of the distances in Wing's patent US12399508B2).
The decision is "is anyone there?" and the answer a controller acts on is who, and where. The drone's other look question,
[`clear_path`](clear_path.md), asks about walls; nothing asked of it before this one is about the people below it.

| | |
| --- | --- |
| Category | C5 What matters here |
| Asked | by no job. The drone asked it during the job, at a look point, in its slot 3 whole, until the final review's S2 (repair R1) took it off: 2 rows over the 36 dev cases, under `abilities.MIN_ABILITY_N`, in the drone's counted clearance `now` cell with no per-family diagnostic and no readable gate. A job that wires it (`JobSpec.looks`, a `look_shares` slot and `look_phases["drop_zone_people"]`, which the drone declared as `inspect`, `descend` and `lower`: over the drop point and lowering; a job that asks it without the phases raises) asks it in those phases, and only when the disc of 4 m around the camera's floor point on the ground lies wholly inside the ask frame's view (below). Elsewhere a dead band |
| Text (slots) | "Before you lower the package: where are the people within 4 m of the point below you? Answer their positions in map coordinates, or an empty list if there are none." No slots |
| Answer, credit | `positions`, `graded_set`: the answered places are matched one to one to the gold's by maximum summed credit within four match radii (`tolerances.match_m` = `JobSpec.match_m`, 2.0 m in the city), each pair earning the four quarters on its own item's resolution (`resolutions_m`: one pixel at the person's hull centroid from the ask frame's camera) and its direction quarter within the match radius alone (a list item is a place, 0.11.0), a missed person costing one and a false place (outside every don't-care circle) `fp_weight` 0.5 of one; an empty gold's one thing is the empty list, `s = 1 / (1 + 0.5 |false|)` (0.11.0). The catastrophe tolerance (below) is on every row |
| Scope, `needs` | `now`; see, pose |
| Gold function | `questions_hazard.drop_zone_people(w, k) -> Moment \| None`, over `zone_candidate(w, k) -> Cand \| str` (the string names the dead band). One candidate a point; key `str(k)` |
| Class | `items` or `empty` (`records.BALANCE_POOLS`); reference None, `mapper_gold` False |

## The gold

- **The people** are those with an evidence sighting in the ask's frame `k` (`person_seen.hull`) whose floor hull has a
  point within `ZONE_M` = 4 m of the camera's floor point. Each item is the hull's floor centroid (`approach_rule.centroid`),
  the place the credit measures from. People are never named, so the items are anonymous.
- **The view.** The disc of radius 4 m around the camera's floor point, on the plane z = 0, must lie wholly inside the frame:
  its middle and 72 points of its boundary (every 5 degrees) project, through the delivered camera pose and the camera's
  focal lengths, in front of the camera and inside the 640 x 480 image. At the drone's lowering height of about 7 m the nadir
  view covers +-5.9 m by +-4.4 m, so a 4 m disc fits inside the short half-axis and a 5 m disc did not (the review measured
  4 of 20 lowering frames with a person within 5 m and no pixel of them at all). It is the same test of every row, empty
  or not: an empty list is a reading of what the frame shows. On the dev root every frame in the three phases passes it.
- **A person with pixels and no evidence sighting** (a glimpse, a body cut by the frame, a missing depth) is a **don't-care
  circle** (`answers.Positions.dont_care`): the truth footprint's circumscribed radius plus the match radius, about the
  footprint's centre, when the footprint lies within `ZONE_M` plus the match radius (6 m) of the camera's floor point. The robot
  saw them and the rule cannot place them to the credit's precision (at the lowering height the median person within 5 m has
  308 px, under the 1,024 px floor), so a list that names them is not charged and one that leaves them out is not either.
  An empty gold may carry circles.
- **The empty list** is the gold when no evidence person is in the zone; the draw below keeps a share of them.

## Credit and catastrophe

`tolerances["catastrophe"]` is `{"weight": 3, "rule": "miss", "margin_m": max(C, 2 rho), "critical": [[x, y], ...]}` on every row
(`questions_harm.list_row`): a person whose hull lies within C = 2.4 m of the camera's floor point is critical, and an answer
that misses one is the error that lowers a package onto a person (`credit_kernel.CATASTROPHE_PERSON`). The margin takes the
largest resolution of the gold's items. A row with no critical person has an empty list there.

## Constants

| Name | Value | Source |
| --- | --- | --- |
| `ZONE_M` | 4 m | At the lowering height of about 7 m the nadir view covers +-5.9 m x +-4.4 m (fx = fy = 381, 640 x 480): a 4 m disc fits inside the short half-axis (review B2) |
| `look_phases` | `inspect`, `descend`, `lower` | The job's declaration (`JobSpec.look_phases`, as `clear_phases` is `clear_path`'s): the drone's status phases over the drop point, before and while it lowers. A job that asks the family without declaring them raises, and no phase is named in the compiler |
| `GROUND_Z` | 0 | The plane the disc lies on; a person stands within 0.15 m of it (the plaza), under a pixel |
| `RING_POINTS` | 72 | A boundary point every 5 degrees |
| `EMPTY_KEEP_PERCENT` | 1 | The balance draw: the floor of 100 (1 - p) / (2 p) for the share p = 0.967 of empties (keep 1.7), so the empty class is at most a third of the rows; `stable_hash(case id, "drop_zone_people", point, "empty") % 100` under it |
| weight 3 | | `CATASTROPHE_PERSON` |

**The subject** is the recent view (`questions_harm.stretch`): `w:<first>-<k>`, so the same people are not asked about at
points four frames apart.

## Supply, baselines and difficulty

- **Wired (0.10.0 draft), then unwired by the final review's S2.** The
  family had the drone's third slot to itself, whole, because its fill is too low to share one. Of its 593 draws over the 36 drone cases, 413 fell outside the
  three phases and 174 more found nobody within 4 m (1% of those are kept); **2 questions** were asked, both in short
  cases (a fill of 0.003 a drawn slot). That is under the 20 rows a counted cell needs, so the family is a diagnostic by
  supply. The counted-cell rule has no per-family flag, so its 2 rows sit inside the drone's clearance `now` cell (98
  questions) at about 2%. The final review took it off the drone for that reason; the code, the gold and this page stay for a
  dataset with the supply, and the drone's other slots are unchanged.
- **Supply if every frame were asked** (dev root `release-060`: 36 drone cases, 3,132 frames at least 4 into their task
  and not its last). 2,107 are out of the three phases (67%), the other 1,025 pass the view test and are
  answerable: 991 `empty` (247 with a don't-care circle) and **34 `items`**, each with one person, in 16 cases of 10 scripts (26
  in `inspect`, 5 in `descend`, 3 in `lower`; 7 with a critical person). A person within 4 m is an evidence sighting in 3% of the
  frames, and in a further 24% a person with pixels and no evidence sighting is a don't-care circle. At most **19** `items` rows lie 8 frames apart (the subject rule). After the draw
  (1%): 46 rows (1.5% of the frames tried), 34 `items` and 12 `empty`. That is under the 20 rows a counted cell needs
  ([eval.md](../eval.md), "Diagnostics"), so the family is a diagnostic by rule until a build supplies more.
- **Answerers.** `table_hazard.ANSWERERS`: the floor centroids of the hulls, in the ask's frame, of every person track whose hull has
  a point within 4 m of the ask's camera floor point. The oracle and `memory_table` read the same hulls the gold takes and
  reproduce it: on every fourth answerable frame of the dev root (241 rows, 8 of them `items`) both score 1.0000 on every row. The view test is not part of the answer: it chooses which
  points are asked. `current_view` holds the latest frame, which is all a `now` gold reads, so it reproduces the gold;
  `path_log` (`baselines_path_hazard`) reads no picture and pedestrians walk where
  they like, so it answers the pool of other scripts' golds where that beats abstaining, else `unknown`.
- **Difficulty.** The people below a drone at 7 m are a few hundred pixels each, seen from above, so the agent must find them,
  decide which are within 4 m of the point under it, and place each in the map; and answer the empty list on the many frames
  where nobody is there. Results: none; no job asks the family.
