# `clear_path`

How far the robot's body could move before it touched anything, in a direction it can move: ahead, back, or (ground)
turned in place to its left or right and then driven. During, scope by evidence. Code: `stbench.questions_clear` (the
spec, the words and the dispatch), `stbench.clear_ground` and `stbench.clear_fly` (the two golds),
`stbench.corridor_screens` (the ground screens), `stbench.wall_seen` (the flier's wall record), `stbench.grid` (the one
corridor rule the gold and every system's mapper call), `stbench.table_clear` and `stbench.baselines_path_clear` (the
answerers). The record and questionnaire rules are in [questions.md](../questions.md). New in 0.8.0, asked in every
direction the robot can move since 0.9.0. It is the controller's own question, "can I move without touching anything?",
asked at look points; it supersedes [`path_ahead`](path_ahead.md) and [`rear_clear`](rear_clear.md), which were asked at
decision points and which no job asks. Its kind is **harm**: what an action now would hit. It is a **structure family**:
part of its answer is the place's fixed structure, which the camera sees; the map reader that answers it from the public
map is an internal design diagnostic of `stbench check`, not a published reference (below). Every gate and
the evidence verdict pass on every job since the final review's repair; the helper's `memory` cells failed the evidence
verdict until then, and the cause was the draw and not the place (below, "The helper's `memory` cells").
[`open_heading`](open_heading.md) is its sibling, asked over the same cells.

| | |
| --- | --- |
| Category | C1 Room to move |
| Asked | during, at a look point drawn from the job's slot table (`JobSpec.look_shares`): the drone's first slot at 60%, the helper's first slot at 50% beside `closest_approach`, the vacuum's first slot at 50% beside `tangle_ahead`. The AMR does not ask it (below). The direction is drawn from the job's declared `JobSpec.clear_directions`: the helper's ahead, back, left and right, the vacuum's ahead, back and left, the drone's ahead and back. Only where the job's `clear_phases` hold the point's status phase, and for a flier also where the base is at least `fly_floor_m` up |
| Text (slots) | ground, ahead: "If you drove straight ahead now, how far could your body go before it touched anything? Answer from what you have seen, in metres from the front of your body; answer {cap} if nothing is within {cap} m." Back: "If you reversed straight back now, ... in metres from the back of your body; ..." Left or right: "If you turned in place to face your {left \| right} and then drove straight, how far could your body go before it touched anything? Answer from what you have seen, in metres from the front of your body; ...". Flier: "If you flew straight {ahead \| back} now at this height, how far could your body go before it touched anything you have seen? Answer in metres from the edge of your body." (`mode`, `direction`, `cap`) |
| Answer, credit | `number` or `unknown`; `graded` on the error against the job's clearance C, or a tenth of the gold where that is more (0.11.0): the four quarters ([questions.md](../questions.md), "Credit"). The harmful side is an answer longer than the gold (`harm` `over`: the need is halved), the direction quarter is the same side of `T = L + C` (the body's length plus C) and, where the gold is at least `T`, within a factor of two. The row's `resolution_m` is the largest of the grid's 2 cm sub-cell, the stopper's floor contact error and the tie gap; the flier's is one pixel at the wall's foot. The catastrophe tolerance (reported score only) is `over` with `threshold_m` = `T`: a corridor said longer than it is counts only where the true one is under `T` and the answer is not (final review S5) |
| Scope, `needs` | by evidence. Ground: `now` when the corridor over frame k's own cells (the latest frame) gives the same number as the window's within the row's resolution, else `memory` (an `unknown` gold is `memory`). Flier: `now` exactly when the latest frame saw the wall the body meets, else `memory`. See, pose |
| Gold function | the `clear_path` entry of `questions_clear.GOLDS`, called with the world and the point: `clear_fly.moment` for a robot that flies, else `clear_ground.moment`; a `Moment` or None (a dead band) |
| Class | `answerable` or `unknown` (`records.BALANCE_POOLS`), the world's. Unsliced: no slice is drawn. A job may keep only a share of the `unknown` golds (`JobSpec.look_unknown_keep`: the drone's 75%, the helper's 35%) |

The prompt states the body (`robot.json`'s `body`: its shape and size, its height, that the base is its middle, and for a
ground robot the climb and the band's top) and defines the corridor (`llm_render.CORRIDOR_TARGET`): the body moves
straight, and the distance is how far it can go along the heading before any part of it touches anything at its own
height, counted as the distance its front edge (ahead) or rear edge (back) moves; for a reverse it is measured from the
back of the body, and for a turn the body turns in place first and the distance is measured from the front edge after the
turn. The words do not name a strip: a box sweeps its rectangle and a disc sweeps a disc, from the shape in `robot.json`.
"Answer from what you have seen" leaves `unknown` as the answer where the robot never looked, where "anything you have
seen" could be read as calling unseen space clear. Without those numbers "how far could your body go" has no answer.

## Why it is hard

The controller needs the first thing inside its own width, and that is rarely the nearest thing in the picture:
- a chair leg 2 cm outside the corridor does not count, and a box 2 cm inside it does;
- a box on the floor stops a robot where its contact shows, and a console's top edge only where it can be carried down
  to the floor;
- a sofa whose underside is above the vacuum's 9.5 cm does not stop it, and the vacuum reads all of this from 8 cm up;
- ahead is mostly in view, and a strong model reads it to a few centimetres. Back and the turns are what the camera saw
  before the robot turned: the agent must keep where the table edge, the chair and the wall were, put them in the frame
  the robot would have after turning, and find the first inside the body's width. Or it must know that it never looked
  there: a camera pitched 15 degrees down rarely saw the space beside and behind a robot that has just turned at a
  counter, and then `unknown` is the right answer and a guess costs the full penalty;
- for the drone the camera points down at 58% of the points, so the wall ahead is at the image's edge or gone, and the
  wall behind was seen from another place and heading, and the drone answers at its own height, so it reasons from
  where the wall meets the ground up the facade.

## The gold: a ground robot

`grid.corridor_clear` over **what the robot last saw of every cell up to the point**: the latest free-space cells of every
keyframe of the window up to `k` (`w.latest_cells` over the window, not the latest frame alone; truth rule 2), so a cell
seen again takes its latest state and nothing after `k` is read. The corridor starts at the body's edge for the
move (`grid.front_edge`: the base plus half the body's length, never the camera):
- **ahead:** the front edge along the delivered heading, the cells counted past `robots.blind_m`;
- **back:** the rear edge, with no turn, along the opposite heading, every cell counted;
- **left, right:** the front edge on the heading turned in place by plus or minus 90 degrees, every cell counted.

The cap is `questions_words.cap_of` (5 m, or a metre inside the depth range: 3 m for the vacuum). The direction is drawn at
each look point by `stable_hash(case id, k, "direction")` over `JobSpec.clear_directions`, from the case id and the point
alone, so the draw reads no world, gold or answer (two worlds that differ only behind the robot draw the same direction).
An `unknown` gold is one where under 80% of the counted cells were seen free (the 80% rule, as before).

**The turn sweep.** A box body (the helper) turning in place sweeps a disc of its half-diagonal. For a turned direction a
seen occupied sub-cell inside that disc and outside the footprint makes the point ask nothing: the body could not make the
turn (measured, it bound at no drawn point of the helper or the AMR). A disc body turns inside its own footprint.

The rule, shared with [`path_ahead`](path_ahead.md), has the parts 0.8.0 changed:
- **The stop is read at the 2 cm sub-cells** (`grid.stop`). The answer is the least distance the body moves before it
  touches an obstacle sub-cell's centre (`grid.reach`), less `grid.SUB_M / 2`, capped. The 0.7.0 rule reported the near
  edge of the first blocking 0.1 m cell, a median 4 cm short of the obstacle (q90 8 cm), which is more than C/2 on 35%
  of the vacuum's golds and 14% of the helper's. The counted cells (the 80% rule: `unknown` when fewer than 80% of them
  were seen free) are those the body sweeps before the stopping fine cell, as before.
- **The body's own shape sweeps** (`robots.Body.shape`, the same function for every robot; final review B2). A **box**
  (the helper, the AMR) sweeps its rectangle, `width_m` across: a sub-cell inside the width is touched after its
  distance along the heading. A **disc** (the vacuum) of radius r sweeps a disc whose front point starts at the edge: a
  sub-cell at distance `a` ahead of the edge and `y` to the side (|y| at most r) is touched after
  `a + r - sqrt(r² - y²)`, up to r later than `a` at the disc's side, and one beside the front arc (`a` below 0, outside
  the disc) is touched as the disc moves on. The first 0.8.0 compile measured only the strip, and 8 of the vacuum's 29
  short and medium golds were 5 to 12 cm short of the disc's travel (the vacuum along a wall that converges on its side:
  gold 0.018 where the disc goes 0.129); the question asks how far the body could go, and the gold now answers that. The
  seen cells are those the body sweeps (a disc leaves out the rectangle's far corners). The drone's body is a disc too
  and its wall corridor follows the same rule (`grid.raycast` takes the body).
- **A surface at most 3 mm over the climb is floor** (`grid.DEPTH_QUANTUM_M`; `grid.band` is the climb plus the quantum
  to the body's height plus half the under-margin), for every robot. The depth's millimetre rounding lifts a 1 cm rug
  over the helper's 1 cm climb, and it stopped the helper's corridor at 21 of 87 short-case points; a 4 mm surface still
  stops it. No 0.7.0 gold changes: no shipped family reads the band.
- **`grid.raycast`** (the public map's rectangles against the same corridor) moved to `grid`, so a gold or a reference may
  call it.

An `unknown` gold is a candidate of the `unknown` pool. An answerable gold is asked only if it passes the screens
below, else the point is a dead band.

**Scope.** `now` exactly when the gold is a number and the corridor over `w.latest_cells([k])`, the cells of frame `k`
alone, gives the same number within the row's resolution (`clear_ground.frame_agrees`); else `memory`, and an `unknown`
gold is `memory`. Frame `k` is the latest frame, what a system holds when it is asked (it gets one frame at a time), so
an answerable `memory` row is exactly one the latest frame cannot answer, and the agent must hold what it has and has
not seen. Until 0.10.0 the corridor was read over the cells of the look window, the up to 8 frames a system was sent at
once. On 0.9.0's data the change moves 35 of the 78 `now` rows to `memory` (drone 9 of 32, helper 12 of 26, vacuum 14 of
20), leaving 43, and the question and its gold are unchanged. The move changes which cells reach the 20-question
minimum ([methodology.md](../methodology.md), rule 4: until 0.10.0 a cell under 20 questions was left out of the
headline; since 0.11.0 it weighs `n / 20`): on the lane-S recompile of 0.9.0's data the helper's and the vacuum's
clearance `now` cells hold 14 and 6 questions (they held 26 and 20) and become thin, so the headline's `now` column is
distance plus the drone's clearance cell (23 questions, from 32), and the drone's clearance `memory` cell holds 22 (from
13) and becomes counted. The rule is not loosened for this: the question redesign of 0.10.0 raised question density
instead (below). (0.8.0's rule, every evidence keyframe in the look window, would have put rows the window's cells alone
answer in `memory`: the vacuum's current view then led by +0.36 on those.)

## The screens

They read the timeline and the depth and remove questions only; they read no gold of another family and no answer
(`corridor_screens.py`). A question is asked only where a camera-only reader could reach C/2. They run on the moved
corridor (the moved start and heading) over the window's free-space states; the 0.8.0 thresholds stand.

1. **The stopper is tied and visible** (`stopper_tied`; `corridor_screens.stopper` returns the same verdict with the
   contact's least one-pixel error and the gap from the stop to that contact). In the keyframe that last observed the
   stopping fine cell, the obstacle points the body touches (inside the corridor and band) are found; the one touched
   first is the stop and its instance the stopper (`grid.reach`: for a disc not always the nearest ahead). Tied: a point
   of that instance lies within max(2 cm, slant / fy) of the floor, is touched within C/2 after the stop (inside the
   corridor widened by C/2), and its one-pixel error (d² + H²) / (H fy) is at most C/2 (the ego families' surface
   tie, applied to the stop). Visible: a wall or other structure, a fixed piece, or a thing recognisable in that
   frame. A cap gold needs neither. It drops cords under the pixel floor and a table top whose legs stand 0.4 m behind
   its edge (an overhang read as a stop).
2. **The side screen** (`CLEAR_SIDE_M` 0.01 m). Not asked when the body 1 cm narrower, or 1 cm wider, on each side (a
   disc's radius 1 cm smaller or larger) gives a gold more than C/2 from it; an `unknown` there counts as moved. The
   grid's own docstring says an obstacle within 1 cm of the body's side may fall either way. A disc is the more
   sensitive there: an obstacle tangent to it is touched only after the whole radius, one just outside it never, so the
   screen drops the tangent cases and keeps a wall that converges on the side by more than a centimetre.
3. **The height screen** (`CLEAR_HEIGHT_M` 0.005 m). Not asked when a thing the body sweeps before the stop (a box's
   rectangle; a disc's path: its centre's track widened by the radius, which takes in the sides beside the front arc)
   has a horizontal edge within 0.5 cm of the band's: its top and the band's floor (the climb plus the 3 mm quantum,
   `grid.band`), its underside and the band's top, or, when it hangs, its bottom edge and the band's top (a picture
   frame's lower edge). A bath mat or a remote, the coffee table's shelf over the vacuum or a frame's lower edge over
   the helper, decides the gold on a height no frame shows. The map's fixed floor coverings are exempt (they are under the
   band's floor by construction).
4. **Phases.** Asked only where the status phase is in `JobSpec.clear_phases`, the phases in which the robot drives:
   helper `navigating` and `carrying`; vacuum `cleaning`. In the others the status answers it (a docked vacuum, an AMR
   at a station: a held-out constant per phase scored +0.55 to +1.00), and methodology §1 says own path and progress are
   not asked. The vacuum's `transit` is left out (final review S3): it leaves its dock on a fixed route, the same poses
   and golds recur in different scripts, and a held-out constant per phase scored +0.103 over `all_unknown` on those
   rows, past the 0.10 form margin.
5. **The turn sweep** (above), for a turned direction of a box body.

**The credit's resolution** is `max(0.02, the stopper's contact error, the tie gap)`, so the largest of the grid's
sub-cell, what one pixel misses at the floor contact in the frame the stop rests on, and how far beyond the stop that
contact lies. The screens already keep the first two at most C/2; the credit's full quarter is what an ideal reader of
that frame reaches.

## The gold: a flier

`clear_fly.moment` asks only where the base is at least `JobSpec.fly_floor_m` up (the drone's 8.35 m) and the phase is in
`clear_phases` (every phase of the drone's status: the floor decides). The direction is ahead or back, drawn by
`stable_hash(case id, k, "direction")` over the drone's `clear_directions` (`("ahead", "back")`), the same draw as
0.8.0's.

- **Why a floor.** The gold assumes the drone flies above everything but the walls. Landed, it sits at 0.27 m among
  benches, planters, people and cars; hovering at 7.01-7.15 m, its 0.30 m body reaches down to 6.86 m, and the square's
  street trees (every 6 m along its walls) have crowns up to 6.90 m. 65 of 215 drawn points were landed or beside a
  crown. The street lamps are taller: the render stands one every 28 m in the kerb strip of every sidewalk of the map
  (`render/scenery_layout.py` `_strip_posts`, `POST_HEIGHT_M["lamp"]` = 7.95 m; `render/scenery.py` `_lamp` draws the
  pole to 7.5 m, an arm to 7.9 m reaching 1.7 m over the street and a head at 7.78-7.95 m). They are fixed scenery, not
  in the map, so the walls-only gold cannot see them and the floor must clear them. 8.35 m is the lamp head, plus the
  body's half-height 0.15 m, plus 0.25 m: the place's own dressing, read once from the render code's declared sizes and
  stated in the job file with its source (the gold code never imports `render/`). (0.8.0 first set 7.3 m, the tallest
  crown plus the same margins; the final review found the lamps, latent on the pilot's data.)
- **The gold** is the 1.3 m body's corridor from its edge (the base plus the body's radius along the heading, or its
  opposite) against the map's wall boxes (`MapSpec.wall_pieces()`, `grid.raycast` through `wall_seen.hit`), capped at
  `wall_seen.CAP_M` (60 m). Every movable is under 3.3 m and the drone never flies below the floor, which clears the
  trees, signals and lamps too, so only walls can stop it.
- **A corridor into open air is not asked**: a gold at the cap is None, because no frame certifies empty air.
- **Knowable.** The wall the corridor stops at (the first wall it meets: the body touches it before any beyond) is seen
  when some keyframe of the window up to `k` saw some stretch of it: `wall_seen.bins` records, for each keyframe, the
  (wall, 1 m bin) pairs its depth saw (points more than 0.5 m above the floor within 0.35 m of the wall's box across it
  and within 0.5 m of its ends, binned along the wall), and any bin of that wall will do, however far along from where
  the body meets it. A building face here is one flat plane, so the stretch met is inferred from the metres shown, and
  "anything you have seen" has a number: the distance. The gold is `unknown` only when the corridor meets a wall that
  no keyframe of the window showed at all, even if a wall beyond it was seen (the body touches the unseen one first);
  the `unknown` pool's candidate is thinned to 75% (`look_unknown_keep`). (0.8.0 first read the hit's bin and its two
  neighbours; the final review found 2 of the pilot's 9 `unknown` golds were walls seen 4-5 m along their face, with a
  wall beyond seen too.)
- **Scope** is `now` exactly when the latest frame, `k`, saw that wall; the evidence is the latest such keyframe, so the
  scope is `now` when that keyframe is `k` itself (`questions_world.scope_of`). This is the same rule as the ground's, since
  a wall seen in the latest frame is the whole of the gold's evidence.
  Pitched down 45 degrees no frame shows the facade at the drone's height, so the reader infers that a building face
  continues upwards; every wall here is 18.9-44.1 m tall with no opening, so the inference is sound for this place.
- **The resolution** is one pixel at the wall's foot from the evidence frame (`rules.resolution_m`, the ground z from the
  map's ground under the hit), at least 2 cm.

## Not asked

- **The AMR's corridors, in any direction.** Ahead: under the stopper screen 80 of 145 in-transit golds are the 5 m cap,
  a held-out constant scores +0.33 and the map reader +0.85, and in the stationary phases the status answers it. Back: a
  constant scores +0.24. Left and right pass the constant (+0.00 and -0.11) but 44% and 40% are the 5 m cap and the map
  reader leads by +0.53 and +0.80: mostly layout, and the AMR has no ask to spare. The AMR's harm questions are the
  people's: [`closest_approach`](closest_approach.md), [`approach_mem`](approach_mem.md) and
  [`closest_person`](closest_person.md).
- **The vacuum's right turn.** A held-out constant leads by +0.28 (the design's measure, on the rows it would have been
  asked at): the vacuum follows edges with the wall at its right.
- **Sideways flight**: a constant per direction scores +0.15 (the drone flies about 5 m from a side wall).

## Answerers, references and gates

- **Tables** (`table_clear`): the ground answer is `grid.corridor_clear` over the store's cells of the window up to the
  ask's point at the ask's pose and the moved start and heading, so `memory_table` reproduces the gold; a flier's is
  `table_clear_fly.answer`, the same functions (`wall_seen`) over the store's own depth record and the public map. The
  oracle reproduces it (the ceiling gate).
- **`path_log`** (`baselines_path_clear.own_answer`, answered through `baselines_path_own.answer`): the robot's own log
  only, `baselines_path_space.detour` along the moved heading (`clear_move.move`) over its own swept cells, else
  `unknown`, and kept only where it beats abstaining on other scripts: a detour is no proof of an obstacle (on the helper
  it scored -0.20), so the rule answers only where the lower end of the 95% interval of its gain over `unknown`, over the
  scripts of the prior, is above 0 (`baselines_path_own.kept`, the one bar every own-log rule of `path_log` stands by,
  `open_heading`'s too; each prior record carries the rule's answer as `own`). An `in_pool` prior keeps the rule unasked.
  No map, no pool of answers: its `answer_keys.levels` on this template are the bare template alone (nothing for a pool
  to find), and the check's reach rule does not name it for a reach rate of 0 there (`answer_keys.pooled`). It is gated.
- **`map_reader`** (`baselines_map.MapReader`, the clear answer in `baselines_map_clear`), the family's declared
  **reference**, never a shortcut: on the ground the public map's obstacles raycast from the moved start along the moved
  heading (the body's sweep and the sub-cell stop), answered as it is; for a flier the wall raycast against the walls
  taller than the base's height; on a held-out layout, which has no map, the pose-keyed pool's best response where the
  held-out rule allows; `unknown` on every other template. **`other_case`** is a reference here too. The maps are
  public, so any system that knows its pose can raycast one. In the home, behind and to the sides the clutter decides and
  not the walls, so the reader scores far below its ahead corridors. The free check of 0.9.0 (the dev
  root `dev-090`, every direction asked, all lengths) scores it 0.20 on the helper, 0.66 on the vacuum and 0.69 on the
  drone, against `all_unknown`'s 0.19, 0.08 and 0.16 (leads of +0.01, +0.57 and +0.53); the design measured the helper's
  back, left and right corridors alone at -0.51 to -1.15 (+0.51 and +0.57 on the vacuum's back and left). On the dev
  root of Release 0.6 (dev-080b, 0.8.0, ahead corridors) it read +0.56 on the helper, +0.46 on the vacuum and +0.69
  on the drone: the drone's gold is `unknown` where no frame showed the wall it would hit, which a reader of the map
  without the pictures cannot tell, and the ground robots' golds are read over what the depth showed, which stops short
  of the map's fixed structure on some rows. On a public place a `clear_path` score is a claim only about a system that
  holds no copy of the place's map; a held-out place makes it a claim about every system. The `stbench check` report
  prints the family above the map reader.
- **Gated**: `blind` (the held-out best constant: +0.00 on the helper and the drone and -0.30 on the vacuum over the
  directions asked, in the free check of 0.9.0), `form`, `questionnaire_blind`, `all_unknown`,
  the balance rule, `path_log` (own log) and, on `memory` rows, `current_view` and `current_view_prior` with no pose key
  (`answer_keys.levels` drops the pose-keyed level for a template whose spec names `map_reader`). `ceiling`, `knowable`,
  `separation` and the balance check (`answerable` against `unknown`) as for any family. Each declared direction was
  measured alone on the dev root: the current view alone scores 0.00 on the rows that need memory (measured in 0.9.0,
  when the current view was the last batch of up to 8 frames). In 0.10.0 the check's `memory_needed` also reads the
  `_in_pool` prior; the vacuum and the drone pass, and the helper's evidence cells do not (next section).

## The helper's `memory` cells: the decision-frames-first draw made a channel, and it is removed

The check's `memory_needed` gate asks whether a system that keeps no memory of the case can answer the family's `memory`
rows. For `clear_path` the memoryless systems are `current_view` (the latest frame alone) and `current_view_prior` with
its `_in_pool` twin (the gold that other scripts' records held with the same phase and the same quantised answer of
`current_view`, the corridor the latest frame alone gives, with no pose key, since a pose key would learn the layout).
The vacuum and the drone passed. On the helper, after the 0.10.0 redesign and until the final review's repair, two of the
**evidence** verdict's cells failed. The first diagnosis was recurring routes, accepted as the fixed structure the vision
allows; the final review corrected it.

**What the check said** (the dev root, 36 helper cases, 106 questions, before the repair):

| Cell | `current_view_prior_in_pool` over `all_unknown` | Rows | The gate allows |
| --- | --- | --- | --- |
| all `memory` rows (the point verdict) | +0.033 [-0.082, +0.143] | 75 | 0.05: passes |
| `memory` rows with an answerable gold (evidence) | +0.136 [0.022, 0.283] | 62 | lower bound above 0: fails |
| `memory` rows of long cases (evidence) | +0.177 [0.018, 0.375] | 18 | lower bound above 0: fails |

**The cause was the draw, not the place.** The final review split the 106 rows by whether their look point is a frame of
`JobSpec.look_decisions`, the decision frames the draw puts first:
- 44 of the helper's 106 `clear_path` rows sat at decision frames (9 of the vacuum's 315). The carrier gold (`right`, 0.56
  m: 10 rows in 5 scripts) had 8 of its rows at `before_place`.
- With the check's own gates the decision-frame rows alone read +0.211 (answerable +0.405 [0.181, 0.672]), the other frames
  +0.000, and the rows without `before_place` passed every gate (+0.010 [-0.072, 0.111]).
- A decision frame puts a structure question at the same plan pose, by the same furniture, in every script, so the
  corridor length at that frame recurs from script to script and a prior keyed on the latest frame learns it. A question's
  presence at a decision frame reveals plan timing and layout: harmless for a family whose answer depends on the world at
  that moment, a channel for a structure family. (Lane J3's "no direction, draw-dependent" was true and missed the cause.)

**The repair** (repair R1) takes `before_place` out of the helper's `look_decisions`
([job-helper.md](../job-helper.md)). It changes no gate, no threshold and no share. On the compile after it (the dev root,
36 helper cases, 96 `clear_path` questions: 29 `now`, 55 `memory` answerable, 12 `unknown`) 28 of the 96 rows sit at decision
frames, the memory cells read +0.000 [0.000, 0.000] in every cell (all 67 `memory` rows, the 55 answerable, the 17 of long
cases), and `stbench check` prints `EVIDENCE VERDICT PASSES` and `ALL GATES PASS`. The second form of the repair, never
drawing a decision frame first for a structure family, was not needed and is kept for a family that still fails. The
carrier gold is still there in other frames (`right` 0.56 m, 4 rows in 4 scripts), where it does not carry the gate.

**What it means for a reader.** A helper score on this family is no longer an exception to read with a caveat; the map
reader (0.337 on the helper's rows in the release check) is an internal design diagnostic of `stbench check`, not a
published reference (the agent is never given a map), and on a public place a `clear_path` score is a claim only about a
system with no copy of the map (methodology rule 5).

## Supply and difficulty

The free check of 0.9.0 (over the dev root `dev-090`, every direction asked, all lengths). The held-out
best constant, the shortcuts and the separation are leads over `all_unknown` (the gates' bars are 0.10
for the shortcuts and 0.40 for the separation; a negative constant is worse than answering `unknown`); the last two
columns are scores:

| Job | Rows | Held-out best constant | Shortcuts' largest lead | Separation | Map reader | `all_unknown` |
| --- | --- | --- | --- | --- | --- | --- |
| helper | 47 | +0.00 | +0.00 | +0.81 [+0.70, +0.92] | 0.20 | 0.19 |
| vacuum | 24 | -0.30 | +0.02 | +0.92 [+0.80, +1.00] | 0.66 | 0.08 |
| drone | 45 | +0.00 | +0.00 | +0.84 [+0.74, +0.93] | 0.69 | 0.16 |

The rows asked in the short and medium cases of the same compile, by direction:

| Job | Direction | Rows | `now` | `memory` answerable | `unknown` |
| --- | --- | --- | --- | --- | --- |
| helper | ahead | 11 | 11 | 0 | 0 |
| helper | back | 8 | 0 | 6 | 2 |
| helper | left | 10 | 5 | 4 | 1 |
| helper | right | 8 | 4 | 1 | 3 |
| vacuum | ahead | 6 | 6 | 0 | 0 |
| vacuum | back | 4 | 4 | 0 | 0 |
| vacuum | left | 11 | 7 | 2 | 2 |
| drone | ahead | 19 | 16 | 3 | 0 |
| drone | back | 18 | 8 | 3 | 7 |

The helper has 37 rows in all (20 `now`, 11 `memory` answerable, 6 `unknown`), the vacuum 21 and the drone 37. The
median `rho / N` of the answerable golds is 0.13 on the helper (q90 0.31), 0.21 on the vacuum (q90 0.50) and 0.04 on the
drone (q90 0.10). The drone's clearance `memory` cell (13 questions) and the vacuum's (4, now that `open_heading` is an
ability of its own) stay under the 20-question line, so they were left out of the headline (a thin cell weighs `n / 20`
since 0.11.0); the helper's (30) counts.

**In 0.10.0** (the dev root, 36 cases a job, after the final review's repair; the points are any frame now
and the shares are the slot tables'), the family asks 96 questions on the helper (66 in the short and medium cases; 29
`now`, 55 `memory` answerable and 12 `unknown`; unknown share 0.12; a fill of 0.258 a drawn slot; 106 before the repair),
315 on the vacuum (180; unknown share 0.02; fill 0.429) and 196 on the drone (144; unknown share 0.09; fill 0.566). The
clearance cells it feeds all count (with the other harm families in them): the drone's `memory` cell holds 100 questions
and its `now` cell 96, the vacuum's 346 and 102, the helper's 67 and 51. Every gate and the evidence verdict pass on all
three jobs. The map reader scores 0.329 on the helper, 0.704 on the vacuum and 0.816 on the drone, and the oracle and
`memory_table` score 1.000 on all three. A `now` row's `age_s` is 0 (final review S6; 9 `clear_path` rows of 0.9.0's data
named an earlier frame and were aged from it), and its catastrophe is charged only where the true corridor is under the
body's length plus C and the answer is not (S5): a drone corridor's median gold is 24 m against a 2.4 m margin, and a
30 m answer to it hits nothing. The 0.11.0 release holds the same counts (96, 315 and 196 questions in 36 cases a job,
66, 180 and 144 of them in the short and medium cases); its check passes every gate and the evidence verdict, the map
reader reads 0.337, 0.706 and 0.816 there against `all_unknown`'s 0.125, 0.016 and 0.092, and the oracle and
`memory_table` read 1.000.

The precision-and-forgetting proxies (each model's own `clear_path` errors on `now` rows, its `ego_mem` errors and
abstentions on `memory` rows and its `unknown` recall on unknown rows) bound what precision costs on the turned corridors:
sol +0.43 (helper) and +0.71 (vacuum), astra +0.90 and +0.83, luna +0.18 and +0.17. They assume the right stopper, where
0.7.0's log reviews put 18-74% of sol's lost credit. The 0.11.0 results are in [report.md](../report.md), Section 4.
