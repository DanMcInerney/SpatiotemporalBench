# `open_heading`

Which way gives the most room: the heading, turned in place, that would let the body go farthest before it touched
anything the robot has seen. During, scope by evidence. Code: `stbench.questions_heading` (the spec, the words and the
gold), `stbench.clear_profile` (the profile of runs by heading), `stbench.grid` (the one corridor rule, shared with
[`clear_path`](clear_path.md)), `stbench.corridor_screens` (the screens), `stbench.table_heading`,
`stbench.baselines_path_heading` and `stbench.baselines_map_heading` (the answerers), `stbench.answers` (the `heading`
answer). The record and questionnaire rules are in [questions.md](../questions.md). New in 0.9.0. Its kind
is **help**: which action now best moves the task forward. A vacuum cornered under a table or a helper backed against a
counter chooses where to go by where there is room; the answer is the action, scored by what it gives up. It is a
**structure family**, like `clear_path`: it reads fixed structure, which the camera sees; the map reader that answers it from
the public map is an internal design diagnostic of `stbench check`, not a published reference (below).
**Counted since 0.11.0.** Its ability is **clearance** (`abilities.FAMILY_ABILITY`) and its rows count in the headline,
with a credit that is the same kernel on the same need rule as every distance, and a direction that is room to move
(below, "Counted since 0.11.0"). In 0.9.0 and 0.10.0 its ability was **heading**, a diagnostic (`abilities.DIAGNOSTIC`):
asked, gated and reported, as the help kind and as its own ability, and counted in no headline mean, because its credit
was a cliff (below, "The credit is a cliff", kept as history).

| | |
| --- | --- |
| Category | C4 Which way |
| Asked | during, at a look point drawn from the job's slot table (`JobSpec.look_shares`): the helper's third slot at 15% (beside `ego_now` at 20% and `approach_mem` at 30%), the vacuum's third slot at 20%. Through 0.9.0 it was the helper's second slot at 20% and the vacuum's at 15%. The AMR and the drone do not ask it (below). Only where the job's `clear_phases` hold the point's status phase. At most one candidate a point; none is a dead band. It carries no subject (it names no thing and no time), so a point may ask it beside a corridor or an `ego_mem` question |
| Text (slots) | "If you had to turn in place and drive straight away from here, which way, of those you have seen, would let your body go farthest before it touched anything? Answer the heading in degrees to the left of the way you face now, from -180 to 180 (to your right is negative)." The only slot is `cap` (the robot's corridor cap), which a table system needs and the words do not state |
| Answer, credit | `heading` (`{"type": "heading", "deg_left": x}`, x a finite number in [-180, 180]; anything else is malformed). `regret` ([questions.md](../questions.md), "Credit"): the answered heading is rounded to the profile's step and its run `d` read from the row's profile (0 for a heading into space never seen); the regret `r` is the best run less `d`, in metres, and the credit is the four quarters on `e = r` with the need `N' = max(C, 0.1 best, 2 rho)` (0.11.0; N = C through 0.10.0). The direction quarter is room to move: a run of `T`, the body's length plus C, or longer, or the best run less a pixel where no heading has that much (0.9.0 and 0.10.0: a run at least half the best); the harmful side (the first two terms of the need halved) is a run under `T` |
| Scope, `needs` | by evidence: `now` when the profile of frame k alone (the latest frame) has a best heading whose regret on the window's profile is within the resolution, else `memory`. See, pose |
| Gold function | the `open_heading` entry of `questions_heading.GOLDS`, called with the world and the point: a `Moment` or None (a dead band) |
| Class | `answerable` only (`records.BALANCE_POOLS`): a point where no heading was seen asks nothing, so there is no `unknown` gold and the balance gate reads one pool |

The prompt defines the answer (`llm_render.HEADING_TARGET`): degrees to the left of the way the robot faces, -180 to 180,
the body turned in place, the run measured to the first thing the body would touch. The body and the corridor are defined
as for `clear_path`.

## Why it is hard

The agent must hold the whole surround, in view and remembered, and compare the room in every direction. Measured on the
dev root, the best heading is not in what the latest frame shows at 17 of the helper's 19 asked points in the short and
medium cases and 9 of the vacuum's 13 (under 0.9.0's rule, which read the look window's up to 8 frames, 11 of 19 and 7 of
13); in the 0.11.0 release the latest frame gives it at 2 of the helper's 54 questions and 33 of the vacuum's 134. A
constant heading cannot answer it: straight ahead or straight back is the best at only 5-10% of points.

## The gold

The profile: for each heading at 1 degree steps from the robot's heading, the body turned in place, `grid.corridor_clear`
over the latest cells of every keyframe of the window up to the point (`clear_path`'s window, the left and right turns at
every angle), from the turned body's edge, and `null` where the corridor was not seen (under 80% of its counted cells
seen free). `clear_profile.profile` computes it vectorised over the occupied and counted sub-cells (each sub-cell's
touch distance per heading bin), so one point's 360 headings take well under a tenth of a second, and it equals
`grid.corridor_clear` at every heading it samples.

The **best run** is the largest of the profile and the gold heading is the one nearest ahead among those within the
corridor's 2 cm sub-cell of it (`clear_profile.best_index`). The gold row carries the whole profile (`tolerances.profile_m`, 360 values for a 1 degree
step, `null` for an unseen heading) with `step_deg`, `resolution_m` (the best run's, as `clear_path`'s: the largest of the
2 cm sub-cell, the stopper's contact error and the tie gap), `need_m` (C) and `threshold_m` (the body's length plus C),
so the scorer reads the regret of any answer without the world. The gold answer is the gold heading as a `heading`.

**Screens.** The best heading's corridor runs to the cap, or passes `clear_path`'s height and side screens with a tied and
visible stopper whose one-pixel error and tie gap are at most C/2; the turn sweep (a box body's turn must clear the seen
obstacles inside the disc of its half-diagonal: `clear_profile.turn_swept`); the moving phases only (`clear_phases`); and
a point with no heading seen asks nothing. A flier is never asked. On the dev root's compile the screens leave the
helper 24 rows and the vacuum 16, in all lengths (below). The final review's probe of those rows found the vacuum's best
run the 3 m cap at 80 to 100% of its points, with a median of 7.5 to 17 optimal headings where the design counted two,
5 degrees apart (the gold is the nearest ahead of the tied headings). The best constant heading still scores about
-0.50 there.

## Answerers, references and gates

- **Tables** (`table_heading`): the profile over the store's cells of the window up to the ask's point, and its best
  heading, so `memory_table` reproduces the gold and the oracle reaches the ceiling.
- **`path_log`** (`baselines_path_heading.own_answer`, answered through `baselines_path_own.answer`): the heading of the
  robot's longest straight run over its own swept cells (headings 5 degrees apart, the run unbroken from the turned
  body's edge; no map, no pool), else `unknown`, and kept only where it beats abstaining on other scripts: it stands by
  the same bar as `clear_path`'s detour (`baselines_path_own.kept`, [eval.md](../eval.md)). On the dev root's compile the
  rule is never kept, so `path_log` abstains on `open_heading`; without the bar it scored -0.32 on the helper. It is
  gated.
- **`map_reader`** (`baselines_map.MapReader`, the heading answer in `baselines_map_heading`), the declared **reference**,
  never a shortcut: the best heading of the public map's raycast profile from the delivered pose (headings 1 degree
  apart, as the gold's profile is, turned in place, from the turned body's edge), answered as it is, as
  [`clear_path`](clear_path.md)'s reference answers its ray; on a held-out layout, which has no map, the pose-keyed pool's
  learned answer alone, `unknown` with none. A pool never stands beside the ray: the first build offered the ray to the
  pool's held-out rule as one more candidate, which scored it on other cases' golds and abstained wherever the pool was
  coarse (a pose recurs in few scripts), so the free check of 0.9.0 printed 0.000 for both jobs (the final review, S1).
  The review's probe of the fixed reader expects about -0.10 on the helper and +0.09 on the vacuum, against
  `all_unknown`'s 0.00 (the release check reads it -0.02 and -0.01): a map does not tell a cluttered home's open way,
  because what the robot has seen of the clutter decides it and the map holds the walls and the fixed pieces alone. That
  is what a layout-memorising agent has, and it is the family's declared reference in `stbench check`, a design
  diagnostic.
- **Gated**: `blind` (the held-out best constant heading abstains, +0.00), `form`, `questionnaire_blind`, `all_unknown`,
  the balance rule, `path_log` and, on `memory` rows, `current_view` and `current_view_prior` with no pose key.
  `ceiling`, `knowable` and `separation` as for any family. With one balance pool (`answerable`) the balance gate reads the
  family against `all_unknown`, which scores 0 on every row of a family without an `unknown` gold.

The free check of 0.9.0 (over the dev root `dev-090`, all lengths; helper / vacuum; leads over
`all_unknown`): the shortcuts' largest lead +0.00 / +0.00 (the gate's bar is 0.10); the held-out best
constant heading +0.00 / +0.00 (the balance rule); the current view and the current view with a pose key lead +0.00 on
the `memory` rows (`memory_needed`); `separation` +1.00 / +1.00 (interval +1.00 to +1.00); `path_log` abstains, 0.000 on both jobs (its in-pool twin, which
keeps the rule unasked, scores -0.324 / -0.223 and is harmless, since a gate reads the larger). The reference
`map_reader` printed 0.000 on both jobs before its fix (above).

## Counted since 0.11.0: room to move on the one need rule

The credit is the kernel on the regret with `N' = max(C, 0.1 g, 2 rho)` and `g` the best run (`credit_kernel.KAPPA`),
the rule every distance is scored on ([question-design.md](../question-design.md), section 6):
- **The direction is room to move.** The answered heading's run is `T` or longer (`T` the body's length plus C), or at
  least the best run less a pixel where no heading has `T`. A heading with that room earns the quarter however far short
  of the best it falls; the harmful side is the complement, a run under `T`, where the first two terms of the need are
  halved.
- **The need** is C while the best run is within ten of it (1.0 m on the vacuum, 1.5 m on the helper) and a tenth of the
  best run beyond, never finer than two pixels. A 1.5 m regret on a 5 m best scores -0.25, where it scored -0.5.
- **It counts.** `open_heading` is a clearance question (`FAMILY_ABILITY`), so its rows sit in the clearance cells of
  the headline; the help kind is still a split of the reports.

What decided it. The robot's own log, the longest straight run it drove (the in-pool twin of `path_log`, unbarred), was
scored on every compiled heading row of the 0.10.0 dev root (54 helper and 134 vacuum rows), and the three models and the
constants on the 38 rows among the 42 reference cases (the intervals are over scripts):

| Credit | Own log, helper | Own log, vacuum | Luna / sol / astra | Best constant, helper / vacuum | Map reader, helper / vacuum |
| --- | --- | --- | --- | --- | --- |
| 0.10.0: direction half the best, N = C | -0.171 | -0.371 | -0.596 / -0.130 / +0.053 | -0.60 / -0.78 | -0.23 / -0.27 |
| room to move, N = C | -0.115 | -0.214 | -0.425 / -0.038 / +0.118 | -0.42 / -0.55 | -0.13 / -0.11 |
| **room to move, N' (shipped)** | **+0.037** [-0.24, +0.34] | **-0.143** [-0.38, +0.08] | **-0.334 / +0.028 / +0.162** | **-0.40 / -0.52** | **-0.05 / -0.02** |
| room to move, a fifth of the best run | +0.254 [-0.02, +0.52] | -0.000 | -0.173 / +0.157 / +0.257 | -0.25 / -0.39 | +0.16 / +0.14 |
| room to move, then linear in the room | +0.380 [+0.10, +0.64] | +0.100 [-0.11, +0.30] | -0.048 / +0.252 / +0.316 | -0.12 / -0.28 | +0.29 / +0.25 |

Held out, every constant heading abstains (0.00) under every credit. Why the last row is not used: the helper's swept
path nearly always has room, so a credit that pays every heading with room in proportion to that room lets the robot's
own log alone beat abstaining by +0.38, far past the shortcut line (0.10). The shipped credit keeps the log at +0.04 on
the helper and -0.14 on the vacuum, graded (a 1.5 m regret on a 5 m best is -0.25, not -0.5), and brings the map reader
to the floor (-0.05 and -0.02), so that diagnostic no longer beats abstaining; luna, sol and astra stay apart (a
hard family). Counted in clearance, it lowers the apartment's headline by 0.04 to 0.05 for sol and astra on 0.10.0's
answers.

**The own-log screen** (`heading_screen`, `FamilySpec.own_log_screen`, off): the fallback declared before the check, for
a job on which `path_log` or its in-pool twin then reads over +0.10. A heading point is a dead band when the robot's
own-log heading would score 0 or more under the row's credit, as `World.told` never asks what the robot's log tells. The
design measured its cost at 25 of the helper's 54 rows and 41 of the vacuum's 134. It is not on in the specs as merged;
a family that still failed after it would leave that job, by rule, and never be re-tuned. The free check of the 0.11 dev
root passes in every reading on every job (`ALL GATES PASS`, `EVIDENCE VERDICT PASSES`, one job at a time), the largest
shortcut lead on this family being +0.04 on the helper and +0.00 on the vacuum, against a gate line of +0.10.

## The credit is a cliff (history: 0.9.0 and 0.10.0)

The calibration read every miss and abstention of the 12 apartment rows (sol, luna, astra and the `map_reader`, on the 47
cases of `dev-090`) against the gold. The golds are right and the credit is not layered, so the family is a diagnostic
in 0.9.0.

**The golds are right.**
- All 12 profiles, recomputed at the delivered pose over the window up to the point, equal the gold rows' `profile_m`.
- The public map's raycast, independent geometry, picks the gold's opening on the 4 rows it was checked on.
- The convention is followed: no answer is explained by a mirrored or map-frame reading, and astra lands within 0.7
  degrees of the gold on 4 rows, two of them openings 1 degree wide. Full credit is reachable.

**The credit is a cliff.**
- A run changes by metres within a few degrees at a doorway, a cord's end or a furniture edge, while the regret is read
  in metres against N = C, 10 to 15 cm. Past a regret of 4C (0.4 to 0.6 m) only the direction quarter is left, and it
  needs a run of at least half the best.
- Over all 40 compiled rows, the contiguous window around the gold that scores at least 0 (as much as `unknown`) has a
  median width of 5 degrees, and 55% of rows are at or under 5 degrees. In the 12 calibration rows, 7 are at or under 3.
- So answers 2 to 10 degrees off the right opening score -0.4 to -1: six of sol's eight misses are that near. Answers that
  leave 1 to 2 m of room score -1, the same as driving into a wall: 3 of sol's 10 answers, 3 of luna's 12 and 1 of
  astra's 7 leave 0.96 to 1.85 m and score -1. A seen heading drawn at random scores -0.90.
- At 6ac394a44679244d/42f4b7e7bde9b437 (vacuum) the best opening is the 3.0 m cap: +0.97 at 108 degrees, +0.68 at 107 and
  -1 at 106. sol answered 105 (a run of 1.37 m), astra 104 (1.33 m) and luna 135 (0.96 m), all -1, each run more than
  three times the vacuum's T = 0.44 m, so none of them is harmful. At ad98f2fe86c2dbce/a01ca9d5f3fef785 (helper) the best
  run is 4.41 m at -82 degrees and nowhere else: sol, 3 degrees off, scores -0.5, and astra, 11 degrees off, -0.43.
- It breaks two rules of the vision: roughly right answers still score, and harm costs more than caution. It is also why
  the calibration missed the design by about 1.0 for sol: the design expected sol +0.63 (helper) and +0.71 (vacuum) and
  astra +0.87 and +0.85, and the calibration gave sol -0.60 and +0.33, astra +0.46 and -0.33 (`clear_path`, credited by the
  same kernel on distances, landed within about 0.3 of its design). The design assumed few tied openings and a smooth
  profile; the final review had already found 7.5 to 17 optimal headings where it counted two.
- No single parameter restores the layering, and any tolerance chosen now would be chosen after reading the scores. The
  variants rescored from the gold rows' own profiles (a robust profile over a few degrees, the direction quarter as room
  to move, credit for the best run within 2 degrees of the answer) move sol's mean on the 12 rows between -0.50 and
  -0.03 and do not make the random heading score above -0.71.

**What 0.9.0 did.** `open_heading` was mapped to the ability `heading` and `heading` was in `DIAGNOSTIC`: the family was
still asked, still gated, and still reported as the help kind (`by.kind` shows its mean, flagged as counted in no mean)
and as an ability of its own (`by.ability`), but it was counted in no `now` or `memory` cell. The stored answers rescore
with no recompile and no rerun. On the calibration the apartment headline of sol moves from +0.345 to +0.459, astra's
from +0.652 to +0.682, luna's from -0.116 to -0.035, the `map_reader`'s from +0.139 to +0.208 and the floor's
(`all_unknown`) from +0.061 to +0.081; downtown and the warehouse do not ask the family. The cells that stay counted do
not thin: the vacuum's clearance `now` cell falls from 26 to 20 golds in the dataset (20 is not under the minimum) and
the helper's from 35 to 26. (0.10.0's scope rule, `now` as the latest frame, moves more clearance rows to `memory`, and
these two cells then hold 6 and 14 golds and are thin: [questions.md](../questions.md), "How a gold is computed". The
redesigned questions of 0.10.0 count them again: on the compile after lane J2, the vacuum's clearance `now`
cell holds 102 questions and the helper's 35.)

**0.11.0 is that redesign** (above), decided on the 0.10.0 rows' own-log measurement and on the stored answers' rescore,
not on a model's score.

## Limits

- **Only the best heading is checked against the screens.** The screens (above) keep a point when its best heading's
  corridor passes them; the regret of any other answered heading is read from the same profile, so a heading the model
  answers can read a run whose stop the camera cannot resolve (a height within 5 mm of the band's edge, an untied
  stopper). This touches partial credit alone, not the ceiling: the oracle and `memory_table` reach 1.00, because the
  best heading is the one that is screened.

## Not asked

- **The AMR.** Every best run is the 5 m cap, with a median of 24 of 72 headings optimal; the held-out constant leads by
  +0.23 and the robot's own log by +0.60.
- **The drone.** The open high street is the longest run from every point of the square, a layout fact (0.8.0's map
  reader scored +1.00 on the drone's corridors).
- **A heading or route to the dock or a visited place.** These are the pose log. The way to a thing the robot saw is
  [`way_to`](way_to.md) since 0.11.0 (the helper's), scored by the progress through the floor seen and not by a bearing.

## Supply and difficulty

On the 0.11.0 compile of the dev root the family asks 54 questions on the helper (40 in the short and medium cases; 2
`now` and 52 `memory`) and 134 on the vacuum (85; 33 `now` and 101 `memory`); with `open_heading` in clearance the
helper's clearance `now` cell holds 53 questions and its `memory` cell 138 (119 without `approach_mem`'s 19), the
vacuum's 135 and 447, so no cell is thin. In 0.10.0 (the dev root, 36 cases a job; the points are any frame and the
shares are the slot tables') the family asks 58 questions on the helper (44 in the short and medium cases; 2 `now` and
56 `memory`; a fill of 0.509 a drawn slot, from its third slot at 15%) and 134 on the vacuum (85; 33 `now` and 101
`memory`; a fill of 0.457, from its third slot at 20%). The helper's heading `now` cell, 2 questions, was thin; the
family was a diagnostic anyway. The next paragraph is the compile of 0.9.0's data with `now` as the latest frame, before
the redesign.

Compiled on the dev root (the free check's whole-root compile of the job tables, with `now` as the latest frame): the
helper has 19 rows in the short and medium cases (2 `now`, 17 `memory`) and 24 in all lengths (2 `now`, 22 `memory`),
the vacuum 13 (4 `now`, 9 `memory`) and 16 in all (4 `now`, 12 `memory`). Under 0.9.0's rule, which read the look window,
the helper had 8 `now` and 11 `memory` in the short and medium cases (9 and 15 in all lengths) and the vacuum 6 and 7
(6 and 10 in all). The median
`rho / N` is 0.13 for the helper (q90 0.27) and 0.20 for the vacuum. In 0.9.0 and 0.10.0 `open_heading` was the whole of
the help kind, which held under 20 questions in every cell, so the kind was a split of the reports and not a cell of
the headline ([eval.md](../eval.md)). It was the ability `heading`, a diagnostic, and no part of clearance: the helper's
clearance `memory` cell had 30 short and medium questions without it (41 with it), the vacuum's 4 (11 with it; a
diagnostic either way). An agent that remembers nothing scores the best heading of what it last saw, +0.04 (helper) and +0.30
(vacuum) over all kept points (measured in 0.9.0, when the current view was the last batch of up to 8 frames). The precision-and-forgetting proxies score sol +0.63 and +0.71, astra +0.87 and +0.85, luna
-0.33 and -0.36.

**Results of the calibration** (the 12 apartment rows of `dev-090`, helper 9 and vacuum 3, penalised score): sol -0.37
(helper -0.60, vacuum +0.33), luna -0.62, astra +0.26 (helper +0.46, vacuum -0.33), the `map_reader` -0.21 and
`all_unknown` 0.00. Against the proxies above that is about 1.0 below the design for sol; the cause is the credit, not the
gold ("The credit is a cliff").
