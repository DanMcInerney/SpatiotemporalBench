# `way_to`

"If you had to go now to the {thing} you saw, which way would you set off to get there by the shortest way your body fits
through, of the ways you have seen? Answer the heading in degrees to the left of the way you face now, from -180 to 180
(to your right is negative)." (0.11, ability *memory*, kind **help**): the heading, turned to in place, whose first
metre of driving brings the robot nearest a thing it saw before, through the floor it has seen. During, `memory`. Code:
`stbench.questions_way` (the spec, the words and the gold), `stbench.route_field` (the walkable floor, the distance field
and the outcome of every heading), `stbench.table_way`, `stbench.baselines_path_way` (the answerers), `stbench.
heading_screen` (the own-log fallback, off).

It is the help kind's core decision, "the way to the goal". A person says "bring me my glasses"; the helper must recall
where it saw them and which doorway to take. It composes three things, a remembered place, a remembered layout and the
current pose, and it is scored by what the answer gives up, not by a distance: a doorway wrong costs the whole regret.
Only the home helper asks it. The vacuum has no fetch decision (its way home is a fact of the layout its own log often
gives); the AMR is dispatched by a fleet manager that holds a map and has few nameable things; the drone flies over the
streets, so the way to a ground thing is mostly its bearing, which `ego_mem` already asks.

| | |
| --- | --- |
| Category | C4 Which way |
| Asked | during, at a look point, in the helper's second slot beside `ego_mem` (`JobSpec.look_shares`: `ego_mem` 65, `way_to` 35, the 35 that slot drew nothing on, so `ego_mem` keeps its share), only in the phases of `JobSpec.look_phases["way_to"]` (`navigating`, `carrying`: where the helper drives). A job that asks it without declaring the phases raises |
| Text (slots) | as above. Slots: `kind`, `attributes` (the thing, as `ego_mem` names it) and `reach_m` (R) and `room_m` (T), which a table system needs and the words do not state, as `open_heading`'s `cap` |
| Answer, credit | `heading` (`{"type": "heading", "deg_left": x}`, x in [-180, 180]), credit `route`: the kernel on the regret `e = max(o) - o_i` of the row's outcomes with N = C and `rho` the field's 0.1 m cell, so the need is N' = max(C, a tenth of R, 2 `rho`) = 0.2 m on the helper, a null heading `e = max(o) + step_m`, scored -1 and harmful. See [questions.md](../questions.md), "Credit" |
| Scope, `needs` | `memory`: the thing is not in the recent view (the last 8 frames of the task up to the point), as `ego_mem` requires. See, pose |
| Gold function | `questions_way.way_to(w, k) -> Moment \| None`: the point's `answerable` and `unknown` pools, key `str(k)`; None when nothing is answerable |
| Slice | sliced (`SLICED`): the questionnaire draws the unknown slice per point (`UNKNOWN_PERCENT`, 20%). `answer_class` is `answerable` or `unknown` (`records.BALANCE_POOLS`) |
| Evidence | the thing's sightings at its place (`ego_mem`'s) and the keyframes that certified the cells of the shortest route from the base; `age_s` is from the oldest |

## Who is asked about

The subjects and the `unknown` pool are `ego_mem`'s (`questions_ego.ego_mem`, [ego_mem.md](ego_mem.md)): a named thing seen
before the recent view and not in it, unmoved since, which no look-alike hides and the robot's own log does not tell
about; an `unknown` real thing with no pixel in the window, or an absent description. A point asks one family or the
other, and a thing is named once a case (the subject rule), so a screen added to the pool (the S4 glimpse screen, a
resolvability change) reaches both. The words carry no gold and the form predicts no slice: a seen thing with no seen route
is a dead band, never an `unknown` gold.

## The gold

`route_field.route` over the cells and the delivered poses of the window up to the point, and nothing after it (truth
rule 2):

1. **Walkable floor.** The 0.1 m fine cells whose latest state is `free` (`grid.latest`, the states `clear_path` reads)
   or that the body swept (`grid.swath` over the delivered poses of each task: driven space is known free; the robot is put
   elsewhere between two tasks, so no segment joins them), less any cell last seen `occupied`; **eroded** by half the body's
   width, so a cell is walkable only where every cell the body would touch standing on its centre is known too; plus the
   cells of the base's own centre line. A cell never seen is not walkable: a claim that floor is empty needs floor seen
   empty (truth rule 2).
2. **The goal set.** The walkable cells within R (1.0 m) of the thing's middle at its **last recognisable sighting** (not
   where it stands now: "as you last saw it").
3. **The field.** `geo(c)`, the 8-connected Dijkstra distance in metres from the goal set; a diagonal step needs both cells
   it passes beside. `Field.near` takes it out to the floor the body touches but cannot stand on (within half the body's
   width and a fine cell of a walkable one), because a heading that runs until it touches something arrives there.
4. **Outcomes.** For each heading at 1 degree, the corridor of `clear_profile.profile` (the swept cells counted seen free,
   `max(R, T)` long) gives the run. A run under T (the body's length plus C: no room to move), or a corridor not seen empty
   (under 80% of its cells), or an arrival on floor with no route, is **null**. Else with `m = min(run, R)` the outcome is
   `R (geo(base) - geo(base + m u)) / m`, in metres: how much nearer one R of driving along that heading brings the robot.
   (The base's value is `geo` at its cell; the arrival's is `Field.near`.)
5. **The answer** is the heading of the greatest outcome, the one nearest ahead of equals (a plateau, where a metre of
   driving lands in the goal set whatever the heading, is a tie, and every heading on it earns full credit).

The row (`tolerances`) holds `profile_m` (360 outcomes, `null` where there is none), `step_deg` (1), `step_m` (R, 1.0 m, the
unit an outcome is told over), `resolution_m` (0.1 m, the field's fine cell) and `need_m` (C, 0.15 m). The scorer reads
the regret of any answer from the row alone.

**Dead bands** (nothing is asked): the phase is not a driving one; no subject has a recognisable route; no route exists
through floor seen or driven; the robot is within R of the goal set already (`geo(base) < R`: a metre of driving would pass
it); no heading brings it nearer (the best outcome is not above 0); the best outcome is above the geometric bound (final
review S5: a drive of R cannot bring the robot nearer than the field's metric lets it, `route_field.max_outcome`, 1.44 per R
on the helper, and a best outcome beyond it means the eroded floor and the body's corridor disagree there; it dropped 1 of
the 30 rows the compile had counted (the table below is after it), an outcome of 2.99, the others lying between 0.94
and 1.21); the thing is told
(`World.told`, never a subject).

**Constants and their sources.** R = 1.0 m is the re-find radius of the helper's job (HomeRobot's real-world find rule and
Habitat's ObjectNav success radius, `job_helper`); D = R because the heading is judged by where one re-find radius of driving
takes the robot (a design-time estimate: D = T, 0.49 m, would make a constant heading on uniform bearings score about
+0.09, and D = 1 m about -0.10); C = 0.15 m and T = 0.34 + 0.15 = 0.49 m are the job's; the field is the 0.1 m fine cell of `grid` and 8-connected
(octile: a straight heading earns up to 8% less than the best diagonal one, below the 0.1 m resolution of the row); the
erosion is by half the body's width, 0.165 m, read conservatively (any part of a neighbouring cell within it). A gap
narrower than about 0.5 m is closed (the price of reading by whole cells); a door is 0.8 m or more.

## Answerers and references

- **Tables** (`table_way`): the same `route` call over the store's latest cells and the steps' poses up to the ask, toward
  the middle of the latest recognisable sighting of the track the description names (`table_ego.ego_mem`'s choice); `unknown`
  with no such track or no route. `oracle` and `memory_table` reproduce the gold (1.00); `current_view`, whose store holds
  the latest frame, has no track of a thing out of it and gives `unknown`.
- **`path_log`** (`baselines_path_way.own_answer`, through `baselines_path_own.answer`): "turn back the way you came", the
  heading from the base at the ask toward the base pose `LOOK_WINDOW` (8) steps earlier; `unknown` with fewer steps or no
  movement. It stands only where it beats abstaining on other scripts (`baselines_path_own.kept`). It is gated; its in-pool
  twin keeps the rule unasked.
- **Gated**: `blind` (a held-out best constant heading abstains), `form`, `questionnaire_blind`, `other_case` (a fact of
  the script, not the layout: the thing's place differs by script), `path_log`, and on `memory` rows `current_view` and
  `current_view_prior`. `ceiling`, `knowable`, `separation` and the balance rule as for any `memory` family. It is not a
  structure family (no `map_reader` reference: `map_reader` answers `unknown`).
- **The own-log screen** (`heading_screen.told_by_log`, `FamilySpec.own_log_screen`, off): a point where the robot's own log
  already points the best way is a dead band, the declared fallback if `path_log` or its twin fails the shortcut line. It
  was not needed (below).

## Measured

On the dev root's helper (the 0.10.0 dev data, 16 scripts, 36 cases: 16 short, 16 medium, 4 long), compiled with the
0.11 code (`stbench compile`, 12 workers, 95 s for the 36 cases):

| | short and medium (32 cases) | all lengths (36) |
| --- | --- | --- |
| look points | 608 | |
| points that draw `way_to` (slot 2, 35%) | 208 | |
| not a driving phase | 27 | |
| no `ego_mem` subject | 28 | |
| a subject but no route | 31 | |
| a routable subject | 122 | |
| **asked** (the subject rule keeps the end questions' things) | 25 | 29 |
| answerable / `unknown` | 19 / 6 | 22 / 7 |

The compile counted 30 rows over all lengths before the bound above, which dropped one of them (a long case's); the 25 short
and medium rows are the same after it. The design expected about 45 on the short and
medium cases (a fill of 0.25 a drawn slot); 25 is a fill of 0.12. A point
that draws the family and has a routable subject (122) does not always ask it: the end questions draw first and reserve
their things, and a thing is named once a case. A subject has no route when the robot never saw the floor between it and
the thing (110 of 520 subjects at the drawn points), is within R of it (50) or has no walkable floor near it (24); the
other 336 have a route. The field costs 0.03 s a subject (median; 0.05 at the 90th percentile, 0.12 the most, with other
jobs running), so the compile of the helper's 36 cases took 80 to 95 s on 12 workers. Every best heading is a turn: all 19
asked golds are at least 30 degrees from straight ahead, 14 of 19 at 120 degrees or more (the thing was seen a recent
view ago, usually behind the robot), and 167 of the 336 routable subjects have a best heading at least 30 degrees from the
straight bearing to the thing (the way is round something). A constant heading cannot answer it: over the 25 rows the
best constant, 160 degrees, scores -0.32 against abstaining's +0.24 (on the 19 answerable rows alone 150 degrees scores
-0.18, straight back -0.23 and straight ahead -0.63), and the free check's best held-out constant abstains.

The free check on the helper (`stbench check`, `--verdict evidence`, 36 cases; these are the 0.11.0 release's figures, on
29 rows from 11 scripts) passes every gate for
`way_to`: `ceiling` 1.00 (the oracle and `memory_table`), `knowable` +0.00 [0.00, 0.00], `memory_needed` +0.00,
`shortcuts` +0.03 (the largest lead over `all_unknown`, the form adversary's in-pool twin; the line is 0.10; `path_log`
abstains and its in-pool twin scores -0.27), `separation` +0.76 [+0.61, +1.00] (line 0.40) and the balance rule (the held-out
best constant is `unknown`, +0.00).
