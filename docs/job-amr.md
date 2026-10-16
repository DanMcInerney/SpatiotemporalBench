# The tote AMR's job, `amr/1`

A MiR250-class tote robot works a warehouse: it leaves its charger, docks at the tote shelves and the packing tables (or runs
an order-picking loop into a rack aisle and on to a packing table), reverses out of every station it docks at, and comes
back to its charger, with unobserved gaps between tasks through which the warehouse's own processes change the hall. A
station whose bay holds a drum is not docked at: the robot stops at T, is asked whether the bay is clear, waits and drives
on (0.5.0-control; "Balance from the world"). The code is `job_amr.py` (the `SPEC` and the script), `job_amr_plan.py` (constants, the map contract, the plan),
`job_amr_path.py` (a task's primitives), `job_amr_drops.py` (the balance from the world and the skip's wait),
`job_amr_subgoal.py` (what the robot is doing, per keyframe), `job_amr_checks.py` (its `validate` rules) and
`world_processes_warehouse.py` (the gaps). A design-time gate script (not shipped) was the free gate of the first version.

## The map

A map is read by kind, never by name or id (`job_amr_plan.check_map` refuses a map that lacks a piece, naming the map and
the kind). `warehouse_a` is the one it ships with; M4's warehouses keep the contract.

- exactly one `charger` (the job's anchor): 1.6 m clear in front of it and 0.5 m to either side;
- stations, `shelving` (where totes are loaded) and `packing_table` (where they are delivered), each with exactly one `bay`
  piece touching its front face (the zone rule insets a zone by 0.05 m, so a station never blocks its own bay);
- rack aisles: regions of kind `aisle`, lined by `pallet_rack` pieces whose floor-level `rack` surfaces are the pick faces.
  An aisle's mouth is its longest edge shared with a region of another kind, and **a crossing is any boundary between an
  `aisle` region and a region of another kind** (the dock on `warehouse_a`, a pack area on a fulfilment layout), found on
  the polyline, never by an opening (the warehouse has none) or by a name;
- `staging_lane` pieces (keep-outs to the AMR; the places a `slots_empty` question would list, which `amr/1` does not ask),
  none within 0.65 m of a station's T (`T_FACE_M` 1.8-2.4) or of its side legs;
- **stations off the lanes' band**: a station's reverse corridor (the body's width, 5 m behind the docked rear edge) lies
  on open floor, not across a lane, or a rear drop finds no room and the corridor's first thing is a lane's pallet
  (`warehouse_a`: the packing table F09 and its bay F20 at y 6.5, the shelving F11 and its bay F22 at y 6.4 and the
  charger F19 at y 1.4, the lanes' band being y 0.6-3.4; `tests/test_job_amr_drops.py` measures the gap);
- floor markings (lanes, bays) at most 0.005 m high: the AMR's climb is 0.01 m and the depth's millimetre rounding lifts
  paint, so a 0.01 m marking reads as an obstacle in every frame.

Any other region is open floor to transit.

## The tasks

`replenish` brings totes from a shelving piece to one or two packing tables; `collect` is the reverse (the same stations,
the tables in the other order, then the shelving), the brief swap's partner; `pick` is an order-picking run: one to two pick
faces of one aisle, in order from its mouth, then one packing table. The kinds are drawn 0.35, 0.25 and 0.40 from
`make_rng(map, seed, "amr/1", "job")`; every task draws the same numbers in the same order whatever its kind, so the swap
pair uses one station set. A task's `params` hold piece ids (the job's, never delivered) and the region kinds it works in;
no assignment names one. **The swap pair's streams are not identical**: a collect visits a replenish's stations in the
other order and `loaded` flips at the other end, so under the swapped brief the route contradicts the words. A pick has
no partner and refuses the swapped brief, so the brief swap runs on the windows whose swapped brief can be composed: a
window holding a `pick` task is left out of every arm of the swap, and the arms run on the same cases.

The status is the phase, `battery_pct` (drawn 60-100 at each task's start, draining 0.15% a minute: a MiR250 runs about
13 h) and `loaded`: false at the start, true from the end of the first hold the robot makes (the shelving of a replenish,
the first table of a collect, the first pick face; the next station when that one was skipped) until the end of the last
planned stop's hold, if the robot docked there; it changes only at the end of an `at_station` hold, so the last keyframe
of a hold still has the old value. A skipped last stop makes no hold, so the robot ends the task still carrying what it
loaded (`loaded` true at the closing hold), and a task that skips every stop never loads.

## The manoeuvres

| Phase | What the base does | Decision |
| --- | --- | --- |
| `waiting` | docked on the charger, head-on (the front edge on the charger's face, which an anchor allows), 10 s | |
| `undocking` | reverses 1.0 m off the charger, at 0.2-0.3 m/s | |
| `transit` | turns in place and drives (A*, 0.8-1.2 m/s) to the arrival waypoint W, 2.0 m from T along a station's face on the side nearer the robot (then the other side, then 1.5 m and 1.0 m, when not free), and along the face to T; turns to face the station | `before_station` at T, facing the station; its `response` is "go" or "skip" |
| `waiting` | **a skipped visit only:** holds at T, facing the station, 10-20 s (the bay holds a drum), then drives on from T in `transit` | |
| `approach` | straight from T to the docked pose (the front edge 0.12 m from the face, centred on the bay), 0.2 m/s | |
| `at_station` | a dock's transfer or a pick face's wait for the picker, 15-40 s | `before_undock` on the hold's last keyframe (a dock only) |
| `undocking` | reverses along the same line to T with the heading kept | |
| `transit`, `approach`, `waiting` | back to the point 1.0 m in front of the charger, a turn to face it, a forward approach and a hold | |

T is the base 1.9 m from the station's face (`T_FACE_M`, tuned within 1.8-2.4 below), so the front edge stands 1.5 m from it. The arrival comes in along
the face and crosses the reverse corridor (the body's width, 5 m behind the docked rear edge) only at T: what is behind a
station is then seen on the way in. M2's vacuum drove in along its corridor and failed (docs/jobs.md, "Back-outs").

A **skipped visit** (the bay holds a drum across the docking line) has the `transit`, the `before_station` and then a
`waiting` hold at T; there is no `approach`, `at_station`, `before_undock` or `undocking`, and the next leg starts at
T, as it does after an undock. The phases of a task are therefore `waiting`, `undocking`, then for each station
`transit, approach, at_station, undocking` or `transit, waiting`, then `transit, approach, waiting`. A task that skips
every station has no `at_station` hold; two of the AMR's events (the first and the last stop) then do not resolve in it,
as an absent moment (`World.events`).

A pick run goes to the aisle (a `before_crossing` 0.5 m before the way crosses into the aisle's region, facing the
boundary, or at its segment's start when that is nearer), to each pick pose (the heading along the aisle, the body's side
0.35 m from the rack's face, centred on the pick face, an `at_station` hold with no decision), turns, goes out (a
`before_crossing` before the way crosses out) and docks at a packing table as a station.

**Keep-outs.** A task is planned on `route.robot_free_space(..., keep_out=...)`: every staging lane's footprint, loaded or
not (with the lanes' pallets the only obstacles, A* between the west and east stations would cut across empty lanes and
round loaded ones, and the robot's own path log would read each lane's load, the answer `slots_empty` asks for), every
bay from the station's face to 0.7 m short of T (so no way crosses a bay, whatever a drum in it says, and T and the leg
along the face stay free) and, behind each station, the reverse corridor's extension (so nothing sweeps a place a rear drop
may stand in). The docking, the approach and the reverse are placed by hand and never touch the planner.

## Balance from the world

A constant answer must earn nothing, so the world varies by coins that read no world (`job_amr_drops.py`), every move at the
task's start less 30 s to the whole second, after the gap's changes, never onto a lane or off one.

- **The bay coin.** For visit j of task i, `make_rng(map, seed, "amr/1", "bay", i, j).random() < 0.5` says the visit's bay
  is blocked, else clear; the draw reads no world. Before the task is planned every thing standing in a bay it visits
  moves out (`move_item`). Once the free space is planned (so the drum's move cannot show in it), each blocked visit gets
  one of the world's drums (0.88 m tall, far over the zone's 0.3 m) moved into its bay **across the docking line**, its
  centre within 0.10 m of the line, wholly inside the bay's keep-out (T - 0.7 m deep): the keep-out is what the planner
  already keeps out of, so a drum there changes no free cell and the path to the `before_station` is the same whatever
  the coins say. A tote (0.32 m) is never used: its top is 2 cm over the threshold. The world holds two drums off the
  lanes, so a task blocks at most two of its bays; a visit whose coin says blocked and that finds no drum (or no spot) is
  not blocked, and docks.
- **The response.** The coin is enforced in the gap, so no change is seen happening; the response comes after the
  decision and follows the bay's state. A visit whose bay holds a drum has `Mark(BEFORE_STATION)` and then `Hold(T,
  heading, s, "waiting")`, with s from `make_rng(map, seed, "amr/1", "hold", i, j).uniform(10, 20)` (a stream of its own,
  so no other draw moves with it; the frame policy may end the hold up to 1 s after s), then the next leg from T: no
  approach, `at_station`, `before_undock` or undocking. The decision's `response` is "skip"; every other
  `before_station` says "go". The look window a `zone_clear` question reads ends at the decision, so it never holds the
  response. A job check holds the response to the world (below).
- **The rear drop.** With 0.6, after the task is planned, a pallet or a drum is moved (`clutter.floor_spot`, which keeps the
  floor connected) into the reverse corridor's extension behind a docked visit, from T's rear edge plus 0.3 m to the
  corridor's 5 m end, clear of every body the task sweeps by 0.25 m. A skipped visit has no corridor to reverse along and
  gets none (its coin is not read; the other visits keep their indices).

## The warehouse's processes

`world_processes_warehouse.py`: every change of a gap at one whole second, 30 s inside it; each thing re-drawn by the start's
law at a half-life rate (`half_life_p`): pallets 2 h, totes, bins and boxes 3 h, drums and sacks 8 h (each through
`move_item`), the pallet jacks and the roll cage 1 h (through `generate.light_start`, off the charger's apron). No thing is
added or removed. The numbers are modelling choices (no published rate fits a hall this small); `SOURCES` says so.
The pickers (`people_warehouse`, three a script) are an overlay on the finished timeline, not a process of the gaps: they
keep off the bays (so the coin stays the bay's state) and 0.3 m off the robot's body, and the gate's depth ray-casts
each where it stands at the frame's time.

## Checks

`job_amr_checks.py` (codes `job_response`, `job_phase_order`, `job_charger`, `job_loaded`, `job_decisions`, `job_targets`,
`job_task_time`, `job_gap`, `job_span`): every `before_station` has the response "go" or "skip", no other decision has one,
and the response is "skip" exactly when a carried, light or clutter thing taller than the zone's 0.3 m stands on the floor
over the bay's zone at the decision's time (the robot answers the world); the phases of a task are those its plan and its
responses give (`transit, waiting` for a skipped station); its first and last keyframes are docked on the charger;
`loaded` follows the rule above (true after the first hold made, false after the last planned stop's if it docked, so
true at the end after a skipped last stop); the decisions are the plan's (a `before_undock` only for a docked visit);
every `before_station` stands at its station's T facing it, the bay nearest its base is that station's; every
`before_undock` is the last `at_station` keyframe of a docked visit; every `before_crossing` lies in transit at most
0.5 m before an aisle's boundary; a task takes at most 900 s; the gaps lie in 10 minutes to 3 hours and the script
spans at most 7 hours.

## The subgoal

`job_amr_subgoal.subgoals(tl, m)` (`JobSpec.subgoals`) gives the text of what the robot is doing at each keyframe, as part of
its task, from the timeline alone. It is the robot's own knowledge: stations by the map's furniture names (an ordinal
when a name repeats on the map: `shelving unit 1`, `packing table 2`), an aisle by its region's name, never an id. It is
**causal**: one forward scan whose state is the phases and decisions of the keyframes so far; a keyframe's text is written
before its own decision is counted, so no text depends on a coin decided at or after it, and the response to a
`before_station` shows only in what follows it. The texts, by what the robot is doing:

| Robot | Text |
| --- | --- |
| at the charger, undocking from it | `leaving the charger` |
| `transit` to the n-th of N stations | `driving to {station} to {pick up totes \| drop off totes \| collect empty totes \| return the empty totes} (stop n of N)` |
| `transit` to a pick face | `driving into the {aisle} to pick (pick i of n)` |
| at a pick face | `waiting at the pick face while it is loaded (pick i of n)` |
| `approach`, `at_station`, `undocking` at a station | `docking at {station}`, `{loading \| unloading} at {station}`, `reversing out of {station}` |
| `waiting` at T (after a skip) | `waiting at {station}: its bay is blocked` |
| the `transit` after a skip | `skipping {station}; ` before the leg's own text (`driving to ...` or `returning to the charger`) |
| `transit`, `approach` home; the closing hold | `returning to the charger`; `charging` |

The purpose is the task kind's: a replenish picks up totes at the shelving and drops them off at its tables, a collect
collects the empty totes at the tables and returns them at the shelving, and a pick's table is a drop-off. Texts repeat
across scripts (the same station, the same stop), so `path_log` forms pools on them.

## What the AMR asks

**0.11.0** (`job_amr.py`). Slot 3 gives half of its points to a new family, `approach_mem`: "If you kept
moving as you are now, how close would you come in the next 5 seconds to the person you saw last, where you last saw them?"
([families/approach_mem.md](families/approach_mem.md)). It is `closest_approach` for the person now out of view, the AMR's
blind aisle end: a picker seen 20 s ago and hidden by the rack now, whom the planner keeps as an obstacle. The slot is now
`ego_now` 50 and `approach_mem` 50; slots 1 and 2 are unchanged and whole. The half comes from `ego_now`, the warehouse's
most-asked during family (astra +0.71, sol +0.59), whose mean a halving does not move (105 questions on the dev root fall to
about 52). The supply is large: a person within 10 m who was seen recently and is out of view is 41% of the keyframes,
against about 10% for someone in view. The family is a harm question of the clearance ability in its `memory` cell, next to
`closest_person` and `closest_approach`'s walking rows; a static-layout system has no sighting to answer from and scores
at the floor. **Measured** (the 0.11 code on the dev root's 36 AMR cases, scoped; [families/approach_mem.md](families/approach_mem.md)
has the detail): the AMR asks 245 questions, of which `approach_mem` 47 (in 28 cases; the design expected 60-100), `ego_now`
65 (105 before; the design expected about 52), `closest_approach` 24, `closest_person` 34, `ego_mem` 28 and `task_things` 47.
Every `approach_mem` row is a person (no "no one": the proxy's 22% did not appear, so the keep stays at its declared
100), with a median age of 5.3 s and a median gold of 3.28 m. The family passes every gate on the AMR in the evidence
verdict: `ceiling` +1.00, `knowable` +0.00, `memory_needed` +0.00 (`current_view` answers 10 and scores -0.86),
`shortcuts` +0.00, `separation` +1.00, and the balance rule. The release's clearance cells hold 22 questions in `now`
and 83 in `memory`.

**0.10.0** (`job_amr.py`; the reasoning is [question-design.md](question-design.md)). A look point has three slots, each
whole (100% of points): `closest_approach` (slot 1: "If you kept moving as you are now, how close would the person nearest
you come to you in the next 5 seconds?", [families/closest_approach.md](families/closest_approach.md)), `ego_mem` (slot 2)
and `ego_now` (slot 3); at its end `task_things` and `closest_person` as before. The frames of `before_crossing` and
`before_station` come first in the look draw; `before_undock` has no supply. `ego_mem` and `ego_now` are asked only of
things on the floor (stock on a rack is no obstacle to a tote robot); the filter cut `ego_now` by 26% and `ego_mem` by 44%.
It still asks no `clear_path` and no `open_heading`.
- **`closest_approach` after lane J2** and the final review's repair. Validity fixes: a person's motion counts only when
  the sensors pair it, which they do for exactly one evidence person in the ask's frame and in the earlier frame used (a
  detection's key names nothing, so a frame with two people is a dead band). It bites on the AMR, whose warehouse holds
  several people (the rule removed 186 of its 530 `person` rows and none of the helper's): the first wiring's gold
  paired a walker's two sightings by the truth's identity, so the oracle led `memory_table` by +0.119 on `knowable` (the
  bar is 0.02). The family now passes every gate on the AMR (24 questions from 14 scripts, 18 with a person and 6 "no
  one"). Lane J2 also tagged a row with a person `memory`, since the robot's velocity then needed the frame before the ask;
  the final review's S3 delivers that velocity with every step (`velocity`, the controller's odometry), so a row whose person
  stands is `now` again (16 of the 18) and only the 2 rows with a walking person are `memory`; the question says "the
  person you can see nearest you".
- **Measured** on the dev root after lane J2 (short and medium cases): 804 look points, 127 look questions
  (`closest_approach` 19, `ego_mem` 22, `ego_now` 86) and 66 end questions, one question in 37.5 frames of look questions and
  24.7 of all (0.9.0: 1 in 35.8); harm is 25% of the questions. The AMR is supply-limited: its slots are whole already, a
  person is in range at few of the draws of `closest_approach` (a fill of 0.023, 24 questions in 1,043 draws, all lengths),
  and the floor holds few things for `ego_mem` to name twice. The family page holds the counts.
- **The AMR's clearance cells after the repair.** The AMR asks no `clear_path`. Its clearance `now` cell held only the "no
  one" rows (6 questions, thin) while every row with a person was `memory`; with the robot's velocity delivered it holds 22
  (6 "no one" and 16 still-person rows) and counts, so the AMR's `now` column is distance (`ego_now`, 105 questions) and
  clearance. Its clearance `memory` cell holds 36 (the 2 walker rows and 34 `closest_person`) and counts
  ([families/closest_approach.md](families/closest_approach.md), "Scope").

**0.9.0 (history).** The table is 0.8.0's: `ego_now` at 70% (slot 1), `ego_mem` at 35% (slot 2), and at its end
`task_things` and `closest_person`. Only the credit changes ([questions.md](questions.md), "Credit"): every
answer earns its four quarters against C = 0.45 m (a list against its 1 m match radius, per item), a closest approach
answered wider than it was counts against the answer twice as fast, and `ego_now`'s resolution is the largest of the
three jobs' (median `rho / N` 0.21, q90 0.55). It asks no `clear_path` in any direction (ahead and back are answered by a
constant, and to the sides 44% and 40% of the golds are the 5 m cap and the map reader leads by +0.53 and +0.80) and no
`open_heading` (every best run is the cap), so `JobSpec.clear_directions` is unused.

**0.8.0** (task-critical questions: harder questions, not more of them). During the job the
AMR asks `ego_now` at 70% of its look points (slot 1) and `ego_mem` at 35% (slot 2); at its end `task_things`
and `closest_person` ([families/closest_person.md](families/closest_person.md)): "During your first task, how close did
a person come to you, as you saw them?", credited on C = 0.45 m. A warehouse robot must be aware of the people on its
floor (ISO 3691-4 requires it to stop before contact with a person in its path), and the near-miss record is what a
fleet keeps; the three pickers share one body, so the question asks for "a person". One question a case, about the first
task of the window (in hash order) that passes its screens. `object_where` left the AMR's end to make room.

It asks **no `clear_path`**: under the stopper screen 80 of 145 in-transit golds ahead are the 5 m cap, a held-out
constant scores +0.33 on the 26 rows the budget allows, the map reader +0.85, and while the robot stands at a station or
waits the status answers it, so `JobSpec.clear_phases` is empty. Its back corridor is not asked either (`rear_clear`'s
history above; the AMR's constant scored +0.29). The prompt now states the body (a box 80 cm long and 58 cm wide, 65 cm tall).

**0.7.0** (the 0.6.0 mapping is unchanged). No decision kind asks a family:
`zone_clear` and `out_of_view` are retired, and the three decision points, the bay coin, the skip and the drum stay in
the path, so the scripts and renders are unchanged. The bay's blocker is told by the robot's own log (the response
"skip" and the subgoal), so `World.told` keys on the job's `zones` and no question names it. During the job the AMR
asks at look points (`JobSpec.looks`): `ego_now` ("where is the thing you can see, from your camera") and `ego_mem`
("as you last saw it, where is the thing, from your camera now"); its brief names the two kinds in one sentence. At its
end it asks `object_where` and `task_things` (`adjacent` is retired; `object_was` and `same_place` are not asked since
0.6.0: 0.42 a medium case, under the supply line), the last with the
predicate "within 2 m of the way you drove": the things within 2 m of the base's path over all of the task's steps (every
phase, `waiting` included, to the task's end: a final stop that was blocked and skipped is still driven to, and the
drive there is part of the way; 0.6.0 lane R3), a don't-care band of 0.3 m about that distance, and light
furniture and people don't-care. The predicate drops `floor` (it was 1.06 items a task and 35% empty with it; the stock
the racks hold off the floor stays don't-care), and `empty_keep_percent` is 43: 22 of the 41 tasks of release-c's short
and medium cases are empty after `told` and `RELAX` (p 0.537), and 100 (1 - p) / (2 p) = 43. Its offsets are credited on the clearance C = 0.45 m (the MiR250's corridor margin) and
its places on the re-find radius R = 1 m, and a list of places matches within 1 m ([questions.md](questions.md),
"Credit"). The sections below are the 0.5.0-control job as built.

**0.5.0-control**. During the job:
- `before_station` asks `zone_clear`: "is the bay ahead of you clear", a `now` question over the bay's zone (the bay
  piece inset by 0.05 m, taller than 0.3 m), the gold read from the decision's look window alone. The bay is blocked or clear by
  the coin above, so the answer varies from case to case, the place cannot predict it, and the robot's response (skip or
  dock) follows it after the ask. The family had failed the shortcut gate in 0.4 because its `memory` rows were all clear
  bays; a now-only gold has no memory rows.
- `before_undock` and `before_crossing` ask `out_of_view` (a `memory` question, with the rack stock made don't-care
  places). `before_undock` stands only before a docked visit.
- No `rear_clear` (the rear drop lands from T's rear edge plus 0.3 m, so the reverse of 1.5 m is clear by construction and
  no decision rides on it), no `path_ahead` and no `slots_empty` (the reasons below).

At its end: `object_where` ("where did you last see the X", the critical, unmoved things), `object_was` (the things that
moved, with the event anchors; the events are those of `JobSpec.events`) and `adjacent` (held-out layouts only; inert on
the one map). `distance_between` left every job. The distance ladder is (0.5, 1, 2, 4) m: the warehouse's sight lines
are 5 to 10 m.

The events: "when you left the charger" is the task's first `transit` step; "when you reached your first stop" and "when you
finished at your last stop" its first and last `at_station` step (a station that was skipped is not a stop; a task that
skips them all has neither); "when you got back to the charger" its last `waiting` step, the end of the closing hold. (A
skip holds at T in `waiting`, so the 0.4 rule, the first `waiting` after a `transit`, would have named the wait at a
blocked bay.)

## The gate (the first version's)

This section is the free gate of the first version of the job (0.4), and the mapping it chose is superseded by the
section above.

The design-time gate script (not shipped; run on `warehouse_a`, seeds 101-130, with generation and `validate` alone over
seeds 101-200 for (a)) is the free gate, run on the merged code: scripts of the job as built from the pipeline
(`build.generate_checked`: the bay coin, the rear drops, the warehouse's processes, the three pickers a script and
`validate`), one-task windows, a depth image ray-cast at the robot's camera for every keyframe (millimetres; the boxes of
the walls, the fixed, light and carried things at the task's start and each picker where it stands at the frame's time),
and each gold asked of the compiler's own `World` over `grid.observe` of that depth: `questions_zone.zone_clear` at
`before_station`, `questions_space.rear_clear` and `path_ahead` at `before_undock` (the delivered pose rows, the zone
rule of `zones.py`), and `questions_zone.slots_empty` for the record (the gate gives the job the lanes' rule; `amr/1`
asks none). Each shortcut's expected score (`blind`, `path_log`'s best constant per station and its map raycast, the form
adversary on `slot:view`, `phase` and `k_in_task`) is set against `all_unknown` + 0.10; `path_log`'s raycast is the walls
and the blocking fixed pieces of at least 0.05 m. The script could set `job_amr_path.T_FACE_M` to try another T (the path
module holds the copy of the plan's constant that docks and keep-outs read).

**As built** (`T_FACE_M` 1.9, the open-corridor `warehouse_a`; seeds 101-130: 30 scripts, 153 tasks, 279 docked visits):

| Item | Measure | Needs | Holds |
| --- | --- | --- | --- |
| (a) | worlds that fit (generated and validated), seeds 101-200 | all | 100 of 100 |
| (b) | reverse corridor swept before the decision, median | at most 0.4 | 0.363 |
| (e) | reverse corridor observed before the decision, median | at least 0.8 | 1.000 |
| (r) | `rear_clear`: drops realised / wanted | within 0.1 of `REAR_DROP_P` | 0.520 / 0.609 (F09 62/94, F10 43/99, F11 26/47, F12 14/39) |
| (r) | unknown golds; rungs of the answered (0.25 m, cap 5 m) | | 0.036; 13, the cap holding 0.38 |
| (r) | `blind`; `path_log` (best constant per station; its raycast); form, against `all_unknown` + 0.10 | at most +0.10 | +0.000; -0.093 (-0.125); +0.000 |
| (f) | `path_ahead` at the docked pose: unknown share, the answered golds; `blind`, `path_log`, form | | 0.412, every one 0; +0.000, +0.018, +0.050 |
| (v) | bay cells observed in the `before_station` look window, median | at least 0.8 | 0.866 |
| (z) | `zone_clear`: no / yes / unknown (no of the answered) | no 0.45-0.55 of the answered | 0.480 / 0.520 / 0.000 (0.480) |
| (z) | coins enforced; `blind`, `path_log`, form against `all_unknown` + 0.10 | all; at most +0.10 | 278 of 279 (one blocked bay found no drum); +0.039, -0.018, +0.039 |
| (s) | `slots_empty`: empty-gold share; lanes passed a task | at most 0.52 | 0.144; 3.77 of 4 |
| (s) | every shortcut against `all_unknown` + 0.10 | at most +0.10 | **+0.278** |

At T = 2.0, the value the first gate chose, the same gate gave (b) 0.410 against its bound of 0.4 and the rest within the
bounds ((v) 0.824, (r) 0.534 realised, (z) shortcuts -0.082, -0.018 and +0.039), so T = 1.9, inside the 1.8-2.4 tuning range
and read on measure seeds, is the one that meets every bound that can be met.

What holds and what does not, and why:
- **(b)** is the measure T moves. The approach sweeps the corridor from the docked rear edge to T's rear edge (T - 0.52 =
  1.38 m of the 5 m), so (b) is at most 0.4 only if the median counted corridor is at least 3.45 m long (1.38 / 0.4): a
  rear drop (0.520 of the visits) or a picker within 3.45 m behind a station puts it over. At T = 2.0 the need was 3.7 m
  and (b) was 0.410 on the merged code (0.386 on the first gate, which had a stand-in zone rule and no pickers).
- **The open corridors work.** With the stations in the lanes' band only 8 of 94 and 14 of 47 visits at the first packing
  table and the first shelving found room for a drop (a corridor over a lane has none, and the roll cage stood in the gap
  behind F09); with F09, F11 and the charger off the band 62 of 94 and 26 of 47 do, and every corridor lies 0.41 m or more
  north of a lane (`tests/test_job_amr_drops.py`). F12's 14 of 39 is the thin one; the cause was not isolated.
- **(v) and (b) pulled T apart on the first gate** (nearer to sweep less, farther to see the bay); on the merged gate both
  improve from 2.0 to 1.9 ((b) 0.410 to 0.363, (v) 0.824 to 0.866), and T below 1.9 is untried on it. The first gate, on the
  map with the stations in the band and its stand-in rule (seeds 101-130): T 2.0 gave (b) 0.410 and (v) 0.795, 2.1 gave 0.437 and 0.807, 2.2 gave
  0.472 and 0.818 with (z) failing at +0.151 (the golds were `unknown` only in the `now` view, which then told the answer),
  and 1.8 (4 seeds) gave 0.366 and 0.722. On the open map the first gate gave, at T 2.0, (b) 0.386, (v) 0.824, 0.508
  realised against 0.577 wanted and shortcuts at most +0.085, and at 2.1 (b) 0.404 with `path_log`'s constant at +0.129,
  so T = 2.0 went with the map. A T below 2.1 with a drum standing in the next bay's outer end closed a station's
  arrival in 2 of 100 worlds (0.45 m from the leg, the planner keeping 0.61 m), so a drum stands inside its bay's keep-out
  (`job_amr_path.bay_keep_out`, T - 0.7 m deep), which keeps every leg free.
- **(z)'s `view`** is decided by coverage (zones-C), so the wording must not predict the answer: `now` holds 86 blocked and
  92 clear, `last` 48 and 53 (178 and 101 visits), and `zone_clear` is never `unknown` on these scripts. No picker blocks a
  bay: every one of the 144 clear coins has a clear gold, and 134 of the 135 blocked ones a blocked gold (the other found no
  drum). The scope is `now` in 183 visits, `memory` in 96.
- **(f)** fails in effect: the station's face is 0.12 m ahead of every undock, so each answered gold is 0 and the family
  carries only whether the corridor was seen (0.412 `unknown`: `path_ahead` reads the decision's look window alone). Its
  shortcuts stay under the limit (+0.050 at most), but a gold that is 0 whenever it is answered asks nothing. The first
  gate, which read the task up to the decision, gave +1.000.
- **(s)** fails the shortcut test by a wide margin: the lanes hold few things (a pallet's receptacles are racks and lanes
  by area, 88% racks), so the best constant list (the lanes most often last seen clear: F08 0.67, F05 0.44, F06 0.24 and F07
  0.05) scores +0.278 above `all_unknown`. Its empty-gold share is within M2's rule.

The mapping, as first decided from the free gate: `before_station` asks `("where_was_i", "out_of_view", "zone_clear")`
((v) and (z) hold) and `before_undock` asks `("where_was_i", "out_of_view", "rear_clear")` ((a), (b), (e) and (r) hold);
`before_crossing` keeps the two M2 families; no decision asks `path_ahead`; and `ends` does not gain `slots_empty`, so the
job holds no slot rule (`slot_zones` is empty) and the lanes are keep-outs only.

**What the AMR asked in 0.4** (the release gate check on the first six scripts, seeds 2001-2006, 18 cases of three
lengths). The release is 12 AMR scripts (seeds 2001-2012, 36 cases: short,
medium and long), and these six ship in them. During the job, at `before_station`, `before_undock` and
`before_crossing`: `where_was_i` and `out_of_view`. At its end: `object_where` and `distance_between` (and `adjacent`,
on held-out layouts only; the first draft has none). Those four hold: at the six-script check `object_where` (88
questions, separation +0.78), `distance_between` (26, +0.85), `where_was_i` (32, +1.00) and `out_of_view` (31, +1.00)
passed, and they pass the binding gate checks (0.4.0's over its short and medium cases, 0.4.1's over all its cases) with
no BROKEN line. `when_seen` (28 questions, +0.64) passed the six-script check too and stayed in the job's `ends` until
the binding check, where its 39 questions pass their gates but `questionnaire_blind` is BROKEN (0.154 against
abstaining's 0.256), a line the export refuses; with the long cases in (64 questions) it was not. It is dropped, so no
0.4.0 or 0.4.1 job asks `when_seen`. The rest were dropped by the six-script check:
- `zone_clear` (12 questions) failed shortcuts and separation: `blind` scored +0.67, because after the scope fix its
  `memory` rows are clear bays ([families/zone_clear.md](families/zone_clear.md), "Balance", foresaw it). `rear_clear`
  (12) failed `memory_needed`: a learned shortcut under the pool's own winner scored +0.083, one question's margin. Both
  left `before_station` and `before_undock`, which now ask `where_was_i` and `out_of_view` like `before_crossing`.
- `changed` (2 questions in 18 cases; its oracle 0.60, an open instrument question for a later release) and `order` (4;
  `other_case` +0.25) left `ends`.

With the drone's `zone_clear` also dropped (for good, after a re-read over its 12 scripts), the first draft has no `now`
question. The job's `zones` rule for the bay stays: a rule no family reads is inert.

**Sizes** (seeds 101-130 on the 0.5.0-control job: 30 scripts of 4-6 tasks, 153 tasks): a script holds 309 to 595
keyframes (median 456); the longest task takes 237 s of the 900 s allowed. Keyframes and minutes by kind
(minimum/median/maximum; median minutes): replenish (54 tasks) 80/97/116, 2.9; collect (32) 76/93/116, 2.8; pick (67)
51/84/113, 2.8. Decision points a task: replenish 3.76, collect 3.69, pick 3.54 (a skipped visit has no
`before_undock`). A task's first decision stands at keyframe 14 to 41: a replenish's `before_station` at 35-41, a
collect's at 14-15 (its first table is the near one) and a pick's `before_crossing` at 14-29. The questionnaire asks a
kind of memory families (`before_undock`, `before_crossing`) only from `GAP_STEPS` (16) keyframes into the task, and a
kind of `now` families (`before_station`: `zone_clear`) from `NOW_MIN_STEPS` (4). Eligible points a task, of a kind that
asks a family: `before_station` 1.82 (every one of them; 1.61 under the 16-step rule it had until the questionnaire
let `now` kinds in early), `before_undock` 0.95 and `before_crossing` 0.76.

**The coins and the responses** (the same 30 scripts; the free generation check, no render). Every one generated and
`validate`d clean (0 of 30 skipped). Of the 279 `before_station` decisions 134 say "skip" (0.480) and 145 "go"; the coin
said blocked at 135 (0.484), so one blocked bay found no drum and docked. The waits at T run 11.0 to 19.9 s. A replenish or
collect skips none of its stations in 15 of 86, some in 55 and all of them in 16 (every stop skipped, so the task has no
`at_station` hold: 8 of each); a pick skips its table in 31 of 67. 53 tasks end loaded (a skipped last stop after a hold
made). The subgoal has 80 distinct texts over the 30 scripts and none carries an id, a region id or the map's name.

The processes' realised shares over the gaps (the first version's scripts of these seeds; moved things / things; the
half-life draw's expectation): pallets 0.29 (0.27), totes 0.20 (0.19), bins 0.19 (0.19), boxes 0.19 (0.19), drums 0.07 (0.08), sacks 0.07
(0.08); the pallet jacks and the roll cage 0.18 (0.44): `light_start` finds no place for most of the draws of a piece 1.55 m
long between lanes 2 m apart, and the piece stays.
