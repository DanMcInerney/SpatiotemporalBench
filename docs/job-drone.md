# The delivery drone's job, `drone/1`

A Wing-class tether drone on `downtown_a` (`job_drone.py`, M3). A script is one to six flights: the drone leaves its nest
in the square, flies the streets at 15 m to a drop point, inspects it from above, descends to 7 m, lowers the package on
its tether, releases it, climbs out and flies home. An *inspection* flight is the same flight with the lowering not made.
A delivery over a drop zone that holds something does not lower: it **aborts** (below) and flies the inspection's hover.
Between flights the drone is home and unobserved, and the downtown changes. Its numbers are `robots.py`'s (M0's: a 1.30 m
disc keeping 3 m from anything it cannot fly over, 8-12 m/s approach, 1-2 m/s descent, 0.5-1.0 m/s on the tether, 3-5
m/s climb, a gimbal from -90 to -20 degrees) and the design's ("approach about 15 m; lowering from about 7 m"). The
drone's body, tether and package are not drawn: the camera looks down, so a real drone's frames would show only the
package, which no question reads.

**One departure from a real Wing flight**: the nest is in the district, so there is no 45 m cruise (a Wing cruise
would put roofs and the horizon in view, which the downtown does not have). Every observed keyframe stays
below the lowest roof (18.9 m) and inside the street canyons.

## What the job reads from a map

By kind only (M4's suburb and courtyards keep it): exactly one `nest` (the job's anchor: a pad under 0.05 m
on bare ground, nothing within 3.65 m of its centre); `drop_zone` pieces (markings on their ground, centres at least
3.65 m from every wall; `name` is the address a brief names); open air over the regions. Walls obstruct a flier at any
height (`validate_ego`'s body rule); fixed pieces lower than the base's height less 3 m are flown over
(`robots.flies_over`). A map without a nest or a drop zone refuses the job (`ValueError`).

## One flight

| Phase | What the drone does | Pitch | Decision |
| --- | --- | --- | --- |
| `preflight` | holds on the pad, base 0.27 m up (the pad's 0.02 m and the landing gear's 0.25 m), 6 s | -20 | |
| `takeoff` | the gimbal to -45, then straight up to 15 m | -45 | |
| `approach` | A* over the streets at 15 m to above the drop zone's centre | -45 | |
| `inspect` | the gimbal to -90, then hovers 6-12 s | -90 | `before_descent`, on the last frame |
| `descend` | straight down to 7 m above the zone's ground | -90 | `before_lower`, on the last frame; asks `zone_clear` |
| `lower` | hovers while the tether pays out to the ground at the lowering speed | -90 | |
| `release` | hovers 2 s | -90 | |
| `climb_out` | reels the tether in, then straight up to 15 m | -90 | `before_return`, on the last frame |
| `return` | the gimbal to -45, turns, and flies the way back | -45 | |
| `land` | the gimbal to -90, then straight down onto the pad | -90 | |

The status is the phase, `battery_pct` (an integer in 95-100 at each flight's start, draining 2.5% a minute) and
`tether_m` (0 but while lowering, releasing and reeling in). Speeds are drawn once a flight from the robot's ranges (the
takeoff and the climb out at `climb`, approach and return at `approach`, descent and landing at `descend`, the tether at
`lower`; `cruise` is never used). A flight to the square takes about 75 s (62-88 s on seeds 101-130) and 40-50 keyframes; one to a street zone up to about 105 s and 65.

**The way** is the map's alone. A* on a 0.5 m grid of the map's own (`job_drone_path.flight_free`), the walls grown by
the body's radius, its 3 m clearance and a cell's half-diagonal, every fixed piece below 12 m flown over, the polyline
smoothed to chords of free cells; the way back is the same polyline reversed. Nothing the world moves is in it (everything
it moves is under 3.3 m), so no flight's path tells of the world. A vertex is a turn in place.

**The frames** are `keyframes.frames` under the robot's policy (a frame each 3 m, 30 degrees of heading or of camera
pitch, 3 s, none closer than 1 s). `keyframes` takes a base height and a camera pitch for a flier (`Move.z`, `Tilt`,
`Frame.z` and `.pitch`; travel counts the 3-D length); every other robot's primitives name neither and sample as they
always did. A `Tilt` belongs to the phase it prepares the pitch of, and no tilt (0.4-0.75 s) outlasts the 1 s between
frames, so every keyframe's pitch is its phase's.

**The swap pair is one flight.** An inspection flies a delivery's flight to the byte in its poses, keyframe times and
decision points (`before_lower` included). Over the hover at 7 m its phase is `inspect` where a delivery's is `lower`,
`release` and the reel-in of `climb_out`, and `tether_m` stays 0; both kinds put a `Mark()` at the end of the lowering,
of the release and of the reel-in, so every required frame, and with it every keyframe time, is the same. That status is
the pair's only difference, recorded for the brief swap. A delivery that aborts flies the inspection's hover the same
way, so it is the inspection's flight to the byte in everything but its response.

## The plan, the events and the relevance

`plan_flights` draws from `make_rng(map, seed, "drone/1", "job")` and the map alone: each flight is an inspection with
probability 0.25, else a delivery, to a drop zone drawn uniformly among the zones the world can set (below). `params`
hold `address` (the zone's name: the only key a brief reads), `zone` (its piece id), the zone's centre as `zone_x` and
`zone_y` (the point the near rule below is measured from) and the region kinds of the zone and the nest (for
`relevance`). Briefs: "Its current assignment is to deliver a package to the drop point at Square West." and
"... to inspect the drop point at ...". The gaps are `job_gaps.gaps` (10 minutes to 3 hours, a span of at most 7 hours),
which validates nothing: the job's checks do (`job_gap`, `job_span`).

Events: "when you took off" (the first `takeoff` step), "when you started lowering the package" (the first `lower`), "when
you released the package" (the first `release`) and "when you landed" (the last `land`); an inspection, and a delivery
that aborts, resolves the first and last only. No family asks about an event any more (the drone asks nothing at its
end), so the events are inert. Relevance: every downtown kind in the drop zone's region and the
nest's, the phases approach, inspect, descend, lower, release and return, and **within 12 m of the drop zone's centre**
(`NEAR_M`, `Relevance.near`): the ground half-width the camera sees from the 15 m inspection height (15 m x tan 40
degrees, half the 80 degree horizontal field, is 12.6 m, rounded down). A thing farther off is situational, never asked.
`before_departure` is not a decision of this job (at the nest it falls in the task's first two look windows, where nothing is
asked). `changed` and `out_of_view` are never asked: both read the robot band's free space and the floor at the robot's
base, which for a flier is in the air.

**The credit** is the job's own: an offset is credited against the clearance C = 2.4 m and a place against the re-find
radius R = 2 m ([questions.md](questions.md), "Credit"; 0.7.0 and 0.8.0 gave each a third of the credit at C/2, C and
2C, or R/2, R and 2R). They are declared constants, read from no gold, and the system prompt states them. `JobSpec.ladder` is 1, 2, 4
and 8 m, the unit of the downtown seen from 7 to 15 m up, and its first rung doubled is the 2 m matching radius. Before
0.7.0 a place or a distance earned full credit within 1 m of the truth, three quarters within 2 m, half within 4 m and a
quarter within 8 m.

## The zone coin: balance from the world, and the response

Each flight's drop zone is **blocked** when `make_rng(map, seed, "drone/1", "zone", i)` reads below 0.5, else **clear**:
a coin that reads no world, so no form, key or pool can predict it. Before the flight (at its start less
`changes.EDGE_BUFFER`, 0 for the first, after every change the gap made) a drop enforces the coin both ways
(`job_drone_zone.enforce`), with the world's own things and laws:

- **blocked**: an item whose receptacles include the kind of the surface under the zone (`plaza`: food carts and portable
  toilets; `loading zone`: vans, box trucks, skips and dumpsters) moves onto that surface, squared up, covering the zone
  materially (30% of its own footprint or of the zone's), colliding with no other item; a zone an item already covers
  that much is left;
- **clear**: every item within 5 cm of the zone moves out by the start's law (`world_processes_home.move_item`); one the
  law finds no place for is given one by moving the (at most two) items that stand where it fits.

The zone is the piece's footprint inset by 0.05 m (the zone rule's `EDGE_M`: a piece that only touches a zone never
blocks it). Pedestrians never enter a drop zone, and nothing else moves in view, so the coin is the zone's
state through the whole flight. The drone flies over every item (every downtown kind is under 3.3 m, its lowest pass 7
m). The way is planned before the drop and reads none of it: the script with the drops left out has the same poses,
times and decision points.

**The response: an abort.** The drone asks itself at `before_lower` whether the drop zone is clear, and then acts on what
the zone holds. After the drop, a delivery whose zone holds a standing item at the flight's start (taller than 0.3 m
over the zone's ground, its footprint overlapping the zone: the state `zone_clear` reads) is **aborted**:
- it flies `hover_phases(INSPECT)`, an inspection's hover (`inspect` for the three holds the lowering, the release and
  the reel-in take, with the same marks), so no keyframe time or pose moves;
- `tethers` is given `deliver and not aborted`, so no tether is paid out;
- it climbs out, `before_return` comes, and it flies home with the package;
- the decision's `response` is `abort`.

Any other flight, an inspection over an occupied zone included, goes on, response `go`; `before_descent` and
`before_return` carry no response. The response is the world's, read after the drop (`job_drone_zone.standing`), not the
coin: a coin the drop could not honour (a box truck the world cannot move) gives the zone's true state. It comes after
the decision, so a question asked at `before_lower` (which sees the frames up to that keyframe) never sees it.

**Nothing before the question tells the coin.** Scripted with every coin blocked and with every coin clear, a seed's
keyframes (poses, times, camera, phases, statuses) and subgoals up to and including each `before_lower` are identical,
and every pose and time to the end of the script is the same. The drone's body, tether and package are not drawn, so an
abort changes no pixel of the flight: it changes the status (`tether_m`), the phases and the subgoal from the first
keyframe of the hover on. An inspection's flights do not change at all. What tells the coin is the street below, seen in
the frames at `before_lower`, which is what the question asks about. The response also ends a realism defect of the
last release: its drone lowered packages onto parked trucks in 22 of 34 cases, from a coin with no response.

**A zone is planned only where the world can set it** (`job_drone_zone.settable`, from the map and its catalog alone): an
item can be put onto it, and every kind that fits the surface under it has a home clear of every drop zone, so a clear
zone never asks the world to put a thing nowhere. On `downtown_a` this leaves the square's three zones (`F26` Square
West, `F27` Square East, `F28` Square South) and drops the two street zones: a 7.3 m box truck fits only a loading zone
(2.5 x 14.8 m), and a 3 m zone in the middle of one leaves 5.9 m each side, so the truck covers the zone whichever way it
stands. The start's world holds a truck on a strip in 27 of seeds 101-130 (13 of them one on each strip), so a clear
coin on a street zone could not be honoured there: before the rule about three quarters of the street zones' flights
came out blocked, whatever their coin said (West Street 27 of 36, East Street 23 of 30). With the street zones moved to
the south end of their loading zones (centre y 44.1 instead of 49.5; 11.8 m beside) they are settable and 148 of 150
measured flights held their coin; the rule then plans them with no code change. That map edit is outside the job.

## The downtown's processes

`world_processes_downtown.py` (loaded by `world_processes.gaps` for the downtown catalog): `start` does nothing; `gap`
makes every change at one whole second inside the gap (at least 30 s from both ends), drawing each item again by the start's
law at a half-life (`half_life_p`): cars, vans, pickups and box trucks 1 h (on-street parking turnover: an average stay
of 1 h 52 min in Everett, 74% of stays under 2 h in Carpinteria, under 30 min at retail bays in Melbourne),
food carts 6 h, skips, dumpsters and portable toilets 48 h (modelling choices: no source gives a stay for the unmetered
kinds). `SOURCES` names each number. No thing is added or removed.

## The checks

`job_drone_checks.py`, run through `JobSpec.checks`: `job_phase_order`, `job_nest` (the first and last keyframe on the
pad within a centimetre), `job_heights` (15 m in transit, 7 m above the zone's ground over the hover, nothing above 15 m
or below the pad), `job_gimbal` (the phase's pitch, within the limits, never turned faster than 60 degrees a second),
`job_tether`, `job_decisions`, `job_battery`, `job_task_time` (600 s), `job_gap` and `job_span`, `job_targets` (the
flight's region, and the plan's zone centre).

`job_response` ties the response to the world: it reads the carried items' placements at the flight's first keyframe and
flags a `before_lower` that is not `abort` exactly when the flight is a delivery and its zone holds a standing item (else
`go`), and a response on any other decision. The same reading decides which phases the flight must run (`job_phase_order`,
`job_heights`) and whether it pays out tether (`job_tether`): a delivery over an occupied zone must run the inspection's
phases with none, a delivery over a clear one the lowering's. The phases follow the world, not the stored response.

## The subgoal

`job_drone_subgoal.subgoals(tl, m)` gives one line for each keyframe: what the drone is doing now, as part of its flight.
It reaches the agent in the header of every frame and `path_log` in its record. It names the drop point by the plan's
address (the assignment states it) and the phase, and nothing else. It is causal: the text at keyframe k reads the
flight's plan, the phases up to k, the decision at k (its kind) and the responses strictly before k, so it does not
change when the timeline is cut after k, nor when the coin of a decision at or after k falls the other way. The one
response, the abort, shows from the first keyframe of the hover on and never in the text at `before_lower`.

| Phase | Text |
| --- | --- |
| `preflight` | "preparing to deliver a package to the drop point at {address}" or "preparing to inspect the drop point at {address}" |
| `takeoff` | "taking off from the nest" |
| `approach` | "flying to the drop point at {address}" |
| `inspect`, at 15 m | "inspecting the drop point at {address} from above" |
| `descend` | "descending over the drop point at {address}" (the text at `before_lower`) |
| `lower` | "lowering the package at {address}" |
| `release` | "releasing the package" |
| `climb_out`, at the hover's height | "reeling the tether in" |
| `inspect`, after the descent | an inspection: "inspecting the drop point at {address} from close above"; after an abort: "holding over the drop point at {address}: it is not clear to lower" |
| `climb_out`, climbing | "climbing away from the drop point" |
| `return` | "flying back to the nest", and " with the package" after an abort |
| `land` | "landing on the nest" |

The hover's height is the base's at `before_lower`; a `climb_out` keyframe at or below it is the reel-in.

## What the drone asks

**0.11.1** asks what 0.11.0 did, except that the corridor is asked only where its evidence frame reads the wall's foot
to within C/2 at one pixel ([families/clear_path.md](families/clear_path.md), "Tied to its foot"). The release holds 262
questions on the drone's 36 cases: `clear_path` 194 and `ego_mem` 68 (0.11.0: 264, `clear_path` 196). In the short and
medium cases that leaves 197 look questions (`clear_path` 142, `ego_mem` 55), one question in 12.8 frames (the 2,530
frames of the short and medium cases over 197); harm is 72% of them and 74% of all (194 of 262), and the clearance cells
hold 96 questions in `now` and 98 in `memory`.

**0.11.0** asks what 0.10.0 did (below).

**0.10.0** (`job_drone.py`; the reasoning is [question-design.md](question-design.md)). A look point has two slots:
`clear_path` at 60% (slot 1) and `ego_mem` at 70% (slot 2); nothing at its end. The frames of `before_lower` and
`before_descent` come first in the look draw. The corridor's rules are 0.9.0's (ahead and back, at least 8.35 m up).
- **`drop_zone_people` is not asked** ([families/drop_zone_people.md](families/drop_zone_people.md)). The 0.10.0 draft gave
  it the third slot whole ("Before you lower the package: where are the people within 4 m of the point below you?", in
  `inspect`, `descend` and `lower` only). It drew 2 questions over the 36 dev cases, under `abilities.MIN_ABILITY_N`, so
  no gate could read it and its 2 rows sat inside the clearance `now` cell with no per-family diagnostic; the final
  review's S2 took it off the drone, with its phase rule, and rebalanced nothing else. The family, its gold and its page
  stay for a dataset with the supply.
- **Measured** on the dev root after the repair (short and medium cases): 433 look points and 199 look questions
  (`clear_path` 144, `ego_mem` 55), one question in 12.7 frames (0.9.0: 1 in 29.8); harm is 72% of the questions (all
  lengths: 196 of 264, 74%). The drone's clearance cells hold 96 questions in `now` and 100 in
  `memory`. (These are 0.11.0's; 0.11.1's are above.)

**0.9.0 (history).** The table is 0.8.0's: `clear_path` at 30% (slot 1), `ego_mem` at 85% (slot 2), nothing at the end,
and the job keeps 75% of its `unknown` corridors. The corridor's directions are now declared (`JobSpec.clear_directions`
= ahead, back; the draw is 0.8.0's, so no gold moves) and the credit is 0.9.0's ([questions.md](questions.md),
"Credit"): four quarters against C = 2.4 m, the resolution one pixel at the wall's foot from the evidence frame, an
answer longer than the true distance counting against it twice as fast. Sideways flight is not asked (a constant scores
+0.15; the drone flies about 5 m from a side wall) and neither is `open_heading` (the open high street is the longest run
from every point of the square, a layout fact).

**0.8.0** (task-critical questions: harder questions, not more of them). The drone must not
hit a building on its way out and back. During the job it asks `clear_path` at 30% of its look points (slot 1) and
`ego_mem` at 85% (slot 2), and **nothing at its end**: `object_where` left (a delivery drone re-finds nothing on
the ground) and its end ask funds the corridor, and `ego_now` left (nothing it names can be hit from 15 m).
`clear_path` ([families/clear_path.md](families/clear_path.md)) asks "If you flew straight {ahead | back} now at this
height, how far could your body go before it touched anything you have seen?", the direction drawn by `stable_hash(case
id, k, "direction") % 2`, credited on C = 2.4 m (rungs 1.2, 2.4 and 4.8 m).
- **Every phase of its status**, but only where the base is at least `JobSpec.fly_floor_m` = 8.35 m up. Landed (0.27 m
  on the nest, among benches, planters, people and cars) and hovering at 7.01-7.15 m (the body reaches down to 6.86 m)
  the walls-only gold is wrong: 65 of 215 drawn points before the floor. The square's street trees stand along its
  walls every 6 m, trunk to 3.40 m above the 0.15 m sidewalk and crown up to 3.5 m tall, so a crown top reaches 6.90 m,
  and the street lamps are taller still: the render stands one every 28 m in the kerb strip of every sidewalk of the
  map (`render/scenery_layout.py`, `POST_HEIGHT_M["lamp"]` = 7.95 m, drawn by `render/scenery.py`), fixed scenery that
  is not in the map. 8.35 m is the lamp head, plus the body's half-height 0.15 m, plus 0.25 m. It is the place's own
  dressing, read once from the render code's declared sizes and stated in the job file with its source. (The first
  0.8.0 floor, 7.3 m, cleared the crowns but not the lamps.)
- **The gold** is the 1.3 m body's corridor against the map's wall boxes; only walls can stop it
  (every movable is under 3.3 m, and the floor clears the trees, signals and lamps). The wall the corridor meets first
  is **seen** when some frame of the window showed any stretch of it: a building face is one flat plane, so a wall seen
  a few metres along its face has a distance. The gold is `unknown` only where no frame of the window showed that wall
  at all, even if a wall beyond it was seen (the body touches the unseen one first);
  a corridor into the open high street (the 60 m cap) is not asked, since no frame certifies empty air. The job keeps
  75% of its `unknown` golds.
- **The drone stays the easiest place**: its corridors are bounded only by static building walls, which the camera sees
  (the agent is never given a map), and nothing else at its height can be hit. Its hard hazards (pedestrians it can
  resolve, wires, trees, poles, traffic) would need renders. A `clear_path` score here is a claim only about a system that
  holds no copy of the map. The 0.11.0 design looked for a drone family about something that changes or is remembered over
  time (a person remembered at the drop, the drop clearance, a people list, a route, vehicles, change between flights) and
  found none that survives its gates on this data
  ([question-design.md](question-design.md), section 3).

**0.7.0** (the 0.6.0 mapping is unchanged). No decision kind asks a family:
`zone_clear` is retired, and the decision points, the zone coin and the abort stay in the path, so the scripts and
renders are unchanged. The zone's blocker is told by the robot's own log (the response `abort` and the subgoal), so
`World.told` keys on the job's `zones` and no question names it. During the job the drone asks at look points
(`JobSpec.looks`): `ego_now` and `ego_mem`, placed as an offset from its camera (which hangs below its base) along the
floor, and credited on the clearance C = 2.4 m (Wing's delivery-point clearance from buildings and trees), the same
metres for a thing 5 m off and one 20 m off. At its end it asks `object_where` alone, credited on the re-find radius R = 2
m (`object_was` and `same_place` are not asked since 0.6.0: 0.33 a medium case, under the supply line). It asks no
`task_things`: the map's floor
surfaces within 10 m of the drop point score +0.25 to +0.40 against its gold, because vehicles stand in parking lanes and
loading zones (a layout fact, truth rule 5), and with the vehicles made don't-care it leaves 0.25 items a task and 80%
empty. The drone's memory column therefore holds memory alone, and it has no change or relevance column. (Before 0.6.0
dropped it, `same_place`'s "no" by a place seen empty ("vacated") was not asked of the drone: a flier's ground points
all read as floor from 15 m.) The sections below are the 0.5.0-control
job as built.

In the 0.5.0-control job: `zone_clear` ("Is the drop zone clear of anything taller than 0.3 m right now?") at `before_lower`, the
last keyframe of the descent, asked of the frames up to it. The drone is at least 16 steps into its flight there in
every flight (16-25 steps in the measure below), so the questionnaire always may ask it; `before_descent` is 16 steps in
in fewer than half the flights (71 of 150 below), and it and `before_return` ask nothing. The gold is now-only: the
zone's state over the decision's look window alone (a blocker seen certifies "no"; at least 80% of its cells seen free is "yes"; else
"unknown"), from code over the timeline and the visibility log. The zone is the `drop_zone` piece's footprint less 5
cm, with a band from 0.3 to 3.5 m over its ground (a 3.3 m box truck's roof is inside it). The column is `now`.

At its end: `object_where` alone ("Where did you last see the X?"), about the things of the downtown's kinds within 12 m of
the drop zone's centre. `distance_between` leaves the job; `clearance_then` is not asked (after the response the drone's
own path and subgoal tell what the zone held); `order` and `when_seen` stay dropped.

**Measured, free, no render** (seeds 101-130, 4-6 flights: 30 scripts, 150 flights, 7,417 keyframes): every seed
generated (skip rate 0 of 30, against a declared bound of 10%) and every script passes `stbench validate`, the leak scan
included. The zone held a standing item at the flight's start in 82 of 150 flights (0.547; 0.559 of the 111 deliveries),
inside the 0.4-0.6 the coin should give; 148 of 150 coins were honoured (the two others are West Street box trucks the
world cannot move on a clear coin, and the drone aborted both: it answers for the zone it finds, not for the coin).
Responses: 62 `abort` (all deliveries over a held zone) and 88 `go` (49 deliveries, 39 inspections). `before_lower` lies
16 to 25 steps into its flight: eligible in 150 of 150 flights, so 1.00 eligible decisions a task (and 71 of 150 flights
reach 16 steps at `before_descent`, which asks nothing).

## The free measures (the first draft's)

What follows is the first draft's history and its measures of the zone coin; the mapping it describes is superseded by
"What the drone asks" above, and its numbers (ray-cast depth over the old band, the 0.4.x mapping) are not this
release's. The design-time script (not shipped; it ran on `downtown_a`, seeds 101-130, and M4's suburb and courtyards
re-ran it unchanged) measured, on scripts as built, with the depth ray-cast at the drone's camera in centimetres over
the walls, the fixed pieces and the items at each keyframe's time (the zone bands only: a zone band keeps only the zones'
cells), each gold by its family's own rule, and each shortcut's expected penalised score as its best response over the
other scripts' golds at the question's key:

| Item | Measure | Needs |
| --- | --- | --- |
| (z) | `zone_clear` at `before_descent` and `before_lower`: shares of `no`, `yes`, `unknown` golds; coins enforced; `blind`, `path_log` and the form adversary | coins all enforced; `no` 0.45-0.55 of the answered; each shortcut at most all_unknown + 0.10 |
| (r) | `clearance_then` at the release, by surface | the same |
| (v) | median share of the zone's cells observed in the look window | at least 0.8 |
| (w) | least distance from the footprint to a wall face | at least 3.0 m |
| (k) | frames a flight, decisions a flight and past the first 16 steps, `zone_clear` questions a short case | reported |

The design mapped `zone_clear` onto `before_descent` and `before_lower` iff (z) and (v) held, and `clearance_then`
into `ends` iff (r) held, on this table re-run on the merged code. **As first mapped**: `zone_clear` at both,
`before_return` asking nothing, `clearance_then` in `ends`.

**What the drone asked in 0.4.x**: nothing during
the job, and at its end `object_where` and `distance_between`. The release is 12 drone scripts (seeds 2001-2012, 36
cases), and the final check over all of them (0.4.x, the point verdict) passes both:
`object_where` (168 questions, separation +0.81 [+0.75, +0.87]) and `distance_between` (77, +0.84 [+0.77, +0.94]). The
rest were dropped:
- `zone_clear` and `clearance_then`, first at the release gate check on the first six scripts (seeds 2001-2006, 18
  cases), where `other_case` scored +0.27 over `all_unknown` at 49 questions and +0.21 at 24 (bound 0.10), with
  `clearance_then`'s `blind` BROKEN at -0.21. The zone families' golds are balanced by a coin that reads no world
  (`zone_clear`'s were 26 `yes` and 23 `no`), so those leads could be chance. They were mapped back and read once more,
  by a rule declared before the data: ship them only if the final check over all 12 scripts, the six they failed on
  included, passes them on the point verdict. It did not: `zone_clear` (96 questions) fails `shortcuts`
  (`blind_in_pool` +0.104 against 0.10, with `blind` BROKEN at -0.135), and `clearance_then` (49) fails `memory_needed`
  (`current_view_prior_in_pool` +0.143) and `shortcuts` (`form_in_pool` +0.143, with `blind` BROKEN at -0.122). Both are
  dropped for good.
- `order` asked 5 questions (0.28 a case) and left `ends`.
- `when_seen` (28 questions, separation +0.93) passed the first check, but an independent review found its `time` answer
  reachable from the clock alone: the drone ends a flight by going home, so a thing's last sighting falls 12-67 s before
  the window's end. After the instrument's repair (time answers learned as lags behind each case's own clock) it fails
  `memory_needed` and `shortcuts` (+0.30: a memoryless learner of the lag answers it), and it was dropped. It was later
  dropped from the AMR and the vacuum as well, so no 0.4.0 or 0.4.1 job asks it
  ([families/when_seen.md](families/when_seen.md)).

0.5.0-control asks `zone_clear` again, now at `before_lower` alone and with a response the drone acts on (above), so the
coin is a decision and the chance tilts that failed it under the point verdict are read under the evidence verdict
([eval.md](eval.md)); `distance_between` and `clearance_then` left the job for the reasons given in "What the drone asks".

The script also printed, for each shortcut, its score when every flight's golds are redrawn as one fair coin each
(`--null`, 100 samples, keys kept): a pool of n questions learns a tilt of about 0.8 / sqrt(n) from a coin with nothing
to find, so at these sizes a fair coin beats `all_unknown` + 0.10 itself in a fifth to two fifths of the samples, and a
shortcut's score means something only against that spread.

**Results** (the lane's run, seeds 101-130, 4-6 flights each, 150 flights, one-task cases, the zone families as the lane
had them; no pedestrians yet; the ray-cast depth in centimetres):

| Item | `downtown_a` as shipped (the square's 3 zones) | street zones moved to the strips' south ends (5 zones) |
| --- | --- | --- |
| coins enforced | 150 of 150 | 148 of 150 (two box trucks the world cannot move) |
| (z) `zone_clear` golds | n 300: no 0.533, yes 0.467, unknown 0 | n 300: no 0.547, yes 0.453, unknown 0 |
| (z) `blind`, `path_log`, `form` | +0.067, +0.173, +0.177 (`k_in_task`); fair coin -0.010 +- 0.154, +0.004 +- 0.155, +0.057 +- 0.110 | +0.040, +0.107, +0.093 (`answer_type`); fair coin -0.020 +- 0.166, -0.012 +- 0.165, +0.075 +- 0.081 |
| (r) `clearance_then` golds | n 111: no 0.541, yes 0.459, unknown 0 | n 111: no 0.523, yes 0.441, unknown 0.036 (4 flights); loading zone n 50, plaza n 61 |
| (r) shortcuts | +0.081, +0.081, +0.270 (`k_in_task`); fair coin 0.000 +- 0.178 (both), +0.077 +- 0.126 | -0.207, -0.180, -0.090; loading zone -0.120, +0.200, -0.120; plaza -0.492, -0.492, -0.049 |
| (v) look-window coverage | median 1.000, min 0.914 / 0.917, every view `now` | median 0.996, min 0.000: 10 flights `last` (a truck's roof, below) |
| (w) footprint to wall | 4.350 m | 3.603 m |
| (k) frames a flight | 44.4 delivery, 44.8 inspection; 223 a script | 49.4, 49.5; 247 |
| (k) steps into the task | `before_descent` 12-16 (12 of 150 at 16 or more), `before_lower` 15-19 (147 of 150), `before_return` 24-33 | 13-22 (71), 16-25 (150), 25-39 (150) |
| (k) `zone_clear` a short case | 0.517 (before_lower 49%, before_return 49%, before_descent 3% of draws) | 0.616 |

Every gold equals its flight's coin (411 of 411 on the shipped map): the state the world is made to hold is what the
depth shows, whenever the zone is in view. What the table says:
- **Balance holds**: `no` is 0.533 and 0.547 of the answered, inside 0.45-0.55, with the coin enforced everywhere.
- **No shortcut has a mechanism**: every key a shortcut reads (the pose, the phase, `k_in_task`, the room words, the
  slots) is a function of the map, the seed and the plan, drawn from streams the coin does not share, so what `path_log`
  (+0.173) and the form adversary (+0.177) exceed `all_unknown` + 0.10 by is the finite sample's tilt: a fair coin does the
  same in 29% and 40% of samples of this size (this run's rows lie at the 87th and 93rd percentile of that spread). The
  interval cannot certify (z) on 30 scripts, and M3's dev data (about half a question a short case, below) is thinner
  still: the point verdict, with the null in view, is what can be read. More seeds (the golds equal the coins, so a run
  without ray-cast would do) shrink the spread as 1 / sqrt(n).
- **`before_descent` cannot ask** on the square's zones (12-25 m from the nest: the flight is 12-16 steps old at the
  decision, the questionnaire wants 16), and `before_return`'s `where_was_i` competes with `before_lower` (8-13 steps
  apart, `GAP_STEPS` 16): they split the draw 49% / 49%, so a short case asks `zone_clear` 0.517 times. Leaving
  `before_return` asking nothing makes it about 1.0. Either way the interval verdict (which wants n of about 270) is out
  of reach on 23 cases.
- **The street zones need a taller band.** A 3.3 m box truck covering a zone has its roof above the band's 3.0 m top, so
  from straight above the zone is unobserved (the 10 `last` views, all gold `no`, so the wording `as you last saw it`
  would predict the answer; and the release's `clearance_then` is `unknown` for 4 flights). `zones.TOP_M` above 3.3 m
  (3.5) removes it. The shipped zones are unaffected: their pads hold only carts and toilets, under 2.4 m.

## Tests

`tests/test_job_drone.py` (plan, zone coin, gaps, briefs, events, the checks' teeth), `tests/test_job_drone_flight.py`
(validity, walls, nest, camera, tether, decisions from the plan alone, the swap pair), `tests/test_job_drone_control.py`
(the abort response, the identity of the script up to `before_lower` under both coins, the aborted hover, the subgoal's
texts and causality, the declared control fields, the response checks' teeth), `tests/test_world_processes_downtown.py`
and `tests/test_keyframes.py`'s flier cases; `tests/drone_kit.py` holds the shared scripts (`generated`, and `forced`,
which patches `job_drone.coin_blocked` to say blocked or clear for every flight). Each module builds its scripts once
(seeds 101-104, 4-6 flights, and seed 101 under both forced coins); the ray-cast measures are no test.
