# Jobs: what a robot does in a script

A script is a robot doing its job in a built world. `job_registry.py` holds the registry, `job_<robot>.py` one job each,
and `generate.generate_script` runs them: the built world, `clutter.place`, the light furniture's starts, then the job's
plan and path. This page holds the vacuum's two jobs: `vacuum/1` (`job_vacuum.py`, M1), which stays frozen, and
`vacuum/2` (`job_vacuum2*.py`, M2, below), the vacuum's job that ships. The other robots' jobs are
[job-helper.md](job-helper.md), [job-amr.md](job-amr.md) and [job-drone.md](job-drone.md); what each job asks is
[questions.md](questions.md).

## One task

The plan draws a room, uniformly from the regions other than the dock's own (`EXCLUDE_DOCK_ROOM`: a hall task crosses no
doorway and asks nothing during the job; M2's multi-room tasks bring the hall back): `TaskPlan("clean_room", {"room":
<region kind>})`, the region's id as the task's private target. The generator writes one task a script in M1; the
battery starts at an integer in 90-100 (the start set of `robots.py`) and falls 1.1% a minute (a Roomba's 90-minute run),
and each phase's speed is drawn once a task from the robot's range (cleaning 0.28-0.31 m/s, transit 0.30-0.35,
undocking and docking 0.10-0.15). Turns are in place at 45 degrees a second. The brush is on exactly in `cleaning`.

## The phases

| # | Phase | What the base does | Decision |
| --- | --- | --- | --- |
| 1 | `docked` | stands at the docked pose: the disc touching the dock's front face, facing out (4.94, 5.05, heading 0) | |
| 2 | `undocking` | forward 0.5 m off the dock | |
| 3 | `transit` | turns in place and drives to the room's edge: A* on the vacuum's free space | `before_doorway`, 0.5 m before the door line, facing it |
| 4 | `cleaning` | the edge pass, then strips 0.20 m apart along the room's longer axis | `room_done`, on the last cleaning keyframe |
| 5 | `transit` | back to the point 0.4 m in front of the docked pose | `before_doorway`, as in 3, inside the room |
| 6 | `docking` | turns to the docked heading and reverses 0.4 m onto the dock | |
| 7 | `docked` | stands at the docked pose again | |

Decision points come from this plan alone: the world moves where they fall (clutter, a pulled-out chair), never
whether they exist. `vacuum/1` asks nothing at either (its `decisions` map holds no family, and its end families are the
default, `object_where` alone): `path_ahead` was asked at `before_doorway` and left it in M1, and no job asks it since
([families/path_ahead.md](families/path_ahead.md)). The decision points stay in the path and the keyframes, so the
scripts and their ids do not change.

**The free space** is `route.robot_free_space`: a 2.5 cm grid whose free cells keep the body's radius (0.17 m), its
clearance (0.02 m) and a cell's half-diagonal from every wall and from `route.robot_obstacles`: what `validate`'s body
rule reads. The vacuum passes under a piece with an underside of 0.10 m or more (the dining table, the coffee table, the hall
console, the desk, the chairs, the stool) and meets its legs; the sofa (0.08 m) and the bed (0.05 m) it cannot enter. Door
leaves are obstacles in their state, the dock is one, and so is any floor item or clutter above 2 cm. A 10 cm grid, with the
half-diagonal that goes with it, closed the passage between the kitchen's open door leaf and its counter (0.8 m wide, for a
robot that needs 0.38 m), so a kitchen task cleaned about a fifth of the floor the vacuum could reach.

**The edge pass and the strips** are `coverage.py`. The edge pass is the outer contour of the room's reachable free floor,
5 cm inside it, simplified, counter-clockwise, from the cell nearest the room's entrance; a piece against a wall is part of
the edge, a piece in the open is a hole. The strips are parallel to the longer side of the room, cut into the runs of free
floor they cross. Runs on neighbouring strips that overlap form a chain; the robot cleans a chain at a time, back and
forth, and `FreeSpace.path` joins the ends. An obstacle in the way therefore leaves a chain each side of it. Every point of
every path lies in a free cell, so the body keeps its clearance between keyframes too.

## The frames

`keyframes.frames` samples the job's primitives (`Move`, `Turn`, `Hold`, `Mark`) under the robot's `FramePolicy` (the
vacuum's: a frame each 1.0 m, 45 degrees or 5 s, none closer than 1.0 s). A frame is required at the start, at the end of
every turn (a vertex of the path), at every `Mark` (a decision) and at the end of every phase's run of primitives (M1's
question 2: "when you left the dock" is the last `undocking` frame). Two required frames at one instant merge when they
share a phase and carry no two decisions; otherwise the robot holds still until a second has passed, which is why the
docked frame comes a second after the docking one. So the straight lines between frames are the path, and every vertex is a
frame.

## The events and the relevance

The four events resolve from the delivered steps alone (`events.resolve`): the last `undocking` frame, the first
`cleaning` frame, the last `cleaning` frame (the `room_done` decision's) and the first `docked` frame after the `docking`
ones. The relevance names the floor-level kinds (shoes, bag, toy, box), the light furniture (chair, stool, laundry basket),
the task's room and the hall, and the phases cleaning, transit and docking. Floor clutter is not a relevance kind: no M1
question names it.

## The light furniture's starts

Light furniture starts each script somewhere else (`generate.light_starts`, from its own key), so what the corridor meets
varies and M2's questions about it start from per-script places. A chair is pulled out up to 0.6 m against its facing
direction (never into its table) and turned up to 15 degrees; the stool and the laundry basket move 0.3-1.0 m in a uniform
direction to a uniform heading. A start must lie in the piece's room, clear of blocking fixed pieces, other light pieces,
the keep-out round every interior door, the gap of every arch, the dock's apron (1.2 m in front, 0.5 m each side) and floor
items and clutter (the legs only, for a piece on legs: the world lays a toy under a seat); after 60 draws without one the
piece keeps its map place (the stool, hemmed in by the counters, does so in most scripts).

## What the checks hold

`job_vacuum._checks`, run by `validate`: the phases run in the order above (each present), the brush is on exactly in
`cleaning`, the first and last frame of a task stand at the docked pose (1 cm, 1e-3 rad), and a task holds exactly
`before_doorway` in transit, `room_done` on its last `cleaning` frame, `before_doorway` in transit.

## Sizes

On `apartment_a`, over seeds 1-150 with the real clutter (`clutter.place`), a one-task script holds 191-234 entities
(median 215) of which 20-44 are floor clutter, and these frames (minimum, median, maximum): bathroom 63/97/129, bedroom
36/76/108, kitchen 160/182/205, living room 209/273/345, a mean of about 160 a script. (Counts of 55-120, 120-220 and
180-355, a mean of about 130, were made with stand-in clutter, `tests/job_kit.py`, before the real clutter was in.)
The living room is 15 m2, its free floor about 6 m2: 30 m of strips, 17 m of edge pass and four frames for each U-turn.
Generating a script takes about half a second.

# `vacuum/2`: the vacuum's full job (M2, `job_vacuum2*.py`)

Scripts of 2 to 6 tasks on `apartment_a` (`stbench generate|build --tasks 2-3` or `--tasks 4-6`: each seed draws its own
count, from its own key, the same in `generate` and in a rebuild). `vacuum/1` stays frozen beside it; `vacuum/2` reuses its helpers by import.
The plan is `job_vacuum2_plan.py` (the map and the seed alone), the path and the declarative part `job_vacuum2.py`, the
job's own `validate` rules `job_vacuum2_checks.py`, the back-out mechanism `job_vacuum2_backout.py`, the doorway coin
`job_vacuum2_doorway.py` (0.5.0-control) and the subgoal `job_vacuum2_subgoal.py`.

## The plan and the gaps

- **Task kinds.** `clean_rooms` cleans one room, or two with probability 0.25, drawn uniformly without replacement from the
  map's regions; a task's first room is never the dock's (the hall: a hall-only task crosses no doorway and asks nothing
  during the job), so the hall comes back only as a second room. `resume` finishes the room the task before it was sent
  home in. Task 0 is `clean_rooms`.
- **The stop.** A `clean_rooms` task that is not the script's last, and whose last room is not the dock's, is sent home
  with probability 0.3 (by the resident's app: a vacuum in an occupied home is interrupted that way; a low
  battery would never fire, a room takes 2-8 minutes of a 90-minute charge), after the brush has swept 30-70% of that
  room's reachable floor (by area, not by path length: the edge pass is a third of a living room's path and most of its
  floor, so a stop by length left a hole too small to resume). The next task is then a `resume` over that room.
- **The gaps.** n - 1 of them, each log-uniform from 10 minutes to 3 hours, shrunk toward 10 minutes when their sum passes
  `SPAN_S - n * TASK_MAX_S` (7 hours less every task at its longest, 20 minutes), so a script spans at most 7 hours: a case
  clock starts 07:00-09:59 and so ends by 17:00, never at night under the one daytime light every frame has. They depend on
  the seed alone (`gaps(map, seed, n)`), never on the world. The robot is docked and its battery restarts in 90-100 at
  every task: nothing carried over shows where a window sits.
- **The brief** names the rooms only: "Its current assignment is to clean the kitchen, then the hall." / "...to finish
  cleaning the bedroom." (`stop` and `kinds` are the job's, never delivered).

## The path of a task

The script calls the world's gap callback before each task (`world_processes.gaps`: `between(lo, hi, keep_open)`, `lo` None
before the first task) with the dock's region and the task's rooms as `keep_open`, then plans the task on
`route.robot_free_space(..., shut=<every interior door off the ways between them>)`: each such door is an obstacle in both
leaf positions, so no door's state can change a path and the robot's own path log learns none (the doors on the ways are
kept open by the callback). The task is `vacuum/1`'s phases, once for each room: docked, undocking, transit to the first
room, then for each room cleaning (the edge pass, then strips) and transit, docking, docked; a room whose doorway is
blocked is skipped (below, "The doorway coin"). Before the task is planned, every doorway is cleared.

- **A resumed room** has no edge pass: the strips only, those runs of a strip that come within the brush's reach (0.125 m)
  of floor the stopped task's brush left unswept (`grid.swath` over its brush-on keyframes; `coverage.strips`'s optional
  `mask`). Stopped and resumed together reach what a whole clean of the room reaches, within a few points (a stop inside the
  edge pass leaves the wall margin only that pass reaches).
- **The keyframes are the path.** Every brush-on vertex is a keyframe: `Driver.begin` frames the first point of a cleaning
  run (the phase's first frame is otherwise a metre on), and `Driver(exact=True)` frames a vertex that turns under a degree
  and puts a decision at the start of a segment in that segment's phase. So `grid.swath` over consecutive brush-on keyframes
  equals, cell for cell, `grid.swath` over the primitives' cleaning moves (`uncovered`'s gold reads it).
- **Speeds and status** as in `vacuum/1`: one speed a phase a task, the battery an integer in 90-100 falling 1.1% a minute,
  the brush on exactly in `cleaning`.

## What `vacuum/2` asks

**0.11.0** (`job_vacuum2.py`). The slots below are unchanged. At its end the job asks `object_where` and `task_things`
again, now as the list of what the brush could catch ([families/task_things.md](families/task_things.md)): "on the floor
of the rooms you cleaned that your brush could catch" (cords and cloth; bath mats, pedestal mats and clothes piles recur
at the same spots from script to script and are don't-care, as in `tangle_ahead`). `open_heading` counts in the
headline. The release holds 711 questions on 36 cases: `clear_path` 315, `open_heading` 134, `tangle_ahead` 133,
`ego_mem` 83, `object_where` 24 and `task_things` 22.

**0.10.0** (`job_vacuum2.py`; the reasoning is [question-design.md](question-design.md)). A look point has three slots, harm
first: slot 1 is `clear_path` or `tangle_ahead` (50% each), slot 2 `ego_mem` (100) and slot 3 `open_heading` (20); at its
end `object_where` alone. A look point is any frame at least 4 steps into its task, and the frames of `before_doorway` and
`room_done` come first in the draw (`before_backout` never occurs: `SITE_KINDS` is empty).
- **`tangle_ahead`** ([families/tangle_ahead.md](families/tangle_ahead.md)) is new: "List anything lying on the floor within
  1 m of you, ahead or to your sides, that your brush could catch (a cable bundle, an extension cord, a power strip or a
  towel; a bath mat, a clothes pile or a pedestal mat counts neither way)." A cord or cloth wound round
  the brush stops the robot until a person frees it. It is asked in `cleaning` and `transit` (`JobSpec.look_phases`) and
  lists extension cords, power strips, cable bundles and floor towels; bath mats, pedestal mats and clothes piles recur at
  the same spots from script to script and are don't-care. A missed item within 0.3 m of the front edge is a catastrophe.
- **`clear_path`** is as in 0.9.0 (ahead, back, left; asked in `cleaning` only), now at 50% of the points.
- **`ego_mem`** is asked only of things on the floor, since a mug on a table is no obstacle to a vacuum, and its recent view
  slides with the point.
- **Measured** on the 0.10.0 development compile (short and medium cases): 981 look points, 416 look questions (`clear_path` 180,
  `tangle_ahead` 81, `ego_mem` 70, `open_heading` 85) and 21 end questions, one question in 14.0 frames of look
  questions and 13.3 of all (0.9.0: 1 in 69.2); harm is 65% of the questions. The clearance cells hold 102 questions in
  `now` and 346 in `memory`, so the `now` cell, thin in 0.9.0's recompile (6), counts.

**0.9.0** (history; the table is `job_vacuum2.py`'s). The questions
are decisions its brain faces (harm, help, situation, reuse; [vision.md](vision.md)), credited by their consequence
([questions.md](questions.md), "Credit"). During the job the vacuum asks `clear_path` at 20% of its look
points (slot 1), and `ego_mem` at 30% or `open_heading` at 15% (slot 2); at its end `object_where` alone.
- **`clear_path` in three directions** (`JobSpec.clear_directions` = ahead, back, left; each drawn equally by a hash of
  the case and the point). Back is "If you reversed straight back now, how far could your body go before it touched
  anything?", from the back of the body; left is "If you turned in place to face your left and then drove straight, ...",
  from the front edge after the turn. The gold reads what the robot last saw of every cell up to the point
  ([families/clear_path.md](families/clear_path.md)). **Right is not asked:** a held-out constant leads by +0.28, since
  the vacuum follows edges with the wall at its right. The back corridor 0.8.0 left out (the robot's own swept path
  behind it answered within C/2 at 31% of points) is asked now: the gold reads the window's cells, and `path_log`'s
  detour stands only where it beats abstaining on other scripts. The vacuum is a disc, so its turn needs no sweep check.
- **`open_heading`** ([families/open_heading.md](families/open_heading.md)): which heading, turned in place, gives the
  body the longest run before it touches anything it has seen, credited by its regret on C = 0.10 m.
- Both are asked only in `JobSpec.clear_phases`, `cleaning`, as before.
- **`task_things` was dropped from the vacuum in 0.9.0** (the final review of 0.9.0;
  [families/task_things.md](families/task_things.md)). The family failed the binding evidence verdict here: the
  recurring-spots rule of `path_log` (`path_log_in_pool`) scores +0.258 over `all_unknown` on the long cases (95%
  interval +0.151 to +0.376) and +0.135 on the medium ones, a layout prior no sensor reads. The home's things stand at the
  same spots from script to script (90% of the vacuum's 68 distinct gold items have another script's item within
  0.5 m), and "on the floor of the rooms you cleaned" reaches every such spot. The rule ([methodology.md](methodology.md),
  rule 5): a family that fails a gate is fixed or dropped, never forced. The job then declared no predicate
  (`JobSpec.task_things` was None, as the drone's) and its prompt had no list paragraph (`brief.asks_lists`). The vacuum
  lost its relevance cell; the helper (route predicate) and the AMR kept theirs. Raising the credit's false-item weight
  or narrowing the match radius would tune a credit to pass a gate, so neither was done. A way back needed a different
  question (a held-out place, or a list limited to things off the layout's recurring spots); 0.11.0 made it, above.
- **Why 20 and 15.** The design declared `clear_path` at 25 and `open_heading` at 20. On the real compile the turned
  corridors have more answerable points than the design counted (the stopper takes the nearest contact within C/2), and
  the medium mean asks rose from 3.19 to 3.56. Cutting `open_heading` alone until no mean rose left it at 5 and two
  rows, too few to gate, so 5 points moved from `clear_path` to `open_heading` instead: every length's mean asks and
  questions stay at or under 0.7.0's, with 21 `clear_path` rows and 13 `open_heading` rows in the short and medium cases
  (no score is in the rule). The vacuum's clearance `memory` cell has 11 short and medium questions (14 in all), a
  diagnostic.

The 0.8.0 paragraph below is the history of the rest; its statement that the back corridor is not asked is no longer
true.

## What `vacuum/2` asked in 0.8.0

**0.8.0** (task-critical questions: harder questions, not more of them). During the job the
vacuum asks `clear_path` at 25% of its look points (slot 1) and `ego_mem` at 36% (slot 2); at its end `object_where` and
`task_things`. `ego_now` left the job (16 rows in 0.7.0, under the 20-question line in any table that holds the
budget). `clear_path` ([families/clear_path.md](families/clear_path.md)) asks "If you drove straight ahead now, how far
could your body go before it touched anything?", credited on C = 0.10 m (rungs 5, 10 and 20 cm), capped at 3 m (a metre
inside the camera's depth range). It is asked only in `JobSpec.clear_phases`, `cleaning`: docked, a
vacuum's corridor is the status, and in `transit` it leaves the dock on a fixed route (the same poses and golds in
different scripts), so a phase constant would score it. The vacuum is a disc, so the gold is how far its round body
could go (a wall converging on its side stops it later than the strip ahead says). The vacuum's band is 0.02 m to 0.095 m, so a sofa's or bed's underside above it is
overhead and the coffee table's 10.0 cm shelf over it is not asked (the height screen); the vacuum reads all of this
from 8 cm up. Its back corridor is not asked: the robot's own swept path behind it answers within C/2 at 31% of points,
M2's failed back-out site again ([Back-outs](#back-outs-the-mechanism-and-the-gate-that-failed)). No person question is
asked: the resident is never placeable from the 8 cm camera. The prompt now states the body (a disc 34 cm across, 9 cm
tall, climbing 2 cm, passing under anything 9.5 cm up).

**0.7.0** (the history of this section). No decision kind asks a family (`JobSpec.decisions` maps every kind to nothing); the decision points and the doorway coin
stay in the path, so the scripts and renders are unchanged, and the blocker a coin sets is told by the robot's own log
(its response and its subgoal), so `World.told` keys on the job's `zones` and no question names it. The job asks during
the job at look points (`JobSpec.looks`): `ego_now` ("where is the thing you can see, from your camera") and `ego_mem`
("as you last saw it, where is the thing, from your camera now"). At its end (`JobSpec.ends`) it asks `object_where` and
`task_things` (`object_was` and `same_place` are not asked: 0.10 a
medium case on this job), with the predicate "on the floor of the rooms you cleaned": the things on
the robot's floor (over 0.02 m) inside the rooms of the task's `cleaning` steps, doorway ways and a footprint across a
room's edge being don't-care; light furniture and people are don't-care by what they are. Its brief names the two look
kinds in one sentence. `adjacent` is retired. Its offsets are credited on the clearance C = 0.10 m and its places on the
re-find radius R = 1 m, and a list of places matches within 0.5 m ([questions.md](questions.md), "Credit"). The
sections below describe the decisions as 0.5.0-control built them (history).

## The decisions, as built

Decision points come from the plan and the doorway coins alone (the world moves where they fall, never whether they
exist). In 0.5.0-control `vacuum/2` asked `zone_clear` (now) at `before_room`, the doorway into a room to clean, and
nothing at `before_doorway`, `room_done`, `cleaning_stopped` or `before_backout`; its end families (`JobSpec.ends`) were
`object_where` and `adjacent`, and `adjacent` asks only on a held-out layout, so it asked nothing on `apartment_a`. Its
place ladder (`JobSpec.ladder`) is `(0.25, 0.5, 1, 2)` m, whose first rung doubled is the 0.5 m matching radius: a home's
rooms are 3-5 m and its things seen at 1-3 m. The gate checks decided the rest
([questions.md](questions.md) has the table):
- M2's fast gate check on look-dev seeds 141-145 dropped `out_of_view` and `uncovered` (their golds were empty after
  their declared fallbacks: 93% at 0.1 m, and 100% where a task stops) and took `order` and `distance_between` out of
  `ends` (under one question a case). 0.5.0-control does not bring `uncovered` back: the 8 cm camera never certifies a
  0.5 m cell wholly free and unswept.
- M2's gate check on the dev render dropped `changed` (it failed `memory_needed`, `shortcuts` and `separation`: 77% of
  its golds were the empty set, which the current view and the question's form guessed).
- The binding gate check over the shipped cases dropped `when_seen` from `ends`: on its 13 questions it fails
  `memory_needed` and `shortcuts` (`current_view_prior_in_pool` and `blind_in_pool` +0.269 each) and `blind` and
  `path_log` are BROKEN. 0.5.0 dropped `where_was_i`
  (a lookup of the frame header's own pose), so until 0.5.0-control the vacuum asked nothing during the job.
- `path_ahead` and `rear_clear` are not asked, because the back-out gate failed and no plan carries a back-out (below).

| Task | Decision sequence |
| --- | --- |
| `clean_rooms`, one room, not the hall's | `before_room` (transit), `room_done` (last cleaning frame), `before_doorway` (transit); a skipped room: `before_room` alone |
| `clean_rooms`, rooms A then B | `before_room`, `room_done`, then `before_room`, `room_done` for B (no decision into or out of the dock's room, the hall), then `before_doorway` out of B; each room skipped drops its `room_done`, and `before_doorway` follows the last room only if it was cleaned |
| a task that stops | `before_doorway` into its last room (not eligible), `cleaning_stopped` in place of the last `room_done` |
| `resume` | `before_doorway` (not eligible), `room_done`, `before_doorway` |
| a site room of a `clean_rooms` task | one `before_backout` (transit) between the doorway decision and the room's first cleaning (none shipped: see below) |

The four events are `vacuum/1`'s: "when you finished cleaning" is the last cleaning frame of the task (the stop, in a stopped
task). A task that skips every room has no cleaning frame, and its two cleaning events resolve to nothing.

## The doorway coin (0.5.0-control, `job_vacuum2_doorway.py`)

Before it drives into a room the robot has a question a controller asks: is the doorway ahead clear? The answer must
vary and the robot must act on it, so a coin sets the doorway and the robot then responds. The question is `zone_clear`
over the way through the interior opening ahead (`zone_shapes.doorway_way`: across the opening less 0.1 m from each jamb,
from the wall's plane on the door's swing side to 0.3 m past its other face, 0.3 m past both for the arch), judged for
anything taller than 0.02 m, the vacuum's climb (`ZoneRule("the doorway ahead of you", DOORWAY, height_m=0.02)`).

- **Which rooms** (`job_vacuum2_plan.eligible`): the rooms of a `clean_rooms` task, but not the dock's room (the hall: no
  doorway is crossed to enter it), not the room a stopping task stops in (the resume sweeps what that task's brush left, so
  the stop must be reached) and none of a `resume`. The decision there is `before_room`; every other doorway entered, and
  the one out of the last room when it was cleaned, is `before_doorway`, which asks nothing (an exit is never blocked, and
  asking it would add rows that are always yes). A robot that has skipped a room it was crossing another to reach stands in
  that one already: no doorway ahead, no decision, no coin.
- **Where the decision stands.** `decision_mark`: 0.5 m of path before the doorway's crossing, walking back over the
  segments. `vacuum/1`'s `crossing` stops at the start of the segment that holds the crossing, and a path that turns at
  the door left the decision 0.19 m from it, too near for a blocker to lie in the way and stay clear of the body (the
  doorway's nearest way, `zones.zone_of`, is the doorway of the decision, which `Coins.close` checks).
- **The coin.** `make_rng(map, seed, "vacuum/2", "doorway", task, room).random() < 0.5` says blocked. It reads no world, and
  it is drawn when the decision is reached.
- **Before the task is planned** every doorway is cleared (`clear_ways`): each carried thing or floor clutter piece standing
  in the way of any interior opening moves out by the world's own law (`move_item`, `clutter.move_floor`) at `t_edge`, the
  task's start less `changes.EDGE_BUFFER` (0 for task 0), a whole second inside the gap after its changes. That restores the
  floor the last task's blockers closed, and a clear coin finds nothing there.
- **When the task is planned up to a blocked decision**, `Coins.close` moves one thing into the way (two for the 1.2 m arch,
  `SINGLE_MAX_M`) at the same `t_edge`, so no change is ever seen: a carried thing of the home's floor kinds (bag, box,
  shoes, toy), else floor clutter taller than 2 cm, drawn from `make_rng(.., "doorway_at", task, room)`. The things lie with
  their centres within 0.12 m of the wall's plane and in the way, the group centred on the gap within 0.1 m, so that the
  widest free width across the opening is under 0.38 m (the body 0.34 m and twice its 0.02 m clearance: the planner
  cannot pass), and at least 0.24 m (its radius, its clearance and 0.05 m) from every point of the path so far, the
  decision pose included, where it turns round. They skip `FloorGuard`, as the AMR's drum does. The planner is then asked
  (no free cell in the opening's gap); a coin that finds no thing, no spot, or a doorway the planner still crosses is a
  `ValueError`, and the seed is skipped by the declared rule: never a doorway that says blocked and is not.
- **The response**, written on `DecisionPoint.response`: "go" through a clear doorway (nothing changes), "skip" for a blocked
  one. A skip cuts the way in at the decision's `Mark` (the primitives up to it are the clear case's, so the keyframes up to
  and including the decision are the same whichever way the coin falls), leaves the room uncleaned and plans what is left
  on the free space with the blockers in the world: to the next room's doorway, or home, the brush off. The rest of the
  script goes on from the world as the gap leaves it; the next task clears the doorways again before it is planned.
- **Skipped rooms shorten tasks, and a task can clean nothing.** A skip drops the room's cleaning and its `room_done`
  (and `before_backout`), and an exit decision follows a cleaned last room only. `rooms` in the brief is unchanged: the
  brief states the assignment, not what became of it.

## The subgoal (`job_vacuum2_subgoal.py`)

Each header says what the robot is doing now, as part of its task, from the plan's room words and the phases and decisions
so far; the text of keyframe k reads the phases up to it, the kinds of the decisions up to and including it and the
responses of the decisions strictly before it, never a coin decided at or after it (`tests/test_job_vacuum2_subgoal.py`
cuts the timeline and flips responses to show it). The forms:
- undocking and the docked keyframes before it: "leaving the dock to clean the kitchen, then the hall" (a resume: "to finish
  cleaning the bedroom");
- transit to a room: "driving to the kitchen to clean it (room 1 of 2)" (a resume: "to finish cleaning it"; no count for
  one room), the same at the `before_room` decision whichever way its coin falls;
- cleaning: "cleaning the kitchen (room 1 of 2)" (a resume: "finishing the bedroom");
- after a skip, to the next doorway decision or the end of that transit: "leaving the kitchen uncleaned: its doorway is
  blocked";
- transit with every room handled: "returning to the dock"; then "docking" and "docked".

## What the checks hold (`job_vacuum2_checks.py`)

Per task: the phases run docked, undocking, transit, cleaning and transit once for each room cleaned, docking, docked
(`job_phase_order`); the brush is on exactly in `cleaning` (`job_brush`); the first and last frame stand at the docked pose
(`job_dock`); the decisions are exactly the plan's sequence given the responses, each in its phase, with `room_done` or
`cleaning_stopped` on the last cleaning frame of its room (`job_decisions`); the task takes at most 20 minutes
(`job_task_time`); the targets are the plan's rooms (`job_targets`). The coin, over the world the script holds: a
`before_room` is answered "go" or "skip" and no other decision is answered (`job_response`); it is answered "skip" exactly
when something stands in the way of the doorway ahead at its time, and at a task's start nothing stands in any doorway but
at the doorways its skips are answered at (`job_doorway`); the things that close a doorway leave no free width across it as
wide as the body plus twice its clearance (`job_doorway_gap`) and keep the body's clearance and 0.05 m from the path up to
the decision (`job_doorway_near`). Per script: each gap is 10 minutes to 3 hours (`job_gap`), the span at most 7 hours
(`job_span`), a `resume` follows a task that stops and names the room it stopped in, and a task that stops is followed by one
(`job_resume`). `build._generate` and `stbench generate` run `validate` (these included) before a script is written: an
invalid script is refused, naming its seed, before anything is rendered.

## Windows

`case.windows(n, rng, starts)`: the first task of a window is uniform over the tasks whose kind the job does not `continue`
(a `resume`: its kind would reveal that earlier tasks exist, and its path the earlier coverage), and a window's size is
uniform over the sizes some allowed start leaves room for; with every start allowed it makes `rng.integers` calls as before.
`build.write_cases` passes the starts. A 4-6-task script gets one window of each length (3 cases), a script of 2 or 3 tasks a short and a medium one (2). The
0.11.0 release has 16 scripts of each job, 12 of 2 or 3 tasks and 4 of 4 to 6, so 16 short, 16 medium and 4 long cases a
job (0.4.0 shipped `short` and `medium` cases only).

## Back-outs: the mechanism, and the gate that failed

The mechanism is `job_vacuum2_backout.py`: in a site room of a `clean_rooms` task (a room holding a fixed piece of
`SITE_KINDS` whose underside the vacuum passes under) the cleaning begins with one back-out: from where it is the robot goes
to an approach point outside the piece's footprint, turns onto an approach line through the piece's centre (its two axes,
both ways, offsets 0, +-0.1, +-0.2 m across, the approach point 0.25 to 1.2 m outside the footprint's edge, the first valid
line), drives in 0.3 m to the footprint's middle (stopping 0.25 m short of what it would meet, where it can: it sees it, it
has no contact sensor), marks `before_backout` facing in, and reverses to the approach point with its heading kept. No
back-out in a resumed room. A room with no valid line raises, naming the seed and the room.

**It ships off: `SITE_KINDS` is empty and no plan carries a `before_backout`** (the kind stays declared). The
feasibility gate was run over seeds 101-200 (free, no render; the corridor's cells are the fine cells `rear_clear` would
count, those before the first thing in it: over the whole 3 m of the corridor (e) can never reach 0.8, since the floor
behind a back-out here is under a metre clear). With the 0.5 corridor drop of part 2 stood in by a draw (a 0.25 x 0.15 m
box 0.3-2.0 m behind the approach point, clear of what is there and of the swept bodies):

| Measure | Living room (coffee table), 128 back-outs | Kitchen (dining table), 116 | Needs |
| --- | --- | --- | --- |
| (a) worlds with a back-out | 100 of 100 (and 200 of 200, seeds 101-300) | 100 of 100 (200 of 200) | every world |
| (b) corridor swept before the decision, median | **0.58** (0.12 over all 3 m) | **0.62** | at most 0.4 |
| (c) rungs of the true clear distance behind | 5, the largest holds 0.47 | **3**, the largest **0.54** | 4 or more, none over half |
| (d) drop realised (stand-in) | 0.56 | 0.45 | 0.4-0.6 |
| (e) corridor in view before the decision, median | 1.00 (0.30 over all 3 m) | 1.00 (0.18) | at least 0.8 |
| (f) rungs of the true clearance ahead | 7, the largest holds 0.50 | 6, 0.47 | 4 or more, none over half |
| (g) clear distance behind in the map raycast's rung | **0.47** | **0.87** | at most 0.4 |
| (h) the same ahead (reported) | 0.29 | 0.16 | |

(b) and (g) fail at both sites and (c) at the kitchen. The reason is the floor plan: the clear floor behind a back-out is
0.4-1.0 m (the counters, the walls, the sofa), so the drive-in itself (0.3-0.5 m) is most of the corridor and a map raycast
gives the rung whenever nothing is dropped in it. The desk is closed by its chair in 99 of 100 worlds, and at the hall
console (0.3 m deep, against the wall) the clear floor ahead is under 0.25 m in every world that has a line (33 of 40
tried). Variants tried at the living room, over the same worlds, (a) /
(b) / (f)'s largest rung / (g): the first valid line with the approach point 0.25-0.6 m out and no standoff 94% / 0.42 /
0.52 / 0.42; the standoff and three distances 96% / 0.40 / 0.56 / 0.34; the shipped five distances with the standoff first
100% / 0.58 / 0.50 / 0.47; the standoff or none for each line before the next 100% / 0.49 / 0.69 / 0.41. No variant meets
(a), (b), (f) and (g) together, so `rear_clear` and `path_ahead` are not mapped to the vacuum's back-outs (`rear_clear`
goes to M3's AMR; no job asks `path_ahead`) and none ships. `tests/test_job_vacuum2_backout.py` tests the mechanism with `SITE_KINDS` patched.

## Sizes (seeds 101-200, back-outs off; before the doorway coin)

A script of 4-6 tasks holds 355-1,533 keyframes (median 792, mean 841; the design's long case was about 720) and takes 1.2 s
to generate (median; 2.5 s at most; before the world's processes and the residents). The longest task takes 794 s, 66% of
`TASK_MAX_S` (20 minutes). By task kind and rooms (keyframes minimum/median/maximum; minutes median/maximum):

| Task | Tasks | Keyframes | Minutes |
| --- | --- | --- | --- |
| `clean_rooms`, one room | 291 | 25/106/352 | 2.7/9.3 |
| `clean_rooms`, two rooms | 118 | 89/308/486 | 7.9/13.2 |
| `resume` | 94 | 32/76/271 | 2.0/7.2 |

A task holds 3-5 decision points (mean 3.35); counted from the task's first step with a look-window cut after every decision point
(`look_windows.windows`), 2.3 a task (70% of them) fall after its first two look windows, and no task has none. The realised
shares of seeds 101-200: 0.289 of the 409 `clean_rooms` tasks have two rooms (the plan's 0.25), 0.315 of the 298 tasks
allowed to stop do (0.3). `uncovered` at the end of a room (a free upper bound: the 0.5 m cells wholly inside the room,
touching no fixed piece the robot cannot pass, that no brush sweep of the room, the stopped task's included for a resume,
touched at all, observation ignored): 82.7% of the 527 `room_done` rooms hold at least one such cell (median 2, at most 12,
mean 2.4), and 97.9% of the 94 `cleaning_stopped` rooms (median 4, at most 16, mean 5.2).

## With the world's processes and the resident (seeds 101-130)

The same scripts as generated at the merge, `world_processes` between the tasks and `people.add` over the finished script
(`stbench generate --tasks 4-6`, then `stbench validate`, the leak scan included): all 30 are valid and make 90 cases (3 a
script). A script of 4-6 tasks holds 498-1,533 keyframes (median 879, mean 949) and takes 2.8 s to generate, validate and
write (median; 4.9 s at most). The resident is home at the start of 56% of the 155 tasks and at 58% of the keyframes; with
occlusion ignored it has a body part inside the camera's image (a margin of 40 px) in 9.5% of them (median per script 10.7%,
at most 17.9%). The `uncovered` bound: 75% of the 174 `room_done` rooms (the hall's 12 none) and
all 24 `cleaning_stopped` rooms hold a cell that could be positive. Tests (`tests/test_world_in_scripts.py`): every change
of a non-mover entity falls at a whole second inside a gap, 30 s from its ends and at one instant a gap (118 changes in
four scripts, 14 of them doors); none falls between two keyframes of a task; the doors on a task's ways are open through
it; every case clock ends by 17:00.

## The doorway coin on seeds 101-130 (the free generation check, 0.5.0-control)

`stbench`'s `build.generate_checked` (the generator, `validate` with the job's checks, and `episode.write_script`) over seeds
101-130 of `apartment_a`, `--tasks 2-6` (each seed draws its own count), no render; then `stbench validate` on every
written script, the leak scan included. A seed that raises is skipped by the declared rule.

| Measure | Seeds 101-130 | Seeds 131-230 (a wider look) |
| --- | --- | --- |
| scripts made / seeds | 30 / 30 (skip rate 0) | 99 / 100 (skip rate 0.01: seed 131 found no pair of floor things that close the arch with the first room's path beside it) |
| `stbench validate` | 30 of 30 clean | 99 of 99 clean |
| tasks | 125 | 386 |
| `before_room` decisions | 105 (0.84 a task: 33 tasks none, 79 one, 13 two) | 295 (0.76 a task) |
| answered "skip" / "go" | 54 / 51 (blocked share 0.514) | 127 / 168 (0.431) |
| by doorway, "go" / "skip" | D1 10/4, D2 10/19, D3 17/14, D4 10/10, O1 4/7 | |
| things in a blocked doorway | one 47 times, two (the arch) 7 times; shoes 22, boxes 19, bags 13, toys 7 | |
| the decision's place | 0.27-0.50 m from the door line (median 0.33), 5-315 steps into the task (median 7; 6 of 105 past 16) | 26 of 295 past 16 |
| nearest blocker to the base at the decision | 0.247-0.471 m (median 0.342; the least a blocker may be is 0.24) | |
| tasks that clean no room | 35 of 125, each by skipping every room it was to enter | |
| a person in the way at a decision | none of the 105 | |

- **The supply the questionnaire sees was far below the decisions made, until its rule changed.** `questionnaire.
  decision_points` asked only at points 16 or more steps into their task (`GAP_STEPS`). A first room's doorway is 2-4 m
  from the dock, so its decision falls 5-15 steps into the task: 368 of the 400 `before_room` decisions of seeds 101-230
  do (151 at step 5), and the 32 past 16 (0.06 a task) are the entries of second rooms. Under that rule the vacuum
  asked `zone_clear` 0.06 times a task. A `now` question is answered from the look window it follows and needs no history, so a
  kind that asks only `now` families is now a candidate from `NOW_MIN_STEPS` (4) steps into its task (the spacing of
  asked points stays 16): the 105 `before_room` decisions of seeds 101-130 are all candidates, and the draw (the
  windows `case.windows` draws, 60 case ids each) asks `zone_clear` 0.90 times a short case, 1.93 a medium and 3.22 a
  long one, where the 16-step rule gave 0.00, 0.10 and 0.17; over seeds 101-230, 0.82, 1.65 and 2.96 against 0.05, 0.13
  and 0.22. A short case (one task) asks none when its task has no `before_room` decision, 33 of the 125 tasks of
  seeds 101-130 (26%). The exits, which lie late, are never blocked.
- **A skip shortens the task** (a task that skips every room is the dock, the hall and a doorway: 15-25 keyframes, median 19) and
  leaves the rooms it skipped unseen: their critical things are in `object_where`'s unknown pool, not its answerable one.

