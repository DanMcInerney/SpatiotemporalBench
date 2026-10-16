# `helper/1`: the home helper's job (M3, `job_helper*.py`)

Scripts of one to six tasks on `apartment_a` (`stbench generate|build --tasks N|A-B`): a Stretch-class home helper (0.34 x
0.33 x 1.41 m, a 1.30 m head camera at -15 degrees, `robots.ROBOTS["helper"]`, M0's numbers) waits on the clear apron of
the home's charging corner, carries a thing from one surface to another or brings one to its spot, and parks again, with
unobserved gaps (and their world changes, and the resident) between tasks. Its camera sees counters, tables and shelves
the vacuum's 0.08 m camera never could: that is what the job adds.
The plan and the gaps are `job_helper_plan.py` (the map and the seed alone), the surfaces it can reach and the poses it
reaches them from `job_helper_reach.py`, the path and the declarative part `job_helper.py`, the thing it carries
`job_helper_carry.py`, the window it puts it down in and the coin that sets it `job_helper_spot.py` (on
`clutter_surface.py`'s moves), the words for what it is doing `job_helper_subgoal.py`, the job's own `validate` rules
`job_helper_checks.py`. No map edit, no render change, no change to a robot's numbers.

**0.5.0-control.** The carry is real: the thing a task moves is a world entity that leaves its
source and appears at its target, the camera tilts down for the grasp and the place, a fetch puts the thing on a surface
and drives home, the helper asks `zone_clear` once a task, at `before_place`, about the spot it is about to put the thing
down on (a coin sets the spot occupied or free before the task and the helper responds after the ask), and each frame's
header says what it is doing (the subgoal). The helper's scripts are rendered again.

## What the design asked, and what the home allows

The design asked for laundry from the washer-dryer to the wardrobe and a drink from the fridge
to a table in another room. On `apartment_a` the helper cannot reach either: the planner keeps 0.355 m from every obstacle
(`route.robot_free_space`: the circumscribed disc of its 0.34 x 0.33 m body, 0.237 m, its 0.10 m clearance and a cell's
half diagonal), and the home was drawn for a 0.17 m vacuum. Measured over every face of each piece (front edge 0.3-0.6
m out, slides of 0.4 m, in the piece's room, reachable from the park pose), with the clearance at 0.10, 0.06, 0.03 and
0.0 m:

| Piece | Why nothing can stand within reach of it |
| --- | --- |
| fridge `F01` | in the kitchen's corner, its front face against the end of counter `F02` and its north side against counter `F06`: no free floor on any side |
| washer-dryer `F08` | in a pocket (x 0.65-1.4, y 2.1-2.75) closed by the dining chair `L01`, which always stands in the 0.75 m column, and by a 0.45 m strip between the dining table and the north wall |
| nightstands `F31`, `F32` | in channels 0.45 and 0.65 m wide between the bed and a wall |
| desk `F33` | its chair stands in front of it |
| wardrobe `F34` | only its west end has a free pose, and it has no surface to take a thing from or put one on |

Lowering the clearance does not help (the pieces are boxed in by other pieces), so this is a map matter, not a robot
number, and neither is changed (M2's dev render runs on this map, and the helper's numbers shape the
frozen home's worlds through the floor guard and the dressing guard). The call (2026-10-07, option 2): the
job takes its sources and targets from the surfaces the helper can reach, found by a rule over the map and the robot, and
the laundry and fridge words are dropped (a stated simplification: a job must run in every world).

## The plan

- **Task kinds.** `tidy` carries a thing from a surface in one room to a surface in another room; `fetch` brings a thing
  from a surface in a room other than the park room to a surface of the park room, the counted piece nearest the park pose
  (`job_helper_plan.fetch_target`, from the map alone, no draw: the hall console F20 on `apartment_a`, 0.95 m from the
  pose), and then drives home to its spot. Each task is a `tidy` with probability
  0.5 (`TIDY_P`, a modelling choice: no question depends on it). A tidy's source room is uniform over the qualified rooms,
  its target room over the others; a fetch's source room is uniform over the qualified rooms but the park room. The piece a
  task prefers in a room is uniform over the room's counted pieces. Every task draws the same five numbers in the same
  order, so no draw depends on a task's kind. A map with fewer than two qualified rooms, or whose park room is not one,
  refuses the job (`ValueError`, naming the map).
- **The gaps** are `job_gaps.gaps` with the helper's id and `TASK_MAX_S` = 600 s (a check; the longest task measured is 137
  s): n - 1 of them, each log-uniform from 10 minutes to 3 hours, shrunk to `SPAN_S - n * TASK_MAX_S`, so a script spans at
  most 7 hours and every case clock ends by 17:00. They depend on the seed alone (`gaps(map, seed, n)`).
- **The brief** names the rooms only: "Its current assignment is to carry something from the living room to the hall." /
  "...to bring something from the living room to its spot." (`regions`, `source`, `target` and `park_room` are the job's,
  never delivered; a fetch's `target` is `fetch_target`).

## The park pose

On the axis of the map's one `dock` piece, 0.63 m in front of its front face, facing the centre of the map's outside door:
(5.40, 5.05) heading -90 degrees on `apartment_a`, inside the dock's apron (1.2 m in front of the face, 0.5 m beyond each
side), which the clutter and the gaps' light furniture keep clear (`anchors = {"dock"}` keeps the light starts off it too).
One pose a map, never a world's. Why not the hall's anchor (5.4, 3.0): something stands within the planner's reach (0.337
m) of it, by the plan review's measure, in 51 of start worlds 101-200 (extension cords, cable bundles, power strips) and in 55 after five gaps, so a fixed
pose there collides, and a pose found per world would leak the clutter through the path; on the apron the pose is clear in
every world (the review's measure of 100 of 100, the fixture's worlds after up to five gaps here).

## The reachable surfaces (`job_helper_reach.py`)

- **A reach pose** is a cell of the helper's free space in the world (with every interior door off the task's ways shut in
  both leaf positions) whose centre lies 0.47-0.77 m from a surface piece's footprint (the front edge 0.3 m, `REACH_M`, to
  0.6 m, `REACH_MAX_M`, from it), in the piece's own room and reachable from the park pose, the base facing the nearest
  point of the footprint. Of those cells the pose is the one nearest the ideal: on the piece's front axis, the front edge
  0.3 m from the face (so a slide along the face, a back-off, or a corner, in order of distance).
- **A surface piece** is a fixed, blocking piece with a surface (a counter, a table, a shelf; not a rug).
- **A piece counts** when at least 0.15 m2 of the map's own floor (every piece at its map place, the interior doors open, no
  clutter, nothing carried) is a reach pose of it; **a room qualifies** when at least two of its pieces count, so a world
  that clutters one piece's reach floor leaves another of the room (`MIN_REACH_AREA_M2`, `MIN_ROOM_PIECES`, the two
  constants chosen from the measure below). The script tries the piece a task prefers first, then the room's others in id
  order: **the world moves where a reach falls (which piece of the room), never whether the task exists or in which room**,
  and a room with none raises, naming the job, map, seed, task and pieces.
- **Measured** on the start worlds of seeds 101-130 (clutter, light furniture at their starts; `keep_open` = the hall and
  the piece's room), the share of worlds in which each piece has a reach pose, and the map's own reach floor:

  | Piece | worlds with a pose | map floor (m2) | counts |
  | --- | --- | --- | --- |
  | coffee table `F11` | 30 of 30 | 1.19 | yes |
  | hall console `F20` | 30 of 30 | 0.21 | yes |
  | TV console `F12` | 29 of 30 | 0.67 | yes |
  | sofa `F10` | 26 of 30 | 0.35 | yes |
  | vanity `F40` (the bathroom's only counted-size piece) | 25 of 30 | 0.40 | yes, but one in its room |
  | shoe rack `F21` | 23 of 30 | 0.42 | yes |
  | dining table `F07` (the kitchen's only one) | 29 of 30 | 0.06 | no |
  | bookshelf `F13` | 7 of 30 | 0.11 | no |
  | bath shelf `F43` | 0 of 30 | 0.06 | no |
  | counters `F02 F03 F04 F06`, bed `F30`, nightstands `F31 F32`, desk `F33` | 0 of 30 | 0 | no |

  So the helper works in two rooms, the hall (console `F20`, shoe rack `F21`) and the living room (sofa `F10`, coffee table
  `F11`, TV console `F12`). **No single piece is the guarantee** (the shoe rack has a pose in 23 worlds of 30, the sofa in
  26): the room is, through the fallback: the job generated with 0 of 200 scripts (seeds 101-200, one task and 4-6 tasks)
  raising, over 897 reach poses of which 65 (7%) were another piece of the room than the one the plan preferred (and
  0 of 200 more on seeds 201-300, over 879 poses and 59 such). The
  per-piece threshold (29 of 30) the design asked for cannot be a rule over the map (the hall console's pose lies
  inside the dock's apron, which the world keeps clear, and the shoe rack's floor is as large as the TV console's), so the
  guarantee is the room's and is measured, not declared.
- A tidy therefore goes hall to living room or back, and a fetch from the living room to the hall: every task crosses
  the doorway into the living room twice, and plans differ by kind, direction and the pieces the room offers.

## The path of a task

The script calls the world's gap callback before each task (`world_processes.gaps`) with the park room and the task's two
rooms as `keep_open`, then plans on `route.robot_free_space(..., shut=<every interior door off the ways between them>)`:
each such door is an obstacle in both leaf positions, so no door's state can change a path and the robot's own path log
learns none (the doors on the ways are kept open by the callback). The phases (a tidy and a fetch run the same):

| Task | Phases, in order |
| --- | --- |
| `tidy`, `fetch` | waiting (2 s at the park pose), navigating to the source's reach pose, the turn to face the piece, the camera tilting down, grasping (10-30 s held still), the camera tilting up (the first frame of carrying), carrying to the target's reach pose, the turn, the camera tilting down, placing (8-20 s), the camera tilting up (the first frame of navigating), navigating back to the park pose, the turn to the park heading, waiting |

- **Facing is exact.** The turn to face a piece (and the park heading) is made however small it is (`_face`, where
  `Driver.turn` leaves a turn under a degree to the next segment), so the `before_grasp`, `before_place` and park keyframes
  face their piece within a micro-radian. The driver is `exact` (every vertex is a keyframe).
- **Speeds and status.** The navigating and carrying speeds are drawn once a task from `ROBOTS["helper"].speeds_mps`; the
  grasp and place durations too, from the task's stream, `make_rng(map, seed, "helper/1", "job", i)`, after the battery
  (an integer drawn in 80-100 falling 0.5% a minute: a modelling choice, a mobile manipulator's few hours on a charge). The
  status is `{battery_pct}`; every task starts waiting at the park pose with a fresh battery, so nothing carried over shows
  where a window sits. No new draw moves one of these: the thing, the coin and the response have streams of their own.
- **The camera tilts.** The head camera is at its default pitch, -15 degrees, while the helper drives and waits, and at
  -45 degrees (`TILT_DEG`) from the `before_grasp` frame through the last grasping frame and on the `before_place` frame
  and through the placing hold (`TILT_DPS` 30 a second, a modelling choice: a pan-tilt head; a 30-degree tilt is one
  second, a frame at its end). At 1.30 m, the picture's middle line at -45 degrees meets a 0.8 m console 0.5 m ahead
  of the camera and a 0.35 m shoe rack 0.95 m ahead (the reach puts the footprint 0.47-0.77 m from the base), where
  -15 degrees saw the low ones at the picture's edge. Each frame carries its own
  pitch (`camera_pose` of the frame's pitch; the pose rows and the header already show it); the limits are -60 to +20
  degrees (`robots.py`). The base stands at z 0.
- **Not drawn:** the helper's body, arm and gripper, and what the gripper holds between the grasp and the place (the
  robot's own payload, as the vacuum's brush is): the thing has no placement then (below).

## The real carry (`job_helper_carry.py`)

A task's thing is a world entity. With the two reach poses known (`job_helper._spot`), `prepare` does, in this order:

1. **The windows.** The source window and the target window are `zone_shapes.surface_window` of the two poses: 0.4 m
   of the largest surface of the piece the base faces, centred on the heading ray's hit point, across the surface's depth.
2. **The thing** (`choose_thing`, a function of the world as the gap left it and the plan alone): the carried item on the
   source piece nearest the source window's centre, within 0.3 m of the window (the arm's reach), small enough to lie on
   the target surface and with a place for it when the target window is blocked (`placeable`). When the piece holds
   none, one such item of the world that lies on a raised surface is put in the window `EDGE_BUFFER` (30 s) before the
   task starts (at 0 for the first task): a seeded draw from `make_rng(map, seed, "helper/1", "thing", i)` among the
   items whose kind lists the source surface as a receptacle, then the rest, to the free spot nearest the window's
   centre. Failing both: the nearest item on the piece at any distance, or one put anywhere on the piece. (On
   seeds 101-130, 47% of the tasks found a thing already on the piece and 53% had one put in reach; none needed one put
   anywhere on the piece.)
3. **The coin and the response** (below), then **the place** the thing ends at.
4. **The placements** (`commit`, after the frames are made): the thing's placement on the source ends at the time of the
   task's **last grasping keyframe** (`ledger.remove`), and a new one starts at the time of the **last placing
   keyframe** on the surface it is put on (`on` = that piece's entity). Between them it has no placement, so
   `render/build_scene.Scene.apply_time` hides it and no `render/` change is needed. `w.carried[eid].spot` is updated, so
   the next gap's processes move it like any carried thing.
5. **`Task.targets`** is (source room, target room, the thing, the source piece's entity, the target piece's entity): the
   pieces are the ones the reach poses face, which the world may move from the plan's preferred ones. The thing is never
   asked about at the end (`World.told()`): the subgoal and the robot's own log name it, its source and its target.

A task the world gives no thing or no place for raises `ValueError`, naming the job, the map, the seed, the task and the
reason, and the seed is skipped by the build's rule. `validate` has a generic rule for the carry (`task_placement`): a
placement of a non-mover entity that starts strictly inside a task belongs to an entity in the task's `targets`; the
world changes only in the gaps, and in the carry.

## The place spot: window, coin and response (`job_helper_spot.py`)

At `before_place` the helper asks itself whether the spot it is about to put the thing down on is clear. The zone rule is
`ZoneRule("the spot in front of you where you are putting it down", SPOT, height_m=0.01, top_m=0.5)`: the target window,
judged from its surface's height, a thing taller than 0.01 m and based lower than 0.5 m above it blocking it.

- **The coin.** `job_helper_plan.spot_blocked(map, seed, i)` is `make_rng(map, seed, "helper/1", "spot", i).random() <
  0.5`: it reads no world. It is enforced on the world `EDGE_BUFFER` before the task, after the task is planned
  (`enforce`, draws from `"spot_world"`), on surfaces only, so the floor's free space and every planned path are what
  they were:
  - **clear**: everything that stands in the window grown by 0.05 m (`DEEP_M`) moves off it, to a place that is not in it.
    A carried item moves by the world's own law (`world_processes_home.move_item`) tried on a copy of the world and kept
    only if it ends on a raised surface (`clutter_surface.move_to_surface`: a floor drop is thrown away and the item drawn
    again); a surface clutter piece by its kind's places (`clutter_surface.move_surface`, mirroring `clutter.move_floor`),
    else slid to the nearest free spot outside the window (`slide_surface`).
  - **blocked**: when nothing stands in the window shrunk by 0.05 m, one carried item that lies on a raised surface (not
    the thing) moves onto it, reaching more than 0.05 m into it (`placement.sample_box`).
  - A failed enforcement is not an error: the response follows the world as it then is. Of 147 tasks (seeds 101-130) the
    coin was blocked in 67; 62 of them were blocked, 5 stayed clear (no item found a place on the window), and of the 80
    clear coins 71 were clear and 9 stayed blocked (a clutter piece that could not leave a crowded console).
- **The response** (`DecisionPoint.response`, written after the ask, so the during question at `before_place`, which reads
  the look window up to the decision, never sees it): `elsewhere` when something stands in the window by more than 0.01 m
  (`SLIVER_M`: the zone grid's sub-cell is 0.02 m and counts when its centre is inside), else `go`. It starts at the
  frame after the decision keyframe.
  - `go`: the thing goes to the window's centre (the free spot nearest it when it is wider than the window and meets a
    neighbour), a quarter turn to the surface, its longer side along the surface's.
  - `elsewhere`: the free spot of the same surface nearest the window's centre that is outside the window (5 mm
    clear of it: `place_spot`, a scan on a 0.02 m grid, no draw), else the target room's other counted piece, in id order: the helper
    drives from its decision pose to that piece's reach pose (tilting up and down again; this drive is `placing`, so
    `before_place` stays the last carrying frame) and puts the thing at the free spot nearest that piece's window. A task
    with no place for it raises `ValueError`. (26 of 147 tasks, 18%, ended on the room's other piece: the hall console is
    crowded.)
- **Identity up to the decision.** The thing, the reach poses, the path, the frames, the statuses and the subgoal up to
  and including `before_place` are the same whichever way the coin falls (the test forces the coin both ways); only the
  response and what follows differ. The coin and the enforcement read no keyframe, and the thing is chosen before them.
- **Why the gold is balanced.** The coin reads no world; the path to the decision is planned before the enforcement; the
  subgoal at the decision and before is coin-independent; the realised blocked share was 0.48 on seeds 101-130 (declared
  band 0.4-0.6).

## The subgoal (`job_helper_subgoal.py`)

One text a keyframe, from the timeline alone and causal: it reads the task's kind, its thing and pieces (`Task.targets`),
the phases up to and including the keyframe and the `before_place` response strictly before it (the test cuts the ego after
each keyframe and flips the coin). The thing is named as a person would (`descriptions.phrase`: "red mug"), the pieces by
the map's names ("hall console"; an ordinal if a name repeats on the map).

| Where | Text |
| --- | --- |
| waiting, before it has left | waiting at its spot to {move or bring} the {thing} from the {source} to the {target} |
| navigating, before it has placed | going to the {source} to pick up the {thing} |
| grasping | picking up the {thing} from the {source} |
| carrying (the text at `before_place`) | carrying the {thing} from the {source} to the {target} |
| placing, a `go` | putting the {thing} down on the {target} |
| placing, an `elsewhere` | putting the {thing} down on a free spot, since the spot on the {target} is occupied |
| navigating, after it has placed | going back to its spot |
| waiting, after it has placed | waiting at its spot |

## What the helper asks

**0.11.0** (`job_helper.py`'s `SPEC`; 0.10.0's table with `way_to` in the second slot and `approach_mem` in the third; the reasoning is
[question-design.md](question-design.md), and each share has its reason in the job file's comments). The questions are decisions its brain faces (harm, help, situation, reuse;
[vision.md](vision.md)), credited by their consequence ([questions.md](questions.md), "Credit"). A look point is any frame
at least 4 steps into its task but not its last, and the frames of `before_grasp` and `before_doorway` come first in
the draw (`JobSpec.look_decisions`: the decisions a fetch turns on; `before_place` is not among them since the final
review's B1: a decision frame puts a structure question at the same plan pose in every script, and with `before_place`
first the helper's `clear_path` memory cells failed the evidence verdict). A point has three slots, harm first;
the share is the percent of points at which a slot draws the family. At its end the helper asks `object_where` and
`task_things`.

| Slot | Family: share | Why |
| --- | --- | --- |
| 1, the harm questions (whole) | `clear_path` 50, `closest_approach` 50 | Half of the slot is the least that gives `closest_approach`, the family with the lowest fill, its 20 rows (`abilities.MIN_ABILITY_N`) over the 36 dev cases (800 points at 0.5 x 0.063 = 25); the corridor, at the same half, has the rows to spare |
| 2, what it saw before | `ego_mem` 65, `way_to` 35 | The fill of `ego_mem` is high, capped by the distinct things a case holds (0.313 a drawn slot on the compile after the repair). In 0.10.0 it had 25 and `person_mem` had 40; `person_mem` left (below) and its 40 went to `ego_mem` at 65, and the slot drew nothing at the other 35 points in a hundred. 0.11.0 gives `way_to` ([families/way_to.md](families/way_to.md)) those 35, so `ego_mem` keeps its 65 and its supply |
| 3, the rest | `ego_now` 20, `open_heading` 15, `approach_mem` 30 | Sized so that the three slots give about 0.5 questions a point. `open_heading` counts in the headline since 0.11.0, as a clearance question ([families/open_heading.md](families/open_heading.md)); through 0.10.0 it was a diagnostic. 0.11.0 gives `approach_mem` ([families/approach_mem.md](families/approach_mem.md)) 30 of the 65 that the slot drew nothing on, so `ego_now` and `open_heading` keep their shares and the helper loses no rows. The design expects 15 to 35 rows (about 400 points x 0.3 x the evidence rule's fill), which join the large pooled apartment clearance cell |

No score enters a share. The lanes measured, per drawn slot, fills of `clear_path` 0.293, `closest_approach` 0.063,
`ego_mem` 0.582, `ego_now` 0.359 and `open_heading` 0.467 (`person_mem` 0.076, when it was asked). The first compile,
with `person_mem` still in slot 2, gave `clear_path` 0.275, `closest_approach` 0.046, `ego_mem` 0.512, `ego_now` 0.342 and
`open_heading` 0.509 (`person_mem` 0.061) and one look question in 12.5 frames of the short and medium cases (one
question in 9.3 of all, over the one-in-15 floor); the helper has the supply to go higher, and its harm share (39% of its
look questions, 30% of all) is what a larger first slot would lift. That compile was the last one measured on the helper's
old slot 2.

**Measured after the final review's repair** (the dev root, `person_mem` gone from slot 2 and `before_place`
from the decision frames; short and medium cases): 608 look points and 312 look questions, one look question in 11.5 frames
and one question in 8.7 of all (410 questions over 3,577 frames; 12.5 and 9.3 before); the fills per drawn slot are
`clear_path` 0.258, `closest_approach` 0.051, `ego_mem` 0.313, `ego_now` 0.338 and `open_heading` 0.491. All lengths: 513
questions, 118 of them harm (23% of all, 30% of the look questions). The `ego_mem` draws that `person_mem` left yield fewer
than the fill suggested (a thing an earlier look has named is not asked again): 168 `ego_mem` questions over all lengths
(171 before). The helper's clearance cells held 51 questions in `now` and 67 in `memory`, and its heading `now` cell 2 (thin, and a
diagnostic through 0.10.0); in the 0.11.0 release, with `open_heading` and `approach_mem` counted, they hold 53 and 138. On the 0.11.0 compile `way_to` asks 29 questions over the helper's 36 cases (25 on the short
and medium ones: the end questions reserve their things first, so it asked fewer than the 45 the design expected), and
every (job, ability, column) cell of the dataset holds at least 20 questions.

- **`clear_path` in four directions** (`JobSpec.clear_directions` = ahead, back, left, right; each drawn equally by a hash
  of the case and the point): how far it can drive on, back away from a counter, or turn in place and drive, before its
  body touches a table edge, a chair or a console. It is asked while it drives (`JobSpec.clear_phases`: `navigating`,
  `carrying`); in `waiting`, `grasping` and `placing` its status answers it (back, waiting: +1.00). The gold reads what it
  last saw of every cell up to the point; most of the space behind and beside it was not looked at, so the reversed and
  turned golds are often `unknown`, and the job keeps 35% of those (`look_unknown_keep`). A box body's turn sweeps a disc
  of its half-diagonal, and a point where a seen obstacle stands in that disc asks nothing
  ([families/clear_path.md](families/clear_path.md)). All four directions are asked: lane J3 measured the helper's memory
  gate direction by direction to see whether one carried its failure, and no direction explains it (below).
- **`closest_approach`** ([families/closest_approach.md](families/closest_approach.md)): how close the person it can see
  nearest the helper would come to it in the next 5 seconds if both kept moving as they are, in any status phase. The helper's home
  holds one resident or none, which is a generator invariant (`JobSpec.one_person`, and the job's checks fail a world with
  two), so the pairing of a person's sightings across frames, which the sensors do not give, is not ambiguous here: the
  helper never meets the family's `TWO_PEOPLE` dead band, and its dead bands are all of one kind, a person row in the range
  of a "no one". The resident is in view in 4.5% of the
  helper's frames. After the final review's repair the family has 22 questions over all lengths (19 in the short and medium
  cases) on the dev root, 16 with the resident in view and 6 "no one", every one `now`: the robot's own velocity is
  delivered with every step, and the resident never walks at a frame tried, so frame k alone gives each gold. It passes
  every gate (its `memory_needed` gate has no `memory` row to read).
- **`approach_mem`** ([families/approach_mem.md](families/approach_mem.md); 0.11.0): how close the helper would come in the
  next 5 seconds to the person it saw last, where it last saw them, if it kept moving as it is: `closest_approach` for the
  resident out of view, as when it reverses with the resident behind it. It is asked in any status phase, in slot 3, and is a
  harm question of the clearance `memory` cell. The person is the one of the latest evidence sighting, held still where
  they were seen; the point is a dead band when the resident is in view (that is `closest_approach`'s), was never seen in
  the task, was walking at the sighting, or moved more than C/2 = 0.075 m out of view (the truth screen). With one resident
  or none the family never meets its `TWO_PEOPLE` dead band. The design expected 15 to 35 rows, mostly golds of 2-3 m
  from sightings a median 14 s old. **Measured** (the 0.11 code on the dev root's 36 helper cases; 541 questions in all):
  19 rows in 15 cases, all a person (no "no one"), a median gold of 2.36 m (p10 1.09 m, p90 4.45 m), none under the
  threshold 0.49 m, a median age of 7.7 s (13 `recent`, 6 `task`); `ego_now` is 49 questions and `open_heading` 54. Its
  scoped check reads ceiling 1.00, `knowable` +0.00, `memory_needed` +0.00, shortcuts +0.00 and separation 1.00 over
  19 rows in 8 scripts, the best constant `unknown`.
- **`ego_mem`** and **`ego_now`** are asked of every thing (the floor filter applies to the vacuum and the AMR only).
- **`open_heading`** ([families/open_heading.md](families/open_heading.md)): which heading gives it room to turn and
  leave, credited by its regret on the one need rule, `N' = max(C, a tenth of the best run, 2 rho)` with C = 0.15 m, its
  right direction being room to move (a run of the body's length plus C, or longer). It is asked in the same phases as
  the corridor. It is a clearance question and counts in the headline since 0.11.0; through 0.10.0 it was the diagnostic
  ability `heading`, counted in no mean.
- **`way_to`** ([families/way_to.md](families/way_to.md); 0.11.0): which way, of those it has seen, to set off to reach a
  thing it saw before the recent view, by the shortest way its body fits through. It is asked in `DRIVING`
  (`navigating`, `carrying`; `JobSpec.look_phases["way_to"]`), a help question in the `memory` ability, and the gold is
  computed over the floor the helper saw or drove ([docs/questions.md](questions.md), "Credit"). Its need is C = 0.15 m widened
  to `2 rho` = 0.2 m by the field's 0.1 m cell, over a step of R = 1 m.

**`person_mem` is no longer asked** ([families/person_mem.md](families/person_mem.md)). It asked where the resident is
from the camera now, as last seen, in the old slot 2. On the compile after lane J2 it failed `separation`: the table
beat `blind` by +0.368 (interval +0.130 to +0.706) against the gate's 0.4. It had 19 questions, 7 answerable and 12 with
an `unknown` gold. A blind `unknown` scores 1 on those 12 and 0 on the other 7, so the lead of a perfect reader is the
answerable share, 7 of 19 or +0.368, and cannot reach 0.4 whatever the reader does: the family could not tell an agent
that perceives and remembers from one that guesses `unknown`. The family is dropped by the evidence rule and not
re-tuned. The resident in view stays covered by `closest_approach`; where a resident who is out of
view is goes unasked of an offset, an open gap for when the answerable supply allows; since 0.11.0
[`approach_mem`](families/approach_mem.md) asks what that resident means for the robot's motion (how close it would come to
where they were last seen), which needs no `unknown` slice. The invariant `one_person` stays and the job's
checks still hold it: `person_mem` was the family that read it, and `closest_approach` does not, but with one resident
it never meets its `TWO_PEOPLE` dead band.

**`clear_path` no longer fails the evidence verdict, and the first diagnosis of why it did was wrong.** After the first
0.10.0 compile the check's `memory_needed` evidence cells of the helper's `clear_path` failed: a memoryless prior keyed on
the corridor the latest frame shows (`current_view_prior_in_pool`) beat `all_unknown` on the answerable rows by +0.136
(interval +0.022 to +0.283, 62 rows) and on the long cases by +0.177 (18 rows). It was first read as the helper's recurring
routes in one home and accepted as fixed structure. The final review found the cause in the draw: 44 of the 106
rows sat at decision frames, 8 of the carrier gold's 10 at `before_place`, and a decision frame puts a structure question at
the same plan pose, by the same furniture, in every script. With `before_place` out of `look_decisions` (repair R1) the
cells read +0.000 in every case (96 questions: 29 `now`, 55 `memory` answerable, 12 `unknown`; 67 `memory` rows), and every
gate and the evidence verdict pass. No gate, threshold or share changed. The detail is on
[families/clear_path.md](families/clear_path.md).

**0.9.0 (history).** During the job the helper asked, at a look point, `ego_now` (30%) or `clear_path` (70%) in slot 1,
and `person_mem` (45%), `ego_mem` (30%) or `open_heading` (20%) in slot 2; at its end `object_where` and `task_things`.
Share had moved from the questions in view to those that need the space the camera cannot see now. On the J1 compile of
the dev root the job had 37 `clear_path` rows in the short and medium cases (20 `now`, 11 `memory` and 6 `unknown`),
and 19 `open_heading` rows (8 `now`, 11 `memory`; the design declared it at 25, and the budget's stop rule cut it to 20).
The helper's clearance `memory` cell, a diagnostic in 0.8.0, counted (41 short and medium questions, 52 in all).

The 0.8.0 paragraph below is the history of the rest; its statement that the back corridor is not asked is no longer
true.

## What the helper asked in 0.8.0

**0.8.0** (task-critical questions: harder questions, not more of them). During the job the
helper asks, at a look point, `ego_now` or `clear_path` in slot 1 (50% each) and `person_mem` (60%) or `ego_mem` (32%)
in slot 2; at its end `object_where` and `task_things`. It is the job with the most to know: how far it can drive
before it meets furniture or clutter (a table top that overhangs its 1.41 m body, a cord, a bag) and where the resident
is.
- **`clear_path`** ([families/clear_path.md](families/clear_path.md)): ahead, in the phases `navigating` and `carrying`
  (`JobSpec.clear_phases`), credited on C = 0.15 m. A rug or mat is 1.0 cm, the helper's own climb: the 3 mm allowance
  over the climb makes it floor, as the depth's millimetre rounding would otherwise lift it over (it stopped the
  corridor at 21 of 87 short-case points). Its back corridor is not asked (29 answerable rows of 206 points after the
  stopper screen, mostly `unknown`).
- **`person_mem`** ([families/person_mem.md](families/person_mem.md)): where the resident is from the camera now, as
  last seen. The home holds one resident in 14 shifts and none in 2, which is a generator invariant
  (`JobSpec.one_person`): the job's checks fail a helper world with two persons, naming the script. The helper keeps 7%
  of its `unknown` `person_mem` golds.
- `ego_now` and `ego_mem` shrink to the shares above, so `object_where` weighs more inside the apartment's memory
  ability; that is reported, not tuned. The prompt now states the body (a box 34 cm long and 33 cm wide, 1.41 m tall).

**0.7.0** (the history of this section.) No decision kind asks a family:
`zone_clear` is retired, and the three decision points, the spot coin and the `elsewhere` response stay in the path, so
the scripts and renders are unchanged. The window's blocker is told by the robot's own log (the response and the
subgoal), and so is the thing the task carries, so `World.told` keys on the job's `zones` and the third target of each
task. During the job the helper asks at look points (`JobSpec.looks`): `ego_now` and `ego_mem`; its brief names the two
kinds in one sentence. At its end it asks `object_where` and `task_things` (`object_was` and `same_place` are not asked
since 0.6.0: 0 a medium case here), the last with the predicate "on the floor
within 1.2 m of the way you went to fetch and put down the thing": the things on the robot's floor within 1.2 m of the
base's path over the task's `navigating`, `grasping`, `carrying` and `placing` steps up to its last `placing` step, a
don't-care band of 0.4 m about that distance, and light furniture and people don't-care.
`empty_keep_percent` is 26 (the Phase A read: p 0.659 of the tasks empty). Its offsets are credited on the clearance C =
0.15 m (OWMM-Agent's pick radius) and its places on the re-find radius R = 1 m, and a list of places matches within 0.5 m
([questions.md](questions.md), "Credit"). The decisions below are the 0.5.0-control job as built.

## The decisions, as built

Decision points come from the plan alone (the world moves where they fall, never whether they exist). A `before_doorway`
is 0.5 m before the doorway into the room a leg enters, at the **last crossing into that room** (`job_vacuum_path.crossing`,
`last=True`): one per room-changing leg, so a detour round a shut door or through the kitchen cannot add or move a
decision (every doorway the way crosses would have made the count a function of the world).

| Task | Decision sequence |
| --- | --- |
| `tidy` hall to living room | `before_grasp` (navigating, the grasp from the hall's piece), `before_doorway` (carrying, into the living room), `before_place` (carrying), `before_doorway` (navigating, into the hall) |
| `tidy` living room to hall | `before_doorway` (navigating, into the living room), `before_grasp`, `before_doorway` (carrying, into the hall), `before_place` |
| `fetch` from the living room | `before_doorway` (navigating), `before_grasp`, `before_doorway` (carrying, into the hall), `before_place` (at the hall console) |

Every task holds four, on `apartment_a`. Only `before_place` asks a family: `zone_clear`, now-only, about the target window
(above). The earlier during families fell
at the release gate check and are not asked: `where_was_i` failed
shortcuts and `out_of_view` had BROKEN learned shortcuts. The end family is `object_where` alone ("Where did you last see
the {thing}?", critical things that did not move, never the thing the helper carried); `distance_between` left every
job. Every task has one `before_place` well past the 4 steps a now decision needs into its task (`questionnaire.NOW_MIN_STEPS`), so one decision a task
is eligible to be asked (seeds 101-130: 1.00 a task). The declared ladder is the home's, (0.25, 0.5, 1, 2) m.

What the checks decided in 0.4 (the release gate check on the first six scripts, seeds 2001-2006, 18 cases of three
lengths; the release is 12 helper scripts, seeds 2001-2012, 36 cases, short, medium and long, and these six
ship in them):
- **Asked.** `object_where` passes (79 questions, separation +0.75 [+0.70, +0.80]) and `distance_between` passes (18
  questions, separation +0.83). M3's fast gate check on look-dev windows had dropped `distance_between` (separation
  +0.33), and M3's review found that a misread: 8 of 12 questions were unknown, which capped separation at 0.33 (the
  unknown-share fault, since fixed). Its look-dev windows of 1-3 tasks also could not ask `changed` (it needs three
  tasks), so `changed`, `order` and `distance_between` were read again on the release windows.
- **Dropped.** `where_was_i` (23 questions) failed shortcuts: `other_case` scored +0.24 over `all_unknown` (bound 0.10),
  since the helper's routes from its spot repeat. `out_of_view` (38) passed its gates, but its learned shortcuts `blind`
  and `path_log` scored -0.114 under abstaining, BROKEN lines, after the family's one repair round (the doors).
  `changed` fails shortcuts (`blind` +0.33, `current_view` +0.67, n 6). `order` asks 2 questions in 18 cases, under one a
  case. `when_seen` (36 questions, separation +0.81) was dropped after the repair of its time shortcut:
  the helper and the drone both end a task by going home, so a thing's last sighting falls 12-67 s before the window's
  end, and once time answers are learned as lags behind the case's own clock, a learner that reads no memory fails
  shortcuts (`blind_in_pool` +0.15). It was later dropped from the AMR and the vacuum as well, so no 0.4.0 or 0.4.1 job
  asks it ([families/when_seen.md](families/when_seen.md)).

`path_ahead` and `rear_clear` are not asked: the helper never reverses (it turns in place) and its reach poses face fixed
furniture the public map gives (M1's `path_ahead` lesson). A doorway question would need a scripted resident that reacts,
which no job has.

The four events: "when you left your spot" (the first navigating keyframe), "when you started carrying" (the first carrying
one), "when you finished carrying" (the last carrying keyframe: the `before_place` decision's) and "when you got back to
your spot" (the first waiting keyframe after the navigating ones). Relevance: every carried kind of the home catalog
(`catalog.HOME.kinds`: the helper's business is what lies on surfaces), the regions of the task's rooms (the park room's
kind included) and the phases navigating and carrying.

## What the checks hold (`job_helper_checks.py`)

Per task: the phases run waiting, navigating, grasping, carrying, placing, navigating, waiting (`job_phase_order`); the
first and last keyframe stand at the park pose, 1 cm and 1e-3 rad (`job_park`); the decisions are exactly the plan's
sequence (`expected_decisions`), each in its phase, with the `before_grasp` on the last navigating keyframe of the first run
and the `before_place` on the last carrying one (`job_decisions`), the base at each within reach of a counted piece of the
plan's room and facing it within 1e-3 rad (`job_reach`); the task takes at most 600 s (`job_task_time`). The carry:
- `job_targets`: the targets are the plan's two rooms, the thing and the two pieces;
- `job_thing`: the thing is a carried entity that stood on the source piece until the last grasping keyframe, has no
  placement until the last placing keyframe, stands from then on a counted piece of the target room (the planned one
  unless the response is `elsewhere`) and, on the planned piece after an `elsewhere`, outside the window; and its two
  placement changes are the only ones it has in the task;
- `job_tilt`: every keyframe's pitch is -15 or -45 degrees, -45 from `before_grasp` through the last grasping keyframe and
  at `before_place` and the last placing keyframe, -15 while waiting;
- `job_spot`: the `before_place` response is the window's state at that moment (something taller than 0.01 m reaching more
  than 0.01 m into it gives `elsewhere`).
Per script: each gap is 10 minutes to 3 hours (`job_gap`) and the span at most 7 hours (`job_span`). `validate`'s generic
`task_placement` rule keeps the world still inside a task apart from the thing. `build._generate` and `stbench generate` run
`validate` (these included) before a script is written.

## A residual, recorded

The home's rugs and floor mat are 0.01 m high, the helper's `climb_m`, and millimetre depth lifts their points over it, so
the helper's band reads a rug as an obstacle: `out_of_view`'s seen-empty rule never sees a rug's place empty, and a thing
that left a rug stays listed there (a ghost both the gold and the mapper keep: the oracle still scores 1). The bathroom's
mats (0.012-0.015 m) are obstacles to the helper outright, one more reason its vanity is not a reliable surface. The
helper's numbers stay M0's.

## Sizes (seeds 101-130, 4-6 tasks, free)

Measured by `stbench generate` over 30 scripts (no render; the free generation check of the 0.5.0-control lane), all
valid under `validate`, none raising:

| | 4-6 tasks |
| --- | --- |
| scripts raising / with `validate` findings | 0 of 30 / 0 |
| keyframes a script (min, median, max) | 200, 309.5, 559 |
| `tidy` keyframes a task (min, median, max); seconds | 40, 71, 112; 68-143 |
| `fetch` keyframes a task (min, median, max); seconds | 40, 60, 94; 62-131 |
| seconds to generate a script (min, median, max) | 1.2, 1.7, 3.9 |
| decisions a task | 4 (2 `before_doorway`, 1 `before_grasp`, 1 `before_place`) |
| `before_place` decisions at least 4 steps into the task (asked: `NOW_MIN_STEPS`), a task | 1.00 |
| coin blocked / realised blocked (`elsewhere`) | 67 of 147 / 71 of 147 (0.48) |
| put on the room's other piece | 26 of 147 |

Seeds 101-230 (130 scripts, 654 tasks) give the same picture: none raising, none with a `validate` finding, the coin
blocked in 307 tasks (0.47) and the response `elsewhere` in 333 (0.51), 104 tasks ending on the room's other piece.

The tilts add about a tenth to a task's frames over the 0.4 numbers (a tidy had a median of 63 keyframes, a fetch 50;
a fetch now also drives to the console and back). The old skip rate was 0; the first version of the thing's choice skipped a fifth of the seeds on the crowded hall console
and TV console, and the rule above (any item with a place when the window is blocked, the room's other pieces, the nearest
on the piece) skips none of these 30. The test `test_the_skip_rate_stays_under_a_tenth` keeps it under one in ten.

The poses far from the ideal are mostly the sofa and the coffee table (their fronts face each other 0.55 m apart: a body
cannot stand between them) and, at the TV console (61%) and the shoe rack (57%), clutter or a chair nearby; the hall
console's are all at the ideal (the apron is clear). The reach-pose measures of the 0.4 release (seeds 101-200: 897 poses,
7% another piece of the room than the plan's, none raising) are unchanged: the carry reads the same poses.
