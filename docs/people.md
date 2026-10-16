# People: who is there, how they move, what the truth holds

A home with nobody in it is a showroom: a robot vacuum on a weekday meets a resident standing at a counter, walking to
another room, in the way; a warehouse has pickers on its floor and a downtown pedestrians on its pavements. This page is
how a script gets them (`people.py`, `people_home.py`, `people_warehouse.py`, `people_downtown.py`, `mover_walk.py`).
People are scenery with a place in the truth: the questions that involve a person ask how near one is or was ("What
counts as seeing a person"), and none gives one a name, a colour or a look. Every number below is a modelling choice,
recorded with its note in the catalog's `SOURCES`.

## Who

A person (a resident, a picker, a pedestrian) is a `mover` entity of kind `person` with no attributes (a person is not
described: no colour, no name), the box of its body (`BOX_M`: length along its facing, width, height), drawn as an
ordinary mover asset at scale 1. There is no pose, no gait and no pet. Ids come after every other entity's, and
`person_kind("person")` gives `validate` the camera clearance (and a default box) as a catalog's mover kinds do.

The overlay is one generic module for every catalog. A catalog's people are the file `people_<catalog>.py`, which
declares `KIND`, `MODEL`, `BOX_M`, `COUNT`, `P_HOME`, `SPEED_MPS`, `DWELL_S`, `SAME_ROOM`, `CAMERA_M`, `ROBOT_KEEP_M`,
`START_ROBOT_M` and `SOURCES`, and may declare `WALK_ON`, `KEEP_OFF` and `GRID_M` (below); a catalog with no such file
has no people. One person kind has several bodies: every module shares `KIND` and `CAMERA_M` (a test holds it), and
each names its own `MODEL` by id and its own `BOX_M`, which equals that model's measured box (a test holds it).

| | `people_home` | `people_warehouse` | `people_downtown` |
| --- | --- | --- | --- |
| People, body | 1 resident, the barefoot `resident` | 3 pickers, the shod `worker` | 8 pedestrians, the `worker` |
| `COUNT`, `P_HOME` | 1, 0.5 | 3, 0.8 | 8, 0.7 |
| `SPEED_MPS`, `DWELL_S` | 0.8-1.2, 20-120 | 1.0-1.4, 10-60 | 1.1-1.5, 5-45 |
| `SAME_ROOM` | 0.7 | 0.6 | 0.5 |
| `CAMERA_M`, `ROBOT_KEEP_M`, `START_ROBOT_M` | 0.8, 0.2, 1.5 | 0.8, 0.3, 2.0 | 0.8, 0.5, 5.0 |
| `WALK_ON`, `KEEP_OFF`, `GRID_M` | the floor, none, 0.05 | the floor, `bay`, 0.05 | `sidewalk` and `plaza`, `drop_zone`, 0.1 |

The resident is barefoot so that nothing on its feet can be read as the tracked `shoes`; the worker (work clothes and
closed boots) is for the places where a barefoot person was not realistic.

## When they are there

`generate_script` ends with `people.add(timeline, map, rng)`: people are an overlay on the finished script, because a
walk must keep clear of a robot path that exists only after the job's script. Each task, independently, has each person
there with probability `P_HOME`, so a guess learns neither presence nor absence. In the script's first task a person is
there from t = 0. Otherwise it arrives, jumps from one task's last pose to the next one's first, or leaves, at **one
whole second inside a gap**, at least 30 s (`changes.EDGE_BUFFER`) from both its ends: the robot never sees it appear,
vanish or move between two keyframes. Within a task a person's pose changes only at keyframe times. A gap too short for
such a moment raises `ValueError` where a person must arrive or leave in it. With `COUNT` above 1 each person is drawn
after the others and treats their placements as obstacles, so no two overlap.

## How they move

A task's walk is `mover_walk.Walker` on the person's free floor, advanced a keyframe at a time:
- **Legs**: stand for a time drawn from `DWELL_S`, pick a waypoint (in the region it stands in with probability
  `SAME_ROOM`, else in another; never in a doorway), walk the A* path at one speed drawn from `SPEED_MPS`, stand again.
  A waypoint never depends on where the robot is. A walk starts and ends in the largest connected stretch of free floor.
- **Free floor**: `GRID_M` cells, a cell free when its centre is half the box's width plus 0.05 m (0.33 m for a 0.56 m
  body, arms hanging) from every wall piece and every obstacle, so every open interior door of the home (0.80 to 0.90 m
  wide) lets a person through; at each step the footprint at its heading is also checked against the obstacles.
- **Obstacles**: everything that stands at any time from the previous task's last keyframe (or t = 0) to the next task's
  first keyframe (or the script's end): blocking fixed furniture above the person's feet, every placement of light
  furniture and of things on its floor in that span, each door leaf in every state it has then. A person's first and
  last poses of a task stand through the gaps around it, so they clear everything the world changes there too.
- **The robot**: a step is not taken (the person waits) when its footprint would come within `max(ROBOT_KEEP_M, the
  robot's own clearance + 0.01 m)` of the robot's body at that keyframe or of the body swept to the keyframe before and
  after it (a rectangle's circumscribed disc), or its centre within `CAMERA_M` of the camera; a person who is waiting in
  a place the robot then comes up to is not nudged aside: the whole walk is drawn again, from a new start (up to
  `people.ATTEMPTS`, 1,000 draws a task; a person that finds none is not there for that task). It starts a task at least
  `START_ROBOT_M` from the robot's first pose. **The robot never reacts to a person**: its path is planned before the
  overlay. That is a stated simplification; in a real home people step round a working vacuum. The drone's own
  clearance is 3 m, so near its pad a pedestrian keeps 3.01 m from its disc.
- **A flier passes over people**: at a keyframe where the robot flies over the person (`robots.flies_over`: the robot's
  base height less its clearance, against the person's own floor plus box height) nothing of the robot keeps the person
  off, neither the body, its sweep nor the camera: a pedestrian may walk under a drone hovering 7 m up. `validate` reads
  the same rule.
- One stream a person and one for its presence and each task (`rng.spawn`), so a task's walk does not depend on how many
  tasks the script has.

## Where they walk

- **Walk surfaces** (`WALK_ON`, fixed-piece kinds). A person stands at the one height of those pieces' tops (`ValueError`
  naming the map when they differ by over 0.01 m, or the map has none) and keeps its whole footprint on their union: a
  body never overhangs a kerb into the roadway or the cars parked along it. Neighbouring pieces up to 0.01 m apart are
  one surface (the maps inset them by 0.005 m). The surface is exact: the free grid's obstacles include the rectangles
  that cover the map off the pieces, which needs the pieces axis-aligned (`ValueError` otherwise). A fixed piece whose top
  is at most the walk height (to 1e-6 above the 0.01 m) is ground, never an obstacle; a carried item or clutter obstructs
  a walker when its base is within `FLOOR_ITEM_Z` (0.1 m) of the walk height (a food cart on the plaza does, a car in the
  roadway below does not).
- **Keep-offs** (`KEEP_OFF`): fixed-piece kinds a walk never enters, grown by the walker's clearance: the AMR's `bay`s
  and the downtown's `drop_zone`s. `amr/1` and `drone/1` balance each zone's answers by a coin that a drop enforces
  before the task, and a person standing in a zone would turn a clear coin blocked, making `no` the likelier answer
  (methodology rule 5: a constant answer must earn nothing).
- **The downtown**: sidewalks and the plaza at 0.15 m. Four stretches of its walk area are not connected (pedestrians
  never cross a roadway), and a walk lives in the largest: the high street's south sidewalk and the square, 895 m² of the
  1,759 m² of free walk area; the other three (the high street's north sidewalk with the sidewalks of the side streets
  that meet it, in three stretches) are never walked. Nobody enters the nest's yard (it is no walk piece). No dogs and no moving traffic: no family needs
  them, and no free photoreal dog is in hand (the downtown's taxis and buses stay retired).
- **The warehouse**: the floor of the dock and the three aisles, 208 m² connected, less the racks, stations, tables and
  the bays.

## What the truth holds

A person is an entity and its placements in the timeline, in the visibility log and the depth when in view, and in the
`detections` channel as `person` with no attributes and no `state` key (a detection's `extent_m` is the model's box). It
is never named by a question: the question side treats a person as a don't-care place, and the free-space golds and the
mapper read a standing person as an obstacle where it was seen. Nothing names it to an agent but what the agent sees of
it: pixels, depth and that detection.

`validate` checks it with the mover rules: `mover_collision` (against walls, fixed and light furniture, things on its
floor and door leaves in the state they have then, and other movers), `mover_camera_clearance` and `unknown_mover_kind`,
and the body rule keeps the robot's clearance from its footprint. A mover is read on its own floor and by its own box: a
fixed piece whose top is at most the mover's base height is the ground it stands on, so a person at 0.15 m on a 0.15 m
sidewalk is valid and one at 0 m over it is not (the home's movers, at 0, keep the rule as it was).
`people.without_people(timeline)` is the script before the overlay, which the realism probe compares worlds by
(`realism_world.rebuild_world`).

## What counts as seeing a person (0.8.0)

A person is named by no question, but families ask where one is and how near. `amr/1` and `helper/1` ask
[closest_approach](families/closest_approach.md) and [approach_mem](families/approach_mem.md), `amr/1` also
[closest_person](families/closest_person.md); [drop_zone_people](families/drop_zone_people.md) and
[person_mem](families/person_mem.md) are written and asked by no job. A sighting of a person is **evidence**
(`person_seen.evidence`, one neutral function that the gold and the oracle channel's person `floor_hull` both call) when
it has:
- at least `RECOG_PX` (1,024) pixels and `RECOG_LUM`;
- a depth at every pixel;
- the contact tie at the job's clearance C: a pixel of the feet within max(2 cm, a pixel's height) of the floor **and
  within C/2 of the hull's nearest point to the camera, along the floor**, read to within C/2 at one pixel. A standing
  person's belly or back projects about 10 cm beyond their heels, so the second clause fails a clear sighting when C/2
  is 7.5 cm (46% of the helper's clear sightings): a sighting that fails is a row the robot saw, not evidence;
- a bounding box that does not touch the image's **bottom** edge. The top and the sides may cut the person.

`recognise`'s rule (a box 2 pixels from every edge) exists because a thing cut by the frame may be a sliver of itself and
misnamed; a person cut at the chest by the top of the frame is plainly a person, and their feet place them. Under the
strict rule 41 of the AMR's 46 closest-approach golds, which are the close passes, would be dropped. Nothing is named but
"a person", so no colour or look-alike screen applies. A person who is not evidence is still a row of the visibility
log, and a task that holds one nearer than the gold by more than C/2 is not asked.

## Simplifications

One standing body, not a gait; waypoints uniform over the reachable free floor (no preference for counters, shelves or
doorways, which dwell would suggest); no pet and no traffic; the robot ignores people; pedestrians never cross a roadway.
Each is the simpler choice under methodology §3 and can be made richer without changing a file format.

## Cost

A person's walk builds a free grid and a walker (a setup of seconds for the downtown's million cells): measured on hand
scripts of 60 keyframes (a task), three pickers take 1.7 s and eight pedestrians 12.5 s (1.5 s each, most of it the
walker's setup), so a downtown script costs about that for each task its pedestrians are there for.
