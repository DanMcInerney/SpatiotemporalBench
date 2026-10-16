# The world between tasks: what changes, when, how often, and why

A script is several tasks of one robot's job, and between two tasks the apartment goes on: someone moves a mug, pulls
a chair out, shuts a door. `world_processes.gaps` is the callback a job's `script` calls once before each task
(`between(lo, hi, keep_open)`); `world_processes_home.py` is the home's processes. The robot never sees a change happen:
every change of a gap is written at one whole second, at least 30 s from both ends of the gap (`changes.EDGE_BUFFER`).

## The law: a thing's state is drawn again by the law it started with

Every process re-draws a thing's state from the same law `build_world`, `clutter.place` and `generate.light_starts`
drew it from, so the world at any task is distributed as the world at the first, and nothing in a case's window shows how
many tasks came before it. A thing is drawn again over a gap of `g` seconds with the chance `1 - 2^(-g/H)` for its
half-life `H` (`world_processes.half_life_p`): 0 for no gap, 1/2 for a gap of `H`. A short gap moves little, a long one
most things, and no thing moves twice in a gap.

| Process | Law of the new state | H |
| --- | --- | --- |
| Carried things (30-40 mugs, books, laptops, shoes...) | a thing of a floor kind (shoes, bag, toy, box) is left on the floor with `clutter.DROP_SHARE` (a half), within `DROP_REACH_M` (0.8 m) of one of its anchor pieces (`clutter.floor_spot`); else, or for any other kind, it goes to a receptacle other than its own, at least 0.6 m away (`placement.find_spot`), and one that lands on a mat or a rug must split no planned robot's floor | 5 h |
| Chairs (4) | pulled out 0 to 0.6 m against their facing from their **map** place and turned up to 15 degrees (`generate.light_start`) | 6 h |
| The stool and the laundry basket | 0.3 to 1.0 m from their map place, any direction and yaw | 12 h |
| Floor clutter (a fan, a bin, a lamp: 20-45 pieces) | placed again by its kind's places (`clutter.move_floor`: its rooms, beside the piece it stands by) | 12 h |
| Interior doors (D1-D4) | open with 70%, shut otherwise; the entrance (D0) is always shut | 2 h |

Light furniture is drawn from its map place, never from where it stands, so nothing drifts over a script. A new place
for a piece is valid when the start's rule holds (`generate.start_valid`): inside its own room, `FURNITURE_MARGIN` (5 cm)
from blocking fixed pieces, other light pieces and the **open leaf of every interior door** (a door opens and closes
between tasks), at least `dressing_boxes.WALL_PROUD_MAX_M` (12 cm) from the walls, off every door's keep-out and every
arch's gap, off the dock's apron, and, for a piece on legs, its legs off floor things. After the start (t > 0) it must
also split no planned ground robot's free floor (`clutter_reach.FloorGuard.allows_piece`: a robot that passes under the
piece meets its leg squares, any other the footprint; each leg is blocked before the next is asked about). After 60
draws without a valid place a piece stays. Surface clutter stays put in M2, and nothing is added or removed.

The half-lives are modelling choices, calibrated and not asserted (home move rates beyond 3RScan's sparse rescans have no
primary source). `world_processes_home.SOURCES` holds the text for each constant; a 3 h gap draws
34% of the carried things (the moves of an early version of the world, over its 0.5-6 h gaps, were of that size), 29% of
the chairs, 16% of the stool,
basket and floor clutter, and 65% of the door states.

## One instant a gap, in an order

All changes of one gap happen at the same whole second `t_g`, in this order: **doors**, light furniture, carried
things, floor clutter. Each is checked against the world as the changes before it in the call left it, with the thing
that moves taken out of it (`World.without`): it has left at `t_g`, so another thing may take its place and nothing can
land where another thing has not yet left. The doors come first so that the floor guard of every later change sees each
door leaf as it will stand when the call ends, which is how the next task is planned.

## Doors

A closed door the robot's path must go round would show in its own path (paths leak). So every door of
`doors_on_ways(m, keep_open)` (the doors on some way between the rooms the next task travels between: the dock's
region and the rooms it works in) is open at `hi`, and each task is planned with every other door as an obstacle in
both of its leaf positions: no door's state ever changes a path. Every other interior door opens and closes freely, seen
from the hall. A door forced open for one task is drawn again, open with 70%, after it, so a forced door does not stay
open more often than at the start. `route_back` is not asked in M2 (its asking rule would fix its gold on a one-cycle
layout).

## What the free floor guards

`clutter_reach.FloorGuard` (the planners' free space, `route.robot_free_space`, each door leaf in its state at `t_g`)
refuses a thing that would cut a planned robot's floor in two. It is built when the first thing a robot must go round is
asked about, from the world as it stands then. `clutter.floor_spot` (a floor drop; the AMR's corridor drop is the same
call with an `area` and the task's swept bodies to `avoid`) and `clutter.move_floor` are `clutter.place`'s floor rules
at a time `t`, with light furniture kept off at both its places, the map's and where it stands.

## What it comes to

Seeds 101-130 and 141-200 (90 worlds), each driven through the start call and five gaps of one hour (450 gaps):

| Process | Drawn a gap, mean (range) | Moved a gap, mean (range) | Realised share |
| --- | --- | --- | --- |
| Doors (D1-D4) | 2.3 (1-4) | 0.68 changed state (0-4) | 0.29 (a door drawn open that was open does not change) |
| Light furniture (6 pieces) | 0.57 (0-3) | 0.11 (0-1) | 0.20 |
| Carried things | 4.5 (0-12) | 4.3 (0-11) | 0.95 |
| Floor clutter | 1.8 (0-7) | 1.6 (0-6) | 0.91 |

After each call 17% of the four interior doors are shut (0-75% a call; the ones the next task needs are always open,
the others 30% of the time). A gap call takes 0.21 s (median 0.19 s, longest 0.85 s), most of it the floor guard's free
space (25 ms a robot, taken when the first thing a robot must go round is asked about). After five gaps 59% of the
members of identical sets are still on their set's surface (387 of 654), and 22% of the sets are whole (48 of 222).

## Residuals, stated and not fixed

- A moved member of an identical set (mugs, glasses, bowls, plates, towels) leaves its set, as an early version's moves did.
- The start's second try for a floor drop (its room's open floor) is not repeated by a gap's drop.
- A thing that finds no spot stays, so the realised share is under the drawn one.
- Light furniture moves rarely: one drawn piece in five finds a place. The floor guard refuses most places in the tight
  dining and desk areas (a chair is four legs that together plug a passage of 0.4 m, and the helper, which cannot pass
  under it, goes round the whole footprint), and the leaf and wall rules refuse more. Of 30 tries each, the chairs
  beside the kitchen's arch and the dining table, at the desk and at the living room's table found 10, 3, 4 and 4
  places (27, 3, 4 and 7 with the vacuum alone guarded; 30, 30, 19 and 30 with no guard).
- The stool's map place is 0.025 m from the kitchen's wall (y = 2.75), inside the 12 cm the walls are kept off, so no
  start or gap ever moves the stool: it stays at its map place.
