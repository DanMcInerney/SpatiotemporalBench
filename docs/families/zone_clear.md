# `zone_clear`

**Retired in 0.6.0.** No job asks it, none of 0.11.0's four included: `JobSpec.decisions` maps no decision kind to a
family, and the questions are asked from the camera at look points and at the end. The module, the
zone rule and the coins stay: each coin still sets its zone and the robot still answers it, so the blocker is told
(`World.told`) and no question names it. What follows is the family as 0.5.0-control built it.

Is the place the next manoeuvre uses clear of anything taller than a height, right now? During, `now`. Code:
`stbench.questions_zone` (the gold), `stbench.zones` and `stbench.zone_shapes` (the one zone rule the table systems call
too). The record and questionnaire rules are in [questions.md](../questions.md); the zone rule is shared with
[clearance_then.md](clearance_then.md) and [slots_empty.md](slots_empty.md).

| | |
| --- | --- |
| Category | C1 Room to move |
| Asked | during, at a decision point of a kind that asks it (`JobSpec.decisions`); the kind must have a rule in `JobSpec.zones` (a job that asks without one is a bug: `ValueError` naming the job and the kind); a pose with no zone for the rule (a map with no piece of its kind, a base facing no surface) is a dead band, not asked. A kind that asks only `now` families is a candidate from 4 steps (`questionnaire.NOW_MIN_STEPS`) into its task, not the 16 (`GAP_STEPS`) a memory family waits for: the answer is the look window the question follows, so the first room's doorway, 5-15 steps into a vacuum's task, can be asked. 0.5.0-control maps it at one decision kind per job, each with a coin and a response (the job's page): `drone/1` at `before_lower` (the drop zone), `amr/1` at `before_station` (the bay ahead), `helper/1` at `before_place` (the surface window in front of the robot) and `vacuum/2` at `before_room` (the doorway's way) |
| Text (slots) | "Is {zone} clear of anything taller than {height} m right now?" (`zone`: the rule's `words`; `height`: the rule's `height_m`, 0.3 m by default) |
| Answer, credit | `yesno` (yes: clear), exact; `unknown` is a gold (slice `unknown`) |
| Scope, `needs` | `now`; see, pose; `mapper_gold` (one rule, so `knowable` does not apply) |
| Gold function | `questions_zone.zone_clear(w, k)`: a `Cand`, critical, subjects the time (`w:<k>`) and the zone's piece (`e:<id>` of its timeline entity; a doorway's is its door's; an arch has none, so its question carries only `w:<k>`) |
| Slice | none (a during family draws no slice) |

## The zone

A job declares which place a decision kind names (`JobSpec.zones[kind]`, a `zones.ZoneRule`: `words`, `piece`, a
`margin_m`, a `height_m` and a `top_m`). The zone comes from the public map and the robot's **delivered** base and
heading, never from the plan or the world, so a table system given the rule finds the same one. `piece` is one of:

- **a furniture kind** (`bay`, `drop_zone`): the piece of that kind whose centre is nearest the base (the first in map
  order on a tie). Its polygon is the piece's footprint grown by the margin and **inset by `zones.EDGE_M` (0.05 m)**: a
  station touches its bay's face and spans it at 0.47 to 2 m, so without the inset the station's own face would read as
  an occupant of an empty bay. It lies on a **ground** (`zones.ground_z`: the top of the tallest other fixed piece of at
  most 0.2 m that holds the piece's centre, else 0, so a marking on a plaza is judged from the plaza's top);
- **`zones.DOORWAY`**: the way through the interior opening nearest the base (`zone_shapes.doorway_way`), on the floor:
  across the opening 0.1 m short of each jamb (the open leaf of a door stands within 0.04 m of its hinge jamb), along it
  from 0.05 m inside the swing-side face of a door to 0.3 m past the other face, or 0.3 m past both faces of an arch;
- **`zones.SPOT`**: the window of the surface the base faces (`zone_shapes.surface_window`): the first blocking fixed
  piece the heading ray meets within 1.5 m, its largest surface, and a rectangle 0.4 m along that surface's long side,
  centred on the hit point, across its whole inset depth, at the surface's height. It needs the base's heading (the
  delivered `yaw_deg`): `zones.zone_of` raises without one, and finds none (a dead band) when the ray meets no piece, or
  a piece with no surface.

The band is `(height_m, top_m)` above the zone's ground (`top_m` is 3.5 m by default, a thing put down on a surface
needs far less).

## The gold

A zone's cells are the fine cells (0.1 m) whose centres lie in the polygon. The states are `grid.observe` of each
keyframe's depth **with the zone's own ground as the floor and its band**, never the robot's band (`World.zone_states`;
a flier's own band is not read). The gold is `zones.zone_state` over the latest state of each cell **of the decision's
look window alone** (`World.window_ks(k)`), the look window that ends at the decision:

- **blocked (no)** when an occupied cell holds an obstacle sub-cell (0.02 m) whose centre lies in the polygon (a thing
  beside the zone, whose points fall in an edge cell, does not block it: the corridor's rule). One such blocker in the
  look window certifies the answer whatever else was seen; the evidence is the look window's keyframes of the blocking cells;
- else **clear (yes)** when at least 80% of the cells are free; the evidence is the keyframes of every observed zone
  cell;
- else **unknown**: the zone was not seen wholly enough to call it clear and nothing in it was seen to block it.

The question is about right now, so there is no earlier frame and no "as you last saw it" form: what an earlier look window
saw changes no gold, and the scope is `now` on every row (the current view gives the gold, methodology rule 4). A zone
nobody observed in the look window is `unknown` with no evidence.

**History.** The 0.4 and 0.5.0 form chose "right now" or "as you last saw it" by coverage and read the window for the
second. On the AMR its `memory` rows were bays seen whole and clear before the approach and under 80% in the last
look window (then a batch the stream sent), every gold `yes`, so a learned shortcut answered them: it failed the AMR's release gate check (`blind` +0.67)
and left the job. The drone's version, on golds balanced by a fair coin but with no response, failed `shortcuts` by
a tilt a fair coin gives (`blind_in_pool` +0.104 against 0.10 at 96 questions) and left too. Asking only about the look window
removes the AMR's rows: the answer is in the pictures the robot has just been sent. The drone's coin now has a response,
and the shortcut gates read a tilt with its interval (the evidence verdict, [eval.md](../eval.md)).

## Balance

A job balances the answer by the world, never by the choice of points: each decision's zone is made
blocked or clear by a coin that reads no world (the job's lane, applied in the gap before the task), and the scripted
robot responds after the question is asked, following the zone's true state (the drone aborts the delivery, the AMR
skips the dock, the helper places on a free part of the surface, the vacuum skips the room). The coin and the response
are in the job's page; nothing before the decision tells which way the coin fell.

## What a system needs

`needs` is what a system needs to answer: the picture and the pose. The gold is the zone's state over the recorded
depth. A system on the camera alone (`rgb_pose`, the default channel) judges from the picture whether anything taller
than the height stands in the place the question names; a system given the depth (the gate channel `detections`) may
measure it. The table systems read the depth through the mapper and call the gold's functions, so `oracle` and
`memory_table` give the gold, and `current_view` equalled it on every row while a message carried the look window.

**Under one-frame delivery** (0.10.0) the scope stays `now` and the gold stays over the look window's cells, but a system
gets one frame at a time, so the table answerers (`oracle`, `memory_table`, `current_view`) see one frame of that window
and no longer reproduce the gold. No job asks the family, so no score is affected; a job that asks it again first
reworks it for one-frame delivery.
