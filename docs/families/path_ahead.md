# `path_ahead`

How far the robot could move straight ahead, from the depth it was just sent. During, `now`. Code:
`stbench.questions_space` (the gold), `stbench.grid` (the one rule a system's mapper calls too), `stbench.freespace` (the
rule applied to a script's files). The record and questionnaire rules are in [questions.md](../questions.md). **No job
asks it in 0.11.0.** Since 0.8.0 [`clear_path`](clear_path.md) supersedes it: the same corridor, asked at look points,
read at the 2 cm sub-cells with a 3 mm allowance over the climb (`grid.DEPTH_QUANTUM_M`: a surface at most that high is
floor, which changed no shipped gold), with the screens and phases of its page. This page records the rule and why no
job asked it at a decision point.

| | |
| --- | --- |
| Category | C1 Room to move |
| Asked | by no job. During, at a decision point of a kind whose `JobSpec.decisions` would name it. It left `vacuum/1`'s doorways (M1) and `vacuum/2`'s back-outs (M2), and `amr/1` does not map it at `before_undock` (the station's face is 0.12 m ahead of every undock: "Where it is asked", below) |
| Text (slots) | "How far could you move straight ahead now before reaching anything? Answer {cap} if nothing is within {cap} m." (`cap`) |
| Answer, credit | `number`, the distance ladder 0.25, 0.5, 1, 2 m |
| Scope, `needs` | `now`; see, pose (the gold reads the depth, which the camera-only channel does not send: below) |
| Gold function | `questions_space.path_ahead(w, k)`: a `Cand`, never None (the rule's `unknown` is a gold, not a dead band) |

## The gold

`grid.path_ahead` over the latest free-space cells of the look window that ends at the decision point, at the decision's
delivered pose, with `cap` = min(5 m, depth range - 1 m) (`questions_words.cap_of`: 3 m for the vacuum); the system's
mapper calls the same function over the same depth, so the oracle reproduces it exactly. An answer about "now" from older
frames would be a guess about space that may have changed, so no other look window is read. The free-space rule
(`grid.py`, `freespace.py`):

- `grid.points` unprojects every pixel with depth above 0 at its centre; a pixel of depth 0 (no surface, nearer than
  `near_m`, past `far_m`, the robot's own body) adds no point and certifies nothing;
- `grid.observe` makes a 0.1 m cell `occupied` when any point lies above what the robot drives over and at most the
  body's height plus half its under-margin (the vacuum's band is 0.023 m to 0.095 m), `free` when at least 4 floor points
  lie in it, and leaves it absent otherwise; points above the band are overhead and ignored, so the floor under a table
  is seen between its legs, and the vacuum's 0.10 m shelf is overhead as it is for the planner;
- **the corridor** is the body's width wide and `cap` long (for a box; a disc sweeps a disc, `grid.reach`, final review B2: [clear_path.md](clear_path.md)). `grid.corridor_clear(states, robot, start, heading, cap,
  blind)` is the rule from any start along any heading, and `path_ahead` is that from the body's front edge
  (`grid.front_edge`: the base plus half the body's length along the heading, never the camera) along the heading, past
  `robots.blind_m`. Only what stands within the body's width can stop it: `observe` also keeps, for an occupied cell, the
  0.02 m sub-cells that hold its points, and a cell stops the corridor when one of them has its centre inside the
  corridor's rectangle (a point within 1 cm of the body's side may fall either way). A door frame 0.19 m off the axis of
  the vacuum, whose half-width is 0.17 m, does not stop it, however near the 0.1 m cell holding it has its centre. The
  answer is the distance to the near edge of the first such cell (its centre's distance along the corridor, less 0.05
  m), or `cap`; `unknown` when fewer than 80% of the cells counted are free. The cells counted are the 0.1 m cells whose
  centres lie in the corridor, beyond the camera's blind distance and before that cell, leaving out an occupied one whose
  obstacle stands outside the body's width (it says nothing of the corridor); the evidence is the look window's keyframes that
  observed them and the cell that stops it;
- **Known limit.** An overhang whose underside lies in the band, seen edge-on (a sofa's base from the vacuum's 0.08 m
  camera), marks its front cells occupied and the floor under it free; a corridor stops at the front, which is right.
- **Yield.** From a camera 0.08 m up, the floor thins out past about 2.3 m: a 0.1 m cell there often holds fewer than
  four floor points. Over 72 headings and positions on a flat floor (ray-cast depth, `tests/space_kit.render_depth`) the
  counted cells of an empty, open corridor to 3 m are 79% to 90% free (mean 84%), so it answers 3 at 69 of the 72 and
  `unknown` at three: the 80% rule sits near its edge at the cap (`tests/test_freespace.py` pins one alignment each
  side). `path_ahead` is answerable wherever something stands within about 2.3 m.

## Where it is asked, and where it is not

The family is sound; the question is whether a site leaves it something a shortcut cannot answer.
- **Doorways (M1)** dropped it, on look-dev evidence: `other_case` (the same moment in another script) beat `all_unknown`
  by 0.50 against a 0.10 margin, and `path_log`'s map raycast scored 0.988 at the leaving doorway, since the doorway
  and the room's layout are on the public map. Its gold was also wrong at two of five
  doorways, the 0.1 m cells catching a door frame outside the body's width: the 2 cm sub-cells above are that repair.
- **Back-outs (M2)** were to carry it with `rear_clear` (the chair legs ahead of a vacuum driving in under a table
  vary by seed). Their free gate failed on the rear corridor, which the vacuum had swept on the way in, and both left
  the vacuum ([jobs.md](../jobs.md), "Back-outs: the mechanism, and the gate that failed").
- **The AMR (M3)** docks with its front edge `DOCK_GAP_M` 0.12 m from a station's face and turns to face it from T,
  1.8-2.4 m out, so the corridor ahead ends at a fixed piece the public map names, at nearly the same distance at
  every visit: the best constant per station and `path_log`'s raycast are expected to answer it. The AMR's free gate
  measured it as gate (f), which did not hold on the merged map ([job-amr.md](../job-amr.md), "The gate (the first
  version's)"), so `amr/1` does not ask it; `path_ahead` waits for a site whose clearance ahead is the world's (a robot
  that crosses open floor).

## What a system needs

`needs` is what a system needs to answer, and the gold may read more: the picture and the pose. The gold is the corridor over the recorded depth; a system on the camera alone (`rgb_pose`, the default channel) judges the distance from the picture, within the job's tolerance ladder, and one given the depth (`detections`, the gate channel) may measure it.

## Beside it

`grid.place_free(states, xy, radius)` is the place test two later families share: of the fine cells whose centres lie
within `radius` of `xy`, at least 80% are `free` and none `occupied` (a place seen empty; a cell nobody observed counts
against it). The corridor behind the robot is [rear_clear](rear_clear.md): `corridor_clear` from the rear edge,
`grid.front_edge(robot, base, yaw + pi)`, over the latest cells of the window instead of the look window.

**Under one-frame delivery** (0.10.0) the scope stays `now` and the gold stays over the look window's cells, so a
system, which gets one frame at a time, holds one frame of what the gold read. No job asks the family (`clear_path`
supersedes it), so no score is affected; a job that asks it again first reworks it for one-frame delivery.
