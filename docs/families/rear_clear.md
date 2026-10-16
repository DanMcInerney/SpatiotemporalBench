# `rear_clear`

How far the robot could reverse straight back, as it last saw the space behind it. During, scope by evidence (`now` when
the latest frame, the one at the decision, holds every look at the corridor, else `memory`). Code:
`stbench.questions_space` (the gold), `stbench.grid` (`corridor_clear`, the one rule `path_ahead` and a system's mapper
call too). The record and questionnaire rules are in [questions.md](../questions.md). **No job asks it in 0.11.0.**
Since 0.8.0 [`clear_path`](clear_path.md) supersedes it as a look family, and since 0.9.0 `clear_path` asks the back
corridor of the drone, the helper and the vacuum (the AMR asks no `clear_path`). In 0.8.0, when `clear_path` asked the
drone's back corridor only, the ground robots' back corridor was left out (after the stopper screen the helper kept 29
answerable rows of 206 points, the vacuum's own swept path answered it within C/2 at 31% of points, the AMR's constant
scored +0.29). The corridor is read at the 2 cm sub-cells with a 3 mm allowance over the climb, as `clear_path`'s page
says.

| | |
| --- | --- |
| Category | C1 Room to move |
| Asked | by no job. During, at a decision point of a kind whose `JobSpec.decisions` would name it. `amr/1` mapped it at `before_undock` and dropped it at the release gate check (12 questions; it failed `memory_needed`: a learned shortcut under the pool's own winner scored +0.083, one question's margin); `vacuum/2` never asked it (its back-outs failed their gate, below). No other job maps it |
| Text (slots) | "If you reversed straight back now, how far could you go before reaching anything you have seen? Answer {cap} if nothing is within {cap} m." (`cap`: `questions_words.cap_of`, 5 m for the AMR) |
| Answer, credit | `number`, the distance ladder 0.25, 0.5, 1, 2 m |
| Scope, `needs` | `evidence` (`now` or `memory` per question, below); see, pose (the gold reads the depth: [path_ahead.md](path_ahead.md), "What a system needs") |
| Gold function | `questions_space.rear_clear(w, k)`: a `Cand`, never None (the rule's `unknown` is a gold, not a dead band) |

## The gold

`grid.corridor_clear(states, robot, start, heading, cap, 0.0)` with:

- **`states`**: `World.latest_cells(range(rec.k0, k + 1))`, each fine cell's last observed state over the window's
  keyframes up to and including the decision. Nothing before the window is evidence (a system is sent the case's stream
  only), and nothing after the decision has been sent.
- **`heading`**: the delivered yaw at `k` (`sensors/pose.jsonl`, `yaw_deg` to 3 places) plus pi; **`start`**:
  `grid.front_edge(robot, base, heading)`, the point half the body's length from the delivered base along it, the body's
  rear edge. Never the base, never the camera (which stands ahead of the base, facing the other way).
- **no blind band** (`blind` 0.0): a camera faces forward, so nothing of the corridor is out of its sight *now* by a
  band; its cells were seen on the way in or not at all. A cell nobody saw counts against the 80% rule, however near
  the rear edge it lies (`test_the_cells_behind_the_rear_edge_count_unseen`).
- **the corridor** is the body's width wide (0.58 m for the AMR) and `cap` long. The answer is the distance to the
  near edge of the first fine cell (0.1 m) that holds an obstacle sub-cell (0.02 m) whose centre lies in the
  corridor's rectangle, its centre's distance less 0.05 m, or `cap`; `unknown` when fewer than 80% of the cells counted
  (those with centres in the corridor, nearer than that cell, less an occupied one) are `free`. What stands 0.31 m off
  the AMR's axis (half-width 0.29 m) never stops it, even in a fine cell whose centre is inside the width; what stands
  0.27 m off does (`test_the_corridor_is_the_bodys_width`).
- **the band** is the robot's own (`grid.band`: from the AMR's 0.01 m climb plus the 3 mm depth allowance, 0.013 m, to
  0.675 m). A pallet, a tote, a drum or a person behind it stops the corridor. **A floor marking does not**: paint is at
  most 0.005 m ([maps README](../../src/stbench/maps/README.md), "Markings are at most 0.005 m high"), under the climb
  even after the depth's millimetre rounding. At 0.01 m (the climb itself) the rounding lifts points a hair over it and
  the corridor stops at the stripe: why a bay's 0.01 m stripe became 0.005 m
  (`test_a_floor_marking_never_stops_the_corridor`; `test_warehouse_map.py` measures the map's own).

The record: slot `cap` only; evidence the sorted keyframes that last observed a counted cell or the cell that stops the
corridor (never empty on an answerable gold); entities none; subjects the decision's time `w:<k>`; critical; the
ladder; slice `unknown` when the gold is.

**Scope** is `questions_world.scope_of(w, k, evidence)`: `now` exactly when the evidence is non-empty and every
keyframe in it is the decision's own frame `k`, the latest frame, else `memory`. A robot facing forward seldom sees what
is behind it in the frame it answers at, so the AMR's undocks are mostly `memory`; an `unknown` gold that nothing saw
has no evidence and is `memory` too. A family of scope `evidence` counts as `memory` for the declared `now` share, and
the measured share is reported beside it.

## What a robot sees of its corridor

Ray-cast on `tests/rear_kit.py` (the AMR's camera, millimetre depth, an empty 5 m corridor, axis y = 4.98, 300 fine
cells): a camera facing along the corridor from its far end sees 90% (one look from 1.0 m, the corridor ahead of it);
turned round at the bay to look back along it, 88% (the 0.53 m nearest the rear edge is in the blind band of that
camera, and an obstacle near the edge then leaves the share under 80%: `unknown`). Seen from the side, as the AMR does
on the lane of a side arrival ([job-amr.md](../job-amr.md)), one look sees at most 75% of it (the cone and the thinning
floor points past about 5 m), so the answer is a number when something stands within a few metres and the cells before
it were seen, and `unknown` when only one look from the lane covers a corridor that is clear to the cap; the whole drive
along the lane (six looks a metre apart) sees 96%. Fixed geometry behind a station and the lane's looks decide how
often the gold is `unknown`; the AMR's free gate ((e) and (r) in [job-amr.md](../job-amr.md)) measures it.

## Where it is asked, and where it is not

- **The vacuum's back-outs** (M2's A2) were its site in the plan and failed their free gate: the vacuum drives in along
  the corridor it backs out of, so 0.58-0.62 of it is swept before the decision (the bound is 0.4), and the map's raycast
  gives the true rung behind at 47-87% of back-outs (bound 40%). The family left the vacuum for the AMR
  ([jobs.md](../jobs.md), "Back-outs: the mechanism, and the gate that failed").
- **The AMR's undocks** (M3): the robot arrives along a station's face and turns at T ([job-amr.md](../job-amr.md)), so
  the reverse corridor is crossed only at T and seen on the way in, not swept. The AMR's free gate held on the merged map
  (items (b), (e) and (r) of its table), and `amr/1` mapped the family at `before_undock`. It was dropped at the release
  gate check (12 questions; `memory_needed`, above), so no job asks it.

## Beside it

The table systems answer it with `grid.corridor_clear` over their latest cells at the ask's pose; `path_log` from the
public map and its own path with the same sub-cell rule, ignoring floor markings (`table_space`,
`baselines_path_space`). Its sibling `path_ahead` is [path_ahead.md](path_ahead.md).
