# `uncovered`

Which cells of the room the robot is leaving it saw as open floor but did not clean. During, `memory`. Code:
`stbench.questions_cover` (the gold), `stbench.grid_cover` (the one rule the table systems call too), `stbench.grid` (the
free-space states and `swath` it reads). The record and questionnaire rules are in [questions.md](../questions.md).
**No job asks it** (below, and "Declared before data"): `vacuum/2` dropped it at M2's fast gate check.

| | |
| --- | --- |
| Category | C5 What matters here |
| Asked | during, at a decision point of a kind whose `JobSpec.decisions` names it (no job names it now: `vacuum/2` dropped it at M2's fast gate check, its golds being empty after the declared fallback, 100% where a task stops); a base that stands in no room is a dead band and not asked, and so is a room with no cell wholly inside and clear of every blocked rectangle (the apartment's hall), whose gold would be the empty set whatever the robot did (methodology rule 5) |
| Text (slots) | "Which cells of the room you are leaving did you see as open floor but not clean?" (none) |
| Answer, credit | `cells` (each `[i, j]`, `i = floor(x / 0.5)`), F1 over the cells with the gold's don't-care cells dropped from both sides |
| Scope, `needs` | `memory`; see, pose, status (the brush); `mapper_gold` (one rule, so `knowable` does not apply). The gold reads the recorded depth, which the camera-only channel does not send: a system there judges the open floor from the picture |
| Gold function | `questions_cover.uncovered(w, k)`: a `Cand`, never `unknown`; None when the base at `k` is in no region |

## The gold

The room is the public map's region that holds the delivered base at the decision (`MapSpec.region_of`). Three things
are read, all over the window up to the decision, none from the world's own state:

- **The swath** (`grid_cover.brush_swath`): the fine cells (0.1 m) the robot's tool named `brush` (`grid_cover.brush_of`) covered. Over each run
  of consecutive delivered steps with `status.brush` true, the tool's centre (the base plus the tool's offset turned by
  the yaw) is covered with the tool's shape and size by `grid.swath` (the straight segments between consecutive
  keyframes are the path: `keyframes.frames` puts a frame at every vertex); a single brush-on step covers its own
  footprint, and a brush-off step ends a run, so transit and a charging gap cover nothing. **Every run of the window
  counts**: a room cleaned in part by an earlier task and finished in a later one is covered by both (no window starts at
  a `resume` task, so a window that holds a resumed room holds the stopped task's sweeps of it). A robot with no `brush`
  tool raises.
- **The states** (`World.latest_cells`): each fine cell's last observed state over the window's keyframes
  (`free`, `occupied`, or never observed), from the depth the system was sent. Nothing unobserved changes a gold.
- **The blocked rectangles** (`grid_cover.blocked`): the map's wall pieces and each fixed piece's obstacles for this
  robot (its four leg squares when the robot passes under it, else its whole footprint; rugs never). This is
  layout-constant and read from the public map. A **light piece's footprint is never in it**, since world processes move
  light furniture where no frame sees: a hidden move changes no gold (truth rule 2).

`grid_cover.cover` then takes each 0.5 m cell whose closed square meets the room polygon:

1. a corner of its square outside the room (the boundary counts as inside): **don't-care**;
2. its closed square touches a blocked rectangle (to 1 µm): **don't-care**;
3. of its 25 fine cells some but not all are swept: **don't-care**; all swept: negative;
4. none swept: negative when any fine cell is `occupied` (something stands in it) or none was observed; **positive** when
   all 25 are `free`; else (some unobserved, the rest `free`) **don't-care**.

Every cell not listed is a negative, so an answer naming a cell outside the room is a false positive; a cell on a room
edge that lies on the 0.5 m grid is touched from outside by one more ring of cells, which are don't-care by rule 1 (the
apartment's rooms are 0.05 m inside the grid, so it has none). The evidence is the sorted distinct keyframes of the
latest states of the positive cells' fine cells; with no positive cell, of every observed fine cell of the cells wholly
inside the room; the decision's own keyframe is added to both.

The swath alone cannot answer it: a cell a log of the robot's own path calls open (`grid_cover.open_cells`: wholly
inside, unblocked, no fine cell swept) is a positive only if the depth showed all 25 fine cells free, and a chair leg,
clutter, a box with an area behind it no frame saw each make a negative or don't-care of it (`tests/test_questions_cover.py`
holds a worked room: the swath-only answer scores F1 0.571 against its gold, which scores 1).

## Declared before data

M2 has no pre-gate render. The free checks decide first, then the dev gate check:

- **The free bound** (needs no render, no depth): over the generated `vacuum/2` scripts (seeds 101-130 and 141-200, never
  21-30) count the `room_done` decisions whose room holds at least one cell that could be positive (`open_cells` over the
  smallest window a case can hold: the decision's task and the tasks it continues). That bounds the non-empty share from
  above. **Under 45%** the empty share would pass 55% whatever the depth shows: `uncovered` is mapped to
  `cleaning_stopped` only from the start. The same count at `cleaning_stopped` is reported.
- **At the dev gate check:** a (layout, job) group whose `uncovered` golds are more than 55% empty
  (`check_report.json`'s `empty_set_share`) takes the named fallback, `cleaning_stopped` only, if it is not already there,
  else the family is dropped; if `path_log` passes it, it becomes own-path only or is dropped.

Measured: the free bound passed. On the `vacuum/2` scripts (seeds 101-200, [jobs.md](../jobs.md)) 82.7% of the 527
`room_done` rooms hold a cell that could be positive (median 2, at most 12) and 97.9% of the 94 `cleaning_stopped` rooms
(median 4, at most 16). On 30 one-task `vacuum/1` scripts of `apartment_a` (seeds 101-130, a proxy before the `vacuum/2`
scripts existed: one room a task, the same strips) the count had been 24 of 30 `room_done` rooms (80%), mostly one or two
cells (the kitchen 6 of 10, the bedroom 8 of 8, the bathroom 7 of 9, the living room 3 of 3). The dev gate check then
dropped the family: its golds were empty after the declared fallback, 100% where a task stops
([questions.md](../questions.md)), so no job asks it.
