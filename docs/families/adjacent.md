# `adjacent`

**Retired in 0.6.0 (an internal checkpoint, folded into 0.7.0 in the [CHANGELOG](../../CHANGELOG.md)). No job asks it in
0.11.0.** No job's `ends` names it: it asks only on a held-out layout, the shipped data hold none, and on a public layout
it asks nothing in any case. The module stays. What follows is the family as 0.5.0-control built it.

A point in the room that connects two rooms, asked only on a held-out layout. End, `memory`. Code:
`stbench.questions_adjacent` (the gold), `stbench.adjacency` (the rule for which rooms are joined, which rooms were
toured and which point the gold names, called by the gold and by the table systems alike), `stbench.table_adjacent` and
`stbench.baselines_path_adjacent` (the answers of the table systems and of `path_log`). The record and questionnaire
rules are in [questions.md](../questions.md).

| | |
| --- | --- |
| Category | C4 Which way |
| Asked | by no job. End, once per moment, **on a layout of `map_files.HELD_OUT` only**: on a public layout the family has no moment, so it takes no slot and changes no public questionnaire. Through 0.5.0 the `ends` of `vacuum/2` and `amr/1` named it (`helper/1` did not, and a flier is never asked, so `drone/1` did not); no job's `ends` names it in 0.11.0, and the shipped data have no held-out layout to ask it on |
| Text (slots) | "Give a point in the {space} that connects the {a} to the {b}." (`space`: "room" for a `home` job, else "area"; `a` and `b`: the two room kinds, in sorted order) |
| Answer, credit | `position` (xy), credit `region`: 1 for a point inside the connecting room's polygon at least 0.2 m from its walls (`tolerances.erode_m`), else 0; never `unknown` |
| Scope, `needs` | `memory`; see, pose (the gold reads the delivered poses and the observed cells: a run without the camera could answer none of these); `mapper_gold` (the gold and the table systems call `adjacency`, one rule, so `knowable` does not apply) |
| Gold function | `questions_adjacent.moments(w)`: one `Moment` for each askable pair, keyed "<kind a>\|<kind b>", one answerable candidate |
| Reference | `other_case`, which `gates._group` leaves out of the family's gates on a held-out layout (any case of an unseen layout carries the layout, which is the ability measured) |
| Slice | none: there is no unknown slice |

## Why only on held-out layouts

A layout's topology is the same in every case of the layout. On a public layout `blind` and `other_case` learn "the
bedroom and the kitchen are joined through the hall" from the other cases, whatever the stream says. On an unseen
layout the topology can only come from the robot's own path and what it saw, which is the ability the family measures.

## Which rooms are joined (`adjacency`)

Two regions are **joined** by a door or an arch cut through the wall between them (the entrance, whose other side is
"outside", joins nothing), or by an **open edge**: collinear polygon edges (within 2 mm) of two regions of one level
that overlap along a run of at least `JOIN_MIN_M` (0.8 m, the narrowest door `check_map` allows) that no wall piece
stands on (`MapSpec.wall_pieces`; a wall face against the edge, within 2 mm, stands on none of it). Two rooms with only a
wall between them have polygons a wall thickness apart and share no edge. `connector(map, a, b)` is the one region
joined to both when a and b are different and not joined to each other, else None. `regions_along`, `crossed` and
`first_crossings` read a path's base positions through the same map: positions in no room (a doorway) are dropped,
repeats collapsed, and a crossing is a pair of consecutive rooms that are joined.

## Moments

A moment is a pair of regions A and B, each of a kind **unique in the layout**, with R, their connector. It is asked
only when all of these hold:

- **Crossed.** The window's delivered base positions cross A-R and R-B, each in either direction. The robot's own path
  showed both boundaries.
- **R is not where the window starts or ends.** R holds neither the window's first delivered base position nor any
  position of the end look window (the last look window of the case, the frames just before the end questions:
  `World.end_window`). Both jobs
  start and end each task at their dock or charger, so a connector that held the first position would be answered by
  "where I started", and one that held a position of the window by "where I am", with no sensor. Crossing A-R and R-B
  inside the end look window puts a position in R, so this also keeps `current_view`, which holds the latest frame
  alone, from ever having the path that shows the connection (rule 4); an out-and-back route re-crosses on the way home, so such
  windows ask nothing.
- **Toured.** A and B were each toured over the window: at least `adjacency.TOURED` (half, the share `changed` asks) of
  their inside fine cells observed, in any state (`World.latest_cells` over the window's keyframes), so the camera saw
  both rooms and a perceiver can tell which they are.
- **Only what the camera identifies.** Neither A nor B is a region whose kind a brief of the window names
  (`answer_keys.rooms_named` over the texts `brief.compose` writes for the window's tasks, the first one's paragraph
  included, on the gate channel, as `path_log` and `suite_priors` read them), a region whose kind the job's relevance
  lists for a task of the window (where the job takes the robot: the vacuum's hall, a drone's drop-zone and nest
  regions), nor the region holding the window's first delivered base position. Otherwise the pose log, the briefs and a
  prior learned on the public layouts ("the dock stands in the hall", "a drop point lies in a garden") place the room,
  and the connector is the room the path entered on leaving it: an answer that needs no sensor.

The gold is `{"type": "position", "xy": <R's centroid>}` (to 4 decimals; no moment when the centroid is not 0.2 m inside
R), the gold row's `region` R's polygon, tolerances `{"erode_m": 0.2}`. **Evidence:** the script keyframes of the two
first crossings (A-R and R-B: where the path first enters the second room of each). **Subjects:** the three regions
(`r:<A id>`, `r:<R id>`, `r:<B id>`), so no other end question names a room of the pair (`changed` among them).
**Critical** when any of the three kinds is in a window task's relevance regions.

A robot that flies is never asked: the table's `Mapper` keeps no robot band for a flier (`table.Mapper.band` is None),
so no table system observes a cell of any room, no room is toured and none could answer. The gold keeps that guard on
the robot; `drone/1` does not list the family in its `ends`.

## The answers

- **The table systems** (`oracle`, `memory_table`; `current_view` by the same code on its one frame): the map is the
  case layout's, on every layout (they are instruments: room identity is a perfect perceiver's knowledge). The regions
  `a` and `b` name are the layout's regions of those kinds (each unique, else `unknown`); R is their `connector` (None:
  `unknown`); the delivered base positions of every step up to the ask must `crossed` A-R and R-B and both rooms must be
  `toured` in the cells observed so far; then the answer is the gold's point (`adjacency.answer_point`), else `unknown`.
  The gold's other rules (a brief, the relevance or the start names a room; R holds the window's first position or a
  position of the end look window) decide which questions are asked, not what a system can derive, so the table does not
  read them. `current_view`, which holds the latest
  frame alone and so never its crossings, answers `unknown`. The `space` slot is words and is never read.
- **`path_log`** (`baselines_path_adjacent`) holds the layout's region kinds and never a held-out layout's geometry (it
  models an agent that learned the public layouts, and reads "clean the study" in its briefs). A room it can place is one
  whose kind a brief names alone; with both placed, the room that connects them is where its routes to the two part (the
  last position of its route to the first that is still on its route to the second). The moment rule leaves out every
  pair a brief names, so on real cases it places nothing and answers `unknown` (its pool is empty by the rule below):
  the route stays so that a leak in the moment rule would show as `path_log` scoring.
- **A fact of the layout is learned from public layouts only.** For a family whose reference is `other_case`,
  `suite_priors.prior_records` keeps a record only from a script on a public layout (so `blind`, `questionnaire_blind`,
  `path_log` and `current_view_prior` abstain on a held-out layout, and the form adversary pools the same way), and
  `suite_report.broken_instruments` does not count the emptiness as a vacuous gate there (the family's numbers carry
  `suite_report.ABSTAINS` as a `note`). A family of any other reference keeps the pool of its own layout and job, which on
  a held-out layout is stronger than an honest agent's prior, so its gate is the stricter.

## Declared before data (methodology rule 5)

- **The free yield.** On generated scripts of each held-out layout with its job (seeds 101-130, 4-6 tasks and one task,
  no render, `STBENCH_MAPS` set): the askable moments a case by window length, as numbers only (no room or kind of a
  held-out layout is ever written down). Touring needs observed cells, so this bound counts the crossing, connector
  (start and end look window), brief, relevance and start rules only and says so; the dev gate check gives the real count.
- **It ships only with at least 10 rows passing on some held-out layout.** A layout with fewer than 10 rows gets no
  verdict there, reported. If `path_log` (given the region kinds of a held-out layout, never its geometry) beats
  `all_unknown` by more than 0.10 on a layout, the moment rule has a sensor-free leak and the family is dropped.
- **The pose-and-status shortcuts.** A design-time script (not shipped; `--map`, `--robot`, `--job`, `--seeds
  101-130`, generated scripts, no render, every cell counted toured) scores fixed answers no sensor is needed for
  against each moment's gold with the family's credit: the window's first delivered position, its last, the medoid of
  its path and the medoid of the positions where a run of a task-critical phase ends (`job.relevance(plan).phases`,
  the job's own; no robot is named) in the window's first task, its last task and every task. It prints each answer's penalised mean against `all_unknown` and the pass rule, declared
  here before any held-out number: no answer beats `all_unknown` by more than 0.10 (the shortcut margin), and the last
  position by more than 0.05 (the memory margin: it is what `current_view` holds). It prints counts and means only,
  never a room, a kind or a place, and is not run with a held-out dataset committed.
- **`other_case` is the family's reference, not a shortcut, on held-out layouts**: `gates._group` leaves it
  out of the family's gates there only.
