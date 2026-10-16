# `changed`

Where in a room something was added, moved or removed between two of the window's tasks. End, `memory`. Code:
`stbench.questions_changed` (the gold) and `stbench.change_events` (the event rules, which the table systems call too).
The record and questionnaire rules are in [questions.md](../questions.md).

**Not asked in 0.11.0.** No job's `ends` names it: it left every job at a gate check. 77% of the vacuum's golds were the
empty set, which the current view and the question's form guess, and the helper's and the AMR's checks failed it or found
too few questions (below). The module and its tests stay.

| | |
| --- | --- |
| Category | C6 Change |
| Asked | by no job. End, once per moment (the job's `ends` would name it; a window of fewer than three tasks has no moment). `vacuum/2` dropped it at M2's gate check (77% of its golds were the empty set). `helper/1`: M3's fast gate check took it out (no question in 10 look-dev cases, whose windows of 1-3 tasks cannot ask it), M3's review put it back to be read on the release windows, and the release gate check dropped it (it fails shortcuts: `blind` +0.33, `current_view` +0.67, n 6). `amr/1` took it out of `ends` at the release gate check (2 questions in 18 cases; its oracle 0.60, an open instrument question for a later release). `drone/1` never asked it |
| Text (slots) | "Where in the {room} did something change between your {first} task and your {second} task? List where each thing was added or moved to, and where each was removed from." (`room`: the room's kind, `tasks`: the two case task indices, `[i, j]`) |
| Answer, credit | `positions` (xy, no height), `f1` with `match_m` 0.5 m; the gold may carry don't-care regions of radius 0.5 m |
| Scope, `needs` | `memory`; see, pose (the gold certifies "removed" and "added" from the recorded depth, which the camera-only channel does not send) |
| Gold function | `questions_changed.moments(w)`: one `Moment` for each (room, i, j), keyed "<room>\|<i>\|<j>", one answerable candidate |
| Slice | none: a moment's gold is an event list, and an empty list is an answer |

## Moments

A moment is a room of the public map and two tasks of the window at least two apart (`j - i >= 2`): the world's
processes act only between tasks, so tasks are the passes, and more elapsed time means more change. A medium window's
tasks stay with `object_where` (a `changed` question names two whole tasks as times, so it would otherwise leave
`object_where` nothing to ask in a window of two). As built, the declared fallback is taken
(`questions_changed.FIRST_LAST`, below): a room gets one moment, for the window's first and last tasks. The room is
named by its kind, so only a kind that is **unique in the layout** is asked. It must be **toured in both tasks**: at
least `adjacency.TOURED` (half) of its inside fine cells observed, in any state, in that task's keyframes
(`World.latest_cells`; `adjacency.toured`, the rule `adjacent` asks too); else there is no moment, a dead band, never an
empty gold from a room the robot did not look at.

## What a side reads of a thing

The things are entities of mobility `light`, `carried` and `clutter`. For each, in each of the two tasks:

- **A sighting counts only when the thing stood on the floor then** (`World.on_floor`: its base within the robot's
  `climb_m` of the robot's own base height). A thing moved from a table to the floor has no `a` and a `b`; a mug moved
  along a table has no sighting at all. The table systems read the same rule from what they hold (the box bottom against
  the delivered base height of the step).
- `a` is its xy at the last seen-bound floor sighting in task i (`Sightings.seen`, 269 px at the global bound), `b` the
  same in task j. A dead-band sighting in a task with no seen one adds its place to `unsure`: a thing only glimpsed in
  a task has no place the robot can vouch for. One in a task that has a seen sighting, before or after it, adds
  nothing: nothing moves within a task, so a glimpse after the last seen sighting is at the thing's own seen place.
- A thing whose **description** (kind and describable attributes) another entity **sighted in the window** shares has
  all its floor sightings' places `unsure`: the table tells things apart by description, so it can tell that twin too.
  An entity no frame shows changes nothing (truth rule 2).
- A **mover** (a resident) gives only `unsure` places, every sighting of it in either task: a person is somewhere else
  in every pass, which is not a change.

## The rules (`change_events`)

`events(things, inside, free_i, free_j)` and `swallow(events, dont_care, match_m)`, one function for both sides:

- `a` and `b`, `d = |a - b|`: under `MOVED_NO_M` (0.15 m) nothing; to `MOVED_YES_M` (0.45 m) a don't-care place at each
  end inside the room; else an event at `b` when `b` is inside the room (and a don't-care place at `a` when that is),
  or at `a` when only `a` is.
- `a` only, inside the room: an event when the place was seen empty in task j (`grid.place_free` over the latest cells
  of task j's keyframes, a disc of the thing's `min(size[0], size[1]) / 2`), else a don't-care place.
- `b` only, inside the room: the same against task i.
- A thing with `unsure` places gives those inside the room as don't-care places and no event.
- `swallow`: an event within `match_m` of a don't-care place becomes one, until none moves (the scorer before
  0.5.0-control dropped an answer there before matching, so a gold item there could never be matched; now it matches
  first and the rule only keeps the gold as it was); both lists are sorted to 4 decimals and don't-care places within
  0.01 m are one.

The gold is `{"type": "positions", "items": [{"xy"}...], "dont_care": [{"xy", "radius": 0.5}...]}`, tolerances
`{"match_m": 0.5}`. **Evidence:** the keyframes of the sightings the surviving events read, and the first keyframe of
each of the two tasks (never empty). **Subjects:** the room (`r:<region id>`), the two tasks as times
(`w:<k0>-<k1 - 1>` each, script keyframes: no other end question may name a time inside them, methodology rule 6) and the
things with an event. Critical when the room is among the relevance regions of task i's or task j's plan.

## Declared before data (methodology rule 5)

A constant empty answer scores 2E - 1 at an empty share E (F1 is 1 when both lists are empty), so the family fails its
shortcut gate through `blind` and the form adversary once E passes 0.55. M2 has no pre-gate:

- **Before the dev render (no render):** over a long window's moments on `vacuum/2`'s generated scripts with the world
  processes (seeds 101-130 and 141-200), the share in which some thing of the three mobilities on the room's floor moved
  at least `MOVED_YES_M`, or came onto or left the floor, between task i's end and task j's start is an upper bound on
  the non-empty share. Under 45%: the fallback from the start, one moment per room for the window's first and last tasks
  only (the longest elapsed time); under 45% for those too, the family is dropped before data.
- **At the dev gate check:** more than 55% of a group's golds empty: the fallback if not taken, else the family is
  dropped. The fallback was taken at M2's fast gate check (62% of the golds empty over pairs two apart), and the code
  keeps it (`FIRST_LAST = True`).
