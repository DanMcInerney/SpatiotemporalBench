# `slots_empty`

Which places of a kind that the robot passed were clear of anything taller than a height when it last saw them? End,
`memory`. Code: `stbench.questions_zone`. The record and questionnaire rules are in [questions.md](../questions.md); the
zones and their states are those of [zone_clear.md](zone_clear.md).

**Not asked in 0.11.0.** No job declares a `slot_zones` rule, so the family has no moment in any shipped job. `amr/1`,
whose staging lanes it would list, dropped its `slot_zones` because a constant list of the lanes most often clear beat
`all_unknown` by more than a shortcut may (+0.278, item (s) of the gate table in [job-amr.md](../job-amr.md)). The
module and its tests stay.

| | |
| --- | --- |
| Category | C5 What matters here |
| Asked | by no job. End, once per rule of `JobSpec.slot_zones` (keyed by its `piece_words`) with at least one place passed (the job's `ends` would name the family). `amr/1` dropped its `slot_zones`, since a constant list of the lanes most often clear beat `all_unknown` by more than a shortcut may |
| Text (slots) | "Which {places} that you passed were clear of anything taller than {height} m when you last saw them? Give where each one is." (`places`: the rule's `piece_words`; `height`: its `height_m`) |
| Answer, credit | `positions`, F1 at `match_m` (below) with don't-care places; never `unknown`: nothing clear is the empty list |
| Scope, `needs` | `memory`; see, pose; `mapper_gold` |
| Gold function | `questions_zone.slots_empty(w)`: one `Moment`, one candidate; critical; evidence the keyframes of the passed places' states; subjects below |
| Slice | none |

## The gold

Every piece of the rule's kind is a zone (`zones.zones_of`, in map order). A zone is **passed** when any of its cells
was observed in the window (the zone band's states over every window keyframe). For a passed zone, with its state over the
window (`zones.zone_state`):

- a cell of it observed in the **end look window** (the recent view would answer it): a don't-care place at its centroid;
- else **clear**: an item at its centroid; **blocked**: nothing; **unknown** (seen, but under 80% free): a don't-care place.

A zone never passed is nothing (and is not listed: "that you passed"). A don't-care place has the match radius.

**`match_m`**: the least of half the longest side of the rule's zones and 0.45 x the least distance between two of
their centroids, at least `rules.MATCH_M` (0.5 m), written on the gold row's tolerances and on every don't-care place,
so no two zones' items are ever one match.

**Subjects**: the entity of every passed zone's piece, and every light or carried thing with a sighting in the window
whose placement at its last sighting overlaps a passed zone (the pallet standing on a lane): no other end question names
a thing whose place tells a slot's state.

## Declared before data

A constant empty answer scores 2E - 1 at an empty-gold share E: the free gate (the job's lane) maps the family only where
the share of empty golds is at most 0.52 and no shortcut beats `all_unknown` by 0.10 (methodology rule 5).
