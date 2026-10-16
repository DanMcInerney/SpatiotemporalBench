# `order`

Which of two things of one kind the robot saw first. End, `memory`. Code: `stbench.questions_things` (shared with
[`when_seen`](when_seen.md) and [`distance_between`](distance_between.md)). The record and questionnaire rules are in
[questions.md](../questions.md).

**Not asked in 0.11.0.** No job's `ends` names it: each job took it out at a gate check of 0.4.0, for too few questions
(under one a case) or a failed shortcut gate (the AMR's `other_case` +0.25), as below. The module and its tests stay.

| | |
| --- | --- |
| Category | C3 Locate |
| Asked | by no job. End, once per relevance kind of the window (the job's `ends` would name it). Each job took it out of `ends`. `vacuum/2` did so at M2's fast gate check (under one question a case). `helper/1`: M3's fast gate check on look-dev windows took it out (0.1 a case), M3's review put it back to be read on the release windows, and the release gate check found 2 questions in 18 cases (under one a case). `amr/1` took it out at the release gate check (4 questions; `other_case` +0.25). `drone/1` took it out at the release gate check (5 questions, 0.28 a case) |
| Text (slots) | "Did you first see the {thing} before the {thing}?" (`first`, `second`: each a description `{"kind", "attributes"}`) |
| Answer, credit | `yesno`, exact |
| Scope, `needs` | `memory`; see |
| Gold function | `questions_things.order(w)`: one `Moment` per relevance kind, keyed by the kind, holding its orderable pairs |
| Slice | none: the unknown pool is empty and the family is not in `SLICED` |

## Pairs

A pair is two *settled* things of one kind (the relaxed rule of 0.5.0: a recognisable sighting, none in the last
look window, every later sighting within `MOVED_NO_M` of the last recognisable one's place;
[`distance_between`](distance_between.md), "Pairs") such that:

- their first recognisable sightings lie at least `ORDER_GAP_S` (30 s) apart;
- the later-seen one has no visibility row, at any pixel count, **at or before** the earlier one's first recognisable
  sighting: a detection at that same keyframe would make the table's order of first sightings wrong for the oracle and the
  table alike (a ceiling failure the compiler must not ask).

The evidence is the two first recognisable sightings, sorted, and their entities.

**Which is named first is a coin per pair**, `stable_hash(case id, the two subjects in sorted order, "named") % 2`: it
reads no answer, so the gold (yes exactly when the thing named first was seen first) is independent of the naming order and
yes and no balance within each kind.

**Two things of one kind only.** A pair of kinds would name where each lives (shoes on the hall's rack and mat, which every
task leaves from; toys on rugs, the sofa and the bed), and a form that reads `slot:first.kind` alone would predict the
answer per kind pair even under a fair coin. With one kind the two slots differ only in attributes, which are drawn
independently of placement. The priors loosen the nested descriptions' `attributes`, then their `kind`.

There is no unknown slice: a thing nobody saw has no first sighting to order. The pair is named by `first` and `second`,
each exactly `{"kind", "attributes"}` (`attributes` `{}` at the least), so the priors' keys have one shape across records.
