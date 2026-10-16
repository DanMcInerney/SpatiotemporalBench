# `when_seen`

When the robot last saw a thing. End, `memory`. Code: `stbench.questions_things` (shared with [`order`](order.md) and
[`distance_between`](distance_between.md)). The record and questionnaire rules are in [questions.md](../questions.md).

**Status: no job asks this family in 0.11.0.** It is in the registry and its code is kept. 0.4.0's gate checks dropped
it from all four jobs: `helper/1` and `drone/1` after the repair of its time shortcut, and `amr/1` and `vacuum/2` at the
binding gate check over the shipped cases (below).

**In 0.5.0** a recognisable sighting is judged one frame at a time, with a box clear of the image's edge and a
depth as well as the pixels and light below ([`object_where`](object_where.md), "Evidence"). `when_seen` keeps the
strict *settled* rule below (no sighting at all after the last recognisable one), because its gold is a time; `order`
and `distance_between` use the relaxed one ([`distance_between`](distance_between.md), "Pairs").

| | |
| --- | --- |
| Category | C3 Locate |
| Asked | by no job. End, once per relevance kind of the window (a job's `ends` would name it). `helper/1` and `drone/1` dropped it after the repair of its time shortcut (below): the helper's (36 questions, separation +0.81 before the repair) fails shortcuts (`blind_in_pool` +0.15), the drone's (28, +0.93 before the repair) fails `memory_needed` and `shortcuts` (+0.30). `vacuum/2` and `amr/1` had passed the repaired gates (the AMR's release gate check: 28 questions, separation +0.64; the vacuum's shortcuts sat 0.03-0.04 under the floor) and were dropped at the binding gate check over the shipped cases (below) |
| Text (slots) | "When did you last see the {thing}?" (`kind`, `attributes`) |
| Answer, credit | `time` (the local clock, `YYYY-MM-DDTHH:MM:SS`), the ladder 15, 30, 60 and 120 s (`questions_things.TIME_LADDER_S`, the minute scale) |
| Scope, `needs` | `memory`; see |
| Gold function | `questions_things.when_seen(w)`: one `Moment` per relevance kind, keyed by the kind |
| Slice | sliced (`SLICED`): the unknown slice is drawn per moment (`questionnaire`) |

## Things

The things are the three end families' shared ones: carried things of the relevance kinds of the window's tasks
(`JobSpec.relevance` of every task's plan, the kinds sorted) whose description is unique among the world's describable
things (`World.things`); light furniture and clutter are never asked. A thing is named by its kind and describable
attributes. A thing is one of:

- *settled* (answerable): it has a recognisable sighting in the window (at least 1,024 px and a mean luminance of
  `recognise.RECOG_LUM`); it has no visibility row, at any pixel count, in the window's last look window (the recent view would
  answer: methodology rule 4); and no sighting at or above its glimpse bound comes after its last recognisable one (the memory
  table would read that later, unrecognised sighting where the gold reads the earlier one). The gold is the case clock of
  its last recognisable sighting, which is the evidence;
- *unknown*: it exists in the window and no visibility row shows it anywhere in it, or it is a description of a relevance
  kind that matches nothing in this world, drawn from other seeds' worlds of the map by the generator's own sampler
  (`questions_where.never_seen`);
- *skipped*: anything else (a glimpse, only dark, a sighting too late to be settled).

The question is about the time of an observation, never of an unseen event: a thing moved while the robot was away is
still last seen when the robot saw it. The same-task rule of `object_where` does not apply (the question asks about the
time, not the thing's state then). Both pools hold the same kinds, so no kind predicts the slice. The ladder is written
on unknown rows too, so a family's rows look alike. A gold is what was last seen; nothing unobserved changes it (truth
rule 2).

## The time shortcut, and why two jobs dropped the family

A `time` gold can be reached from the clock alone when a job ends every task the same way. The helper and the drone end a
task by going home, so a thing's last sighting falls 12-67 s before the window's end, and a rule that answers a fixed
number of steps before the end scored +0.30 over `all_unknown` on the drone (a learned lag +0.15 on the helper). No gated
shortcut could see it: each case is on its own day, and the learned shortcuts scored time candidates against other
cases' clocks. The instrument was repaired once: a prior record holds the delivered clock of its ask step, a
pool learns each time gold as its lag behind that clock and answers on the asker's clock, and `other_case`'s time
answers are put back on the asking case's clock. The `shortcuts` and `memory_needed` gates also read each learned
shortcut under the pool's own winner (`<name>_in_pool`) as well as under the held-out rule, taking the larger score.
Under the repaired gates the family passed in `vacuum/2` and `amr/1` and failed in `helper/1` and `drone/1`, where it
was dropped.

## The binding gate check dropped it from the AMR and the vacuum

0.4.0's binding gate check was one whole-root check over the 84 short and medium cases it shipped (point verdict). On it:
- **`vacuum/2`** (13 questions) fails `memory_needed` (`current_view_prior_in_pool` +0.269 over the 0.05 bound) and
  `shortcuts` (`blind_in_pool` +0.269 over 0.10), and `blind` and `path_log` are BROKEN (-0.115 under abstaining's
  0.077).
- **`amr/1`** (39 questions) passes its gates, but `questionnaire_blind` is BROKEN (0.154 against abstaining's 0.256),
  a line the export refuses. With the long cases in (64 questions) it was not.

A failing family is dropped and never forced, so both jobs ask it no more. No job in 0.4.0 or any later release asks
`when_seen`.
