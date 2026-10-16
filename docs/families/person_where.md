# `person_where`

Where a person was, as the robot last saw them, at a named moment. End, `memory`, temporal. Code:
`stbench.questions_change` (the gold), with `stbench.questions_where` (`answerable_at`) and `stbench.questions_world`
(`World.steady`). The record and questionnaire rules are in [questions.md](../questions.md). New in 0.5.0.

**Not asked in 0.11.0.** No job's `ends` names it. It left `vacuum/2` and `helper/1`, the only jobs that named it, at
the 0.5.0 dev gate read, because it failed `time_needed` in all three reads (the resident mostly stays where they were
last seen, so a notebook that keeps one place for each person, with no time, scores above the floor). The module and
its tests stay.

| | |
| --- | --- |
| Category | C3 Locate |
| Asked | by no job. End, once per moment of the job's events in each task of the window (a job's `ends` would name it), at most once a case. `helper/1` and `vacuum/2` named it in the supply rule's starting map (below) until the dev gate read took it out of both, for failing `time_needed` in all three reads; `amr/1` and `drone/1` never did. It has no 0.4.x record |
| Text (slots) | "Where was the {person}, as you last saw them, {event}{task}?": "Where was the person, as you last saw them, when you started cleaning in your second task?" (`kind`, `attributes`, `event`, `task`: `object_where`'s slots) |
| Answer, credit | `position` (xy), the distance ladder widened by `rules.widening` of the person's own box (a mover's kind is not in the item catalog) |
| Scope, `needs` | `memory`; see, pose, status. `FamilySpec.temporal` is true |
| Gold function | `questions_change.person_where(w)`: one `Moment` for each (task, wording) the window's events resolve, keyed "<task>\|<wording>" |
| Slice | none: only a moment at which the person was seen is asked, and the unknown pool is empty |

## Who is asked about

A person is a `mover` entity of kind `person` with no attributes (`docs/people.md`), so a person's description is the
kind alone, and it names one person only when the world has exactly one: the home's resident. The warehouse's three
pickers and the downtown's eight pedestrians share one description each, so no question can tell one from another and
none is asked (`questions_change.movers`: the movers whose description is unique in the world, the rule
`object_where` applies to a thing). Of the shipped jobs that is `vacuum/2` and `helper/1`, which share the home.

## The gold

For a moment (a wording resolved over a task to `k_e`), the person is asked about when:

1. its last recognisable sighting at or before `k_e` lies in the moment's own task (`answerable_at`; the
   recognisability rule is [`object_where`](object_where.md)'s, one frame at a time);
2. nothing seen between that sighting and the moment puts the person elsewhere (`World.steady`): every sighting of the
   person from that one to `k_e`, a glimpse or a dark one included, lies within `MOVED_NO_M` (0.15 m) of it; and
3. the person has no visibility row, at any pixel count, in the window's last look window (the recent view would answer:
   methodology rule 4).

The gold is the placement's (x, y) at that sighting, which is the evidence. A person walks while the robot is away, so
the place the robot last saw them in an earlier task is not where they are by the window's end. Rule 2 keeps a person
the robot glimpsed walking off from being asked about the place it left, and rule 3 keeps the last look window from answering
a question about an earlier moment.

The person is the question's subject (`e:<id>`), and the subject rule forbids two end questions about one subject, so a
case asks it once. It is critical only where the job's relevance kinds name people; none of the four jobs' does, so its
rows are situational in the report's relevance split (`questions_where.critical`).

## Why it is temporal, and the gates

It asks for a place at an earlier time of a thing that moves between tasks. `latest_table` (one latest sighting a
track, no event table) answers with the person's latest place whatever the moment, which is the gold only where the
person did not move after it (and `first_table`, the first sighting, with the first); `time_needed` holds the larger of
the two to `all_unknown` + 0.10 on the family, and `memory_table` reads
the latest sighting inside the moment's task at or before it (`table_change.place_at`). The family is unsliced, so
`all_unknown` scores 0 on it. The risk declared before the data is that `blind`, which learns from other scripts'
records, finds the kind or the place the resident favours in the answer; a family that fails a gate is dropped by rule
and not re-tuned. What happens where the robot did not look changes no gold (truth rule 2).

## Supply

The family asks one question a case at most, in a case whose person is seen in a task and is steady to a moment of it.
The supply rule ([questions.md](../questions.md)) keeps it in a job's `ends` only when the whole-root compile asks at
least 0.5 of its questions a case in some shipped length; otherwise it leaves the job before any gate is read.
