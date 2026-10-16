# `clearance_then`

Was the place the job used clear of anything taller than a height at a named moment of the job? End, `memory`. Code:
`stbench.questions_zone`. The record and questionnaire rules are in [questions.md](../questions.md); the zone and its
states are those of [zone_clear.md](zone_clear.md).

**Not asked in 0.11.0.** No job's `ends` names it. `drone/1`, the one job that did, dropped it for good at the 0.4.0
release gate check, because its gold fails the shortcut gates (below). The module and its tests stay. No shipped job's
`JobSpec.zones` keys an event wording (the keys are decision kinds), so a job that named it again would still get no
moment until it declared one.

| | |
| --- | --- |
| Category | C1 Room to move |
| Asked | by no job. End, once per moment: a wording that is both a key of `JobSpec.zones` and of `JobSpec.events` and that resolved in the task (the job's `ends` would name the family). `drone/1` named it and dropped it at the release gate check (`other_case` +0.21 at 24 questions, and `blind` BROKEN at -0.21; its golds are balanced by a fair coin, which left the lead possibly chance). It was mapped back and read once more, by a rule declared before the data, over all 12 drone scripts (seeds 2001-2012): 49 questions, and it fails `memory_needed` (`current_view_prior_in_pool` +0.143) and `shortcuts` (`form_in_pool` +0.143, with `blind` BROKEN at -0.122). It is dropped for good, and no later release asks it: after a response (the drone aborts, the AMR skips, the helper places elsewhere, the vacuum skips the room) the robot's own path and subgoal tell the zone's state, so an end question about it is one about its own path, not a memory of the world; the state of its passages and goal places was asked in-run, while it was undecided ([zone_clear.md](zone_clear.md)) |
| Text (slots) | "Was {zone} clear of anything taller than {height} m {event}{task}?" (`zone`, `height`: the rule's; `event`: the wording; `task`: the case task index, None when the window holds one task) |
| Answer, credit | `yesno` (yes: clear), exact; `unknown` is a gold |
| Scope, `needs` | `memory`; see, pose, status (the event is resolved from the steps' phases); `mapper_gold` |
| Gold function | `questions_zone.clearance_then(w)`: one `Moment` for each (task, wording), keyed "<task>\|<wording>", its one candidate in the answerable pool; critical, subjects the time (`w:<k_e>`) and the zone's piece (`e:<id>`) |
| Slice | none: the rule gives `unknown` (a zone nothing observed in the moment's look window) |

## The moment and the state

`World.events` resolves each wording of the job's events in each task of the window to a script keyframe `k_e`, as a
system resolves it from the delivered steps. The zone is the rule's zone for the delivered pose at `k_e`
(`zones.zone_of`: the nearest piece or way, or the window the base faces along its delivered heading). Its state is
`zones.zone_state` over **the keyframes of the look window that holds `k_e`, up to and including `k_e`**: what had
been seen at or before the moment, never after it. A thing that walks into the zone after `k_e` but in the same window
changes nothing; a zone seen only in an earlier window is `unknown` (the look window of the moment is what the question
reads). A window of one task has no task phrase.

## What is not asked

A moment whose zone has any cell observed in the window's last look window is not asked: the recent view would answer
it (methodology rule 4). A map with no piece of the rule's kind gives no moment. Two moments about one zone share its piece
subject, so the subject rule asks one of them in a case.

**Under one-frame delivery** (0.10.0) the table answerer of this family sees a look window of one frame, so it no longer
reproduces the gold. No job asks the family, so no score is affected; a job that asks it again first reworks it for
one-frame delivery.
