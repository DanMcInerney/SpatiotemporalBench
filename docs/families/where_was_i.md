# `where_was_i`

Where the robot's base was at a named clock a minute or more ago. During, `memory`, own path (`reference` `path_log`).
Code: `stbench.questions_self` (the gold). The record and questionnaire rules are in [questions.md](../questions.md).

**Status: no job asks this family in 0.11.0, and none has since 0.5.0.** It is in the registry and its code is kept,
but `vacuum/2` and `amr/1`, the two jobs that asked it in 0.4.x, no longer name it (`JobSpec.decisions` maps no decision
kind to a family), so no case contains one. It was dropped because it is a lookup of the frame header. The pose and the
clock are printed in every frame the robot is sent, so a question that names a clock a minute back and asks where the
base was reads one line of the stream; both reference agents scored 100% on it, and it measures neither what the robot
saw nor what it remembers of the world ([methodology.md](../methodology.md), section 1: own path and progress are not
asked). A replacement that is anchored in perception ("where were you when you first saw X") would need its own check
first, so none is built.

The text below is the family as 0.4.x asked it.

| | |
| --- | --- |
| Category | C3 Locate |
| Asked | by no job (above). In 0.4.x: during, at a decision point of a kind whose `JobSpec.decisions` names it (`vacuum/2`: `before_doorway`; `amr/1`: `before_station`, `before_undock` and `before_crossing`, where it passed the release gate check with 32 questions and separation +1.00; `helper/1` mapped it at `before_doorway`, `before_grasp` and `before_place` and dropped it at the release gate check, where `other_case` scored +0.24 over `all_unknown` since the helper's routes repeat; `drone/1` does not ask it); a point with no step old enough is a dead band and not asked |
| Text (slots) | "Where were you at {clock}?" (`clock`: an ISO clock the stream delivered at a step of the window) |
| Answer, credit | `position` (xy, metric `xy`), the distance ladder 0.25, 0.5, 1, 2 m |
| Scope, `needs` | `memory`; pose |
| Gold function | `questions_self.where_was_i(w, k)`: a `Cand`, or None in a dead band |

## The gold

- **The steps that can be named** are the window's keyframes `s` before the first keyframe of the decision's look window
  (`World.window_ks(k)`, at most 8 steps) with `t(k) - t(s) >= MIN_AGO_S` (60 s), in keyframe
  order. Nothing before the window counts: the stream delivers no clock before the case's first step, so a window's
  first minute has no step to name, and a point there is not asked. A window of several tasks may name a step of an
  earlier task, whose clock the stream delivered as the case's own.
- **The named step** is `eligible[stable_hash(case id, k, "where_was_i") % len(eligible)]`: the draw reads the case id,
  the decision and the window's times alone, so the clock asked reveals nothing of the world or of the path.
- **The gold** is the delivered base's (x, y) at that step (`sensors/pose.jsonl`'s `pose.pos`, to 4 decimals): the base,
  never the camera, which stands ahead of it (the vacuum's 0.13 m). Always answerable; the evidence is the one step, no
  entity; subjects are the decision's time `w:k` and the step's `w:s`; critical.
- **The slot** is `World.clock(s)`, the clock the stream delivered at `s` (`case.case_clock`, as `case.case_rows` writes
  it): never the script's own time. Two steps of a task are at least `min_dt_s` (1 s) apart, so a clock names one step.

## Why metres, not the room

A credit `region` (a point in the room the robot was in) was considered. That variant is predicted without
memory: the room the robot was in at a clock is the plan's room order at that offset (`other_case` at the same moment)
or the room it is in now (`current_view_prior`), so it would fail the gates by construction. The metres are predicted by
neither, and their memoryless answer, where the robot is now, is what `current_view` gives, so the memory-needed gate
measures it. The answer from the robot's own log, the pose at the nearest logged step, is `path_log`'s reference.

## Beside it

The table systems answer it from the step they hold nearest the clock (`table_self`); `path_log` from its pose log
(`baselines_path_self`).
