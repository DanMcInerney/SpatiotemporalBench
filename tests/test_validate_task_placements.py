"""`validate`'s rule that the world does not change inside a task (`task_placement`): a placement of a non-mover entity
that starts strictly inside a task belongs to an entity in the task's `targets`. Each test breaks one thing in the
hand-built timeline (tests/builders.py: the home helper in the kitchen of `apartment_a`; task 0 runs from 0 to 3.0 s,
task 1 from 7200 s) and checks the one code it expects."""

from __future__ import annotations

from dataclasses import replace

from builders import MUG, MUG2, TABLE, minimal_timeline
from stbench.geometry import quat_z
from stbench.timeline import Placement
from stbench.validate import validate
from validate_kit import codes, with_mover


def moved_mug2(t: float):
    """The fixture with its second mug moved at `t` s to (2.2, 2.15, 0.75) on the table, clear of the first one."""
    tl = minimal_timeline()
    old = tl.placements[4]
    assert old.entity == MUG2
    tl.placements[4:5] = [replace(old, t1=t), Placement(MUG2, t, None, (2.2, 2.15, 0.75), quat_z(1.0), TABLE)]
    return tl


def test_a_placement_inside_a_task_must_belong_to_its_targets(apt):
    """Contract: a mug set down anew at 1.0 s, inside task 0 (0 to 3.0 s), is `task_placement` when the task's targets
    do not name it, and nothing when they do (the thing a helper carries). Regression caught: a script whose world
    changes under the robot's camera (a thing that appears or moves while the robot works), which the `object_where`
    rules would read as a thing that did not move; or a rule that refuses the carry itself."""
    tl = moved_mug2(1.0)
    assert codes(tl, apt) == {"task_placement"}
    named = moved_mug2(1.0)
    named.ego.tasks[0] = replace(named.ego.tasks[0], targets=("R3", MUG2))
    assert validate(named, apt) == []
    other = moved_mug2(1.0)
    other.ego.tasks[0] = replace(other.ego.tasks[0], targets=("R3", MUG))
    assert codes(other, apt) == {"task_placement"}


def test_a_change_between_tasks_or_at_a_tasks_edge_is_not_inside_it(apt):
    """Contract: only a placement that starts strictly between a task's first and last keyframe counts: one in the gap
    (the world's processes, 3600 s here), one at the first keyframe's time (the coins set the world at the script's
    start, 0 s) and one at the last keyframe's time are not inside. Regression caught: a rule that refuses the world's
    own changes in the gaps, or the coins at the first task's start."""
    assert validate(moved_mug2(3600.0), apt) == []
    assert validate(moved_mug2(3.0), apt) == []  # the last keyframe of task 0
    assert validate(moved_mug2(7200.0), apt) == []  # the first keyframe of task 1
    assert codes(moved_mug2(7201.0), apt) == {"task_placement"}  # inside task 1


def test_a_mover_walks_while_a_task_runs(apt):
    """Contract: a mover (a resident, a taxi) is placed anew all the time, and the rule leaves it out: a mover that
    starts to stand at a new place at 1.0 s, inside task 0, is not `task_placement`. Regression caught: a rule that
    refuses every script with a resident."""
    assert validate(with_mover(minimal_timeline(), (2.0, 1.0), t0=1.0), apt) == []
