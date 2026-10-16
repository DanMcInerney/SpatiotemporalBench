"""`path_log`'s answer to `way_to` (`baselines_path_way`) and the family's registrations. The rule is "turn back the way
you came": the heading from the base at the ask toward the base pose `look_windows.LOOK_WINDOW` (8) steps earlier, from
the delivered poses alone."""

from __future__ import annotations

from stbench import baselines_path, baselines_path_own, baselines_path_way, questions_text, table_answers
from stbench.look_windows import LOOK_WINDOW
from stbench.robots import ROBOTS

HELPER = ROBOTS["helper"]
SLOTS = {"kind": "mug", "attributes": {}, "reach_m": 1.0, "room_m": 0.49}


def rule(poses, base, yaw_deg, slots=SLOTS):
    return baselines_path_way.own_answer(HELPER, poses, slots, base, yaw_deg)


def drive_east(n=12):
    """`n` delivered poses (x, y, yaw in degrees) one metre apart along +x, the last the ask's."""
    return [(float(i), 0.0, 0.0) for i in range(n)]


def test_the_own_log_rule_turns_back_toward_where_the_robot_was_eight_steps_ago():
    """Contract: the heading from the base at the ask toward the pose sent `LOOK_WINDOW` steps before the ask, in
    degrees to the left of the way the robot faces, in (-180, 180]. Hand cases: driving east, 12 poses, the ask at (11,
    0): the pose 8 steps earlier is the 4th (3, 0), straight behind: facing east (0) that is 180; facing north (90) it
    is 90 to the left; facing south (-90) it is -90 (270 wrapped); a robot back at the place it was eight steps ago has
    no direction to turn (`unknown`). Regression caught: a heading in the world frame, an angle past 180, the wrong end
    of the log, or another number of steps."""
    poses = drive_east()
    assert LOOK_WINDOW == 8 and poses[-1 - LOOK_WINDOW] == (3.0, 0.0, 0.0)
    assert rule(poses, (11.0, 0.0), 0.0) == {"type": "heading", "deg_left": 180.0}
    assert rule(poses, (11.0, 0.0), 90.0) == {"type": "heading", "deg_left": 90.0}
    assert rule(poses, (11.0, 0.0), -90.0) == {"type": "heading", "deg_left": -90.0}
    assert rule([(0.0, 0.0, 0.0)] * 4 + [(2.0, 2.0, 0.0)] * 5 + [(0.0, 0.0, 0.0)], (0.0, 0.0), 0.0)["type"] == "unknown"


def test_the_rule_needs_a_log_that_long_and_a_robot_that_has_moved():
    """Contract: with `LOOK_WINDOW` poses or fewer there is no pose that far back, and a robot within a fine cell of it
    has no direction to turn: both are `unknown`. Hand cases: 8 poses (index 0 to 7: none 8 before the last) unknown, 9
    poses answer; a robot that has not moved from the pose 8 steps ago unknown. Regression caught: a heading made up
    from the first pose of a short log, or an angle from a zero vector."""
    assert rule(drive_east(8), (7.0, 0.0), 0.0)["type"] == "unknown"
    assert rule(drive_east(9), (8.0, 0.0), 0.0) == {"type": "heading", "deg_left": 180.0}
    still = [(1.0, 1.0, 0.0)] * 12
    assert rule(still, (1.0, 1.0), 0.0)["type"] == "unknown"


def test_the_rule_reads_the_pose_log_alone():
    """Contract: `path_log` reads no picture and no thing: the answer is the same for every thing the question names
    (`path_log` knows the kind of nothing). Hand case: a mug and a bag, and a different R and T, answer alike on one
    log. Regression caught: a rule that looks the thing up (a place from a prior, the faced places) and so would not be
    an own-log rule held to the bar of `baselines_path_own`."""
    poses = drive_east()
    other = {"kind": "bag", "attributes": {"color": "red"}, "reach_m": 2.0, "room_m": 0.9}
    assert rule(poses, (11.0, 0.0), 30.0) == rule(poses, (11.0, 0.0), 30.0, other)


def test_the_family_is_registered_in_every_registry_it_belongs_to():
    """Contract: the family's modules are found by the registries that the other families' are (`questions_text`'s
    family list, `table_answers`', `baselines_path`'s and the own-log bar's), so the compile asks it, the table systems
    answer it and `path_log` answers it through the bar every own-log rule stands by. Regression caught: a module the
    registry skips (a mistyped name is skipped, not an error), so the family is asked and nobody answers it."""
    spec = questions_text.FAMILIES["way_to"]
    assert (spec.when, spec.answer_type, spec.credit, spec.scope, spec.reference, spec.own_log_screen) == (
        "during", "heading", "route", "memory", None, False)
    assert "way_to" in questions_text.LOOKS and "way_to" in questions_text.SLICED
    assert "way_to" in table_answers.ANSWERERS and "way_to" in baselines_path.PATH_ANSWERERS
    assert baselines_path_own.rules()["way_to"] is baselines_path_way.own_answer
    assert not questions_text.FAMILIES["open_heading"].own_log_screen
