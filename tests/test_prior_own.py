"""The `own` key of a prior record (`suite_priors`, `answer_keys.RECORD_KEYS`): `path_log`'s own-log answer to a
question of a template with an own-log rule (`clear_path`, `open_heading`), which `baselines_path_own.kept` weighs the
rule by. Hand cases on `synth.Synth`'s script of one task of 20 steps in the kitchen with a decision point after step 8,
where the robot heads +x (the script turns it 45 degrees a step, 360 degrees in eight): the base poses are set by
hand."""

from __future__ import annotations

import pytest

from baseline_kit import EVENTS, number, play
from own_kit import heading
from stbench import mapspec, suite_priors
from stbench.baselines_path import PathLog
from stbench.blind import Prior
from stbench.questions_words import clear_slots
from stbench.stream_protocol import GATE_CHANNEL
from stream_kit import case_on, during_q, gold_row
from synth import HOME, Synth

BD = "before_doorway"
UNKNOWN = {"type": "unknown"}
# The bent path of `test_path_log_clear` (round a box at x 1.5 to 1.7 on the line y 1.5), on steps 0 to 8; the robot
# stands at (0.5, 1.5) heading +x at the ask. After it (steps 9 to 19) it drives on east along the line to (5.5, 1.5),
# which the system has not been sent at the ask.
BENT = {0: (3.5, 1.5), 1: (3.5, 1.5), 2: (3.5, 1.5), 3: (3.5, 1.5), 4: (2.0, 1.5), 5: (1.9, 0.9), 6: (1.2, 0.9),
        7: (1.1, 1.5), 8: (0.5, 1.5)}
# The line y 1.5 from x 0.5 (step 0) to x 2.5 (step 8; steps 1 to 7 stand at the room's centre, x 2.0, on it). After it
# the robot drives on east to (5.5, 1.5).
LINE = {0: (0.5, 1.5), 8: (2.5, 1.5)}
ON_EAST = {k: (5.5, 1.5) for k in range(9, 20)}
CASES = {
    # ahead: the corridor from the front edge (0.67, 1.5) east: the axis cells at 0.72 to 1.22 swept, those at 1.32 to
    # 1.72 not and 1.82 on again, so the first unswept axis point is at s = 0.65 from the edge: 0.6 (the detour).
    # back: the corridor from the rear edge (0.33, 1.5) west: its first cell (0.28) is unswept: no detour, `unknown`
    "bent": (BENT | ON_EAST, [("clear_path", clear_slots("drive", "ahead", 3.0), "number", number(0.6)),
                              ("clear_path", clear_slots("drive", "back", 3.0), "number", UNKNOWN)]),
    # the run west from the rear edge (2.33, 1.5) is 20 cells, 2.0 m, the longest (ahead the first cell is unswept): 180
    "line": (LINE | ON_EAST, [("open_heading", {"cap": 3.0}, "heading", heading(180.0))]),
}


@pytest.mark.parametrize("name", CASES)
def test_a_record_carries_the_answer_the_system_gives_at_the_ask(tmp_path, name):
    """Contract (`answer_keys.RECORD_KEYS`, `baselines_path_own`): the record of a question of a template with an
    own-log rule holds `path_log`'s own-log answer, the very one the system gives at the ask: from the ask step's base
    pose and heading, the slots of the question (the direction of a `clear_path`) and the delivered poses of the steps
    up to the ask (9 of them, 0 to 8), none of those after it, and the robot as its `start` message gives it. Hand
    cases (above): the bent path answers 0.6 ahead and `unknown` back; the line answers 180 degrees left. Each is asked
    of the system on the stream (kept unasked, an `in_pool` prior) and built as the record of the script: the same
    answer. The poses after the ask would change both (the east run fills the box's cells: no detour; it is a 3.0 m
    run ahead: heading 0), and the record of a question of any other template (`path_ahead`) holds none. Regression
    caught: a rule judged on records built from other poses than the system holds (all the window's, which includes
    steps after the ask, or a direction that ignored `back`), so what `kept` scores is not what the system answers; or
    an own-log rule with no record answer."""
    poses, asked = CASES[name]
    synth = Synth(tasks=[[(HOME.rooms[0], 20)]], decisions=[(8, BD)], poses=poses)
    qs = [during_q(n, 8, template=t, slots=slots, answer_type=kind) for n, (t, slots, kind, _) in enumerate(asked, 1)]
    other = during_q(9, 8)
    case = case_on(tmp_path, synth, [*qs, other])
    run, _ = play(case, PathLog, channel=GATE_CHANNEL, map=mapspec.load_map("apartment_a"), events=EVENTS,
                  prior=Prior((), in_pool=True))
    for q, (_, _, _, want) in zip(qs, asked, strict=True):
        assert run.answers[q["qid"]] == want, q["qid"]
        assert suite_priors.record_for(case, q, gold_row(q), None)["own"] == want, q["qid"]
    assert suite_priors.record_for(case, other, gold_row(other), None)["own"] is None
