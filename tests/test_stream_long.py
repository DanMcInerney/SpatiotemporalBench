"""A multi-task case through `stream.run_case` (plan §5.1's B3: "multi-task cases run end to end"): a six-task window
of an eight-task script streams its tasks, gaps, statuses and frames as the timeline holds them, every number relative
to the case. The script is `synth.Synth` with three keyframes (1.5 s apart) in each task and tasks 7,200 s apart, so by
hand the unobserved gap before a task is 7,200 - 2 x 1.5 = 7,197 s; the window is tasks 2 to 7 (script keyframes 6 to
23) with one decision point, at script keyframe 10: the second step of the case's second task."""

from __future__ import annotations

from stream_kit import K, case_on, clock, during_q, end_q, observed, ops, play
from synth import Synth

GAP_S = 7197.0   # 7,200 s between starts, less the 2 x 1.5 s a task's last keyframe is after its first


def long_case(root):
    """Eight three-step tasks in the kitchen; the case is the last six, asked a during question at the decision point
    (script keyframe 10) and an end one."""
    synth = Synth(tasks=[[(K, 3)] for _ in range(8)], decisions=[(10, "before_doorway")])
    return case_on(root, synth, [during_q(1, 10), end_q(2)], a=2, b=7, length="long")


def test_a_long_case_streams_as_the_timeline_holds_it(tmp_path):
    """Contract (plan §5.1): a six-task window starts at step 0 and streams: a `task` message before the first step of
    each task, with its index from 0, its clock (the case's start clock plus 7,200 s a task) and the gap since the
    last step of the one before (0 for the first, 7,197 s for each later one, by hand); every step's `status.task` is
    the case's index of its task, never the script's; each step is one frame, and the during question is asked right
    after the decision point's frame (at 7,200 + 1.5 s, delivered as it is, to the tenth); the end
    ask follows the last frame, at the last step; the state is measured at each task's end and at `finish`. Regression
    caught: a multi-task case broken end to end (a task message lost, a gap or an index that carries the script's
    numbers, a task message that falls inside the previous task)."""
    case = long_case(tmp_path)
    run, log = play(case)
    assert run.status == "complete" and run.failures == {} and set(run.answers) == {q["qid"] for q in
                                                                                      [during_q(1, 10), end_q(2)]}
    first_two_tasks = ["start", "task", *["observe"] * 3, "task", "observe", "observe", "ask", "observe"]
    assert ops(log) == [*first_two_tasks, *["task", *["observe"] * 3] * 4, "ask", "finish"]
    assert observed(log) == [[k] for k in range(18)]
    tasks = [entry[1:5] for entry in log if entry[0] == "task"]
    assert [t[0] for t in tasks] == [0, 1, 2, 3, 4, 5]
    assert [t[1] for t in tasks] == [clock(case, 7200 * i) for i in range(6)]
    assert [t[2] for t in tasks] == [0.0] + [GAP_S] * 5
    assert tasks[0][3].startswith("You are the onboard assistant of a") and all(
        "onboard assistant" not in t[3] for t in tasks[1:])   # the paragraph once, then each task's own assignment
    steps = [s for entry in log if entry[0] == "observe" for s in entry[1]]
    assert [s["k"] for s in steps] == list(range(18)) and [s["status"]["task"] for s in steps] == [
        k // 3 for k in range(18)]
    asks = [(entry[1], entry[2]) for entry in log if entry[0] == "ask"]
    assert asks == [("during", {"k": 4, "clock": clock(case, 7201.5)}),
                    ("end", {"k": 17, "clock": clock(case, 5 * 7200 + 3)})]
    assert set(run.state_bytes) == {"0", "1", "2", "3", "4", "5", "finish"}
