"""`vacuum/2`'s subgoal (`job_vacuum2_subgoal`): the text of what the robot is doing at each keyframe. Hand-made egos
for the wording, seeds 101-104 for the scripts (`job_kit.generated2`, with the coins forced where a contract needs
them)."""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

import job_kit
from stbench.job_vacuum2_plan import eligible
from stbench.job_vacuum2_subgoal import subgoals
from stbench.mapspec import load_map
from stbench.timeline_ego import DecisionPoint, Ego, Keyframe, Task, TaskPlan

pytestmark = pytest.mark.xdist_group(name="test_job_vacuum2")  # shares the worker, and the cache, of the job tests

M = load_map("apartment_a")
SEEDS = (101, 102, 103, 104)
TEXT = re.compile(
    r"^(leaving the dock to (clean|finish cleaning) the [a-z ,]+"
    r"|driving to the [a-z ]+ to (clean|finish cleaning) it( \(room [12] of 2\))?"
    r"|cleaning the [a-z ]+( \(room [12] of 2\))?"
    r"|finishing the [a-z ]+"
    r"|leaving the [a-z ]+ uncleaned: its doorway is blocked"
    r"|returning to the dock|docking|docked)$")


def all_blocked(_m, _seed, _i, _j) -> bool:
    return True


def all_clear(_m, _seed, _i, _j) -> bool:
    return False


def hand(plan: TaskPlan, phases: list[str], decisions: dict[int, tuple[str, str | None]]):
    """A one-task script with these phases a keyframe and these decisions at keyframes ({k: (kind, response)}), over a
    real timeline's world (the subgoal reads the ego alone)."""
    _, donor = job_kit.generated2(102)
    kfs = [Keyframe(k, float(k), (0.0, 0.0, 0.1), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 0.0, 0.0, 0, phase, {})
           for k, phase in enumerate(phases)]
    ego = Ego(kfs, [Task(0, 0, len(kfs), plan, ())], [DecisionPoint(k, *d) for k, d in sorted(decisions.items())])
    return replace(donor, ego=ego)


KITCHEN_HALL = TaskPlan("clean_rooms", {"rooms": "kitchen, then the hall", "kinds": ["kitchen", "hall"]})
PHASES = ["docked", "undocking", "transit", "transit", "transit", "cleaning", "cleaning", "transit", "transit",
          "cleaning", "cleaning", "transit", "docking", "docked"]


def test_the_words_of_each_moment_by_hand():
    """Contract (FD-0.5.0-control-impl §2): the subgoal names the rooms as the brief does and where the robot is in
    them. Hand cases: a kitchen-then-hall task that goes through the kitchen's doorway reads, keyframe by keyframe,
    "leaving the dock to clean the kitchen, then the hall" (docked, undocking), "driving to the kitchen to clean it
    (room 1 of 2)" (transit, the decision included), "cleaning the kitchen (room 1 of 2)", "driving to the hall to
    clean it (room 2 of 2)", "cleaning the hall (room 2 of 2)", "returning to the dock", "docking", "docked"; the
    same task with the kitchen skipped reads "leaving the kitchen uncleaned: its doorway is blocked" from the
    keyframe after the decision to the end of that transit, then cleans the hall as room 2 of 2; a resumed bedroom
    reads "leaving the dock to finish cleaning the bedroom", "driving to the bedroom to finish cleaning it" (no
    count for one room), "finishing the bedroom", "returning to the dock". Regression caught: a wording that gives
    the plan's step away (the stop's share, a region id) or counts a skipped room as cleaned."""
    went = hand(KITCHEN_HALL, PHASES, {3: ("before_room", "go"), 6: ("room_done", None), 10: ("room_done", None)})
    assert subgoals(went, M) == [
        *["leaving the dock to clean the kitchen, then the hall"] * 2,
        *["driving to the kitchen to clean it (room 1 of 2)"] * 3,
        *["cleaning the kitchen (room 1 of 2)"] * 2,
        *["driving to the hall to clean it (room 2 of 2)"] * 2,
        *["cleaning the hall (room 2 of 2)"] * 2,
        "returning to the dock", "docking", "docked"]
    skip = ["docked", "undocking", "transit", "transit", "transit", "transit", "cleaning", "cleaning", "transit",
            "docking", "docked"]
    skipped = hand(KITCHEN_HALL, skip, {3: ("before_room", "skip"), 7: ("room_done", None)})
    assert subgoals(skipped, M) == [
        *["leaving the dock to clean the kitchen, then the hall"] * 2,
        *["driving to the kitchen to clean it (room 1 of 2)"] * 2,
        *["leaving the kitchen uncleaned: its doorway is blocked"] * 2,
        *["cleaning the hall (room 2 of 2)"] * 2,
        "returning to the dock", "docking", "docked"]
    resume = TaskPlan("resume", {"rooms": "bedroom", "kinds": ["bedroom"]})
    phases = ["docked", "undocking", "transit", "cleaning", "cleaning", "transit", "docking", "docked"]
    assert subgoals(hand(resume, phases, {2: ("before_doorway", None), 4: ("room_done", None)}), M) == [
        *["leaving the dock to finish cleaning the bedroom"] * 2, "driving to the bedroom to finish cleaning it",
        *["finishing the bedroom"] * 2, "returning to the dock", "docking", "docked"]


def test_a_skip_text_lasts_until_the_next_doorway_decision():
    """Contract: after a skip the text stays "leaving the ... uncleaned" through the rest of the transit up to the next
    doorway decision, where it becomes the next room's "driving to ..." (the same words the clear case reads at the
    decision), so a robot that has skipped one room and drives to another reads both. Hand case: bathroom then
    kitchen, the bathroom skipped at keyframe 3, the kitchen's `before_room` at keyframe 7. Regression caught: a skip
    text that outlives the leg, or a next room announced a keyframe late."""
    plan = TaskPlan("clean_rooms", {"rooms": "bathroom, then the kitchen", "kinds": ["bathroom", "kitchen"]})
    phases = ["docked", "undocking", "transit", "transit", "transit", "transit", "transit", "transit", "transit",
              "cleaning", "cleaning", "transit", "docking", "docked"]
    tl = hand(plan, phases, {3: ("before_room", "skip"), 7: ("before_room", "go"), 10: ("room_done", None)})
    got = subgoals(tl, M)
    assert got[3] == got[2] == "driving to the bathroom to clean it (room 1 of 2)"
    assert got[4:7] == ["leaving the bathroom uncleaned: its doorway is blocked"] * 3
    assert got[7] == got[8] == "driving to the kitchen to clean it (room 2 of 2)"
    assert got[9] == "cleaning the kitchen (room 2 of 2)"


@pytest.mark.parametrize("coins", [None, all_blocked, all_clear])
def test_every_text_is_in_the_closed_vocabulary_and_leaks_nothing(coins):
    """Contract: over the scripts of seeds 101-104, one text a keyframe, each of the closed forms (the regular
    expression here) and none naming a region id, an entity id or the coin; with blocked coins the skip text appears,
    with clear ones it does not. Regression caught: a text that carries the plan's private params (the stop's share,
    region ids), or one that says "skip" or "coin"."""
    skip_texts = 0
    for seed in SEEDS:
        _, tl = job_kit.generated2(seed, (), coins)
        texts = subgoals(tl, M)
        assert len(texts) == tl.n_keyframes
        for text in set(texts):
            assert TEXT.match(text), text
            assert not re.search(r"\bR\d\b|\bE\d{4}\b|\bD\d\b|\bO\d\b|coin|stop", text), text
        skip_texts += sum("uncleaned" in text for text in texts)
    assert (skip_texts > 0) == (coins is not all_clear)


def test_the_subgoal_is_causal():
    """Contract (FD-0.5.0-control-impl §2): the subgoal of keyframe k does not change when the timeline is cut after
    k, nor when any coin decided at or after k falls the other way. Seeds 101 and 102 with their own coins: for every
    7th keyframe k, the subgoals of the script cut after k (the tasks clipped, the decisions after k gone) agree with
    the whole's up to k; and flipping the response of a `before_room` decision from skip to go (the rest untouched)
    leaves every subgoal up to and including its keyframe alone and changes the next. Seed 103 with the
    coin of its first `before_room` forced each way (a real script each: the skip leaves the room): equal up to the
    decision, the skip text after it only when blocked. Regression caught: a text that reads the response at its own
    keyframe or later, or a count that includes a room the robot has not reached."""
    for seed in (101, 102):
        _, tl = job_kit.generated2(seed)
        full = subgoals(tl, M)
        kfs, tasks, decisions = tl.ego.keyframes, tl.ego.tasks, tl.ego.decisions
        for k in range(0, len(kfs), 7):  # every 7th keyframe, and the decisions below
            cut = Ego(kfs[:k + 1], [replace(t, k1=min(t.k1, k + 1)) for t in tasks if t.k0 <= k],
                      [d for d in decisions if d.k <= k])
            assert subgoals(replace(tl, ego=cut), M) == full[:k + 1], (seed, k)
        for d in (d for d in decisions if d.kind == "before_room" and d.response == "skip"):
            flipped = [replace(x, response="go") if x is d else x for x in decisions]
            other = subgoals(replace(tl, ego=replace(tl.ego, decisions=flipped)), M)
            assert other[:d.k + 1] == full[:d.k + 1], (seed, d)
            assert other[d.k + 1] != full[d.k + 1], (seed, d)

    def coin(i, j, blocked):
        return lambda _m, _seed, task, room: blocked if (task, room) == (i, j) else False

    _, base = job_kit.generated2(103, (), all_clear)
    first = next(d for d in base.ego.decisions if d.kind == "before_room")
    task = base.task_of(first.k)
    room = next(j for j in range(len(task.targets)) if eligible(M, task.plan, j))
    _, a = job_kit.generated2(103, (), coin(task.index, room, True))
    _, b = job_kit.generated2(103, (), coin(task.index, room, False))
    da = next(d for d in a.ego.decisions if d.kind == "before_room" and a.task_of(d.k).index == task.index)
    ta, tb = subgoals(a, M), subgoals(b, M)
    assert ta[:da.k + 1] == tb[:da.k + 1]
    assert ta[da.k + 1].startswith("leaving the") and "uncleaned" in ta[da.k + 1]
    assert "uncleaned" not in tb[da.k + 1]
