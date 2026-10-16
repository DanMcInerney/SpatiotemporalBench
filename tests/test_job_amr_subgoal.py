"""`amr/1`'s subgoal (`job_amr_subgoal`, 0.5.0-control): the text of what the robot is doing, per keyframe. The texts
are read by hand on timelines built for the purpose (a plan and a run of phases, decisions and responses over
`warehouse_a`'s names: the shelving units F11 and F12 are "shelving unit 1" and "shelving unit 2", the packing tables
F09 and F10 "packing table 1" and "packing table 2", the aisle R2 the "west aisle"); the causal and leak contracts on
the scripts of `test_job_amr` (`amr_kit.generated`, one xdist group of its own)."""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

import amr_kit
from stbench.job_amr_subgoal import subgoals
from stbench.job_registry import get
from stbench.leak import RULES
from stbench.timeline_ego import DecisionPoint, Ego, Task, TaskPlan

pytestmark = pytest.mark.xdist_group(name="test_job_amr_subgoal")

M, JOB = amr_kit.M, get("amr/1")


@pytest.fixture(scope="module")
def scripts():
    return {seed: amr_kit.generated(seed, 3) for seed in amr_kit.SEEDS}


def timeline_of(plan: TaskPlan, rows: list):
    """A timeline of one task: `plan` over a keyframe for each row, a phase or (phase, decision kind, response)."""
    base = amr_kit.generated(101, 1)
    kfs, decisions = [], []
    for k, row in enumerate(rows):
        phase, kind, response = (row, None, None) if isinstance(row, str) else row
        kfs.append(replace(base.ego.keyframes[0], k=k, t=float(k), phase=phase, task=0))
        if kind:
            decisions.append(DecisionPoint(k, kind, response))
    return replace(base, ego=Ego(kfs, [Task(0, 0, len(kfs), plan, ())], decisions))


def test_a_replenish_reads_by_stop_and_by_what_the_robot_made_of_each():
    """Contract (0.5.0-control): the texts of a replenish at the shelving unit F11 and the packing tables F09 and F10,
    the first docked, the second skipped and the third docked, by hand: leaving the charger (waiting, undocking); a
    transit "driving to shelving unit 1 to pick up totes (stop 1 of 3)", whose last keyframe carries the
    `before_station`; "docking at", "loading at", "reversing out of" the unit; "driving to packing table 1 to drop off
    totes (stop 2 of 3)" up to and including that decision, then the skip: "waiting at packing table 1: its bay is
    blocked" and a transit that begins "skipping packing table 1; " before "driving to packing table 2 to drop off
    totes (stop 3 of 3)"; docked at table 2 ("unloading"), then "returning to the charger" (transit and approach) and
    "charging". Regression caught: a text that names a station by its id, a skipped visit that reads as docked, a stop
    count that counts skipped stops out, or a closing hold that reads as a wait at a blocked bay."""
    plan = TaskPlan("replenish", {"stations": ["F11", "F09", "F10"], "regions": ["dock"]})
    rows = ["waiting", "waiting", "undocking", "transit", ("transit", "before_station", "go"), "approach", "at_station",
            ("at_station", "before_undock", None), "undocking", "transit", ("transit", "before_station", "skip"),
            "waiting", "waiting", "transit", ("transit", "before_station", "go"), "approach", "at_station",
            ("at_station", "before_undock", None), "undocking", "transit", "approach", "waiting"]
    shelf, one, two = "shelving unit 1", "packing table 1", "packing table 2"
    leg1 = f"driving to {shelf} to pick up totes (stop 1 of 3)"
    leg2 = f"driving to {one} to drop off totes (stop 2 of 3)"
    leg3 = f"driving to {two} to drop off totes (stop 3 of 3)"
    assert subgoals(timeline_of(plan, rows), M) == [
        "leaving the charger", "leaving the charger", "leaving the charger", leg1, leg1, f"docking at {shelf}",
        f"loading at {shelf}", f"loading at {shelf}", f"reversing out of {shelf}", leg2, leg2,
        f"waiting at {one}: its bay is blocked", f"waiting at {one}: its bay is blocked", f"skipping {one}; {leg3}",
        f"skipping {one}; {leg3}", f"docking at {two}", f"unloading at {two}", f"unloading at {two}",
        f"reversing out of {two}", "returning to the charger", "returning to the charger", "charging"]


def test_a_pick_run_reads_by_pick_and_a_collect_by_what_it_returns():
    """Contract (0.5.0-control): a pick of two faces of the west aisle (R2) and a packing table that is skipped reads,
    by hand, "driving into the west aisle to pick (pick 1 of 2)" and "waiting at the pick face while it is loaded
    (pick 1 of 2)", the same for the second, "driving to packing table 2 to drop off totes (stop 1 of 1)" through the
    crossing out and the decision, the wait at the blocked bay, and "skipping packing table 2; returning to the
    charger"; a collect of F10, F09 and the shelving unit F11 loads at the tables ("collect empty totes") and unloads
    at the unit ("return the empty totes"). Regression caught: a pick face that reads as a station, a stop count that
    includes the faces, or a collect that reads as a replenish."""
    pick = TaskPlan("pick", {"stations": ["F10"], "aisle": "R2", "faces": [["F13", "S1"], ["F13", "S2"]],
                             "regions": ["aisle", "dock"]})
    rows = ["waiting", "undocking", ("transit", "before_crossing", None), "at_station", "at_station", "transit",
            "at_station", "transit", ("transit", "before_crossing", None), ("transit", "before_station", "skip"),
            "waiting", "transit", "approach", "waiting"]
    aisle1 = "driving into the west aisle to pick (pick 1 of 2)"
    aisle2 = "driving into the west aisle to pick (pick 2 of 2)"
    face1 = "waiting at the pick face while it is loaded (pick 1 of 2)"
    face2 = "waiting at the pick face while it is loaded (pick 2 of 2)"
    out = "driving to packing table 2 to drop off totes (stop 1 of 1)"
    assert subgoals(timeline_of(pick, rows), M) == [
        "leaving the charger", "leaving the charger", aisle1, face1, face1, aisle2, face2, out, out, out,
        "waiting at packing table 2: its bay is blocked", "skipping packing table 2; returning to the charger",
        "returning to the charger", "charging"]
    collect = TaskPlan("collect", {"stations": ["F10", "F09", "F11"], "regions": ["dock"]})
    dock = ["transit", ("transit", "before_station", "go"), "approach", "at_station",
            ("at_station", "before_undock", None), "undocking"]
    rows = ["waiting", "undocking", *dock, *dock, *dock, "transit", "approach", "waiting"]
    texts = subgoals(timeline_of(collect, rows), M)
    assert texts[2:4] == ["driving to packing table 2 to collect empty totes (stop 1 of 3)"] * 2
    assert texts[5:7] == ["loading at packing table 2"] * 2
    assert texts[8] == "driving to packing table 1 to collect empty totes (stop 2 of 3)"
    assert texts[11] == "loading at packing table 1"
    assert texts[14] == "driving to shelving unit 1 to return the empty totes (stop 3 of 3)"
    assert texts[17:19] == ["unloading at shelving unit 1"] * 2 and texts[-1] == "charging"


def test_a_plan_that_lists_no_stations_is_read_with_generic_words():
    """Contract (the integration of 0.5.0-control: `episode.pose_rows` asks every registered job for its subgoals, so a
    hand-built timeline of `amr/1`, as the test kits make, must be given texts whatever its plan holds): a plan with no
    `stations`, no `faces` and no `aisle` (a kit's `{"room": ...}`), over a run with two decisions, reads one non-empty
    text a keyframe, by hand: leaving the charger, "driving to the next stop", "docking at the station", "working at the
    station", "reversing out of the station", "waiting at the station: its bay is blocked" and the skip that follows,
    then "returning to the charger" and "charging". A pick plan with a face list and no aisle names "the aisle".
    Regression caught: a KeyError on `params["stations"]` that fails every pose file written for a hand-built AMR
    timeline, or a station index past the plan's list."""
    rows = ["waiting", "undocking", "transit", ("transit", "before_station", "go"), "approach", "at_station",
            ("at_station", "before_undock", None), "undocking", "transit", ("transit", "before_station", "skip"),
            "waiting", "transit", "approach", "waiting"]
    texts = subgoals(timeline_of(TaskPlan("replenish", {"room": "dock"}), rows), M)
    go = "driving to the next stop"
    assert texts == ["leaving the charger", "leaving the charger", go, go, "docking at the station",
                     "working at the station", "working at the station", "reversing out of the station", go, go,
                     "waiting at the station: its bay is blocked", f"skipping the station; {go}",
                     "returning to the charger", "charging"]
    pick = TaskPlan("pick", {"faces": [["F13", "S1"]]})
    assert subgoals(timeline_of(pick, ["waiting", "undocking", "transit", "at_station", "waiting"]), M)[2] == (
        "driving into the aisle to pick (pick 1 of 1)")


def test_the_subgoal_is_causal(scripts):
    """Contract (FD-0.5.0-control-impl §2, the key test): `subgoals(tl)[k]` does not change when the timeline is cut
    after k, nor when the response of every decision at or after k is taken away; and a coin decided at k falling the
    other way (the same script with visit j of task i blocked or clear) changes no text up to and including the
    decision, then the texts after it tell what the robot did ("waiting at ...: its bay is blocked" against "docking
    at ..."). Cuts are made at every decision, the keyframe before and the one after it, in all three scripts.
    Regression caught: a subgoal that reads a response or a phase from a keyframe after its own (the `path_log`
    shortcut gate would then read the coin from the text)."""
    for seed, tl in scripts.items():
        full, ks = subgoals(tl, M), tl.ego.keyframes
        assert len(full) == len(ks) and all(full)
        for d in tl.ego.decisions:
            for k in (d.k - 1, d.k, d.k + 1):
                tasks = [Task(t.index, t.k0, min(t.k1, k + 1), t.plan, t.targets) for t in tl.ego.tasks if t.k0 <= k]
                cut = replace(tl, ego=Ego(ks[:k + 1], tasks, [x for x in tl.ego.decisions if x.k <= k]))
                assert subgoals(cut, M) == full[:k + 1], (seed, d.k, k)
                blind = [replace(x, response=None) if x.k >= k else x for x in tl.ego.decisions]
                assert subgoals(replace(tl, ego=Ego(ks, tl.ego.tasks, blind)), M)[:k + 1] == full[:k + 1]
    for i, j in ((0, 0), (1, 1)):
        blocked, clear = amr_kit.forced(101, 3, i, j, True, rear=False), amr_kit.forced(101, 3, i, j, False, rear=False)
        k = amr_kit.asked(blocked, i, j).k
        assert k == amr_kit.asked(clear, i, j).k
        a, b = subgoals(blocked, M), subgoals(clear, M)
        assert a[:k + 1] == b[:k + 1]
        assert a[k + 1].startswith("waiting at ") and a[k + 1].endswith(": its bay is blocked")
        assert b[k + 1].startswith("docking at ")


def test_the_subgoal_names_the_robots_places_and_nothing_else(scripts):
    """Contract (FD-0.5.0-control: the subgoal is the robot's own knowledge; `leak.py`): no text of any script carries
    an entity id, a region id, a map id of a piece or a region (F09, R2, D1), the word "truth" or the map's name; the
    stations are named by the map's furniture names with an ordinal (shelving unit 1 and 2, packing table 1 and 2),
    never "bay" or "drum" (what is in the bay is what the robot is asked); and the texts repeat across scripts (a pool
    for the `path_log` key to form). Regression caught: an id or the world's state in a text the agent reads."""
    seen: dict[int, set[str]] = {}
    for seed, tl in scripts.items():
        texts = set(subgoals(tl, M))
        seen[seed] = texts
        for text in texts:
            assert not any(rx.search(text) for rx in RULES.values()), text
            assert not re.search(r"(?<![A-Za-z0-9])[FRDO]\d+(?![A-Za-z0-9])", text) and "warehouse" not in text.lower()
            assert "drum" not in text and "pallet" not in text
            assert not re.search(r"(?:shelving unit|packing table)(?! \d)", text)  # the ordinal always stands
    assert len(seen[101] & seen[102]) >= 5
