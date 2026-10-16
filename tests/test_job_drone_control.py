"""`drone/1`'s control work (0.5.0-control lane J1): the abort response at `before_lower`, the subgoal and the job's
declared control fields, over `downtown_a`. The scripts are seeds 101-104 as they generate (`drone_kit.generated`) and
seed 101 with every flight's coin forced both ways (`drone_kit.forced`: `job_drone.coin_blocked` patched), built once
for the module. A zone's state is read from the finished timeline's placements (`drone_kit.zone_state`), not from the
code under test. Seed 101's flights are deliver, deliver, inspect, deliver, all over the square."""

from __future__ import annotations

from dataclasses import replace
from unittest import mock

import pytest

import drone_kit
from stbench import job_drone, questions_zone
from stbench.generate import generate_script
from stbench.job_registry import get
from stbench.timeline_ego import Ego
from stbench.validate import validate

pytestmark = pytest.mark.xdist_group(name="test_job_drone_flight")  # one worker builds the scripts once

SEEDS = (101, 102, 103, 104)
M = drone_kit.downtown()
JOB = get("drone/1")
DELIVER, INSPECT = "deliver", "inspect"
SWAP = {"abort": "go", "go": "abort"}
INSPECTION_RUNS = ["preflight", "takeoff", "approach", "inspect", "descend", "inspect", "climb_out", "return", "land"]
DELIVERY_RUNS = ["preflight", "takeoff", "approach", "inspect", "descend", "lower", "release", "climb_out", "return",
                 "land"]


def timelines():
    """Every script the contracts read: the natural ones and seed 101 under both forced coins."""
    return [drone_kit.generated(s) for s in SEEDS] + [drone_kit.forced(101, True), drone_kit.forced(101, False)]


def lower_k(tl, task) -> int:
    """The keyframe of the flight's `before_lower`."""
    return next(d.k for d in tl.decisions_in(task.k0, task.k1) if d.kind == "before_lower")


blocked_at_start = drone_kit.blocked_at_start


def runs(tl, task) -> list[str]:
    """The flight's phases, each run once."""
    phases = [kf.phase for kf in tl.ego.keyframes[task.k0:task.k1]]
    return [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]]


def texts(tl, task) -> list[str]:
    """The flight's subgoal texts, each run once."""
    found = JOB.subgoals(tl, M)[task.k0:task.k1]
    return [t for i, t in enumerate(found) if i == 0 or t != found[i - 1]]


def test_a_delivery_aborts_exactly_when_its_zone_holds_an_item():
    """Contract (FD-0.5.0-control-impl §4, drone/1): the response at a flight's `before_lower` is `abort` exactly when
    the flight is a delivery and its drop zone holds a standing item (taller than 0.3 m over the zone's ground, its
    footprint overlapping the zone) at the flight's first keyframe, read from the finished timeline's placements; it is
    `go` otherwise, an inspection over a blocked zone included; the other two decisions carry none. Over seeds 101-104
    and seed 101 with the coins forced both ways, both responses occur and every script validates. Regression caught:
    a response read from the coin (not the world), an inspection that aborts, a response on a decision that makes none,
    or an abort script the validator refuses."""
    seen, inspect_over_item = set(), 0
    for tl in timelines():
        assert validate(tl, M) == []
        for task in tl.ego.tasks:
            held = blocked_at_start(tl, task)
            for d in tl.decisions_in(task.k0, task.k1):
                want = ("abort" if held and task.plan.kind == DELIVER else "go") if d.kind == "before_lower" else None
                assert d.response == want, (tl.episode.seed, task.index, d.kind)
            seen.add(tl.decisions_in(task.k0, task.k1)[1].response)
            inspect_over_item += held and task.plan.kind == INSPECT
    assert seen == {"abort", "go"} and inspect_over_item >= 1


def test_the_response_follows_the_world_and_not_the_coin():
    """Contract: the response is the world's after the coin was enforced, never the coin itself. With every coin forced
    to blocked but the drops enforcing the opposite of each coin, every zone is clear, no flight aborts, and the script
    is the one the clear coins give, response for response. Regression caught: a response taken from `coin_blocked`
    (the abort would then follow the coin over an empty zone)."""
    real = job_drone.enforce
    n = len(drone_kit.generated(101).ego.tasks)
    with (mock.patch.object(job_drone, "coin_blocked", lambda name, seed, i: True),
          mock.patch.object(job_drone, "enforce",
                            lambda w, m, rng, piece, blocked, t: real(w, m, rng, piece, not blocked, t))):
        tl = generate_script(drone_kit.MAP, 101, "drone", "drone/1", n)
    assert {d.response for d in tl.ego.decisions if d.kind == "before_lower"} == {"go"}
    assert tl.ego == drone_kit.forced(101, False).ego


def test_the_script_up_to_before_lower_is_the_same_under_both_coins():
    """Contract (FD-0.5.0-control-impl §4, "Contracts for every job"): scripted with every coin blocked and with every
    coin clear, the same seed gives a script whose keyframes (poses, times, camera, phases, statuses) and subgoals up to
    and including each flight's `before_lower` are identical, whose decision points lie on the same keyframes, and
    whose every pose and time is the same to the end: only the response, and after it the hover's phases, the tether and
    the text, differ. The zones do differ (blocked and clear at each flight's start). Regression caught: a path, a
    frame time, a status or a subgoal that tells the coin before the question is asked."""
    blocked, clear = drone_kit.forced(101, True), drone_kit.forced(101, False)
    assert len(blocked.ego.keyframes) == len(clear.ego.keyframes) and blocked.ego.tasks == clear.ego.tasks
    assert [(d.k, d.kind) for d in blocked.ego.decisions] == [(d.k, d.kind) for d in clear.ego.decisions]
    sub_b, sub_c = JOB.subgoals(blocked, M), JOB.subgoals(clear, M)
    for task in blocked.ego.tasks:
        assert blocked_at_start(blocked, task) and not blocked_at_start(clear, task)
        k_d = lower_k(blocked, task)
        assert blocked.ego.keyframes[task.k0:k_d + 1] == clear.ego.keyframes[task.k0:k_d + 1]
        assert sub_b[task.k0:k_d + 1] == sub_c[task.k0:k_d + 1]
        for a, b in zip(blocked.ego.keyframes[task.k0:task.k1], clear.ego.keyframes[task.k0:task.k1], strict=True):
            assert (a.t, a.pos, a.rot, a.base, a.yaw, a.cam_pitch, a.task, a.status["battery_pct"]) == (
                b.t, b.pos, b.rot, b.base, b.yaw, b.cam_pitch, b.task, b.status["battery_pct"])
        if task.plan.kind == DELIVER:
            assert sub_b[k_d + 1] != sub_c[k_d + 1]


def test_an_aborted_delivery_runs_the_inspections_phases_with_no_tether():
    """Contract: a delivery aborted at `before_lower` flies the inspection's phases, hovering at the lowering height
    as `inspect` for the time a lowering takes, and pays out no tether at any keyframe; the same flight with a clear
    zone runs the delivery's phases and lowers 7 m of tether. Both fly the same poses (the frame times are the
    inspection's and the delivery's alike). By hand the aborted run is preflight, takeoff, approach, inspect, descend,
    inspect, climb_out, return, land. Regression caught: an abort that still pays out the tether or lowers, or one that
    moves the frames."""
    blocked, clear = drone_kit.forced(101, True), drone_kit.forced(101, False)
    deliveries = [t for t in blocked.ego.tasks if t.plan.kind == DELIVER]
    assert len(deliveries) >= 2
    for task in blocked.ego.tasks:
        assert runs(blocked, task) == INSPECTION_RUNS
        assert {kf.status["tether_m"] for kf in blocked.ego.keyframes[task.k0:task.k1]} == {0.0}
        if task.plan.kind == DELIVER:
            assert runs(clear, task) == DELIVERY_RUNS
            assert max(kf.status["tether_m"] for kf in clear.ego.keyframes[task.k0:task.k1]) == pytest.approx(7.0)
        else:
            assert runs(clear, task) == INSPECTION_RUNS


def test_the_subgoal_texts_by_hand():
    """Contract (FD-0.5.0-control-impl §2, the drone's templates): a flight's subgoal, each text once in a row, is by
    hand: for a delivery that goes on, preparing to deliver, taking off, flying to the drop point, inspecting it from
    above, descending, lowering the package, releasing it, reeling the tether in, climbing away, flying back and
    landing; for one aborted, the same to the descent and then holding over the drop point because it is not clear to
    lower, climbing away and flying back with the package; for an inspection, preparing to inspect and inspecting from
    close above over the hover. Each names the drop point by the plan's address, and nothing names a piece id or a coin.
    Regression caught: a text that skips a phase, names the wrong place, or tells the abort before the hover."""
    def hand(a: str) -> tuple[list[str], list[str], list[str]]:
        common = ["taking off from the nest", f"flying to the drop point at {a}",
                  f"inspecting the drop point at {a} from above", f"descending over the drop point at {a}"]
        go = [f"preparing to deliver a package to the drop point at {a}", *common, f"lowering the package at {a}",
              "releasing the package", "reeling the tether in", "climbing away from the drop point",
              "flying back to the nest", "landing on the nest"]
        abort = [f"preparing to deliver a package to the drop point at {a}", *common,
                 f"holding over the drop point at {a}: it is not clear to lower", "climbing away from the drop point",
                 "flying back to the nest with the package", "landing on the nest"]
        inspect = [f"preparing to inspect the drop point at {a}", *common,
                   f"inspecting the drop point at {a} from close above", "climbing away from the drop point",
                   "flying back to the nest", "landing on the nest"]
        return go, abort, inspect

    blocked, clear = drone_kit.forced(101, True), drone_kit.forced(101, False)
    assert [t.plan.kind for t in blocked.ego.tasks] == [DELIVER, DELIVER, INSPECT, DELIVER]
    for task in blocked.ego.tasks:
        go, abort, inspect = hand(task.plan.params["address"])
        if task.plan.kind == DELIVER:
            assert texts(clear, task) == go and texts(blocked, task) == abort
        else:
            assert texts(clear, task) == texts(blocked, task) == inspect
    assert blocked.ego.tasks[0].plan.params["address"] == "Square West"
    for tl in (blocked, clear):
        for text in JOB.subgoals(tl, M):
            assert not any(word in text for word in ("F2", "F3", "zone_", "coin", "blocked", "abort"))


def test_a_plan_with_no_address_gets_the_same_texts_without_it():
    """Contract (the integration of 0.5.0-control: `episode.pose_rows` asks every registered job for its subgoals, so a
    hand-built timeline of `drone/1` whose plan states no address must still be given texts): a flight whose plan is
    stripped of its params reads as the same flight with " at <address>" taken out of each text, one non-empty text a
    keyframe, and no "None". Regression caught: a KeyError on `params["address"]` that fails every pose file written
    for a hand-built drone timeline, or a text that prints the missing address."""
    tl = drone_kit.forced(101, True)
    tasks = [replace(t, plan=replace(t.plan, params={})) for t in tl.ego.tasks]
    bare = JOB.subgoals(replace(tl, ego=replace(tl.ego, tasks=tasks)), M)
    with_address = JOB.subgoals(tl, M)
    expected = [text.replace(f" at {t.plan.params['address']}", "")
                for t in tl.ego.tasks for text in with_address[t.k0:t.k1]]
    assert bare == expected and all(bare) and not any("None" in text for text in bare)


def cut(tl, k):
    """The timeline cut after keyframe `k`: its keyframes to `k`, the tasks and the decisions that reach them."""
    ego = tl.ego
    tasks = [replace(t, k1=min(t.k1, k + 1)) for t in ego.tasks if t.k0 <= k]
    return replace(tl, ego=Ego(ego.keyframes[:k + 1], tasks, [d for d in ego.decisions if d.k <= k]))


def flipped_from(tl, k):
    """The timeline with the response of every decision at keyframe `k` or later turned the other way."""
    ds = [replace(d, response=SWAP[d.response]) if d.response is not None and d.k >= k else d
          for d in tl.ego.decisions]
    return replace(tl, ego=replace(tl.ego, decisions=ds))


def test_the_subgoal_is_causal():
    """Contract (FD-0.5.0-control-impl §2, the causality contract): the text at keyframe k does not change when the
    timeline is cut after k, nor when the response of any decision at or after k is turned the other way; the hover
    after an abort does read it (turned to a go, its text changes), so the test sees the flip. Read over every
    keyframe of seed 101 under both forced coins (aborts and goes both) and of seed 102 as it generates. Regression
    caught: a text that reads the future (a later phase, the length of the flight) or the response of the decision at
    its own keyframe, which would tell the coin to the agent before it answers."""
    seen_flip = 0
    for tl in (drone_kit.forced(101, True), drone_kit.forced(101, False), drone_kit.generated(102)):
        full = JOB.subgoals(tl, M)
        assert len(full) == tl.n_keyframes
        for k in range(tl.n_keyframes):
            assert JOB.subgoals(cut(tl, k), M)[k] == full[k], k
            assert JOB.subgoals(flipped_from(tl, k), M)[k] == full[k], k
        for task in tl.ego.tasks:
            k_d = lower_k(tl, task)
            if next(d for d in tl.ego.decisions if d.k == k_d).response == "abort":  # the hover shows the abort
                assert JOB.subgoals(flipped_from(tl, k_d), M)[k_d + 1] != full[k_d + 1]
                seen_flip += 1
    assert seen_flip >= 3


def test_the_job_declares_its_control_fields():
    """Contract (FD-0.5.0-control-impl §4, §6, §7): `drone/1` has a zone for `before_lower` alone, the drop zone (which
    families it asks, none during the job since 0.6.0, is `test_jobs_ask_060`'s); its ladder is 1, 2, 4 and 8 m; each
    plan's relevance carries the drop zone's centre (`F26` Square West (40, -6) and the like, from the map) as its near
    point with a radius of 12 m, and the plan states that centre; the job's zone rules pass the compiler's consistency
    check. Regression caught: a coin with no rule, a relevance with no near point (every thing of the district
    critical), or a centre that is not the zone's."""
    assert set(JOB.decisions) == {"before_descent", "before_lower", "before_return"}
    assert set(JOB.zones) == {"before_lower"} and JOB.zones["before_lower"].words == "the drop zone"
    assert JOB.zones["before_lower"].piece == "drop_zone"
    assert JOB.ladder == (1.0, 2.0, 4.0, 8.0)
    questions_zone.check_rules(JOB)
    centres = {"F26": (40.0, -6.0), "F27": (60.0, -6.0), "F28": (60.0, -18.0), "F29": (20.3, 44.1),
               "F30": (79.7, 44.1)}
    seen = set()
    for seed in range(1, 41):
        for plan in JOB.plan(M, seed, 5):
            zone = plan.params["zone"]
            rel = JOB.relevance(plan)
            assert (plan.params["zone_x"], plan.params["zone_y"]) == centres[zone]
            assert rel.near == (centres[zone],) and rel.near_m == 12.0 and not rel.floor
            assert rel.regions == {plan.params["region"], plan.params["nest_region"]}
            seen.add(zone)
    assert seen == set(centres)
    assert get("drone/1").subgoals is not None


@pytest.mark.parametrize(("edit", "code"), [
    ("an abort over a clear zone", "job_response"),
    ("a go over a zone that holds an item", "job_response"),
    ("a response at before_descent", "job_response"),
    ("no response at before_lower", "job_response"),
    ("an aborted delivery that lowers", "job_phase_order"),
    ("an aborted delivery that pays out tether", "job_tether"),
    ("a delivery over a clear zone that does not lower", "job_phase_order"),
    ("a plan whose zone centre is off", "job_targets"),
])
def test_the_response_checks_have_teeth(edit, code):
    """Contract: `job_drone`'s checks tie the response and the flight's phases to the world: they flag an abort over a
    clear zone, a go over a zone that holds an item (an inspection's go over one is fine), a response on a decision
    that makes none, a missing response, a delivery over a held zone that runs the lowering's phases or pays out tether,
    a delivery over a clear zone that only hovers, and a plan whose stated zone centre is not the zone's. The unedited
    scripts pass. Regression caught: a response rule no script can break, or a phase rule that follows the stored
    response instead of the world."""
    blocked, clear = drone_kit.forced(101, True), drone_kit.forced(101, False)
    assert JOB.checks(blocked, M) == [] and JOB.checks(clear, M) == []
    first = blocked.ego.tasks[0]
    k_d = lower_k(blocked, first)
    descent = next(d.k for d in blocked.ego.decisions if d.kind == "before_descent" and d.k >= first.k0)

    def with_decision(tl, k, response):
        ds = [replace(d, response=response) if d.k == k else d for d in tl.ego.decisions]
        return replace(tl, ego=replace(tl.ego, decisions=ds))

    def with_flight_of(tl, other):
        kfs = list(tl.ego.keyframes)
        kfs[first.k0:first.k1] = other.ego.keyframes[first.k0:first.k1]
        return replace(tl, ego=replace(tl.ego, keyframes=kfs))

    if edit == "an abort over a clear zone":
        broken = with_decision(clear, lower_k(clear, first), "abort")
    elif edit == "a go over a zone that holds an item":
        broken = with_decision(blocked, k_d, "go")
    elif edit == "a response at before_descent":
        broken = with_decision(clear, descent, "go")
    elif edit == "no response at before_lower":
        broken = with_decision(clear, lower_k(clear, first), None)
    elif edit == "an aborted delivery that lowers":
        broken = with_flight_of(blocked, clear)
    elif edit == "an aborted delivery that pays out tether":
        kfs = list(blocked.ego.keyframes)
        hover = next(kf for kf in kfs[first.k0:first.k1] if kf.k > k_d)
        kfs[hover.k] = replace(hover, status={**hover.status, "tether_m": 3.0})
        broken = replace(blocked, ego=replace(blocked.ego, keyframes=kfs))
    elif edit == "a delivery over a clear zone that does not lower":
        broken = with_flight_of(clear, blocked)
    else:
        tasks = list(blocked.ego.tasks)
        tasks[0] = replace(first, plan=replace(first.plan, params={**first.plan.params, "zone_x": 41.0}))
        broken = replace(blocked, ego=replace(blocked.ego, tasks=tasks))
    assert code in {p.code for p in JOB.checks(broken, M)}
