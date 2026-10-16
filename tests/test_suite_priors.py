"""The priors and the options (`suite_priors`, plan §4.8): the records other scripts' questions give, the pool a prior
system finds in them, the case `other_case` is fed and where it is asked, and each system's `RunOptions` from its flags.
The keys are checked on compiled synthetic cases through the real stream; the rest on hand-made cases, whose scripts are
20-step tasks with decision points at steps 3 and 12 of each task: the base stands at the room's centre, (2, 1.5) in the
kitchen and (6, 1.5) in the living room, and heads 45 degrees times (step in task mod 8): 180 at step 12, 135 at step
19."""

from __future__ import annotations

import json
import pickle
from dataclasses import replace
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest import mock

import pytest

import zone_kit as zk
from baseline_kit import EVENTS, play, question, rows, started, step
from check_kit import asked_records, hand_case, hand_synth, synthetic_dataset
from run_kit import scored_case
from stbench import answer_keys, baselines, job_registry, mapspec, suite_priors, table_answers
from stbench import case as case_mod
from stbench.baselines import AllUnknown, Baseline, CurrentView, MemoryTable
from stbench.seeding import stable_hash
from stbench.stream import CaseRun
from stbench.stream_protocol import System
from stbench.suite import RunOptions
from stbench.task_predicate import Anchor, TaskThings
from stream_kit import case_on, clock, during_q, end_q, gold_row, qid
from synth import HOME, WAREHOUSE, Synth

NM = baselines.NM_SYSTEM
BD = "before_doorway"
KEYED = ("blind", "path_log", "current_view_prior")   # the systems `answer_keys.levels` keys


def number(v: float) -> dict:
    return {"type": "number", "value": v}


def at(x: float, y: float) -> dict:
    return {"type": "position", "xy": [x, y]}


def run_of(answers: dict) -> CaseRun:
    return CaseRun(answers, "complete", {}, {}, {}, None, "0" * 16)


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    """Two compiled cases of the synthetic script (`space_kit.things_run`): a decision after step 21, in `transit`."""
    return synthetic_dataset(tmp_path_factory.mktemp("compiled"), 2)


def test_prior_keys_agree(compiled):
    """Contract (plan §4.8, map SG-H23): at every ask of a case on the gate channel, the key `answer_keys.levels` gives
    the record `suite_priors.record_for` writes of the question equals the key it gives the record the prior system
    itself builds there (`ask_record`, E1's hook), for each system that keys on records, with the `nm` the first wave's
    `current_view` gave. Hand anchor: the during question (asked after step 21, the robot's front edge at x = 0, heading
    +x, in `transit`, `current_view` answering 2.2) keys `current_view_prior` on ("path_ahead", "transit", "-1,0,0",
    ">=2"). The record's `clock` is also the one the system builds at the same ask (`now["clock"]` for `blind`, the
    step's for the others): a time gold is learned as a lag behind it. Regression caught: writing records the systems
    do not look up (a different rounding of a pose, a missing phase, rooms from other words), so a prior's pool is
    empty and its shortcut gate passes vacuously; or a clock a step off the one the stream delivers (a shifted `now` or
    `ask_step`), which would offset every lag silently."""
    for case in compiled:
        qs, gs = rows(case)
        nm = play(case, CurrentView)[0].answers
        built = {}
        for name in KEYED:
            b = baselines.BASELINES[name]
            built[name] = asked_records(case, b.cls, **suite_priors.options_for(name, case, [case], {}).opts)
            assert set(built[name]) == {q["qid"] for q in qs}, name
            for q, g in zip(qs, gs, strict=True):
                rec = suite_priors.record_for(case, q, g, nm[q["qid"]] if b.prior_nm else None)
                assert answer_keys.levels(name, rec) == answer_keys.levels(name, built[name][q["qid"]]), (name, q)
                assert rec["clock"] == built[name][q["qid"]]["clock"], (name, q)
        during = next(g["qid"] for g in gs if g["scope"] == "now")
        first = answer_keys.levels("current_view_prior", built["current_view_prior"][during])[0]
        assert first == ("path_ahead", "transit", "-1,0,0", ">=2")


def test_a_record_is_what_the_systems_are_sent(tmp_path):
    """Contract (plan §4.8): `record_for` gives a question's template, slots and `when`; the clock, phase, base and
    heading of the step the ask is about (a during question's decision step, an end question's last step of the window),
    read from the script's pose rows in the delivered numbering, the clock as the stream delivers it (the case's start
    clock and the seconds into the window, 1.5 a step and 7200 a task here: 18 s at step 12 of the first task, 7218 at
    step 12 of the second, 7229 at its last step, 28.5 s in, rounded half up); the rooms the briefs sent so far name (a
    during ask in the first of two tasks has been sent one brief, the end ask both); `nm`; the other end questions as
    `{template, slots}` (none for a during question); no anchor (that is a `task_things` record's) and no own-log
    answer (a `clear_path` record's); the camera pose of every step of the window up to the ask's, the ask's last (13
    for the first during ask at step 12 of the script, 33 for the second at step 32, all 40 steps for an end ask); the
    gold. The window here is two tasks, kitchen then living room, and the second during question is asked after step 32
    of the script (step 12 of its task). Regression caught: the rooms of tasks not yet begun named in a during ask's
    key, the end step read at the window's start or the script's end, a sibling list that holds the question itself, or
    cameras of steps after the ask (the ego rule would then judge a record on views it had not had)."""
    synth = Synth(tasks=[[(HOME.rooms[0], 20)], [(HOME.rooms[1], 20)]],
                  decisions=[(3, BD), (12, BD), (23, BD), (32, BD)])
    qs = [during_q(1, 12), during_q(2, 32), end_q(3, slots={"kind": "mug", "attributes": {"color": "red"}}),
          end_q(4, slots={"kind": "book", "attributes": {"color": "blue"}})]
    case = case_on(tmp_path, synth, qs, a=0, b=1, length="medium")
    gold = {q["qid"]: gold_row(q) for q in qs}
    first, second, red, blue = (suite_priors.record_for(case, q, gold[q["qid"]], None) for q in qs)
    common = {"script": case.script.id, "nm": None, "phase": "docked", "subgoal": None, "anchor": None, "own": None}
    cams = {r["gold"]["qid"]: r.pop("cameras") for r in (first, second, red, blue)}
    assert [len(cams[q["qid"]]) for q in qs] == [13, 33, 40, 40]
    assert cams[qs[1]["qid"]][-1] == suite_priors.ask_step(case, qs[1]).row["camera"]
    assert first == {**common, "template": "path_ahead", "slots": {"cap": 2}, "when": "during", "base": (2.0, 1.5),
                     "clock": clock(case, 18), "yaw": 180.0, "rooms": ("kitchen",), "siblings": [],
                     "gold": gold[qs[0]["qid"]]}
    assert second == {**first, "base": (6.0, 1.5), "clock": clock(case, 7218), "rooms": ("kitchen", "living room"),
                      "gold": gold[qs[1]["qid"]]}
    assert red == {**common, "template": "object_where", "slots": qs[2]["slots"], "when": "end", "base": (6.0, 1.5),
                   "clock": clock(case, 7228.5), "yaw": 135.0, "rooms": ("kitchen", "living room"),
                   "gold": gold[qs[2]["qid"]],
                   "siblings": [{"template": "object_where", "slots": qs[3]["slots"]}]}
    assert blue["siblings"] == [{"template": "object_where", "slots": qs[2]["slots"]}]
    assert suite_priors.record_for(case, qs[1], gold[qs[1]["qid"]], number(1.4))["nm"] == number(1.4)
    step = suite_priors.ask_step(case, qs[1])
    assert (step.task, step.k_in_task, step.row["k"], step.clock) == (1, 12, 32, clock(case, 7218))


def test_a_task_things_record_carries_the_anchor_of_its_task(tmp_path):
    """Contract (0.6.0 repair, `answer_keys.RECORD_KEYS`): the record of a `task_things` question holds the anchor of
    the task it names, built by `task_predicate.anchor_of` from that task's pose rows as delivered and the job's
    predicate, so that `path_log`'s recurring-spots rule can be scored on a record as it is on an ask; the record of any
    other question holds none, and neither does one of a job that declares no predicate. Hand values: the window is two
    tasks, the second in the living room, the base standing at (6, 1.5) for its 20 steps (all `docked`), so a `point`
    anchor over `docked` is that one point for task 1 and the kitchen's (2, 1.5) for task 0. Regression caught: the
    anchor built from the wrong task's steps (all of them, say: the median of (2, 1.5) and (6, 1.5) is (4, 1.5)), or a
    record that carries one for a question that is not `task_things`."""
    pred = TaskThings("within 2 m of where you docked", "point", frozenset({"docked"}), reach_m=2.0)
    synth = Synth(tasks=[[(HOME.rooms[0], 20)], [(HOME.rooms[1], 20)]], decisions=[(12, BD)])
    qs = [end_q(1, template="task_things", slots={"phrase": pred.phrase, "task": 1}, answer_type="positions"),
          end_q(2, template="task_things", slots={"phrase": pred.phrase, "task": 0}, answer_type="positions"),
          end_q(3)]
    case = case_on(tmp_path, synth, qs, a=0, b=1, length="medium")
    gold = {q["qid"]: gold_row(q) for q in qs}
    declared = job_registry.get
    with mock.patch.object(job_registry, "get", lambda name: replace(declared(name), task_things=pred)):
        second, first, other = (suite_priors.record_for(case, q, gold[q["qid"]], None) for q in qs)
    assert second["anchor"] == Anchor("point", segments=(((6.0, 1.5), (6.0, 1.5)),))
    assert first["anchor"] == Anchor("point", segments=(((2.0, 1.5), (2.0, 1.5)),))
    assert other["anchor"] is None
    assert suite_priors.record_for(case, qs[0], gold[qs[0]["qid"]], None)["anchor"] is None   # no predicate declared


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Hand cases, each with a gold: A0 and A1 the two tasks of one kitchen script; B0 (a `before_doorway` cap of 3) and
    C0 two other scripts' one-task cases, B0 a kitchen and C0 a living room; D0 a two-task case of a fourth script; W0 a
    warehouse case; V0 a kitchen case of the vacuum's job and robot, on the warehouse's layout: it differs from B0 in
    its layout alone."""
    root = tmp_path_factory.mktemp("world")
    a = hand_synth(1, tasks=2)
    return {"A0": hand_case(root, a, 0), "A1": hand_case(root, a, 1), "B0": hand_case(root, hand_synth(2), cap=3),
            "C0": hand_case(root, hand_synth(3, room=HOME.rooms[1])),
            "D0": scored_case(root, hand_synth(4, tasks=2), [during_q(1, 12)], a=0, b=1, length="medium"),
            "W0": hand_case(root, hand_synth(5, room=WAREHOUSE.rooms[0], stage=WAREHOUSE)),
            "V0": hand_case(root, hand_synth(6, stage=replace(HOME, map_name=WAREHOUSE.map_name)))}


def test_priors_never_come_from_the_own_script(world):
    """Contract (map X-H14, plan §4.8): the records a case's prior holds come from the cases of other scripts on its
    layout and job, in the order given, and never from its own script (two cases of one script are windows of one world
    and one path: the prior would hold its own answer), nor from another layout's, nor another job's. Hand values: a
    hand case asks two questions and D0 one, so A0's prior is B0's two, C0's two and D0's one; B0's the four of A's two
    cases, C0's two and D0's one; the warehouse case W0 and the vacuum case V0 share a layout and not a job, and V0
    shares B0's job and not its layout, so neither has another script on both. Regression caught: a prior that
    memorises the answer, or one pooled across layouts or across jobs, which dilutes each place's own facts."""
    cases = list(world.values())
    sid = {name: case.script.id for name, case in world.items()}

    def scripts(name: str) -> list[str]:
        return [r["script"] for r in suite_priors.prior_records(world[name], cases, {}, nm=False)]

    assert scripts("A0") == [sid["B0"]] * 2 + [sid["C0"]] * 2 + [sid["D0"]] == scripts("A1")
    assert scripts("B0") == [sid["A0"]] * 4 + [sid["C0"]] * 2 + [sid["D0"]]
    assert scripts("W0") == [] == scripts("V0")


def test_ask_map_is_total(tmp_path):
    """Contract (map X-H15): each own during question's decision keyframe maps to the other window's j-th decision point
    of its kind (j its index among the own window's points of that kind), else the other's last point of that kind,
    else the last step of the other window, so every question is asked. Own: the second task of a script (its steps
    20-39), decision points before_doorway at 23 and 32 and room_done at 36, a question at each. Other, holding one
    before_doorway (8) and no room_done: 23 to 8, 32 to the last before_doorway, 8 again, and 36 to the window's last
    step, 19. Other, with before_doorway at 6 and 14 and room_done at 17: 23 to 6, 32 to 14, 36 to 17. Both streams run
    with every question answered. Regression caught: a question never asked, or a target that ends no look window of the
    other stream."""
    room = HOME.rooms[0]
    own = case_on(tmp_path, Synth(tasks=[[(room, 20)], [(room, 20)]], seed=1,
                                  decisions=[(3, BD), (12, BD), (23, BD), (32, BD), (36, "room_done")]),
                  [during_q(1, 23), during_q(2, 32), during_q(3, 36), end_q(4)], a=1)
    one = case_on(tmp_path, Synth(tasks=[[(room, 20)]], seed=2, decisions=[(8, BD)]), [during_q(1, 8)])
    two = case_on(tmp_path, Synth(tasks=[[(room, 20)]], seed=3, decisions=[(6, BD), (14, BD), (17, "room_done")]),
                  [during_q(1, 6)])
    assert suite_priors.ask_map(own, one) == {23: 8, 32: 8, 36: 19}
    assert suite_priors.ask_map(own, two) == {23: 6, 32: 14, 36: 17}
    for other in (one, two):
        run, _ = play(own, AllUnknown, stream_from=other, ask_map=suite_priors.ask_map(own, other))
        assert run.status == "complete" and run.failures == {}
        assert set(run.answers) == {qid(n) for n in (1, 2, 3, 4)}


class Plain(System):
    """A system the options are built for and never run."""


def test_options_come_from_the_flags_alone(world, monkeypatch):
    """Contract (plan §4.8): `options_for` reads the system's `Baseline` flags and nothing else, and gives exactly what
    each asks for: `needs_prior` the prior records of the other scripts (here A's two cases and C's; B's own script is
    left out and the warehouse case is another layout), carrying `NM_SYSTEM`'s answers from the earlier wave when
    `prior_nm` (six, in the hand values below, and a ValueError without that wave or for an answer it failed to give);
    `needs_map` the public map of the layout; `needs_events` the job's four named moments; `other_stream` the other case
    and where its during question is asked (B0, whose decision at step 12 is its second `before_doorway`, fed A0's
    stream, whose second is step 12), with the maps that put a clock and a task its questions name onto that stream (B0
    names none: the clock map is empty, its one task maps to A0's one). A system with no flag gets nothing.
    Regression caught: a system handed more than it may have, a first wave skipped, or a flag that gives an option
    that is not its own."""
    monkeypatch.setattr(baselines, "BASELINES", {
        "plain": Baseline(Plain), "blind": Baseline(Plain, needs_prior=True),   # `blind`: in `gates.LEARNED`, so a twin
        "keyed": Baseline(Plain, needs_prior=True, prior_nm=True), "map": Baseline(Plain, needs_map=True),
        "events": Baseline(Plain, needs_events=True), "other": Baseline(Plain, other_stream=True)})
    a0, a1, b0, c0, w0 = (world[n] for n in ("A0", "A1", "B0", "C0", "W0"))
    cases = [a0, a1, b0, c0, w0]
    assert suite_priors.options_for("plain", b0, cases, {}) == RunOptions({})
    prior = suite_priors.options_for("blind", b0, cases, {})
    assert list(prior.opts) == ["prior"] and prior.stream_from is None and prior.ask_map is None
    assert [r["script"] for r in prior.opts["prior"]] == [a0.script.id] * 4 + [c0.script.id] * 2
    assert [r["nm"] for r in prior.opts["prior"]] == [None] * 6
    twin = suite_priors.options_for("blind_in_pool", b0, cases, {})   # the in-pool twin: the same records
    assert (prior.opts["prior"].in_pool, twin.opts["prior"].in_pool) == (False, True)
    assert list(twin.opts["prior"]) == list(prior.opts["prior"])
    with pytest.raises(ValueError, match="no baseline named 'plain_in_pool'"):   # a system with no prior has no twin
        suite_priors.options_for("plain_in_pool", b0, cases, {})
    done = {(NM, a0.id): run_of({qid(1): number(1.0), qid(2): at(1, 1)}),
            (NM, a1.id): run_of({qid(1): number(1.5), qid(2): at(2, 2)}),
            (NM, c0.id): run_of({qid(1): number(0.5), qid(2): at(3, 3)})}
    keyed = suite_priors.options_for("keyed", b0, cases, done)
    assert [r["nm"] for r in keyed.opts["prior"]] == [number(1.0), at(1, 1), number(1.5), at(2, 2), number(0.5),
                                                      at(3, 3)]
    with pytest.raises(ValueError, match="no current_view run of case"):
        suite_priors.options_for("keyed", b0, cases, {})
    failed = {**done, (NM, a1.id): run_of({qid(1): number(1.5), qid(2): {"type": "failed", "reason": "ask error"}})}
    with pytest.raises(ValueError, match="gave no answer to question"):
        suite_priors.options_for("keyed", b0, cases, failed)
    placed = suite_priors.options_for("map", b0, cases, {})
    assert list(placed.opts) == ["map", "zones", "slot_zones", "task_things"]
    assert placed.opts["map"].name == "apartment_a"
    assert (placed.opts["zones"], placed.opts["slot_zones"]) == ({}, ())   # the vacuum declares no zone
    assert placed.opts["task_things"] is None   # and, until 0.6.0's jobs, no `task_things` predicate
    events = suite_priors.options_for("events", b0, cases, {})
    assert list(events.opts) == ["events"] and events.opts["events"] == dict(job_registry.get("vacuum/1").events)
    assert set(events.opts["events"]) == {"when you left the dock", "when you started cleaning",
                                          "when you finished cleaning", "when you got back to the dock"}
    other = suite_priors.options_for("other", b0, [b0, a0, c0, w0], {})
    assert (other.stream_from.id, other.ask_map) == (a0.id, {12: 12})
    assert list(other.opts) == ["clock_map", "task_map", "clock_back"]
    assert (other.opts["clock_map"], other.opts["task_map"]) == ({}, {0: 0})
    assert set(other.opts["clock_back"]) == set(suite_priors._window(a0).at.values())   # all A0's delivered clocks


def test_other_case_prefers_the_same_plans_and_says_what_is_missing(world):
    """Contract (plan §4.8): `other_case` is fed a case of another script on the same layout, job, robot and length;
    with none it raises ValueError naming what is missing. B0 is a kitchen case (short): with only C0 (a living room)
    it takes C0; A1 is of A0's own script and D0 is a medium window of two tasks, so neither is a candidate alone; the
    warehouse case W0 is not one for a flat, and nor is V0, the vacuum's kitchen case on another layout. Regression
    caught: the stream of a case of the wrong layout or length, or of the case's own script (its own world), or a
    silent fallback when there is none."""
    a0, a1, b0, c0, d0, v0, w0 = (world[n] for n in ("A0", "A1", "B0", "C0", "D0", "V0", "W0"))
    assert suite_priors.other_case_for(b0, [b0, c0, w0, v0]).id == c0.id
    for case, cases, why in [(b0, [b0], "no case of another script"), (a0, [a0, a1], "no case of another script"),
                             (b0, [b0, d0], "of length short"), (b0, [b0, w0], "layout apartment_a"),
                             (b0, [b0, v0], "layout apartment_a")]:
        with pytest.raises(ValueError, match=why):
            suite_priors.other_case_for(case, cases)


def test_other_case_takes_the_same_room_whatever_the_hash_ranks_first(tmp_path):
    """Contract (plan §4.8): among the cases it could be fed, `other_case` takes one of the same task plans (the same
    room) before the case-id hash chooses. B0 is a kitchen case (id ...b0), A0 a kitchen (...a1) and C0 a living room
    (...c0), the ids fixed so that the hash of (B0, C0) is below that of (B0, A0): with no preference C0 would be fed.
    A0 is, whichever order the cases come in, and with only C0 it is C0. Regression caught: a preference for the same
    room that is lost (the stream of another room's frames for half of all random ids, which a test on random ids
    cannot tell from a hash that happened to agree)."""
    def make(cid, synth, **kw):
        with mock.patch.object(case_mod, "claim_id", lambda dataset, entry: cid):
            return hand_case(tmp_path, synth, **kw)

    b0 = make("00000000000000b0", hand_synth(2))
    a0 = make("00000000000000a1", hand_synth(1))
    c0 = make("00000000000000c0", hand_synth(3, room=HOME.rooms[1]))
    assert stable_hash(b0.id, c0.id) < stable_hash(b0.id, a0.id)   # the arrangement: the hash alone prefers C0
    assert suite_priors.other_case_for(b0, [b0, c0, a0]).id == a0.id == suite_priors.other_case_for(b0, [a0, c0, b0]).id
    assert suite_priors.other_case_for(b0, [b0, c0]).id == c0.id


def test_table_systems_get_the_map(world, tmp_path, monkeypatch):
    """Contract (E3a §2): `options_for` gives `map`, the case layout's public `MapSpec`, to the eight table systems
    (`oracle`, `memory_table`, `current_view`, `current_view_prior`, `kind_range`, `other_case`, `latest_table`,
    `first_table`, for the families that name a room) and to `path_log` and the reference `map_reader`, and to no other
    system; a table system's answer context carries it as `ctx["map"]`, None for a system built without one. Hand case:
    B0's options on the apartment; one answerer, registered for the test, notes the `ctx["map"]` it was handed in a
    system given a map and in one given none. Regression caught: a room-scoped answerer with no room to read."""
    a0, a1, b0, c0, w0 = (world[n] for n in ("A0", "A1", "B0", "C0", "W0"))
    cases = [a0, a1, b0, c0, w0]
    done = {(NM, case.id): run_of({qid(1): number(1.0), qid(2): at(1, 1)}) for case in (a0, a1, c0)}
    given = {name: suite_priors.options_for(name, b0, cases, done).opts.get("map") for name in baselines.BASELINES}
    assert {name for name, m in given.items() if m is not None} == {
        "oracle", "memory_table", "current_view", "current_view_prior", "other_case", "latest_table", "first_table",
        "path_log", "kind_range", "map_reader"}
    assert {m.name for m in given.values() if m is not None} == {"apartment_a"}

    seen = []
    monkeypatch.setitem(table_answers.ANSWERERS, "ghost",
                        lambda store, slots, ctx: seen.append(ctx["map"]) or {"type": "count", "value": 1})
    apartment = mapspec.load_map("apartment_a")
    for name, opts in (("mapped", {"map": apartment}), ("bare", {})):
        system = started(MemoryTable, tmp_path / name, events=EVENTS, **opts)
        system.observe([step(0, 0.0, 0.0)])
        system.ask("end", {"k": 0, "clock": ""}, [question("ghost", {}, answer_type="count")])
    assert seen == [apartment, None]


def test_table_systems_get_the_jobs_zone_rules(world, monkeypatch):
    """Contract (E §1, the review's M3): `options_for` gives `zones` and `slot_zones`, the job's rules, with the map to
    each system that needs the map (here `oracle`, `memory_table` and the memoryless `current_view`), and nothing to one
    that needs none; the rules pickle, as the suite's pool sends a run's options to its worker. A memoryless baseline
    that held no rule would answer every zone question `unknown` and make the memory-needed gate vacuous. Hand case:
    B0's options with the registry answering the tote robot's kit job (the rules are not looked up on B0's map).
    Regression caught: `current_view` without the rules, or a rule that no longer crosses a process."""
    monkeypatch.setattr(job_registry, "get", lambda job_id: zk.JOB)
    b0 = world["B0"]
    for name in ("oracle", "memory_table", "current_view"):
        opts = suite_priors.options_for(name, b0, [b0], {}).opts
        assert (opts["zones"], opts["slot_zones"]) == (zk.JOB.zones, zk.JOB.slot_zones), name
        assert pickle.loads(pickle.dumps(opts)) == opts and opts["slot_zones"] == (zk.LANES,)
    assert suite_priors.options_for("all_unknown", b0, [b0], {}).opts == {}


def stub_case(cid: str, script: str, tasks: int, length: str = "long", plans: tuple = ()) -> SimpleNamespace:
    """A case as `other_case_for` reads one, its facts stubbed: its ids, its layout, job, robot and length, and a window
    of `tasks` tasks with the given plans."""
    return SimpleNamespace(id=cid, script=SimpleNamespace(id=script),
                           info={"layout": "apartment_a", "job": "vacuum/2", "robot": "vacuum", "length": length},
                           window=SimpleNamespace(rec=SimpleNamespace(a=0, b=tasks - 1), plans=plans))


def test_other_case_finds_a_partner_for_every_window(monkeypatch):
    """Contract (E3a §2, the review's B1): `other_case` is fed a case of another script on the same layout, job, robot
    and length, the same task count preferred, else the nearest (the fewer on a tie), so a long window of a size that
    occurs once in a dataset still finds a partner; it raises, naming the length, only when no other script has one.
    Hand cases, facts stubbed: long windows of 4, 5 and 6 tasks, one of each over three scripts: the 5 takes the 4 (one
    task away either side, the fewer), the 4 takes the 5 and the 6 the 5; a second script's 4 is taken by the 4 before
    the 5, though the 5 is as near; a window of the 4's own script is never its partner; a medium window with no other
    medium window raises 'of length medium', as does a short one beside long ones. Regression caught: `stbench check`
    refusing a dataset whose long window sizes occur once (a refusal at exit 2 about half the time with ten scripts of
    4 to 6 tasks)."""
    l4, l5, l6 = stub_case("c4", "s1", 4), stub_case("c5", "s2", 5), stub_case("c6", "s3", 6)
    also4, own4 = stub_case("d4", "s4", 4), stub_case("e4", "s1", 4)
    medium, short = stub_case("m3", "s5", 3, "medium"), stub_case("s1", "s6", 1, "short")
    monkeypatch.setattr(suite_priors, "_info", lambda case: case.info)
    monkeypatch.setattr(suite_priors, "_window", lambda case: case.window)

    def partner(case, *cases):
        return suite_priors.other_case_for(case, [case, *cases]).id

    assert [partner(l5, l4, l6), partner(l4, l5, l6), partner(l6, l4, l5)] == ["c4", "c5", "c5"]
    assert partner(l4, l5, also4, l6) == "d4"
    assert partner(l4, own4, l6) == "c6"
    for case, others in ((medium, (l4, l5, l6)), (short, (l4, medium))):
        with pytest.raises(ValueError, match=f"of length {case.info['length']} among the .* case {case.id}"):
            suite_priors.other_case_for(case, [case, *others])


def own_questions(tmp_path, *slots: dict) -> SimpleNamespace:
    """A case whose `questions.jsonl` holds one question a slots dict."""
    path = tmp_path / "questions.jsonl"
    path.write_text("".join(json.dumps({"qid": f"q{i}", "when": "end", "slots": slot}) + "\n"
                            for i, slot in enumerate(slots)), encoding="utf-8")
    return SimpleNamespace(questions=path)


def clocks(start: str, *offsets: int) -> tuple[str, ...]:
    """The delivered clocks of a task that starts at `start` (a whole second), `offsets` whole seconds after it, to the
    tenth of a second as the stream writes them (`...:SS.0`), written out here."""
    return tuple((datetime.fromisoformat(start) + timedelta(seconds=s)).strftime("%Y-%m-%dT%H:%M:%S") + ".0"
                 for s in offsets)


def test_other_case_asks_at_the_same_moment(tmp_path, monkeypatch):
    """Contract (E3a §2, plan §4.8: "clocks in slots translate by task-relative offset"): `other_case`, fed another
    stream, is asked a clock- or task-named question at the same moment of that stream. `task_map(own, other)` sends
    each of own's case task indices to itself or to the other's last when past it; `clock_map` sends a clock to the
    other's delivered clock nearest the same offset into the mapped task, the earlier on a tie. Hand case: own holds 5
    tasks starting at 08:00, 10:00, ... 16:00; the other 4, starting at 09:10, 10:10, 11:10 and 12:10, each with steps
    0, 60, 149, 151 and 300 s after its start. `task_map` is 0-3 to themselves and 4 to 3. Own's 10:02:30 (150 s into
    its task 1) is 150 s into the other's task 1, 10:12:30, halfway between its steps 10:12:29 and 10:12:31: the
    earlier, 10:12:29. Own's 16:01:00 (60 s into task 4) is 60 s into the other's task 3, 12:11:00, a step of it; own's
    17:00:00, past the end of its task 4, is the other's last step of task 3, 12:15:00. A `MemoryTable` fed the other
    stream and given the maps answers a question as the other stream was asked it: a registered answerer reads the step
    at the clock (k 7, the other's task 1 holding k 5 to 9 and 10:12:29 its third) and the steps of the task (5, of the
    other's task 3, for own's task 4); `tasks` [2, 4] become [2, 3] and a clock or task the maps lack is kept.
    Regression caught: `other_case` answering a clock or task its stream never had, so its gate passes for nothing."""
    own = own_questions(tmp_path, {"clock": "2026-03-02T10:02:30.0"}, {"clock": "2026-03-02T16:01:00.0"},
                        {"clock": "2026-03-02T17:00:00.0"}, {"task": 4})
    own.window = SimpleNamespace(rec=SimpleNamespace(a=0, b=4), plans=(),
                                 clocks=tuple(clocks(f"2026-03-02T{8 + 2 * v:02d}:00:00", *range(0, 200, 10))
                                              for v in range(5)))
    offsets = (0, 60, 149, 151, 300)
    other = SimpleNamespace(window=SimpleNamespace(rec=SimpleNamespace(a=0, b=3), plans=(),
                                                   clocks=tuple(clocks(f"2026-03-02T{9 + v:02d}:10:00", *offsets)
                                                                for v in range(4))))
    monkeypatch.setattr(suite_priors, "_window", lambda case: case.window)
    task_map, clock_map = suite_priors.task_map(own, other), suite_priors.clock_map(own, other)
    assert task_map == {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}
    assert clock_map == {"2026-03-02T10:02:30.0": "2026-03-02T10:12:29.0",
                         "2026-03-02T16:01:00.0": "2026-03-02T12:11:00.0",
                         "2026-03-02T17:00:00.0": "2026-03-02T12:15:00.0"}

    system = started(MemoryTable, tmp_path / "table", events={}, clock_map=clock_map, task_map=task_map)
    for v, task_clocks in enumerate(other.window.clocks):
        system.task(v, task_clocks[0], 0.0, f"task {v}")
        system.observe([{**step(5 * v + i, 0.0, 0.0), "clock": clock} for i, clock in enumerate(task_clocks)])
    monkeypatch.setitem(table_answers.ANSWERERS, "at_clock", lambda store, slots, ctx: {
        "type": "count", "value": store.nearest(slots["clock"])["k"]})
    monkeypatch.setitem(table_answers.ANSWERERS, "in_task", lambda store, slots, ctx: {
        "type": "count", "value": len(store.task_steps(slots["task"]))})
    now = {"k": 19, "clock": ""}
    asked = [question("at_clock", {"clock": "2026-03-02T10:02:30.0"}, "a", "count"),
             question("in_task", {"task": 4}, "b", "count"), question("in_task", {"task": 3}, "c", "count")]
    assert system.ask("end", now, asked) == {"a": {"type": "count", "value": 7}, "b": {"type": "count", "value": 5},
                                             "c": {"type": "count", "value": 5}}
    kept = system.translated(question("ghost", {"tasks": [2, 4], "clock": "elsewhere", "task": 9}))
    assert kept["slots"] == {"tasks": [2, 3], "clock": "elsewhere", "task": 9}
    assert system.translated(question("ghost", {"task": None}))["slots"] == {"task": None}


def test_other_case_answers_on_the_asking_cases_clock(tmp_path, monkeypatch):
    """Contract (plan §4.8; `clock_map`'s inverse): `other_case` is fed another case's stream, so a time answer is a
    clock of that stream, a clock of another day from the asking case's gold, which it could only lose against.
    `clock_back(own, other)` sends each clock the other's stream delivers to the clock of own's task of the same number
    (own's last, when it holds fewer) the same seconds into it. Hand case: own holds 2 tasks starting at 08:00 and
    10:00, the other 3 starting at 09:10, 10:10 and 11:10, each with steps 0, 60, 149, 151 and 300 s after its start.
    The other's 10:12:29 (149 s into its task 1) is 10:02:29, 149 s into own's task 1; its 09:11:00 (60 s into task 0)
    is 08:01:00; its task 2 has no counterpart, so it goes to own's last task: 11:15:00 (300 s in) is 10:05:00, 300 s
    into own's task 1. A `MemoryTable` fed the other's stream and given the map answers a question of the registered
    kind with the clock of its last step, 11:15:00, as 10:05:00; a clock the map lacks, and an answer that is no time,
    are kept. Regression caught: `other_case` scored against another day's clock, which it can never match, so that a
    stream that does carry the answer sits at the floor of its time families for nothing."""
    offsets = (0, 60, 149, 151, 300)
    own = SimpleNamespace(window=SimpleNamespace(clocks=tuple(clocks(f"2026-03-02T{8 + 2 * v:02d}:00:00", *offsets)
                                                              for v in range(2))))
    other = SimpleNamespace(window=SimpleNamespace(clocks=tuple(clocks(f"2026-03-02T{9 + v:02d}:10:00", *offsets)
                                                                for v in range(3))))
    monkeypatch.setattr(suite_priors, "_window", lambda case: case.window)
    back = suite_priors.clock_back(own, other)
    assert len(back) == 15
    assert [back[c] for c in other.window.clocks[0]] == list(clocks("2026-03-02T08:00:00", *offsets))
    assert [back[c] for c in other.window.clocks[1]] == list(clocks("2026-03-02T10:00:00", *offsets))
    assert [back[c] for c in other.window.clocks[2]] == list(clocks("2026-03-02T10:00:00", *offsets))
    assert back["2026-03-02T10:12:29.0"] == "2026-03-02T10:02:29.0"
    assert back["2026-03-02T11:15:00.0"] == "2026-03-02T10:05:00.0"

    system = started(MemoryTable, tmp_path / "table", events={}, clock_back=back)
    for v, task_clocks in enumerate(other.window.clocks):
        system.task(v, task_clocks[0], 0.0, f"task {v}")
        system.observe([{**step(5 * v + i, 0.0, 0.0), "clock": clock} for i, clock in enumerate(task_clocks)])
    monkeypatch.setitem(table_answers.ANSWERERS, "last_clock", lambda store, slots, ctx: {
        "type": "time", "clock": ctx["now"]["clock"]})
    asked = question("last_clock", {}, "a", "time")
    assert system.ask("end", {"k": 14, "clock": ""}, [asked]) == {
        "a": {"type": "time", "clock": "2026-03-02T10:05:00.0"}}
    assert system.untranslated({"type": "time", "clock": "elsewhere"}) == {"type": "time", "clock": "elsewhere"}
    assert system.untranslated({"type": "count", "value": 5}) == {"type": "count", "value": 5}
    plain = started(MemoryTable, tmp_path / "plain", events={})   # a system fed its own case's stream answers as asked
    assert plain.untranslated({"type": "time", "clock": "2026-03-02T11:15:00.0"}) == {
        "type": "time", "clock": "2026-03-02T11:15:00.0"}


def test_a_clock_put_back_on_the_asking_case_keeps_its_tenth_of_a_second(monkeypatch):
    """Contract (0.10.0, delivered clocks carry tenths): `clock_back` writes the asking case's clock in the delivered
    shape, `YYYY-MM-DDTHH:MM:SS.s`, with the tenth kept, so a time answer put back on it loses nothing the stream
    delivered. Hand case: own's one task starts at 08:00:00.0, the other's at 09:10:00.0 with steps 0.0, 1.5 and 3.0 s
    in; the clocks go back to 08:00:00.0, 08:00:01.5 and 08:00:03.0. Regression caught: a put-back clock cut to the
    second (08:00:01 for 08:00:01.5), which scores a time answer 0.5 s wrong against a gold that was right."""
    own = SimpleNamespace(window=SimpleNamespace(clocks=(("2026-03-02T08:00:00.0",),)))
    other = SimpleNamespace(window=SimpleNamespace(
        clocks=(("2026-03-02T09:10:00.0", "2026-03-02T09:10:01.5", "2026-03-02T09:10:03.0"),)))
    monkeypatch.setattr(suite_priors, "_window", lambda case: case.window)
    assert suite_priors.clock_back(own, other) == {"2026-03-02T09:10:00.0": "2026-03-02T08:00:00.0",
                                                   "2026-03-02T09:10:01.5": "2026-03-02T08:00:01.5",
                                                   "2026-03-02T09:10:03.0": "2026-03-02T08:00:03.0"}
