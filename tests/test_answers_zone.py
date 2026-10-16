"""The answers to `zone_clear`, `clearance_then` and `slots_empty`: `table_zone` for the table systems and
`baselines_path_zone` for `path_log` (the gold is `questions_zone`'s, pinned by hand in `test_questions_zone.py`).

**The end-to-end case** is a run of the tote robot on `zone_kit`'s map from real ray-cast depth, written to files and
played through `run_case` on `detections`. Its five rendered keyframes (`DEPTH_KS`; the base is (3.9, 1.5) facing west,
the bay B1 ahead, unless said) and what they show:
- 5: from (8.2, 2.0) facing east, the staging lanes: L1 whole and free (87%), L2's near end with a 0.5 m box standing
  in it (40%, blocked); L3 and L4 are never in view;
- 17 (the decision's look window, 16-17): B1 with a 0.5 m box in it (85% in view);
- 18 ('when you reached the station'): B1 with a 0.2 m box in it, which is no obstacle to a zone that starts at 0.3 m;
- 25 ('when you finished at the station'): the base is at (3.9, 4.5) and the bay B2 shows a 0.5 m box;
- 36 (the end look window, 34-39): the base faces east and sees neither bay nor lane.
The golds by hand: `zone_clear` at 17 is no (the look window decides); 'reached' is yes (B1, the 0.2 m box, seen in look
window 18-25 up to 18); 'finished' is no (the nearest bay at 25 is B2, with the box); `slots_empty` lists the centre of
L1, (10, 2) (L2 is blocked, L3 and L4 were not passed).

**The hand case** (`feed`) is twelve steps of two tasks at the base (3.9, 1.5) facing west, whose zone cells are put in
by hand (`VIEWS`): look windows 0-3 (task 0: approach, at_station at 1 and 2, undocking), 4-5, then task 1: 6-9 (the
same phases), 10-11. Keyframe 1 sees B1 and L1 free; 2 puts 30 occupied cells in B1 and sees half of L3 and all of L4; 7
sees B1 free; 8 sees L2 with 60 occupied cells; 11 sees L4. The events are 'reached' (the first at_station) and
'finished' (the last)."""

from __future__ import annotations

import math
from dataclasses import replace
from unittest import mock

import pytest

import zone_kit as zk
from answers_kit import compiled_case, scored
from baseline_kit import UNKNOWN, play, question, record, rows, started, step
from shape_kit import APARTMENT, NORTH
from space_kit import box
from stbench import freespace, job_registry, questions_world, zones
from stbench.baselines import CurrentView, MemoryTable, Oracle
from stbench.baselines_path import PathLog
from stbench.table import Mapper, Seen, Store
from stbench.table_zone import rule_of
from stbench.zones import ZoneRule
from zone_kit import B1, L1, L2, L3, L4, paint

# One xdist group (`--dist loadgroup`): the end-to-end case is compiled and played once, on one worker.
pytestmark = pytest.mark.xdist_group(name=__name__)

B1_BOX, LOW_BOX = box(1.2, 1.6, 1.2, 1.8, 0.0, 0.5), box(1.2, 1.6, 1.2, 1.8, 0.0, 0.2)
B2_BOX, L2_BOX = box(1.2, 1.6, 4.2, 4.8, 0.0, 0.5), box(11.9, 12.3, 1.5, 2.5, 0.0, 0.5)
BAND = (0.0, (0.3, 3.5))   # the bays' and lanes' ground and band: every zone of the kit's job is on the floor
JOB = replace(zk.JOB, assignments={"clean_room": "Its current assignment is to move totes at the {room}."})
OPTS = {"events": zk.EVENTS, "map": zk.MAP, "zones": JOB.zones, "slot_zones": JOB.slot_zones}
YES, NO = {"type": "yesno", "value": True}, {"type": "yesno", "value": False}


DEPTH_KS = (5, 17, 18, 25, 36)


def zone_run():
    s = zk.script(seed=7)
    s.depth_ks = set(DEPTH_KS)
    s.poses[5], s.yaws[5] = (8.2, 2.0), 0.0
    s.poses.update(dict.fromkeys(range(22, 26), (3.9, 4.5)))
    s.yaws.update(dict.fromkeys(range(34, 40), 0.0))
    s.frame_boxes = {5: [L2_BOX], 17: [B1_BOX], 18: [LOW_BOX], 25: [B2_BOX]}
    return s


@pytest.fixture(scope="module")
def played(tmp_path_factory):
    """The case "c0" over `zone_kit`'s map and job, compiled, and `oracle`, `memory_table` and `current_view` played
    over it through the stream: `(case, questions, golds, {class name: (run, system)})`."""
    with mock.patch.object(questions_world, "load_map", lambda name: zk.MAP):
        case = compiled_case(tmp_path_factory.mktemp("zones"), "c0", zone_run(), JOB, 0, 0)
    qs, golds = rows(case)
    with mock.patch.object(job_registry, "get", lambda job_id: JOB):
        return case, qs, golds, {cls.__name__: play(case, cls, **OPTS) for cls in (Oracle, MemoryTable, CurrentView)}


def test_the_oracle_answers_the_zone_families(played):
    """Contract (E §3): through `run_case` on `detections`, `oracle` and `memory_table` score 1 on all three zone
    families, the mapper's zone states read from the real depth the stream delivers (each zone on its own band, kept in
    the zones' cells) and the moments resolved from the delivered steps. Hand-worked from the scene above: the
    decision's look window holds B1 with a 0.5 m box, so `zone_clear` is no; 'when you reached the station' is yes (a
    0.2 m box is under the 0.3 m the band starts at, which the robot's own band would call occupied) and 'when you
    finished at the station' is no (at 25 the nearest bay is B2, with its box); `slots_empty` is the one item (10, 2),
    L1. Regression caught: a zone, band, look window or moment that differs from the gold's, or a zone judged on the
    robot's band."""
    _, qs, golds, runs = played
    asked = list(zip(qs, golds, strict=True))
    assert [(q["decision_k"], q["slots"]) for q, g in asked if g["family"] == "zone_clear"] == [
        (17, {"zone": zk.BAY.words, "height": 0.3})]
    assert {(g["family"], q["slots"].get("event")): g["gold"] for q, g in asked} == {
        ("zone_clear", None): NO, ("clearance_then", zk.REACHED): YES, ("clearance_then", zk.FINISHED): NO,
        ("slots_empty", None): {"type": "positions", "items": [{"xy": [10.0, 2.0]}]}}
    for name in ("Oracle", "MemoryTable"):
        run, _ = runs[name]
        assert run.status == "complete" and run.failures == {}, name
        assert scored(golds, run.answers) == dict.fromkeys((q["qid"] for q in qs), 1.0), name


def test_current_view_gives_the_gold_on_the_now_rows(played):
    """Contract (VISION rule 4): the one `now` row of the case, `zone_clear` at the decision, is answered by
    `current_view`, a system that keeps the last look window alone, exactly as the gold has it, and every other row of
    the case is `memory`. Hand-worked from the scene above: the decision's look window holds B1 with a 0.5 m box, so the
    gold is no and `current_view`'s store (the look window 16-17) says no. Regression caught: a `zone_clear` that reads
    an earlier look window (a `memory` row, which the current view cannot answer) or one whose gold the memoryless
    baseline does not reproduce."""
    _, qs, golds, runs = played
    now = {g["qid"] for g in golds if g["scope"] == "now"}
    assert now == {q["qid"] for q, g in zip(qs, golds, strict=True) if g["family"] == "zone_clear"} and len(now) == 1
    assert {g["scope"] for g in golds if g["qid"] not in now} == {"memory"}
    run, _ = runs["CurrentView"]
    assert run.status == "complete" and run.failures == {}
    assert {qid: score for qid, score in scored(golds, run.answers).items() if qid in now} == dict.fromkeys(now, 1.0)


def test_the_mapper_keeps_each_zone_band(played):
    """Contract (E §3): a step's zone cells equal `freespace.script_states` of the same keyframe with the zone's ground
    and band, keeping the cells of the job's zones, and the robot band's cells are the free-space rule's own. At
    keyframe 18 the 0.2 m box stands in B1: the robot's band marks its cells occupied (a 0.2 m thing is over the AMR's
    climb), the zone band, which starts at 0.3 m, marks none. Regression caught: a zone judged on the robot's band, a
    zone band that differs from the gold's, or the robot's families moved by the zones."""
    case, *_, runs = played
    keep = zones.band_cells(zk.MAP, [*zk.JOB.zones.values(), *zk.JOB.slot_zones])
    assert set(keep) == {BAND}
    store = runs["MemoryTable"][1].store
    assert {k: store.zone_cells[BAND][k] for k in DEPTH_KS} == freespace.script_states(
        case.script, DEPTH_KS, *BAND, keep=keep[BAND])
    assert {k: store.cells[k] for k in DEPTH_KS} == freespace.script_states(case.script, DEPTH_KS)
    assert not any(store.cells[k] or store.zone_cells[BAND][k] for k in range(40) if k not in DEPTH_KS)
    assert any(s == "occupied" for s in store.cells[18].values())
    assert store.zone_cells[BAND][18] and not any(s == "occupied" for s in store.zone_cells[BAND][18].values())


# -- by hand ----------------------------------------------------------------------------------------------------
class HandMapper(Mapper):
    """A mapper that has seen `views` ({keyframe: the cells every zone band shows}) as the depth would have shown it."""

    def __init__(self, real: Mapper, views: dict):
        vars(self).update(vars(real))
        self.views = views

    def observe(self, step: dict):
        cells = self.views.get(step["k"])
        return None if cells is None else Seen({}, dict.fromkeys(self.zone_cells, cells))


VIEWS = {1: {**paint(B1), **paint(L1)},
         2: {**paint(B1, "occupied", n=30, start=100), **paint(L3, n=196), **paint(L4)},
         7: paint(B1),
         8: {**paint(L2), **paint(L2, "occupied", n=60, start=100)},
         11: paint(L4)}
PHASES = ["approach", "at_station", "at_station", "undocking", "waiting", "waiting",
          "approach", "at_station", "at_station", "undocking", "waiting", "waiting"]
WINDOWS = [range(0, 4), range(4, 6), range(6, 10), range(10, 12)]   # a `task` message before 0 and before 6
CLOCK = "2026-03-02T08:00:00"


def feed(cls, tmp_path, **opts):
    """A `cls` over the kit's map and rules whose mapper is `HandMapper(VIEWS)`, sent the hand case, and what it
    answered to `zone_clear` (the bay B1) after the look windows ending at 3, 5, 9 and 11."""
    system = started(cls, tmp_path, **{**OPTS, **opts})
    system.mapper = HandMapper(system.mapper, VIEWS)
    system.store = Store(system.mapper)
    during = {}
    for window in WINDOWS:
        if window.start in (0, 6):
            system.task(window.start // 6, CLOCK, 0.0, "")
        system.observe([step(k, 3.9, 1.5, 180.0, PHASES[k]) for k in window])
        during[window[-1]] = system.ask("during", {"k": window[-1], "clock": CLOCK}, [clear()])["q1"]
    return during, system


def clear(words: str = zk.BAY.words) -> dict:
    return question("zone_clear", {"zone": words, "height": 0.3}, answer_type="yesno")


def then(event: str, task: int | None) -> dict:
    return question("clearance_then", {"zone": zk.BAY.words, "height": 0.3, "event": event, "task": task},
                    qid=f"{task}|{event}", answer_type="yesno")


EMPTY = {"type": "positions", "items": []}
SLOTS = question("slots_empty", {"places": "staging lanes", "height": 0.3}, qid="lanes", answer_type="positions")
ENDS = [then(zk.REACHED, 0), then(zk.FINISHED, 0), then(zk.REACHED, 1), then(zk.FINISHED, 1), then(zk.FINISHED, None),
        SLOTS]


def test_a_table_answers_the_zone_families_from_what_it_holds(tmp_path):
    """Contract (E §1): `memory_table` answers each zone family as the gold does, from the zone cells it kept and the
    look window each step arrived in. Hand-worked on the hand case: `zone_clear` after the look window 0-3 is no (B1 is
    covered there, and the cell states are the latest: 30 occupied at 2 over the free of 1), after 4-5, which saw
    nothing of B1, `unknown` (the table holds what 1 and 2 showed, and the question is about right now), after 6-9 yes
    (7 saw B1 free) and after 10-11 `unknown` again. The end moments: task 0's 'reached' (step 1) yes, since the blocker
    came at 2, after it, though in the same look window; 'finished' (step 2) no, the moment's own step counting; task
    1's 'reached' (7) and 'finished' (8) yes, and with no task named the last task's. `slots_empty` lists L1 only: L2
    blocked (8), L3 half seen (unknown), L4 in the end look window (11). `other_case`'s map `task_map` {0: 1} puts task
    0's 'finished' on task 1's.
    Regression caught: a state after the moment read, an earlier look window's cells read for it, `zone_clear` answered
    from an earlier look window (after 4-5 and 10-11 the table holds what it saw of B1 and says `unknown`, as the gold
    does), or a zone passed judged on the wrong steps."""
    during, system = feed(MemoryTable, tmp_path / "m")
    assert during == {3: NO, 5: UNKNOWN, 9: YES, 11: UNKNOWN}
    ends = system.ask("end", {"k": 11, "clock": CLOCK}, ENDS)
    assert ends == {"0|when you reached the station": YES, "0|when you finished at the station": NO,
                    "1|when you reached the station": YES, "1|when you finished at the station": YES,
                    "None|when you finished at the station": YES,
                    "lanes": {"type": "positions", "items": [{"xy": [10.0, 2.0]}]}}
    _, system = feed(MemoryTable, tmp_path / "o", task_map={0: 1})
    assert system.ask("end", {"k": 11, "clock": CLOCK}, [then(zk.FINISHED, 0)]) == {
        "0|when you finished at the station": YES}


def test_current_view_answers_the_look_window_only(tmp_path):
    """Contract (E §1, the review's M3): `current_view` holds the rules and the map, and its store is the last look
    window alone: `zone_clear` is answered from the look window (no after 0-3, which holds the blocker; yes after 6-9,
    which saw B1 free), and is `unknown` for a zone the look window did not cover though an earlier one saw it (after
    4-5 and after 10-11), which is also what `memory_table` answers, the family asking only about right now. An end
    question that names a task is `unknown`, the store holding no message of any task; with none named it answers about
    where it stands, and the end look window 10-11 saw nothing of B1: `unknown`; `slots_empty` lists no lane (L4 is the
    only one in the look window). Regression caught: a memoryless baseline that remembers, or one that holds no rule and
    so never answers (a vacuous memory-needed gate)."""
    during, system = feed(CurrentView, tmp_path / "v")
    assert during == {3: NO, 5: UNKNOWN, 9: YES, 11: UNKNOWN}
    assert system.ask("end", {"k": 11, "clock": CLOCK}, ENDS) == {
        "0|when you reached the station": UNKNOWN, "0|when you finished at the station": UNKNOWN,
        "1|when you reached the station": UNKNOWN, "1|when you finished at the station": UNKNOWN,
        "None|when you finished at the station": UNKNOWN, "lanes": EMPTY}


PLACES = {"before_room": zk.WAY, "before_place": zk.SPOT}
D1_WAY = zones.zone_of(APARTMENT, zk.WAY, (3.9, 1.1), math.pi / 4)
TABLE_WINDOW = zones.zone_of(APARTMENT, zk.SPOT, (2.1, 0.9), NORTH)


def test_a_table_finds_a_doorway_and_a_spot_as_the_gold_does(tmp_path):
    """Contract: `memory_table` answers `zone_clear` for a doorway's way and for the window of the surface the base
    faces from the pose and heading the stream delivered (`pose.yaw_deg`), reading the look window alone. Hand case on
    `apartment_a`, hand-set cells (`HandMapper`), one look window of two steps each, the question asked after each:
    - at (3.9, 1.1) heading 45 degrees, D1's way seen free at step 1: yes;
    - at (2.1, 0.9) heading 90, the dining table's window seen free at 3: yes;
    - the same base heading 270 (the sink unit's window, which nothing showed), with the table's window seen free at
      5: `unknown`; heading 0 (nothing within 1.5 m is faced): `unknown`, whatever was seen;
    - back at D1 with nothing seen in the look window (8-9), though step 1 saw its way: `unknown`.
    Regression caught: a spot chosen from the base alone (the heading not read, or read from the camera), or a zone
    answered from a look window before the current one."""
    views = {1: paint(D1_WAY), 3: paint(TABLE_WINDOW), 5: paint(TABLE_WINDOW), 7: paint(TABLE_WINDOW)}
    system = started(MemoryTable, tmp_path, events=zk.EVENTS, map=APARTMENT, zones=PLACES, slot_zones=())
    system.mapper = HandMapper(system.mapper, views)
    system.store = Store(system.mapper)
    system.task(0, CLOCK, 0.0, "")

    def asked(ks: range, x: float, y: float, yaw_deg: float, rule: ZoneRule) -> dict:
        system.observe([step(k, x, y, yaw_deg, "cleaning") for k in ks])
        return system.ask("during", {"k": ks[-1], "clock": CLOCK}, [clear(rule.words)])["q1"]

    assert asked(range(0, 2), 3.9, 1.1, 45.0, zk.WAY) == YES
    assert asked(range(2, 4), 2.1, 0.9, 90.0, zk.SPOT) == YES
    assert asked(range(4, 6), 2.1, 0.9, 270.0, zk.SPOT) == UNKNOWN
    assert asked(range(6, 8), 2.1, 0.9, 0.0, zk.SPOT) == UNKNOWN
    assert asked(range(8, 10), 3.9, 1.1, 45.0, zk.WAY) == UNKNOWN


def test_a_table_without_the_rules_or_the_map_says_so(tmp_path):
    """Contract (E §1): a system built without the job's zone rules answers the zone families `unknown`, a rule the
    mapper does not hold by those words too; one given the rules and no map raises, naming the map; and a slot rule is
    found by its `piece_words` among the slot rules only. Hand case: the kit's rules on one step. Regression caught: a
    system that fails on a missing option instead of abstaining, a zone answered from no map, or a slot rule found
    among a zone's rules that happens to share its plural."""
    lane_ahead = ZoneRule("the lane ahead", "staging_lane", piece_words="staging lanes", margin_m=0.1)
    one = [step(0, 3.9, 1.5, 180.0, "at_station")]
    asks = [clear(), then(zk.REACHED, None), SLOTS]
    bare = started(MemoryTable, tmp_path / "b", events=zk.EVENTS)
    bare.observe(one)
    assert bare.ask("during", {"k": 0, "clock": CLOCK}, [asks[0]]) == {"q1": UNKNOWN}
    assert bare.ask("end", {"k": 0, "clock": CLOCK}, asks[1:]) == {"None|when you reached the station": UNKNOWN,
                                                                    "lanes": UNKNOWN}
    unmapped = started(MemoryTable, tmp_path / "u", events=zk.EVENTS, zones=JOB.zones, slot_zones=JOB.slot_zones)
    unmapped.observe(one)
    for when, ask in zip(("during", "end", "end"), asks, strict=True):
        with pytest.raises(ValueError, match="needs the case layout's public map"):
            unmapped.ask(when, {"k": 0, "clock": CLOCK}, [ask])
    named = started(MemoryTable, tmp_path / "n", **{**OPTS, "zones": {**JOB.zones, "x": lane_ahead}})
    assert (rule_of(named.store, "the lane ahead"), rule_of(named.store, "staging lanes", plural=True)) == (
        lane_ahead, zk.LANES)
    assert rule_of(named.store, "the moon") is None and rule_of(named.store, "the staging lanes") == zk.LANES
    named.observe(one)
    assert named.ask("during", {"k": 0, "clock": CLOCK}, [clear("the moon")]) == {"q1": UNKNOWN}


def prior(q: dict, when: str, gold: dict, credit: str, **tolerances) -> list[dict]:
    """Six records of three other scripts for `q`, each with the gold row `gold` (scored by `credit`), at the base,
    heading and phase of the hand case's step 3."""
    out = []
    for i in range(6):
        r = record(q, gold, script=f"s{i % 3}", when=when, phase="undocking", base=(3.9, 1.5), yaw=180.0)
        out.append({**r, "gold": {**r["gold"], "family": q["template"], "credit": credit, "tolerances": tolerances}})
    return out


def test_path_log_answers_the_zone_families_from_its_pool(tmp_path):
    """Contract (E §2): `path_log` reads no depth, so it answers each zone family from other scripts' gold rows at its
    key and nothing else: the best response over the pool, `unknown` with none. Hand case: six records of three
    scripts for each family, at the phase and pose of the hand case's step 3 (the base (3.9, 1.5), heading west), whose
    golds are no, no, and the one item (10, 2); `path_log` answers those, and `unknown` with an empty prior.
    Regression caught: a `path_log` that answers the zone families from nothing (a vacuous floor), or one that
    answers a zone it never saw."""
    item = {"type": "positions", "items": [{"xy": [10.0, 2.0]}]}
    asked = [("during", clear(), NO, "exact", {}), ("end", then(zk.FINISHED, None), NO, "exact", {}),
             ("end", SLOTS, item, "f1", {"match_m": 1.125})]
    pool = [r for when, q, gold, credit, tol in asked for r in prior(q, when, gold, credit, **tol)]
    for records, want in ((pool, [NO, NO, item]), ([], [UNKNOWN] * 3)):
        log = started(PathLog, tmp_path / str(bool(records)), map=zk.MAP, events=zk.EVENTS, prior=records)
        log.observe([step(k, 3.9, 1.5, 180.0, "undocking") for k in range(4)])
        got = [next(iter(log.ask(when, {"k": 3, "clock": CLOCK}, [q]).values())) for when, q, *_ in asked]
        assert got == want
