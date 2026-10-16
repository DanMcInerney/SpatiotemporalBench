"""`adjacent` answered by the table systems (`table_adjacent`): the oracle over `adjacent_kit`'s six-room apartment, and
the table's rules by hand. The kit's map is patched into `map_files.HELD_OUT` for each test that compiles (a held-out
map itself is never a fixture). Expected values are worked out in the docstrings."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

import adjacent_kit as ak
from answers_kit import compiled_case
from baseline_kit import gold, play, question, rows, scores, started, step
from baseline_kit import record as prior_record
from stbench import map_files, questions_world, score, table_answers
from stbench.baselines import CurrentView, MemoryTable, Oracle
from stbench.baselines_path import PathLog
from stbench.table import Store

CID = "7c1d5a9e3b2f4a68"
SLOTS = {"space": "room", "a": "bedroom", "b": "kitchen"}
UNKNOWN = {"type": "unknown"}
CORRIDOR_POINT = {"type": "position", "xy": [4.6, 1.5]}
# The robot starts in the hall (its dock), goes to the corridor and through the kitchen's door, looks round the kitchen
# (three steps, headings 45, 135 and 225 degrees: each sees about a fifth of the room's floor, so the three tour more
# than half of it), comes back and goes out through the corridor's open edge into the bedroom, looks round there (45,
# 135 and 315 degrees) and returns to the hall; the eight steps of the end look window are in the hall. Only those six
# steps have a depth image, rendered from the floor alone (the hand depth has no wall boxes, so it also sees floor
# through the walls, which only helps the touring: the rule under test is the table's).
WALK = [ak.HALL, ak.COR, ak.D2, *[ak.KIT] * 3, ak.D2, ak.COR, ak.COR_E, *[ak.BED] * 3, ak.COR_E, *[ak.HALL] * 8]
LOOKS = {3: 45, 4: 135, 5: 225, 9: 45, 10: 135, 11: 315}   # keyframe: heading in degrees


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    """`WALK` as a one-task case of the kit with the depth `LOOKS` render (the gold read from the depth files), compiled
    with the kit's job over the kit map, the kit's map named a held-out layout: the case."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(map_files, "HELD_OUT", (*map_files.HELD_OUT, ak.MAP_NAME))
        patch.setattr(questions_world, "load_map", lambda name: ak.kit_map())
        script = ak.script(WALK, depth_ks=set(LOOKS), yaws={k: math.radians(d) for k, d in LOOKS.items()})
        return compiled_case(tmp_path_factory.mktemp("adjacent"), CID, script, ak.JOB, 0, 0)


def test_the_table_reproduces_the_gold(case):
    """Contract (E §5): through `run_case` on `detections`, over the case compiled from the kit's tour (the robot
    crossed corridor-kitchen and corridor-bedroom, looked round both rooms and ended in the hall), the one
    question is "Give a point in the room that connects the bedroom to the kitchen." with the corridor's centre
    (4.6, 1.5) as its gold, and both `oracle` and `memory_table` answer exactly that point and score 1. Regression
    caught: the table and the gold reading the regions, the crossings or the touring differently."""
    [q], [g] = rows(case)
    assert q["slots"] == SLOTS and g["gold"] == CORRIDOR_POINT and g["reference"] == "other_case"
    for cls in (Oracle, MemoryTable):
        run, _ = play(case, cls, map=ak.kit_map(), events=ak.JOB.events)
        assert run.status == "complete" and run.failures == {}, cls.__name__
        assert run.answers[q["qid"]] == CORRIDOR_POINT and scores(case, run) == {q["qid"]: 1.0}, cls.__name__


def test_current_view_lacks_the_crossings(case):
    """Contract (rule 4): `current_view` holds the end look window alone, which spent its eight steps in the hall, so
    the path that showed the connection is not in what it holds: on the case above it answers `unknown` and scores 0
    where `memory_table` scores 1. Regression caught: a memoryless baseline that remembers the crossings."""
    [q], _ = rows(case)
    run, system = play(case, CurrentView, map=ak.kit_map())
    assert run.failures == {} and [s["k"] for s in system.store.steps_until(10**6)] == list(system.latest)
    assert run.answers[q["qid"]] == UNKNOWN and scores(case, run) == {q["qid"]: 0.0}


# -- the table by hand -------------------------------------------------------------------------------------------------
def hand_store(walk: list[tuple[float, float]], seen: dict[str, float] | None = None) -> Store:
    """A store of the steps of `walk` (one task, k = the index), holding at step 0 the fine cells of each room of `seen`
    that room's share observed (default: every room fully), from its lower left."""
    store = Store()
    store.begin_task(0, "2026-03-02T08:00:00", 0.0, "")
    for k, (x, y) in enumerate(walk):
        store.add_step(step(k, x, y))
    for rid, share in (seen or dict.fromkeys(ak.ROOMS, 1.0)).items():
        cells = ak.room_cells(rid)
        store.cells.setdefault(0, {}).update(dict.fromkeys(cells[:round(share * len(cells))], "free"))
    return store


def ask(store: Store, slots: dict = SLOTS, m="kit") -> dict:
    """The table's `adjacent` answer through the registry, as the vacuum asked at the store's last step."""
    kit = ak.kit_map() if m == "kit" else m
    now = store.steps[max(store.steps)]
    return table_answers.answer(store, question("adjacent", slots), {"now": now, "map": kit, "robot": None,
                                                                    "events": None})


def test_the_table_by_hand():
    """Contract (E §1): the table answers the corridor's centre when its steps cross both boundaries (corridor-kitchen
    and corridor-bedroom, either way: here the bedroom first) and it observed half or more of each room's cells, in any
    state. It answers `unknown` when one boundary was never crossed, a room under half observed (49%), a kind is
    not unique in the layout (two bedrooms) or names no room, or no one room connects the two (the joined kitchen and
    study, the walled hall and study); the `space` word is never read. A table given no map says so. Regression caught:
    an answer from the map alone (no crossing), an untoured room named, a connector chosen where the layout has two or
    none, or a slot read that is only words."""
    walk = [ak.COR, ak.BED, ak.COR_E, ak.COR, ak.D2, ak.KIT, ak.COR]
    assert ask(hand_store(walk)) == CORRIDOR_POINT
    assert ask(hand_store(walk), {**SLOTS, "space": "area"}) == CORRIDOR_POINT
    assert ask(hand_store(walk[:3])) == UNKNOWN   # the kitchen's boundary never crossed
    assert ask(hand_store([ak.KIT, ak.D2, ak.COR, ak.D2, ak.KIT])) == UNKNOWN   # the bedroom's never crossed
    assert ask(hand_store(walk, {**dict.fromkeys(ak.ROOMS, 1.0), ak.KIT_ID: 0.49})) == UNKNOWN
    assert ask(hand_store(walk, {**dict.fromkeys(ak.ROOMS, 1.0), ak.BED_ID: 0.5})) == CORRIDOR_POINT
    assert ask(hand_store(walk, {**dict.fromkeys(ak.ROOMS, 1.0), ak.BED_ID: 0.49})) == UNKNOWN
    m = ak.kit_map()
    twin = replace(m, regions=tuple(replace(r, kind="bedroom") if r.id == ak.BATH_ID else r for r in m.regions))
    assert ask(hand_store(walk), m=twin) == UNKNOWN
    assert ask(hand_store(walk), {**SLOTS, "b": "attic"}) == UNKNOWN
    assert ask(hand_store(walk), {**SLOTS, "a": "kitchen", "b": "study"}) == UNKNOWN
    assert ask(hand_store(walk), {**SLOTS, "a": "hall", "b": "study"}) == UNKNOWN
    with pytest.raises(ValueError, match="needs the case layout's map"):
        ask(hand_store(walk), m=None)


# -- path_log -------------------------------------------------------------------------------------------------------
KINDS = tuple(r.kind for r in ak.kit_map().regions)
CORRIDOR = [list(p) for p in ak.kit_map().region(ak.COR_ID).polygon]
TO_BEDROOM = [ak.COR, ak.COR_E, *[ak.BED] * 3]   # its route to the bedroom: the corridor, its east end, the bedroom
TO_KITCHEN = [ak.COR, ak.D2, *[ak.KIT] * 3]      # and to the kitchen: the corridor, the kitchen's door, the kitchen
BEDROOM, KITCHEN = "Please clean the bedroom.", "Its current assignment is to clean the kitchen."


def region_gold(answer: dict) -> float:
    """The penalised score of `answer` against the corridor as a region gold (credit `region`, eroded 0.2 m)."""
    return score.penalised(answer, {**gold({"type": "position", "xy": [4.6, 1.5]}), "family": "adjacent",
                                    "credit": "region", "region": CORRIDOR, "tolerances": {"erode_m": 0.2}})


def played(tmp_path, tasks: list[tuple[str, list]], prior: list[dict] | None = None) -> dict:
    """`path_log` over the tasks (a brief and the walk of each) in turn, given the kit's region kinds and no map, asked
    the kit's question after the last step."""
    system = started(PathLog, tmp_path, kinds=KINDS, events=ak.JOB.events, prior=prior or [])
    k = 0
    for i, (brief, walk) in enumerate(tasks):
        system.task(i, f"2026-03-02T0{8 + i}:00:00", 0.0, brief)
        system.observe([step(k + j, *xy) for j, xy in enumerate(walk)])
        k += len(walk)
    return system.ask("end", {"k": k - 1, "clock": ""}, [question("adjacent", SLOTS)])["q1"]


def test_path_log_places_only_what_its_briefs_give(case, tmp_path):
    """Contract (E §2): `path_log`, given the kit's region kinds and no map, places a room only by a brief that names it
    alone, and the room that connects two it has placed is where its routes to them part. Hand case: the task "clean
    the bedroom" (the corridor, its east end, three steps in the bedroom) and "clean the kitchen" (the corridor, the
    kitchen's door, three steps in the kitchen): the routes share the first position and the second of each is 1.4 m or
    more off the other's, so they part at (4.6, 1.5), the corridor's centre, which scores 1 against the gold. It
    abstains (`unknown`) when one brief names the bedroom and none the kitchen, when none names either, when one brief
    names both (its steps cannot be told apart), and on the kit's own case (the brief names a room the kit does not
    have), which scores 0. Regression caught: a `path_log` weaker than the agent it models (a vacuous gate), or one
    that answers a pair the moment rule asks, which would pass the shortcut gate for nothing."""
    placed = played(tmp_path / "both", [(BEDROOM, TO_BEDROOM), (KITCHEN, TO_KITCHEN)])
    assert placed == CORRIDOR_POINT and region_gold(placed) == 1.0
    assert played(tmp_path / "swapped", [(KITCHEN, TO_KITCHEN), (BEDROOM, TO_BEDROOM)]) == CORRIDOR_POINT
    assert played(tmp_path / "one", [(BEDROOM, TO_BEDROOM), ("Please clean the living room.", TO_KITCHEN)]) == UNKNOWN
    assert played(tmp_path / "none", [("Please clean the living room.", TO_BEDROOM)]) == UNKNOWN
    assert played(tmp_path / "both-in-one", [("Please clean the bedroom and the kitchen.", TO_BEDROOM)]) == UNKNOWN
    [q], _ = rows(case)
    run, _ = play(case, PathLog, kinds=KINDS, events=ak.JOB.events)
    assert run.failures == {} and run.answers[q["qid"]] == UNKNOWN and scores(case, run) == {q["qid"]: 0.0}


def test_path_log_answers_from_its_pool_when_it_has_one(tmp_path):
    """Contract (E §2): with a supported pool (five records from three scripts, each the corridor's region as the gold)
    and no room it can place, `path_log` answers the pool's best response, a point of the corridor, scoring 1; with no
    pool it says `unknown`. Regression caught: a pool ignored, or an answer without a pool invented."""
    def record_of(script: str) -> dict:
        rec = prior_record(question("adjacent", SLOTS), CORRIDOR_POINT, script=script)
        return {**rec, "gold": {**rec["gold"], "family": "adjacent", "credit": "region", "region": CORRIDOR,
                                "tolerances": {"erode_m": 0.2}}}

    pool = [record_of(f"s{n}") for n in (1, 1, 2, 2, 3)]
    walk = [("Please clean the living room.", TO_BEDROOM)]
    assert region_gold(played(tmp_path / "pool", walk, pool)) == 1.0
    assert played(tmp_path / "bare", walk) == UNKNOWN
