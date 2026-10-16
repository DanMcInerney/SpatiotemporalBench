"""The answers to `object_was`, `same_place` and `person_where` (`table_change`, `baselines_path_change`): the table
systems from the sightings of their tracks and the cells their mapper kept, `path_log` from its pool. Hand-worked
values; the oracle runs on windows of two tasks whose golds `test_questions_change_rows` and the question tests work out
by hand."""

from __future__ import annotations

from collections import Counter

import change_draw_kit as dk
from answers_kit import compiled_case
from baseline_kit import ROBOT, play, question, rows, scores, started, step
from change_kit import CHANGE_JOB, GAP_S, person, thing
from space_kit import EVENTS, vacuum_run
from stbench import table_answers
from stbench.baselines import FirstTable, LatestTable, MemoryTable, Oracle
from stbench.baselines_path import PathLog
from stbench.robots import ROBOTS
from stbench.table import Store

# Case ids of `two_tasks()` that between them ask `object_was` at both states and `same_place` as yes, moved and
# vacated.
CIDS = ("ba023a0f1b6e4cee", "9e47857239158f62", "beb46726dcb53092")
FINISHED, SET_OFF = "when you finished sweeping", "when you set off"


def two_tasks():
    """Two tasks (28 keyframes each), the bags seen at keyframes 3 and 31 (offset 3 of each, the camera at (2.13, 1.5)):
    E0001 red stays at (5.13, 1.5) (`same_place` yes); E0002 blue 3.13 -> 3.23 (yes); E0003 green 5.13 -> 5.73 (0.60 m:
    moved); E0004 cyan (3.13, 2.0) -> (6.13, 2.0) (3.00 m: moved, and asked by `object_was` at either state); E0006
    yellow at (1.0, 0.0) in task 0 only, whose place task 1's render sees empty (vacated); a person who is at (1, 2) in
    task 0 and (3, 2) in task 1."""
    s = vacuum_run(2)
    for eid, colour, a, b in (("E0001", "red", (5.13, 1.5), (5.13, 1.5)), ("E0002", "blue", (3.13, 1.5), (3.23, 1.5)),
                              ("E0003", "green", (5.13, 1.5), (5.73, 1.5)),
                              ("E0004", "cyan", (3.13, 2.0), (6.13, 2.0))):
        thing(s, eid, "bag", colour, [a, b], dk.SEEN)
    thing(s, "E0006", "bag", "yellow", [(1.0, 0.0), (7.0, 2.5)], [3])
    person(s, "P1", [(0.0, (1.0, 2.0)), (GAP_S, (3.0, 2.0))], [3, 31])
    return s


def test_the_oracle_and_the_memory_table_answer_the_families(tmp_path):
    """Contract: through `run_case` on `detections`, the oracle and `memory_table` score 1 on every question of three
    cases of a window of two tasks that between them ask `object_was` at its earlier and its later state,
    `person_where`, and `same_place` as yes (the red and blue bags), moved (the green and cyan ones) and vacated (the
    yellow bag, whose place the depth of task 1 shows empty). They read the sightings of the named task, the first place
    in the second task, and the cells of its steps. Regression caught: an answerer reading the latest sighting where the
    gold reads another, reading another task's sightings, a vacated no the mapper cannot certify (the oracle under the
    ceiling), or a yes/no whose threshold the gold's yes and no do not clear."""
    asked = set()
    for n, cid in enumerate(CIDS):
        case = compiled_case(tmp_path / f"case{n}", cid, two_tasks(), CHANGE_JOB, 0, 1)
        asked |= {(g["family"], g["answer_class"]) for g in rows(case)[1]}
        for cls in (Oracle, MemoryTable):
            run, _ = play(case, cls, events=EVENTS)
            assert run.status == "complete" and run.failures == {}
            assert set(scores(case, run).values()) == {1.0}, (cid, cls.__name__)
    assert asked >= {("object_was", "earlier"), ("object_was", "later"), ("person_where", None),
                     ("same_place", "yes"), ("same_place", "moved"), ("same_place", "vacated")}


def sighted(store: Store, k: int, phase: str, *dets: dict) -> None:
    """Step `k` (clock 08:00:<k>) of the vacuum in the phase, delivered with the detections."""
    store.add_step(step(k, 0.0, 0.0, phase=phase, detections=list(dets)))


def det(key: str, kind: str, attributes: dict, x: float, y: float) -> dict:
    return {"d": key, "kind": kind, "attributes": attributes, "bbox_px": [0, 0, 40, 40],
            "center_cam": [0.0, 0.0, 1.0], "center_map": [x, y, 0.1], "extent_m": [0.2, 0.2, 0.2]}


SEEN = {3: [("r", "bag", {"color": "red"}, 1.0, 1.0), ("b", "bag", {"color": "blue"}, 2.0, 2.0)],
        4: [("g", "bag", {"color": "green"}, 3.0, 3.0)], 5: [("p", "person", {}, 0.5, 0.5)],
        13: [("r", "bag", {"color": "red"}, 4.0, 1.0), ("b", "bag", {"color": "blue"}, 2.1, 2.0)],
        17: [("b", "bag", {"color": "blue"}, 2.4, 2.0)],
        15: [("p", "person", {}, 6.0, 6.0)]}   # keyframe -> (key, kind, attributes, x, y) of the detections there


def hand_store() -> Store:
    """Two tasks of ten steps (task 0 keyframes 0 to 9, task 1 10 to 19; the first step of each undocks, the others
    clean, so 'when you finished sweeping' is keyframe 9 and 19) and the detections of `SEEN`: a red bag at (1, 1) at
    keyframe 3 and (4, 1) at 13; a blue bag at (2, 2) at 3 and (2.1, 2) at 13 and (2.4, 2) at 17; a green bag at (3, 3)
    at 4 (task 0 only); a person at (0.5, 0.5) at 5 and (6, 6) at 15."""
    store = Store()
    for task, k0 in ((0, 0), (1, 10)):
        store.begin_task(task, f"2026-03-02T08:00:{k0:02d}", 0.0, "Clean.")
        for k in range(k0, k0 + 10):
            sighted(store, k, "undocking" if k == k0 else "cleaning", *(det(*d) for d in SEEN.get(k, [])))
    return store


def ask(store: Store, family: str, slots: dict, events=EVENTS) -> dict:
    return table_answers.answer(store, question(family, slots), {"events": events, "robot": ROBOT})


def was(colour: str, task: int) -> dict:
    return {"kind": "bag", "attributes": {"color": colour}, "task": task}


def same(colour: str) -> dict:
    return {"kind": "bag", "attributes": {"color": colour}, "tasks": [0, 1]}


def free_around(store: Store, k: int, x: float, y: float) -> None:
    """The fine cells (0.1 m) around (x, y) that step `k` observed free: what `Mapper.observe` keeps of a depth image
    that saw the floor there."""
    u, v = int(x / 0.1), int(y / 0.1)
    store.cells[k] = {(i, j): "free" for i in range(u - 3, u + 4) for j in range(v - 3, v + 4)}


def test_the_table_reads_the_named_task_and_the_first_place_in_the_second():
    """Contract: with the job's event table, `object_was` answers the latest sighting of the tracks the description
    names inside the task the question names (4 places), `unknown` when there is none; `same_place` is yes when the last
    place in the first task and the first place in the second are under 0.30 m apart (the middle of the gap between the
    gold's 0.15 and 0.45), no when further, no when the second task never sighted the thing and the cells of its steps
    show the first place empty, `unknown` otherwise. Hand-worked on `hand_store`: the red bag is (1, 1) in task 0 and
    (4, 1) in task 1; the green bag, seen in task 0 only, is (3, 3) in task 0 and `unknown` in task 1; `same_place` of
    the red bag is no (3 m), of the blue bag yes (0.1 m: the first place in task 1, (2.1, 2), and not its latest, (2.4,
    2)), of the green bag `unknown` until a step of task 1 observed the floor around (3, 3) free, then no; cells
    observed in task 0 do not count, and a step that saw an occupied cell there leaves it `unknown`. Regression caught:
    an answerer reading outside its named task, the wrong threshold, a vacated no from space seen in another task or
    with an occupied cell, or one invented without any cell."""
    store = hand_store()
    assert ask(store, "object_was", was("red", 0)) == {"type": "position", "xy": [1.0, 1.0]}
    assert ask(store, "object_was", was("red", 1)) == {"type": "position", "xy": [4.0, 1.0]}
    assert ask(store, "object_was", was("green", 0)) == {"type": "position", "xy": [3.0, 3.0]}
    assert ask(store, "object_was", was("green", 1)) == {"type": "unknown"}
    assert ask(store, "object_was", was("red", 2)) == {"type": "unknown"}   # a task the stream never had
    who = {"kind": "person", "attributes": {}, "event": FINISHED}
    assert ask(store, "person_where", {**who, "task": 0}) == {"type": "position", "xy": [0.5, 0.5]}
    assert ask(store, "person_where", {**who, "task": 1}) == {"type": "position", "xy": [6.0, 6.0]}
    assert ask(store, "same_place", same("red")) == {"type": "yesno", "value": False}
    assert ask(store, "same_place", same("blue")) == {"type": "yesno", "value": True}
    assert ask(store, "same_place", same("green")) == {"type": "unknown"}
    free_around(store, 2, 3.0, 3.0)   # seen in task 0: no evidence about task 1
    assert ask(store, "same_place", same("green")) == {"type": "unknown"}
    free_around(store, 12, 3.0, 3.0)
    assert ask(store, "same_place", same("green")) == {"type": "yesno", "value": False}
    store.cells[12][(30, 30)] = "occupied"
    assert ask(store, "same_place", same("green")) == {"type": "unknown"}


def stored(base_z: float = 0.0, height: float = 0.2) -> Store:
    """Two tasks of three steps (keyframes 0 to 2 and 10 to 12; the base on the floor at 0), a green bag seen at
    keyframe 1 standing on `base_z` with the box height `height`, and in task 1 the floor around its place (3, 3) seen
    free."""
    store = Store()
    for task, k0 in ((0, 0), (1, 10)):
        store.begin_task(task, f"2026-03-02T08:00:{k0:02d}", 0.0, "Clean.")
        for k in range(k0, k0 + 3):
            seen = [{**det("g", "bag", {"color": "green"}, 3.0, 3.0), "center_map": [3.0, 3.0, base_z + height / 2],
                     "extent_m": [0.2, 0.2, height]}] if k == 1 else []
            sighted(store, k, "cleaning", *seen)
    free_around(store, 11, 3.0, 3.0)
    return store


def test_a_vacated_no_needs_the_band_to_see_the_thing_gone():
    """Contract: as the gold, the table says the thing left only when the robot's band can show it gone
    (`table_change.can_vacate`): the robot drives (`ctx["robot"]`), the sighting stood on the floor of the base at that
    step (the bounding-box middle less half its height within `climb_m`) and the thing's top is above the band's floor.
    Hand-worked on a green bag whose place the cells of task 1 show free: the vacuum (climb 0.02 m) says no for a bag
    0.2 m high on the floor; unknown for a drone, for the same bag standing 0.8 m up (a shelf) and for a mat 0.01 m
    high; unknown with no robot in the context. Regression caught: a table that certifies a vacated no where the gold
    does not ask (the oracle then answers questions the compiler never asks), or one that does not where the gold
    does."""
    def answer(store, robot):
        ctx = {"events": EVENTS, "robot": robot}
        return table_answers.answer(store, question("same_place", same("green")), ctx)

    no, unknown = {"type": "yesno", "value": False}, {"type": "unknown"}
    assert answer(stored(), ROBOT) == no
    assert answer(stored(), ROBOTS["drone"]) == unknown and answer(stored(), None) == unknown
    assert answer(stored(base_z=0.8), ROBOT) == unknown and answer(stored(height=0.01), ROBOT) == unknown


def test_with_no_event_table_the_answerers_read_every_sighting():
    """Contract: a table that reads no event table (`latest_table`, `current_view`) keeps no time: `object_was` and
    `person_where` answer the latest sighting the store holds whatever task or moment the text names, and `same_place`
    reads that one sighting at both ends, so it answers yes and never no, even for a place its cells show empty.
    Hand-worked on `hand_store`: the red bag is (4, 1) for task 0 (the later place, not the one asked), the green bag
    (3, 3) for a task it was not seen in, the person (6, 6); `same_place` of the red bag, which moved 3 m, is yes, and
    of the green bag yes though step 12 saw its place empty. Regression caught: an answerer that reads the event table
    or the task when it has none (the time-blind notebook then looks time-aware, or raises), or a notebook that says
    no."""
    store = hand_store()
    assert ask(store, "object_was", was("red", 0), None) == {"type": "position", "xy": [4.0, 1.0]}
    assert ask(store, "object_was", was("green", 1), None) == {"type": "position", "xy": [3.0, 3.0]}
    who = {"kind": "person", "attributes": {}, "event": SET_OFF, "task": 0}
    assert ask(store, "person_where", who, None) == {"type": "position", "xy": [6.0, 6.0]}
    assert ask(store, "same_place", same("red"), None) == {"type": "yesno", "value": True}
    free_around(store, 12, 3.0, 3.0)
    assert ask(store, "same_place", same("green"), None) == {"type": "yesno", "value": True}
    assert ask(store, "object_was", was("pink", 0), None) == {"type": "unknown"}
    assert ask(Store(), "same_place", same("red"), None) == {"type": "unknown"}


class TimeBlind(Oracle):
    """The table that keeps each track's latest sighting and reads no event table (`latest_table`'s shape): the
    oracle's store with `reads_events` off, so no `events` option is needed or passed and every answerer gets
    `ctx["events"] is None`."""

    reads_events = False


def test_a_table_with_no_event_table_runs_a_case_without_raising(tmp_path):
    """Contract: a table system that reads no event table answers every question of the three families of a whole case
    without raising, so `check` can run it: `object_was` gives the thing's latest place, `person_where` the person's
    latest place and `same_place` yes whatever the golds say. Hand-worked on the three cases of `two_tasks()` (run with
    no `events` option): the cyan bag's latest place is (6.13, 2.0) and the person's (3, 2); every `same_place` answer
    is yes, right for the red and blue bags and wrong for the green, cyan and yellow ones. Regression caught: an
    answerer that reads `ctx["events"]` or the task when there is none (a raise makes `stbench check` refuse the run),
    or a time-blind table that looks time-aware."""
    seen = set()
    for n, cid in enumerate(CIDS):
        case = compiled_case(tmp_path / f"case{n}", cid, two_tasks(), CHANGE_JOB, 0, 1)
        run, _ = play(case, TimeBlind)
        assert run.status == "complete" and run.failures == {}
        asked = {q["qid"]: q for q in rows(case)[0]}
        for qid, answer in run.answers.items():
            template, slots = asked[qid]["template"], asked[qid]["slots"]
            if template == "same_place":
                assert answer == {"type": "yesno", "value": True}
            elif template == "object_was":
                assert slots["attributes"] == {"color": "cyan"} and answer == {"type": "position", "xy": [6.13, 2.0]}
            else:
                assert template == "person_where" and answer == {"type": "position", "xy": [3.0, 2.0]}
            seen.add(template)
    assert seen == {"object_was", "same_place", "person_where"}


def test_neither_time_blind_notebook_beats_the_floor_on_a_case_of_moved_things(tmp_path):
    """Contract (DECISIONS, "0.5.0's code review"): through `run_case` on `detections`, over a compiled case whose only
    family is `object_was` asked of many moved things, the oracle scores 1 on every question while `first_table` is
    right on exactly the earlier-state rows and `latest_table` on exactly the later-state rows, each wrong (penalised
    -1) on the other half, so the two score +1 and -1 on every row and their means are opposite. Hand-worked on twelve
    bags that move 6.5 m in the gap after task 0 (seen at keyframes 3 and 31; the last rung is under 2.9 m): the case
    asks twelve questions, one a draw, the later ones for the draws the hand coin lands on, evidence keyframe 3 for an
    earlier row and 31 for a later one. Regression caught: a family that a one-sided notebook answers (every moved thing
    asked at its earlier state gives `first_table` the whole family, so `time_needed` fails by construction), or
    notebooks that cannot run the case."""
    cid, job = CIDS[0], dk.only("object_was")
    case = compiled_case(tmp_path / "case", cid, dk.many_moved(), job, 0, 1)
    golds = {g["qid"]: g for g in rows(case)[1]}
    later = sum(dk.hand_coin(cid, "object_was", f"draw:{j}") for j in range(12))
    assert Counter(tuple(g["evidence_k"]) for g in golds.values()) == {(3,): 12 - later, (31,): later}
    seen = {}
    for name, cls, opts in (("oracle", Oracle, {"events": EVENTS}), ("first", FirstTable, {}),
                            ("latest", LatestTable, {})):
        run, _ = play(case, cls, **opts)
        assert run.status == "complete" and run.failures == {}
        seen[name] = scores(case, run)
    for qid, g in golds.items():
        earlier = tuple(g["evidence_k"]) == (3,)
        assert (seen["oracle"][qid], seen["first"][qid], seen["latest"][qid]) == (
            (1.0, 1.0, -1.0) if earlier else (1.0, -1.0, 1.0)), g["evidence_k"]
    assert sum(seen["first"].values()) == -sum(seen["latest"].values()) == 12 - 2 * later


# -- path_log -------------------------------------------------------------------------------------------------------
ASK = "2026-03-02T08:00:05"


def prior_of(family: str, slots: dict, golds: list[dict], credit: str, tolerances: dict) -> list[dict]:
    """One record of `family` asked with `slots` from each of the scripts that held the golds, as `suite_priors` writes
    them (the same slots, so the exact question is the first level that holds enough of them)."""
    out = []
    for n, value in enumerate(golds):
        gold = {"qid": f"g{n}", "family": family, "gold": value, "slice": "answerable", "answer_class": None,
                "range_m": None, "credit": credit, "region": None, "tolerances": tolerances,
                "metric": "xy" if value["type"] == "position" else None, "scope": "memory", "reference": None,
                "critical": True, "subjects": [], "evidence_k": [1], "entities": []}
        out.append({"script": f"s{n}", "template": family, "slots": slots, "when": "end", "clock": ASK,
                    "phase": "cleaning", "base": (0.0, 0.0), "yaw": 0.0, "rooms": (), "nm": None, "siblings": [],
                    "gold": gold})
    return out


def path_answer(tmp_path, prior: list[dict], family: str, slots: dict) -> dict:
    system = started(PathLog, tmp_path, map=None, events=EVENTS, prior=prior)
    system.task(0, "2026-03-02T08:00:00", 0.0, "Its current assignment is to clean the kitchen.")
    system.observe([step(k, 0.5, 1.9, phase="undocking" if k == 0 else "cleaning") for k in range(6)])
    return system.ask("end", {"k": 5, "clock": ASK}, [question(family, slots)])["q1"]


def test_path_log_learns_the_change_families_from_their_own_records(tmp_path):
    """Contract: `path_log` answers the three families with no picture, from the pool of other scripts' records of the
    question's own template: `object_was` and `person_where` as it answers `object_where` (the pool's best place),
    `same_place` the pool's best yes/no, and `unknown` when the template has no pool (the records of `object_where`
    are not `object_was`'s). Hand-worked: five scripts that each asked the exact question and had the gold (2, 1), yes
    or no are answered with it; the same question with no records is `unknown`, and so is `object_was` with five
    records under the template `object_where`. Regression caught: a family with no `path_log` answerer, one that reads
    another template's records, or one that answers from no pool."""
    here = {"type": "position", "xy": [2.0, 1.0]}
    ladder = {"ladder": [0.25, 0.5, 1.0, 2.0]}
    yes, no, unknown = {"type": "yesno", "value": True}, {"type": "yesno", "value": False}, {"type": "unknown"}
    pair = {"kind": "bag", "attributes": {"color": "red"}, "tasks": [0, 1]}
    who = {"kind": "person", "attributes": {}, "event": FINISHED, "task": 0}
    for n, (family, slots, gold, credit, tolerances) in enumerate((
            ("object_was", was("red", 0), here, "ladder", ladder), ("person_where", who, here, "ladder", ladder),
            ("same_place", pair, yes, "exact", {}), ("same_place", pair, no, "exact", {}))):
        prior = prior_of(family, slots, [gold] * 5, credit, tolerances)
        assert path_answer(tmp_path / f"{n}-pool", prior, family, slots) == gold, (family, gold)
        assert path_answer(tmp_path / f"{n}-none", [], family, slots) == unknown, family
    other = prior_of("object_where", was("red", 0), [here] * 5, "ladder", ladder)
    assert path_answer(tmp_path / "other", other, "object_was", was("red", 0)) == unknown
