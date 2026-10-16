"""`adjacent` (questions_adjacent), by hand on `adjacent_kit`'s six-room apartment: the vacuum walks from room to room,
the end look window is its last eight steps, and every room is observed (hand free-space states) unless a test says not.
Like both real jobs the walks start and end at the robot's dock, here in the hall, a room the questions are not
about."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import combinations

import pytest

import adjacent_kit as ak
import family_kit as fk
from space_kit import JOB as SPACE_JOB
from space_kit import record as space_record
from stbench import map_files, mapspec, questionnaire, questions, score
from stbench.adjacency import connector
from stbench.geometry import quat_z
from stbench.job_registry import Relevance
from stbench.questions_adjacent import moments
from stbench.questions_text import FAMILIES
from stbench.questions_world import World
from stbench.records import check_family
from stbench.robots import ROBOTS
from stbench.timeline import Entity, Placement, State

EARLY = [ak.HALL, ak.COR, ak.D2, ak.KIT, ak.D2, ak.COR, ak.COR_E, ak.BED, ak.COR_E]  # hall, corridor, kitchen, ...
KITCHEN_ONLY = EARLY[:6]  # hall, corridor, kitchen, corridor: the first half of EARLY, the bedroom never visited
HOLD = [ak.HALL] * 8
BEDROOM_KITCHEN = "bedroom|kitchen"


@pytest.fixture
def held_out(monkeypatch):
    """The kit's map is a held-out layout for this test (a held-out map itself is never a fixture)."""
    monkeypatch.setattr(map_files, "HELD_OUT", (*map_files.HELD_OUT, ak.MAP_NAME))


def keys(w) -> list[str]:
    return [m.key for m in moments(w)]


def the_cand(w):
    """The window's one candidate, in its one moment."""
    [m] = moments(w)
    assert m.unknown == []
    [c] = m.answerable
    return c


def test_the_moment_by_hand(held_out):
    """Contract: on the kit, a walk from the hall to the corridor, kitchen, corridor, bedroom and corridor (the
    crossings of the kitchen and the bedroom first made at keyframes 3 and 7) and eight steps back in the hall asks one
    question, about the bedroom and the kitchen (the hall's pairs are left out, the walk having started there): the
    room that connects them is the corridor (R2), the gold is a point of it (its centre, (4.6, 1.5), the first point at
    least 0.2 m inside), credit `region` on its polygon eroded 0.2 m, scope `memory`, reference `other_case`, the
    evidence the two first crossings (3, 7), the subjects the three rooms (R5, R2, R3), not critical when the job's
    relevance names none of them and critical when it lists the corridor; asked by the questionnaire as the job's end
    family and stamped into valid question and gold rows. Regression caught: a gold that is the wrong room (A or B
    for R), a point on the corridor's wall, evidence or subjects that name the wrong keyframes or rooms, or a family
    the registry cannot compile."""
    w = ak.world(ak.script(EARLY + HOLD))
    assert w.tl.episode.map == ak.MAP_NAME and w.end_window == range(9, 17)
    c = the_cand(w)
    assert c.family == "adjacent" and c.slots == {"space": "room", "a": "bedroom", "b": "kitchen"}
    assert c.gold == {"type": "position", "xy": [4.6, 1.5]} and c.slice == "answerable" and c.entities == ()
    assert c.region == [[3.1, 0.0], [6.1, 0.0], [6.1, 3.0], [3.1, 3.0]] and c.tolerances == {"erode_m": 0.2}
    assert c.evidence_k == (3, 7) and c.subjects == ("r:R5", "r:R2", "r:R3") and not c.critical
    assert c.decision_k is None and keys(w) == [BEDROOM_KITCHEN]
    assert FAMILIES["adjacent"].render(c.slots) == "Give a point in the room that connects the bedroom to the kitchen."
    listed = replace(ak.JOB, relevance=lambda plan: Relevance(frozenset(), frozenset({"corridor"}), frozenset()))
    assert the_cand(ak.world(ak.script(EARLY + HOLD), listed)).critical
    spec = FAMILIES["adjacent"]
    assert check_family(spec) == [] and (spec.credit, spec.reference, spec.scope) == ("region", "other_case", "memory")
    assert [x.key for x in questionnaire.end(w)] == [c.key]
    (q,), (g,) = questions.records([c])
    assert q["text"] == "Give a point in the room that connects the bedroom to the kitchen."
    assert (g["credit"], g["scope"], g["reference"], g["region"]) == ("region", "memory", "other_case", c.region)


def test_adjacent_is_asked_only_on_held_out_maps(monkeypatch):
    """Contract (design §8.3, D1): the family asks nothing of a layout outside `map_files.HELD_OUT`, however clearly
    the path crossed its rooms, and on a held-out layout asks every pair the hand lists. On the kit, a tour that
    starts in the hall, visits the corridor, kitchen, study and bedroom, goes back to the hall and ends with eight steps
    there, patched into `HELD_OUT`: bedroom|kitchen (evidence 4, 12), through the corridor, and corridor|study (4, 6),
    through the kitchen; the hall's pairs are not asked, the hall being where the robot started, hall|study (a wall
    between them, no connector), kitchen|study and the joined pairs have no connector, and the bathroom none.
    Regression caught: a layout fact asked of a public layout, which `blind` learns from the other cases, or a pair
    listed that has no connector."""
    tour = [ak.HALL, ak.D1, ak.COR, ak.D2, ak.KIT, ak.ARCH, ak.STUDY, ak.ARCH, ak.KIT, ak.D2, ak.COR, ak.COR_E, ak.BED,
            ak.COR_E, ak.COR, ak.D1, ak.HALL] + HOLD
    w = ak.world(ak.script(tour))
    assert moments(w) == [] and questionnaire.end(w) == []
    monkeypatch.setattr(map_files, "HELD_OUT", (*map_files.HELD_OUT, ak.MAP_NAME))
    found = {m.key: m.answerable[0] for m in moments(w)}
    assert sorted(found) == ["bedroom|kitchen", "corridor|study"]
    assert {key: (c.evidence_k, c.subjects) for key, c in found.items()} == {
        "bedroom|kitchen": ((4, 12), ("r:R5", "r:R2", "r:R3")), "corridor|study": ((4, 6), ("r:R2", "r:R3", "r:R4"))}
    assert {key: tuple(c.gold["xy"]) for key, c in found.items()} == {"bedroom|kitchen": (4.6, 1.5),
                                                                       "corridor|study": (4.55, 4.6)}


def test_the_end_look_window_never_holds_the_connecting_room(held_out):
    """Contract (rules 4 and 5): a question is asked only when no position of the end look window (the last eight steps,
    all `current_view` holds, and where a robot that docks at the end of a task is when it is asked) lies in the
    connecting room, so neither the end look window's own path nor "answer where I am" scores. On the kit, with both
    boundaries crossed early (corridor-kitchen, corridor-bedroom): eight steps back in the hall ask the bedroom|kitchen
    question; both boundaries crossed again in the end look window, eight steps in the corridor, a look window that
    reaches the corridor with its last step only and one that has it with its first step only (seven in the hall after
    it) ask nothing; the corridor visited just before the look window asks it; and a walk that crossed the kitchen's
    boundary only, early, asks nothing. Regression caught: a connecting room the robot stands in, or stood in a step
    ago, whose centre "where I am" would score (both jobs end at their dock), or a question asked with a boundary never
    crossed."""
    assert keys(ak.world(ak.script(EARLY + HOLD))) == [BEDROOM_KITCHEN]
    assert keys(ak.world(ak.script(EARLY + [ak.COR] + HOLD))) == [BEDROOM_KITCHEN]
    assert keys(ak.world(ak.script(EARLY + EARLY))) == []
    assert keys(ak.world(ak.script(EARLY + [ak.COR] * 8))) == []
    assert keys(ak.world(ak.script(EARLY + HOLD[:-1] + [ak.COR]))) == []
    assert keys(ak.world(ak.script(EARLY + [ak.COR] + HOLD[:-1]))) == []
    assert keys(ak.world(ak.script(KITCHEN_ONLY + HOLD))) == []


def test_the_connecting_room_is_not_where_the_window_starts(held_out):
    """Contract (rule 5): a question is asked only when the connecting room does not hold the window's first delivered
    position, the robot's dock at the start of a task, so "answer where I started" scores nothing either. On the kit,
    the walk from the hall asks the bedroom|kitchen question; the same walk started one step earlier in the corridor
    asks nothing; and as the second task of a script whose first task stays in the corridor it asks the question in
    the window that starts there and nothing in the window that starts with the first task. Regression caught: a
    connecting room the robot started in (the reviewer's blocker), or the start read from the script's first step
    instead of the window's."""
    assert keys(ak.world(ak.script(EARLY + HOLD))) == [BEDROOM_KITCHEN]
    assert keys(ak.world(ak.script([ak.COR] + EARLY + HOLD))) == []
    s = ak.script([ak.COR] * 3, EARLY + HOLD)
    assert keys(ak.world(s, first=1)) == [BEDROOM_KITCHEN] and keys(ak.world(s, first=0)) == []


def test_both_rooms_must_be_toured(held_out):
    """Contract: both rooms of the pair were toured, at least half their inside fine cells observed in any state over
    the window. On the kit, the kitchen (870 cells) 40% observed asks nothing, 50% (435) and 60% ask the question,
    and so does the bedroom (900 cells) 60% observed as obstacles only (any state counts); the bedroom 40% asks
    nothing. Regression caught: a room named that the camera never showed, or a room of which only free cells
    count."""
    walk = ak.script(EARLY + HOLD)
    assert keys(ak.world(walk, observed={ak.KIT_ID: (0.4, "free")})) == []
    assert keys(ak.world(walk, observed={ak.KIT_ID: (0.5, "free")})) == [BEDROOM_KITCHEN]
    assert keys(ak.world(walk, observed={ak.KIT_ID: (0.6, "free")})) == [BEDROOM_KITCHEN]
    assert keys(ak.world(walk, observed={ak.BED_ID: (0.6, "occupied")})) == [BEDROOM_KITCHEN]
    assert keys(ak.world(walk, observed={ak.BED_ID: (0.4, "occupied")})) == []
    assert keys(ak.world(walk, observed={ak.KIT_ID: (0.0, "free")})) == []


def test_a_room_the_robot_can_place_without_the_camera_is_not_asked(held_out):
    """Contract (rule 5): a room is not asked when a system places it without the camera. On the kit's walk
    (asked as it stands): a task whose brief says "clean the kitchen" leaves no question about the kitchen, whichever
    task of a two-task window says it; a job whose relevance lists the bedroom, none about the bedroom, whichever task
    of a two-task window it lists it for (a brief that names no room, so the relevance alone decides); a walk that
    starts in the bedroom, none about it (the hall's pair is asked), and the same walk started one step earlier in the
    hall asks the bedroom's, the hall's pairs now being the ones left out. Regression caught: a question the briefs,
    the job's relevance or the pose log (the start region) answer without a sensor, or only the window's last task's
    brief or relevance read."""
    walk = EARLY + HOLD
    assert keys(ak.world(ak.script(walk))) == [BEDROOM_KITCHEN]
    assert keys(ak.world(ak.script(walk, rooms=("kitchen",)))) == []
    assert keys(ak.world(ak.script(EARLY, HOLD, rooms=("living room", "living room")))) == [BEDROOM_KITCHEN]
    assert keys(ak.world(ak.script(EARLY, HOLD, rooms=("kitchen", "living room")))) == []
    assert keys(ak.world(ak.script(EARLY, HOLD, rooms=("living room", "kitchen")))) == []
    listed = replace(ak.JOB, relevance=lambda plan: Relevance(frozenset(), frozenset({"bedroom"}), frozenset()))
    assert keys(ak.world(ak.script(walk), listed)) == []
    by_task = replace(ak.JOB, assignments={"clean_room": "Its current assignment is a tour."},
                      relevance=lambda plan: Relevance(frozenset(), frozenset({plan.params["room"]}), frozenset()))
    for rooms, asked in ((("living room", "living room"), True), (("bedroom", "living room"), False),
                         (("living room", "bedroom"), False)):
        assert keys(ak.world(ak.script(EARLY, HOLD, rooms=rooms), by_task)) == ([BEDROOM_KITCHEN] if asked else [])
    from_bedroom = [ak.BED, ak.COR_E, ak.COR, ak.D2, ak.KIT, ak.D2, ak.COR, ak.COR] + HOLD
    assert keys(ak.world(ak.script(from_bedroom))) == ["hall|kitchen"]   # the hall's pair, not the bedroom's
    assert keys(ak.world(ak.script([ak.HALL, *from_bedroom]))) == [BEDROOM_KITCHEN]


def test_a_kind_the_briefs_paragraph_names_is_not_asked(held_out):
    """Contract (rule 5): the briefs that count are the texts `brief.compose` writes, the paragraph of the window's
    first task included, as `path_log` and `suite_priors` read them, not the assignment templates alone. On the kit's
    walk, the bedroom renamed "floor", a word of the paragraph's place ("one floor of a furnished home"), is unique in
    the layout and crossed and toured like the bedroom, and no question names it; renamed "attic", a word no text
    holds, the question is asked. Regression caught: the gold reading the assignment templates, so that a kind the
    brief a system is sent names alone is asked and `path_log` places it."""
    m = ak.kit_map()

    def renamed(kind: str):
        return replace(m, regions=tuple(replace(r, kind=kind) if r.id == ak.BED_ID else r for r in m.regions))

    walk = ak.script(EARLY + HOLD)
    assert keys(ak.world(walk, kit=renamed("floor"))) == []
    assert keys(ak.world(walk, kit=renamed("attic"))) == ["attic|kitchen"]


def test_evidence_is_in_script_keyframes(held_out):
    """Contract: evidence names script keyframes, as every private record does, not steps of the window: the kit's
    walk as the second task of a script, after a first task of five steps, has its two first crossings at script
    keyframes 5 + 3 and 5 + 7. Regression caught: evidence counted from the window's first step."""
    w = ak.world(ak.script([ak.HALL] * 5, EARLY + HOLD), first=1)
    assert w.rec.k0 == 5 and the_cand(w).evidence_k == (8, 12)


def test_the_gold_scores_by_region(held_out):
    """Contract: the gold row scores through `score.row` by the connector's polygon eroded 0.2 m: the gold's own
    point and one 0.3 m inside the corridor's west wall score 1; one 0.1 m inside it, one in the bedroom and one in
    the kitchen score 0. Regression caught: a gold region that is the wrong room, or one not eroded (a point on the
    wall scored right)."""
    (_,), (g,) = questions.records([the_cand(ak.world(ak.script(EARLY + HOLD)))])
    info = {"case": "c0", "script": "s0", "layout": ak.MAP_NAME, "robot": "vacuum", "job": "vacuum/1",
            "length": "short"}

    def credit(x, y):
        return score.row({"type": "position", "xy": [x, y]}, g, when="end", info=info)["s"]

    assert [credit(*g["gold"]["xy"]), credit(3.4, 1.5), credit(5.8, 0.4)] == [1.0, 1.0, 1.0]
    assert [credit(3.2, 1.5), credit(3.1, 1.5), credit(*ak.BED), credit(*ak.KIT)] == [0.0, 0.0, 0.0, 0.0]


def test_only_unique_kinds_are_named(held_out):
    """Contract: a room is named by its kind only where the kind is unique in the layout. On the kit, with the
    bathroom also of kind "bedroom", no question names a bedroom (neither room is in a pair), and with the bathroom
    its own kind the bedroom|kitchen question is asked. Regression caught: "the bedroom" asked where there are two."""
    m = ak.kit_map()
    twin = replace(m, regions=tuple(replace(r, kind="bedroom") if r.id == ak.BATH_ID else r for r in m.regions))
    walk = ak.script(EARLY + HOLD)
    assert keys(ak.world(walk, kit=twin)) == []
    assert keys(ak.world(walk, kit=m)) == [BEDROOM_KITCHEN]


def test_the_space_word_follows_the_domain(held_out):
    """Contract: the question says "room" for a home job and "area" for the others (a warehouse's or a city's
    open-plan places). Regression caught: "room" asked of an open-plan warehouse area."""
    walk = ak.script(EARLY + HOLD)
    said = {domain: FAMILIES["adjacent"].render(the_cand(ak.world(walk, replace(ak.JOB, domain=domain))).slots)
            for domain in ("home", "warehouse", "delivery")}
    assert said == {"home": "Give a point in the room that connects the bedroom to the kitchen.",
                    "warehouse": "Give a point in the area that connects the bedroom to the kitchen.",
                    "delivery": "Give a point in the area that connects the bedroom to the kitchen."}


def test_an_unseen_change_changes_no_gold(held_out):
    """Contract (truth rule 2): the gold reads the public map, the delivered path and the observed cells, never the
    world behind them. On the kit's walk, a bag that moves across the apartment and a door that shuts, neither in any
    frame, leave the moments and every gold as they were. Regression caught: a gold or a moment that reads the
    timeline's entities or states."""
    s = ak.script(EARLY + HOLD)
    inp, rec = ak.inputs(s), space_record(s, a=0, b=0)
    bag = Entity("E0101", "bag", {"color": "red"}, "carried", (0.3, 0.3, 0.3))
    door = Entity("E0102", "door", {"between": ["hall", "kitchen"]}, "fixed", (0.9, 0.04, 2.0))
    still = quat_z(0.0)
    changed = replace(inp.tl, entities=[*inp.tl.entities, bag, door], states=[State("E0102", "open", 0.0, 6.0, True)],
                      placements=[*inp.tl.placements, Placement("E0101", 0.0, 6.0, (5.0, 4.0, 0.0), still, None),
                                  Placement("E0101", 6.0, None, (7.5, 3.0, 0.0), still, None),
                                  Placement("E0102", 0.0, None, (3.05, 1.5, 0.0), still, None)])
    before, after = moments(World(inp, rec)), moments(World(replace(inp, tl=changed), rec))
    assert [m.key for m in before] == [BEDROOM_KITCHEN] and after == before


def test_a_flier_is_never_asked(held_out):
    """Contract: the table's `Mapper` keeps no robot band for a robot that flies, so no table system observes a cell of
    any room and none could answer: `adjacent` asks a flier nothing, whatever cells the gold itself is handed. The kit's
    walk, every room observed in the hand states, asks the bedroom|kitchen question of the vacuum and nothing of the
    same window with the drone as the robot. Regression caught: a question asked of a flier that no table system can
    answer, which would hold the family's ceiling gate at 0 there."""
    s = ak.script(EARLY + HOLD)
    assert keys(ak.world(s)) == [BEDROOM_KITCHEN]
    flier = replace(ak.inputs(s), robot=ROBOTS["drone"])
    assert moments(World(flier, space_record(s, a=0, b=0))) == []


def test_adding_adjacent_to_a_jobs_ends_changes_no_public_questionnaire():
    """Contract (M4 integration, the merge of C and E): `adjacent` has no moment on a public layout and a family with no
    moments takes no slot, so a job that lists it in its `ends` asks the same questions of a public layout with it as
    without. One task of the synthetic vacuum over the public apartment's name, for a job that ends `object_where` and
    `when_seen`, compiled in memory with and without `adjacent` appended (the question ids are drawn afresh each time,
    so they are left out): the same questions and gold rows, more than a few. Regression caught: a family with no
    moments taking a slot, or moving the draw of the others, on a public layout."""
    s = fk.things_by_task(fk.multi_run(1), 1)
    job = replace(SPACE_JOB, ends=("object_where", "when_seen"))

    def compiled(of) -> list[list[dict]]:
        return [[{k: v for k, v in row.items() if k != "qid"} for row in rows] for rows in fk.compiled(s, 0, 0, job=of)]

    without = compiled(job)
    assert len(without[0]) >= 5 and compiled(replace(job, ends=(*job.ends, "adjacent"))) == without


def test_every_held_out_map_present_has_two_askable_pairs():
    """Contract: a held-out layout holds at least two pairs of rooms, each of a kind unique in the layout, with one
    connecting region, so the held-out family has something to ask there. Read over each held-out map in
    `STBENCH_MAPS` (skipped when none is there). Regression caught: a held-out layout on which the family is empty."""
    present = [name for name in map_files.HELD_OUT if name in map_files.private_maps()]
    if not present:
        pytest.skip("no held-out map in STBENCH_MAPS")
    for name in present:
        m = mapspec.load_map(name)
        kinds = Counter(r.kind for r in m.regions)
        unique = [r for r in m.regions if kinds[r.kind] == 1]
        pairs = [(a.id, b.id) for a, b in combinations(unique, 2) if connector(m, a.id, b.id) is not None]
        assert len(pairs) >= 2, f"{name}: {len(pairs)} askable pairs"
