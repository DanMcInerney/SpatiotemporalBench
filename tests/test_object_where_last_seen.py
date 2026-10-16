"""The scripted systems answer `object_where` with no event ("Where did you last see ..."): the memory table from every
sighting it holds, `path_log` from every step it logged and, as candidate places, the map pieces its subgoals name.
Hand-worked values on `test_table.mug_store` (steps 0-5; a red mug at (1, 2) at step 1 and (3, 2) at 4, a blue one at
(5, 5) at 2) and on `apartment_a`."""

from __future__ import annotations

from baseline_kit import EVENTS, at, question, record, started, step
from stbench import mapspec, table_answers
from stbench.baselines_path import PathLog, named_pieces, piece_place
from test_path_log import BED, KITCHEN, START, ask_where, room_prior, scripted_steps
from test_table import EVENTS as MUG_EVENTS
from test_table import ROBOT, mug_store

LAST_SEEN = question("object_where", {"kind": "box", "attributes": {"color": "brown"}})
APARTMENT = mapspec.load_map("apartment_a")


def seen(color: str, ctx_events) -> dict:
    """The table's answer to 'where did you last see the {color} mug?' with this event table (or none)."""
    q = {"template": "object_where", "slots": {"kind": "mug", "attributes": {"color": color}}}
    return table_answers.answer(mug_store(), q, {"now": None, "latest": [], "robot": ROBOT, "events": ctx_events})


def test_the_table_answers_a_question_with_no_event_from_every_sighting():
    """Contract: an `object_where` with no `event` slot is answered from the latest sighting of the description in the
    whole store, to the window's end, with or without an event table (the table systems and `current_view` alike), and
    `unknown` for a description nothing matches; it names no wording, so a job's events are not consulted and none
    raises. Hand case: the red mug is last at (3, 2) (step 4, the 2.00004 rounded away), the blue one at (5, 5), a green
    one nowhere; the answer is the same with the job's events as with none. Regression caught: a question with no event
    raising a KeyError in the answerer (the slot read as required), or answered from the first sighting."""
    for events in (MUG_EVENTS, None):
        assert seen("red", events) == {"type": "position", "xy": [3.0, 2.0]}
        assert seen("blue", events) == {"type": "position", "xy": [5.0, 5.0]}
        assert seen("green", events) == {"type": "unknown"}


def last_seen_prior():
    """`test_path_log.room_prior` for the question with no event: ten other scripts, in five that cleaned the kitchen
    the brown box was last seen at KITCHEN, in five that cleaned the bedroom at BED."""
    return [record(LAST_SEEN, at(*place), script=f"{room}{i}", rooms=(room,))
            for room, place in (("kitchen", KITCHEN), ("bedroom", BED)) for i in range(5)]


def ask_last_seen(tmp_path, steps):
    """`path_log` on the apartment, told to clean the kitchen, fed `steps`, asked where it last saw the brown box."""
    system = started(PathLog, tmp_path, map=APARTMENT, events=EVENTS, prior=last_seen_prior())
    system.task(0, "2026-03-02T08:00:00", 0.0, "Its current assignment is to clean the kitchen.")
    system.observe(steps)
    return system.ask("end", {"k": steps[-1]["k"], "clock": ""}, [LAST_SEEN])["q1"]


def test_path_log_reads_every_logged_step_when_the_question_names_no_event(tmp_path):
    """Contract: for a question with no event `path_log` offers the places its camera faced at any logged step, to the
    window's end, and learns the answer from the prior's golds of the same kind and rooms; an anchored question (an
    event) still reads the steps up to the event. Hand case (`test_path_log`'s kitchen prior: the brown box at (2, 2) in
    five kitchen scripts): a robot that faces away from the places until the
    event, step 3, and toward them after it (steps 4 and 5) answers `unknown` to 'where was the brown box when you
    started cleaning', which it could only have faced afterwards, and (2, 2) to 'where did you last see the brown
    box', which reads to step 5; one that faces them from the start answers (2, 2) to both, and one that never does
    `unknown` to both. Regression caught: the end of the window read as no step at all (the question with no event
    answered `unknown` always), or an event assumed."""
    opts = {"map": APARTMENT, "events": EVENTS}
    late = scripted_steps(0.5, 1.9, 180.0, after=(0.5, 1.9, 0.0))
    assert ask_where(PathLog, tmp_path / "anchored", late, room_prior(), "kitchen", **opts) == {"type": "unknown"}
    assert ask_last_seen(tmp_path / "late", late) == at(2.0, 2.0)
    assert ask_last_seen(tmp_path / "facing", scripted_steps(0.5, 1.9, 0.0)) == at(2.0, 2.0)
    assert ask_last_seen(tmp_path / "away", scripted_steps(0.5, 1.9, 180.0)) == {"type": "unknown"}
    assert START in EVENTS


def test_the_pieces_a_subgoal_names_are_found_by_their_names_on_the_map():
    """Contract: `named_pieces` finds the map pieces whose names occur in the subgoal texts as whole words, in any case,
    in map order; a name that repeats on the map is followed by an ordinal, the n-th piece of that name in map order,
    and a name with no ordinal names every piece of it; an ordinal past the last, a part of a word and a name the map
    lacks name none. By hand on `apartment_a` (kitchen counters F02, F04 and F06, a dining table F07): 'carrying the
    mug from the Dining Table to the kitchen counter 2' names F04 and F07 (map order puts F04 first); 'the kitchen
    counter' alone F02, F04 and F06; 'kitchen counter 4' none; 'dining tables' none; 'a sofabed' none. Regression
    caught: a subgoal's piece missed for its capital letters or its ordinal, or every counter taken for one the text
    singles out."""
    def ids(*texts):
        return [f.id for f in named_pieces(APARTMENT, texts)]

    assert ids("carrying the mug from the Dining Table to the kitchen counter 2") == ["F04", "F07"]
    assert ids("waiting at the kitchen counter") == ["F02", "F04", "F06"]
    assert ids("the kitchen counter 1", "the kitchen counter 3") == ["F02", "F06"]
    assert ids("the kitchen counter 4", "the kitchen counter 0", "dining tables", "a sofabed", "") == []


def test_a_pieces_place_is_the_middle_of_its_largest_surface():
    """Contract: `piece_place` is the centre of the piece's largest surface at that surface's height, and for a piece
    with no surface the centre of its footprint at the piece's height. By hand on `apartment_a`: the dining table F07
    (2.1, 1.9) at 0.75 m, the bookshelf F13 on the first of its two equal surfaces (0.4 m), the fridge F01 (no surface)
    at (0.4, 0.4) and 1.8 m. Regression caught: a place at the piece's base, which no camera would face when
    the thing is on its top."""
    by_id = {f.id: f for f in APARTMENT.furniture}
    assert piece_place(by_id["F07"]) == (2.1, 1.9, 0.75)
    assert piece_place(by_id["F01"]) == (0.4, 0.4, 1.8)
    x, y, z = piece_place(by_id["F13"])
    assert (round(x, 3), round(y, 3), z) == (0.9, 6.0, 0.4)


def test_path_log_offers_the_pieces_its_subgoals_name_as_candidate_places(tmp_path):
    """Contract: the places `path_log` may answer with include the map pieces named by the subgoals of its logged steps,
    those its camera faced at a logged step up to the question's moment; a subgoal after the moment, a piece behind the
    camera and a step with no subgoal add nothing. By hand from (0.5, 1.9) facing +x (camera 0.17 m ahead): with no
    subgoal a kind with no receptacle offers nothing; with 'going to the dining table' on step 1 the dining table's
    top (2.1, 1.9, 0.75) is offered, by step 3 and by step 1 but not by step 0; the same text only on step 5 is not
    offered by step 3; 'going to the kitchen counter 3' (F06 at (0.35, 1.4), behind the camera) adds nothing.
    Regression caught: a subgoal that names where the thing is read as nothing (the shortcut then weaker than a log that
    says it), or a piece offered that the camera never faced."""
    def log(steps):
        system = started(PathLog, tmp_path / f"s{len(list(tmp_path.iterdir()))}", map=APARTMENT, events=EVENTS)
        system.observe(steps)
        return system

    def steps(subgoals):
        return [step(k, 0.5, 1.9, 0.0, phase="cleaning", **({"subgoal": subgoals[k]} if k in subgoals else {}))
                for k in range(6)]

    text = "going to the dining table"
    assert log(steps({})).faced_places("gizmo", [], until=3) == []
    assert log(steps({1: text})).faced_places("gizmo", [], until=3) == [(2.1, 1.9, 0.75)]
    assert log(steps({1: text})).faced_places("gizmo", [], until=1) == [(2.1, 1.9, 0.75)]
    assert log(steps({1: text})).faced_places("gizmo", [], until=0) == []
    assert log(steps({5: text})).faced_places("gizmo", [], until=3) == []
    assert log(steps({1: "going to the kitchen counter 3"})).faced_places("gizmo", [], until=3) == []
