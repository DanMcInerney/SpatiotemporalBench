"""`out_of_view` (questions_view): the places, as last seen, of the movable things on the floor near the robot that it
cannot see now, by hand on `view_kit`'s run (the decision at keyframe 21 has the look window 19-21 and the base at
(-0.17, 0); radius 3 m, height 0.3 m, the bands 0.5 m and 0.05 m)."""

from __future__ import annotations

import pytest

import family_kit as fk
import view_kit as vk
from space_kit import box, vacuum_run
from stbench import questions_view, score
from stbench.questions_swap import gold_changes_under_swap
from stbench.questions_text import FAMILIES
from stbench.records import check_gold, check_question
from view_kit import BAG, GLIMPSE_PX, K2, SHOES, TOY, VASE, K, ask, looking, person, thing

BUILT_HEIGHT = questions_view.HEIGHT  # as built, read before any test patches it


@pytest.fixture(autouse=True, scope="module")
def primary_height():
    """These tests are worked out by hand at the text's primary height, `HEIGHT_M` (0.3 m); the height as built is the
    fallback (M2's fast gate check), pinned by `test_the_built_height_is_the_fallback`."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(questions_view, "HEIGHT", questions_view.HEIGHT_M)
        yield


def test_the_built_height_is_the_fallback():
    """Contract (DECISIONS, "M2's fast gate check"): `out_of_view` asks about things taller than 0.1 m, the declared
    fallback, taken when every gold of 15 on look-dev seeds 141-145 was empty at 0.3 m. Regression caught: the
    primary height restored without a new fast gate check."""
    assert BUILT_HEIGHT == questions_view.FALLBACK_HEIGHT_M == 0.1



def places(gold: dict) -> tuple[list, list]:
    """(the items' places, the don't-care places) of a positions gold, as written."""
    return [i["xy"] for i in gold["items"]], [r["xy"] for r in gold.get("dont_care", [])]


def hand_world(in_view: bool = True):
    """Things around the base (-0.17, 0), each seen at keyframe 3 (before the look window 19-21): E1 a chair 1.0 m
    ahead; E2 shoes at 1.0 m; E3 a 0.314 m bag at 1.0 m; E4 a chair at 3.2 m; E5 one at 3.6 m; E6 a 0.4 m vase on a 0.8
    m table at 1.8 m; E7 a chair 1.0 m behind, which a 100 px row at keyframe 20 puts in view now (`in_view`); P1 a
    person 2.0 m off, P3 one 5 mm from it, P2 one at 6.2 m."""
    s = vacuum_run()
    thing(s, "E1", (0.83, 0.0))
    thing(s, "E2", (-0.17, 1.0), SHOES, kind="shoes", mobility="carried")
    thing(s, "E3", (-0.17, -1.0), BAG, kind="bag", mobility="carried")
    thing(s, "E4", (3.03, 0.0))
    thing(s, "E5", (-0.17, 3.6))
    thing(s, "E6", (0.83, 1.5), VASE, kind="vase", mobility="carried", z=0.8)
    thing(s, "E7", (-1.17, 0.0))
    if in_view:
        s.see(20, "E7", GLIMPSE_PX)
    person(s, "P1", (-0.17, 2.0))
    person(s, "P3", (-0.17, 2.005))
    person(s, "P2", (6.0, 0.0))
    return s


def test_out_of_view_by_hand():
    """Contract: the gold lists the movable things taller than 0.3 m on the floor within 3 m that no row of the look
    window shows, each at its last-sighting place, and makes a don't-care place of every case the robot cannot tell
    apart, so no score hinges on a boundary or an in-view thing. Hand values at the decision at keyframe 21: E1, a 0.9 m
    chair 1.0 m away, is the one item, at (0.83, 0), evidence its sighting 3 and the decision 21; E2 (0.1 m) is nothing;
    E3 (0.314 m, inside the 0.05 m band) is don't-care at (-0.17, -1); E4 at 3.2 m (inside the 0.5 m band of the radius)
    is don't-care at (3.03, 0); E5 at 3.6 m is nothing; E6 on a 0.8 m table is nothing (not on the floor); E7, tall and
    seen before, has a row in the look window and so is don't-care where it is now, (-1.17, 0), and without that row it
    is an item; a person is never an item, only don't-care at its last place, (-0.17, 2) for P1 (P3, 5 mm off, merges
    with it) and none for P2, 6.2 m away, since no item or answer can touch a place past radius + 0.5 m. The text and
    slots are the family's: radius 3 m (the vacuum's depth range less a metre). Regression caught: a thing in view
    counted as out of view, a boundary case scored, a thing on a table or a low one listed, or a person made an item."""
    c = ask(hand_world())
    assert c.gold == {"type": "positions", "items": [{"xy": [0.83, 0.0]}],
                      "dont_care": [{"xy": xy, "radius": 0.5}
                                    for xy in ([-1.17, 0.0], [-0.17, -1.0], [-0.17, 2.0], [3.03, 0.0])]}
    assert (c.family, c.slice, c.decision_k, c.evidence_k, c.entities, c.critical) == ("out_of_view", "answerable", K,
                                                                                         (3, K), (), True)
    assert c.subjects == (f"w:{K}", "e:E1") and c.tolerances == {"match_m": 0.5} and c.scope is None
    assert c.slots == {"height": 0.3, "radius": 3.0}
    assert FAMILIES["out_of_view"].render(c.slots) == (
        "Where, as you last saw them, are the movable things taller than 0.3 m standing on the floor within 3 m of you "
        "that are not in any of the images you were just sent?")
    items, care = places(ask(hand_world(in_view=False)).gold)
    assert items == [[-1.17, 0.0], [0.83, 0.0]] and [-1.17, 0.0] not in care


def test_as_last_seen():
    """Contract: an item is where the robot last saw the thing, never where it is now (truth rule 2). Hand-worked: a
    chair seen at keyframe 3 at (0.83, 0) is moved at t 9 s (keyframe 6) to (-1.17, 0), a place the robot never looks
    at; at the decision at 21 the gold is the one item at (0.83, 0) with evidence 3 and 21, and nothing at its true
    place, not even a don't-care. Regression caught: the true place leaking into the gold."""
    s = vacuum_run()
    thing(s, "E1", None, where=[(0, 9.0, (0.83, 0.0, 0.0)), (9.0, None, (-1.17, 0.0, 0.0))])
    c = ask(s)
    assert c.gold == {"type": "positions", "items": [{"xy": [0.83, 0.0]}]}
    assert c.evidence_k == (3, K) and c.subjects == (f"w:{K}", "e:E1")


def gap_world(look: bool = True, blocked: bool = False):
    """Two tasks; a chair seen at P = (0.83, 0) in task 0 (keyframe 3) and, in the gap (t 3,600 s), moved 0.6 m to
    (0.83, 0.6); in task 1 the robot stands at (-0.5, 0) heading +x at keyframe 38, before the decision at 49 (look
    window 47-49), with a render there when `look`. `blocked` stands a box at P in that render. Only keyframes 47-49 and
    38 carry depth: a render of task 0's look window would show P as bare floor while the chair stood there."""
    s = vacuum_run(2)
    s.depth_ks = {47, 48, 49}
    thing(s, "E1", None, where=[(0, 3600.0, (0.83, 0.0, 0.0)), (3600.0, None, (0.83, 0.6, 0.0))])
    if look:
        looking(s, 38, (-0.5, 0.0))
    if blocked:
        s.frame_boxes[38] = [box(0.63, 1.03, -0.2, 0.2, 0.0, 0.9)]
    return s


def test_a_place_seen_empty_since_is_dont_care():
    """Contract: when the keyframes after a thing's last sighting and before the look window certify its place empty,
    the robot saw it was gone, so that place is don't-care and the thing is no item (the review's m5: a table's track of
    a thing moved in a gap would otherwise be a ghost the gold lacks). Hand-worked on two tasks: a chair seen at (0.83,
    0) in task 0 and moved 0.6 m in the gap; at keyframe 38 the robot looks at (0.83, 0) from 1.33 m with a ray-cast
    render of bare floor (free fine cells: `place_free` over the chair's half-width 0.2 m): the gold at 49 is no item
    and a don't-care at (0.83, 0), the new place never seen. Without the look it is the item at (0.83, 0); with a box
    standing at (0.83, 0) in the render (something there: not seen empty) it is the item too; and a chair that arrives
    at (0.83, 0) at t 4 s, after a render of the bare place at keyframe 2 and before its sighting at 3, is the item (a
    look before the sighting says nothing of what happened since). Regression caught: an item the robot itself saw
    gone, a place called empty that was only looked at, or one looked at before the thing was last seen there."""
    gone = ask(gap_world(), K2, 0, 1).gold
    assert places(gone) == ([], [[0.83, 0.0]])
    for s in (gap_world(look=False), gap_world(blocked=True)):
        assert places(ask(s, K2, 0, 1).gold) == ([[0.83, 0.0]], [])
    s = vacuum_run()
    thing(s, "E1", None, where=[(4.0, None, (0.83, 0.0, 0.0))])
    looking(s, 2, (-0.5, 0.0))
    assert places(ask(s).gold) == ([[0.83, 0.0]], [])


def test_a_glimpse_after_the_last_sighting_is_dont_care():
    """Contract: a sighting that is not recognisable (here a glimpse) decides no item: a thing that was only glimpsed,
    or glimpsed 0.15 m or more from its last recognisable sighting after it, is don't-care at that sighting and at every
    glimpse after it (a track may hold any of them: a memory reads its latest detection), and a glimpse before the last
    recognisable sighting is ignored. Hand-worked at 21: E1 recognisable (1,500 px) at keyframe 4 at (0.83, 0), moved at
    t 9 s, glimpsed at 100 px at keyframe 9 at (-0.17, 1.2): don't-care (0.83, 0) and (-0.17, 1.2), no item; E2
    glimpsed at keyframe 2 at (0, -1), moved, recognisable at keyframe 4 at (-1.17, 0): the item (-1.17, 0) and nothing
    at (0, -1); E3 only glimpsed, at keyframe 5 at (1.5, -1.5): don't-care there; a person seen at keyframe 4 at (1,
    1.5) and glimpsed at 9 after moving to (-1, -1.5) is don't-care at both. Regression caught: a gold hinging on a
    sighting a person could not name, which a detector would read the other way."""
    s = vacuum_run()
    thing(s, "E1", None, at=(4,), where=[(0, 9.0, (0.83, 0.0, 0.0)), (9.0, None, (-0.17, 1.2, 0.0))])
    s.see(9, "E1", GLIMPSE_PX)
    thing(s, "E2", None, at=(4,), where=[(0, 4.0, (0.0, -1.0, 0.0)), (4.0, None, (-1.17, 0.0, 0.0))])
    s.see(2, "E2", GLIMPSE_PX)
    thing(s, "E3", (1.5, -1.5), at=(5,), px=GLIMPSE_PX)
    thing(s, "P1", None, vk.PERSON, kind="person", mobility="mover", attrs={}, at=(4,),
          where=[(0, 9.0, (1.0, 1.5, 0.0)), (9.0, None, (-1.0, -1.5, 0.0))])
    s.see(9, "P1", GLIMPSE_PX)
    assert places(ask(s).gold) == ([[-1.17, 0.0]], [[-1.0, -1.5], [-0.17, 1.2], [0.83, 0.0], [1.0, 1.5], [1.5, -1.5]])


def test_a_thing_glimpsed_in_the_look_window_keeps_its_old_place_dont_care():
    """Contract: a thing with a row in the look window is in view, so no item, and the places a track of it from before
    the look window would hold (its last sighting, every glimpse after it) are don't-care like a mover's, so a table
    that keeps the old track is not penalised (`knowable`, VISION rule 2). Hand-worked on two tasks: a chair seen at
    (0.83, 0) in task 0, moved in the gap to (0.83, 0.6) and only glimpsed (100 px) at keyframe 48 of the look window
    47-49, its old place never looked at again: the gold at 49 has no item, and don't-care places at (0.83, 0) (the old
    track) and (0.83, 0.6) (where it is). The same chair with no row in the look window is the item at (0.83, 0).
    Regression caught: a ghost at the old place of a thing the gold calls in view."""
    s = gap_world(look=False)
    assert places(ask(s, K2, 0, 1).gold) == ([[0.83, 0.0]], [])
    s.see(48, "E1", GLIMPSE_PX)
    assert places(ask(s, K2, 0, 1).gold) == ([], [[0.83, 0.0], [0.83, 0.6]])


def test_the_swallow_rule():
    """Contract: an item within 0.5 m of a don't-care place becomes one (repeated until none moves), since the scorer
    drops answer points inside a don't-care region and would drop the right answer; a swallowed item leaves the
    evidence and the subjects. Hand-worked at 21 with people at (0, 1) and (1, -1) and chairs at (0, 1.4), (0, 1.8) and
    (1, -1.6): the chair 0.4 m from the first person is swallowed, the next chair 0.4 m from that one with it, the
    chair 0.6 m from the second person stays the one item (evidence 3 and 21, subject e:D); without the people all
    three are items. Regression caught: a right answer dropped by the scorer, or a chain left half swallowed."""
    def world(people: bool):
        s = vacuum_run()
        for eid, xy in (("A", (0.0, 1.4)), ("B", (0.0, 1.8)), ("D", (1.0, -1.6))):
            thing(s, eid, xy)
        if people:
            person(s, "P1", (0.0, 1.0))
            person(s, "P2", (1.0, -1.0))
        return s

    c = ask(world(True))
    assert places(c.gold) == ([[1.0, -1.6]], [[0.0, 1.0], [0.0, 1.4], [0.0, 1.8], [1.0, -1.0]])
    assert c.evidence_k == (3, K) and c.subjects == (f"w:{K}", "e:D")
    assert places(ask(world(False)).gold) == ([[0.0, 1.4], [0.0, 1.8], [1.0, -1.6]], [])


def test_the_fallback_height(monkeypatch):
    """Contract: the slot `height` is the one thing the declared fallback changes: with 0.1 m the shoes (0.1 m) are
    don't-care and the 0.2 m toy and the 0.314 m bag are items, where at the default 0.3 m the shoes and the toy are
    nothing and only the bag (inside the band) is don't-care. Hand-worked at 21 with shoes at (0.83, 0), a toy at
    (-0.17, -1.2) and a bag at (-0.17, 1.5); setting `questions_view.HEIGHT` to the fallback is the one-line switch the
    coordinator makes, and the text then reads 'taller than 0.1 m'. Regression caught: a fallback that changes nothing,
    or a band misread."""
    s = vacuum_run()
    thing(s, "E1", (0.83, 0.0), SHOES, kind="shoes", mobility="carried")
    thing(s, "E2", (-0.17, -1.2), TOY, kind="toy", mobility="carried")
    thing(s, "E3", (-0.17, 1.5), BAG, kind="bag", mobility="carried")
    assert places(ask(s).gold) == ([], [[-0.17, 1.5]])
    low = ask(s, height=questions_view.FALLBACK_HEIGHT_M)
    assert places(low.gold) == ([[-0.17, -1.2], [-0.17, 1.5]], [[0.83, 0.0]]) and low.slots["height"] == 0.1
    monkeypatch.setattr(questions_view, "HEIGHT", questions_view.FALLBACK_HEIGHT_M)
    assert ask(s).slots["height"] == 0.1
    assert "taller than 0.1 m" in FAMILIES["out_of_view"].render(ask(s).slots)


def test_hidden_changes_never_change_a_gold():
    """Contract: nothing the robot did not see changes a gold (VISION rule 4). Hand-worked at 21 with two identical oak
    chairs X and Y seen at keyframe 3 at (0.83, 0) and (-0.17, -1.2) and a birch chair W seen at (-1.17, 1.2): the gold
    is W as the one item and the twins' two places don't-care; the identical-item swap check (X and Y exchange
    everything from t 0, and from t 20 s) finds no changed gold; and adding a second birch chair and a person no frame
    ever shows, in range, leaves the gold as it was (an unseen twin is no twin). Regression caught: a gold reading what
    was not seen."""
    def world(extra: bool):
        s = vacuum_run()
        for eid, xy in (("X", (0.83, 0.0)), ("Y", (-0.17, -1.2))):
            thing(s, eid, xy, attrs={"color": "oak"})
        thing(s, "W", (-1.17, 1.2), attrs={"color": "birch"})
        if extra:
            thing(s, "Z", (1.2, 1.0), attrs={"color": "birch"}, at=())
            person(s, "U", (-1.0, -1.0), at=())
        return s

    base = ask(world(False)).gold
    assert places(base) == ([[-1.17, 1.2]], [[-0.17, -1.2], [0.83, 0.0]])
    assert ask(world(True)).gold == base
    inp, rec = vk.inputs_and_record(world(False))
    for t in (0.0, 20.0):
        assert gold_changes_under_swap(inp, rec, "X", "Y", t) == []


def twin_world(a_colour: str):
    """Two tasks and two chairs seen at keyframes 3 and 35: A stands at (0, -1.5) all along, B at (0.83, 0) in task 0
    and, moved in the gap, at (-1.17, 1) in task 1; B is oak and A is `a_colour`. Only keyframes 47-49 carry depth, so
    no place is ever seen empty."""
    s = vacuum_run(2)
    s.depth_ks = {47, 48, 49}
    thing(s, "A", (0.0, -1.5), attrs={"color": a_colour}, at=(3, 35))
    thing(s, "B", None, attrs={"color": "oak"}, at=(3, 35),
          where=[(0, 3600.0, (0.83, 0.0, 0.0)), (3600.0, None, (-1.17, 1.0, 0.0))])
    return s


def test_a_twin_is_never_an_item():
    """Contract: identity between identical things is not observable, so a gold must not depend on it (`knowable`,
    VISION rule 2): a thing with a twin, another entity of its description that some frame delivered by the end of the
    look window shows, is never an item, and every place either was sighted at before the look window is don't-care,
    where a table that tells its tracks apart by description keeps one track a place. Hand-worked on two tasks at 49
    with two oak chairs, A still at (0, -1.5) and B moved in the gap from (0.83, 0) to (-1.17, 1), both seen at 3 and 35
    and the old place never looked at again: no item, don't-care at (-1.17, 1), (0, -1.5) and (0.83, 0). With A birch
    the chairs are told apart: items at (-1.17, 1) and (0, -1.5), and nothing at (0.83, 0). A twin first shown after the
    look window (keyframe 25 of the one-task window, the look window being 19-21) is no twin yet, so the chair at (0.83,
    0) stays an item, while one shown at keyframe 2 makes both don't-care. Regression caught: an item a table would list
    as a ghost where the gold, knowing which chair is which, has none, or a gold that reads frames the ask has not been
    sent."""
    assert places(ask(twin_world("oak"), K2, 0, 1).gold) == ([], [[-1.17, 1.0], [0.0, -1.5], [0.83, 0.0]])
    assert places(ask(twin_world("birch"), K2, 0, 1).gold) == ([[-1.17, 1.0], [0.0, -1.5]], [])

    def one_task(twin_at: int):
        s = vacuum_run()
        thing(s, "B", (0.83, 0.0), attrs={"color": "oak"})
        thing(s, "A", (-1.17, 0.0), attrs={"color": "oak"}, at=(twin_at,))
        return s

    assert places(ask(one_task(25)).gold) == ([[0.83, 0.0]], [])
    assert places(ask(one_task(2)).gold) == ([], [[-1.17, 0.0], [0.83, 0.0]])


def test_rows_pass_the_validators():
    """Contract: every row the compile writes for the family passes `check_question` and `check_gold`, a positions gold
    with its don't-care regions included: credit f1 with `match_m` 0.5, scope `memory`, no metric, the evidence and
    subjects the gold names. Hand-worked on the by-hand world compiled as a one-task case: the decision at 21 is the
    one point of the job's `before_doorway` far enough into the task (16 steps), so one `out_of_view` row, asking the
    text of its slots. Regression caught: a positions gold the scorer cannot read, or a row that does not fit the
    spec."""
    qs, gs = fk.compiled(hand_world(), job=vk.VIEW_JOB)
    ((q, g),) = ((q, g) for q, g in zip(qs, gs, strict=True) if g["family"] == "out_of_view")
    assert check_question(q) == [] and check_gold(g) == []
    assert (q["when"], q["decision_k"], q["answer_type"], q["needs"], q["slots"]) == (
        "during", K, "positions", ["see", "pose"], {"height": 0.3, "radius": 3.0})
    assert q["text"] == FAMILIES["out_of_view"].render(q["slots"])
    assert (g["credit"], g["tolerances"], g["scope"], g["metric"], g["reference"], g["critical"]) == (
        "f1", {"match_m": 0.5}, "memory", None, None, True)
    assert g["gold"]["items"] == [{"xy": [0.83, 0.0]}] and len(g["gold"]["dont_care"]) == 4
    assert (g["evidence_k"], g["subjects"], g["entities"], g["slice"]) == ([3, K], [f"w:{K}", "e:E1"], [], "answerable")


def test_items_rest_on_recognisable_frames_and_a_clipped_exit_frame_keeps_them():
    """Contract: an item is where the robot last saw the thing recognisably (enough pixels and light, a box clear of the
    image's edge, a depth), and the keyframe of that frame is the evidence; a later sighting that is not recognisable
    but lies within 0.15 m of that place (the frame of a thing leaving the view, cut by the edge or depth-less) leaves
    it an item; one 0.15 m or more away makes both places don't-care, since a memory that reads the latest detection
    holds the thing there; a thing with no recognisable sighting at all (cut by the edge, then a 400 px detection) is
    don't-care at the places it was sighted and never an item. Hand-worked at 21 on chairs: E1 at (0.83, 0)
    recognisable at keyframe 3, cut by the image's edge at 9: the item (0.83, 0), evidence 3 (not 9) and 21; E2 at
    (-0.17, -1.2), recognisable at 3, depth-less at 9 after moving 0.1 m to (-0.17, -1.3): the item (-0.17, -1.2); E3
    at (-1.17, 0), recognisable at 3, cut by the edge at 9 after moving 0.3 m to (-1.17, 0.3): no item, don't-care at
    both places; E4 at (0.83, 1) cut by the edge at 3 and detected at 400 px at 5: don't-care at (0.83, 1). Regression
    caught: a sliver at the frame's edge taken as the evidence of an item (an `out_of_view` row the agent was shown no
    image of), or a thing that left the view kept at a place the table does not hold."""
    s = vacuum_run()
    edge = [0, 100, 40, 140]
    thing(s, "E1", (0.83, 0.0))
    s.see(9, "E1", 1500, bbox=edge)
    thing(s, "E2", None, where=[(0, 9.0, (-0.17, -1.2, 0.0)), (9.0, None, (-0.17, -1.3, 0.0))])
    s.see(9, "E2", 1500, depth=None)
    thing(s, "E3", None, where=[(0, 9.0, (-1.17, 0.0, 0.0)), (9.0, None, (-1.17, 0.3, 0.0))])
    s.see(9, "E3", 1500, bbox=edge)
    thing(s, "E4", (0.83, 1.0), at=())
    s.see(3, "E4", 1500, bbox=edge)
    s.see(5, "E4", 400)
    c = ask(s)
    assert places(c.gold) == ([[-0.17, -1.2], [0.83, 0.0]], [[-1.17, 0.0], [-1.17, 0.3], [0.83, 1.0]])
    assert c.evidence_k == (3, K) and c.subjects == (f"w:{K}", "e:E1", "e:E2")


def test_a_large_thing_widens_the_match_and_its_dont_care_radius():
    """Contract: the matching radius of a row is 0.5 m plus the largest `rules.widening` of its items, and a don't-care
    place's radius 0.5 m plus its own thing's, written on the row, since a large thing's surface is seen where its
    centre is not. Hand-worked at 21 with a pallet 1.2 x 1.0 m (widening 0.35): pallet P at (0.83, 0) and a chair at
    (-0.17, -1.2) are the items, `match_m` 0.85; a second pallet Q at (2.73, 0), 2.9 m away and so within 0.5 m of the
    radius, is don't-care with radius 0.85, a bag in the height band at (-0.17, 1.5) with radius 0.5. Through the
    scorer, an answer naming P 0.6 m from its centre (and the chair) scores 1, one 0.9 m off scores under 1, and the
    0.6 m answer scored by the old fixed 0.5 m radius does not score 1. Without a large item `match_m` is 0.5.
    Regression caught: agents naming the right pallet scored -1 for the surface-to-centre error of a thing wider than
    a metre."""
    s = vacuum_run()
    thing(s, "P", (0.83, 0.0), vk.PALLET, kind="pallet", mobility="carried")
    thing(s, "C", (-0.17, -1.2))
    thing(s, "Q", (2.73, 0.0), vk.PALLET, kind="pallet", mobility="carried")
    thing(s, "BG", (-0.17, 1.5), BAG, kind="bag", mobility="carried")
    c = ask(s)
    assert places(c.gold) == ([[-0.17, -1.2], [0.83, 0.0]], [[-0.17, 1.5], [2.73, 0.0]])
    assert [r["radius"] for r in c.gold["dont_care"]] == [0.5, 0.85] and c.tolerances == {"match_m": 0.85}
    (row,) = (g for g in fk.compiled(s, job=vk.VIEW_JOB)[1] if g["family"] == "out_of_view")

    def answered(offset: float) -> float:
        items = [{"xy": [0.83, offset]}, {"xy": [-0.17, -1.2]}]
        return score.credit({"type": "positions", "items": items}, row)

    assert answered(0.0) == 1.0 and answered(0.6) == 1.0 and answered(0.9) < 1.0
    assert score.credit({"type": "positions", "items": [{"xy": [0.83, 0.6]}, {"xy": [-0.17, -1.2]}]},
                        {**row, "tolerances": {"match_m": 0.5}}) < 1.0
    plain = vacuum_run()
    thing(plain, "C", (-0.17, -1.2))
    assert ask(plain).tolerances == {"match_m": 0.5}


def test_an_empty_gold_stays():
    """Contract: nothing seen near is the empty list, an honest answer to 'what was seen' and never `unknown`: the row
    is still written, answerable, with the matching radius 0.5, no don't-care place, and the decision as its only
    evidence. Hand-worked at 21 with one chair 3.6 m away (past the radius and its band: nothing). Regression caught:
    an empty gold dropped or made `unknown`, which would hide its share from the gates (the declared bound is 0.525)."""
    s = vacuum_run()
    thing(s, "E1", (-0.17, 3.6))
    c = ask(s)
    assert c.gold == {"type": "positions", "items": []} and c.slice == "answerable"
    assert (c.tolerances, c.evidence_k, c.subjects) == ({"match_m": 0.5}, (K,), (f"w:{K}",))
