"""`changed` (questions_changed), by hand on `changed_kit`'s living room: three tasks of 28 keyframes whose gaps move,
remove and add things, a resident, identical shoes and a mug on a table."""

from __future__ import annotations

import re
from dataclasses import replace
from functools import lru_cache

import changed_kit as ck
import family_kit as fk
from stbench import questionnaire, questions, questions_changed
from stbench.questionnaire import conflicts
from stbench.questions_changed import moments
from stbench.questions_text import FAMILIES
from stbench.records import check_family, check_gold, check_question
from stbench.seeding import stable_hex

HAND = {"type": "positions",
        "items": [{"xy": [4.6, 1.2]}, {"xy": [4.6, 2.0]}, {"xy": [6.2, 2.4]}],
        "dont_care": [{"xy": list(p), "radius": 0.5} for p in
                      [(5.4, 1.6), (5.8, 0.4), (6.2, 0.9), (6.6, 0.4), (7.4, 0.8), (7.4, 2.4), (7.8, 0.4),
                       (7.8, 2.4)]]}


@lru_cache(maxsize=1)
def rendered():
    """The by-hand script and its inputs over the rendered views, whose depth is read once a process."""
    s = ck.by_hand()
    return s, fk.inputs(s, ck.JOB)


def hand_world(cid: str = "c0"):
    s, inp = rendered()
    return ck.world(s, cid=cid, inp=inp)


def only(w):
    """The window's one `changed` candidate."""
    [m] = moments(w)
    assert m.unknown == []
    [c] = m.answerable
    return c


def hand_only(s, observed=None, **kwargs):
    """The one candidate of the window of script `s` over hand free-space states (the whole room observed, free)."""
    observed = dict.fromkeys(range(len(s.tasks)), 1.0) if observed is None else observed
    return only(ck.world(s, inp=ck.hand_inputs(s, observed), **kwargs))


def test_changed_by_hand():
    """Contract: the gold of tasks 0 and 2 in the living room (by hand, `changed_kit.by_hand`; things on the floor
    count, the task's last seen sighting is where a thing was, the shared place test certifies "added" and "removed"):
    - E0001 the red bag moved (7.4, 0.8) to (4.6, 1.2): an event at (4.6, 1.2), a don't-care place at (7.4, 0.8);
    - E0002 the blue box gone from (6.2, 2.4), a place task 2 saw empty: an event there;
    - E0004 the green bag on a table in tasks 0 and 1 and on the floor at (4.6, 2.0) in task 2: it has no `a` (a thing
      up on a table has no floor sighting) and a `b`, where task 0 saw the floor empty: an event at (4.6, 2.0);
    - E0003 the yellow toy added at (7.8, 2.4), task 0 saw it empty, but 0.4 m from where the resident stood (7.4, 2.4)
      in task 2: swallowed, a don't-care place;
    - E0005 the mug moved 0.6 m on a table: nothing, no floor sighting;
    - E0006 and E0007, identical black shoes: don't-care places (5.8, 0.4), (6.6, 0.4) and (7.8, 0.4), though one moved;
    - P0001 the resident at (5.4, 1.6) in task 0, at (7.4, 2.4) and (6.2, 0.9) in task 2: don't-care places.
    So the events are (4.6, 1.2), (4.6, 2.0) and (6.2, 2.4), the don't-care places those eight of radius 0.5 m. The
    evidence is the first keyframes of the two tasks (0 and 56) and the sightings the events read: E0001 at 7 and 67,
    E0002 at 9, E0004 at 67 (the keyframe in a task that faces the place); the subjects the room, the two tasks as
    times (`w:0-27`, `w:56-83`) and the three things. Regression caught: the per-sighting rule read differently from
    the table's (a thing on a table counted as standing on the floor), a twin or a resident reported as a change, or
    an event the robot could not have seen."""
    w = hand_world()
    assert w.tasks == [(0, 0, 28), (1, 28, 56), (2, 56, 84)]
    [m] = moments(w)
    assert m.key == "living room|0|2" and m.unknown == []
    [c] = m.answerable
    assert c.family == "changed" and c.slots == {"room": "living room", "tasks": [0, 2]}
    assert c.gold == HAND and c.slice == "answerable" and c.tolerances == {"match_m": 0.5}
    assert c.evidence_k == (0, 7, 9, 56, 67) and c.entities == ()
    assert c.subjects == ("r:R2", "w:0-27", "w:56-83", "e:E0001", "e:E0002", "e:E0004") and c.critical
    assert FAMILIES["changed"].render(c.slots) == (
        "Where in the living room did something change between your first task and your third task? List where each "
        "thing was added or moved to, and where each was removed from.")


def test_the_last_sighting_of_a_task_decides():
    """Contract: a thing is where its last seen-bound sighting of the task put it: the red bag also seen (1,500 px) at
    keyframe 14, later in task 0 than its keyframe 7, is read there (the evidence 14 for 7, the gold as it was); the
    blue box glimpsed (100 px) at keyframe 8, before its sighting at 9, is still an event at (6.2, 2.4); and the twin
    shoe E0007 seen only at 100 px in task 2 keeps (7.8, 0.4) a don't-care place, a twin's glimpses counting as its
    sightings do (the gold as it was). Regression caught: a thing read at its first sighting, or a twin's glimpsed
    place forgotten."""
    base = hand_only(ck.by_hand())
    later = hand_only(ck.by_hand(extra=lambda s: s.see(14, "E0001", 1500)))
    assert later.evidence_k == (0, 9, 14, 56, 67) and replace(later, evidence_k=base.evidence_k) == base
    assert hand_only(ck.by_hand(extra=lambda s: s.see(8, "E0002", 100))) == base
    assert hand_only(ck.by_hand(extra=lambda s: s.see(63, "E0007", 100))) == base


def test_a_glimpse_adds_doubt_only_where_a_task_has_no_sighting():
    """Contract: nothing moves within a task, so a dead-band glimpse after a task's last seen sighting is at the
    thing's own seen place and adds no doubt about where it is; a thing only glimpsed in a task has no place the robot
    can vouch for, and stays unsure. By hand, each on `by_hand`'s gold (events (4.6, 1.2), (4.6, 2.0), (6.2, 2.4)):
    - the red bag glimpsed (100 px) at keyframe 70 of task 2 after its sighting at 67 (the move to (4.6, 1.2)), the blue
      box at keyframe 12 of task 0 after its sighting at 9 (the removal from (6.2, 2.4)), the green bag at keyframe 75
      of task 2 after its sighting at 67 (the addition at (4.6, 2.0)): each event stands, the gold, the evidence and the
      subjects as they were;
    - a white bag at (7.0, 2.0) in tasks 0 and 1 and (5.2, 2.6) in task 2, seen only in task 2, glimpsed in task 0
      at keyframe 7: no event at (5.2, 2.6) though task 0 saw it empty, and (7.0, 2.0) a don't-care place; seen in
      task 0 (1,500 px) instead, it moved: an event at (5.2, 2.6), with the white bag among the subjects.
    Regression caught: a seen, certified change erased by a glimpse of the thing standing where it was seen (three of
    the vacuum's thirteen `changed` moments), or a thing only glimpsed read as seen."""
    base = hand_only(ck.by_hand())
    for k, eid in ((70, "E0001"), (12, "E0002"), (75, "E0004")):
        assert hand_only(ck.by_hand(extra=lambda s, k=k, eid=eid: s.see(k, eid, 100))) == base

    def white(seen, *, glimpse=False):
        def add(s):
            ck.put(s, "E0011", "bag", {"color": "white"}, [(7.0, 2.0), (7.0, 2.0), (5.2, 2.6)], seen=seen)
            if glimpse:
                s.see(7, "E0011", 100)
        return add

    glimpsed = hand_only(ck.by_hand(extra=white([2], glimpse=True)))
    assert glimpsed.gold["items"] == base.gold["items"] and glimpsed.subjects == base.subjects
    assert glimpsed.gold["dont_care"] == sorted(
        [*base.gold["dont_care"], {"xy": [7.0, 2.0], "radius": 0.5}], key=lambda item: item["xy"])
    seen = hand_only(ck.by_hand(extra=white([0, 2])))
    assert seen.gold["items"] == [{"xy": [4.6, 1.2]}, {"xy": [4.6, 2.0]}, {"xy": [5.2, 2.6]}, {"xy": [6.2, 2.4]}]
    assert seen.subjects == ("r:R2", "w:0-27", "w:56-83", "e:E0001", "e:E0002", "e:E0004", "e:E0011")


def test_the_place_test_reads_the_things_narrow_half():
    """Contract: a removal is certified over a disc of `min(size[0], size[1]) / 2` around the old place, so a long
    thing is not asked for more floor than its width: a crate 0.3 x 0.9 m gone from (5.8, 1.5), with only the
    half of the living room west of x = 6 observed (free) in task 2, is an event there (the 0.15 m disc lies inside
    what was seen; the 0.45 m of its long side would reach 0.25 m past it), and no don't-care place is written when
    there is none. Regression caught: the wrong half of the size, which turns a certified removal into a don't-care
    place."""
    s = ck.script(3)
    ck.put(s, "E0010", "crate", {"color": "brown"}, [(5.8, 1.5), (5.8, 1.5), None], size=(0.3, 0.9, 0.3))
    c = hand_only(s, {0: 1.0, 1: 1.0, 2: 0.5})
    assert c.gold == {"type": "positions", "items": [{"xy": [5.8, 1.5]}]}
    assert c.subjects == ("r:R2", "w:0-27", "w:56-83", "e:E0010")


def test_only_pairs_two_apart(monkeypatch):
    """Contract: `changed` asks the pairs of a window's tasks at least two apart, and only of a room toured in both:
    a 2-task window has none, a 3-task window the pair (0, 2), a 5-task window six pairs, and under the fallback
    (`FIRST_LAST`, as built) the first and last tasks only, (0, 4) of 5; and on a 2-task window the end questions with
    `changed` named are, for any case id, exactly those without it (by hand: one `changed` question taken first would
    leave `object_where` nothing, its event keyframes lying inside the two tasks). Regression caught: adjacent tasks,
    little elapsed time, crowding `object_where` out of a medium window (the plan review's M5), or the fallback not
    taken."""
    five = ck.by_hand(5)
    w5 = ck.world(five, inp=ck.hand_inputs(five, dict.fromkeys(range(5), 1.0)))
    assert [m.key for m in moments(w5)] == ["living room|0|4"]
    monkeypatch.setattr(questions_changed, "FIRST_LAST", False)
    pairs = {2: [], 3: ["living room|0|2"],
             5: [f"living room|{i}|{j}" for i, j in ((0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 4))]}
    for n, want in pairs.items():
        s = ck.by_hand(n)
        w = ck.world(s, inp=ck.hand_inputs(s, dict.fromkeys(range(n), 1.0)))
        assert sorted(m.key for m in moments(w)) == want
    s = ck.by_hand(2)
    inp = ck.hand_inputs(s, {0: 1.0, 1: 1.0})
    where_only = replace(ck.JOB, ends=("object_where",))
    for i in range(8):
        cid = stable_hex("two apart", i, n=16)
        both = [c.key for c in questionnaire.end(ck.world(s, cid=cid, inp=inp))]
        alone = [c.key for c in questionnaire.end(ck.world(s, cid=cid, job=where_only, inp=inp))]
        assert both == alone and both


def test_an_unseen_twin_changes_nothing():
    """Contract: an event stands unless another entity some frame shows has the thing's description: a second red bag
    that no frame shows (E0008, standing at (4.0, 0.3)) leaves the red bag's event at (4.6, 1.2) and the whole gold as
    it was; shown in task 1 at 100 px (a glimpse, enough to be sighted) it makes the first bag a twin, and the event
    becomes a don't-care place (at (4.6, 1.2) and where it was, (7.4, 0.8)). Regression caught: an entity the robot
    never saw turning an event into don't-care (truth rule 2), or a twin the table can see missed."""
    def twin(**kwargs):
        return lambda s: ck.put(s, "E0008", "bag", {"color": "red"}, [(4.0, 0.3)], **kwargs)

    base = hand_only(ck.by_hand())
    assert hand_only(ck.by_hand(extra=twin(seen=[]))) == base
    shown = hand_only(ck.by_hand(extra=twin(seen=[1], px=100)))
    assert shown.gold["items"] == [{"xy": [4.6, 2.0]}, {"xy": [6.2, 2.4]}]
    assert {"xy": [4.6, 1.2], "radius": 0.5} in shown.gold["dont_care"]
    assert shown.subjects == ("r:R2", "w:0-27", "w:56-83", "e:E0002", "e:E0004")


def test_a_room_not_toured_is_not_asked():
    """Contract: a room is asked about only when both tasks observed at least half of its inside fine cells, in any
    state: of the living room's 1,200, 600 free (or 600 occupied) in task 2 and all in the others is a moment, 599
    in task 2, or 599 in task 0, none; and the kitchen, which no keyframe observes, never is one. Regression caught:
    an empty gold, "nothing changed", from a room the robot never looked at."""
    s = ck.by_hand()
    for state in ("free", "occupied"):
        w = ck.world(s, inp=ck.hand_inputs(s, {0: 1.0, 1: 1.0, 2: 0.5}, state=state))
        assert [m.key for m in moments(w)] == ["living room|0|2"]
    for seen in ({0: 1.0, 1: 1.0, 2: 599 / 1200}, {0: 599 / 1200, 1: 1.0, 2: 1.0}, {0: 1.0, 1: 1.0}):
        assert moments(ck.world(s, inp=ck.hand_inputs(s, seen))) == []


def test_a_room_kind_must_be_unique():
    """Contract: a room is named in the question by its kind, so a layout with two rooms of one kind asks about
    neither (here the kitchen renamed a living room too, and the living room toured). Regression caught: a question
    whose answer depends on which of two rooms of one name is meant, which a reader cannot know."""
    s = ck.by_hand()
    inp = ck.hand_inputs(s, dict.fromkeys(range(3), 1.0))
    assert len(moments(ck.world(s, inp=inp))) == 1
    twice = replace(inp, map=replace(inp.map, regions=tuple(replace(r, kind="living room") for r in inp.map.regions)))
    assert moments(ck.world(s, inp=twice)) == []


def test_critical_follows_the_relevance_regions():
    """Contract: the question is critical when the room is in the relevance regions of either task's plan: the job's
    regions kitchen and living room (and so the living room) make it critical, regions of the kitchen alone do not.
    Regression caught: a question marked critical that no task's relevance names, or one missed."""
    s = ck.by_hand()
    assert hand_only(s).critical
    kitchen = replace(ck.JOB, relevance=lambda plan: replace(ck.JOB.relevance(plan), regions=frozenset({"kitchen"})))
    assert not hand_only(s, job=kitchen).critical


def test_two_tasks_are_two_named_times():
    """Contract: `changed` names its two tasks as times (`w:0-27` and `w:56-83`, the tasks' keyframes) so that no other
    end question of a 3-task window names a time inside them: over 24 case ids, whenever `changed` is asked, every
    other question's `w:` times lie in task 1 (keyframes 28 to 55), and it is asked in some of them, with questions
    beside it. The hand world gains a white bag that stays at (5.0, 1.0), which `changed` does not name: its entities
    (E0001, E0002 and E0004) leave no other answerable thing in task 1, and a moment is asked only when the subject
    rule allows a candidate in both its pools. Regression caught: end questions hinting at each other (VISION rule 6):
    an `object_where` about the first task's event beside a question that names the first task."""
    s = ck.by_hand(extra=lambda s: ck.put(s, "E0008", "bag", {"color": "white"}, [(5.0, 1.0)]))
    inp = ck.hand_inputs(s, dict.fromkeys(range(3), 1.0))
    asked, beside = 0, 0
    for i in range(24):
        cid = stable_hex("named times", i, n=16)
        end = questionnaire.end(ck.world(s, cid=cid, inp=inp))
        changed = [c for c in end if c.family == "changed"]
        others = [c for c in end if c.family != "changed"]
        for c in changed:
            assert c.subjects[1:3] == ("w:0-27", "w:56-83")
            assert not any(conflicts(c.subjects, o.subjects) for o in others)
            times = [int(m[1]) for o in others for t in o.subjects if (m := re.fullmatch(r"w:(\d+)", t))]
            assert all(28 <= k < 56 for k in times)
            asked, beside = asked + 1, beside + len(others)
    assert asked > 0 and beside > 0


def test_hidden_changes_never_change_a_gold():
    """Contract: a gold reads what the robot saw. A pink bag at (5.0, 0.5) in tasks 0 and 2, moved to (7.0, 2.8) in
    the gap before task 1 and back in the gap before task 2 where no frame saw it, gives the gold of the world in which
    it never moved, and that of the world without it; two identical shoes that swap places in task 2 give the gold of
    the world in which they did not (an equal evidence and equal subjects too). Regression caught: a gold reading what
    was not seen: the timeline's moves rather than the sightings."""
    def pink(stands):
        return lambda s: ck.put(s, "E0009", "bag", {"color": "pink"}, stands, seen=[0, 2])

    def shoes(places):
        def add(s):
            ck.put(s, "E0006", "shoes", {"color": "black"}, places[0], size=ck.SHOES)
            ck.put(s, "E0007", "shoes", {"color": "black"}, places[1], size=ck.SHOES)
        return add

    base = hand_only(ck.by_hand())
    there, hidden = [(5.0, 0.5)] * 3, [(5.0, 0.5), (7.0, 2.8), (5.0, 0.5)]
    assert hand_only(ck.by_hand(extra=pink(there))) == base
    assert hand_only(ck.by_hand(extra=pink(hidden))) == base
    kept = [[(5.8, 0.4)], [(6.6, 0.4), (6.6, 0.4), (7.8, 0.4)]]
    swapped = [[(5.8, 0.4), (5.8, 0.4), (7.8, 0.4)], [(6.6, 0.4), (6.6, 0.4), (5.8, 0.4)]]
    skip = ("E0006", "E0007")
    assert hand_only(ck.by_hand(skip=skip, extra=shoes(kept))) == base
    assert hand_only(ck.by_hand(skip=skip, extra=shoes(swapped))) == base


def test_rows_pass_the_validators():
    """Contract: every compiled row of a window that asks `changed` passes `check_question` and `check_gold`, the
    family's spec passes `check_family`, and the family is asked among `object_where`'s over a few case ids (a gold
    is `positions` with don't-care regions, the credit `f1` with `match_m`, scope `memory`). Regression caught: a row
    the harness could not read, or a gold the scorer refuses."""
    assert check_family(FAMILIES["changed"]) == []
    s, inp = rendered()
    found = set()
    for i in range(6):
        w = ck.world(s, cid=stable_hex("rows", i, n=16), inp=inp)
        qs, gs = questions.compile_inputs(w.inp, w.rec)
        assert [check_question(q) for q in qs] == [[]] * len(qs) and [check_gold(g) for g in gs] == [[]] * len(gs)
        for g in gs:
            found.add(g["family"])
            if g["family"] == "changed":
                assert (g["credit"], g["scope"], g["tolerances"], g["gold"]["type"]) == (
                    "f1", "memory", {"match_m": 0.5}, "positions")
    assert found == {"changed", "object_where"}
