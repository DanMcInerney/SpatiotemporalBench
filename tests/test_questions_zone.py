"""`zone_clear`, `clearance_then` and `slots_empty` (questions_zone), by hand on `zone_kit`'s run: the tote robot at
(3.9, 1.5) facing west, the bay B1 ahead (352 cells) and B2 3 m off, the decision at keyframe 17 (the look window
16-17), the events "when you reached the station" at 18 and "when you finished at the station" at 25 (the look window
18-25), the end look window 34-39, four staging lanes. The states are hand-set (`Views`, `paint`) except where a test
ray-casts."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

import zone_kit as zk
from space_kit import SpaceScript, box, use_job
from stbench import job_registry, questions, questions_world, questions_zone, zones
from stbench.events import EventRule
from stbench.questions_text import FAMILIES
from stbench.zones import ZoneRule
from synth import RECOG, WAREHOUSE
from zone_kit import REACHED, Views, paint

K = zk.DECISION_K
CELLS = zones.zone_cells(zk.B1)
NO_BAYS = replace(zk.MAP, furniture=tuple(f for f in zk.MAP.furniture if f.kind != "bay"))


def ask(views: Views, s: SpaceScript | None = None, **kw):
    """`zone_clear`'s candidate at the kit's decision (keyframe 17) over `views`."""
    return questions_zone.zone_clear(zk.world(s or zk.script(), views, **kw), K)


def then(views: Views, s: SpaceScript | None = None, **kw):
    return questions_zone.clearance_then(zk.world(s or zk.script(), views, **kw))


def slot_moments(views: Views, s: SpaceScript | None = None, **kw):
    return questions_zone.slots_empty(zk.world(s or zk.script(), views, **kw))


def test_zone_clear_reads_the_decisions_look_window_alone():
    """Contract: `zone_clear` asks about right now. Its gold is `zones.zone_state` over the decision's look window
    (keyframes 16-17) and no earlier frame: a blocker in the look window certifies "no", at least 80% of the zone's 352
    cells (282) seen free gives "yes", anything else is "unknown" (a gold of the unknown slice), and no question has a
    "last saw it" form (the family's spec is `now`, its slots are the zone and the height). Hand values:
    - the look window sees all of B1 free: yes, evidence (17,), the question of B1's entity P01 at 17;
    - 282 cells free (0.801): yes; 281 (0.798): unknown, whatever the bay looked like earlier;
    - the bay seen whole and free at 12 (an earlier look window) and nothing in the look window: unknown, no evidence;
      never seen: the same;
    - 317 cells free (90%) is yes, and 317 with one occupied among them is no, evidence (17,);
    - 100 cells seen (28%) with one blocker: no, evidence (17,), the look window alone decided;
    - 30 cells taken at 12 and the look window seeing 100 other cells free: unknown (a blocker seen before the look
      window is not the look window's), evidence (17,); the bay free at 12 and 30 cells taken at 17: no;
    - a robot band that says blocked everywhere is never read: every request is for ground 0 and band (0.3, 3.5).
    The text is 'Is the bay ahead of you clear of anything taller than 0.3 m right now?'. Regression caught: an answer
    from an earlier frame (the `memory` rows of the AMR's first `zone_clear`, all clear bays: `blind` scored +0.67), a
    wording of "as you last saw it", a zone judged on the robot's band, or evidence naming frames before the look
    window."""
    spec = FAMILIES["zone_clear"]
    assert (spec.scope, spec.needs, spec.slots) == ("now", ("see", "pose"), ("zone", "height"))
    c = ask(Views({16: paint(zk.B1), 17: paint(zk.B1)}))
    assert c.slots == {"zone": "the bay ahead of you", "height": 0.3}
    assert c.gold == {"type": "yesno", "value": True} and c.slice == "answerable"
    assert (c.family, c.evidence_k, c.decision_k, c.subjects, c.critical, c.tolerances, c.entities) == (
        "zone_clear", (17,), K, (f"w:{K}", "e:P01"), True, {}, ())
    assert spec.render(c.slots) == "Is the bay ahead of you clear of anything taller than 0.3 m right now?"
    assert ask(Views({17: paint(zk.B1, n=282)})).gold == {"type": "yesno", "value": True}
    short = ask(Views({12: paint(zk.B1), 17: paint(zk.B1, n=281)}))
    assert (short.gold, short.slice, short.evidence_k) == ({"type": "unknown"}, "unknown", (17,))
    for views in (Views({12: paint(zk.B1)}), Views()):
        c = ask(views)
        assert (c.gold, c.slice, c.evidence_k) == ({"type": "unknown"}, "unknown", ())
    clear90, blocked90 = ask(Views({17: paint(zk.B1, n=317)})), ask(Views({17: {**paint(zk.B1, n=317),
                                                                              CELLS[100]: "occupied"}}))
    assert clear90.gold["value"] is True
    assert (blocked90.gold["value"], blocked90.evidence_k) == (False, (17,))
    glimpse = ask(Views({17: {**paint(zk.B1, n=100), CELLS[50]: "occupied"}}))
    assert (glimpse.gold["value"], glimpse.evidence_k) == (False, (17,))
    taken = {**paint(zk.B1), **paint(zk.B1, "occupied", n=30, start=100)}
    before = ask(Views({12: taken, 17: paint(zk.B1, n=100)}))
    assert (before.gold, before.slice, before.evidence_k) == ({"type": "unknown"}, "unknown", (17,))
    assert ask(Views({12: paint(zk.B1), 17: taken})).gold["value"] is False
    views = Views(zone={17: paint(zk.B1)}, robot={17: paint(zk.B1, "occupied")})
    assert ask(views).gold == {"type": "yesno", "value": True} and set(views.calls) == {(0.0, (0.3, 3.5))}


def test_clearance_then_reads_up_to_the_moment():
    """Contract: `clearance_then` reads the keyframes of the look window that holds the event, up to and including it,
    never after it, and is not asked of a zone the end look window observed. Hand values on the look window 18-25: the
    bay all free at 18 and 30 cells of it taken by something that walks in at 20 (the whole bay observed again, those
    cells occupied): 'when you reached the station' (18) is yes, evidence (18,); 'when you finished at the station' (25)
    reads 18-25 and is no, evidence (20,); a bay seen only at 17 (an earlier look window) is unknown at both, slice
    unknown; a bay with a cell in the end look window (36) gives no moment; an event wording the zones do not name gives
    none; neither does a map without bays. The moment is keyed '0|<wording>', its question names the bay's entity P01
    and the time, and its text is 'Was the bay ahead of you clear of anything taller than 0.3 m when you reached the
    station?'. Regression caught: a later state leaking into a past moment, an earlier look window read for it, or a
    current-view answer."""
    walk = {**paint(zk.B1), **paint(zk.B1, "occupied", n=30, start=100)}
    s = zk.script()  # a person walks in at keyframe 19 and is seen at 20: the one change that happens in view
    s.item("M0001", "person", {}, [(0, s.t_of(19), (6.0, 1.5, 0.0)), (s.t_of(19), None, (1.5, 1.5, 0.0))],
           mobility="mover", size=(0.5, 0.5, 1.7))
    s.see(20, "M0001", RECOG)
    moments = then(Views({18: paint(zk.B1), 20: walk}), s)
    assert [m.key for m in moments] == [f"0|{REACHED}", "0|when you finished at the station"]
    assert all(m.unknown == [] and len(m.answerable) == 1 for m in moments)
    reached, finished = (m.answerable[0] for m in moments)
    assert reached.slots == {"zone": "the bay ahead of you", "height": 0.3, "event": REACHED, "task": None}
    assert (reached.gold, reached.evidence_k, reached.subjects) == ({"type": "yesno", "value": True}, (18,),
                                                                    ("w:18", "e:P01"))
    assert (finished.gold, finished.evidence_k, finished.subjects) == ({"type": "yesno", "value": False}, (20,),
                                                                       ("w:25", "e:P01"))
    assert (reached.family, reached.slice, reached.decision_k, reached.critical, reached.tolerances,
            reached.entities) == ("clearance_then", "answerable", None, True, {}, ())
    assert FAMILIES["clearance_then"].render(reached.slots) == (
        "Was the bay ahead of you clear of anything taller than 0.3 m when you reached the station?")
    earlier = [m.answerable[0] for m in then(Views({17: paint(zk.B1)}))]
    assert [(c.gold, c.slice, c.evidence_k) for c in earlier] == [({"type": "unknown"}, "unknown", ())] * 2
    assert then(Views({18: paint(zk.B1), 36: paint(zk.B1, n=1)})) == []
    unnamed = replace(zk.JOB, events={**zk.EVENTS, "when you left": EventRule("first", "undocking")})
    assert len(then(Views({18: paint(zk.B1)}), job=unnamed)) == 2
    assert then(Views({18: paint(zk.B1)}), m=NO_BAYS) == []


def test_clearance_then_names_the_task_of_a_window_of_several():
    """Contract: in a window of several tasks each task's events are their own moments, keyed '<task>|<wording>' with
    the case task index in the slot and the text, and read that task's look window: task 1's events fall at keyframes 58
    and 65 (the look window 58-65). Hand values: the bay free at 18 and, in task 1, taken at 58 by five cells: task 0's
    moments are yes, task 1's are no ('... when you reached the station in your second task?'). Regression caught: a
    moment of task 1 read over task 0's keyframes, or a task phrase on a window of one."""
    s = zk.script(2)
    taken = {**paint(zk.B1), **paint(zk.B1, "occupied", n=5, start=100)}
    moments = then(Views({18: paint(zk.B1), 58: taken}), s, a=0, b=1, length="medium")
    assert [m.key for m in moments] == [f"{i}|{w}" for i in (0, 1) for w in (REACHED, "when you finished at the "
                                                                                       "station")]
    c = [m.answerable[0] for m in moments]
    assert [x.slots["task"] for x in c] == [0, 0, 1, 1]
    assert [x.gold["value"] for x in c] == [True, True, False, False]
    assert [x.subjects[0] for x in c] == ["w:18", "w:25", "w:58", "w:65"]
    assert FAMILIES["clearance_then"].render(c[2].slots) == (
        "Was the bay ahead of you clear of anything taller than 0.3 m when you reached the station in your second "
        "task?")


def lanes_world(*, second_lane: int | None = None):
    """Four lanes in a window of the kit: L1 seen empty at 5, L2 seen with something standing in it (60 cells occupied)
    at 5, L3 seen at `second_lane` cells (never when None), L4 seen empty at 5 and again at 36 (the end look window).
    Three pallets, all sighted at 5: E0001 on L2, E0002 on L3, E0003 off the lanes (10, 8)."""
    s = zk.script()
    for eid, xy in (("E0001", (12.5, 2.0)), ("E0002", (15.0, 2.0)), ("E0003", (10.0, 8.0))):
        s.item(eid, "pallet", {"color": "brown"}, [(0, None, (*xy, 0.0))], size=(1.2, 1.0, 0.9))
        s.see(5, eid, RECOG)
    at5 = {**paint(zk.L1), **paint(zk.L2), **paint(zk.L2, "occupied", n=60, start=100), **paint(zk.L4)}
    if second_lane is not None:
        at5.update(paint(zk.L3, n=second_lane))
    return s, Views({5: at5, 36: paint(zk.L4)})


def test_slots_empty_by_hand():
    """Contract: `slots_empty` lists, among the places of the kind that the robot passed (any cell observed in the
    window), those last seen clear, as one item at each zone's centroid; a place in view at the end is a don't-care
    place (the current view), a partly seen one too, a blocked one is nothing and a place never passed is nothing.
    Hand values on lanes L1..L4 (centres (10, 2), (12.5, 2), (15, 2), (17.5, 2), zones 1.3 x 2.7 m): L1 empty: one
    item at (10, 2); L2 with 60 occupied cells: nothing; L3 never seen: nothing; L4 seen again in the end look window: a
    don't-care place at (17.5, 2). The match radius is the least of half the longest side (1.35) and 0.45 x the
    2.5 m between two lanes (1.125), at least 0.5: 1.125, on the tolerance and on the don't-care place. The evidence
    is the keyframes that decided the passed places (5, and 36 for L4); the subjects are the passed lanes' entities
    (P05, P06, P08) and the pallet standing on L2 as last seen (E0001), never E0002 (on a lane nobody passed) or E0003.
    Seen at 196 of L3's 392 cells it is passed and unknown: a don't-care place at (15, 2), and E0002 a subject.
    Regression caught: a place never seen listed as empty, a place seen at the end asked, or two end questions hinting
    at each other (the pallet that tells a slot's state)."""
    s, views = lanes_world()
    (moment,) = slot_moments(views, s)
    assert moment.key == "staging lanes" and moment.unknown == [] and len(moment.answerable) == 1
    (c,) = moment.answerable
    assert c.gold == {"type": "positions", "items": [{"xy": [10.0, 2.0]}],
                      "dont_care": [{"xy": [17.5, 2.0], "radius": 1.125}]}
    assert (c.family, c.slots, c.slice, c.tolerances, c.evidence_k, c.entities, c.decision_k, c.critical) == (
        "slots_empty", {"places": "staging lanes", "height": 0.3}, "answerable", {"match_m": 1.125}, (5, 36), (), None,
        True)
    assert c.subjects == ("e:E0001", "e:P05", "e:P06", "e:P08")
    assert FAMILIES["slots_empty"].render(c.slots) == (
        "Which staging lanes that you passed were clear of anything taller than 0.3 m when you last saw them? Give "
        "where each one is.")
    s, views = lanes_world(second_lane=196)
    part = slot_moments(views, s)[0].answerable[0]
    assert part.gold["dont_care"] == [{"xy": [15.0, 2.0], "radius": 1.125}, {"xy": [17.5, 2.0], "radius": 1.125}]
    assert part.subjects == ("e:E0001", "e:E0002", "e:P05", "e:P06", "e:P07", "e:P08")
    assert slot_moments(Views(), s) == []
    assert slot_moments(Views({36: paint(zk.L3)}), s)[0].answerable[0].gold == {
        "type": "positions", "items": [], "dont_care": [{"xy": [15.0, 2.0], "radius": 1.125}]}


def test_every_family_compiles_on_the_kit():
    """Contract: the three families go through the compiler on the kit's case, every row passing the record validators,
    one question each: the bay at the decision (yes, now), one of the two moments about the bay (they share its entity,
    so the subject rule keeps one) and the lanes (L1 seen empty: one item at (10, 2), subjects P05). The rows need the
    picture and the pose, never the depth (`needs`: see, pose; the moment's also the status), and a bay seen whole at 12
    (an earlier look window) and not in the decision's look window is still asked, `unknown` and `now`, never a `memory`
    row. Regression caught: a family whose rows the harness could not read (a subject kind, a tolerance, a scope no
    validator allows), a `needs` the camera-only channel cannot meet, or a `zone_clear` row of the memory column."""
    s = zk.script()
    views = Views({5: paint(zk.L1), 17: paint(zk.B1), 18: paint(zk.B1), 25: paint(zk.B1)})
    qs, gs = questions.compile_inputs(zk.inputs(s, views), zk.record(s))
    assert Counter(g["family"] for g in gs) == {"zone_clear": 1, "clearance_then": 1, "slots_empty": 1}
    by = {g["family"]: (q, g) for q, g in zip(qs, gs, strict=True)}
    q, g = by["zone_clear"]
    assert q["text"] == "Is the bay ahead of you clear of anything taller than 0.3 m right now?"
    assert (q["answer_type"], q["needs"], q["decision_k"], g["gold"], g["scope"], g["credit"], g["slice"]) == (
        "yesno", ["see", "pose"], K, {"type": "yesno", "value": True}, "now", "exact", "answerable")
    q, g = by["clearance_then"]
    assert q["text"] in {"Was the bay ahead of you clear of anything taller than 0.3 m when you reached the station?",
                         "Was the bay ahead of you clear of anything taller than 0.3 m when you finished at the "
                         "station?"}
    assert (q["needs"], g["gold"], g["scope"], g["subjects"][1:]) == (["see", "pose", "status"],
                                                                       {"type": "yesno", "value": True}, "memory",
                                                                       ["e:P01"])
    q, g = by["slots_empty"]
    assert (q["answer_type"], q["needs"], g["gold"], g["credit"], g["tolerances"], g["subjects"], g["evidence_k"]) == (
        "positions", ["see", "pose"], {"type": "positions", "items": [{"xy": [10.0, 2.0]}]}, "f1", {"match_m": 1.125},
        ["e:P05"], [5])
    earlier = Views({12: paint(zk.B1)})   # seen whole in an earlier look window, nothing in the decision's
    _, golds = questions.compile_inputs(zk.inputs(s, earlier), zk.record(s))
    (row,) = [g for g in golds if g["family"] == "zone_clear"]
    assert (row["gold"], row["scope"], row["slice"], row["evidence_k"]) == ({"type": "unknown"}, "now", "unknown", [])


def test_hidden_changes_never_change_a_gold():
    """Contract (truth rule 2): a gold reads what depth showed, never where things are. Pallets added in the bay B1 and
    the lane L1 that no frame saw (the states unchanged, no sighting) change none of the four candidates the kit
    asks (the bay at the decision, both moments, the lanes); a 0.5 m box added in the bay in the scene a frame
    ray-cast (keyframes 17 and 18) changes the gold to no. Regression caught: a gold reading footprints or the
    timeline's placements instead of the states."""
    views = Views({5: paint(zk.L1), 17: paint(zk.B1), 18: paint(zk.B1)})

    def golds(s, views):
        w = zk.world(s, views)
        found = [questions_zone.zone_clear(w, K)]
        found += [c for m in [*questions_zone.clearance_then(w), *questions_zone.slots_empty(w)] for c in m.answerable]
        return found

    base = golds(zk.script(), views)
    assert [c.family for c in base] == ["zone_clear", "clearance_then", "clearance_then", "slots_empty"]
    hidden = zk.script()
    for eid, xy in (("E0009", (1.5, 1.5)), ("E0010", (10.0, 2.0))):
        hidden.item(eid, "pallet", {"color": "red"}, [(0, None, (*xy, 0.0))], size=(1.2, 1.0, 0.9))
    assert golds(hidden, views) == base
    empty, boxed = zk.look(zk.B1), zk.look(zk.B1, [box(1.2, 1.6, 1.2, 1.8, 0.0, 0.5)])
    for cells, want in ((empty, True), (boxed, False)):
        found = golds(hidden, Views({5: paint(zk.L1), 17: cells, 18: cells}))
        assert [c.gold["value"] for c in found[:3]] == [want] * 3


def test_a_zone_band_keeps_only_the_zones_cells(tmp_path, monkeypatch):
    """Contract: a case read from its files observes a zone with the zone's own ground and band and keeps only the cells
    inside the job's zones of that band; the robot's band is kept whole, and a band no zone has holds nothing. Hand
    values on a frame of the tote robot at (3.9, 6.4) facing west over a bare floor, with the job's bays and lanes on
    `warehouse_a` (the real map): the bay F22 (1.5, 6.4, zone x 0.75..2.25, y 5.35..7.45: u 7..22, v 53..74) is seen
    whole and free, and the other cells that band holds lie in F23's (1.5, 4.2: v 31..52, partly in view); the
    robot's band holds more than a thousand cells, among them one at u 25, v 64, which no zone has. Regression
    caught: a cache that keeps whole drone frames (tens of thousands of cells per keyframe, a table process each) or
    a zone read through another zone's band."""
    use_job(monkeypatch, zk.JOB)
    s = SpaceScript(tasks=[[("dock", 3)]], stage=WAREHOUSE, poses={0: (3.9, 6.4)}, yaws={0: 3.141592653589793},
                    depth_ks={0})
    inp = questions_world.load_inputs(s.write_case(tmp_path))
    zone = inp.states_of([0], 0.0, (0.3, 3.5))[0]
    f22, f23 = ({(u, v) for u in range(7, 23) for v in vs} for vs in (range(53, 75), range(31, 53)))
    assert f22 <= set(zone) <= f22 | f23 and all(zone[c] == "free" for c in f22)
    robot = inp.states_of([0])[0]
    assert len(robot) > 1000 and (25, 64) in robot and (25, 64) not in zone
    assert inp.states_of([0], 0.0, (0.5, 3.5))[0] == {}
    assert inp.states_of([0], 0.0, (0.3, 3.5))[0] is zone


def test_the_zone_rules_are_found_by_their_words():
    """Contract: a table system finds a question's rule by its words, so `check_rules` refuses a job whose zones are
    ambiguous or unnamed, naming the job: two different rules with one wording (across `zones` and `slot_zones`),
    two different slot rules with one `piece_words`, a slot rule with none, a key that is neither a decision kind nor
    an event wording, and a decision kind that asks `zone_clear` with no zone; one rule under two keys, and
    every registered job, pass. Regression caught: a question whose rule a system could not find, or finds wrongly."""
    check_rules = questions_zone.check_rules
    check_rules(zk.JOB)
    for job in job_registry.all_jobs():
        check_rules(job)

    def refused(match, **fields):
        with pytest.raises(ValueError, match=rf"amr/1.*{match}"):
            check_rules(replace(zk.JOB, **fields))

    other = ZoneRule("the bay ahead of you", "bay", margin_m=0.1)
    refused("share the words", zones={**zk.JOB.zones, REACHED: other})
    refused("share the words", slot_zones=(ZoneRule("the bay ahead of you", "bay", piece_words="bays"),))
    refused("piece_words", slot_zones=(zk.LANES, ZoneRule("the other lanes", "staging_lane",
                                                          piece_words="staging lanes")))
    refused("no piece_words", slot_zones=(ZoneRule("the lanes", "staging_lane"),))
    refused("neither a decision kind nor an event", zones={**zk.JOB.zones, "when you slept": zk.BAY})
    refused("asks zone_clear and has no zone", zones={REACHED: zk.BAY})
    check_rules(replace(zk.JOB, zones={**zk.JOB.zones, "before_crossing": zk.BAY}))


def test_look_window_of_is_the_stream_look_window_that_holds_a_step():
    """Contract: `World.window_of(k)` is the look window the stream puts script keyframe k in, whether or not it ends
    there (`window_ks` is the one that ends there). Hand values from the cuts at 17 and 39: 18 is in 18-25, 17 in 16-17,
    39 in 34-39; a keyframe outside the window is an error. Regression caught: the event's look window read as the
    decision's."""
    w = zk.world(zk.script(2), Views(), 1, 1)
    assert (w.window_of(58), w.window_of(57), w.window_of(79)) == (range(58, 66), range(56, 58), range(74, 80))
    assert w.window_ks(57) == range(56, 58)
    with pytest.raises(ValueError, match="outside the window"):
        w.window_of(10)
