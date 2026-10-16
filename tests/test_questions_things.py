"""`when_seen`, `order` and `distance_between` (questions_things), by hand on `things_kit`'s flat task (keyframes 1.5 s
apart from the case's start clock 2026-03-02T08:00:00; no decision point, so the end look window is keyframes 56 to 59)
and its window of several tasks. Draws are the registry's own hash (`questionnaire.unknown_slice`), counted over ids."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace

import things_kit as tk
from space_kit import SpaceScript
from stbench import questionnaire, questions, score
from stbench.job_registry import Relevance
from stbench.questions_swap import gold_changes_under_swap
from stbench.questions_text import FAMILIES
from stbench.questions_things import (
    GOLDS,
    MIN_APART_M,
    ORDER_GAP_S,
    SLICED,
    TIME_LADDER_S,
    distance_between,
    kinds,
    order,
    settled,
    when_seen,
)
from stbench.questions_where import never_seen
from stbench.records import check_family, check_gold, check_question

LADDER = {"ladder": [15.0, 30.0, 60.0, 120.0]}


def pools(moments, key):
    (m,) = [m for m in moments if m.key == key]
    return m.answerable, m.unknown


def asked_order(later, first=(10, 20), cid="c0"):
    """The `order` candidates of a flat task with red shoes recognisable at `first` and blue shoes shown as `later`."""
    s = tk.flat()
    tk.put(s, "E0001", "shoes", "red", recognised=first)
    tk.put(s, "E0002", "shoes", "blue", **later)
    return pools(order(tk.world(s, cid=cid)), "shoes")[0]


def test_the_constants_are_the_briefs():
    """Contract: the minute-scale ladder, the order gap and the distance floor are fixed before data (design §8.3,
    the brief), the three specs pass `check_family`, and the registry holds the module's families with `when_seen` and
    `distance_between` sliced and `order` not. Regression caught: a changed rung or floor, or a spec the harness cannot
    read."""
    assert (TIME_LADDER_S, ORDER_GAP_S, MIN_APART_M) == ((15.0, 30.0, 60.0, 120.0), 30.0, 0.5)
    assert set(GOLDS) == {"when_seen", "order", "distance_between"}
    assert set(SLICED) == {"when_seen", "distance_between"}
    for name in GOLDS:
        assert check_family(FAMILIES[name]) == [] and FAMILIES[name].when == "end"


def test_when_seen_by_hand():
    """Contract: a settled thing (a recognisable sighting in the window, no row in the end look window, no sighting at
    or above the glimpse bound after its last recognisable one) is asked with the case clock of that last recognisable
    sighting as its gold and evidence, on the 15, 30, 60 and 120 s ladder; the unseen things and the descriptions the
    world lacks are the unknown pool; a thing that only glimpsed, or that a later glimpse or dark sighting follows, or
    that the end look window shows at any pixel count, is in neither; no same-task rule (a thing last seen in task 0 of
    a window over two is asked). Hand-worked on a flat task of 60 keyframes (the end look window 52-59), bags of one
    colour each:
    - red recognisable at k 10 and 30: asked, 30 x 1.5 s = 45 s, '2026-03-02T08:00:45.0', evidence 30;
    - blue recognisable at 10 and 20, glimpsed at 40: skipped (the table would read k 40); green recognisable at 20
      and again at 58 (end look window): skipped; black recognisable at 20 and seen dark (under the luminance floor) at
      45: skipped; brown only glimpsed at 5: skipped;
    - grey never seen: unknown, with the one bag colour the world lacks (orange); a white bag that exists only from
      100 s, after the window's last keyframe (88.5 s), is not a thing the robot could have seen: in neither pool.
    A window of two tasks (28 keyframes each) with a bag recognisable at k 4 only is asked with '2026-03-02T08:00:06.0'.
    Regression caught: the time of an unseen event, or of a sighting the table and the gold read differently (a later
    glimpse), or a thing the current view answers."""
    s = tk.flat()
    tk.put(s, "E0001", "bag", "red", recognised=[10, 30])
    tk.put(s, "E0002", "bag", "blue", recognised=[10, 20], glimpsed=[40])
    tk.put(s, "E0003", "bag", "green", recognised=[20, 58])
    tk.put(s, "E0004", "bag", "black", recognised=[20], dark=[45])
    tk.put(s, "E0005", "bag", "brown", glimpsed=[5])
    tk.put(s, "E0006", "bag", "grey")
    s.item("E0007", "bag", {"color": "white"}, [(100.0, None, (1.0, 1.0, 0.0))])
    w = tk.world(s)
    assert w.end_window == range(52, 60)
    moments = when_seen(w)
    assert [m.key for m in moments] == ["bag", "box", "shoes", "toy"]
    answerable, unknown = pools(moments, "bag")
    (c,) = answerable
    assert (c.slots, c.gold, c.slice, c.evidence_k, c.entities, c.tolerances, c.subjects) == (
        {"kind": "bag", "attributes": {"color": "red"}}, {"type": "time", "clock": "2026-03-02T08:00:45.0"},
        "answerable", (30,), ("E0001",), LADDER, ("e:E0001",))
    assert FAMILIES["when_seen"].render(c.slots) == "When did you last see the red bag?"
    subjects = {c.subjects[0] for c in unknown}
    assert "e:E0006" in subjects and subjects - {"e:E0006"} <= {'d:bag/{"color":"orange"}'}
    assert all(c.gold == {"type": "unknown"} and c.evidence_k == () and c.tolerances == LADDER for c in unknown)
    two = SpaceScript(tasks=[[("kitchen", 28, "cleaning")], [("kitchen", 28, "cleaning")]], decisions=[],
                      depth_ks=set())
    tk.put(two, "E0001", "bag", "red", recognised=[4])
    (c,), _ = pools(when_seen(tk.world(two, 0, 1, length="medium")), "bag")
    assert (c.gold, c.evidence_k) == ({"type": "time", "clock": "2026-03-02T08:00:06.0"}, (4,))


def test_order_by_hand_and_balanced():
    """Contract: two settled things of one kind are asked when their first recognisable sightings lie at least 30 s
    apart and the later-seen one has no visibility row (any pixel count) at or before the earlier one's first
    recognisable sighting; the evidence is the two first sightings sorted with their entities; which is named first is
    a coin per pair that does not read the answer, so the gold is yes exactly when the named-first one is the earlier,
    and across case ids yes is between 0.4 and 0.6 of the answers within each kind. Hand-worked, red shoes
    recognisable at k 10 and 20 (15 s), blue shoes as given:
    - blue at 40 (60 s): asked, evidence (10, 40), entities (E0001, E0002); the text names red first with gold yes or
      blue first with gold no, and both orders occur over 40 ids;
    - blue at 30 (exactly 30.0 s): asked; at 29 (28.5 s) and at 23 (19.5 s): not asked;
    - blue recognisable at 40 but glimpsed at 5, before red's first: not asked; detected (400 px) at 10, red's own
      keyframe: not asked; detected at 11: asked.
    Over 200 ids on a window of four tasks (bags and toys 6 pairs, shoes and boxes 3) the yes share of each kind is in
    [0.4, 0.6]. Regression caught: a form that predicts the answer (the naming order), or a pair the table cannot
    order."""
    (c,) = asked_order({"recognised": [40]})
    assert (c.evidence_k, c.entities, c.gold["type"], c.slice) == ((10, 40), ("E0001", "E0002"), "yesno", "answerable")
    texts = {("Did you first see the pair of red shoes before the pair of blue shoes?", True),
             ("Did you first see the pair of blue shoes before the pair of red shoes?", False)}
    seen = set()
    for cid in tk.ids("order text", 40):
        (c,) = asked_order({"recognised": [40]}, cid=cid)
        seen.add((FAMILIES["order"].render(c.slots), c.gold["value"]))
        red_first = c.slots["first"]["attributes"]["color"] == "red"
        assert c.subjects == (("e:E0001", "e:E0002") if red_first else ("e:E0002", "e:E0001"))
    assert seen == texts
    assert len(asked_order({"recognised": [30]})) == 1
    for none in ({"recognised": [29]}, {"recognised": [23]}, {"recognised": [40], "glimpsed": [5]},
                 {"recognised": [40], "detected": [10]}):
        assert asked_order(none) == [], none
    assert len(asked_order({"recognised": [40], "detected": [11]})) == 1
    yes: Counter = Counter()
    asks: Counter = Counter()
    for w in tk.worlds(tk.things_window(4), tk.ids("order balance", 200), 0, 3, "long"):
        for m in order(w):
            asks[m.key] += len(m.answerable)
            yes[m.key] += sum(c.gold["value"] for c in m.answerable)
    assert asks == {"bag": 1200, "box": 600, "shoes": 600, "toy": 1200}
    for kind, n in asks.items():
        assert 0.4 <= yes[kind] / n <= 0.6, (kind, yes[kind] / n)


def test_order_names_one_kind():
    """Contract: an `order` question names two things of one kind, so the kind pair a form could read is never a clue to
    the answer (the review's M3): every `order` moment is a kind and holds only pairs of it, and over 50 case ids of a
    long window (six tasks, four kinds of things) every compiled `order` question has `first.kind` equal to
    `second.kind`, all four kinds turn up, and the two descriptions differ. Regression caught: a pair of kinds whose
    homes predict the answer (shoes on the hall's rack, toys on rugs)."""
    s = tk.things_window(6)
    asked = []
    for w in tk.worlds(s, tk.ids("one kind", 50), 0, 5, "long"):
        for m in order(w):
            assert m.answerable and all({c.slots["first"]["kind"], c.slots["second"]["kind"]} == {m.key}
                                        for c in m.answerable)
        qs, _ = questions.compile_inputs(w.inp, w.rec)
        asked += [q["slots"] for q in qs if q["template"] == "order"]
    assert len(asked) >= 50 and {a["first"]["kind"] for a in asked} == set(tk.KINDS)
    assert all(a["first"]["kind"] == a["second"]["kind"] and a["first"] != a["second"] for a in asked)


def test_distance_by_hand():
    """Contract: two settled things are asked when their positions at their last recognisable sightings are at least 0.5
    m apart; the gold is that distance in metres (3 places) from the sighting's placement, never the true place later;
    the evidence is the two sightings, sorted by keyframe with their entities; the moments are the unordered pairs of
    relevance kinds, one kind with itself included; the unknown pool is each settled thing with an unseen one of the
    pair's other kind. Hand-worked on a flat task: bags red at (1, 1) seen at k 10, blue at (4, 5) seen at 20 and moved
    to (7, 7) at 45 s, green at (1, 1.3) seen at 15, and a black box at (1, 4) seen at 25:
    - bag|bag: red-blue 5.0 (3-4-5, evidence 10 and 20, not the 8.485 to the place it moved to), blue-green sqrt(22.69)
      = 4.763 (evidence 15 and 20); red-green, 0.3 m apart, is not asked;
    - bag|box: red 3.0, blue sqrt(10) = 3.162, green 2.7 from the box; box|box and every moment of a kind with no thing
      have no answerable pair;
    - the unknown pool of bag|box is the three settled bags each with every box description the world lacks and the
      settled box with every bag description it lacks; each row names one settled thing and one absent description.
    Exactly 0.5 m apart is asked, 0.49 is not. Regression caught: a relative gold near 0, or the true place leaking."""
    s = tk.flat()
    tk.put(s, "E0001", "bag", "red", (1.0, 1.0), recognised=[10])
    tk.put(s, "E0002", "bag", "blue", (4.0, 5.0), recognised=[20], moves=(45.0, (7.0, 7.0)))
    tk.put(s, "E0003", "bag", "green", (1.0, 1.3), recognised=[15])
    tk.put(s, "E0004", "box", "black", (1.0, 4.0), recognised=[25])
    w = tk.world(s)
    moments = distance_between(w)
    assert [m.key for m in moments] == ["bag|bag", "bag|box", "bag|shoes", "bag|toy", "box|box", "box|shoes",
                                        "box|toy", "shoes|shoes", "shoes|toy", "toy|toy"]

    def golds(key):
        return {c.entities: (c.gold, c.evidence_k) for c in pools(moments, key)[0]}

    def number(v):
        return {"type": "number", "value": v}

    assert golds("bag|bag") == {("E0001", "E0002"): (number(5.0), (10, 20)),
                                ("E0003", "E0002"): (number(4.763), (15, 20))}
    assert golds("bag|box") == {("E0001", "E0004"): (number(3.0), (10, 25)),
                                ("E0002", "E0004"): (number(3.162), (20, 25)),
                                ("E0003", "E0004"): (number(2.7), (15, 25))}
    assert [m.key for m in moments if m.answerable] == ["bag|bag", "bag|box"]
    absent = Counter(kind for kind, _ in never_seen(w))
    assert absent["bag"] > 0 and absent["box"] > 0
    unknown = pools(moments, "bag|box")[1]
    assert len(unknown) == 3 * absent["box"] + absent["bag"]
    for c in unknown:
        held = [x for x in c.subjects if x.startswith("e:")]
        assert len(held) == 1 and held[0] in {"e:E0001", "e:E0002", "e:E0003", "e:E0004"}
        assert c.gold == {"type": "unknown"} and c.evidence_k == () and c.entities == ()
        assert c.tolerances == {"ladder": [0.25, 0.5, 1.0, 2.0]}  # the home catalog's things widen by 0
        assert sorted(slot["kind"] for slot in (c.slots["first"], c.slots["second"])) == ["bag", "box"]
    texts = set()
    for cid in tk.ids("distance text", 30):
        both = pools(distance_between(tk.world(s, cid=cid)), "bag|bag")[0]
        (c,) = [c for c in both if c.entities == ("E0001", "E0002")]
        assert c.gold == number(5.0)
        texts.add(FAMILIES["distance_between"].render(c.slots))
    assert texts == {f"How far apart were the {a} bag and the {b} bag, as you last saw them? Answer in metres."
                     for a, b in (("red", "blue"), ("blue", "red"))}

    def apart(dx):
        t = tk.flat()
        tk.put(t, "E0001", "shoes", "red", (1.0, 1.0), recognised=[10])
        tk.put(t, "E0002", "shoes", "blue", (1.0 + dx, 1.0), recognised=[12])
        return [c.gold for c in pools(distance_between(tk.world(t)), "shoes|shoes")[0]]

    assert apart(0.5) == [number(0.5)] and apart(0.49) == []


def test_unknown_slices_per_moment():
    """Contract: `when_seen` and `distance_between` draw their unknown slice per moment (the registry's hash, 20%), and
    `order` draws none; every moment of the long window has both pools, so every moment can be asked either way and
    no kind or pair predicts its slice. Hand-worked on six tasks of four kinds (the unknown pool: the unseen bag and toy
    and the absent descriptions): over 200 ids each moment's unknown share is in [0.12, 0.28]; every `order` moment has
    no unknown candidate. Regression caught: a kind that predicts the slice (a moment with an empty pool is never asked
    as unknown), or `order` given a slice."""
    w = next(iter(tk.worlds(tk.things_window(6), ["c0"], 0, 5, "long")))
    ids = tk.ids("slice share", 200)
    for family, gold in (("when_seen", when_seen), ("distance_between", distance_between)):
        moments = gold(w)
        assert len(moments) == (4 if family == "when_seen" else 10)
        for m in moments:
            assert m.answerable and m.unknown, (family, m.key)
            share = sum(questionnaire.unknown_slice(cid, family, m.key) for cid in ids) / len(ids)
            assert 0.12 <= share <= 0.28, (family, m.key, share)
    assert all(m.answerable and not m.unknown for m in order(w))


def test_hidden_changes_never_change_a_gold():
    """Contract (VISION rule 4): what no frame showed changes no gold. Over six tasks' worth of things (three tasks
    here) with two identical grey toys, both seen: neither is ever a subject (their description is not unique), and
    swapping them from the start changes no candidate's gold, while swapping two different bags does (the check is not
    vacuous: the black bag is then seen where the red one was); adding a box that no frame shows, of a description no
    world has, changes the gold and slice of no candidate that was asked before and adds only unknown ones. The log has
    no `rgb_med` (the label stands for the picture): on a log with `rgb_med` the exchange of two differently coloured
    bags shows a black bag in red, which the render screen drops rather than reports (the question is no longer asked,
    so no gold changes). Regression caught: a gold that reads which of two identical things is which, or an unseen
    thing's existence."""
    s = tk.things_window(3)
    s.rgb_med = False
    tk.put(s, "E0090", "toy", "grey", (6.0, 6.0), recognised=[3, 30])
    tk.put(s, "E0091", "toy", "grey", (6.0, 7.0), recognised=[4, 31])
    w = tk.world(s, 0, 2, "c7", "medium")
    cands = questionnaire.candidates(w)
    assert {"when_seen", "order", "distance_between"} <= {c.family for c in cands}
    assert not any({"e:E0090", "e:E0091"} & set(c.subjects) for c in cands)
    assert gold_changes_under_swap(w.inp, w.rec, "E0090", "E0091", 0.0) == []
    changed = gold_changes_under_swap(w.inp, w.rec, "E0001", "E0002", 0.0)
    assert {"when_seen", "distance_between"} <= {json.loads(c["question"])[0] for c in changed}

    def asked(script):
        return {c.key: (c.gold, c.slice) for c in questionnaire.candidates(tk.world(script, 0, 2, "c7", "medium"))}

    before = asked(s)
    tk.put(s, "E0099", "box", "pink", (6.0, 6.0))
    after = asked(s)
    assert all(after.get(key) == value for key, value in before.items())
    assert {slice_ for key, (_, slice_) in after.items() if key not in before} == {"unknown"}


def test_kinds_are_the_relevance_kinds_of_every_task_of_the_window():
    """Contract: the kinds a window asks about are those of every task's relevance (here the job asks bags in the
    kitchen and shoes in the living room), and a thing of a kind no task asks is never a subject. Hand-worked on two
    tasks of 30 keyframes (the second in the living room): a bag at k 5, shoes at k 35 and a toy at k 6; the window of
    both has the moments bag and shoes (and no toy), the window of the second alone has shoes only. Regression caught:
    the last task's relevance alone (the window's earlier things never asked), or things of any kind."""
    job = replace(tk.THINGS_JOB, relevance=lambda plan: Relevance(
        frozenset({"bag"} if plan.params["room"] == "kitchen" else {"shoes"}), frozenset(), frozenset()))
    s = SpaceScript(tasks=[[("kitchen", 30, "cleaning")], [("living room", 30, "cleaning")]], decisions=[],
                    depth_ks=set())
    tk.put(s, "E0001", "bag", "red", recognised=[5])
    tk.put(s, "E0002", "shoes", "red", recognised=[35])
    tk.put(s, "E0003", "toy", "red", recognised=[6])
    both, second = tk.world(s, 0, 1, length="medium", job=job), tk.world(s, 1, length="short", job=job)
    assert (kinds(both), kinds(second)) == (["bag", "shoes"], ["shoes"])
    assert [(m.key, len(m.answerable)) for m in when_seen(both)] == [("bag", 1), ("shoes", 1)]
    assert [(m.key, len(m.answerable)) for m in when_seen(second)] == [("shoes", 1)]
    assert not any("e:E0003" in c.subjects for m in when_seen(both) for c in m.answerable + m.unknown)


def test_rows_pass_the_validators():
    """Contract: every row compiled for the three families passes `check_question` and `check_gold`, its text is its
    family's render of its slots, a `distance_between` gold is at least the 0.5 m floor and scored on the ladder (an
    answer 0.73 m off scores the rungs 1 m and 2 m: 2 of 4, where the relative credit of a 2 m gold gave it 0.04), a
    `when_seen` gold is a clock, and each family is asked on a window of two tasks and on a long one under the subject
    rule (no two questions share an entity or a description, and at most one an absent description). Hand-worked on
    `things_window`: over 30 ids of each window every one of the three families is asked at least once in all, and no
    rows are invalid. Regression caught: a row the harness could not read, a distance credit that falls to a few
    percent for an error of under a metre, or two questions that name one thing."""
    for n, length in ((2, "medium"), (6, "long")):
        asked: Counter = Counter()
        for w in tk.worlds(tk.things_window(n), tk.ids(f"rows {n}", 30), 0, n - 1, length):
            qs, gs = questions.compile_inputs(w.inp, w.rec)
            subjects = [x for g in gs for x in g["subjects"] if x.startswith(("e:", "d:"))]
            assert len(subjects) == len(set(subjects))
            assert sum(x.startswith("d:") for x in subjects) <= 1
            for q, g in zip(qs, gs, strict=True):
                assert check_question(q) == [] and check_gold(g) == []
                assert q["text"] == FAMILIES[q["template"]].render(q["slots"])
                if q["template"] in GOLDS:
                    asked[q["template"]] += 1
                if g["family"] == "distance_between" and g["slice"] == "answerable":
                    assert g["gold"]["value"] >= MIN_APART_M and g["credit"] == "ladder"
                    off = {"type": "number", "value": g["gold"]["value"] - 0.73}
                    assert score.credit(off, g) == 0.5
        assert set(asked) == set(GOLDS), (n, asked)


def test_settled_holds_through_same_place_sightings_and_fails_on_a_later_glimpse_elsewhere():
    """Contract: a thing is settled when it has a recognisable sighting, none in the end look window, and every later
    sighting (a glimpse, a detection, a clipped or depth-less frame) lies within 0.15 m of its last recognisable place,
    so the memory table, which reads its latest sighting, holds it where the gold read it; a later sighting 0.15 m or
    more away unsettles it (the table would hold it there), unless a recognisable sighting follows, which is the last.
    Hand-worked on a flat task (bags, the end look window 56-59), `last` the last recognisable keyframe:
    - red (1, 1) recognisable at 10, detected (400 px) at 20 and 45: settled, last 10;
    - blue (2, 1) recognisable at 10, moved 0.1 m at 45 s (keyframe 30), glimpsed at 40: settled, last 10;
    - green (3, 1) the same but moved 0.2 m: not settled;
    - black (4, 1) recognisable at 10 and 1,500 px at 20 with its box on the image's edge: that frame is no sighting
      the gold reads, and it is in the same place: settled, last 10;
    - brown (5, 1) recognisable at 10, moved to (5, 2.5) at 45 s and recognisable at 50: settled, first 10, last 50;
    - orange recognisable at 10 and 58 (the end look window) and white at 1,500 px with no depth at 10: not settled.
    Regression caught: an oracle under ceiling (a clipped exit frame or a detection a few centimetres off dropping the
    gold, 112 of 167 in the M0 measure), or a thing that moved and was glimpsed there kept at its old place."""
    s = tk.flat()
    tk.put(s, "E0001", "bag", "red", (1.0, 1.0), recognised=[10], detected=[20, 45])
    tk.put(s, "E0002", "bag", "blue", (2.0, 1.0), recognised=[10], glimpsed=[40], moves=(45.0, (2.1, 1.0)))
    tk.put(s, "E0003", "bag", "green", (3.0, 1.0), recognised=[10], glimpsed=[40], moves=(45.0, (3.2, 1.0)))
    tk.put(s, "E0004", "bag", "black", (4.0, 1.0), recognised=[10])
    s.see(20, "E0004", 1500, bbox=[0, 100, 40, 140])
    tk.put(s, "E0005", "bag", "brown", (5.0, 1.0), recognised=[10, 50], moves=(45.0, (5.0, 2.5)))
    tk.put(s, "E0006", "bag", "orange", (6.0, 1.0), recognised=[10, 58])
    tk.put(s, "E0007", "bag", "white", (7.0, 1.0))
    s.see(10, "E0007", 1500, depth=None)
    found = {x.ref.entity: (x.first, x.last) for x in settled(tk.world(s))["bag"]}
    assert found == {"E0001": (10, 10), "E0002": (10, 10), "E0004": (10, 10), "E0005": (10, 50)}


def test_distance_rows_carry_the_widened_ladder(monkeypatch):
    """Contract: a `distance_between` row's ladder is the distance rungs 0.25, 0.5, 1 and 2 m plus the widening of both
    things' kinds (`rules.widening`: half the longer side beyond 0.25 m), on the answerable and the unknown rows alike,
    so the credit is absolute and a large thing's surface is not charged as an error. Hand values with a bag 1.0 x 0.4
    x 0.5 m (widening 0.25) and a box 1.3 x 0.4 x 0.5 m (0.4): bag-bag 0.5 more, so 0.75, 1, 1.5 and 2.5; bag-box 0.65
    more, so 0.9, 1.15, 1.65 and 2.65, on the answerable pair and on every unknown row of the pair's moment. Regression
    caught: the relative credit (3.6 cm of credit for an error of 0.73 m on a 20 m distance), or a widening counted for
    one thing of the two."""
    monkeypatch.setattr("stbench.questions_where.item_size",
                        lambda kind: {"bag": (1.0, 0.4, 0.5), "box": (1.3, 0.4, 0.5)}.get(kind, (0.1, 0.1, 0.1)))
    s = tk.flat()
    tk.put(s, "E0001", "bag", "red", (1.0, 1.0), recognised=[10])
    tk.put(s, "E0002", "bag", "blue", (4.0, 5.0), recognised=[20])
    tk.put(s, "E0003", "box", "black", (1.0, 4.0), recognised=[25])
    moments = distance_between(tk.world(s))
    (bags,), _ = pools(moments, "bag|bag")
    boxes, unknown = pools(moments, "bag|box")
    assert bags.tolerances == {"ladder": [0.75, 1.0, 1.5, 2.5]}
    assert {tuple(c.tolerances["ladder"]) for c in boxes + unknown} == {(0.9, 1.15, 1.65, 2.65)}
    assert len(unknown) > 2


def test_distance_rows_take_their_rungs_from_the_job():
    """Contract: a `distance_between` row's rungs are the job's ladder (`JobSpec.ladder`, the unit of its place) plus
    the two things' widening, on the answerable and the unknown rows alike; a job that sets none has the home's 0.25,
    0.5, 1 and 2 m. Hand values with a job whose ladder is 1, 2, 4 and 8 m (the drone's) and two bags (no widening): the
    answerable pair's rungs and every unknown row's are 1, 2, 4, 8. Regression caught: rungs fixed in the family, which
    scores a drone's 7 m sight lines on a home's 25 cm."""
    s = tk.flat()
    tk.put(s, "E0001", "bag", "red", (1.0, 1.0), recognised=[10])
    tk.put(s, "E0002", "bag", "blue", (4.0, 5.0), recognised=[20])
    wide = replace(tk.THINGS_JOB, ladder=(1.0, 2.0, 4.0, 8.0))
    moments = distance_between(tk.world(s, job=wide))
    (bags,), _ = pools(moments, "bag|bag")
    rows = [bags] + [c for m in moments for c in m.unknown]
    assert len(rows) > 1 and {tuple(c.tolerances["ladder"]) for c in rows} == {(1.0, 2.0, 4.0, 8.0)}
    (home,), _ = pools(distance_between(tk.world(s)), "bag|bag")
    assert home.tolerances == {"ladder": [0.25, 0.5, 1.0, 2.0]}


def test_a_distance_row_is_critical_when_both_things_are():
    """Contract: an answerable `distance_between` row is critical when each of its two things is critical at its last
    recognisable place (`questions_where.critical`: a relevance kind, in a room of a kind the task works in), and the
    unknown rows are critical by kind. Hand-worked on a flat task (the hand map's kitchen x 0-4 and living room x 4-8,
    y 0-3, both asked): bags red (1, 1), blue (3, 1) and green (1, 4), the last in no room: red-blue (2 m) is
    critical, red-green (3 m) and blue-green (3.606 m) are not; every unknown row is. Regression caught: every row
    critical whatever the places, which leaves the conditioning split no situational rows to read."""
    s = tk.flat()
    tk.put(s, "E0001", "bag", "red", (1.0, 1.0), recognised=[10])
    tk.put(s, "E0002", "bag", "blue", (3.0, 1.0), recognised=[12])
    tk.put(s, "E0003", "bag", "green", (1.0, 4.0), recognised=[14])
    answerable, _ = pools(distance_between(tk.world(s)), "bag|bag")
    assert {c.entities: c.critical for c in answerable} == {("E0001", "E0002"): True, ("E0001", "E0003"): False,
                                                            ("E0002", "E0003"): False}
    unknown = [c for m in distance_between(tk.world(s)) for c in m.unknown]
    assert unknown and all(c.critical for c in unknown)
