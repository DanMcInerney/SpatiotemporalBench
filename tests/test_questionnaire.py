"""The questionnaire: the decision draw, the end questions, the subject rule and the events, by hand on `space_kit`'s
run (events at keyframes 2, 7, 18 and 26, one task of 28 keyframes) and on longer tasks built for the draw."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

import family_kit as fk
from family_kit import fake_cand, fake_spec, multi_run, register, things_by_task
from space_kit import EVENT_KS, JOB, SpaceScript, inputs, record, things_run, vacuum_run
from stbench import job_registry, questionnaire, robots
from stbench.events import EventRule, resolve
from stbench.questionnaire import GAP_STEPS, MAX_DURING, conflicts, drawn
from stbench.questions_select import Moment
from stbench.questions_world import World
from stbench.seeding import stable_hash, stable_hex
from test_architecture import MODULES, string_constants

BAG = (0.374, 0.304, 0.314)


def long_task(decisions, frames=90):
    """One task of `frames` keyframes (docked, then cleaning) with before_doorway decision points at `decisions`."""
    return SpaceScript(tasks=[[("kitchen", frames, "cleaning")]], decisions=[(k, "before_doorway") for k in decisions],
                       depth_ks=set())


def world(script, cid="c0", length="short", job=JOB, a=0, b=None):
    return World(inputs(script, job), record(script, a=a, b=b, cid=cid, length=length))


def test_asked_points_are_spaced():
    """Contract: asked points are at least 16 steps apart, and a short case asks at most 3. Hand-worked on a 90-step
    task with decisions at 3 (before `NOW_MIN_STEPS`, which `JOB`'s `now` family asks from: never), 10, 15, 16, 20, 31,
    32, 47, 63 and 80: every case id draws a set whose members are 16 or more apart, at most 3, and maximal (a candidate
    left out is within 15 steps of one kept, or the quota is full); 16 and 20 are never both asked. Regression caught:
    asks bunched where the world is interesting (`test_questionnaire_minimum` has the minimum into the task)."""
    candidates = [3, 10, 15, 16, 20, 31, 32, 47, 63, 80]
    script = long_task(candidates)
    for i in range(40):
        kept = [d.k for d in drawn(world(script, cid=stable_hex("spaced", i, n=16)))]
        assert kept == sorted(kept) and set(kept) <= set(candidates[1:])
        assert all(b - a >= GAP_STEPS for a, b in zip(kept, kept[1:], strict=False))
        assert len(kept) <= MAX_DURING["short"] and not {16, 20} <= set(kept)
        left = [k for k in candidates[1:] if k not in kept]
        assert len(kept) == MAX_DURING["short"] or all(any(abs(k - o) < GAP_STEPS for o in kept) for k in left)


def test_a_kind_that_asks_no_family_is_never_drawn():
    """Contract: a decision point whose kind has no during family in the job (`before_room`, `room_done`) is never
    asked, and a point asked is the kind's family. Hand-worked on `vacuum_run`: the decisions are before_room at 5,
    room_done at 18 and before_doorway at 21: only keyframe 21 is asked, with `path_ahead`. Regression caught: asking at
    a decision point the job gave no family, at any step of the task."""
    w = world(vacuum_run())
    assert [(d.k, d.kind) for d in drawn(w)] == [(21, "before_doorway")]
    assert [c.family for c in questionnaire.during(w)] == ["path_ahead"]


def test_the_draw_ignores_the_world():
    """Contract: the drawn points and their families depend on the case id and the job's decision points alone. Hand-
    worked: the same run with a thousand things added, moved, seen and unseen, different depth and another seed asks the
    same points with the same families as the bare run (keyframe 21, `path_ahead`), and a longer task draws the same
    points for the same id whatever its world; only the id changes the draw. Regression caught: which points are asked
    revealing the world (design §8.5)."""
    bare = vacuum_run()
    busy = vacuum_run(seed=9, boxes=[], frame_boxes={19: []})
    for i in range(30):
        busy.item(f"E{i:04d}", "bag", {"color": "red"}, [(0, None, (0.5 + i * 0.1, 1.0, 0.0))], size=BAG)
        busy.see(i % 28, f"E{i:04d}", 1500)
    assert [(d.k, d.kind) for d in drawn(world(bare))] == [(d.k, d.kind) for d in drawn(world(busy))] == [
        (21, "before_doorway")]
    plain, crowded = long_task([16, 20, 31, 32, 47, 63, 80]), long_task([16, 20, 31, 32, 47, 63, 80])
    for k in range(40):
        crowded.item(f"E{k:04d}", "box", {"color": "brown"}, [(0, None, (1.0, 1.0 + k * 0.01, 0.0))])
        crowded.see(k, f"E{k:04d}", 1500)
    for i in range(10):
        cid = stable_hex("ignores", i, n=16)
        assert [d.k for d in drawn(world(plain, cid))] == [d.k for d in drawn(world(crowded, cid))]


def test_events_resolve_from_the_job():
    """Contract: for every task of the window (case numbering) and each wording of the job's event table, the keyframe
    `events.resolve` gives over that task's delivered steps, converted back to script numbering, and no compiler module
    names a phase or a wording. Hand values on `vacuum_run` (k 0 docked, 1-2 undocking, 3-6 transit, 7-18 cleaning,
    19-22 transit, 23-25 docking, 26-27 docked): a job with its own table (first transit 3, last transit 22, first
    docking 23, the last step of each phase after docking 25 and 27) resolves to those in task 0, a phase the task never
    has to nothing; a window over the second task of two is task 0 of its case with the same offsets plus 28; a window
    over both holds both tasks' events, task 1's at the offsets plus 28. No `questions*`, `questionnaire`, `grid`,
    `freespace` or `cmd_compile` string constant is a phase of any robot or a wording of `vacuum/1`'s events.
    Regression caught: a special case for one robot's phases, an event on the wrong keyframe or in the wrong
    task, or a compiler that resolves what a system cannot."""
    table = {"first transit": EventRule("first", "transit"), "last transit": EventRule("last", "transit"),
             "first docking": EventRule("first", "docking"), "last docking": EventRule("last", "docking"),
             "last docked after docking": EventRule("last", "docked", after_phase="docking"),
             "never": EventRule("first", "paused")}
    mine = replace(JOB, events=table)
    first_task = {"first transit": 3, "last transit": 22, "first docking": 23, "last docking": 25,
                  "last docked after docking": 27}
    second_task = {"first transit": 31, "last transit": 50, "first docking": 51, "last docking": 53,
                   "last docked after docking": 55}
    assert world(vacuum_run(), job=mine).events == {(0, wording): k for wording, k in first_task.items()}
    assert world(vacuum_run(2), job=mine, a=1).events == {(0, wording): k for wording, k in second_task.items()}
    both = world(vacuum_run(2), job=mine, length="medium", b=1).events
    assert both == {**{(0, wording): k for wording, k in first_task.items()},
                    **{(1, wording): k for wording, k in second_task.items()}}
    assert world(vacuum_run()).events == {(0, wording): k for wording, k in EVENT_KS.items()}
    steps = [{"k": k, "status": {"phase": p}} for k, p in enumerate(
        ["docked"] + ["undocking"] * 2 + ["transit"] * 4 + ["cleaning"] * 12 + ["transit"] * 4 + ["docking"] * 3
        + ["docked"] * 2)]
    assert {w: resolve(rule, steps) for w, rule in JOB.events.items()} == EVENT_KS
    phases = {p for r in robots.ROBOTS.values() for p in r.status.phases}
    wordings = set(job_registry.get("vacuum/1").events)
    compiler = [n for n in MODULES if n.startswith("questions") or n in ("questionnaire", "grid", "freespace",
                                                                         "cmd_compile")]
    assert {"questions", "questionnaire", "questions_world", "grid", "freespace", "cmd_compile"} <= set(compiler)
    held = {n: sorted(c for c in string_constants(MODULES[n].read_text(encoding="utf-8")) if c in phases or c in
                      wordings or any(w in c for w in wordings)) for n in compiler}
    assert {n: hits for n, hits in held.items() if hits} == {}


def test_the_subject_rule(monkeypatch):
    """Contract: no two end questions of a case share an entity, a description or a named time; times conflict when
    equal or overlapping, and every question's entities are among its subjects. Hand values: `conflicts` of e:E1 and
    e:E1 is true, e:E1 and e:E2 false, w:7 and w:7 true, w:3-9 and w:9 true, w:3-8 and w:9 false; two absent
    descriptions, d:bag/{red} and d:bag/{blue}, conflict (a case asks at most one: the decoys give nothing away only
    while they are few) and an absent description and a thing do not; on a world with one red bag the case asks the bag
    once; and a fake family whose two moments name the same time is asked one question of them, whatever the case id.
    During questions are exempt: a case asks its `path_ahead` (subject w:21) beside end questions. Regression caught:
    end questions hinting at each other (VISION rule 6), the rule counting every time a gold reads, or a case of
    several absent descriptions, which an agent answers `unknown` to by their number."""
    assert conflicts(["e:E1"], ["e:E1", "w:2"]) and not conflicts(["e:E1", "w:2"], ["e:E2", "w:7"])
    assert conflicts(["w:7"], ["w:7"]) and conflicts(["w:3-9"], ["w:9"]) and not conflicts(["w:3-8"], ["w:9"])
    assert conflicts(['d:bag/{"color":"red"}', "w:2"], ['d:bag/{"color":"blue"}', "w:7"])
    assert not conflicts(['d:bag/{"color":"red"}', "w:2"], ["e:E1", "w:7"])
    s = vacuum_run()
    s.item("E0001", "bag", {"color": "red"}, [(0, None, (1.0, 1.0, 0.0))], size=BAG)
    s.see(1, "E0001", 1500)
    for cid in (stable_hex("subjects", i, n=16) for i in range(30)):
        taken = questionnaire.end(world(s, cid))
        assert sum("e:E0001" in c.subjects for c in taken) <= 1  # the one bag is named once
        subjects = [x for c in taken for x in c.subjects]
        assert len(subjects) == len(set(subjects))
        assert all(f"e:{e}" in c.subjects for c in taken for e in c.entities)
    register(monkeypatch, fake_spec("fake_t"), lambda w: [
        Moment(k, [fake_cand("fake_t", k, (f"r:{k}", "w:7"))], []) for k in ("a", "b")])
    for cid in (stable_hex("same time", i, n=16) for i in range(30)):
        assert len(questionnaire.end(world(s, cid, job=end_job("fake_t")))) == 1


def draws_unknown(cid: str, family: str, key: str) -> bool:
    """The slice draw, written out by hand (docs/questions.md): a moment of a sliced family is the unknown slice for a
    fifth of the (case id, family, moment key) triples."""
    return stable_hash(cid, family, key, "slice") % 100 < 20


BAG_COLOURS = ("red", "blue", "green", "black")


def moment_key(colour: str) -> str:
    """The key of `object_where`'s moment of the bag of this colour: its description as JSON."""
    return json.dumps(["bag", {"color": colour}], sort_keys=True, separators=(",", ":"))


def four_bags() -> SpaceScript:
    """The run with four answerable bags, each seen recognisably at keyframe 1."""
    s = vacuum_run()
    for i, color in enumerate(BAG_COLOURS):
        s.item(f"E000{i + 1}", "bag", {"color": color}, [(0, None, (1.0 + i, 1.0, 0.0))], size=BAG)
        s.see(1, f"E000{i + 1}", 1500)
    return s


def end_job(*families):
    """`JOB` asking these end families."""
    return replace(JOB, ends=tuple(families))


def moments_of(family, keys):
    """A gold function giving one moment a key, each with one answerable candidate of subjects of its own."""
    return lambda w: [Moment(k, [fake_cand(family, k, (f"r:{family}{k}",))], []) for k in keys]


def pooled(family, keys):
    """A gold function giving one moment a key, each with an answerable and an unknown candidate (slots n: the key, and
    the key with a `u`)."""
    return lambda w: [Moment(k, [fake_cand(family, k, (f"r:{family}{k}",))],
                             [fake_cand(family, f"{k}u", (f"r:{family}{k}/u",), unknown=True)]) for k in keys]


def test_end_takes_families_in_turn(monkeypatch):
    """Contract: the end questionnaire visits the families the job asks one question a turn, in a per-case order
    (`stable_hash(case id, family, "family")`), each family's moments in `stable_hash(case id, family, moment key,
    "moment")` order, and stops at the end size. Hand-worked for a case id that visits `fake_a` (three moments) before
    `fake_b` (one): the turns are A's first moment, B's, then A's second and third; at size 1, 2, 3 and 4 the asked
    questions are the first 1, 2, 3 and 4 of those turns. Regression caught: an `end()` that drains one family before
    the next (size 2 would then ask two of A's), or ignores the size."""
    keys = {"fake_a": ["m0", "m1", "m2"], "fake_b": ["m0"]}
    for family, moments in keys.items():
        register(monkeypatch, fake_spec(family), moments_of(family, moments))
    cid = stable_hex("turns", 0, n=16)
    first = sorted(keys, key=lambda f: stable_hash(cid, f, "family"))
    assert first == ["fake_a", "fake_b"]  # the family with moments to spare goes first: draining it would show
    by_hash = {f: sorted(ms, key=lambda m: stable_hash(cid, f, m, "moment")) for f, ms in keys.items()}
    turns = [(f, by_hash[f][i]) for i in range(3) for f in first if i < len(by_hash[f])]
    assert len(turns) == 4
    w = world(vacuum_run(), cid, job=end_job(*keys))
    for size in (1, 2, 3, 4, 12):
        monkeypatch.setitem(questionnaire.MAX_END, "short", size)
        assert {(c.family, c.slots["n"]) for c in questionnaire.end(w)} == set(turns[:size])


def test_a_job_asks_only_its_end_families(monkeypatch):
    """Contract: the end families asked are the job's `ends`, or M1's `object_where` alone when it names none, so a
    new family module changes no existing compile until a job names it; `ends` naming a family the registry lacks or a
    during family raises, naming it, and an empty `ends` asks nothing. Hand-worked on the run with things: a registered
    fake end family whose gold function raises is never reached by a job with `ends` None, `end` and `candidates`
    alike; named, it is asked once beside `object_where`'s questions, and alone it asks its one. Regression caught: a
    new family module changing every existing compile, or a job's typo asking nothing."""
    def reached(w):
        raise AssertionError("a family the job does not name was asked for its moments")

    register(monkeypatch, fake_spec("fake_a"), reached)
    s = things_run()
    w = world(s, "case-ends")
    assert {c.family for c in questionnaire.end(w)} == {"object_where"}
    assert {c.family for c in questionnaire.candidates(w)} == {"object_where", "path_ahead"}
    register(monkeypatch, fake_spec("fake_a"), moments_of("fake_a", ["m0"]))
    both = world(s, "case-ends", job=end_job("object_where", "fake_a"))
    assert [c.family for c in questionnaire.end(both)].count("fake_a") == 1
    assert {c.family for c in questionnaire.end(both)} == {"object_where", "fake_a"}
    assert [(c.family, c.slots["n"]) for c in questionnaire.end(world(s, job=end_job("fake_a")))] == [("fake_a", "m0")]
    assert questionnaire.end(world(s, job=end_job())) == []
    for named in ("nosuch", "path_ahead"):
        with pytest.raises(ValueError, match=named):
            questionnaire.end(world(s, job=end_job("object_where", named)))


def test_slice_is_drawn_per_moment(monkeypatch):
    """Contract: a sliced family draws each moment's slice on its own, `stable_hash(case id, family, moment key,
    "slice") % 100 < 20` (unknown when true), so the moment never predicts the slice and the questions of one case do
    not predict each other; a family that is not sliced never asks from its unknown pool. Hand-worked on a sliced fake
    family of four moments with both pools, over 200 case ids: each moment's question has the hand draw's slice, the
    unknown share lies in [0.12, 0.28] (the declared 20%), and some case's moments differ; an unsliced family with the
    same pools asks only `answerable`. On `object_where`, the four-bag run (no thing unseen, so its unknown pool is the
    absent descriptions, of which a case asks one) has a moment for each bag, keyed by its description, and asks the
    moments of the hand draw's order up to and including the first that draws the unknown slice (all four when none
    does), each in the slice the hand draw of its key gives: that moment uses the one absent description, and no later
    moment has an unknown candidate to be asked beside. Regression caught: the slice drawn per case (Q15), per
    moment's position, or an unsliced family asked as a sliced one."""
    keys = ["m0", "m1", "m2", "m3"]
    register(monkeypatch, fake_spec("fake_s"), pooled("fake_s", keys), sliced=True)
    register(monkeypatch, fake_spec("fake_p"), pooled("fake_p", keys))
    ids = [stable_hex("slices", i, n=16) for i in range(200)]
    unknown = differing = 0
    for cid in ids:
        want = {k: draws_unknown(cid, "fake_s", k) for k in keys}
        asked = questionnaire.end(world(vacuum_run(), cid, job=end_job("fake_s")))
        assert {c.slots["n"][:2]: c.slice == "unknown" for c in asked} == want and len(asked) == 4
        unknown += sum(want.values())
        differing += len(set(want.values())) > 1
        plain = questionnaire.end(world(vacuum_run(), cid, job=end_job("fake_p")))
        assert len(plain) == 4 and {c.slice for c in plain} == {"answerable"}
    assert 0.12 <= unknown / (4 * len(ids)) <= 0.28 and differing > 0
    for cid in ids[:40]:
        got = questionnaire.end(world(four_bags(), cid))
        order = sorted(BAG_COLOURS, key=lambda c: stable_hash(cid, "object_where", moment_key(c), "moment"))
        want, unknown_drawn = [], False
        for colour in order:
            if draws_unknown(cid, "object_where", moment_key(colour)):
                unknown_drawn = True
                break
            want.append(colour)
        assert sorted(c.slots["attributes"]["color"] for c in got if c.slice == "answerable") == sorted(want)
        assert sum(c.slice == "unknown" for c in got) == unknown_drawn


def test_a_sliced_moment_needs_both_pools(monkeypatch):
    """Contract: a moment of a sliced family is asked only when it has an answerable and an unknown candidate (with one
    side empty it asks nothing, whatever slice it draws, else the slice would be the form's tell), and a moment of a
    family that draws no slice when it has an answerable one. Hand-worked over 40 case ids, among which the hand draw
    gives some moment the unknown slice and some the answerable one: a sliced moment with no unknown candidate is never
    asked, one with both is asked, an unsliced moment with an answerable candidate and none unknown is asked; and the
    run with no thing at all (every `object_where` moment has absent descriptions to be unknown and no answerable
    thing) asks no end question. Regression caught: `or` for `and`, which asks a moment with one pool empty (the
    questions of such a moment all of one slice), or a slice-less family needing a pool it never draws."""
    both = Moment("both", [fake_cand("fake_s", "both", ("r:both",))],
                  [fake_cand("fake_s", "bothu", ("r:b/u",), unknown=True)])
    lone = Moment("lone", [fake_cand("fake_s", "lone", ("r:lone",))], [])
    plain = Moment("plain", [fake_cand("fake_p", "plain", ("r:plain",))], [])
    register(monkeypatch, fake_spec("fake_s"), lambda w: [both, lone], sliced=True)
    register(monkeypatch, fake_spec("fake_p"), lambda w: [plain])
    ids = [stable_hex("pools", i, n=16) for i in range(40)]
    assert {draws_unknown(cid, "fake_s", "lone") for cid in ids} == {True, False}
    for cid in ids:
        asked = {c.slots["n"][:4] for c in questionnaire.end(world(vacuum_run(), cid, job=end_job("fake_s", "fake_p")))}
        assert asked == {"both", "plai"}
        assert questionnaire.end(world(vacuum_run(), cid)) == []


def sharing(blocked: str, keys):
    """A gold function for a sliced fake family: one moment for each key, each with an answerable and an unknown
    candidate. In the `blocked` pool every moment's candidate names the same subject (the one answerable thing
    `e:shared`, or the one shared unknown subject `r:fake_s/absent`: not a `d:` subject, of which a case asks at most
    one), so the first moment asked uses it up; the other pool's candidates name subjects of their own."""
    def moment(k):
        answerable = ("e:shared",) if blocked == "answerable" else (f"r:{k}",)
        unknown = ("r:fake_s/absent",) if blocked == "unknown" else (f"r:fake_s/{k}",)
        return Moment(k, [fake_cand("fake_s", k, answerable)], [fake_cand("fake_s", f"{k}u", unknown, unknown=True)])

    return lambda w: [moment(k) for k in keys]


def hand_asked(cid: str, blocked: str, keys) -> dict[str, str]:
    """What `sharing(blocked, keys)` asks, worked by hand: the moments in `stable_hash(case id, "fake_s", key,
    "moment")` order, each asked in the slice it draws until one draws the `blocked` slice, which takes the shared
    subject; every moment after it finds that pool without an allowed candidate, so none is asked. Keyed by the
    question's slot."""
    asked = {}
    for k in sorted(keys, key=lambda k: stable_hash(cid, "fake_s", k, "moment")):
        unknown = draws_unknown(cid, "fake_s", k)
        asked[f"{k}u" if unknown else k] = "unknown" if unknown else "answerable"
        if unknown == (blocked == "unknown"):
            break
    return asked


def test_the_asked_unknown_share_is_the_declared_one(monkeypatch):
    """Contract: whether a moment of a sliced family is asked does not depend on its slice, so over cases the asked
    questions are `UNKNOWN_PERCENT` unknown (VISION rules 6 and 7: a form that tells the slice lets an all-`unknown`
    agent score on it). Hand-worked on a sliced family of five moments whose answerable candidates all name one thing
    and whose unknown candidates are each their own: the first moment asked draws its slice; one that draws answerable
    takes the thing and no later moment has a candidate in both pools, so a case asks the moments up to and including
    the first that draws answerable (all five when none does). Per case that is the hand draw's prefix, and the
    expected unknown asked is a fifth of the expected asked whatever the number of moments (each draw is a fifth, and
    the moments asked are the prefix of the draws up to a stopping point): over 400 case ids the unknown share lies in
    [0.14, 0.26], where a search of the drawn pool alone (an unknown moment always asked, an answerable one only while
    the thing is free) gives 0.5. Regression caught: a moment skipped when only its drawn pool has no allowed
    candidate, which asks the unknown slice far above its declared share and lets an all-`unknown` agent score
    about 0.45."""
    keys = [f"m{i}" for i in range(5)]
    register(monkeypatch, fake_spec("fake_s"), sharing("answerable", keys), sliced=True)
    s = vacuum_run()
    asked = unknown = 0
    for cid in (stable_hex("share", i, n=16) for i in range(400)):
        got = questionnaire.end(world(s, cid, job=end_job("fake_s")))
        assert {c.slots["n"]: c.slice for c in got} == hand_asked(cid, "answerable", keys)
        asked += len(got)
        unknown += sum(c.slice == "unknown" for c in got)
    assert 0.14 <= unknown / asked <= 0.26


@pytest.mark.parametrize("blocked", ["answerable", "unknown"])
def test_a_moment_whose_pool_the_subject_rule_emptied_is_skipped(monkeypatch, blocked):
    """Contract: a moment of a sliced family with no candidate the subject rule allows in one of its pools is skipped
    whatever slice it draws (the other pool's candidate is free), as one with a pool empty from the start is
    (`test_a_sliced_moment_needs_both_pools`). Hand-worked on two moments whose candidates in the `blocked` pool name
    one subject, over 80 case ids: the moment first in hash order is asked in its drawn slice; the other is asked only
    when the first did not draw the blocked slice, in its own drawn slice, and never otherwise, including the cases
    where it draws the other slice, whose pool is free (some of the 80 ids are such cases, else the test would not tell
    the draw-first search from this one). Regression caught: the slice drawn before the pools are looked at, which asks
    the second moment in the free slice."""
    keys = ["m0", "m1"]
    register(monkeypatch, fake_spec("fake_s"), sharing(blocked, keys), sliced=True)
    s = vacuum_run()
    skipped_free = 0
    for cid in (stable_hex("emptied", i, n=16) for i in range(80)):
        got = questionnaire.end(world(s, cid, job=end_job("fake_s")))
        want = hand_asked(cid, blocked, keys)
        assert {c.slots["n"]: c.slice for c in got} == want
        second = max(keys, key=lambda k: stable_hash(cid, "fake_s", k, "moment"))
        skipped_free += len(want) == 1 and draws_unknown(cid, "fake_s", second) != (blocked == "unknown")
    assert skipped_free > 0


def test_the_subject_rule_holds_at_20_questions(monkeypatch):
    """Contract: a long window of six tasks compiles 20 end questions (the long size) over three families, and no two
    share an entity, a description or a named time, however the families' candidates overlap. Hand-worked: six tasks
    with five things each (`family_kit`), `object_where` and two fake families whose twelve moments each name a time,
    the second's the same twelve as the first's (task t's 'set off' and 'finished sweeping', keyframes 28 t + 2 and
    + 18): 20 questions are asked, from all three families, their subjects all distinct. Regression caught: end
    questions that hint at each other (VISION rule 6), the rule dropped once a second family is in turn."""
    s = things_by_task(multi_run(6), 6)
    times = [f"w:{28 * (i % 6) + (2, 18)[i // 6]}" for i in range(12)]
    for family in ("fake_pairs", "fake_twins"):
        register(monkeypatch, fake_spec(family), lambda w, family=family: [
            Moment(f"p{i}", [fake_cand(family, i, (f"r:{family}{i}", t))], []) for i, t in enumerate(times)])
    for cid in (stable_hex("twenty", i, n=16) for i in range(6)):
        asked = questionnaire.end(fk.world(s, 0, 5, cid, "long", end_job("object_where", "fake_pairs", "fake_twins")))
        subjects = [x for c in asked for x in c.subjects]
        assert len(asked) == 20 and {c.family for c in asked} == {"object_where", "fake_pairs", "fake_twins"}
        assert len(subjects) == len(set(subjects))  # every subject, the times included, once
        assert all(f"e:{e}" in c.subjects for c in asked for e in c.entities)


def test_a_dead_band_skips_the_point(monkeypatch):
    """Contract: a during gold that returns None (a dead band at that decision point) leaves the point unasked and the
    others as drawn: the draw does not see the gold, and no other family is drawn for it. Hand-worked on a 90-step
    task with decision points at 16, 32 and 48 (all three drawn for a short case, 16 apart), a family whose
    gold is None at 32: the questions are those at 16 and 48, and the drawn points are the same three as with a family
    that has no dead band. Regression caught: a crash on None, or a substitute point drawn by what the gold found."""
    def gold(w, k):
        return None if k == 32 else fake_cand("fake_d", k, (f"w:{k}",), decision_k=k)

    register(monkeypatch, fake_spec("fake_d", "during", "now"), gold)
    job = replace(JOB, decisions={"before_doorway": ("fake_d",)})
    for cid in (stable_hex("dead", i, n=16) for i in range(10)):
        w = world(long_task([16, 32, 48]), cid, job=job)
        assert [d.k for d in drawn(w)] == [16, 32, 48]
        assert [c.decision_k for c in questionnaire.during(w)] == [16, 48]
        assert [c.decision_k for c in questionnaire.candidates(w) if c.family == "fake_d"] == [16, 48]


def test_the_end_questions_stay_under_the_cap_and_in_hash_order():
    """Contract: at most 12 end questions for a short case (18 medium, 20 long), and the end rows follow
    `stable_hash(case id, candidate key)`. Hand-worked on the run with four answerable bags: no more than the four
    moments ask, each about a different bag, in an order that is not the world's. Regression caught: an end quota that
    ignores the case's length, or a question order that tells the moment."""
    assert [questionnaire.MAX_END[k] for k in ("short", "medium", "long")] == [12, 18, 20]
    s = four_bags()
    orders = set()
    for cid in (stable_hex("order", i, n=16) for i in range(40)):
        taken = questionnaire.end(world(s, cid))
        assert len(taken) <= 4
        entities = [c.entities for c in taken if c.entities]
        assert len(set(entities)) == len(entities)
        orders.add(tuple(c.slots["attributes"]["color"] for c in taken))
    assert len(orders) > 1
