"""`object_where` (questions_where): "Where did you last see the {thing}?", who is answerable, unknown or skipped, by
hand on `space_kit`'s run (one task of 28 keyframes; the end look window is keyframes 22 to 27) and on three-task
windows of the moved bag."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

import family_kit as fk
from change_kit import MANY_COLOURS
from space_kit import JOB, RELEVANT_KINDS, things_run, vacuum_run
from stbench import questionnaire
from stbench.job_registry import Relevance
from stbench.questions_text import FAMILIES
from stbench.questions_where import moments, never_seen, pools
from stbench.seeding import stable_hex

BAG = (0.374, 0.304, 0.314)  # the catalog's bag: a large thing


def thing_world():
    """`things_run`, the case of its one task, over the hand map (kitchen x 0-4, living room x 4-8)."""
    return fk.world(things_run())


def summary(cands):
    return sorted((c.entities, tuple(c.gold.get("xy", ())), c.evidence_k) for c in cands if c.entities)


def test_object_where_unknown_needs_no_pixel():
    """Contract: the answerable things are those with a recognisable sighting (1,024 px and a mean luminance of
    `recognise.RECOG_LUM`, 0.01) in the window and no visibility row in its last look window, the gold their placement
    at the last such sighting; things seen only at 500 px or only dark (lum 0.005, under the floor) are skipped, not
    unknown; unknown is a thing with no pixel in the window that exists at its last keyframe, or an absent description.
    Hand values on `things_run`:
    - answerable E0001 at (2, 1) from keyframe 10 (it moved 0.9 m at keyframe 5, under `STALE_M`, and its last
      recognisable sighting is the later one) and E0004 at (3.5, 1) from 12;
    - unknown E0005 (a blue bag no frame shows) and E0006, a white box that exists from keyframe 20 and so at the
      window's last keyframe, in that order, then the absent descriptions.
    E0002 (500 px), E0003 (dark), E0007 (a row in the end look window), the mug, the chair and the two identical toys
    are never asked. Regression caught: a seen thing scored as `unknown` (truth rule 2), a thing that only appears later
    left out of the unknown pool, or a thing in view asked."""
    answerable, unknown = pools(thing_world())
    assert summary(answerable) == [(("E0001",), (2.0, 1.0), (10,)), (("E0004",), (3.5, 1.0), (12,))]
    real = [c.subjects[0] for c in unknown if c.subjects[0].startswith("e:")]
    assert real == ["e:E0005", "e:E0006"]
    assert [c.subjects[0].startswith("d:") for c in unknown] == [False, False] + [True] * (len(unknown) - 2)


def test_the_text_and_the_slots_name_the_thing_and_nothing_else():
    """Contract: `object_where` is rendered from the thing's kind and attributes alone, 'Where did you last see the
    {thing}?': the spec's slots are `kind` and `attributes`, every candidate carries exactly those two (answerable,
    unknown and absent descriptions alike), no moment, event or task is a slot, and the text is the render of the slots
    in the one wording, which asks for map coordinates. Hand-worked on `things_run`: 'Where did you last see the red
    bag? Answer in map coordinates.' for E0001, 'the brown box' for E0004, 'the blue bag' for the unseen E0005.
    Regression caught: the render purity broken (a moment or a task a slot or in the text), which would tell the form
    from the world (plan §2.5)."""
    spec = FAMILIES["object_where"]
    assert spec.slots == ("kind", "attributes") and spec.needs == ("see", "pose")
    answerable, unknown = pools(thing_world())
    assert {tuple(c.slots) for c in answerable + unknown} == {("kind", "attributes")}
    texts = {c.entities[0]: spec.render(c.slots) for c in answerable}
    tail = " Answer in map coordinates."
    assert texts == {"E0001": "Where did you last see the red bag?" + tail,
                     "E0004": "Where did you last see the brown box?" + tail}
    assert spec.render(unknown[0].slots) == "Where did you last see the blue bag?" + tail
    assert spec.render({"kind": "shoes", "attributes": {"color": "black"}}) == (
        "Where did you last see the pair of black shoes?" + tail)


def test_there_is_one_moment_for_each_answerable_thing_keyed_by_its_description():
    """Contract: `moments` gives one moment for each answerable thing, keyed by the thing's description (kind and
    attributes as JSON), holding that thing alone in `answerable` and the window's whole unknown pool in `unknown`; no
    moment is keyed by an event or a task. Hand-worked on `things_run`: two moments, the red bag's and the brown box's.
    Regression caught: moments by event wording (a thing asked once for each moment of the job, which was most of the
    questions about the same unmoved bag), or a moment holding two things, which the questionnaire draws one from."""
    w = thing_world()
    answerable, unknown = pools(w)
    found = moments(w)
    red = json.dumps(["bag", {"color": "red"}], sort_keys=True, separators=(",", ":"))
    brown = json.dumps(["box", {"color": "brown"}], sort_keys=True, separators=(",", ":"))
    assert [m.key for m in found] == [red, brown]
    assert [m.answerable for m in found] == [[answerable[0]], [answerable[1]]]
    assert all(m.unknown == unknown for m in found)


def test_unknown_pools_match_the_asked_kinds():
    """Contract: the answerable things, the unknown things and the absent descriptions are all carried things of the
    relevance kinds (here bag, shoes, toy, box), so no kind is answerable-only and none predicts the slice; an absent
    description matches nothing in the world; and over 200 case ids the share of each moment's questions that is the
    unknown slice lies in [0.12, 0.28] (the declared 20%). Regression caught: a kind or a moment that predicts the
    slice, or a never-seen description of a kind the job never asks about."""
    w = thing_world()
    answerable, unknown = pools(w)
    assert {c.slots["kind"] for c in answerable + unknown} <= RELEVANT_KINDS
    assert {"bag", "box"} <= {c.slots["kind"] for c in answerable} & {c.slots["kind"] for c in unknown}
    assert all(c.critical for c in answerable + unknown)
    assert {kind for kind, _ in never_seen(w)} <= RELEVANT_KINDS and len(never_seen(w)) >= 10
    assert all(key not in w.groups for key in never_seen(w))
    ids = [stable_hex("slice share", i, n=16) for i in range(200)]
    for m in moments(w):
        share = sum(questionnaire.unknown_slice(cid, "object_where", m.key) for cid in ids) / len(ids)
        assert 0.12 <= share <= 0.28, (m.key, share)


def test_the_asked_unknown_share_is_the_declared_one():
    """Contract: over many cases of one world the questions `object_where` asks are the unknown slice for the declared
    20% of the moments (`questionnaire.UNKNOWN_PERCENT`), whichever things the case draws: neither a thing nor the
    world's unseen things nor the absent descriptions move the share (a moment is asked only when both its pools allow
    a candidate, so whether it is asked never depends on its slice). Hand-worked on a window of eight bags that stay
    (recognisable at keyframe 1) beside two unseen toys: over 300 case ids the unknown share of the end questions lies
    in [0.12, 0.28]. Regression caught: a moment skipped when its drawn pool is empty, which shifts the share (a skew
    of 0.3 to 0.5 for a declared 0.2)."""
    s = vacuum_run()
    for i, colour in enumerate(MANY_COLOURS[:8]):
        s.item(f"E{i + 1:04d}", "bag", {"color": colour}, [(0, None, (0.4 + 0.4 * i, 1.0, 0.0))], size=BAG)
        s.see(1, f"E{i + 1:04d}", 1500)
    s.item("E0101", "toy", {"color": "purple"}, [(0, None, (3.0, 2.0, 0.0))])
    s.item("E0102", "toy", {"color": "pink"}, [(0, None, (3.5, 2.0, 0.0))])
    asked = unknown = 0
    for cid in (stable_hex("asked share", i, n=16) for i in range(300)):
        got = questionnaire.end(fk.world(s, cid=cid))
        asked += len(got)
        unknown += sum(c.slice == "unknown" for c in got)
    assert asked > 1000 and 0.12 <= unknown / asked <= 0.28, (unknown, asked)


def test_end_questions_skip_things_in_view():
    """Contract: a thing with any visibility row in the window's last look window (keyframes 22 to 27) is never asked at
    the end, answerable or unknown, at any pixel count; the same thing with no pixel there is. Hand-worked: three red
    bags, each recognisable at keyframe 1, are seen again in the end look window at 1,500, 400 (a detection, under the
    floor) and 100 px (a glimpse); a fourth is not: only the fourth is answerable. Regression caught: a `memory`
    question the current view answers (VISION rule 4)."""
    s = vacuum_run()
    for i, (color, px) in enumerate((("red", 1500), ("blue", 400), ("green", 100), ("black", None))):
        s.item(f"E00{i + 1:02d}", "bag", {"color": color}, [(0, None, (1.0 + i, 1.0, 0.0))], size=BAG)
        s.see(1, f"E00{i + 1:02d}", 1500)
        if px:
            s.see(25, f"E00{i + 1:02d}", px)
    answerable, unknown = pools(fk.world(s))
    assert [c.entities for c in answerable] == [("E0004",)]
    assert not any(c.subjects[0] in {f"e:E000{i}" for i in (1, 2, 3)} for c in answerable + unknown)


def test_evidence_never_precedes_the_window():
    """Contract: the sightings a gold may read are those of the case's own window. Hand-worked on two tasks (28
    keyframes each; the case is the second): a red bag recognisably seen only at keyframe 3, in the first task, with no
    pixel in the window, is an unknown thing, not an answerable one; seen at keyframe 31 as well, inside the window, it
    is answerable with evidence 31. Regression caught: evidence from before the case (design §5), which a system that
    is sent only the window could not have."""
    def run(extra):
        s = vacuum_run(2)
        s.item("E0001", "bag", {"color": "red"}, [(0, None, (1.0, 1.0, 0.0))], size=BAG)
        s.see([3, *extra], "E0001", 1500)
        return fk.world(s, 1)

    answerable, unknown = pools(run([]))
    assert answerable == []
    assert "e:E0001" in {c.subjects[0] for c in unknown}
    answerable, unknown = pools(run([31]))
    assert [(c.entities, c.evidence_k) for c in answerable] == [(("E0001",), (31,))]
    assert "e:E0001" not in {c.subjects[0] for c in unknown}


def test_the_need_is_the_re_find_radius_and_the_row_records_the_nearest_range():
    """Contract (0.7.0, 0.9.0): an answerable `object_where` row's need is the home job's re-find radius R (0.5 m, its
    `match_m`), whatever the range, and an unknown row carries the same need with the floor resolution; the row records
    the nearest recognisable slant of the thing in the window as `range_m` (reports only; an unknown row has none),
    which sets no tolerance. Hand values on `things_run` (the camera at keyframe 1 is (2.0919, 1.5919, 0.08), at 10
    (5.9081, 1.5919, 0.08), at 12 (5.9081, 1.4081, 0.08)): E0001 (a bag whose middle is 0.157 m up) is seen at keyframes
    1 and 10, nearest 3.9534 m from the second (the nearest slant at the place it is asked about, (2, 1): at keyframe 1
    it stood 0.9 m away at (1.1, 1), 1.1577 m from the camera, which does not count); E0004 (a box, middle 0.05 m up) is
    seen at 12 only, 2.4426 m. Both rows have need 0.5 (the resolution tests work their resolutions out), and the gold
    rows the scorer reads carry them. Regression caught: a need that scales with the range, the job's four-rung distance
    ladder on a place, a range on an unknown row."""
    answerable, unknown = pools(thing_world())
    got = {c.entities[0]: (c.range_m, c.tolerances["need_m"], c.answer_class) for c in answerable}
    assert got["E0001"][0] == pytest.approx(3.9534, abs=1e-4) and got["E0004"][0] == pytest.approx(2.4426, abs=1e-4)
    assert got["E0001"][1] == 0.5 and got["E0004"][1] == 0.5
    assert {v[2] for v in got.values()} == {"answerable"}
    assert {(c.tolerances["need_m"], c.tolerances["resolution_m"]) for c in unknown} == {(0.5, 0.01)}
    assert {(c.range_m, c.answer_class) for c in unknown} == {(None, "unknown")}
    _, golds = fk.compiled(things_run())
    rows = {g["slice"]: g for g in golds if g["family"] == "object_where"}
    assert rows["unknown"]["tolerances"]["need_m"] == 0.5 and rows["unknown"]["range_m"] is None
    assert rows["answerable"]["tolerances"]["need_m"] == 0.5 and rows["answerable"]["credit"] == "graded"
    assert rows["answerable"]["range_m"] > 0 and rows["answerable"]["answer_class"] == "answerable"


def moved_bag(seen, to=2.0):
    """Three tasks and a red bag at (1, 1) through the first task and at (`to`, 1) from the second (a change in the gap;
    1 m by default, `STALE_M`), recognisable at the keyframes `seen`."""
    s = vacuum_run(3)
    s.item("E0001", "bag", {"color": "red"}, [(0, 7000.0, (1.0, 1.0, 0.0)), (7000.0, None, (to, 1.0, 0.0))], size=BAG)
    s.see(seen, "E0001", 1500)
    return s


def test_the_last_sighting_may_lie_in_an_earlier_task():
    """Contract: a thing is answerable at the place of its last recognisable sighting in the window, whichever task
    that was in: "where did you last see" is a question about the last sighting, and nothing unobserved changes a gold
    (truth rule 2). Hand-worked on three tasks (28 keyframes each) with a red bag at (1, 1) in task 0 and 0.8 m on at
    (1.8, 1) after the first gap (under `STALE_M`), recognisable at keyframes 1 and 29: the gold is (1.8, 1) from
    29; recognisable at keyframe 1 only it is (1, 1) from 1, though the bag has moved by the end; and the moment is the
    bag's alone, keyed by its description, in a window of three tasks as in one. Regression caught: the same-task rule
    of the anchored question kept, which would drop every thing last seen before the final task."""
    for seen, xy, evidence in (([1, 29], [1.8, 1.0], (29,)), ([1], [1.0, 1.0], (1,))):
        w = fk.world(moved_bag(seen, to=1.8), 0, 2, "c1", "medium")
        (c,), _ = pools(w)
        assert (c.slots, c.gold, c.evidence_k, c.entities) == (
            {"kind": "bag", "attributes": {"color": "red"}}, {"type": "position", "xy": xy}, evidence, ("E0001",))
        assert len(moments(w)) == 1
    (c,), _ = pools(fk.world(moved_bag([1, 29], to=1.8), 1))
    assert (c.gold, c.evidence_k) == ({"type": "position", "xy": [1.8, 1.0]}, (29,))


def test_a_non_critical_thing_is_never_answerable():
    """Contract: only critical things are answerable (`questions_where.critical` under any task of the window): a thing
    of a relevance kind whose last place lies in no room of a kind the tasks work in is skipped, not asked, while it
    stays in the unknown pool when no frame shows it. Hand-worked on the hand map (kitchen x 0-4, living room x 4-8)
    with a job that works the living room only: a red bag last seen at (5, 1) is answerable, a blue bag last seen at
    (2, 1) in the kitchen and a green bag at (9, 1) in no room are not, and an unseen brown box is unknown. Regression
    caught: the questions about places the job never works in (the 0.5.0 log review's non-critical rows), or the
    relevance read from the kind alone."""
    s = vacuum_run()
    for eid, colour, x in (("E0001", "red", 5.0), ("E0002", "blue", 2.0), ("E0003", "green", 9.0)):
        s.item(eid, "bag", {"color": colour}, [(0, None, (x, 1.0, 0.0))], size=BAG)
        s.see(1, eid, 1500)
    s.item("E0004", "box", {"color": "brown"}, [(0, None, (6.0, 2.0, 0.0))])
    living = replace(JOB, relevance=lambda plan: Relevance(RELEVANT_KINDS, frozenset({"living room"}), frozenset()))
    answerable, unknown = pools(fk.world(s, job=living))
    assert [c.entities for c in answerable] == [("E0001",)] and all(c.critical for c in answerable)
    assert "e:E0004" in {c.subjects[0] for c in unknown}
    assert not any(c.subjects[0] in {"e:E0002", "e:E0003"} for c in answerable + unknown)
    both = replace(JOB, relevance=lambda plan: Relevance(RELEVANT_KINDS, frozenset({"kitchen", "living room"}),
                                                         frozenset()))
    assert [c.entities for c in pools(fk.world(s, job=both))[0]] == [("E0001",), ("E0002",)]
