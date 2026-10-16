"""`helper/1`'s subgoal (`job_helper_subgoal`): what the helper is doing at each keyframe, from the timeline alone and
causal. The scripts are seeds 101 and 102 over `apartment_a` (`tests/helper_kit.py`, one xdist group); the texts are
the plan's (FD-0.5.0-control-impl §2) and the hand values the map's: the hall console F20, the shoe rack F21, the sofa
F10, the coffee table F11, the TV console F12."""

from __future__ import annotations

from dataclasses import replace
from unittest import mock

import pytest

import helper_kit
from stbench import build, descriptions, job_helper_carry, leak
from stbench.generate import generate_script
from stbench.job_helper_subgoal import subgoals
from stbench.job_registry import get
from stbench.mapspec import load_map
from stbench.timeline import Timeline

pytestmark = pytest.mark.xdist_group(name="helper")  # one worker builds the scripts once; every module reads them

SEEDS = (101, 102)
M = load_map("apartment_a")
JOB = get("helper/1")
PIECES = {"F10": "sofa", "F11": "coffee table", "F12": "tv console", "F20": "hall console", "F21": "shoe rack"}


@pytest.fixture(scope="module")
def scripts():
    return {seed: helper_kit.generated(seed, build.tasks_for((4, 6), "apartment_a", seed, "helper", "helper/1"))
            for seed in SEEDS}


def cut(tl: Timeline, k: int) -> Timeline:
    """The timeline as it stands after keyframe `k`: the keyframes, the decisions and the tasks up to it, the world
    whole (the future of the ego is what a causal text must not read)."""
    kfs = tl.ego.keyframes[:k + 1]
    tasks = [replace(t, k1=min(t.k1, k + 1)) for t in tl.ego.tasks if t.k0 <= k]
    decisions = [d for d in tl.ego.decisions if d.k <= k]
    return replace(tl, ego=replace(tl.ego, keyframes=kfs, tasks=tasks, decisions=decisions))


def test_the_subgoal_is_declared_and_one_text_a_keyframe(scripts):
    """Contract: `helper/1` declares its subgoal function, which gives one non-empty text for every keyframe of a
    script. Regression caught: a job whose pose rows miss a subgoal (the header and `path_log` would read nothing), or a
    function that returns fewer texts than keyframes."""
    assert JOB.subgoals is subgoals
    for _, (_, tl) in scripts.items():
        texts = JOB.subgoals(tl, M)
        assert len(texts) == tl.n_keyframes and all(texts)


def test_the_texts_follow_the_phases_and_name_the_thing_and_its_pieces(scripts):
    """Contract: within a task the text is by the phase and by what has happened up to the keyframe: waiting before it
    leaves ("waiting at its spot to move/bring the {thing} from the {source} to the {target}"), navigating to the source
    ("going to the {source} to pick up the {thing}"), grasping ("picking up the {thing} from the {source}"), carrying
    ("carrying the {thing} from the {source} to the {target}"), placing ("putting the {thing} down on the {target}", and
    after an `elsewhere` "... on a free spot, since the spot on the {target} is occupied"), navigating back ("going back
    to its spot") and waiting at the end ("waiting at its spot"); the thing is named by its description and the pieces
    by the map's names (by hand: the sofa, the coffee table, the TV console, the hall console, the shoe rack).
    Regression caught: a text that names the wrong piece (the planned one when the world moved the helper to another), a
    verb that does not follow the task kind, or a phase with no text."""
    for _, (_, tl) in scripts.items():
        texts = subgoals(tl, M)
        for task in tl.ego.tasks:
            eid, source, target = task.targets[2:5]
            thing = descriptions.phrase(tl.entity(eid).kind, tl.entity(eid).attributes)
            src, dst = (PIECES[next(f.id for f in M.furniture if f.center == tuple(tl.placement_of(x, 0.0).pos[:2]))]
                        for x in (source, target))
            verb = "move" if task.plan.kind == "tidy" else "bring"
            by_phase: dict[str, list[str]] = {}
            for k in range(task.k0, task.k1):
                by_phase.setdefault(tl.keyframe(k).phase, []).append(texts[k])
            assert by_phase["waiting"][0] == f"waiting at its spot to {verb} the {thing} from the {src} to the {dst}"
            assert by_phase["waiting"][-1] == "waiting at its spot"
            assert set(by_phase["grasping"]) == {f"picking up the {thing} from the {src}"}
            assert set(by_phase["carrying"]) == {f"carrying the {thing} from the {src} to the {dst}"}
            assert by_phase["navigating"][0] == f"going to the {src} to pick up the {thing}"
            assert by_phase["navigating"][-1] == "going back to its spot"
            answer = next(d.response for d in tl.decisions_in(task.k0, task.k1) if d.kind == "before_place")
            placing = f"putting the {thing} down on the {dst}" if answer == "go" else (
                f"putting the {thing} down on a free spot, since the spot on the {dst} is occupied")
            assert set(by_phase["placing"]) == {placing}


def test_a_text_reads_nothing_after_its_keyframe(scripts):
    """Contract (the causality test of FD-0.5.0-control-impl §2): `subgoals(tl)[k]` does not change when the timeline's
    ego is cut after keyframe k. Checked at every keyframe of a tidy and a fetch (the keyframes before, at and after the
    `before_place` decision, where the response enters). Regression caught: a text that reads the coin or the response
    of the decision it is about to be asked at, which a model could read as the answer."""
    tl = scripts[101][1]
    whole = subgoals(tl, M)
    for kind in ("tidy", "fetch"):
        task = next(t for t in tl.ego.tasks if t.plan.kind == kind)
        place = next(d.k for d in tl.decisions_in(task.k0, task.k1) if d.kind == "before_place")
        for k in range(task.k0, task.k1):
            if abs(k - place) <= 2 or k % 5 == 0:
                assert subgoals(cut(tl, k), M)[k] == whole[k], (kind, k)


def test_a_coin_decided_at_or_after_a_keyframe_does_not_change_its_text():
    """Contract: with the place spot's coin forced blocked or clear, the first task's texts are the same up to and
    including the `before_place` keyframe, and differ after it (the response is known from the next keyframe on): by
    hand, the placing text of a blocked window says the spot is occupied and a clear one does not. Regression caught:
    a text before the question that tells the coin, so a model could answer it from the subgoal and the `path_log`
    gate would be beaten."""
    runs = {}
    for blocked in (True, False):
        with mock.patch.object(job_helper_carry, "spot_blocked", lambda *_, blocked=blocked: blocked):
            runs[blocked] = generate_script("apartment_a", 103, "helper", "helper/1", 2)
    task = runs[True].ego.tasks[0]
    place = next(d.k for d in runs[True].decisions_in(task.k0, task.k1) if d.kind == "before_place")
    texts = {blocked: subgoals(tl, M)[:task.k1] for blocked, tl in runs.items()}
    assert texts[True][:place + 1] == texts[False][:place + 1]
    after = [(x, y) for x, y in zip(texts[True][place + 1:], texts[False][place + 1:], strict=True) if "free spot" in x]
    assert after and all("free spot" not in y for _, y in after)


def test_no_text_names_an_id_a_region_or_the_map(scripts):
    """Contract (the leak scan): no text of a script holds an entity id, a region id, a piece id or the map's name; the
    pieces are named by what they are. Regression caught: a text built from the plan's ids ("F20", "R2"), which a
    system could use to look a place up on the public map."""
    names = [M.name]
    for _, (_, tl) in scripts.items():
        for text in set(subgoals(tl, M)):
            assert [rule for rule, rx in leak.RULES.items() if rx.search(text)] == [], text
            assert not any(n.lower() in text.lower() for n in names), text
            assert not any(f.id in text.split() for f in M.furniture), text
