"""How early in its task a decision point may be asked, and that the draw reads kinds and family specs alone: hand-built
one-task and two-task scripts on `space_kit`, fake `during` families of each scope."""

from __future__ import annotations

from dataclasses import replace

import pytest

from family_kit import fake_cand, fake_spec, register
from space_kit import JOB, SpaceScript, box, inputs, record
from stbench.questionnaire import GAP_STEPS, NOW_MIN_STEPS, drawn
from stbench.questions_world import World
from stbench.seeding import stable_hex

# Fake during families, one of each scope, and the decision kinds a job could ask them at.
KINDS = {"now_only": ("fake_now",), "now_pair": ("fake_now", "fake_now2"), "memory_only": ("fake_mem",),
         "evidence_only": ("fake_ev",), "mixed": ("fake_now", "fake_mem"), "silent": ()}
JOB_OF_KINDS = replace(JOB, decisions=KINDS)


@pytest.fixture
def families(monkeypatch):
    """`fake_now`, `fake_now2` (scope now), `fake_mem` (memory) and `fake_ev` (evidence) in the registry, each giving a
    candidate at every decision point."""
    for name, scope in (("fake_now", "now"), ("fake_now2", "now"), ("fake_mem", "memory"), ("fake_ev", "evidence")):
        register(monkeypatch, fake_spec(name, "during", scope),
                 lambda w, k, name=name: fake_cand(name, k, (f"w:{k}",), decision_k=k))


def world(script, cid="c0", job=JOB_OF_KINDS, a=0, b=None, length="short"):
    return World(inputs(script, job), record(script, a=a, b=b, cid=cid, length=length))


def tasks_of(frames, n=1):
    return [[("kitchen", frames, "cleaning")] for _ in range(n)]


@pytest.mark.parametrize("kind, offset, asked", [
    ("now_only", NOW_MIN_STEPS - 1, False), ("now_only", NOW_MIN_STEPS, True),
    ("now_pair", NOW_MIN_STEPS - 1, False), ("now_pair", NOW_MIN_STEPS, True),
    ("memory_only", NOW_MIN_STEPS, False), ("memory_only", GAP_STEPS - 1, False), ("memory_only", GAP_STEPS, True),
    ("evidence_only", GAP_STEPS - 1, False), ("evidence_only", GAP_STEPS, True),
    ("mixed", NOW_MIN_STEPS, False), ("mixed", GAP_STEPS - 1, False), ("mixed", GAP_STEPS, True),
    ("silent", NOW_MIN_STEPS, False), ("silent", 40, False)])
def test_a_point_is_asked_from_the_minimum_its_kinds_scopes_set(families, kind, offset, asked):
    """Contract: a decision point is asked from `NOW_MIN_STEPS` (4) steps into its task when every family its kind asks
    has scope `now`, and from `GAP_STEPS` (16) when any is `memory` or `evidence` (a mixed kind takes the later), and a
    kind that asks nothing never; the steps are counted from the task's own start, not the script's.
    Hand-worked on a second task of 30 keyframes (it starts at keyframe 30, so a minimum read from the script's start
    would ask a point at offset 3 as keyframe 33) with one decision point of the kind at the offset, the window being
    that task: `now_only` and `now_pair` at 3 are not asked and at 4 are; `memory_only` at 4 and 15 are not and at 16
    is; `evidence_only` and `mixed` the same; `silent` is not at 4 or 40. Regression caught: `now` questions starved at
    a task's start (every point held to 16 steps), memory questions asked without history (every point let in at 4), a
    mixed kind treated as `now` (`any` for `all`), or the offset read from the wrong start."""
    script = SpaceScript(tasks=tasks_of(30, 2), decisions=[(30 + offset, kind)], depth_ks=set())
    assert [d.k - 30 for d in drawn(world(script, a=1))] == ([offset] if asked else [])


def test_the_early_minimum_keeps_the_spacing():
    """Contract: points a `now` kind is asked from `NOW_MIN_STEPS` still lie `GAP_STEPS` apart, 16 allowed and 15 not.
    Hand-worked on a 40-keyframe task of `JOB`'s `now` family with decision points at 4, 5, 19 and 20: the only sets the
    draw can keep are {4, 20} (16 apart), {5} (15 from 20, 14 from 19) and {19} (15 from 4, 14 from 5), whichever point
    the case id hashes first; over 40 case ids it draws each of them. Regression caught: the lower entry point read as
    the spacing too, which asks 4 and 5, or 19 and 20, together."""
    script = SpaceScript(tasks=tasks_of(40), decisions=[(k, "before_doorway") for k in (4, 5, 19, 20)], depth_ks=set())
    seen = set()
    for i in range(40):
        seen.add(tuple(d.k for d in drawn(world(script, stable_hex("apart", i, n=16), JOB))))
    assert seen == {(4, 20), (5,), (19,)}


def test_the_draw_reads_kinds_and_family_specs_never_the_world():
    """Contract: the points drawn depend on the case id, the decision points' kinds and steps and the families' specs,
    not on the world, so the lower entry point for `now` kinds leaks nothing (design §8.5). Hand-worked: a long task
    with `JOB`'s `before_doorway` (`path_ahead`, a `now` family) at 4, 8, 12, 20, 24, 40, 60 and 80, over a bare world
    (no depth image at all, so every gold is `unknown`) and a busy one (a wall, a box, sixty things seen and a depth
    image at every keyframe): the same points for every one of 40 case ids, and some id draws a point inside the first
    16 steps (else the test would not reach the range the early minimum opened). Regression caught: an entry point or a
    spacing that reads the gold, the evidence or the things around the point."""
    ks = (4, 8, 12, 20, 24, 40, 60, 80)
    bare = SpaceScript(tasks=tasks_of(90), decisions=[(k, "before_doorway") for k in ks], depth_ks=set())
    busy = SpaceScript(tasks=tasks_of(90), decisions=[(k, "before_doorway") for k in ks], depth_ks=None,
                       boxes=[box(2.21, 2.43, -3.0, 3.0, 0.0, 2.0), box(1.41, 1.73, -0.15, 0.15, 0.0, 0.3)])
    for i in range(60):
        busy.item(f"E{i:04d}", "box", {"color": "brown"}, [(0, None, (1.0, 1.0 + i * 0.001, 0.0))])
        busy.see(i % 90, f"E{i:04d}", 1500)
    early = 0
    for i in range(40):
        cid = stable_hex("reads", i, n=16)
        quiet, crowded = (drawn(world(s, cid, JOB)) for s in (bare, busy))
        assert [(d.k, d.kind) for d in quiet] == [(d.k, d.kind) for d in crowded]
        early += any(d.k < GAP_STEPS for d in quiet)
    assert early > 0
