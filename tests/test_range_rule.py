"""The `range_m` of a ladder gold follows the rule the prompt states for its family (`llm_render`'s credit paragraph,
docs/format.md's `range_m` row, `questions_select.Cand`). One thing seen from two keyframes shows the families apart:
the rule is not one number for the question but one per family, and a relative rung is built from it."""

from __future__ import annotations

import pytest

import change_kit as ck
from change_kit import thing
from ego_kit import K, named, put, scene, world
from space_kit import vacuum_run
from stbench.questions_change import object_was
from stbench.questions_ego import ego_mem, ego_now
from stbench.questions_where import pools


def test_each_family_states_its_range_and_takes_its_rungs_from_the_job():
    """Contract: `ego_now`'s and `ego_mem`'s `range_m` is the slant from the camera at the look point to the nearest
    part they ask for (at the height of the visible point nearest it), and their need is the job's clearance C,
    whatever the range; `object_where`'s `range_m` is the nearest recognisable slant at the asked place, and its need
    is the job's re-find radius R (0.7.0), whatever the range. Hand-worked from the cameras of `ego_kit` (look point
    keyframe 13 at (3.0, 1.13, 0.08); keyframe 2 at (2.0, 1.63, 0.08)) and 0.1 x 0.1 x 0.4 m things whose pixels are
    the corners of their boxes (base corners at z = 0, so 0.08 m below the camera):
    - R, a red bag at (1.0, 2.63) seen at keyframes 2 and 13: `ego_now` 2.4313 (the corner (1.05, 2.58): 1.95 along x,
      1.45 along y, 0.08 up), `object_where` 1.4193 (the nearer sighting, keyframe 2: 1.0, 1.0, 0.12 to the middle), not
      the ask's slant;
    - S, black shoes at (2.0, 2.63) seen at keyframe 2 only: `ego_mem` 1.7354 (the corner (2.05, 2.58), seen from
      keyframe 2 and measured from the camera now), `object_where` 1.0072;
    - T, a pink toy at (3.0, 2.0): `ego_mem` 0.8239 (the point (3.0, 1.95) of its south edge: 0.82 along y, 0.08 up),
      `object_where` 1.0730.
    Regression caught: one range for every family (`object_where` taking the ask's slant, `ego_now` the nearest
    sighting), an ego range to the thing's middle (the old answer, 2.5029), or a need that follows the range."""
    s = scene()
    put(s, "R", "bag", "red", (1.0, 2.63), [2, K])
    put(s, "S", "shoes", "black", (2.0, 2.63), [2])
    put(s, "T", "toy", "pink", (3.0, 2.0), [2])
    w = world(s)
    now, mem, where = named(ego_now(w, K)), named(ego_mem(w, K)), {c.entities[0]: c for c in pools(w)[0]}
    assert set(now) == {"R"} and set(mem) == {"S", "T"} and set(where) == {"R", "S", "T"}
    assert now["R"].range_m == pytest.approx(2.4313, abs=1e-3)
    assert where["R"].range_m == pytest.approx(1.4193, abs=1e-3)
    assert mem["S"].range_m == pytest.approx(1.7354, abs=1e-3) and where["S"].range_m == pytest.approx(1.0072, abs=1e-3)
    assert mem["T"].range_m == pytest.approx(0.8239, abs=1e-3) and where["T"].range_m == pytest.approx(1.0730, abs=1e-3)
    for c in (now["R"], mem["S"], mem["T"]):
        assert c.tolerances["need_m"] == 2.0   # C = 2.0, not a share of the range
    assert {c.tolerances["need_m"] for c in where.values()} == {0.5}   # R = 0.5 m, whatever the range


def test_object_was_scales_from_the_nearest_slant_at_the_asked_place():
    """Contract: `object_was`'s `range_m` is the nearest recognisable slant at which the asked place itself was seen,
    not the last sighting's and not the other state's. Hand-worked on a red bag (0.314 m tall, middle 0.157 m up, so
    0.077 m above the camera) that stands at (1.0, 1.0) in task 0, seen from keyframes 2 and 3, and at (2.5, 1.0) in
    task 1, seen from keyframe 29: keyframe 2's camera (2.0, 1.63) is sqrt(1 + 0.63^2 + 0.077^2) = 1.1844 away and
    keyframe 3's (2.13, 1.5) is 1.2381 away, so the earlier state's range is 1.1844 (its evidence is the last sighting,
    keyframe 3); keyframe 29's camera (2.0919, 1.5919) is 0.7231 from (2.5, 1.0), the later state's. Regression caught:
    the last sighting's slant (1.2381) or the other state's range on the row."""
    s = vacuum_run(2)
    thing(s, "E0001", "bag", "red", [(1.0, 1.0), (2.5, 1.0)], [2, 3, 29])
    w = ck.window(s, 0, 1)
    early, late = object_was(w)[0].answerable, object_was(w)[0].unknown
    assert [(c.answer_class, c.evidence_k) for c in early] == [("earlier", (3,))]
    assert [(c.answer_class, c.evidence_k) for c in late] == [("later", (29,))]
    assert early[0].range_m == pytest.approx(1.1844, abs=1e-3) and late[0].range_m == pytest.approx(0.7231, abs=1e-3)
