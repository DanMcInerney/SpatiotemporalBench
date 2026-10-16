"""The resolution and the need of `closest_person` and `person_mem` golds (0.9.0, `questions_people`, `resolution_at`).
Hand-worked on `person_kit`'s scene: the camera is 0.08 m up, so a pixel at d metres on the floor (h = 0.08) is
hypot((d^2 + 0.0064) / (0.08 * 381.36115), sqrt(d^2 + 0.0064) / 381.36115), at least 0.01. The job's clearance C is 2.0
m, the vacuum's body 0.34 m long."""

from __future__ import annotations

import pytest

import person_kit as pk
from ego_kit import K, put
from stbench.questions_people import person_mem
from test_closest_person import case_with, only, scene


def test_closest_person_is_resolved_at_the_nearest_sighting_and_is_harmful_over():
    """Contract (brief 3.7, a closest approach: one pixel at the nearest sighting; `questions_people._task_cand`): the
    row carries C as the need, the harmful side `over` (a distance answered wider than the person came) and the
    threshold body length + C, with the resolution at the sighting that gives the gold. By hand (the module's scene):
    the person at (2.0, 3.63) seen at keyframe 2 (camera (2.0, 1.63)) has its near edge 1.72 m away, at (2.0, 2.63) seen
    at keyframe 13 (camera (2.0, 1.13)) 1.22 m: the gold is 1.22, the sighting keyframe 13. There the point (2.0, 2.35)
    is d = 1.22 m, h = 0.08: (1.4884 + 0.0064) / 30.5089 = 0.04900 radially, 0.00321 sideways, 0.04910 together
    (keyframe 2's would be 0.0973). The threshold is 0.34 + 2.0 = 2.34. Regression caught: the resolution of the farther
    sighting or of the last frame, the harmful side `radial` or null, a threshold without the body."""
    s = scene()
    pk.put_person(s, "A", [(0, 10.0, (2.0, 3.63, 0.0)), (10.0, None, (2.0, 2.63, 0.0))], [2, K])
    _, cand = only(pk.world(s))
    assert cand.gold == {"type": "number", "value": 1.22}
    assert cand.tolerances == {"resolution_m": pytest.approx(0.04910, abs=1e-4), "need_m": 2.0, "harm": "over",
                               "threshold_m": pytest.approx(2.34)}


def test_a_no_one_gold_is_resolved_at_the_cap_from_the_tasks_last_camera():
    """Contract (brief 3.7; `questions_people._task_cand`): the answer 10 ("no one within 10 m") rests on no sighting,
    so its resolution is one pixel at a floor point 10 m from the camera of the task's last keyframe, the base z the
    floor. By hand: d = 10, h = 0.08: (100 + 0.0064) / 30.5089 = 3.2779 radially, 0.02622 sideways, 3.278 together,
    whatever the camera's place; the need is still C = 2.0 (the kernel widens it to twice the resolution). Regression
    caught: a no-one row with the person's rule (no sighting to read: a crash, or the floor of 1 cm, which would make
    the answer 10 a precision the camera cannot give)."""
    _, cand = only(pk.world(scene(), cid=case_with(True, tasks=("0",))))
    assert cand.gold == {"type": "number", "value": 10.0} and cand.answer_class == "no_one"
    assert cand.tolerances == {"resolution_m": pytest.approx(3.278, abs=1e-3), "need_m": 2.0, "harm": "over",
                               "threshold_m": pytest.approx(2.34)}


def test_person_mem_is_resolved_from_the_sighting_camera_and_an_unknown_row_holds_the_floor():
    """Contract (brief 3.7, an offset: at the evidence sighting's camera, to the gold point; `questions_people.
    person_mem`): the row carries C as the need, the harmful side `radial` and the resolution from the camera of the
    sighting the hull was read from, not the look point's. By hand: the person at (2.0, 3.63) seen at keyframe 2 (camera
    (2.0, 1.63, 0.08)) has the gold point (2.17, 3.35), d = 1.7284: (2.9874 + 0.0064) / 30.5089 = 0.09813 radially,
    0.00453 sideways, 0.0982 together; from the look point's camera (3.0, 1.13) it would be d = 2.3701 and 0.1844. A
    world with no row of any person asks `unknown`, whose tolerances hold the floor resolution 0.01. Regression caught:
    the ask's camera used for a remembered person, a missing harm, or an unknown row the record would refuse."""
    s = pk.scene()
    pk.put_person(s, "A", (2.0, 3.63), [2])
    m = person_mem(pk.cut(s, K, pk.one_person()), K)
    (c,) = m.answerable
    assert c.tolerances == {"resolution_m": pytest.approx(0.0982, abs=1e-4), "need_m": 2.0, "harm": "radial"}
    empty = pk.scene()
    put(empty, "T", "bag", "red", (2.0, 1.8), [K])   # a thing is no person
    (u,) = person_mem(pk.cut(empty, K, pk.one_person()), K).unknown
    assert u.gold == {"type": "unknown"}
    assert u.tolerances == {"resolution_m": 0.01, "need_m": 2.0, "harm": "radial"}
