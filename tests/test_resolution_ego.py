"""The resolution and the need of `ego_now` and `ego_mem` golds (0.9.0, `questions_ego._cand`, `resolution_at`): the row
carries C as the need, the harmful side radial and `resolution_m`, one pixel of the camera at the gold point, from the
sighting that tied the point. Hand-worked on `ego_kit`'s scene with the camera raised to 1.30 m
(`resolution_kit.raised`), the look point 13 at camera (3.0, 1.13), keyframe 2 at (2.0, 1.63); the job's clearance C is
2.0 m. One pixel at a floor point d from a camera h above its plane is hypot((d^2 + h^2) / (h fy), sqrt(d^2 + h^2) /
fx), fx = fy = 381.36115, at least 0.01."""

from __future__ import annotations

import math

import pytest

import family_kit as fk
from ego_kit import EGO_JOB, K, named, put, scene
from resolution_kit import raised
from space_kit import record
from stbench.questions_ego import ego_mem, ego_now
from stbench.questions_world import World
from test_ego_support import JOB as SUPPORT_JOB
from test_ego_support import rest, table


def raised_world(s, job=EGO_JOB) -> World:
    return World(raised(fk.inputs(s, job)), record(s))


def row(c) -> dict:
    return c.tolerances


def test_the_row_carries_c_the_radial_harm_and_one_pixel_at_the_gold_point_never_under_a_centimetre():
    """Contract (brief 3.7, `questions_ego._cand`): an `ego_now` row's tolerances are the clearance C as the need, the
    harmful side `radial` (a thing placed farther from the camera than it is) and the resolution at the gold point from
    the camera that saw it. By hand (camera (3.0, 1.13, 1.30), C = 2.0, the bags' base on the floor): a red bag whose
    nearest hull point is (3.0, 4.0) is d = 2.87 away, h = 1.30: (8.2369 + 1.69) / (1.3 * 381.36115) = 0.020023 radially
    and sqrt(9.9269) / 381.36115 = 0.008262 sideways, 0.02166 together; a blue bag whose nearest point is (3.0, 2.0),
    d = 0.87, would be 0.00642 and is held at the floor of 0.01. Regression caught: a resolution read from the gold's
    range alone (no height), the need taken from the job's reacquire radius, a resolution under the 1 cm floor the
    kernel needs, or a row that lost its harmful side."""
    s = scene()
    put(s, "T", "bag", "red", (3.0, 4.1), [K], pts=[(2.9, 4.0, 0.0), (3.1, 4.0, 0.0), (3.0, 4.2, 0.0)])
    put(s, "N", "bag", "blue", (3.0, 2.1), [K], pts=[(2.9, 2.0, 0.0), (3.1, 2.0, 0.0), (3.0, 2.2, 0.0)])
    found = named(ego_now(raised_world(s), K))
    assert list(found) == ["T", "N"]
    assert found["T"].gold == {"type": "offset", "ahead_m": pytest.approx(2.87, abs=1e-4),
                               "left_m": pytest.approx(0.0, abs=1e-4)}
    assert row(found["T"]) == {"resolution_m": pytest.approx(0.02166, abs=2e-5), "need_m": 2.0, "harm": "radial"}
    assert row(found["N"]) == {"resolution_m": 0.01, "need_m": 2.0, "harm": "radial"}


def test_an_ego_mem_row_reads_the_camera_of_the_sighting_not_the_cameras_of_the_ask():
    """Contract (brief 3.7: the evidence sighting's camera): the resolution of a remembered point is one pixel from the
    camera at the sighting that saw it, not the camera at the look point. By hand: a red bag seen only at keyframe 2
    (camera (2.0, 1.63, 1.30)) as the triangle (1.9, 4.5), (2.1, 4.5), (2.0, 4.7) has its nearest point to the look
    point's floor point (3.0, 1.13) at the vertex (2.1, 4.5): 3.37 ahead and 0.9 left. From the sighting's camera that
    point is d = 2.8717 away: (8.2469 + 1.69) / 495.7695 = 0.020043, sideways 0.008266, 0.02168 together; from the look
    point's camera it would be d = 3.4881 away: 0.0296. Regression caught: the ask's camera used for the resolution (a
    remembered point resolved from where the robot stands now), or the first or the middle frame."""
    s = scene()
    put(s, "A", "bag", "red", (2.0, 4.6), [2], pts=[(1.9, 4.5, 0.0), (2.1, 4.5, 0.0), (2.0, 4.7, 0.0)])
    a = named(ego_mem(raised_world(s), K))["A"]
    assert a.gold == {"type": "offset", "ahead_m": pytest.approx(3.37, abs=1e-4),
                      "left_m": pytest.approx(0.9, abs=1e-4)}
    assert row(a) == {"resolution_m": pytest.approx(0.02168, abs=2e-5), "need_m": 2.0, "harm": "radial"}


def test_the_sighting_with_the_least_contact_error_sets_the_resolution_not_the_last_one():
    """Contract (brief 3.7; `questions_ego._cand`): of several evidence sightings, the one whose contact gave the
    surface tie (the least contact error) is the camera the resolution is read from, the earliest of equals. By hand: a
    red bag seen at keyframes 1 (camera (3.0, 3.13, 1.30) after the base is put at (3.0, 3.0) heading north) and 2
    (camera (2.0, 1.63, 1.30)) as the triangle (2.9, 5.0), (3.1, 5.0), (3.0, 5.2): the gold point is (3.0, 5.0).
    Keyframe 1 is 1.87 m from it, its contact pixels read to (3.5069 + 1.69) / 495.77 = 0.01048 and the point to
    0.01205; keyframe 2 is 3.5152 m away and reads to 0.02795 and 0.02999. The resolution is 0.01205, not the last
    frame's 0.02999. Regression caught: the last evidence frame used when a contact tied the point (the edge tie's
    rule), or the first of any."""
    s = scene(poses={1: (3.0, 3.0)}, yaws={1: math.pi / 2})
    put(s, "A", "bag", "red", (3.0, 5.1), [1, 2], pts=[(2.9, 5.0, 0.0), (3.1, 5.0, 0.0), (3.0, 5.2, 0.0)])
    a = named(ego_mem(raised_world(s), K))["A"]
    assert a.evidence_k == (1, 2)
    assert a.gold == {"type": "offset", "ahead_m": pytest.approx(3.87, abs=1e-4),
                      "left_m": pytest.approx(0.0, abs=1e-4)}
    assert row(a)["resolution_m"] == pytest.approx(0.01205, abs=2e-5)


def test_an_edge_tied_point_is_resolved_from_the_last_evidence_frame_at_the_supports_top():
    """Contract (brief 3.7: the last evidence frame when no contact tied the point; `resolution_at`: the plane is the
    base the thing rests on): a bag on a table 0.75 m high seen by its top corners alone (no contact; the camera is 0.08
    m up, under the top) is tied by the table's edge, and its resolution is read from the last evidence frame to the
    gold point on the plane of the table's top. By hand (the table's near edge at y 3.5; C = 0.4): a bag at (3.0, 3.6)
    has its nearest point (3.0, 3.5). Seen at keyframes 1 (camera (3.0, 2.13, 0.08), 1.37 m from the point) and 2
    (camera (2.0, 1.63, 0.08), d = 2.1206): h = 0.75 - 0.08 = 0.67, so keyframe 2 gives (4.4969 + 0.4489) / (0.67 *
    381.36115) = 0.019356 radially, 0.005832 sideways, 0.02022 together; keyframe 1 would give 0.00994, held at 0.01;
    the camera at the look point (d = 2.37) 0.0246; the floor as the plane (h = 0.08) 0.1477. Regression caught: the
    first frame, the look point's camera or the floor used when the support is the plane."""
    s = scene(poses={1: (3.0, 2.0)}, yaws={1: math.pi / 2})
    table(s)
    rest(s, "EDGE", "red", (3.0, 3.6), seen=(1, 2), top_only=True)
    edge = named(ego_mem(fk.world(s, job=SUPPORT_JOB), K))["EDGE"]
    assert edge.evidence_k == (1, 2)
    assert row(edge) == {"resolution_m": pytest.approx(0.02022, abs=2e-5), "need_m": 0.4, "harm": "radial"}


def test_an_unknown_row_carries_the_job_s_need_and_the_floor_resolution():
    """Contract (`questions_ego._unknown`): an unknown `ego_mem` row asks no point, so its tolerances hold C, the radial
    harm and the floor resolution 0.01 (no scorer reads them on an unknown gold, and the row must still be a valid
    graded row). By hand: a blue bag no frame shows is unknown; C = 2.0. Regression caught: an unknown row whose
    tolerances are empty or are the old ladder (the record refused, or a scorer reading a default)."""
    s = scene()
    put(s, "A", "bag", "red", (2.0, 4.6), [2], pts=[(1.9, 4.5, 0.0), (2.1, 4.5, 0.0), (2.0, 4.7, 0.0)])
    put(s, "U", "bag", "blue", (1.0, 1.0), [])
    moment = ego_mem(raised_world(s), K)
    unknown = [c for c in moment.unknown if c.subjects == ("e:U",)]
    assert len(unknown) == 1
    assert row(unknown[0]) == {"resolution_m": 0.01, "need_m": 2.0, "harm": "radial"}
