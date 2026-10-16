"""The support side of the surface tie (`ego_support`, called by `questions_ego._cand`): a thing resting on a surface
whose top is at or above the camera is asked by `ego_now` and `ego_mem` only where it stands at the surface's near edge
(the edge tie), unless a contact pixel at the surface ties it; one hanging on a face has no base. The scene is
`ego_kit`'s (camera at (3.0, 1.13, 0.08) at the look point 13, heading north) with a table 1 x 1 x 0.75 m centred at
(3.0, 4.0): its near edge is the line y = 3.5, x from 2.5 to 3.5. The job's clearance is 0.4 m (C/2 = 0.2 m), so a thing
is asked while its nearest point lies within 0.2 m of the near edge along the sight line.

A thing on a table above the camera shows no contact with it (the camera sees its underside and front face), so the bags
here are seen as their top corners alone, `top_only`, and the edge tie decides. Most tests call `set_back` itself on the
gold point worked out by hand."""

from __future__ import annotations

from dataclasses import replace

import pytest

import family_kit as fk
from ego_kit import CAMERA, EGO_JOB, K, named, put, scene
from stbench.ego_support import contact_bases, set_back, set_back_m
from stbench.geometry import Rect
from stbench.questions_ego import ego_mem, ego_now

JOB = replace(EGO_JOB, clearance=0.4)
SIZE = (0.2, 0.2, 0.1)


def rest(s, eid: str, colour: str, xy, *, seen=(K,), z: float = 0.75, support: str | None = "TABLE",
         top_only: bool = False) -> None:
    """A bag centred at `xy` with its base `z` up, resting on `support` (the placement's `on`), seen at `seen` as the
    eight corners of its box, or its top four alone when `top_only` (a camera below the surface sees no contact)."""
    pts = [(xy[0] + a * SIZE[0] / 2, xy[1] + b * SIZE[1] / 2, c) for a in (-1, 1) for b in (-1, 1)
           for c in ((z + SIZE[2],) if top_only else (z, z + SIZE[2]))]
    put(s, eid, "bag", colour, xy, seen, size=SIZE, where=[(0, None, (*xy, z))], pts=pts)
    s.placements[:] = [replace(p, on=support) if p.entity == eid else p for p in s.placements]


def table(s, xy=(3.0, 4.0), size=(1.0, 1.0, 0.75)) -> None:
    put(s, "TABLE", "table", "oak", xy, (), size=size, mobility="fixed")


def backed(s, eid: str, point, evidence=(K,), job=JOB) -> bool:
    """Whether the screen calls the thing set back, its gold point `point` read from the camera at the look point."""
    return set_back(fk.world(s, job=job), eid, list(evidence), CAMERA[:2], point)


def asked_now(s) -> set[str]:
    return set(named(ego_now(fk.world(s, job=JOB), K)))


def test_the_set_back_is_measured_along_the_sight_line_into_the_footprint():
    """Contract (`ego_support.set_back_m`): how far behind the near edge a point lies, from the camera's floor point.
    By hand, a 2 x 2 rectangle at the origin (edges at +-1): a camera at (0, -5) and a point (0, 0.5) enter at y = -1,
    1.5 behind it; a camera at (-5, -5) and a point (0, 0) enter the corner (-1, -1) at 80% of the way, 0.2 x 7.0711 =
    1.4142 left; a point outside the rectangle (0, 2) is 0 behind; a camera inside it is that far from the point
    (0.5). Regression caught: the distance to the point from the edge's nearest point (which counts the length of an
    oblique edge as depth), the whole distance from the camera, or a point outside counted as inside."""
    rect = Rect(0.0, 0.0, 2.0, 2.0, 0.0)
    assert set_back_m(rect, (0.0, -5.0), (0.0, 0.5)) == pytest.approx(1.5)
    assert set_back_m(rect, (-5.0, -5.0), (0.0, 0.0)) == pytest.approx(1.4142, abs=1e-4)
    assert set_back_m(rect, (0.0, -5.0), (0.0, 2.0)) == 0.0
    assert set_back_m(rect, (0.0, 0.0), (0.5, 0.0)) == pytest.approx(0.5)


def test_a_thing_at_the_edge_of_a_high_table_is_not_set_back_and_one_further_back_is():
    """Contract (the set-back screen, `ego_support.set_back`): on a table above the camera, a thing is set back when its
    nearest point lies more than C/2 behind the table's near edge along the sight line. By hand (C/2 = 0.2; the camera
    at (3.0, 1.13, 0.08), the table's edge at y 3.5, x 2.5 to 3.5): bags 0.2 m square on it (base 0.75 m) at y 3.6
    (nearest point (3.0, 3.5), on the edge: 0 back) and at (2.7, 3.75) (nearest (2.8, 3.65): 0.15 back) are not; at
    (3.3, 3.85) (nearest (3.2, 3.75): 0.25 back) and (3.0, 4.2) (nearest (3.0, 4.1): 0.6 back) they are. A bag on the
    floor (no support), nearest point (2.1, 1.83), is not. Regression caught: no screen (AMR case 5, a box 0.23 m back
    on a table above the camera, and the vacuum's picture frame on a desk), or the edge measured past C/2."""
    s = scene()
    table(s)
    rest(s, "EDGE", "red", (3.0, 3.6))
    rest(s, "NEAR", "blue", (2.7, 3.75))
    rest(s, "BACK", "green", (3.3, 3.85))
    rest(s, "FAR", "yellow", (3.0, 4.2))
    rest(s, "FLOOR", "purple", (2.0, 1.93), z=0.0, support=None)
    points = (("EDGE", (3.0, 3.5)), ("NEAR", (2.8, 3.65)), ("BACK", (3.2, 3.75)), ("FAR", (3.0, 4.1)),
              ("FLOOR", (2.1, 1.83)))
    assert [backed(s, eid, point) for eid, point in points] == [False, False, True, True, False]


def test_the_whole_near_edge_counts_not_only_its_point_nearest_the_camera():
    """Contract (`ego_support.set_back`, along the sight line): a thing at the near edge but to one side of the
    camera's foot is not set back. By hand: a bag at (3.4, 3.6) has its nearest point at the corner (3.3, 3.5), on the
    edge, 0.3 m from the edge's point nearest the camera (3.0, 3.5); the sight line enters the table at that same
    corner, so it is 0 back. Regression caught: the support's nearest point to the camera used in place of its near
    edge (the brief's wording), which screens every thing at an edge but straight ahead."""
    s = scene()
    table(s)
    rest(s, "SIDE", "red", (3.4, 3.6))
    assert not backed(s, "SIDE", (3.3, 3.5))


def test_a_surface_below_the_camera_hides_nothing():
    """Contract (the comparator in `ego_support.set_back`): a thing whose base is below its camera is looked down on, so
    where it stands on the surface is in the frame, however far back. By hand: a mat 3 x 3 x 0.05 m at (3.0, 4.0), a
    bag on it at y 4.2 (base 0.05 m, under the camera's 0.08 m; its nearest point (3.0, 4.1) 1.6 m behind the mat's
    edge, C/2 = 1.0) is not set back; the same bag on a surface 0.1 m up (over the camera) is. Regression caught:
    every resting thing screened whatever the surface's height."""
    for z, expected in ((0.05, False), (0.10, True)):
        s = scene()
        put(s, "MAT", "mat", "oak", (3.0, 4.0), (), size=(3.0, 3.0, z), mobility="fixed")
        rest(s, "ON", "red", (3.0, 4.2), z=z, support="MAT")
        assert backed(s, "ON", (3.0, 4.1), job=EGO_JOB) is expected


def test_the_edge_tie_asks_a_thing_at_a_high_edge_and_not_one_set_back_in_both_families():
    """Contract (`ego_support.edge_tie`, applied by `questions_ego._cand` for both families, the sight line from the
    camera of the sighting): a bag on the table seen by its top corners alone (no contact) is asked by `ego_now` when it
    stands at the edge and not when it stands back; `ego_mem` the same for a bag seen at keyframe 2 and out of the look
    window at the look point. By hand (C/2 = 0.2, the table's edge at y 3.5): a bag at y 3.6 (nearest point (3.0, 3.5))
    is on the edge, from the camera at (3.0, 1.13) at the look point and from (2.0, 1.63) at keyframe 2; a bag at (3.3,
    4.2) (nearest (3.2, 4.1)) is 0.6 back from the first and 0.67 from the second. Regression caught: the tie written
    into one family's candidate alone, or the look point's camera used for a sighting made from another."""
    now, mem = scene(), scene()
    for s, seen in ((now, (K,)), (mem, (2,))):
        table(s)
        rest(s, "EDGE", "red", (3.0, 3.6), seen=seen, top_only=True)
        rest(s, "BACK", "green", (3.3, 4.2), seen=seen, top_only=True)
    assert asked_now(now) == {"EDGE"}
    assert set(named(ego_mem(fk.world(mem, job=JOB), K))) == {"EDGE"}


def test_a_thing_on_a_low_surface_is_asked_through_its_contact_at_the_surface_top():
    """Contract (the contact tie with a support's top as the base): a bag on a mat 0.05 m high, under the camera's
    0.08 m, shows the corners where it meets the mat, which tie it however far back from the mat's edge. By hand (C =
    2.0, C/2 = 1.0; the camera at (3.0, 1.13, 0.08)): a bag at y 4.2 has its nearest corner (3.0, 4.1, 0.05) at d = 2.97
    and H = 0.03, so e = (8.8209 + 0.0009) / (0.03 * 381.36) = 0.771 <= 1.0 and it is asked; seen by its top corners
    alone (no contact) it is not (its mat is below the camera, so there is no edge tie). Regression caught: the floor
    assumed for a thing on a surface, or a thing on a surface below the camera tied only by an edge."""
    for top_only, expected in ((False, {"ON"}), (True, set())):
        s = scene()
        put(s, "MAT", "mat", "oak", (3.0, 4.0), (), size=(3.0, 3.0, 0.05), mobility="fixed")
        rest(s, "ON", "red", (3.0, 4.2), z=0.05, support="MAT", top_only=top_only)
        assert set(named(ego_now(fk.world(s, job=EGO_JOB), K))) == expected


def test_a_thing_hanging_on_a_face_is_never_tied():
    """Contract (`ego_support.contact_bases`, `edge_tie`): a thing whose placement names a support but whose gold point
    lies outside the support's footprint (a coat on a wardrobe's side) rests on nothing: it has no base, and no edge
    tie, though its lowest corners are at the z its placement gives. By hand: the table's footprint is x 2.5 to 3.5, y
    3.5 to 4.5; a bag at (3.0, 3.4) has its nearest point (3.0, 3.3), 0.2 outside it, so its base is None and it is not
    asked; the same bag at the edge (3.0, 3.6), gold (3.0, 3.5), has the base 0.75. Regression caught: a hanging coat
    tied by its hem, or by the edge of the wardrobe it hangs on."""
    s = scene()
    table(s)
    rest(s, "HANG", "red", (3.0, 3.4))
    rest(s, "EDGE", "blue", (3.0, 3.6))
    w = fk.world(s, job=JOB)
    assert contact_bases(w, "HANG", [K], (3.0, 3.3)) == [None]
    assert contact_bases(w, "EDGE", [K], (3.0, 3.5)) == [0.75]
    assert "HANG" not in asked_now(s)
