"""The oracle channel's person `floor_hull`: set exactly when the sighting is evidence by `person_seen`,
every other thing's as for any thing. Hand-worked: the camera at keyframe 13 is (3.0, 1.13, 0.08) heading north."""

from __future__ import annotations

import numpy as np
import pytest

import person_kit as pk
from ego_kit import CLEARANCE_M, K, put
from stbench import episode as ep
from stbench import perception as P
from synth import camera

HULL = [[2.83, 2.85], [3.17, 2.85], [3.17, 3.41], [2.83, 3.41]]   # the person at (3.0, 3.13)
BAG_HULL = [[2.95, 1.95], [3.05, 1.95], [3.05, 2.05], [2.95, 2.05]]   # the 0.1 m bag at (3.0, 2.0)


def channel(s, c_m=CLEARANCE_M):
    """(detections of keyframe 13 by entity id) of the oracle channel built from the script's own files."""
    tl, cam = s.timeline(), camera("mm")
    points = lambda k, e: s.parts.get((k, e), np.empty((0, 3)))   # noqa: E731
    gaps = lambda k, e: s.gaps.get((k, e), 0)   # noqa: E731
    oracle, ids = P.build(tl, s.visibility(), ep.pose_rows(tl), cam, points, gaps, c_m)
    row, key = oracle[K], ids[K]["ids"]
    return {key[d["d"]]: d for d in row["detections"]}


def test_a_person_s_floor_hull_is_set_exactly_when_the_sighting_is_evidence():
    """Contract (FD-task-questions review, section 3.6): a person's detection carries a `floor_hull` exactly
    when `person_seen.hull` holds for that frame, so a table that keeps the hulls can reproduce the people golds. By
    hand, three people seen at keyframe 13 with 1,500 px: A at (3.0, 3.13) with the box cut by the frame's top has the
    footprint rectangle for its hull, B with the box on the bottom edge has none, and C seen in 300 px (detected, since
    215 px is seen) has none. The channel still detects all three. A bag cut by the top has none (the floor's rule
    stands for everything else) and a bag clear of the edges has its 0.1 m square. Regression caught: the oracle
    following `recognisable` for people (A lost: the ceiling gate fails on `closest_person`), or following nothing (B
    and C hulled: the table answers what the gold never asks)."""
    s = pk.scene()
    pk.put_person(s, "A", (3.0, 3.13), [K], bbox=[100, 0, 140, 300])
    pk.put_person(s, "B", (1.0, 3.13), [K], bbox=[100, 100, 140, 480])
    pk.put_person(s, "C", (5.0, 3.13), [K], px=300)
    put(s, "T", "bag", "red", (3.0, 2.0), [K], px=1500)
    s.see([K], "T", 1500, bbox=[100, 0, 140, 300])
    put(s, "U", "bag", "blue", (3.0, 2.0), [K], px=1500)
    det = channel(s)
    assert {e: d["floor_hull"] for e, d in det.items()} == {"A": HULL, "B": None, "C": None, "T": None, "U": BAG_HULL}


def test_a_person_with_pictures_needs_the_clearance():
    """Contract: the contact tie reads the job's clearance, so a channel built with pictures and a person but no `c_m`
    raises, naming the person, and one built without pictures has no hull at all. Regression caught: a default
    clearance (a hull set at a clearance no job declared)."""
    s = pk.scene()
    pk.put_person(s, "A", (3.0, 3.13), [K])
    with pytest.raises(ValueError, match="A is a person"):
        channel(s, c_m=None)
    tl = s.timeline()
    oracle, _ = P.build(tl, s.visibility(), ep.pose_rows(tl), camera("mm"))
    assert [d["floor_hull"] for d in oracle[K]["detections"]] == [None]
