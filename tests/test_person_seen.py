"""The person evidence rule (`person_seen`): when one frame's sighting of a person places them. Hand-worked
values; the camera at keyframe 13 is (3.0, 1.13, 0.08) heading north, the person a 0.34 x 0.56 m footprint at
(3.0, 3.13), so its floor hull is the rectangle (2.83, 2.85)-(3.17, 3.41) and the point of it nearest the camera's floor
point is (3.0, 2.85), 1.72 m ahead."""

from __future__ import annotations

import numpy as np
import pytest

import person_kit as pk
from ego_kit import CLEARANCE_M, K
from stbench import person_seen
from stbench.camera import Camera
from synth import camera

HULL = [[2.83, 2.85], [3.17, 2.85], [3.17, 3.41], [2.83, 3.41]]   # counter-clockwise from the lowest vertex
CAMERA = (3.0, 1.13, 0.08)


def seen_hull(c_m: float = CLEARANCE_M, **sighting):
    """The hull of the person's sighting at keyframe 13 given how it was seen (`put_person`'s keywords), else None."""
    s = pk.scene()
    pk.put_person(s, "P", (3.0, 3.13), [K], **sighting)
    return person_seen.hull(person_seen.frame_of(pk.world(s), "P", K), "P", c_m)


@pytest.mark.parametrize("sighting, evidence", [
    ({"bbox": [100, 0, 140, 300]}, True),           # cut by the frame's top: the feet are in frame
    ({"bbox": [0, 100, 140, 300]}, True),           # cut by the left side
    ({"bbox": [100, 100, 640, 300]}, True),         # cut by the right side
    ({"bbox": [0, 0, 640, 300]}, True),             # top and both sides at once
    ({"bbox": [100, 100, 140, 478]}, True),         # two pixels above the bottom edge, as the floor's rule allows
    ({"bbox": [100, 100, 140, 479]}, False),        # one pixel above it: the feet are cut
    ({"bbox": [100, 100, 140, 480]}, False),        # on the bottom edge
    ({"px": 1024}, True), ({"px": 1023}, False),    # the floor's pixel count, 32 x 32
    ({"lum": 0.01}, True), ({"lum": 0.0099}, False),
    ({"depth": None}, False),                       # no depth for the person at all
    ({"gaps": 1}, False),                           # one pixel without a depth: the hull would miss it
])
def test_the_bottom_edge_is_the_only_edge_that_costs_a_person_their_sighting(sighting, evidence):
    """Contract (FD-task-questions review, section 3.2): a person's sighting is evidence under the floor's conditions
    (1,024 px, `RECOG_LUM` = 0.01, a depth, no pixel without one) with one change: the box may touch the top and the
    sides of the image, never the bottom edge (within 2 px of it, `RECOG_EDGE_PX`), where the feet would be cut. By hand
    from the table: a box cut at the top is evidence, a box whose bottom is 2 px above the edge is, 1 px or 0 px is not;
    1,024 px and a luminance of 0.01 pass, one less does not. The hull is the footprint's rectangle. Regression caught:
    the floor's all-edge rule kept for people (41 of the AMR's 46 closest-approach sightings, cut at the top, lost), or
    an edge rule so loose that a person cut at the feet places them wrongly."""
    assert (seen_hull(**sighting) == HULL) is evidence
    assert seen_hull(**sighting) in (HULL, None)


@pytest.mark.parametrize("c_m, evidence", [(0.2, True), (0.18, False)])
def test_the_contact_tie_decides_at_the_clearance(c_m, evidence):
    """Contract: the sighting is evidence only where a camera-only reader could place the person's nearest foot to C/2:
    a visible contact pixel within C/2 of the nearest point, reading to within C/2 at one pixel. By hand: the nearest
    point is (3.0, 2.85), d = 1.72 m from the camera's floor point, which is H = 0.08 m above the floor, so one pixel
    moves it (d^2 + H^2) / (H fy) = (2.9584 + 0.0064) / (0.08 x 381.36) = 0.0972 m: within C/2 = 0.10 m (C = 0.2), not
    within 0.09 m (C = 0.18). Regression caught: no tie (a person whose feet a reader cannot place is evidence), or the
    tie read at the wrong clearance."""
    assert (seen_hull(c_m) == HULL) is evidence


def test_a_person_whose_feet_no_pixel_shows_is_not_evidence():
    """Contract: the contact tie needs a pixel near the floor (within 2 cm, or one pixel's height at its range) within
    C/2 of the nearest point. Points of the person from 0.9 m up only (the feet behind a table) have none, so the
    sighting is not evidence at any clearance. Regression caught: a tie read from the head, which would place a
    person standing behind something by where their shoulders are."""
    high = pk.body((3.0, 3.13))
    assert seen_hull(pts=high[high[:, 2] > 0.5]) is None
    assert seen_hull(pts=high) == HULL


def test_a_row_the_log_cannot_read_raises_naming_the_person():
    """Contract: as `recognise.recognisable`, a row with enough pixels that lacks `lum`, `bbox` or `depth_med_m`
    (a log that cannot say) raises, naming the person and keyframe, while a row under the pixel count is simply not
    evidence. Regression caught: a missing key counted as a pass or a fail without a word."""
    s = pk.scene()
    pk.put_person(s, "P", (3.0, 3.13), [K])
    pts = pk.body((3.0, 3.13))

    def frame(rec):
        return person_seen.Frame(s.timeline(), camera("mm"), K, CAMERA, rec, lambda k, e: pts)

    ok = {"px": 1500, "bbox": [100, 100, 140, 140], "depth_med_m": 2.0, "lum": 0.25}
    assert person_seen.hull(frame(ok), "P", CLEARANCE_M) == HULL
    assert person_seen.hull(frame(None), "P", CLEARANCE_M) is None
    assert person_seen.hull(frame({"px": 100}), "P", CLEARANCE_M) is None
    with pytest.raises(ValueError, match="P at keyframe 13 has no lum"):
        person_seen.hull(frame({k: v for k, v in ok.items() if k != "lum"}), "P", CLEARANCE_M)
    with pytest.raises(ValueError, match="no bbox"):
        person_seen.hull(frame({**ok, "bbox": None}), "P", CLEARANCE_M)
    assert isinstance(camera("mm"), Camera) and np.isfinite(CAMERA).all()   # the kit's camera is a real one


def test_the_kind_word_is_the_catalogs():
    """Contract: `person_seen.PERSON_KIND` is the kind word every people catalog gives its movers, so the oracle channel
    and the golds treat exactly the entities `people.add` makes. Regression caught: a renamed kind that silently turns
    the person rule off (people fall back to the floor's all-edge rule)."""
    from stbench import people_downtown, people_home, people_warehouse

    assert {people_home.KIND, people_warehouse.KIND, people_downtown.KIND} == {person_seen.PERSON_KIND}
