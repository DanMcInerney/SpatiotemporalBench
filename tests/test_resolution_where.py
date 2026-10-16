"""The resolution and the need of `object_where` golds (0.9.0, `questions_where.tolerances_of`): the need is the job's
re-find radius R, the resolution the larger of one pixel at the last recognisable sighting and the hidden widening; the
row holds no room (the direction quarter of a place is `e <= R` alone). Hand-worked on `space_kit`'s run (the vacuum,
camera 0.08 m up; R = 0.5 m) over the hand map (kitchen x 0-4, living room x 4-8, both y 0-3); one pixel at a floor
point d from a camera h above it is hypot((d^2 + h^2) / (h fy), sqrt(d^2 + h^2) / fx), fx = fy = 381.36115, at least
0.01."""

from __future__ import annotations

import pytest

import family_kit as fk
from resolution_kit import raised
from space_kit import record, things_run, vacuum_run
from stbench.questions_where import pools
from stbench.questions_world import World


def test_a_place_carries_r_and_the_pixel_at_its_last_sighting_and_no_room():
    """Contract (brief 3.7, a place: one pixel at the last recognisable sighting; `tolerances_of`; 0.9.0 review B2): the
    row holds R = 0.5 as the need, no harmful side, the resolution at the place from the camera of the last
    recognisable sighting, and nothing else (no room, so no fact of the map and the kind can earn the direction
    quarter). By hand on `things_run`: E0001 (a bag) stands at (2, 1) from its last recognisable sighting at keyframe
    10, camera (5.9081, 1.5919, 0.08): d = 3.9527, so (15.6239 + 0.0064) / 30.5089 = 0.51234 radially, 0.01037
    sideways, 0.5124 together (its first sighting, at keyframe 1 when it stood elsewhere, does not count); E0004 (a
    box) at (3.5, 1.0) seen at keyframe 12 only, camera (5.9081, 1.4081): d = 2.4424, 0.1958. An unknown row holds the
    floor resolution 0.01. Regression caught: the resolution read from the first sighting or the ask, a need that is
    not R, a `room` key left on the row."""
    answerable, unknown = pools(fk.world(things_run()))
    got = {c.entities[0]: c.tolerances for c in answerable}
    assert got["E0001"] == {"resolution_m": pytest.approx(0.5124, abs=1e-4), "need_m": 0.5, "harm": None}
    assert got["E0004"] == {"resolution_m": pytest.approx(0.1958, abs=1e-4), "need_m": 0.5, "harm": None}
    assert {tuple(c.tolerances.items()) for c in unknown} == {(("resolution_m", 0.01), ("need_m", 0.5),
                                                               ("harm", None))}


@pytest.mark.parametrize(("raise_camera", "size", "at", "resolution"), [
    (True, (1.4, 1.4, 1.5), (5.0, 2.0), 0.45),    # the top above the camera: widened by 1.4 / 2 - 0.25, the pixel 0.01
    (True, (1.4, 1.4, 1.0), (5.0, 2.0), 0.01),    # the camera above the top saw it: not widened, the pixel at the floor
    (False, (1.0, 0.4, 0.5), (2.0, 1.0), 0.5124),  # widened by 0.25 but the pixel at 3.95 m from 0.08 m up is larger
    (False, (1.0, 0.4, 0.5), (5.0, 2.0), 0.25)])  # ... and at 1.0 m the widening is larger than the pixel 0.0328
def test_the_resolution_is_the_larger_of_the_pixel_and_the_hidden_widening(raise_camera, size, at, resolution):
    """Contract (brief 3.7: a place's resolution is the larger of one pixel and `rules.hidden_widening`;
    `questions_where.tolerances_of`): a place that a camera below the thing's top never saw cannot be read closer than
    the thing's half-length beyond 0.25 m, and a pixel coarser than that sets it instead. By hand (keyframe 10, camera
    (5.9081, 1.5919)): with the camera raised to 1.30 m, a 1.4 x 1.4 x 1.5 m thing at (5, 2) has its top above the
    camera, widening 0.7 - 0.25 = 0.45, and a pixel of 0.0069 (held at 0.01): 0.45; 1.0 m tall, under the camera, its
    top was in view and nothing widens it: 0.01. With the vacuum's 0.08 m camera, a 1.0 x 0.4 x 0.5 m thing (widening
    0.25) at (2, 1) is d = 3.9527 away and its pixel 0.5124 sets it; at (5, 2), d = 0.9956, the pixel is 0.0328 and the
    widening 0.25 sets it. Regression caught: the widening added to the pixel, taken without the camera's height (every
    tall thing widened), or the pixel ignored where the widening is zero."""
    s = vacuum_run()
    s.item("E0001", "bag", {"color": "red"}, [(0, None, (*at, 0.0))], size=size)
    s.see([10], "E0001", 1500)
    inp = fk.inputs(s)
    w = World(raised(inp) if raise_camera else inp, record(s))
    (cand,) = pools(w)[0]
    assert cand.tolerances["resolution_m"] == pytest.approx(resolution, abs=1e-4)
    assert cand.tolerances["need_m"] == 0.5
