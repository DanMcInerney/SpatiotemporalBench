"""0.7.0's credit by the job's own numbers (lane D1; the row's shape is 0.9.0's): each job declares its clearance C,
`object_where` is credited on its re-find radius R with the hidden-top widening, and the prompt states exactly those
numbers. Expected values are worked by hand."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field, fields, replace

import pytest

import family_kit as fk
from space_kit import JOB, SpaceScript, things_run, vacuum_run
from stbench import job_registry, nearest_part, questions_text
from stbench.llm_render import detection_line, system_prompt
from stbench.questions_where import nearest_sighting_at, pools
from test_llm_render import public

# job id -> (C, R) in metres, from FD-0.7.0-review "The values to declare" (typed here, not read from the specs)
NUMBERS = {"vacuum/1": (0.10, 1.0), "vacuum/2": (0.10, 1.0), "helper/1": (0.15, 1.0), "amr/1": (0.45, 1.0),
           "drone/1": (2.4, 2.0)}
JOBS = job_registry.all_jobs()


def test_every_registered_job_declares_its_clearance_and_re_find_radius():
    """Contract: each registered job declares a positive clearance C (the margin its robot's own spec keeps beside its
    path) and a re-find radius R (within which it finds a thing again with it in view): vacuum C 0.10 m, helper 0.15 m,
    AMR 0.45 m (the MiR250's 0.435 m a side), drone 2.4 m (Wing's 8 ft from trees); R 1.0 m for the two home jobs
    (HomeRobot's real-world find rule, which they declare as `reacquire`) and the job's `match_m` for the AMR (1.0 m, a
    convention) and the drone (2.0 m, Wing's). By hand from FD-0.7.0-review. Regression caught: a job added without a
    clearance (its ego questions would refuse to compile, or credit by a guess), a value typed in the wrong unit (10 for
    0.10), a home job left at R = `match_m` 0.5 m, or an AMR or drone R that drifts from the `task_things` radius."""
    assert {job.id for job in JOBS} == set(NUMBERS)
    for job in JOBS:
        assert job.clearance_m > 0
        assert (job.clearance_m, job.reacquire_m) == NUMBERS[job.id], job.id
    assert {job.id: job.reacquire_m == job.match_m for job in JOBS} == {
        "vacuum/1": False, "vacuum/2": False, "helper/1": False, "amr/1": True, "drone/1": True}


def test_the_object_where_need_is_the_jobs_re_find_radius():
    """Contract: an `object_where` row's need is `job.reacquire_m`: for a job whose first rung is 0.5 m (match_m 1.0, so
    R 1.0 m) it is 1.0; for one that declares R = 0.8 m, 0.8; the job's own four-rung distance ladder (0.5, 1, 2, 4) is
    not the place's. Unknown rows carry the same need. Regression caught: R taken from the distance ladder or a constant
    (a place in the warehouse credited at the home's radius), or the declared `reacquire` ignored. (The widening and
    the pixel that set the row's resolution are `test_resolution_where`'s.)"""
    wide = replace(JOB, ladder=(0.5, 1.0, 2.0, 4.0))
    s = things_run()
    for job, need in ((wide, 1.0), (replace(JOB, reacquire=0.8), 0.8)):
        answerable, unknown = pools(fk.world(s, job=job))
        assert answerable and unknown
        assert {c.tolerances["need_m"] for c in answerable + unknown} == {need}


@dataclass
class Raised(SpaceScript):
    """A vacuum run in which the camera of some keyframes is raised: `lift` maps a keyframe to the metres its camera is
    higher than the vacuum's own 0.08 m."""

    lift: dict[int, float] = field(default_factory=dict)

    def timeline(self):
        tl = super().timeline()
        frames = [replace(kf, pos=(kf.pos[0], kf.pos[1], kf.pos[2] + self.lift.get(kf.k, 0.0)))
                  for kf in tl.ego.keyframes]
        return replace(tl, ego=replace(tl.ego, keyframes=frames))


def two_views(lift):
    """A red 1.0 x 0.4 x 0.5 m thing at (5, 2) recognisable at keyframes 10 and 12, the camera of keyframe 12 raised by
    `lift`; the cameras are (5.9081, 1.5919) and (5.9081, 1.4081), 0.08 m up (plus the lift at 12)."""
    base = vacuum_run()
    s = Raised(**{f.name: getattr(base, f.name) for f in fields(base) if f.init}, lift={12: lift})
    s.item("E0001", "bag", {"color": "red"}, [(0, None, (5.0, 2.0, 0.0))], size=(1.0, 0.4, 0.5))
    s.see([10, 12], "E0001", 1500)
    return fk.world(s)


def test_the_widening_is_the_least_over_every_sighting_at_the_place_and_the_range_is_the_nearest():
    """Contract (0.7.0, `place_widening`; 0.9.0, the widening sets the row's resolution when it is the larger): the
    widening grows the resolution by the half-length beyond 0.25 m only when the thing's top was above the camera at
    every recognisable sighting of it at the place, since one view from above shows its top face and puts the middle in
    view; `range_m` is still the nearest sighting's slant, and the evidence the last sighting. Hand values: the thing's
    top is 0.5 m and its widening 1.0 / 2 - 0.25 = 0.25 m; keyframe 10 is the nearer view (sqrt(0.9081^2 + 0.4081^2 +
    0.17^2) = 1.0100 m against 1.0972 m at keyframe 12) from a camera 0.08 m up, under the top. With the camera of
    keyframe 12 at 0.78 m (lifted 0.7 m), above the top, the least widening is 0 although the nearest sighting saw only
    the sides, so the resolution is the pixel at keyframe 12, 1.0840 m from the place from 0.78 m up: 0.0069, held at
    the floor 0.01; lifted 0.2 m (0.28 m, still under the top) both views were below the top, the widening 0.25 is
    larger than that pixel (0.0121) and is the resolution. Regression caught: the widening read from the nearest
    sighting alone (the drone's nest-level view widening a car most of whose sightings saw its top), or the last."""
    for lift, resolution in ((0.7, 0.01), (0.2, 0.25)):
        w = two_views(lift)
        near = nearest_sighting_at(w, w.tl.entities[0], (5.0, 2.0))
        assert near is not None and near[1] == 10 and near[0] == pytest.approx(1.0100, abs=1e-3)
        answerable, _ = pools(w)
        assert answerable[0].range_m == near[0] and answerable[0].evidence_k == (12,)
        assert answerable[0].tolerances["resolution_m"] == pytest.approx(resolution, abs=1e-4)


def family_modules():
    """The family modules that exist (a module not written yet is skipped, as the registry does)."""
    for name in questions_text.FAMILY_MODULES:
        try:
            yield importlib.import_module(f"stbench.{name}")
        except ModuleNotFoundError as error:
            if error.name != f"stbench.{name}":
                raise


def asked_families(job):
    return [*job.looks, *(job.ends if job.ends is not None else ("object_where",))]


@pytest.mark.parametrize("family", sorted({f for job in JOBS for f in asked_families(job)}))
def test_no_family_a_job_asks_credits_by_the_relative_ladder(family):
    """Contract: the prompt no longer states a relative share, so no family any registered job asks (its `looks` and
    `ends`) may build its rungs with `rules.relative_ladder` or `relative_tolerances` (they stay for the change
    families, which no job asks): the module that owns the family does not import either. Regression caught: a family
    left on the 0.6.0 ladder, scored in shares of a range the prompt no longer describes."""
    owner = next(module for module in family_modules() if family in module.GOLDS)
    assert {"relative_ladder", "relative_tolerances"}.isdisjoint(vars(owner)), owner.__name__


def test_the_gold_rule_picks_the_point_the_prompt_defines():
    """Contract (0.7.0, FD-0.7.0-review change 6; the prompt half is `test_subgoal_prompt`'s hand sentence):
    `nearest_part` takes, of the visible hull, the point nearest the camera's floor point. By hand, with that point at
    the origin: a thing whose near face alone was shown, the segment (2, -1) to (2, 1), gives (2, 0), and showing its
    back as well, (4, -1) to (4, 1), does not change it; a hull that holds the origin gives (0, 0). Regression caught: a
    gold that takes the thing's true near side or its middle, or one that leaves (0, 0) out."""
    near_face = [[2.0, -1.0], [2.0, 1.0]]
    box = [[2.0, -1.0], [4.0, -1.0], [4.0, 1.0], [2.0, 1.0]]
    over = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    assert nearest_part.nearest_on_hull(near_face, (0.0, 0.0)) == (2.0, 0.0)
    assert nearest_part.nearest_on_hull(box, (0.0, 0.0)) == (2.0, 0.0)
    assert nearest_part.nearest_on_hull(over, (0.0, 0.0)) == (0.0, 0.0)


MUG = {"kind": "mug", "attributes": {"color": "red"}, "bbox_px": [10, 20, 30, 40], "center_cam": [0.1, -0.3, 1.5],
       "center_map": [2.0, 1.0, 0.9], "extent_m": [0.1, 0.1, 0.12]}


def test_a_detection_line_shows_its_floor_hull_none_or_nothing():
    """Contract (0.7.0, FD-0.7.0-review change 8): a detection with a `floor_hull` (map-frame points, counter-clockwise)
    is rendered with it, each coordinate to the millimetre; a null one (the thing not clearly visible in the frame) as
    `floor hull none`; one without the key is the line it was, so the older perception still renders. By hand: a hull
    of (1, 2), (1.5, 2), (1.5, 2.5) gives `floor hull [(1.000, 2.000), (1.500, 2.000), (1.500, 2.500)]` after the box.
    Regression caught: the key dropped from the line (a detections-channel system could not read the outline the oracle
    table answers from), or rounded to the centimetre like the other numbers."""
    plain = ("  - mug (color red): map (2.00, 1.00, 0.90), camera (0.10, -0.30, 1.50), size 0.10 x 0.10 x 0.12 m, "
             "box (10, 20) to (30, 40) px")
    assert detection_line(MUG) == plain
    hull = [[1.0, 2.0], [1.5, 2.0], [1.5, 2.5]]
    shown = ", floor hull [(1.000, 2.000), (1.500, 2.000), (1.500, 2.500)]"
    assert detection_line({**MUG, "floor_hull": hull}) == plain + shown
    assert detection_line({**MUG, "floor_hull": None}) == plain + ", floor hull none"


def test_the_detections_prompt_tells_what_a_floor_hull_is_and_the_camera_prompt_does_not(tmp_path):
    """Contract (0.7.0, FD-0.7.0-review change 8): the detections channel's prompt says a detection may give the floor
    hull of the thing, the outline on the floor of what the camera showed, as map-frame (x, y) points in metres,
    counter-clockwise; the camera channel's prompt names no such thing (it has no detections), and neither mentions a
    depth. Regression caught: a field in the lines that the prompt never explains, or the sentence leaking into the
    camera-only prompt."""
    said = system_prompt(public(tmp_path / "d"), "detections")
    assert "the floor hull of the thing" in said and "counter-clockwise" in said and "depth" not in said
    assert "none when the thing is not clearly visible in the frame" in said
    assert "floor hull" not in system_prompt(public(tmp_path / "c"), "rgb_pose")
