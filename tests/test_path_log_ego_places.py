"""`path_log`'s ego probe, judged where its rule fires and clustered at the job's own radius (0.7.0 lane D6, after the
Opus re-review of the probe). Hand-worked on the vacuum's camera: a base at (x, 1.0) heading north (90 degrees) has its
camera at (x, 1.13), and a chair's face at y = 4.0 is 2.87 m straight ahead, in view up to 2.4 m to either side (the
80 degree field) and never from 8 m off."""

from __future__ import annotations

import pytest

from baseline_kit import EVENTS, offset, question, record, started, step
from stbench.baselines_path import PathLog
from stbench.rules import clearance_ladder

CHAIR = {"kind": "chair", "attributes": {}}


def asked(tmp_path, template: str, records: list[dict], steps: list[dict]) -> dict:
    """`path_log`'s answer to a `template` question about a chair, its prior `records`, asked after `steps`."""
    system = started(PathLog, tmp_path, kinds=("kitchen",), events=EVENTS, prior=records)
    system.task(0, "2026-03-02T08:00:00", 0.0, "Clean the kitchen.")
    system.observe(steps)
    ask = question(template, CHAIR, answer_type="offset")
    return system.ask("during", {"k": steps[-1]["k"], "clock": ""}, [ask])["q1"]


def two_spots(template: str, cameras_of) -> list[dict]:
    """Eight other scripts asked `template` about a chair, four that saw it at spot A, (1.0, 4.0), and four at spot B,
    (9.0, 4.0), each from the base (x, 1.0) heading north of its own spot, 2.87 ahead and 0.0 left on the kit's rungs
    (0.25, 0.5, 1.0, 2.0; a last rung of 2.0 m, so the two spots 8 m apart are two places)."""
    ask = question(template, CHAIR, answer_type="offset")
    return [record(ask, offset(2.87, 0.0), script=f"s{i}", when="during", base=(1.0 if i < 4 else 9.0, 1.0), yaw=90.0,
                   cameras=cameras_of(1.0 if i < 4 else 9.0)) for i in range(8)]


def test_a_kind_at_two_spots_keeps_each_spot_where_it_is_in_view(tmp_path):
    """Contract (the Opus re-review of the probe; `baselines_path_ego._gains`): a place's rule is judged on the records
    whose own camera faced it, the ones it would answer, not on every record of the template and kind. By hand: a chair
    stood at A (1.0, 4.0) in four scripts and at B (9.0, 4.0) in the other four, each asked `ego_now` from 2.87 m in
    front of it. Under A's rule the four A records are answered by another A record's identical point (exact, +1 each)
    and the four B records, whose cameras never had A in view, are not scored; so A gains +1 on four scripts and is
    kept (the interval's low end is 1). Asked from the base (1.5, 1.0) heading north, with A in view, it answers A's
    point from the camera (1.5, 1.13): 2.87 ahead and 0.5 left (1.5 - 1.0). Asked from (9.5, 1.0), the same for B. The
    pool's constant offset, (2.87, 0.0), is what each gets where the place is not kept. Regression caught: a record
    about the other spot scored -1 against a place its camera never faced (A's gain 4 - 4 = 0, its interval's low end
    -1), so a kind at two spots keeps neither."""
    records = two_spots("ego_now", lambda x: (step(0, x, 1.0, 90.0)["camera"],))
    for x in (1.5, 9.5):
        got = asked(tmp_path / str(x), "ego_now", records, [step(0, x, 1.0, 90.0)])
        assert (got["ahead_m"], got["left_m"]) == pytest.approx((2.87, 0.5), abs=1e-3), x


def test_ego_mem_judges_a_place_on_the_records_that_faced_it_at_any_logged_step(tmp_path):
    """Contract (`baselines_path_ego._seen`): `ego_mem`'s rule fires when the place was faced at any logged step up to
    the ask, so a record is judged by every camera it had until its ask. By hand: the eight scripts of the two-spot
    case each faced their chair (heading north, steps 0) and then turned to face away (heading south, 270 degrees, the
    ask's step), so the golds are relative to the south-facing camera: the chair at (1.0, 4.0) from a camera at (1.0,
    0.87) is 3.13 behind and 0.0 left, -3.13 ahead. The south-facing camera never has the chair in view, the
    north-facing one does. Asked after the same turn from the base (1.5, 1.0): camera (1.5, 0.87) heading south has
    left = +x, so the point (1.0, 4.0) is 3.13 behind and 0.5 to the right: (-3.13, -0.5), not the pool's constant
    (-3.13, 0.0). `ego_now` asked at the same step has nothing in view, so it keeps no place. Regression caught: a
    record judged by its ask step's camera alone (nothing in view, so no place is ever scored), or by all records."""
    def cameras(x):
        return (step(0, x, 1.0, 90.0)["camera"], step(1, x, 1.0, -90.0)["camera"])

    mem = two_spots("ego_mem", cameras)
    for r in mem:
        r["yaw"] = -90.0
        r["gold"]["gold"] = offset(-3.13, 0.0)
    turned = [step(0, 1.5, 1.0, 90.0), step(1, 1.5, 1.0, -90.0)]
    got = asked(tmp_path / "mem", "ego_mem", mem, turned)
    assert (got["ahead_m"], got["left_m"]) == pytest.approx((-3.13, -0.5), abs=1e-3)
    now = two_spots("ego_now", cameras)
    for r in now:
        r["yaw"] = -90.0
        r["gold"]["gold"] = offset(-3.13, 0.0)
    assert asked(tmp_path / "now", "ego_now", now, turned) == offset(-3.13, 0.0)


def face_at(c_m: float, xs: list[float]) -> list[dict]:
    """One script a point of `xs`, each asked `ego_now` about a chair from the base (x, 1.0) heading north, its nearest
    part at (x, 4.0): 2.87 ahead and 0.0 left, on the job's clearance ladder `[c/2, c, 2c]` (`rules.clearance_ladder`),
    which sets the golds' matching radius; the golds state no `match_m`."""
    ask = question("ego_now", CHAIR, answer_type="offset")
    records = [record(ask, offset(2.87, 0.0), script=f"s{i}", when="during", base=(x, 1.0), yaw=90.0)
               for i, x in enumerate(xs)]
    for r in records:
        r["gold"]["tolerances"] = {"ladder": clearance_ladder(c_m)}
    return records


def test_the_cluster_radius_follows_the_jobs_clearance(tmp_path):
    """Contract (the Opus re-review of the probe; `baselines_path_ego._radius`): golds are one place within their last
    rung, 2C of the job's clearance, not a fixed 0.5 m. By hand: five scripts saw a chair's face at x = 2.4, 2.7, 3.0,
    3.3 and 3.6 (y = 4.0), 0.3 apart, from the base (x, 1.0), on the clearance C = 0.4 m: rungs 0.2, 0.4, 0.8, so a
    radius of 0.8 m. All five lie within 0.6 m of 3.0: one place, its points the whole face; held out, each record's
    neighbour is 0.3 m off (inside the second rung, +1/3 each), so the place is kept. Asked from the base (4.5, 1.0),
    the camera (4.5, 1.13), it answers the face's nearest point, (3.6, 4.0): 2.87 ahead and 0.9 left. At the old fixed
    0.5 m the face splits into {2.4, 2.7, 3.0} and {3.3, 3.6}, each of which loses on the records beyond it (3.6 is
    0.6 m from 3.0: outside the second rung), so none is kept and the answer is the pool's constant offset, (2.87,
    0.0). With C = 0.02 the radius is 0.04 m, the five points are five places of one point each, nothing is left to
    answer from when its own script is, and the answer is the pool's constant offset, (2.87, 0.0) (every record's
    gold, so it wins held out).
    Regression caught: a radius that ignores the job's rungs (the pool's 0.0 left for the large clearance, a place the
    face is split into)."""
    xs = [2.4, 2.7, 3.0, 3.3, 3.6]
    got = asked(tmp_path / "wide", "ego_now", face_at(0.4, xs), [step(0, 4.5, 1.0, 90.0)])
    assert (got["ahead_m"], got["left_m"]) == pytest.approx((2.87, 0.9), abs=1e-3)
    assert asked(tmp_path / "tight", "ego_now", face_at(0.02, xs), [step(0, 4.5, 1.0, 90.0)]) == offset(2.87, 0.0)
