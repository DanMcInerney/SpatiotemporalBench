"""The apartment's `under_m` against the drawings of `render/furniture.py`, in Blender: the
map tells the planner how much room a piece leaves under it (`furniture[i].under_m`, read by
`geometry.under_obstacles`), while depth and the picture show what the builder drew, so the two must agree.
`tests/blender_under_driver.py` builds each blocking piece alone and casts rays straight up under it; the rule is
applied here. Needs Blender; marked `blender`, skipped when it is not installed. No render and no GPU, a few seconds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import apartment
from stbench import blender_run
from stbench.geometry import Rect, point_in_rect, under_obstacles
from stbench.render_cli import blender_path
from stbench.robots import ROBOTS

REPO = Path(__file__).resolve().parents[1]
# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
TOL_M = 0.001  # a drawing may sit this far under the map's `under_m`
VACUUM = ROBOTS["vacuum"].body
PASSING_M = VACUUM.height_m + VACUUM.under_margin_m  # the highest underside the robot vacuum cannot enter, plus a hair


@pytest.fixture(scope="module")
def hits(tmp_path_factory) -> list[dict]:
    """Per blocking piece of apartment_a, the heights at which rays cast up from below the floor first meet its drawing
    (see the driver)."""
    out = tmp_path_factory.mktemp("under") / "hits.json"
    blender_run.run([blender_path(), "--background", "--factory-startup", "--python-exit-code", "1", "--python",
                     str(REPO / "tests" / "blender_under_driver.py"), "--", "--render", str(REPO / "render"),
                     "--src", str(REPO / "src"), "--out", str(out), "--map", "apartment_a"],
                    "casting rays under the apartment's furniture")
    return json.loads(out.read_text(encoding="utf-8"))


def violations(rows: list[dict], under_m: dict[str, float]) -> list[str]:
    """The ids of the pieces whose drawing does not keep what the map says of its underside. A piece with `under_m`
    above 0 leaves open floor to that height everywhere outside its four corner squares (where the legs are): the first
    thing a ray meets there is at least `under_m` up (less 1 mm), or nothing. A solid piece (0) leaves no way in below
    `PASSING_M`: every ray meets something lower than that."""
    bad = []
    for row in rows:
        under = under_m[row["id"]]
        legs = under_obstacles(Rect(0.0, 0.0, row["size"][0], row["size"][1], 0.0), under)
        for x, y, z in row["points"]:
            if under > 0:
                ok = z is None or z >= under - TOL_M or any(point_in_rect(leg, x, y) for leg in legs)
            else:
                ok = z is not None and z < PASSING_M
            if not ok:
                bad.append(row["id"])
                break
    return bad


def test_every_piece_honours_its_under_m(hits):
    """Contract (plan 2.2 and 2.9: the planner drives a robot under a piece only where `under_m` says there is room, and
    the camera sees through that gap, so the drawing must leave exactly what the map claims): built alone and probed
    with rays from 1 cm under the floor on a 5 cm grid, each blocking piece of the apartment has no thing lower than
    its `under_m` outside its four corner squares, and a piece with `under_m` 0 has something below the vacuum's passing
    height over its whole footprint. The rule itself is shown to bite: an `under_m` the builder does not reach, and a
    piece called solid that has floor under it, are both caught. Regression caught: a map value above its builder (the
    desk's apron lower than the map says, so the planner would route a robot into a box, or depth would certify floor
    the planner never enters), a solid piece with a gap below 10 cm (the toilet's cistern and bowl, the shoe rack's
    shelf: the vacuum's camera sees through it), or a piece the map forgot to give an underside."""
    apt = apartment()
    assert [row["id"] for row in hits] == [f.id for f in apt.furniture if f.blocks]
    under = {f.id: f.under_m for f in apt.furniture}
    assert violations(hits, under) == []
    for row in hits:  # a drawn piece: some ray met it (an empty mesh would pass every rule above)
        assert any(z is not None for _, _, z in row["points"]), row["id"]
    assert violations(hits, {**under, "F33": 0.65}) == ["F33"]
    assert violations(hits, {**under, "F07": 0.0}) == ["F07"]
