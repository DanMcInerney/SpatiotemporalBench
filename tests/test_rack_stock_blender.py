"""The rack stock in Blender (`render/furniture_warehouse.py`): the six racks of
`warehouse_a` are built the way the scene builds them (`tests/blender_rack_stock_driver.py`) and what their stock holds
is checked against numbers worked out here from the rack's own dimensions (4 bays of 2.7 m with a surface rect 2.6 m
long and 1.08 m deep, levels 1.3 m apart under 0.12 m beams, upright posts 0.09 m along the rack at a pitch of 2.6775 m,
0.08 m across, set against the front and back). Needs Blender; marked `blender`, skipped when it is not installed. About
fifteen seconds.

Which bay-levels hold stock is `test_render_assets_blender.py`'s contract; this module is about how it is drawn."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from stbench import blender_run
from stbench.render_cli import blender_path

REPO = Path(__file__).resolve().parents[1]
# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
POST_PITCH, POST_Y, POST_X = 2.6775, 0.09, (0.47, 0.55)  # by hand: (10.8 - 0.09) / 4, and the posts' x range


@pytest.fixture(scope="module")
def stock(tmp_path_factory):
    """What the stock of each rack holds: per rack, a row [material, x0, x1, y0, y1, z0, z1, yaw] per part, built twice,
    and the tilt and shifts of 300 pallets built with room to spare."""
    out = tmp_path_factory.mktemp("stock") / "stock.json"
    blender_run.run([blender_path(), "--background", "--factory-startup", "--python-exit-code", "1", "--python",
                     str(REPO / "tests" / "blender_rack_stock_driver.py"), "--", "--render", str(REPO / "render"),
                     "--src", str(REPO / "src"), "--out", str(out)], "building the racks' stock")
    return json.loads(out.read_text(encoding="utf-8"))


def test_every_part_of_the_stock_stays_in_its_bay_level_clear_of_the_uprights(stock):
    """Contract (plan-realism section 4: the stock stays "inside the same bay-level volume as today"; the free test
    `dressing_zones` keeps tracked things out of free bay-levels, so stock leaking into one would sit in an item).
    By hand: a part in bay b (centre y within [-5.4 + 2.7 b, -2.7 + 2.7 b]) at level v (centre z within [1.3 v, 1.3 v +
    1.3]) lies within 1.3 m of its bay's middle along the rack, within 0.54 m of its middle across it, and from the
    level's height to 1.18 m above it (the next beam's underside); and it does not reach any of the posts, which are
    0.09 m wide at y = -5.355 + 2.6775 i and stand at x = 0.47 to 0.55 and -0.55 to -0.47. Regression caught: a jitter
    that pushes a pallet through an upright or into the neighbouring bay, stock that overhangs the rack's front into the
    aisle, or a stack that touches the beam above."""
    for rack_id, parts in stock["racks"].items():
        assert parts, rack_id
        for mat, x0, x1, y0, y1, z0, z1, _ in parts:
            bay, level = int(((y0 + y1) / 2 + 5.4) // 2.7), int(((z0 + z1) / 2) // 1.3)
            yc = -5.4 + 2.7 * (bay + 0.5)
            assert x0 >= -0.54 and x1 <= 0.54 and y0 >= yc - 1.3 and y1 <= yc + 1.3, (rack_id, mat, bay)
            assert z0 >= 1.3 * level - 1e-9 and z1 <= 1.3 * level + 1.18, (rack_id, mat, level)
            for i in range(5):
                post_y = -5.355 + POST_PITCH * i
                in_y = y0 < post_y + POST_Y / 2 - 1e-6 and post_y - POST_Y / 2 + 1e-6 < y1
                in_x = any(x0 < hi - 1e-6 and lo + 1e-6 < x1 for lo, hi in ((POST_X[0], POST_X[1]), (-POST_X[1],
                                                                                                     -POST_X[0])))
                assert not (in_x and in_y), (rack_id, mat, i)


def test_pallets_stand_off_true_by_up_to_two_degrees_and_three_centimetres(stock):
    """Contract (plan-realism section 4: "jitter of +-2 degrees and +-3 cm"): of 300 pallets built with room to spare,
    none turns more than 2 degrees or shifts more than 3 cm either way, and both reach most of that range (a jitter that
    is never applied, or one in a tenth of the range, is not what the plan asks for). Across the rack the shift is kept
    inside the 1.075 m between the back panel and the surfaces' front edge, so it is not asserted to reach 3 cm there.
    Regression caught: jitter dropped, scaled down, or unbounded."""
    tilt, dx, dy = zip(*stock["jitter"], strict=True)
    assert max(map(abs, tilt)) <= 2.0 + 1e-6 and max(map(abs, dx)) <= 0.03 + 1e-6 and max(map(abs, dy)) <= 0.03 + 1e-6
    assert max(tilt) > 1.8 and min(tilt) < -1.8 and max(dy) > 0.027 and min(dy) < -0.027
    assert max(dx) > 0.005 and min(dx) < -0.005


def test_stock_is_deterministic_and_each_rack_has_its_own(stock):
    """Contract (rule 13: one dressing per map; plan-realism section 4: jitter "seeded by map and bay"): building the
    racks again gives the same parts to the last digit, and the racks that stand the same way (F13, F15 and F17 face +x;
    their bay-levels are the same) do not hold the same stock, so a row of racks does not repeat itself. Regression
    caught: a seed from the clock or the process (every render different), or a seed that ignores the rack (six copies
    of one stock)."""
    assert stock["racks"] == stock["again"]
    west = [json.dumps(stock["racks"][r]) for r in ("F13", "F15", "F17")]
    assert len(set(west)) == 3


def _size(part) -> tuple[float, float]:
    """A turned box part's own (across, along) in metres, from its extents and its turn."""
    _, x0, x1, y0, y1, _, _, yaw = part
    c, s = math.cos(yaw), math.sin(yaw)
    ex, ey = x1 - x0, y1 - y0
    return (c * ex - s * ey) / (c * c - s * s), (c * ey - s * ex) / (c * c - s * s)


def test_carton_sizes_are_mixed_and_a_carton_face_carries_a_small_label(stock):
    """Contract (plan-realism section 4: "mixed carton sizes ... printed black marks and a white shipping label", and
    nothing wrapped: clear film reads white, a tracked pallet colour): the cartons (beige parts a hand's width or more
    across, not the 4 cm corner boards) come in at least 4 different footprints measured to the centimetre (the layouts
    are 2 x 3, 2 x 2, 2 x 4, 3 x 4, 1 x 2 and a mixed one); every white part is a label no bigger than 14.5 x 9.5 cm and
    7 mm thick (3.5 mm and a pallet's turn), there are labels on at least a third of the cartons, and the only other
    colours are black marks, dark straps and the pallet's oak. Regression caught: one carton size everywhere, a sheet of
    white (wrap film) over a stack, a label the size of a carton face, or stock that is no longer beige."""
    parts = [p for rack in stock["racks"].values() for p in rack]
    cartons = [p for p in parts if p[0].startswith("beige") and p[2] - p[1] > 0.1 and p[4] - p[3] > 0.1]
    labels = [p for p in parts if p[0] == "white"]
    assert len({tuple(round(v, 2) for v in _size(p)) for p in cartons}) >= 4
    assert len(labels) >= len(cartons) / 3
    for _, x0, x1, y0, y1, z0, z1, _ in labels:
        assert x1 - x0 <= 0.007 and y1 - y0 <= 0.145 and z1 - z0 <= 0.095
    assert {p[0].split("*")[0] for p in parts} == {"beige", "oak", "white", "black", "dark"}
