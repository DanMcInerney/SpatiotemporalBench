"""The downtown's windows in Blender (`render/facade_shaders.py`: windows with
interiors, so they stop being black): one building of each facade style is rendered straight on in daylight, sunlit and
in shade (`tests/blender_facade_driver.py`: an orthographic camera, 32 samples, the episodes' colour pipeline), and the
display luminance of the middle of every window and of every storefront bay is measured. Needs Blender and a GPU; marked
`blender`, skipped when Blender is not installed. About half a minute (twelve renders)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from stbench import blender_run
from stbench.render_cli import blender_path

REPO = Path(__file__).resolve().parents[1]
# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
STYLES = ("brick_red", "brick_tan", "stone_grey", "glass_blue", "concrete_white", "stucco_cream")
# The darkest a window or shop bay may be, as the mean display luminance (0 to 1) of its middle, with sun on the facade
# and in shade. The black-void windows the city had measured 0.057 to 0.066 at the darkest on every punched wall (and
# 0.066 in a sunlit shop bay): these are about twice that and well under what the new interiors measure (0.14 and 0.29).
FLOOR = {"sun": 0.20, "shade": 0.11}
VARIED = 0.03  # the least standard deviation of the windows' luminance in a building: the old one's was 0.022 in shade


@pytest.fixture(scope="module")
def measured(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("facades") / "facades.json"
    blender_run.run([blender_path(), "--background", "--factory-startup", "--python-exit-code", "1", "--python",
                     str(REPO / "tests" / "blender_facade_driver.py"), "--", "--render", str(REPO / "render"), "--src",
                     str(REPO / "src"), "--out", str(out)], "rendering the facades")
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.mark.parametrize("light", ["sun", "shade"])
@pytest.mark.parametrize("style", STYLES)
def test_no_window_or_shop_is_a_black_void_and_the_windows_differ_from_one_another(measured, style, light):
    """Contract (the city's look: a window shows blinds, curtains, a room or the sky in its glass, never black; and a
    street's windows are not all alike, seeded per window): in a building of each facade style, rendered straight on in
    daylight, every window of the three storeys (a punched wall's 18 and a curtain wall's 36 panes) and every storefront
    bay has a middle at least as bright as `FLOOR` (0.20 with sun on the facade, 0.11 in shade), the luminance of the
    windows varies (standard deviation at least 0.03) and on a punched wall the windows are darker than the wall between
    them, as real glass is (their mean at most 0.9 of it). Regression caught: the old shader's windows (a dark tint and
    a rare pale curtain: 0.057 at the darkest, a deviation of 0.029 in shade), a pane that is all one interior, or a
    shop front painted black."""
    row = measured[f"{style}/{light}"]
    lums = [w["lum"] for w in row["windows"]]
    assert len(lums) in (18, 36) and len(row["shops"]) == 6
    assert min(lums) >= FLOOR[light], (style, light, min(lums))
    assert min(s["lum"] for s in row["shops"]) >= FLOOR[light], (style, light, row["shops"])
    assert statistics.pstdev(lums) >= VARIED, (style, light, statistics.pstdev(lums))
    if row["wall"] is not None:
        assert statistics.mean(lums) <= 0.9 * row["wall"], (style, light, statistics.mean(lums), row["wall"])


def test_a_deep_building_has_a_real_roof_a_parapet_ring_round_a_tar_surface_with_equipment_on_it(measured):
    """Contract (plan-realism section 4, "City": rooftops with air-conditioning units, vents and parapets, which the
    drone at 12 m never sees under the 18.9 m of the lowest roof but a higher camera would): an 18 x 12 m building (216
    m2) has a tar roof facing up at its height over most of its footprint (at least 60%: the parapet's 0.3 m ring and
    the equipment take the rest), a parapet whose top, at 0.9 m above the roof, is only the 0.3 m ring (at most 20 m2 of
    the 216, not a solid block over the whole roof as a face's is), and equipment (condensers, vents, a hatch) that
    stands on the roof no higher than 1.5 m above it. Regression caught: the old solid parapet block over a roof (a
    white slab, no tar to see), a roof with nothing on it, or equipment through the parapet."""
    for key in ("brick_red/sun", "glass_blue/shade"):
        roof = measured[key]["roof"]
        assert roof["tar_area"] >= 0.6 * 216 and 0.0 < roof["parapet_top_area"] <= 20.0, (key, roof)
        assert roof["unit_faces"] >= 12 and 0.4 <= roof["unit_top"] <= 1.5, (key, roof)
