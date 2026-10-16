"""The floor-level look of the floors in Blender (M1's R.2, `render/floor_look.py`): true relief on the tile and wood
floors and a pile on the carpet. One Blender script (`blender_floor_driver.py`) builds a floor from a texture set it
makes (a plateau with a 10 cm groove down the middle, a flat colour and roughness: nothing is downloaded) and the test
reads what it printed. Needs Blender and a GPU, as the render tests do; marked `blender`, skipped when Blender is not
installed. About fifteen seconds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stbench import blender_run
from stbench.render_cli import blender_path

REPO = Path(__file__).resolve().parents[1]
# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
RELIEF_M = 0.005  # the relief the driver gives its floor, peak to peak: 2.5 mm either side of the surface


@pytest.fixture(scope="module")
def result(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("floor") / "result.json"
    blender_run.run([blender_path(), "--background", "--factory-startup", "--python-exit-code", "1", "--python",
                     str(REPO / "tests" / "blender_floor_driver.py"), "--", "--render", str(REPO / "render"),
                     "--src", str(REPO / "src"), "--out", str(out)], "rendering a floor with relief")
    return json.loads(out.read_text(encoding="utf-8"))


def test_a_floor_with_relief_is_displaced_by_its_map_and_no_deeper_than_3_mm(result):
    """Contract (`floor_look.displace` and `subdivide`; the vacuum sees its floor at 5-27 degrees, where a bump of a
    grout line reads flat): a floor given 5 mm of relief shows it in its depth, not only in its shading. Looking
    straight down at it from 0.5 m, the row of the depth image through a 10 cm groove runs between the plateau, 2.5
    mm nearer than the surface (0.5 - 0.0025 m), and the groove, 2.5 mm farther, so its range is 4 to 5.5 mm (the
    map's edge is two pixels soft) and no point is more than 3 mm from the surface, the limit; the same floor without
    relief is flat to 0.3 mm. The material says it: bump and displacement both, the largest displacement 2.5 mm, the
    relief recorded, and the object has an adaptive subdivision. Regression caught: a bump where the picture needs a
    groove (flat depth, a floor that reads as paper at 8 cm), a displacement deeper than the robot can tell from an
    obstacle, or a floor that is not subdivided (a displacement of four vertices)."""
    relief, flat = result["relief"], result["flat"]
    assert relief["made"] and relief["method"] == "BOTH" and relief["prop"] == pytest.approx(RELIEF_M)
    assert relief["max_vertex"] == pytest.approx(RELIEF_M / 2) and relief["modifiers"] == [["SUBSURF", True]]
    assert 0.004 <= relief["max"] - relief["min"] <= 0.0055, relief
    assert max(0.5 - relief["min"], relief["max"] - 0.5) <= result["limit"]  # nowhere more than 3 mm from the surface
    assert abs(relief["min"] - (0.5 - RELIEF_M / 2)) < 6e-4 and abs(relief["max"] - (0.5 + RELIEF_M / 2)) < 6e-4
    assert not flat["made"] and flat["modifiers"] == [] and flat["max"] - flat["min"] < 3e-4
    assert result["limit"] == 0.003


def test_a_carpet_has_a_sheen_and_a_bump_over_its_bump_and_a_wood_floor_has_none(result):
    """Contract (`floor_look.pile`, `wear.worn`): the worn carpet's Principled node has sheen (weight 0.8, roughness
    0.45: fibres scatter light sideways, so it rises as the view grazes) and a bump of fibre tips and tufts whose normal
    is chained over the bump the material already had (its displacement map); a wood floor, worn the same way, has no
    sheen. Regression caught: a carpet that shades like a plank (no pile at the angle the vacuum sees), or a pile that
    replaces the material's own bump instead of adding to it."""
    pile = result["pile"]
    assert pile["sheen"] == pytest.approx(0.8) and pile["sheen_roughness"] == pytest.approx(0.45)
    assert pile["bumps"] == 2 and pile["top_is_bump"] and pile["chained"]
    assert pile["wood_sheen"] == 0.0
