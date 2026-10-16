"""The home's floors in the packaged manifest (M1's R.2): 2K sets, with relief
where a bump reads flat. Free: the manifest is data and `render/floor_look.py` is read as text (it imports `bpy`)."""

from __future__ import annotations

import ast
from pathlib import Path

from stbench import asset_manifest as A

RENDER = Path(__file__).resolve().parents[1] / "render"
HOME_ROOMS = ("living room", "bedroom", "kitchen", "hall", "bathroom")


def _constant(name: str):
    """A module-level literal of `render/floor_look.py`."""
    tree = ast.parse((RENDER / "floor_look.py").read_text(encoding="utf-8"))
    (value,) = [n.value for n in tree.body if isinstance(n, ast.Assign) and [t.id for t in n.targets] == [name]]
    return ast.literal_eval(value)


def test_the_home_floors_are_2k_cc0_sets_and_only_the_tiles_and_the_wood_have_relief():
    """Contract (the vacuum sees its floor at 5-27 degrees and the bottom rows resolve 0.5-1.2 mm a pixel, against the
    1.7-2.0 mm of a 1K set's texels; a bump of a grout line reads flat at that angle): each room's floor of the home is
    an ambientCG material of CC0 whose file is the 2K-JPG set (6 MB and up: a 1K set is 3-9 MB, a 2K one 6-31);
    the table of floors that take true displacement (`floor_look.RELIEF_M`) names exactly the living room's wood and
    the kitchen's, the hall's and the bathroom's tiles (the carpet has a pile instead), and no relief moves a floor by
    more than 3 mm either side of its surface. Regression caught: a floor back on its 1K set, or a relief deeper than
    the robot can tell from an obstacle."""
    m = A.packaged()
    floors = {room: m.floors[room] for room in HOME_ROOMS}
    assert floors == {"living room": "acg_WoodFloor051", "bedroom": "acg_Carpet016", "kitchen": "acg_Tiles139",
                      "hall": "acg_Tiles139", "bathroom": "acg_Tiles107"}
    for ref in set(floors.values()):
        a = m.asset(ref)
        assert a.source == "ambientcg" and a.licence == "CC0-1.0" and a.role == "material", ref
        (f,) = a.files
        assert f.path.endswith("_2K-JPG.zip") and a.estimate_bytes > 6_000_000, ref
    relief = _constant("RELIEF_M")
    assert set(relief) == {floors["living room"], floors["kitchen"], floors["bathroom"]}
    assert max(relief.values()) / 2 <= _constant("MAX_DISPLACE_M") == 0.003
