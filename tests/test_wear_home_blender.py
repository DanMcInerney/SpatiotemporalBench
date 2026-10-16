"""The household's look of M1's R.2 fix round in Blender (`render/wear.py`, `render/wear_surfaces.py`,
`render/furniture.py`): smooth painted walls in a colour a room, the library's bump cut by surface, raw board and dust
cloth under furniture, plinths that match their doors, grubby tile joints; and every other catalog dressed as it was.
One Blender script (`blender_wear_driver.py`) builds the placeholder apartment and the contracts below read what it
printed. Needs Blender; marked `blender`, skipped without it. Nothing is downloaded or rendered: about 15 seconds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stbench import blender_run
from stbench import episode as ep
from stbench.render_cli import blender_path
from test_render_materials_blender import apartment_timeline

# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
REPO = Path(__file__).resolve().parents[1]
ROOMS = {"hall", "living room", "kitchen", "bedroom", "bathroom"}


@pytest.fixture(scope="module")
def found(tmp_path_factory):
    root = tmp_path_factory.mktemp("wear_home")
    paths = ep.write_script(root / "episode", apartment_timeline())
    out = root / "result.json"
    cmd = [blender_path(), "--background", "--factory-startup", "--python-exit-code", "1", "--python",
           str(REPO / "tests" / "blender_wear_driver.py"), "--", "--render", str(REPO / "render"), "--src",
           str(REPO / "src"), "--private", str(paths.private), "--out", str(out)]
    blender_run.run(cmd, "checking the household's look")
    return json.loads(out.read_text(encoding="utf-8"))


def test_the_librarys_bump_is_cut_by_surface_in_a_home_and_left_alone_elsewhere(found):
    """Contract (`wear.BUMP_SCALE`, `wear.use_catalog`; paint, laminate and doors are smooth to the eye at 5 to 30 cm, a
    pre-screen finding): a bump of strength 0.5 on a material the household dresses becomes 0.03 on a wall or ceiling
    (0.5 * 0.06), 0.04 on a door (0.08), 0.015 on porcelain and ceramic (0.03), 0.125 on wood and laminate (0.25), 0.3
    on upholstery (0.6) and 0.1 on a carpet or a rug (0.2: its texture is a fibre pile, not bubbles); the same
    materials in another catalog's scene (the warehouse's) keep the 0.5, because that look is frozen. Regression
    caught: the stucco on a vanity and the crumpled door of the pre-screen, or a household change that moves the
    warehouse."""
    home, other = found["bumps"]["home"], found["bumps"]["warehouse"]
    assert home == {"wall": 0.03, "ceiling": 0.03, "door": 0.04, "smooth": 0.015, "furniture": 0.125, "fabric": 0.3,
                    "carpet": 0.1, "rug": 0.1}
    assert other == {"wall": 0.5, "ceiling": 0.5, "furniture": 0.5, "carpet": 0.5}


def test_a_homes_walls_are_one_flat_paint_a_room_with_the_marks_where_the_fittings_are(found):
    """Contract (`wear_surfaces.wall`; the pre-screen's 'concrete-like grime all over' became smooth painted plaster):
    the household's wall takes the flat colour a material was tinted to and not the texture's picture (another catalog's
    wall still reads its texture); it is painted in a colour a room, five different ones for the apartment's five rooms,
    none of them the plain white of a material with no paint; it carries a mark at each door's two jambs, two spots each
    (a hand at 1 m, shoes at the foot: at least 20 in all for the apartment's five doors, more where furniture stands in
    a corner against a wall); the doorway's strip is a worn threshold and the doors are painted leaves; and once
    `wear.apply` has run the household's look is off again for whatever is dressed next. Regression caught: a wall with
    the library's streaked picture, every room one colour, a look that stays switched on, or a threshold left as the
    plain underlay."""
    w = found["walls"]
    assert w["name"].endswith("+wall") and w["geometry_nodes"] == 1
    paints = [tuple(p) for p in w["paints"]]
    assert len(paints) == len(ROOMS) == len(set(paints)) and all(max(p) <= 1.0 and min(p) > 0.5 for p in paints)
    assert max(max(p) - min(p) for p in paints) >= 0.2 and (1.0, 1.0, 1.0) not in paints
    assert w["spots"] >= 20
    assert w["underlay"].endswith("+threshold") and w["doors"] == ["door"]
    assert w["after_has_geometry"] is False
    assert found["flat_paint"] == {"home": False, "warehouse": True}


def test_what_is_seen_from_below_a_piece_is_raw_board_or_dust_cloth(found):
    """Contract (`furniture.py`, `wear_surfaces.raw` and `cover`; the pre-screen's 'smeared, stretched texture' under
    the coffee table was the top's wood grain and bump at a grazing angle): of the rays cast up from the floor on a 9 x
    9 grid under each fixed piece with an open underside (the dining table, coffee table, desk, hall console, sofa and
    bed), those that reach its underside and not a leg (81 less the legs' squares, at least 20) meet raw board or the
    black dust cloth at least 70% of the time, and no other finish but a rail's or a leg's. Regression caught: a table
    top's wood grain or fabric on the underside the vacuum drives under."""
    under = found["undersides"]
    assert set(under) == {"dining_table", "coffee_table", "desk", "hall_console", "sofa", "bed"}
    for kind, u in under.items():
        total = sum(u["hits"].values())
        plain = sum(n for name, n in u["hits"].items() if name in ("raw", "cover"))
        assert total >= 20 and plain / total >= 0.7, (kind, u)


def test_a_cabinets_plinth_is_its_doors_colour_darker_and_not_a_black_void(found):
    """Contract (`furniture._cabinet`, the wardrobe; 'black plinths on every cabinet are noisy voids', the pre-screen):
    the plinth of a counter, sink unit, vanity, TV console, nightstand or wardrobe, met by a ray 3 cm up from in front,
    is the piece's own colour at three quarters of its brightness (palette key `<colour>*0.75`), not the black `dark`.
    Regression caught: a plinth that renders as a void under every cabinet."""
    p = found["plinths"]
    assert set(p) == {"counter", "sink_unit", "vanity", "tv_console", "nightstand", "wardrobe"}
    for kind, row in p.items():
        assert row["plinth"] == f"{row['colour']}*0.75", (kind, row)


def test_the_joints_of_a_tile_floor_read_its_height_map_and_dust_gathers_round_what_stands_on_it(found):
    """Contract (`wear._grubby_tile`, `wear_surfaces.halo`): a tile material of a manifest set (the bathroom's) dressed
    for the household reads grime in its joints from its own height map (the bump's height, an image: the joints are
    where it is low), and has a dust band round each footprint the scene gave (three boxes: two absolute values each,
    the distance of the box's metric); with no height map the joints are left and the bands stay. Regression caught: a
    spotless grid of white tiles round the toilet and the tub."""
    t = found["tiles"]
    assert t["joints_read_height"] is True and t["absolutes"] == 6
    assert t["bare"] == {"joints": False, "absolutes": 6}
