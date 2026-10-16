"""What the warehouse's dressing builders draw, in Blender (`render/dressing_build_warehouse.py` and the modules under
it). Every Dressing of `warehouse_a`'s layout is built by
`tests/blender_dressing_warehouse_driver.py` (no render and no GPU, no asset library, so a Poly Haven piece is its
stand-in) and what comes out is checked against numbers worked out here by hand. Needs Blender; marked `blender`,
skipped when it is not installed. About fifteen seconds. How the dispatcher places, names and collects the pieces is
`test_dressing_build_blender.py`'s; which pieces the layout emits and where is `test_dressing_layout_warehouse.py`'s;
that no piece meets a forbidden volume is `test_dressing.py`'s."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dressing_kit import layout_of, raw_map
from stbench import blender_run
from stbench.render_cli import blender_path

REPO = Path(__file__).resolve().parents[1]
# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
BOX_EPS_M = 0.005  # the render's tolerance: built geometry stays in its Dressing's box by this much more at most
YELLOW_CLASSES = {"rack_upright_protector", "end_of_aisle_guard", "safety_band"}


@pytest.fixture(scope="module")
def layout() -> list:
    return layout_of("warehouse", raw_map("warehouse_a"))


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    """What the builders made of the layout, built twice: per Dressing a row (see the driver), the digest of each build
    and the objects a builder left linked in the scene."""
    out = tmp_path_factory.mktemp("warehouse") / "built.json"
    blender_run.run([blender_path(), "--background", "--factory-startup", "--python-exit-code", "1", "--python",
                     str(REPO / "tests" / "blender_dressing_warehouse_driver.py"), "--", "--render",
                     str(REPO / "render"), "--src", str(REPO / "src"), "--out", str(out)],
                    "building the warehouse's dressing")
    return json.loads(out.read_text(encoding="utf-8"))


def text_of(d) -> str:
    return d.recipe.split(":", 2)[2]


def test_every_dressing_is_one_loose_object_inside_its_box_at_pass_index_0(built, layout):
    """Contract (the validity guard: dressing is never an entity, and `dressing_zones` holds each Dressing's box clear
    of what a question or the robot depends on, so what is drawn must stay in it): each of the layout's Dressings is
    built as exactly one object, with `pass_index` 0 (instance ids start at 1, so it is in no `instance_map`), an
    identity transform (the dispatcher places it) and no collection (the dispatcher links it); put at the Dressing's
    place and yaw its mesh lies in the Dressing's box (`Dressing.box`, the axis-aligned box after the yaw) by 5 mm at
    most, which checks the frame the builders draw in (x out of the face, y across, z up) against the layout's yaw; and
    it fills at least half the box along each side, so no builder draws a token. Regression caught: a builder that draws
    in the map frame or turned the wrong way (a wall cladding panel across the room), geometry that spills past its box
    (a guard grown into the aisle, a label into a bay's opening), a piece with the entity pass index, or one that links
    itself."""
    assert [(r["cls"], r["recipe"]) for r in built["rows"]] == [(d.cls, d.recipe) for d in layout]
    assert built["linked"] == []
    for d, r in zip(layout, built["rows"], strict=True):
        assert r["objects"] == 1 and r["pass_index"] == 0 and r["identity"], d.recipe
        lo, hi = d.box[:3], d.box[3:]
        assert all(r["world"][i] >= lo[i] - BOX_EPS_M - 1e-6 and r["world"][i + 3] <= hi[i] + BOX_EPS_M + 1e-6
                   for i in range(3)), (d.recipe, r["world"], d.box)
        extent = [r["local"][i + 3] - r["local"][i] for i in range(3)]
        assert all(extent[i] >= 0.5 * d.size[i] for i in range(3)), (d.recipe, extent, d.size)


def test_labels_signs_and_door_numbers_are_dark_print_on_light_plates_at_a_legible_size(built, layout):
    """Contract (readable aisle signs and beam labels; the text is the recipe's,
    drawn as geometry): the 48 beam labels, 3 aisle signs and 4 door numbers each keep their string as a custom property
    of the object and of its mesh, and carry it as print, one connected piece of mesh a letter or digit (a dash counts),
    in a dark material (`rubber_black`) lying 0.5 to 2 mm in front of a white plate (`label_paper`) and inside it. The
    print is as large as the plate takes, so that it reads at a few metres in a 640 x 480 frame (a pinhole of 80 degrees
    sees a 0.06 m letter at 3 m as 8 pixels): a beam label's letters are at least 0.055 m tall and its code at least
    0.25 m wide; an aisle sign's biggest letter at least 0.15 m (its two lines each fill the board's width); a door
    number's digit at least 0.25 m. Regression caught: a label whose text was dropped (a plate with no code), print that
    is a different string, text behind the plate or off it, and print too small to read."""
    seen = {"beam_label": 0, "aisle_sign": 0, "door_number": 0}
    for d, r in zip(layout, built["rows"], strict=True):
        if d.cls not in seen:
            continue
        seen[d.cls] += 1
        text = text_of(d)
        assert r["text_property"] == text and r["mesh_text_property"] == text, d.recipe
        assert {"rubber_black", "label_paper"} <= set(r["materials"]), d.recipe
        assert r["ink_islands"] == len([c for c in text if c not in " |"]), (d.recipe, r["ink_islands"])
        plate = r["plate"]
        for x0, y0, z0, x1, y1, z1 in r["ink"]:
            assert plate[3] + 0.0005 - 1e-6 <= x0 <= x1 <= plate[3] + 0.002 + 1e-6, d.recipe
            assert plate[1] <= y0 and y1 <= plate[4] and plate[2] <= z0 and z1 <= plate[5], d.recipe
        tallest = max(e[5] - e[2] for e in r["ink"])
        width = max(e[4] for e in r["ink"]) - min(e[1] for e in r["ink"])
        if d.cls == "beam_label":
            assert tallest >= 0.055 and width >= 0.25, (d.recipe, tallest, width)
        elif d.cls == "aisle_sign":
            assert tallest >= 0.15 and width >= 0.7, (d.recipe, tallest, width)
        else:
            assert tallest >= 0.25, (d.recipe, tallest)
    assert seen == {"beam_label": 48, "aisle_sign": 3, "door_number": 4}


def test_a_beam_label_has_a_code39_barcode_of_its_code_the_right_way_round(built, layout):
    """Contract (plan-realism section 4: "barcode labels on the beams"): under its code a beam label carries thin bars,
    one box each, five to a character of the code between the two stars of Code 39, so a code of 7 characters has 45,
    with the bars inside the plate; read from the front (a viewer in front of the label looks at it along -x, so his
    right is +y), the symbol starts with the star's two narrow bars then a wide one: the first two bars are equally wide
    and the third is three times as wide as the first. Regression caught: a label with print and no bars, bars of
    another code, a barcode that is mirrored (it starts with the star's end), or bars that run off the plate."""
    labels = [(d, r) for d, r in zip(layout, built["rows"], strict=True) if d.cls == "beam_label"]
    assert len(labels) == 48
    for d, r in labels:
        bars, plate = r["bars"], r["plate"]
        assert len(bars) == 5 * (len(text_of(d)) + 2), d.recipe
        assert all(plate[1] <= y0 and y1 <= plate[4] for y0, y1 in bars) and bars == sorted(bars)
        w = [y1 - y0 for y0, y1 in bars]
        assert w[0] == pytest.approx(w[1], rel=0.05) and w[2] == pytest.approx(3 * w[0], rel=0.05), d.recipe


def test_yellow_hardware_is_open_rails_and_only_protectors_guards_and_the_band_are_yellow(built):
    """Contract (the validity rule: yellow is a tracked colour of five warehouse kinds, so a yellow safety guard is open
    rails and brackets, never a solid block; the wall band is hazard paint, half black): of the 348 pieces only the
    upright protectors, the end-of-aisle guards and the safety bands use the safety yellow, and of each the yellow
    covers at most 0.6 of its box seen from the front (along the way it faces) and from above, worked out by rasterising
    its triangles on a 64 x 64 grid (a solid block covers 1.0; by hand a protector's flat bars cover about 0.4, a
    guard's posts and rails about 0.4 from the front and 0.5 from above, a band's stripes half of its face). Regression
    caught: a protector drawn as a yellow block (a tote or bin colour look-alike on the floor of every aisle), yellow on
    a piece that is not safety hardware, a band that is a plain yellow slab."""
    seen = set()
    for r in built["rows"]:
        if "yellow_coverage" not in r:
            continue
        seen.add(r["cls"])
        front, _, top = r["yellow_coverage"]
        assert front <= 0.6 and top <= 0.6, (r["recipe"], r["yellow_coverage"])
    assert seen == YELLOW_CLASSES


def test_a_high_bay_fixture_hangs_above_the_lens_plane_with_its_lens_at_the_bottom(built, layout):
    """Contract (lighting lane, `lighting.lamps`: a disk light 0.36 m across sits 2 mm under the fixture's lens plane at
    the map's light height 5.8 m, so nothing opaque may hang below that plane inside the footprint, and the visible lens
    is a camera-only emitter so it does not count the light twice): each of the nine fixtures lies between z 5.80 and
    the deck's underside at 5.96 and inside 0.4 m across; its lens (`led_emitter_cool`) is a round disc 0.3 m across,
    centred on the light and just above the plane (3 to 10 mm). Regression caught: a fixture whose bezel or lens hangs
    under the light's plane (it shades its own light), a lens off the light's axis, or a fixture hung from the deck
    whose lens is not at 5.8 m."""
    fixtures = [r for d, r in zip(layout, built["rows"], strict=True) if d.cls == "high_bay_light"]
    assert len(fixtures) == 9
    for r in fixtures:
        x0, y0, z0, x1, y1, z1 = r["world"]
        assert z0 >= 5.80 - 1e-6 and z1 <= 5.96 + 1e-6, r["world"]
        assert x1 - x0 <= 0.4 + 1e-6 and y1 - y0 <= 0.4 + 1e-6
        lx0, ly0, lz0, lx1, ly1, lz1 = r["lens"]
        assert lx1 - lx0 == pytest.approx(0.3, abs=0.01) and ly1 - ly0 == pytest.approx(0.3, abs=0.01)
        assert lx0 + lx1 == pytest.approx(0, abs=1e-3) and ly0 + ly1 == pytest.approx(0, abs=1e-3)
        assert lz0 >= 0.003 and lz1 <= 0.010


def test_a_skylights_pane_is_material_slot_0_and_its_frame_slot_1(built, layout):
    """Contract (lighting lane, `lighting.daylight_panels`: it replaces slot 0 of every `dressing_skylight_*` object
    with its daylight material, which is emissive, so the frame must not be in slot 0): each of the six roof lights has
    two slots, the pane (`skylight_panel`) first and the aluminium frame (`brushed_steel`) second, both with faces.
    Regression caught: a frame in slot 0 (the whole roof light, rails included, glows like the daylight)."""
    lights = [r for d, r in zip(layout, built["rows"], strict=True) if d.cls == "skylight"]
    assert len(lights) == 6
    for r in lights:
        assert r["materials"] == ["skylight_panel", "brushed_steel"] and all(n > 0 for n in r["slot_polys"])


def test_the_builders_make_the_same_pieces_every_time(built):
    """Contract (rule 13, determinism: every episode of a map has the same dressing): building every piece of the
    warehouse twice in one session gives the same meshes, material slots and printed strings, to a hundredth of a
    millimetre. Regression caught: a builder that reads a clock, an unseeded random number or a set's order."""
    first, second = built["digest"]
    assert first == second
