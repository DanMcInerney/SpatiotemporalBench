"""The detailed procedural kinds in Blender: the pallet jack, roll cage, forklift and tote robot, the pallet with its
film, the totes, drums, sacks and bins, the household's glass, bottle, book, remote, laptop, plant, laundry basket and
robots, the clutter that no download covers (`render/procedural_clutter.py`: the manifest's procedural `clutter`
models) and the kinds the look round rebuilt (`procedural_carried.py`, `procedural_toys.py`: a tote, a holdall, a
storage box, a carton, a pair of shoes, a stool, a bear and some blocks). Every builder is run with the real code and
one JSON is read back (`blender_procedural_driver.py`): extents, polygons, a digest, the pixel share of each slot over
eight flat-lit views. Needs Blender (the renders are Cycles on the CPU at 128 x 96 pixels, one sample); marked
`blender`, skipped when it is not installed. Nothing is downloaded. About half a minute for the module (two Blender
runs)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stbench import asset_manifest as A
from stbench import blender_run
from stbench.catalog import CATALOGS
from stbench.mapspec import bundled_maps, load_map
from stbench.render_cli import blender_path

# One xdist group a module (`--dist loadgroup`): its module fixtures are built once, on one worker.
pytestmark = [pytest.mark.blender, pytest.mark.xdist_group(name=__name__),
              pytest.mark.skipif(not Path(blender_path()).exists(), reason="Blender is not installed")]
REPO = Path(__file__).resolve().parents[1]
CLUTTER = {a.builder: a for a in A.packaged().assets if a.role == "clutter" and a.source == A.PROCEDURAL}
# The household's askable kinds rebuilt by the look round: no earlier design to hold their colour's share to, so they
# are held to the clutter's floor, over every slot that takes the colour.
REBUILT = ("tote_bag", "holdall_bag", "storage_box", "carton", "shoes_pair", "round_stool", "teddy_bear", "toy_blocks")
REBUILT_ASSETS = {a.builder: a for a in A.packaged().assets if a.role == "model" and a.builder in REBUILT}
KINDS = ("pallet", "tote", "drum", "sack", "bin", "pallet_jack", "roll_cage", "forklift", "tote_robot", "glass",
         "bottle", "book", "remote", "laptop", "plant", "laundry_basket", "robot_vacuum", "home_robot", "ball_robot",
         *CLUTTER, *REBUILT)
# The share of the object's pixels (mean over the eight views, and the least of them) that the slot carrying the
# catalog colour covered in the design these builders replaced: recorded from it, so that detail added since is held to
# what it showed. The slot is the one the manifest tints (the plant's colour is its leaves: its pot is untinted
# porcelain).
BEFORE = {"pallet": (0.905, 0.898), "tote": (0.963, 0.956), "drum": (0.859, 0.841), "sack": (1.0, 1.0),
          "bin": (1.0, 1.0), "pallet_jack": (0.966, 0.940), "roll_cage": (0.983, 0.975), "forklift": (0.587, 0.466),
          "tote_robot": (0.952, 0.921), "glass": (1.0, 1.0), "bottle": (0.623, 0.601), "book": (0.791, 0.600),
          "remote": (0.847, 0.826), "laptop": (0.605, 0.288), "plant": (0.279, 0.250), "laundry_basket": (0.872, 0.840),
          "robot_vacuum": (0.733, 0.686), "home_robot": (0.896, 0.856), "ball_robot": (0.928, 0.897)}
# How far below that a slot may fall, where real detail is dark and the old shape had none (a pallet jack's steering
# wheels and handle head, a robot's bumper): the colour stays dominant, which is the point. Every other kind is held to
# a hundredth.
DROP = {"pallet_jack": 0.07, "tote_robot": 0.06}
DEFAULT_DROP = 0.01


def sizes() -> dict[str, tuple[float, float, float]]:
    """The catalog's size of each kind: carried and mover kinds of every catalog, light furniture of the maps."""
    out = {}
    for catalog in CATALOGS.values():
        for kind in (*catalog.kinds.values(), *catalog.mover_kinds.values()):
            out[kind.name] = tuple(kind.size)
    for name in bundled_maps():
        for f in load_map(name).furniture:
            if f.mobility == "light":
                assert out.setdefault(f.kind, tuple(f.size)) == tuple(f.size), f.kind
    out.update({builder: tuple(a.native_bbox_m) for builder, a in {**CLUTTER, **REBUILT_ASSETS}.items()})  # model boxes
    return out


def colour_slots(kind: str) -> list[str]:
    """The slots of a builder that take the colour attribute."""
    if kind in REBUILT:
        return list(REBUILT_ASSETS[kind].tint["materials"])
    if kind in CLUTTER:
        return list(CLUTTER[kind].tint["materials"])
    asset = A.packaged().model_for(kind)
    wanted = (asset.tint or {}).get("materials")
    return [wanted[0]] if wanted else ["leaves"]  # the plant: no tint, the leaves are the green


def share_of(r: dict, kind: str) -> dict:
    """The mean and the least (over the views) of the pixel share of every colour slot of a builder together."""
    slots = [r["shares"][slot] for slot in colour_slots(kind)]
    return {"mean": sum(x["mean"] for x in slots), "least": sum(x["least"] for x in slots)}


def run_driver(out: Path) -> dict:
    cmd = [blender_path(), "--background", "--factory-startup", "--python-exit-code", "1", "--python",
           str(REPO / "tests" / "blender_procedural_driver.py"), "--", "--render", str(REPO / "render"), "--kinds",
           ",".join(KINDS), "--out", str(out)]
    blender_run.run(cmd, "building the procedural kinds")
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Two runs of the driver, each in its own Blender process."""
    root = tmp_path_factory.mktemp("procedural")
    return run_driver(root / "a.json"), run_driver(root / "b.json")


def test_every_builder_is_built_to_its_catalog_box(built):
    """Contract (V0 section 5, plan-t3 section 5: the timeline's size is what the questions and the visibility log are
    computed from): the mesh of each kind is exactly its catalog size (a pallet's 1.2 x 1.0 x 1.0, a pallet jack's and
    a cage's from the map), standing on z = 0 and centred on x and y, and the geometry the builder makes is that size to
    within a hundredth before the fit, so that the fit never quietly squashes a detail (a laundry basket's cloth that
    stood 37 mm above the box, a plant's leaves 6 mm out of it, a remote's keys 2.5 mm too high). Regression caught: a
    detail added past the box (a wider rim, a longer handle, a taller crown), which the exact fit hides by squashing the
    whole model, or a kind built to a size the catalog no longer has."""
    want = sizes()
    assert set(KINDS) <= set(want)
    for kind, r in built[0].items():
        size = want[kind]
        assert r["built"] == pytest.approx(size, abs=1e-4), kind
        assert r["built_min"] == pytest.approx((-size[0] / 2, -size[1] / 2, 0.0), abs=1e-4), kind
        assert r["design"] == pytest.approx(size, abs=1e-9), kind
        for have, designed in zip(r["natural"], r["design"], strict=True):
            assert abs(have - designed) <= max(0.001, 0.01 * designed), (kind, r["natural"], r["design"])


def test_the_colour_slot_keeps_its_share_of_what_is_seen(built):
    """Contract (VISION rule 4: the colour attribute is what the pixels show; a model is asked about an object by kind
    and colour): the slot that carries the catalog colour (a pallet's wrap, a jack's body, a cage's frame, a bottle's
    glass, a plant's leaves) covers as much of the object's pixels as it did before the detail was added, to within the
    recorded allowance, in the mean over eight views and in the worst of them, and covers most of what is seen of every
    kind but the forklift (whose dark mast, tyres and seat are a third of it). Regression caught: a pallet whose film
    shrank to a band under a visible load, a jack or robot that went mostly black, a tote whose body is all rim and
    handle: the colour present in the model but not in the picture."""
    for kind, r in built[0].items():
        if kind in CLUTTER or kind in REBUILT:  # new kinds have no earlier design: their colour must show, hand floors
            assert share_of(r, kind)["mean"] >= 0.2, (kind, share_of(r, kind))
            assert share_of(r, kind)["least"] >= 0.1, (kind, share_of(r, kind))
            continue
        mean_before, least_before = BEFORE[kind]
        share = share_of(r, kind)
        drop = DROP.get(kind, DEFAULT_DROP)
        assert share["mean"] >= mean_before - drop, (kind, share, mean_before)
        assert share["least"] >= least_before - drop - 0.02, (kind, share, least_before)
        if kind not in ("forklift", "plant", "laptop", "bottle", "book"):
            assert share["mean"] >= 0.7, (kind, share)


def test_a_pallet_jack_is_not_a_box_and_a_bin_is_not_a_tote(built):
    """Contract (no new look-alike: the questions name the kind): the pallet jack stays an open frame, two forks and a
    tiller (it fills a fifth of its silhouette's rectangle, where a block fills most of it), and the picking bin keeps
    its low front lip (its front fifth stands under 60% as high as its back, where a tote's walls are all one height).
    Regression caught: a pump housing grown into a block that makes the jack read as a box, or a bin given a full-height
    front, which reads as a small tote."""
    assert built[0]["pallet_jack"]["fill"] < 0.35
    assert built[0]["bin"]["front_height"] < 0.6 * built[0]["bin"]["back_height"]
    assert built[0]["tote"]["front_height"] > 0.9 * built[0]["tote"]["back_height"]


def test_a_build_is_the_same_mesh_every_time(built):
    """Contract (the scene's determinism rule: the same world in, the same scene out): building a kind twice in a
    process, and in two processes, gives the same polygons (their corners to a micrometre and their slots, whatever
    order they were made in). Regression caught: a wobble that drew on `random` or the clock, a set iterated in
    memory-address order that decides a shape, or a build that depends on what was built before it."""
    for kind, r in built[0].items():
        assert r["digest"] == r["digest_again"], kind
        assert r["digest"] == built[1][kind]["digest"], kind
    assert len({r["digest"] for r in built[0].values()}) == len(KINDS)


def test_no_closed_piece_of_a_mesh_faces_inwards(built):
    """Contract (the library's meshes are shaded and lit from their normals): every closed piece of a procedural mesh
    (a board, a tyre, a film, a sack) has its faces pointing out of it, so smooth shading and the glossy film's
    highlights are on the outside. Regression caught: the normal recalculation choosing wrongly for a thin, flat piece
    (the old sack, whose two skins meet along its rim, was inside out), a loft wound the wrong way round, or a builder
    that adds a mirrored copy."""
    for kind, r in built[0].items():
        assert r["inside_out"] == 0, kind
