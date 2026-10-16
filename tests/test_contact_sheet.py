"""The sheet of `stbench realism sheet` (`contact_sheet`): the strata by their rules, the draw, the crops and their key,
and the page. Free: synthetic frames and rectangles made here, no Blender."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from stbench import contact_sheet as S
from stbench.geometry import Rect

DESK = Rect(8.65, 3.85, 0.6, 1.2, math.pi)  # the bedroom's desk F33: x 8.35 to 8.95, y 3.25 to 4.45
WALL = Rect(3.0, 0.0, 6.0, 0.1, 0.0)  # a wall piece: y -0.05 to 0.05, x 0 to 6
BASE = (3.0, 3.0)


def test_each_stratum_has_its_rule_and_a_frame_lands_in_the_first_it_fits():
    """Contract (`contact_sheet.stratum`; the sheet's five strata, VISION rule 10's checklist of the vacuum's view): a
    frame in the dock's phases (docked, undocking, docking) is dock whatever else it is; a base inside a piece with an
    open underside is under furniture; a middle block of the depth image whose median is at most 0.3 m is wall-facing
    (0.30 is, 0.31 is not); a cleaning frame whose base is within 0.35 m of a wall or blocking footprint is along walls
    (a wall at y = 0.05: a base at y 0.39 is, y 0.41 is not, in transit it never is); and everything else is open floor.
    By hand on the desk and the wall above. Regression caught: a stratum that never draws (the rule reads a phase that
    does not exist), or a frame counted twice."""
    under, solids = [DESK], [WALL]
    assert S.stratum("docked", (4.94, 5.05), under, solids, None) == "dock"
    assert S.stratum("docking", (8.65, 3.85), under, solids, 0.1) == "dock"  # the dock wins over what is under
    assert S.stratum("cleaning", (8.65, 3.85), under, solids, 0.1) == "under furniture"
    assert S.stratum("cleaning", BASE, under, solids, 0.30) == "wall-facing"
    assert S.stratum("transit", BASE, under, solids, 0.31) == "open floor"
    assert S.stratum("cleaning", (3.0, 0.39), under, solids, 2.0) == "along walls"
    assert S.stratum("cleaning", (3.0, 0.41), under, solids, 2.0) == "open floor"
    assert S.stratum("transit", (3.0, 0.39), under, solids, 2.0) == "open floor"
    assert S.stratum("cleaning", BASE, under, solids, None) == "open floor"
    assert set(S.STRATA) == {"dock", "under furniture", "wall-facing", "along walls", "open floor"}


def test_the_middle_block_of_a_depth_image_is_its_median_with_beyond_range_left_out():
    """Contract (`centre_median`): the median of the middle third of the rows and of the columns, 0 (beyond range) not
    counted, None when nothing in the block has a depth. By hand on a 9 x 12 frame (the block is rows 3 to 5, columns 4
    to 7): the block holds 2.0 everywhere but two zeros and one 0.2, so its median is 2.0; a block of zeros is None;
    anything outside the block does not count. Regression caught: a wall at 0 mm (out of range) read as touching, or the
    whole frame's median in place of the middle's."""
    d = np.full((9, 12), 5.0)
    d[3:6, 4:8] = 2.0
    d[3, 4] = d[4, 5] = 0.0
    d[5, 7] = 0.2
    assert S.centre_median(d) == pytest.approx(2.0)
    d[3:6, 4:8] = 0.0
    assert S.centre_median(d) is None
    d[3:6, 4:8] = 0.25
    assert S.centre_median(d) == pytest.approx(0.25)


def test_the_draw_takes_six_of_each_stratum_with_a_seed_and_from_every_script():
    """Contract (`draw`; six frames a stratum, seeded, spread across scripts): from a stratum of 10 frames of script A
    and 2 of script B six are drawn and both of B's are among them (the scripts take turns); the same seed draws the
    same frames and another seed draws others; a stratum with 3 frames keeps all 3; an empty one is empty.
    Regression caught: a sheet of six frames of one script, or a draw that changes between runs."""
    frames = [{"stratum": "open floor", "script": "A", "k": k} for k in range(10)]
    frames += [{"stratum": "open floor", "script": "B", "k": k} for k in range(2)]
    frames += [{"stratum": "dock", "script": "A", "k": k} for k in range(3)]
    one, again, other = S.draw(frames, 7), S.draw(frames, 7), S.draw(frames, 8)
    assert one == again and one != other
    open_floor = one["open floor"]
    assert len(open_floor) == 6 and {f["k"] for f in open_floor if f["script"] == "B"} == {0, 1}
    assert len({(f["script"], f["k"]) for f in open_floor}) == 6
    assert len(one["dock"]) == 3 and one["wall-facing"] == [] and set(one) == set(S.STRATA)


def test_a_crop_is_the_objects_box_and_a_quarter_of_it_all_round_inside_the_picture():
    """Contract (`crop_box`; the crop check's crop is the box and about 25% of its size of margin, not a fixed 128 px in
    which the object filled 6-12%): the box grown by a quarter of its width and height on each side (1.5 times it),
    centred on the box, at least 64 pixels across, kept inside a 640 x 480 frame. By hand: a box 100 wide and 80 high at
    (270, 190)-(370, 270) gives 150 x 120 centred on it: (245, 170, 395, 290); a 20 x 20 box in the top left corner
    gives the 64 minimum pushed inside: (0, 0, 64, 64); a 20 x 20 box at the bottom right (620, 460, 640, 480) gives
    (576, 416, 640, 480); a box as wide as the frame (0, 100, 640, 140) gives the frame's 640 wide and 64 high (0, 88,
    640, 152). Regression caught: a crop that runs off the picture (a short or black one), one off centre, or one so
    large that the object is a few per cent of it, which is what made the first crop check name the sofa behind the
    box."""
    assert S.crop_box([270, 190, 370, 270], (480, 640)) == (245, 170, 395, 290)
    assert S.crop_box([10, 10, 30, 30], (480, 640)) == (0, 0, 64, 64)
    assert S.crop_box([620, 460, 640, 480], (480, 640)) == (576, 416, 640, 480)
    assert S.crop_box([0, 100, 640, 140], (480, 640)) == (0, 88, 640, 152)


def test_a_stratum_shows_distinct_views_not_the_same_docked_view_six_times():
    """Contract (`same_view`, `draw`): two frames of one map are one view when their bases are under 0.15 m apart and
    their headings under 0.3 rad (heading taken round the circle: 3.1 and -3.1 are 0.08 apart); a stratum draws only one
    of such frames. By hand: the dock stratum of five scripts that each start and end docked at (4.94, 5.05) facing 0,
    plus one undocking frame at (4.5, 5.05) heading 0 and one at (4.94, 5.05) heading 1.0, holds three views and draws
    three frames (not six); frames without a pose, or of another map, are views of their own. Regression caught: the
    sheet's dock stratum of six copies of one docked view (one per script start and end), which showed a reviewer one
    picture six times."""
    dock = [{"stratum": "dock", "script": f"s{i}", "k": k, "pose": ("apartment_a", 4.94, 5.05, 0.0)}
            for i in range(5) for k in (0, 70)]
    dock += [{"stratum": "dock", "script": "s0", "k": 3, "pose": ("apartment_a", 4.5, 5.05, 0.0)},
             {"stratum": "dock", "script": "s1", "k": 4, "pose": ("apartment_a", 4.94, 5.05, 1.0)}]
    drawn = S.draw(dock, seed=1)["dock"]
    assert len(drawn) == 3
    views = {(round(f["pose"][1], 2), round(f["pose"][3], 1)) for f in drawn}
    assert views == {(4.94, 0.0), (4.5, 0.0), (4.94, 1.0)}
    assert S.same_view({"pose": ("m", 0, 0, 3.1)}, {"pose": ("m", 0.1, 0, -3.1)})
    assert not S.same_view({"pose": ("m", 0, 0, 0.0)}, {"pose": ("m", 0.2, 0, 0.0)})
    assert not S.same_view({"pose": ("m", 0, 0, 0.0)}, {"pose": ("n", 0, 0, 0.0)})
    assert not S.same_view({"pose": None}, {"pose": None})
    assert len(S.draw([{"stratum": "dock", "script": "a", "k": k} for k in range(4)], seed=1)["dock"]) == 4


def test_crops_are_of_big_sightings_across_scripts_and_the_key_stays_out_of_the_folder(tmp_path):
    """Contract (`pick_crops`, `write_crops`; the look-alike check): of each class at most 5 sightings of 1,024 to
    2,048 px are taken, no frame twice, scripts in turn (a class seen in two scripts has crops from both); sightings
    outside the pixel range are never taken; the crops are PNGs of the object's box and a quarter of it all round (a 40
    x 30 and a 30 x 30 box are cut at the 64 x 64 minimum, a 50 x 40 one at 75 x 64) named `crop_0001.png` onwards in a
    seeded order, whose names do not say what they are, and the key (class, kind, colour, script, frame) is
    `crop_key.json` in the output folder, not in `crops/`, which holds nothing else. Regression caught: reviewers who
    can read the answer off a file name or a neighbouring file, a class cut from one frame five times, or a crop of an
    object too small to name."""
    for sid in ("A", "B"):
        Image.fromarray(np.full((480, 640, 3), 90, dtype=np.uint8)).save(tmp_path / f"{sid}.jpg")
    strip = [{"script": "A", "k": k, "px": 1500, "bbox": [300, 200, 340, 230]} for k in range(4)]
    strip += [{"script": "B", "k": 0, "px": 2000, "bbox": [10, 10, 40, 40]}]
    strip += [{"script": "A", "k": 9, "px": 1023, "bbox": [0, 0, 5, 5]}, {"script": "A", "k": 10, "px": 2049,
                                                                         "bbox": [0, 0, 5, 5]}]
    mug = {"script": "A", "k": 1, "px": 1100, "bbox": [400, 300, 450, 340], "kind": "mug", "colour": "red"}
    cands = {"clutter:power_strip": strip, "askable:mug": [mug] * 3}
    picks = S.pick_crops(cands, seed=3)
    power = [p for p in picks if p["class"] == "clutter:power_strip"]
    assert len(power) == 5 and {p["script"] for p in power} == {"A", "B"}
    assert len({(p["script"], p["k"]) for p in power}) == 5 and all(1024 <= p["px"] <= 2048 for p in power)
    mugs = [p for p in picks if p["class"] == "askable:mug"]
    assert len(mugs) == 1  # three sightings of one frame make one crop
    out = tmp_path / "sheet"
    key = S.write_crops(picks, lambda s, k: tmp_path / f"{s}.jpg", out, seed=3)
    names = sorted(p.name for p in (out / "crops").iterdir())
    assert names == [f"crop_{i:04d}.png" for i in range(1, len(picks) + 1)]
    sizes = {r["class"]: Image.open(out / "crops" / r["crop"]).size for r in key}
    assert sizes == {"clutter:power_strip": (64, 64), "askable:mug": (75, 64)}
    assert [r["crop"] for r in key] == names and {r["class"] for r in key} == {"clutter:power_strip", "askable:mug"}
    assert json.loads((out / "crop_key.json").read_text(encoding="utf-8")) == key
    assert not (out / "crops" / "crop_key.json").exists()
    firsts = {S.write_crops(picks, lambda s, k: tmp_path / f"{s}.jpg", tmp_path / f"seed{n}", seed=n)[0]["class"]
              for n in range(8)}
    assert firsts == {"clutter:power_strip", "askable:mug"}  # the order is the seed's: not by class, not the same


def test_the_page_shows_every_picture_at_native_size_with_its_caption(tmp_path):
    """Contract (`write_sheet`): `index.html` has an image of 640 x 480 for each of the strata's frames, the standing
    frames and the references, each copied to a file of its own beside it, each with a caption naming its stratum,
    script and frame (or its room, or its file). Regression caught: a sheet that scales the frames (the reviewer
    cannot judge noise at half size), a picture left out, or a caption that does not say what the frame is."""
    for n in ("a", "b", "s", "r"):
        Image.fromarray(np.full((480, 640, 3), 50, dtype=np.uint8)).save(tmp_path / f"{n}.jpg")
    strata = {"dock": [{"script": "A", "k": 0}], "open floor": [{"script": "B", "k": 4}]}
    page = S.write_sheet(tmp_path / "out", strata, lambda s, k: tmp_path / ("a.jpg" if s == "A" else "b.jpg"),
                         [(tmp_path / "s.jpg", "standing camera, kitchen, script A, frame 2")], [tmp_path / "r.jpg"])
    html = page.read_text(encoding="utf-8")
    assert html.count('width="640" height="480"') == 4
    captions = ("dock: script A, frame 0", "open floor: script B, frame 4",
                "standing camera, kitchen, script A, frame 2", "reference r.jpg")
    for caption in captions:
        assert caption in html
    assert len(list(Path(tmp_path / "out" / "frames").iterdir())) == 4
