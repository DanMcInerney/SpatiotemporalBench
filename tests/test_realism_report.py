"""The arithmetic of `stbench realism report` (`realism_report`, `realism_data`, `realism_rows`): counts, the objects of
a frame, light, exposure, range, windows, the luminance floor, the references and where a probe is, each worked by hand
on inputs made here, so a measure that reports the wrong number is caught without a render. (The noise is
`test_realism_noise`'s and the crop check `test_realism_crops`'s.) Free: arrays and dicts, no Blender."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from stbench import realism_data as D
from stbench import realism_report as R
from stbench import realism_rows as C
from stbench import realism_run


def test_a_scripts_entities_are_counted_by_mobility_and_its_floor_clutter_by_support():
    """Contract (`entity_counts`; the target is 160-250 entities a script and 20-45 pieces of clutter on the floor): a
    count by mobility, the total, and the clutter that lies on the floor with no support. By hand: one fixed, one
    carried and one light piece, five clutter pieces of which three lie on the floor and two on a surface. Regression
    caught: a floor count that includes the clutter on surfaces, or a total that leaves a mobility out."""
    rows = [("fixed", False), ("carried", False), ("light", False), *[("clutter", True)] * 3, *[("clutter", False)] * 2]
    assert R.entity_counts(rows) == {"fixed": 1, "carried": 1, "light": 1, "clutter": 5, "total": 8, "floor_clutter": 3}


def test_the_objects_of_a_frame_are_the_entities_and_dressing_pieces_of_215_px_with_trim_left_out():
    """Contract (`objects_in_frame`; Hypersim counts what an image shows and a baseboard is no object): of a debug
    index frame, an entity (index 1 to 999) and a dressing piece (1000 up) count from `min_px` pixels, structure (0)
    never, and a piece of trim (baseboard, door casing, window sill, backsplash) never. By hand with `min_px` 215:
    entities 5 (300 px) and 7 (215) count and 6 (214) does not; pieces 1001 (a picture frame, 215) counts, 1000 (a
    baseboard, 500) and 1002 (a door casing, 5,000) are trim, 1003 (a radiator, 214) is too small: two entities and
    one piece, three objects. Regression caught: trim counted as objects (a frame of baseboards reads as a furnished
    room), a threshold off by one, or entities and pieces counted in each other's range."""
    px = {0: 100_000, 5: 300, 6: 214, 7: 215, 1000: 500, 1001: 215, 1002: 5000, 1003: 214}
    classes = {1000: "baseboard", 1001: "picture_frame", 1002: "door_casing", 1003: "radiator"}
    assert R.objects_in_frame(px, classes, 215) == {"entities": 2, "pieces": 1, "objects": 3}


def test_light_is_the_median_of_the_cards_in_a_room_and_the_cast_of_the_lamps():
    """Contract (`light_table`; the target is 100-300 lx a room under the lamps alone, 300-750 on the worktop, the
    lamps 2,700-3,500 K): lux by room is the median of its cards, the worktop's the median of its cards, the cast the
    median of the lamps-only cards' colour ratios, and the lamps' temperature the range of their record. By hand: the
    hall's three cards 90, 110 and 130 lx read 110; the kitchen's two, 250 and 270, read 260; the worktop's 400, 500
    and 600 read 500; red over green 0.9, 1.0 and 1.1 read 1.0 and blue over green 0.6, 0.7 and 0.8 read 0.7.
    Regression caught: a mean where a median is meant (one card in a corner moves the room), a room's cards pooled
    with another's, or daylight read as the lamps."""
    def card(room, lux, r_g=1.0, b_g=0.7):
        return {"room": room, "lux": lux, "r_g": r_g, "b_g": b_g}

    cards = {"lamps": [card("hall", 90, 0.9, 0.6), card("hall", 110, 1.0, 0.7), card("hall", 130, 1.1, 0.8),
                       card("kitchen", 250), card("kitchen", 270)],
             "daylight": [card("hall", 4000), card("hall", 6000)],
             "worktop": [{"lux": lux, "r_g": 1, "b_g": 1} for lux in (400, 500, 600)]}
    table = R.light_table(cards)
    assert table["lamps"] == {"hall": 110.0, "kitchen": 260.0} and table["daylight"] == {"hall": 5000.0}
    assert table["worktop"] == 500.0 and table["cast"] == {"r_g": pytest.approx(1.0), "b_g": pytest.approx(0.7)}
    assert R.lamp_cct([{"cct_k": 3000.0}, {"cct_k": 3500.0}, {"cct_k": 3000.0}]) == {"min": 3000.0, "max": 3500.0}
    assert R.lamp_cct([]) is None


def test_exposure_clipping_and_crushing_are_a_rooms_medians_and_means():
    """Contract (`by_room`; the target is a mean sRGB luminance of 0.35-0.55 a room): the median EV and median
    luminance of a room's frames and the mean of its clipped and crushed shares. By hand: the kitchen's three frames
    have EV 1, 2 and 3 (median 2), luminance 0.30, 0.40 and 0.50 (median 0.40), clipped 0, 0.03 and 0 (mean 0.01) and
    crushed 0.01, 0.02 and 0.06 (mean 0.03); the bedroom's one is itself. Regression caught: a mean luminance that one
    bright window frame drags, or two rooms pooled."""
    def frame(room, ev, lum, clipped, crushed):
        return {"room": room, "ev": ev, "lum": lum, "clipped": clipped, "crushed": crushed}

    rooms = R.by_room([frame("kitchen", 1.0, 0.3, 0.0, 0.01), frame("kitchen", 3.0, 0.5, 0.0, 0.06),
                       frame("kitchen", 2.0, 0.4, 0.03, 0.02), frame("bedroom", -1.0, 0.2, 0.0, 0.1)])
    kitchen = rooms["kitchen"]
    assert kitchen["frames"] == 3 and kitchen["ev"] == 2.0 and kitchen["lum"] == pytest.approx(0.4)
    assert kitchen["clipped"] == pytest.approx(0.01) and kitchen["crushed"] == pytest.approx(0.03)
    assert rooms["bedroom"] == {"frames": 1, "ev": -1.0, "lum": 0.2, "clipped": 0.0, "crushed": 0.1}


def test_a_card_no_light_reached_counts_in_the_lux_and_has_no_cast():
    """Contract (`light_table`): a grey card with no light on it (0 lux: inside a solid) reads r/g and b/g of 0 over 0,
    NaN, which must not be the cast of the lamps: it is left out of the cast's medians and stays in the lux medians. By
    hand: kitchen cards of 0, 100 and 300 lx read 100 lx, and cards of r/g 1.0, NaN and 1.2 and b/g 0.8, NaN and 0.6
    read r/g 1.1 and b/g 0.7. Regression caught: the report's 'b/g = nan' (one dark card in 225 made the median NaN)."""
    nan = float("nan")
    cards = {"lamps": [{"room": "kitchen", "lux": 0.0, "r_g": nan, "b_g": nan},
                       {"room": "kitchen", "lux": 100.0, "r_g": 1.0, "b_g": 0.8},
                       {"room": "kitchen", "lux": 300.0, "r_g": 1.2, "b_g": 0.6}]}
    table = R.light_table(cards)
    assert table["lamps"] == {"kitchen": 100.0}
    assert table["cast"] == {"r_g": pytest.approx(1.1), "b_g": pytest.approx(0.7)}


def test_a_window_stands_so_many_stops_over_the_room_by_the_box_of_its_sill():
    """Contract (`realism_data.window_boxes`, `realism_report.window_stops`; a window in a daylight frame should be 4.5
    stops or more over the room's median): a window's trim is several pieces of class window_sill round its glass, the
    pieces that touch are one window and its box is theirs together, it is in view when they show 200 pixels and the box
    is 24 across; the window's luminance is the 90th percentile of the linear luma inside its box, over the median of
    the whole frame. By hand: in an 80 x 80 frame a left jamb (ids 1001, 4 x 40), a head (1002, 4 x 40, over the jamb's
    corner) and a right jamb (1005) make the box (10, 10, 50, 50); a lone jamb 10 pixels off (1003, 160 pixels: under
    200 and 4 across) and a baseboard (1004) are no window; the glass (1,024 pixels of the box's 1,600) is 3.2 over a
    room of 0.1, so the window is log2(3.2 / 0.1) = 5 stops over the median. Regression caught: a window measured by the
    brightest pixel of the frame (a lamp or a sun patch is no window), a jamb at the edge counted as a window (its box
    has no glass), or the trim's four pieces measured one by one."""
    index = np.zeros((80, 80), dtype=np.uint16)
    index[10:50, 10:14] = 1001
    index[10:14, 10:50] = 1002
    index[10:50, 46:50] = 1005
    index[10:50, 60:64] = 1003
    index[70:74, 0:40] = 1004
    classes = {1001: "window_sill", 1002: "window_sill", 1005: "window_sill", 1003: "window_sill", 1004: "baseboard"}
    assert D.window_boxes(index, classes) == [(10, 10, 50, 50)]
    luma = np.full((80, 80), 0.1)
    luma[14:46, 14:46] = 3.2
    assert R.window_stops(luma, (10, 10, 50, 50)) == pytest.approx(5.0)


def test_a_frames_range_is_stops_from_its_median_to_its_brightest_and_its_darkest():
    """Contract (`stops`): the stops from the median to the 99.9th percentile (a window, a sun patch) and from the 5th
    percentile to the median (the dark under furniture). By hand: 1,000 pixels, 60 at 0.01, 880 at 0.04 and 60 at 0.64
    have a 5th percentile of 0.01, a median of 0.04 and a 99.9th of 0.64: log2(0.64 / 0.04) = 4 stops bright and
    log2(0.04 / 0.01) = 2 dark. Regression caught: a range from the minimum and maximum (one hot pixel makes it
    20 stops) or a mean in place of the median."""
    lum = np.concatenate([np.full(60, 0.01), np.full(880, 0.04), np.full(60, 0.64)])
    assert R.stops(lum) == {"bright": pytest.approx(4.0), "dark": pytest.approx(2.0)}


def test_the_luminance_floor_is_percentiles_over_big_sightings_of_the_jobs_kinds_and_a_share_under_the_floor():
    """Contract (`luminance_floor`; the target is at most 5% of a kind's sightings under `RECOG_LUM`): over the
    sightings of at least 1,024 px of the job's kinds, the 1st, 2nd and 5th percentiles of `lum`, the candidate (the
    midpoint of the 2nd and the 5th) and, per kind, the share under the floor. By hand: 100 shoe sightings of lum
    0.001, 0.002 ... 0.100 (2,000 px) have percentiles 0.00199, 0.00298 and 0.00595 (the line between neighbours at
    positions 0.99, 1.98 and 4.95), a candidate of 0.004465, and 10 under 0.0105: 0.10; 50 bag sightings of 0.1 are
    all above it; five shoe sightings of 500 px (lum 0.0001) and a plant's sightings are not counted. Regression
    caught: small sightings in the floor (a glimpse is dark), another kind's, or a share of all sightings in place of
    a kind's."""
    shoes = [{"kind": "shoes", "px": 2000, "lum": 0.001 * (i + 1)} for i in range(100)]
    bags = [{"kind": "bag", "px": 1024, "lum": 0.1}] * 50
    noise = [{"kind": "shoes", "px": 500, "lum": 0.0001}] * 5 + [{"kind": "plant", "px": 4000, "lum": 0.0001}] * 3
    out = R.luminance_floor(shoes + bags + noise, ["shoes", "bag"], 0.0105)
    assert out["sightings"] == 150
    assert out["per_kind"]["shoes"] == {"sightings": 100, "share_under": 0.10}
    assert out["per_kind"]["bag"]["share_under"] == 0
    alone = R.luminance_floor(shoes, ["shoes"], 0.0105)
    assert alone["percentiles"] == {"p1": pytest.approx(0.00199), "p2": pytest.approx(0.00298),
                                    "p5": pytest.approx(0.00595)}
    assert alone["candidate"] == pytest.approx(0.004465)
    assert R.luminance_floor([], ["shoes"], 0.01)["sightings"] == 0


def test_a_rows_status_is_ok_outside_reported_or_no_data_against_its_target():
    """Contract (`row`, `within`, `words`, `render_md`): a measure with a target is ok inside it and outside beyond it
    (the bounds count), one with none is reported, one with no value is no data; the words of a target say its bounds;
    the markdown has a line to a measure. Regression caught: a bound that excludes its own value, or a missing measure
    reported as a pass."""
    assert R.row("a", 20, (20, None)).status == "ok" and R.row("a", 19.9, (20, None)).status == "outside"
    assert R.row("b", 0.05, (None, 0.05)).status == "ok" and R.row("b", 0.06, (None, 0.05)).status == "outside"
    assert R.row("c", {"x": 1}, None).status == "reported" and R.row("d", None, (1, 2)).status == "no data"
    assert R.words((100, 300)) == "100 to 300" and R.words((20, None)) == "at least 20"
    assert R.words((None, 0.05)) == "at most 0.05"
    md = R.render_md([R.row("a", 20, (20, None)), R.row("c", 0.12345, None)])
    assert "| a | 20 | at least 20 | ok |" in md and "| c | 0.123 | none | reported |" in md


def _script(sid, total, floor, standing, windows=(), daylight=True):
    """A script's numbers as `realism_cmd._script` makes them, by hand: a kitchen frame at 0.30 mean luminance and a
    bedroom frame at 0.40, lamps-only cards of 400 lx in the kitchen and the hall, the shoes' sightings."""
    frames = [{"room": "kitchen", "ev": 1.0, "lum": 0.30, "clipped": 0.0, "crushed": 0.0},
              {"room": "bedroom", "ev": 1.0, "lum": 0.40, "clipped": 0.0, "crushed": 0.0}]
    lamps = [{"room": room, "lux": 400.0, "r_g": 1.1, "b_g": 0.8, "pos": [0, 0, 0]} for room in ("kitchen", "hall")]
    return {"id": sid, "counts": {"total": total, "floor_clutter": floor}, "free_m2": 40.0, "frames": frames,
            "sightings": [{"kind": "shoes", "px": 2000, "lum": 0.001 * (i + 1)} for i in range(100)],
            "lamps": [{"cct_k": 3000.0}], "patches": [{"mean": 0.5, "sigma": 3.0, "chroma": 0.1, "lag1": 0.0}],
            "relevance": ["shoes"], "daylight": daylight, "windows": list(windows),
            "standing_objects": [{"entities": 5, "pieces": standing - 5, "objects": standing}] * 3,
            "script_objects": [{"entities": 2, "pieces": 2, "objects": 4}], "cards": {"lamps": lamps}}


def _refs():
    """The references' numbers (`realism_data.reference_stats`) as the rows read them, made by hand."""
    return {"basis": "pictures of the robot's own camera", "pictures": 14, "fills": 0.05, "clipped": 0.02,
            "clipped_with_fills": 0.07, "crushed": 0.01, "exposure": {"median": 0.30, "min": 0.22, "max": 0.36},
            "patches": [{"mean": 0.5, "sigma": 1.0, "chroma": 0.5, "lag1": 0.5}]}


def test_the_report_is_built_from_the_scripts_numbers_against_the_targets():
    """Contract (`realism_rows.build_rows`; the vacuum's exposure band is the j7's 0.22-0.36, the kitchen's lamps-alone
    band 150-500 lx and the others' 100-300): two scripts of 158 and 190 entities miss and meet the 160-250 target (the
    spread's minimum is outside), a median standing frame of 25 objects is reported with no target (DECISIONS
    2026-10-06: one view covers a fifth of a room), a kitchen at 0.30 mean luminance meets 0.22-0.36 and a bedroom at
    0.40 misses it (it met the old 0.35-0.55), lamps-alone cards of 400 lx are in the kitchen's band and out of the
    hall's, a floor share under the floor of 0.10 for the shoes misses the 5%, and every measure with no input is "no
    data", not a pass. Regression caught: a report that passes a spread by its median, a kitchen judged by a living
    room's band, a band a stop above the robot, a standing-frame count held to a target again, or a measure reported
    with no input."""
    rows = {r.name: r for r in C.build_rows([_script("a", 158, 30, 25), _script("b", 190, 30, 25)], None, 0.0105, None)}
    assert rows["entities a script"].status == "outside" and rows["entities a script"].value["min"] == 158
    assert rows["floor clutter a script"].status == "ok"
    assert rows["floor clutter a m2 of free floor"].value["median"] == 0.75
    standing = next(r for n, r in rows.items() if n.startswith("objects in the median standing frame"))
    assert standing.value == 25 and standing.status == "reported" and standing.target == "none"
    assert rows["exposure, kitchen: median mean sRGB luminance (median EV 1.00, 2 frames)"].status == "ok"
    assert rows["exposure, bedroom: median mean sRGB luminance (median EV 1.00, 2 frames)"].status == "outside"
    assert rows["exposure, bedroom: median mean sRGB luminance (median EV 1.00, 2 frames)"].target == "0.22 to 0.36"
    assert rows["lamps only, kitchen, lux"].status == "ok" and rows["lamps only, kitchen, lux"].target == "150 to 500"
    assert rows["lamps only, hall, lux"].status == "outside" and rows["lamps only, hall, lux"].target == "100 to 300"
    assert rows["luminance floor, shoes: share under 0.0105 (200 sightings)"].status == "outside"
    assert rows["luminance floor, shoes: share under 0.0105 (200 sightings)"].value == 0.1
    assert rows["lamps only, the worktop, lux"].status == "no data"
    assert rows["the lamps' colour temperature, K"].status == "ok"
    cast = rows["the lamps' cast (red/green, blue/green)"].value
    assert cast == {"r_g": pytest.approx(1.1), "b_g": pytest.approx(0.8)}
    assert next(r for n, r in rows.items() if n.startswith("a window in a daylight frame")).status == "no data"


def test_windows_noise_references_and_crops_are_rows_beside_their_targets():
    """Contract (`realism_rows.build_rows`): a window in a daylight frame is a row against 4.5 stops on the median of
    the frames that show one, the frames of a script with no sun up left out (script a's 5.0, 4.0 and 6.0 read 5.0,
    script b's 1.0 at night is not read); the noise is three rows, the luma sigma, the colour noise over it and the
    lag-1 correlation, each by luminance band with the references' beside it (the frames' 3.0, 0.1 and 0.0 against the
    robot's 1.0, 0.5 and 0.5 in the 0.35-0.65 band); the references' exposure and fills are rows of their own; and the
    crop check is two sets of rows, the forced choice first (its 0 confused crops are ok, free naming's 2 are outside,
    and a kind the job asks about under 4 of 5 is outside whichever protocol). Regression caught: a night frame counted
    as a daylight one, a noise measure left out of the report, or a crop protocol scored as the other."""
    rows = C.build_rows([_script("a", 170, 30, 25, windows=[5.0, 4.0, 6.0]),
                         _script("b", 170, 30, 25, windows=[1.0], daylight=False)], _refs(), 0.0105,
                        {"forced": {"confused": [], "per_kind": {"shoes": {"crops": 5, "right": 5, "rate": 1.0}}},
                         "free": {"confused": [{"crop": "a.png"}] * 2,
                                  "per_kind": {"shoes": {"crops": 5, "right": 3, "rate": 0.6},
                                               "mug": {"crops": 2, "right": 0, "rate": 0.0}}}})
    by = {r.name: r for r in rows}
    window = next(r for n, r in by.items() if n.startswith("a window in a daylight frame"))
    assert window.value == 5.0 and window.status == "ok" and window.target == "at least 4.5"
    assert "median of 3 frames" in window.name
    for what, frames, refs in (("luma σ in 8-bit levels", 3.0, 1.0), ("chroma noise over luma noise", 0.1, 0.5),
                               ("lag-1 correlation of the residual", 0.0, 0.5)):
        row = next(r for n, r in by.items() if n.startswith(f"noise, flat patches: {what}"))
        assert row.value["frames"]["0.35-0.65"] == frames and row.value["references"]["0.35-0.65"] == refs
    assert by["the references' exposure: mean sRGB luminance of a picture (the robot's own frames, fills left out)"
              ].value == {"median": 0.30, "min": 0.22, "max": 0.36}
    assert next(r for n, r in by.items() if n.startswith("the references' fills")).value["clipped_with_fills"] == 0.07
    names = list(by)
    assert names.index("crop check, forced choice: confused crops") < names.index("crop check, free naming: confused "
                                                                                  "crops")
    assert by["crop check, forced choice: confused crops"].status == "ok"
    assert by["crop check, free naming: confused crops"].status == "outside"
    assert by["crop check, forced choice, shoes: right, kind and colour (5 of 5)"].status == "ok"
    assert by["crop check, free naming, shoes: right, kind and colour (3 of 5)"].status == "outside"
    assert by["crop check, free naming, mug: right, kind and colour (0 of 2)"].status == "reported"


def test_a_pictures_luma_and_shares_are_read_the_way_the_camera_model_logs_them():
    """Contract (`realism_data.srgb_luma`, `shares`): the luma of an sRGB-encoded 8-bit picture is the Rec.709 weights
    on the encoded values (white 1.0, mid grey 128/255), the clipped share is the pixels with a channel at 255 and the
    crushed share those at or under 5 levels of luma, the way `camera_model` logs a frame: a 10 x 10 picture of 20
    white, 30 black and 50 mid-grey pixels has clipped 0.2 and crushed 0.3, and with the 20 white ones as a mask (left
    out) clipped 0 and crushed 30 / 80 = 0.375. Regression caught: references measured by another rule than the frames,
    so that a comparison of their shares means nothing."""
    a = np.full((10, 10, 3), 128, dtype=np.uint8)
    a[:2] = 255
    a[2:5] = 0
    assert D.srgb_luma(a[:1, :1])[0, 0] == pytest.approx(1.0)
    assert D.srgb_luma(a[9:, :1])[0, 0] == pytest.approx(128 / 255)
    assert D.shares(a) == (pytest.approx(0.2), pytest.approx(0.3))
    mask = np.zeros((10, 10), dtype=bool)
    mask[:2] = True
    assert D.shares(a, mask) == (pytest.approx(0.0), pytest.approx(0.375))


def _save(path, a):
    Image.fromarray(a).save(path)


def test_the_references_are_the_robots_own_frames_with_their_flat_fills_left_out(tmp_path):
    """Contract (`realism_data.reference_stats`, `robot_references`; DECISIONS 2026-10-06): the references' clipping,
    exposure and noise come from the pictures whose `sources.json` camera note names a robot's front camera, with the
    flat fill (exactly RGB 140, 138, 140: blown highlights the maker's page flattened) left out of every share. By hand
    on a 480 x 10 robot picture, each 10 x 10 block of it 20 fill pixels, 10 white, 20 black and 50 mid-grey (128): the
    fill is 0.2 of it; of the 80 pixels left clipped is 10 / 80 = 0.125, crushed 20 / 80 = 0.25 and the mean sRGB luma
    (10 + 50 * 128 / 255) / 80 = 0.4387; counting the fill as blown, clipped is 30 / 100 = 0.3. A white photo in the
    folder that is no robot's is not read (with no sources.json every picture is). Exposure and noise come from pictures
    at least 480 wide only (a thumbnail is downscaled): of a 480 x 16 one and a 160 x 16 one, the 30 patches of the
    first. Regression caught: fills read as the cleanest highlights and the quietest noise of the references, a Flickr
    photo among the robot's, or a thumbnail's denoised pixels and brighter tone (the j7's thumbnails read 0.46-0.60, its
    frames 0.23-0.36)."""
    block = np.full((10, 10, 3), 128, dtype=np.uint8)
    block[:2] = (140, 138, 140)
    block[2] = 255
    block[3:5] = 0
    robot = np.tile(block, (1, 48, 1))
    checker = np.tile(np.where(((np.indices((16, 16)).sum(axis=0)) % 2 == 0)[..., None], 132, 124).astype(np.uint8)
                      .repeat(3, axis=2), (1, 30, 1))
    folder = tmp_path / "refs"
    folder.mkdir()
    _save(folder / "robot.png", robot)
    _save(folder / "other.png", np.full((10, 10, 3), 255, dtype=np.uint8))
    _save(folder / "wide.png", checker)
    _save(folder / "thumb.png", checker[:, :160])
    note = "Real frame from the Roomba j7 front camera"
    (folder / "sources.json").write_text(json.dumps([
        {"file": "robot.png", "camera_note": note}, {"file": "wide.png", "camera_note": note},
        {"file": "thumb.png", "camera_note": note}, {"file": "other.png", "camera_note": "Cameraphone shot"}]),
        encoding="utf-8")
    stats = D.reference_stats(folder)
    assert stats["files"] == ["robot.png", "thumb.png", "wide.png"] and stats["pictures"] == 3
    assert len(stats["patches"]) == 30 and all(p["sigma"] == pytest.approx(4.0) for p in stats["patches"])
    (folder / "sources.json").write_text(json.dumps([{"file": "robot.png", "camera_note": note}]), encoding="utf-8")
    one = D.reference_stats(folder)
    assert one["files"] == ["robot.png"] and one["patches"] == []
    assert one["fills"] == pytest.approx(0.2) and one["clipped"] == pytest.approx(0.125)
    assert one["crushed"] == pytest.approx(0.25) and one["clipped_with_fills"] == pytest.approx(0.3)
    assert one["exposure"]["median"] == pytest.approx((10 + 50 * 128 / 255) / 80)
    (folder / "sources.json").unlink()
    assert D.reference_stats(folder)["pictures"] == 4


def test_a_probe_is_written_where_the_report_and_the_sheet_read_it(tmp_path):
    """Contract (`realism_data.probe_folder`; `realism probe --out`, `report --probe` and `sheet --probe`): the probe of
    a script is `DIR/<id>`, and `DIR` itself where it is already named by the id, so `--out probe` and `--out
    probe/<id>` (the look procedure's) end in the same place, and the report given `--probe probe` finds it either way;
    the probe writes through the very function the readers use. Regression caught: a probe written to `probe/<id>/<id>`
    that no report reads (the first look-dev had to move its folders by hand)."""
    root = tmp_path / "probe"
    assert D.probe_folder(root, "ab12") == root / "ab12"
    assert D.probe_folder(root / "ab12", "ab12") == root / "ab12"
    assert realism_run.probe_folder is D.probe_folder
