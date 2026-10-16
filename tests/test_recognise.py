"""The recognisability floor (`recognise.RECOG_PX`, `RECOG_LUM`, `RECOG_EDGE_PX`; 0.5.0): a sighting is
recognisable in one frame, from 1,024 px and a mean luminance of 0.01, a bounding box 2 px clear of every edge of the
image and a median depth, because "seen" (269 px) is a 16 x 17 px patch and a sliver at the frame's edge is no thing a
person could name (VISION rule 10). Hand tests: the boundary values are written out, not read from the constants."""

from __future__ import annotations

import pytest

import family_kit as fk
from space_kit import vacuum_run
from stbench.camera import Camera
from stbench.questions_where import LastSeen, answerable_at, pools
from stbench.recognise import recognisable
from stbench.sightings import Sightings

CAMERA = Camera.load({"width": 640, "height": 480, "fx": 400.0, "fy": 400.0, "cx": 320.0, "cy": 240.0})
BAG = (0.374, 0.304, 0.314)


def frame(px: int = 1500, lum: float = 0.25, bbox=(100, 100, 140, 140), depth: float | None = 2.0) -> dict:
    """A visibility record of one frame: `px` pixels of mean luminance `lum` in `bbox` (u0, v0, u1, v1; u1 and v1
    exclusive) at the median depth `depth` (None: no pixel of it has a depth)."""
    return {"id": "E1", "px": px, "bbox": list(bbox), "depth_med_m": depth, "lum": lum}


def test_a_frame_is_recognisable_from_pixels_light_a_clear_box_and_a_depth():
    """Contract (VISION rule 10): `recognisable(row, camera, what)` is true when one frame's record has at
    least 1,024 px (COCO's 32 x 32 px), a mean luminance of at least 0.01 (0.5.0-control: where the pixels carry no
    colour; M1's 0.0058 let a view of a grey bag at 0.0087 count), a bounding box at least 2 px
    from every edge of the 640 x 480 image (`u1` and `v1` exclusive, so u1 <= 638 and v1 <= 478) and a median depth;
    every bound inclusive. Hand values: 1,023 px no and 1,024 yes; lum 0.0099 no and 0.01 yes (0.0087, the grey bag the
    log review found, no); u0 or v0 of 1 no and 2
    yes; u1 of 639 no and 638 yes; v1 of 479 no and 478 yes; no depth (null) no, whatever its pixels and light; no row
    (None) no; a count under the floor is no sighting whatever else the row lacks, but a count over it with no `lum`,
    `bbox` or `depth_med_m` key raises `ValueError` naming the thing, because nothing is silently counted either way.
    Regression caught: a bound moved by one (`>` for `>=`), the floor back at 0.0058, a thing cut by the frame's edge
    (a sliver of itself) or with no depth taken as evidence, or a log row without a field quietly counted as dark,
    clipped or bright."""
    assert recognisable(frame(px=1023), CAMERA, "x") is False and recognisable(frame(px=1024), CAMERA, "x") is True
    assert recognisable(frame(lum=0.0099), CAMERA, "x") is False and recognisable(frame(lum=0.01), CAMERA, "x")
    assert recognisable(frame(lum=0.0087), CAMERA, "x") is False
    assert recognisable(frame(lum=0.0058), CAMERA, "x") is False
    assert recognisable(frame(bbox=(1, 100, 140, 140)), CAMERA, "x") is False
    assert recognisable(frame(bbox=(2, 100, 140, 140)), CAMERA, "x") is True
    assert recognisable(frame(bbox=(100, 1, 140, 140)), CAMERA, "x") is False
    assert recognisable(frame(bbox=(100, 2, 140, 140)), CAMERA, "x") is True
    assert recognisable(frame(bbox=(100, 100, 639, 140)), CAMERA, "x") is False
    assert recognisable(frame(bbox=(100, 100, 638, 140)), CAMERA, "x") is True
    assert recognisable(frame(bbox=(100, 100, 140, 479)), CAMERA, "x") is False
    assert recognisable(frame(bbox=(100, 100, 140, 478)), CAMERA, "x") is True
    assert recognisable(frame(depth=None), CAMERA, "x") is False and recognisable(frame(depth=0.3), CAMERA, "x")
    assert recognisable(None, CAMERA, "x") is False
    assert recognisable({"id": "E1", "px": 1023}, CAMERA, "x") is False
    for missing in ("lum", "bbox", "depth_med_m"):
        row = {k: v for k, v in frame().items() if k != missing}
        with pytest.raises(ValueError, match=f"keyframe 4 for mug.*{missing}"):
            recognisable(row, CAMERA, "keyframe 4 for mug")
    with pytest.raises(ValueError, match="lum"):
        recognisable(frame(lum=None), CAMERA, "keyframe 4 for mug")


def test_look_gives_the_pixels_and_luminance_of_a_counted_sighting():
    """Contract: `Sightings.look(entity, k)` is the `(px, lum)` the log recorded for a sighting the compiler counts
    (seen or in the dead band), None for a keyframe with no sighting, so the Perceive set's recognisability floor reads
    what the compiler counted. a: 1,500 px at k1, 400 px at k3; b: 2,000 px at k3. Regression caught: a look that
    reports a keyframe the compiler does not count, or drops the luminance of one it does."""
    rows = [{"k": 1, "entities": [{"id": "a", "px": 1500, "lum": 0.2}]},
            {"k": 3, "entities": [{"id": "a", "px": 400, "lum": 0.1}, {"id": "b", "px": 2000, "lum": 0.3}]}]
    s = Sightings(rows, CAMERA)
    assert (s.look("a", 1), s.look("a", 3), s.look("b", 3)) == ((1500, 0.2), (400, 0.1), (2000, 0.3))
    assert (s.look("a", 2), s.look("c", 1)) == (None, None)


def test_a_gold_cites_the_last_recognisable_frame_and_a_clipped_one_is_never_evidence():
    """Contract: `World.recognised` is the frames whose own record passes `recognisable`, and an `object_where` gold
    cites the last of them in the window, in that frame's place: a later frame cut by the image's edge or with no depth
    (a thing leaving the view) is a sighting but not evidence, and a thing seen only so is neither answerable nor
    unknown. Hand-worked on one task (events at keyframes 2, 7, 18 and 26): a red bag at (1, 1), moved 0.1 m to (1.1,
    1) at t 9 s (keyframe 6), is recognisable at 3 and 12, cut by the left edge at 6, depth-less at 8 and cut by the
    right edge at 15: `sighted` is 3, 6, 8, 12, 15, `recognised` 3 and 12; `answerable_at` the event 'began sweeping'
    (7) is (1, 1) from 3, at 'finished sweeping' (18) (1.1, 1) from 12, and at 'set off' (2) there is none; the bag
    `object_where` asks is (1.1, 1) from 12, the last recognisable frame, not the sliver at 15; a blue toy seen only
    cut by the edge (3) and depth-less (5) has `recognised` empty and is in neither pool. Regression caught: the 133
    golds M0 lost where the last frame of a thing leaving the view was a sliver (the gold citing that sliver, or
    none), or a sliver's place taken as where the thing was."""
    s = vacuum_run()
    s.item("E0001", "bag", {"color": "red"}, [(0, 9.0, (1.0, 1.0, 0.0)), (9.0, None, (1.1, 1.0, 0.0))], size=BAG)
    s.see([3, 12], "E0001", 1500)
    s.see(6, "E0001", 1500, bbox=[0, 100, 40, 140])
    s.see(8, "E0001", 1500, depth=None)
    s.see(15, "E0001", 1500, bbox=[600, 100, 639, 140])
    s.item("E0002", "toy", {"color": "blue"}, [(0, None, (3.0, 1.0, 0.0))])
    s.see(3, "E0002", 1500, bbox=[100, 0, 140, 40])
    s.see(5, "E0002", 1500, depth=None)
    w = fk.world(s)
    bag, toy = (next(e for e in w.tl.entities if e.id == eid) for eid in ("E0001", "E0002"))
    assert (w.sighted("E0001"), w.recognised("E0001")) == ([3, 6, 8, 12, 15], [3, 12])
    assert (w.sighted("E0002"), w.recognised("E0002")) == ([3, 5], [])
    assert answerable_at(w, (0, 7), bag) == LastSeen((1.0, 1.0), 3)
    assert answerable_at(w, (0, 18), bag) == LastSeen((1.1, 1.0), 12)
    assert answerable_at(w, (0, 2), bag) is None and answerable_at(w, (0, 26), toy) is None
    answerable, unknown = pools(w)
    assert not any("e:E0002" in c.subjects for c in answerable + unknown)
    (c,) = answerable
    assert (c.entities, c.gold, c.evidence_k) == (("E0001",), {"type": "position", "xy": [1.1, 1.0]}, (12,))
