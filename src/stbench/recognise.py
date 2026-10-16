"""The recognisability floor (VISION rule 10): a sighting is a thing a person could name only from enough
pixels and light, in one frame whole enough to see, so only a recognisable sighting counts."""

from __future__ import annotations

from .camera import Camera

# The recognisability floor. RECOG_PX is COCO's small/medium object boundary, 32 x 32 px, at the
# benchmark's own 640 x 480 scale. RECOG_LUM is on the visibility log's `lum`, the mean Rec.709 relative luminance
# (0..1) of the entity's pixels in linear light: 0.01 is about sRGB 0.10 (25 of 255), where the pixels carry no
# colour. Over the kept episodes the 1st, 2nd and 5th percentiles of `lum` over the seen sightings of carried
# things were 0.0059, 0.0080 and 0.0121 on the apartment (3.4% under 0.01) and 0.0081, 0.0129 and 0.0214 on the
# warehouse (1.4%); a view that dark is no better than a glimpse. Of the carried things seen (269 px) in a visit, 59%
# (apartment) and 66% (warehouse) were recognisable in it: nearly every other one for want of pixels, not of light.
RECOG_PX = 1024
# M1 (2026-10-07) set it from the frozen look-dev of the apartment through the vacuum's camera, between the 2nd and 5th
# percentiles of `lum` over sightings of at least 1,024 px of vacuum/1's kinds (0.0053 and 0.0062): 0.0058. The 0.5.0
# log review (2026-10-10, defect 1) found that this left a view the comment above calls "no colour" recognisable: the
# vacuum's grey bag was evidenced at 0.0087, and a decoy "black bag" was asked while a bag in view rendered black.
# 0.5.0-control raised it to the 0.01 the comment names; `colour_seen` names what colour a recognisable view shows.
RECOG_LUM = 0.01
# 0.5.0: the bounding box keeps this many pixels from every image edge (a thing cut by the frame's edge is a sliver of
# itself), and the frame has a depth for the thing. Both are per frame,
# as rule 10 says "in a single frame": no pixel total over two frames, no larger count in one.
RECOG_EDGE_PX = 2


def recognisable(rec: dict | None, cam: Camera, what: str) -> bool:
    """Whether a visibility record of one frame (`visibility.frame_entities`' row of the entity; None for no row) is
    recognisable: at least `RECOG_PX` pixels, a mean luminance of at least `RECOG_LUM`, a bounding box at least
    `RECOG_EDGE_PX` from every edge of the image (`bbox` is `[u0, v0, u1, v1]`, `u1` and `v1` exclusive), and a median
    depth (`depth_med_m` is null when no pixel of the thing has a depth). Raises ValueError, naming `what` (the thing
    and keyframe), when the pixel count passes and the row lacks `lum`, `bbox` or `depth_med_m` (every visibility log
    has all three, so a row without one is a log that cannot say, and nothing is silently counted either way)."""
    if rec is None or rec["px"] < RECOG_PX:
        return False
    missing = [key for key in ("lum", "bbox", "depth_med_m") if key not in rec]
    missing += [key for key in ("lum", "bbox") if rec.get(key, 0) is None]  # only the depth may be null
    if missing:
        raise ValueError(f"the visibility row of {what} has no {' or '.join(missing)}: the recognisability floor "
                         f"needs it")
    u0, v0, u1, v1 = rec["bbox"]
    clipped = min(u0, v0, cam.width - u1, cam.height - v1) < RECOG_EDGE_PX
    return not clipped and rec["depth_med_m"] is not None and rec["lum"] >= RECOG_LUM
