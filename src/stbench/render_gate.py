"""The render's own check: no large surface may go dark partway through a script.

A Blender run has gone wrong in one way so far (FINDINGS, the drone's dark paving): part-way through a drone script the
square's paving (a procedural material, so no texture could be missing) falls to about a five-hundredth of its
brightness in every frame after, while everything else in the scene stays as it was, and the camera's auto-exposure
lifts the frame (up to +2 EV of exposure and +6 EV of gain) and clips every sunlit surface. It is not a function of the
script: a second render of the same script, with the same code, did not show it. So it is checked in the render, from
what the render made.

The check reads the linear pass and the object index (never the JPEG): each entity covering at least `MIN_FRACTION`
of a frame has a mean luminance in that frame, and an entity that falls below `DARK_RATIO` of its own
`REFERENCE_PERCENTILE`th percentile over the script, in at least `MIN_DARK_FRAMES` frames, has gone dark. The reference
is the entity's own, so a dark sofa or the inside of a rack is not flagged, it is a high percentile so that a surface
that is dark for most of the frames it is in view still reads against the frames before, and the entity needs
`MIN_FRAMES` frames in view.

The separating figure is each entity's third-darkest frame over its reference (`statistic`; single-frame minima
overlap, which is why three frames are asked for). The two defective drone scripts read 0.0006 and 0.0011 on the
developed JPEGs (the only record of them), against 0.042 on the JPEGs of the 60 healthy renders of the Release 0.6.0
drafts and 0.0148 on the linear passes of 30 healthy vacuum renders (the worst: a dining table seen from underneath).
`DARK_RATIO` (0.0039) sits 3.8x under the worst healthy linear figure and 3.5x over the worst defective one (a false
positive fails the build, and a miss is caught by the scan, so the margin is not tilted either way). `render_log.json`
records the statistic of every render (`dark_gate`), so each release's renders show how far the healthy ones are
from it.

What it cannot see: a fault that starts before the surface is first in view, an entity that is dark in all its frames
(its reference is dark too), an entity that is never in view for `MIN_FRAMES` frames at `MIN_FRACTION` of the frame, and
index 0 (structure and sky, which are not entities).
"""

from __future__ import annotations

import numpy as np

from .camera_model import luminance

MIN_FRACTION = 0.02  # an entity this large in the frame is measured
MIN_FRAMES = 8  # the frames an entity needs in view for its reference to mean anything
REFERENCE_PERCENTILE = 90  # of the entity's mean luminances over the script
DARK_RATIO = 1 / 256  # a frame this far below the entity's reference is dark
MIN_DARK_FRAMES = 3  # a flicker of one or two frames is not the fault

History = dict[str, list[tuple[int, float]]]  # entity id -> (keyframe, mean luminance) where it was measured


def entity_means(index: np.ndarray, linear: np.ndarray, instance_map: dict[str, str]) -> dict[str, float]:
    """`{entity id: mean luminance}` of each entity that covers at least `MIN_FRACTION` of the frame. `index` is the
    object-index pass, `linear` the scene-linear colour pass (H x W x 3) and `instance_map` the render's own
    `{index: entity id}` (index 0, structure and sky, is not an entity)."""
    ids = index.astype(np.int64).ravel()
    lum = luminance(linear.astype(np.float32)).ravel().astype(np.float64)
    count = np.bincount(ids)
    total = np.bincount(ids, weights=lum, minlength=len(count))
    out = {}
    for i in np.nonzero(count >= MIN_FRACTION * ids.size)[0]:
        entity = instance_map.get(str(int(i)))
        if entity is not None:
            out[entity] = float(total[i] / count[i])
    return out


def _ratios(history: History) -> dict[str, list[tuple[int, float]]]:
    """`{entity: (keyframe, mean over its reference)}` for each entity with a reference to read against."""
    out = {}
    for entity, rows in history.items():
        if len(rows) < MIN_FRAMES:
            continue
        reference = float(np.percentile([m for _, m in rows], REFERENCE_PERCENTILE))
        if reference > 0.0:
            out[entity] = [(k, m / reference) for k, m in rows]
    return out


def dark_surfaces(history: History) -> dict[str, list[int]]:
    """`{entity id: the keyframes where it is dark}` for each entity that has gone dark (at least `MIN_DARK_FRAMES`
    frames under `DARK_RATIO` of its reference)."""
    found = {}
    for entity, rows in _ratios(history).items():
        dark = [k for k, r in rows if r < DARK_RATIO]
        if len(dark) >= MIN_DARK_FRAMES:
            found[entity] = dark
    return found


def statistic(history: History) -> dict:
    """What `render_log.json` records for every render: the entity whose `MIN_DARK_FRAMES`th-darkest frame is lowest
    against its reference, that ratio, how many of its frames are under `DARK_RATIO`, and the threshold. The entity and
    ratio are None when no entity was in view long enough to measure."""
    worst = None
    for entity, rows in _ratios(history).items():
        ratios = sorted(r for _, r in rows)
        if worst is None or ratios[MIN_DARK_FRAMES - 1] < worst[1]:
            worst = (entity, ratios[MIN_DARK_FRAMES - 1], sum(r < DARK_RATIO for r in ratios))
    return {"entity": worst and worst[0], "third_lowest_ratio": worst and round(worst[1], 6),
            "frames_under": worst[2] if worst else 0, "threshold": DARK_RATIO}


def onsets(dark: dict[str, list[int]]) -> dict[str, int]:
    """`{entity: its first dark keyframe}`: what two renders are compared by (a transient fault starts somewhere else
    each time, a darkening the scene itself makes starts at the same keyframe)."""
    return {entity: frames[0] for entity, frames in dark.items()}


def describe(dark: dict[str, list[int]]) -> str:
    """The sentence a failed render says: which entities, in which keyframes."""
    return "; ".join(f"{entity} in keyframes {frames[0]}-{frames[-1]} ({len(frames)} frames)"
                     for entity, frames in sorted(dark.items()))
