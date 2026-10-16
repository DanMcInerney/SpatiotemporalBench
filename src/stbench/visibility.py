"""Private visibility log (`private/<id>/visibility.jsonl`): what each keyframe actually saw.

One line per keyframe, listing EVERY entity with at least one pixel in the instance image (unthresholded):

    {"k": 7, "entities": [{"id": "E0042", "px": 311, "bbox": [u0, v0, u1, v1],
     "depth_med_m": 2.314, "lum": 0.2231, "rgb_med": [r, g, b], "centroid_cam": [x, y, z],
     "centroid_map": [x, y, z]}, ...]}

- `bbox` is in pixels with `u1` and `v1` exclusive.
- `lum` is the mean Rec.709 relative luminance (0..1) of the entity's pixels: each sRGB-encoded JPEG pixel is decoded
  to linear light, then weighted 0.2126 R + 0.7152 G + 0.0722 B.
- `rgb_med` is the median of each sRGB channel (0..255, whole numbers) over the entity's pixels in the delivered JPEG:
  the colour the picture shows it in, which `colour_seen` names. It was added in 0.5.0-control (review defect 1: a
  catalog colour word is not what a render shows); an older row lacks it and its readers fall back to the label.
- `centroid_cam` is the mean of the entity's pixels unprojected through their planar depth at the pixel centres
  (j + 0.5, i + 0.5); `centroid_map` is the same point in the map frame. Both are diagnostics: no gold uses them.
- Pixels with depth 0 (no depth recorded) count towards `px`, `bbox` and `lum` but not towards the depth or the
  centroids; an entity with no depth pixel at all has `depth_med_m` and both centroids null.

The oracle channel's threshold is applied here (`status`); the compiler's stricter one is in `sightings.py`.
Everything is plain numpy over PNG and JPEG files; every transform comes from `stbench.camera`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .camera import Camera, cam_to_map, unproject
from .case import ScriptPaths, depth_move_problem
from .episode import write_jsonl
from .render_encode import read_png16
from .rules import glimpse_px, seen_px  # the seen and glimpse thresholds are owned by rules.py

LUMA = np.array([0.2126, 0.7152, 0.0722])
_s = np.arange(256) / 255.0
SRGB_TO_LINEAR = np.where(_s <= 0.04045, _s / 12.92, ((_s + 0.055) / 1.055) ** 2.4)


def status(px: int, cam: Camera, kind: str | None = None) -> str:
    """The oracle channel's rule: "seen" from the kind's seen_px (215 at 640x480 on the global bound) pixels, "unseen"
    under its glimpse_px (1 on the global bound: at none), "band" in between. The compiler's dead band is wider, up to
    1.25 x seen_px, 1..268 px on the global bound (`sightings.py`, `rules.gold_seen_px`)."""
    if px < glimpse_px(cam, kind):
        return "unseen"
    return "seen" if px >= seen_px(cam, kind) else "band"


def frame_entities(index: np.ndarray, depth: np.ndarray, rgb: np.ndarray, cam: Camera, pos, rot,
                   instance_map: dict[int, str]) -> list[dict]:
    """The visibility records of one keyframe, from its instance image, depth image (camera units), RGB and pose."""
    h, w = index.shape
    if depth.shape != (h, w) or rgb.shape[:2] != (h, w) or (w, h) != (cam.width, cam.height):
        raise ValueError(f"image sizes disagree: instances {index.shape}, depth {depth.shape}, rgb {rgb.shape}, "
                         f"camera {(cam.height, cam.width)}")
    flat = index.ravel()
    order = np.argsort(flat, kind="stable")
    ids, starts, counts = np.unique(flat[order], return_index=True, return_counts=True)
    rows_out = []
    for pid, s, c in zip(ids.tolist(), starts.tolist(), counts.tolist(), strict=True):
        if pid == 0:
            continue
        if pid not in instance_map:
            raise ValueError(f"instance image holds pass_index {pid}, which instance_map.json does not list")
        pix = order[s:s + c]
        v, u = pix // w, pix % w
        rec = {"id": instance_map[pid], "px": int(c), "bbox": [int(u.min()), int(v.min()), int(u.max()) + 1,
                                                              int(v.max()) + 1]}
        d = depth[v, u].astype(np.float64) * cam.depth_scale
        ok = d > 0
        rec["depth_med_m"] = round(float(np.median(d[ok])), 4) if ok.any() else None
        lin = SRGB_TO_LINEAR[rgb[v, u]]
        rec["lum"] = round(float((lin @ LUMA).mean()), 4)
        rec["rgb_med"] = [int(round(float(x))) for x in np.median(rgb[v, u], axis=0)]
        if ok.any():
            cam_pts = unproject(u[ok] + 0.5, v[ok] + 0.5, d[ok], cam).mean(axis=1)
            rec["centroid_cam"] = [round(float(x), 4) for x in cam_pts]
            rec["centroid_map"] = [round(float(x), 4) for x in cam_to_map(cam_pts, pos, rot)]
        else:
            rec["centroid_cam"] = rec["centroid_map"] = None
        rows_out.append(rec)
    return sorted(rows_out, key=lambda r: r["id"])


def load_instance_map(path: str | Path) -> dict[int, str]:
    return {int(k): v for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items()}


def load_pose(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def build(paths: ScriptPaths) -> list[dict]:
    """Visibility rows for every keyframe of a rendered script, from the pose rows' camera block."""
    cam = Camera.load(paths.camera)
    imap = load_instance_map(paths.instance_map)
    poses = load_pose(paths.pose)
    missing = [str(p) for k in range(len(poses)) for p in (paths.rgb(k), paths.depth(k), paths.instances(k))
               if not p.exists()]
    if missing:
        if moved := depth_move_problem([paths]):   # an old layout, not an unrendered script
            raise FileNotFoundError(moved)
        raise FileNotFoundError(f"{len(missing)} rendered files are missing (run `stbench render` first), "
                                f"for example {missing[0]}")
    rows = []
    for pose in poses:
        k = pose["k"]
        with Image.open(paths.rgb(k)) as im:
            rgb = np.asarray(im.convert("RGB"))
        ents = frame_entities(read_png16(paths.instances(k)), read_png16(paths.depth(k)), rgb, cam,
                              pose["camera"]["pos"], pose["camera"]["rot"], imap)
        rows.append({"k": k, "entities": ents})
    return rows


def write(paths: ScriptPaths) -> int:
    """Compute and write `private/<id>/visibility.jsonl`; returns the number of keyframes."""
    rows = build(paths)
    write_jsonl(paths.visibility, rows)
    return len(rows)


def read(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
