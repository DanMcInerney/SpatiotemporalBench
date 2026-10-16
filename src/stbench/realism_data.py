"""What `stbench realism report` reads: a dataset's scripts and a probe folder, as the inputs of `realism_report`.

A probe folder (`stbench realism probe`, one per script, named by the script's id: `probe_folder`) holds `meta.json`,
`cards.json` (the grey cards), `dressing_index.json` (a debug index's id to class), `standing/` (a dataset of its own:
the standing camera's script) and `index_script/` and `index_standing/` (`%06d.npz`: `index` uint16, with the dressing
pieces from `realism_report.DEBUG_BASE` up, and `depth` float32 in metres, 0 beyond range). The timeline is read through
its query helpers, never its lists (`tests/test_architecture.py`). """

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .camera import Camera
from .case import ScriptPaths, read_index
from .mapspec import MapSpec
from .realism_noise import fill_mask, patch_stats
from .realism_report import WINDOW_MIN_PX, objects_in_frame, stops, window_stops
from .rules import seen_px
from .timeline import Timeline, load
from .visibility import LUMA

CRUSHED_LUMA, CLIPPED = 5, 255  # the camera model's: a pixel at or under this 8-bit luma, a channel at this
IMAGES = (".jpg", ".jpeg", ".png")
ROBOT_NOTE = "front camera"  # a reference whose `sources.json` camera_note says this is a robot's own camera's frame
FULL_FRAME_PX = 480  # a reference narrower than this is a thumbnail: not read for exposure or noise
WINDOW_CLASS = "window_sill"  # the dressing class of a window's trim: its pieces' box has the glass inside it
WINDOW_JOIN_PX, WINDOW_LEAST_PX = 4, 24  # trim pieces this near are one window; a window's box is this wide at least
WINDOW_FRAMES = 24  # frames of a script read for the window measure


def probe_folder(root: str | Path, sid: str) -> Path:
    """The folder of a script's probe: `root/<sid>`, or `root` itself where it is already named by the script's id (a
    probe's `--out` and the report's `--probe` both go through here, so what one writes the other reads)."""
    root = Path(root)
    return root if root.name == sid else root / sid


def script_paths(dataset: str | Path) -> list[ScriptPaths]:
    """Every script of a dataset, by id."""
    return [ScriptPaths(dataset, sid) for sid, e in sorted(read_index(dataset).items()) if e.get("kind") == "script"]


def read_log(paths: ScriptPaths) -> dict:
    return json.loads(paths.render_log.read_text(encoding="utf-8"))


def room_name(m: MapSpec, xy: tuple[float, float]) -> str:
    """The name of the room a point is in; "outside" in a doorway or a wall."""
    rid = m.region_of(xy)
    return m.region(rid).name if rid else "outside"


def rgb8(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def srgb_luma(a: np.ndarray) -> np.ndarray:
    """Luminance of 8-bit sRGB-encoded pixels, 0 to 1 (Rec.709 weights on the encoded values: a display's luma)."""
    return (a.astype(np.float64) / 255.0) @ LUMA


def shares(a: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    """The share of an 8-bit image's pixels with a channel at 255 and the share at or under `CRUSHED_LUMA`: what the
    camera model logs for a frame, for a picture that has no log (a reference). Pixels of `mask` are left out."""
    luma = (a.astype(np.float64)) @ LUMA
    keep = np.ones(a.shape[:2], dtype=bool) if mask is None else ~mask
    return float((a == CLIPPED).any(axis=-1)[keep].mean()), float((luma <= CRUSHED_LUMA)[keep].mean())


def exposure_frames(paths: ScriptPaths, tl: Timeline, m: MapSpec) -> list[dict]:
    """Per frame `{k, room, ev, lum, clipped, crushed}`: the room of its base, the frame log's numbers and the picture's
    mean sRGB luminance."""
    out = []
    for f in read_log(paths)["develop"]:
        kf = tl.keyframe(f["k"])
        out.append({"k": f["k"], "room": room_name(m, kf.base[:2]), "ev": f["ev"], "clipped": f["clipped"],
                    "crushed": f["crushed"], "lum": float(srgb_luma(rgb8(paths.rgb(f["k"]))).mean())})
    return out


def noise_patches(paths: ScriptPaths, ks: list[int]) -> list[dict]:
    """The flat patches (`realism_noise.patch_stats`) of the given frames' pictures."""
    return [p for k in ks for p in patch_stats(rgb8(paths.rgb(k)))]


def robot_references(folder: str | Path) -> list[Path] | None:
    """The pictures of a references folder that a robot's own camera took, by its `sources.json` (a camera_note that
    names a front camera); None where the folder has no `sources.json`."""
    notes = Path(folder) / "sources.json"
    if not notes.is_file():
        return None
    entries = json.loads(notes.read_text(encoding="utf-8"))
    return sorted(Path(folder) / e["file"] for e in entries if ROBOT_NOTE in e.get("camera_note", ""))


def reference_stats(folder: str | Path) -> dict:
    """The references' numbers, from the robot's own frames (`robot_references`; every picture where there is no
    `sources.json`) with their flat fills (`realism_noise.FILL`) left out: `fills` (the share of a picture that is fill,
    the mean), `clipped` and `crushed` (shares of the pixels left), `clipped_with_fills` (the fills counted as the blown
    highlights they are) and, from the pictures `FULL_FRAME_PX` wide or more (a thumbnail is downscaled, so denoised,
    and not the camera's own tone), `exposure` (the median, least and greatest mean sRGB luma of a picture, the fills
    left out) and the flat patches."""
    chosen = robot_references(folder)
    basis = "pictures of the robot's own camera" if chosen is not None else "every picture (no sources.json)"
    files = chosen if chosen is not None else sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGES)
    pics = [rgb8(p) for p in files]
    masks = [fill_mask(a) for a in pics]
    pairs = [shares(a, m) for a, m in zip(pics, masks, strict=True)]
    full = [(a, m) for a, m in zip(pics, masks, strict=True) if a.shape[1] >= FULL_FRAME_PX and not m.all()]
    lum = [float(srgb_luma(a)[~m].mean()) for a, m in full]
    blown = [float(((a == CLIPPED).any(axis=-1) | m).mean()) for a, m in zip(pics, masks, strict=True)]
    return {"basis": basis, "pictures": len(pics), "files": [p.name for p in files],
            "fills": float(np.mean([m.mean() for m in masks])) if pics else None,
            "clipped": float(np.mean([c for c, _ in pairs])) if pics else None,
            "clipped_with_fills": float(np.mean(blown)) if pics else None,
            "crushed": float(np.mean([c for _, c in pairs])) if pics else None,
            "exposure": {"median": float(np.median(lum)), "min": min(lum), "max": max(lum)} if lum else None,
            "patches": [q for a, m in full for q in patch_stats(a, m)]}


def sightings(paths: ScriptPaths, tl: Timeline) -> list[dict]:
    """Each line of the visibility log as sightings `{kind, px, lum}` of its entities (the log lists every entity with a
    pixel)."""
    out = []
    for line in paths.visibility.read_text(encoding="utf-8").splitlines():
        for e in json.loads(line)["entities"]:
            out.append({"kind": tl.entity(e["id"]).kind, "px": e["px"], "lum": e["lum"], "id": e["id"]})
    return out


def linear_stops(folder: Path, ks: list[int]) -> list[dict]:
    """`realism_report.stops` of the kept linear passes of the given frames (`%06d.linear.npy`), in the passes' unit."""
    out = []
    for k in ks:
        x = np.load(folder / f"{k:06d}.linear.npy").astype(np.float64)
        out.append({"k": k, **stops(x @ LUMA)})
    return out


def window_boxes(index: np.ndarray, classes: dict[int, str]) -> list[tuple[int, int, int, int]]:
    """The box (u0, v0, u1, v1) of each window in a probe index frame: a window's trim is four pieces of class
    `WINDOW_CLASS` (a jamb each side, a head, a sill board) round its glass, so the pieces whose boxes touch (within
    `WINDOW_JOIN_PX`) are one window and its box is theirs together. A window is in view when its pieces show at least
    `WINDOW_MIN_PX` pixels and its box is at least `WINDOW_LEAST_PX` across (a lone jamb at the edge is none)."""
    boxes = []
    for pid in (p for p in np.unique(index) if classes.get(int(p)) == WINDOW_CLASS):
        vs, us = np.nonzero(index == pid)
        boxes.append([int(us.min()), int(vs.min()), int(us.max()) + 1, int(vs.max()) + 1, int(len(us))])
    joined = True
    while joined:
        joined = False
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                if (a[0] <= b[2] + WINDOW_JOIN_PX and b[0] <= a[2] + WINDOW_JOIN_PX
                        and a[1] <= b[3] + WINDOW_JOIN_PX and b[1] <= a[3] + WINDOW_JOIN_PX):
                    a[:] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]), a[4] + b[4]]
                    boxes.remove(b)
                    joined = True
                    break
            if joined:
                break
    return [(u0, v0, u1, v1) for u0, v0, u1, v1, n in boxes
            if n >= WINDOW_MIN_PX and min(u1 - u0, v1 - v0) >= WINDOW_LEAST_PX]


def window_range(folder: Path, probe: Path, ks: list[int]) -> list[float]:
    """For each of the given frames that has a window in view (`window_boxes` of the probe's index of the script's own
    frames), the stops the brightest window stands over the room (`realism_report.window_stops` of the kept linear pass,
    `folder/%06d.linear.npy`)."""
    classes, out = dressing_index(probe), []
    for k in ks:
        f = probe / "index_script" / f"{k:06d}.npz"
        if not f.exists() or not (folder / f"{k:06d}.linear.npy").exists():
            continue
        if boxes := window_boxes(read_frame(f)[0], classes):
            luma = np.load(folder / f"{k:06d}.linear.npy").astype(np.float64) @ LUMA
            out.append(max(window_stops(luma, b) for b in boxes))
    return out


# -- the probe -----------------------------------------------------------------------------------------------------
def read_frame(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(index, depth) of one probe frame."""
    with np.load(path) as z:
        return z["index"], z["depth"]


def pixel_counts(index: np.ndarray) -> dict[int, int]:
    ids, counts = np.unique(index, return_counts=True)
    return {int(i): int(n) for i, n in zip(ids, counts, strict=True)}


def dressing_index(probe: Path) -> dict[int, str]:
    f = probe / "dressing_index.json"
    return {int(k): v for k, v in json.loads(f.read_text(encoding="utf-8")).items()} if f.exists() else {}


def probe_objects(probe: Path, folder: str, cam: Camera, trim: tuple[str, ...]) -> list[dict]:
    """`realism_report.objects_in_frame` of each frame of `probe/<folder>/` (`index_script` or `index_standing`)."""
    classes = dressing_index(probe)
    return [{"k": int(p.stem), **objects_in_frame(pixel_counts(read_frame(p)[0]), classes, seen_px(cam), trim)}
            for p in sorted((probe / folder).glob("*.npz"))]


def probe_cards(probe: Path) -> dict:
    f = probe / "cards.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def probe_meta(probe: Path) -> dict:
    return json.loads((probe / "meta.json").read_text(encoding="utf-8"))


def standing_paths(probe: Path) -> ScriptPaths:
    """The standing camera's script of a probe folder."""
    return ScriptPaths(probe / "standing", probe_meta(probe)["standing_id"])


def standing_rgb_paths(probe: Path) -> ScriptPaths | None:
    """The script of the standing frames rendered at release settings, None if the probe made none."""
    meta = probe_meta(probe)
    return ScriptPaths(probe / "standing_rgb", meta["standing_rgb_id"]) if meta.get("standing_rgb_id") else None


def timeline_of(paths: ScriptPaths) -> Timeline:
    return load(paths.timeline)

