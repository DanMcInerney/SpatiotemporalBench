"""The sheet of `stbench realism sheet` (M1's R.2): 30 vacuum frames in five strata, the standing
frames and the references, at native size in one `index.html`, and the crops of the look-alike check.

The strata, six frames each, drawn with a seed, each frame in the first stratum it fits: **dock** (phase docked,
undocking or docking), **under furniture** (its base inside a piece with an open underside), **wall-facing** (the
median depth of the middle block at most `WALL_FACING_M`; 0 is beyond range and counts for nothing), **along walls**
(cleaning, the base within `NEAR_WALL_M` of a wall or a blocking footprint) and **open floor** (the rest). A stratum
shows distinct views: of frames whose bases and headings are near enough alike (`same_view`: one docked view at the
start and one at the end of every script) it keeps one. The crops are the object's box and `CROP_MARGIN` of its size
all round, in native pixels (at least `CROP_MIN_PX` a side), `PER_CLASS` of each class (every clutter class, every
dressing class, every askable kind) cut from frames where the object covers `AREA_PX` pixels, across scripts and
poses, written under random names; the key that says what each is (class, kind, colour, script, frame) is a separate
file that no reviewer is given. Pure functions on timelines, maps and arrays; the writing is `write_sheet` and
`write_crops`."""

from __future__ import annotations

import html
import json
import math
import random
import shutil
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import Rect, dist_to_rect, point_in_rect

STRATA = ("dock", "under furniture", "wall-facing", "along walls", "open floor")
PER_STRATUM = 6
DOCK_PHASES = ("docked", "undocking", "docking")
NEAR_WALL_M, WALL_FACING_M = 0.35, 0.3
SAME_VIEW_M, SAME_VIEW_RAD = 0.15, 0.3  # one view: bases this near and headings this alike
AREA_PX, PER_CLASS = (1024, 2048), 5
CROP_MARGIN, CROP_MIN_PX = 0.25, 64  # of the object's box, each side; the least a crop is across (some context)


def centre_median(depth_m: np.ndarray) -> float | None:
    """The median depth (metres) of the middle block of a frame (the middle third of its rows and of its columns),
    leaving out what reads 0 (beyond range); None if nothing in it has a depth."""
    h, w = depth_m.shape
    block = depth_m[h // 3:2 * h // 3, w // 3:2 * w // 3]
    block = block[block > 0]
    return float(np.median(block)) if block.size else None


def stratum(phase: str, base: tuple[float, float], under: Sequence[Rect], solids: Sequence[Rect],
            depth_centre: float | None) -> str:
    """The first stratum of `STRATA` a frame fits: `under` are the footprints of the pieces with an open underside,
    `solids` the walls and blocking footprints, `depth_centre` the frame's `centre_median`."""
    if phase in DOCK_PHASES:
        return "dock"
    if any(point_in_rect(r, base[0], base[1]) for r in under):
        return "under furniture"
    if depth_centre is not None and depth_centre <= WALL_FACING_M:
        return "wall-facing"
    near = min((float(dist_to_rect(np.array([base[0]]), np.array([base[1]]), r)[0]) for r in solids), default=1e9)
    return "along walls" if phase == "cleaning" and near <= NEAR_WALL_M else "open floor"


def same_view(a: dict, b: dict) -> bool:
    """Whether two frames (each with `pose`: (map, x, y, yaw), or none) look at the same thing: one map, bases within
    `SAME_VIEW_M` and headings within `SAME_VIEW_RAD`. A frame without a pose is its own view."""
    if a.get("pose") is None or b.get("pose") is None or a["pose"][0] != b["pose"][0]:
        return False
    turn = abs((a["pose"][3] - b["pose"][3] + math.pi) % (2 * math.pi) - math.pi)
    return math.hypot(a["pose"][1] - b["pose"][1], a["pose"][2] - b["pose"][2]) < SAME_VIEW_M and turn < SAME_VIEW_RAD


def draw(frames: Sequence[dict], seed: int, per: int = PER_STRATUM) -> dict[str, list[dict]]:
    """`per` frames of each stratum, drawn with the seed from `frames` (each `{stratum, script, k, pose, ...}`),
    distinct views (`same_view`: a frame like one already drawn is passed over), spread over the scripts where there are
    several (a round robin over scripts in a seeded order); a stratum with fewer views keeps all of them."""
    rng = random.Random(seed)
    out: dict[str, list[dict]] = {}
    for name in STRATA:
        mine = sorted((f for f in frames if f["stratum"] == name), key=lambda f: (f["script"], f["k"]))
        by_script: dict[str, list[dict]] = defaultdict(list)
        for f in mine:
            by_script[f["script"]].append(f)
        for v in by_script.values():
            rng.shuffle(v)
        order = sorted(by_script)
        rng.shuffle(order)
        picked: list[dict] = []
        while len(picked) < per and any(by_script.values()):
            for s in order:
                while by_script[s] and len(picked) < per:
                    f = by_script[s].pop()
                    if not any(same_view(f, g) for g in picked):  # a view like one drawn already is passed over
                        picked.append(f)
                        break
        out[name] = picked
    return out


def crop_box(bbox: Sequence[int], shape: tuple[int, int], margin: float = CROP_MARGIN, least: int = CROP_MIN_PX
             ) -> tuple[int, int, int, int]:
    """The crop (u0, v0, u1, v1) of an object's box `[u0, v0, u1, v1]`: the box and `margin` of its size on every side,
    at least `least` across, centred on the box and kept inside an image of `shape` (h, w)."""
    span = []
    for lo, hi, limit in ((bbox[0], bbox[2], shape[1]), (bbox[1], bbox[3], shape[0])):
        side = min(max(round((hi - lo) * (1 + 2 * margin)), least), limit)
        start = min(max((lo + hi) // 2 - side // 2, 0), limit - side)
        span.append((start, start + side))
    return span[0][0], span[1][0], span[0][1], span[1][1]


def pick_crops(candidates: dict[str, list[dict]], seed: int, per: int = PER_CLASS) -> list[dict]:
    """`per` sightings of each class (each `{script, k, px, bbox, ...}` with a pixel count inside `AREA_PX`), a frame
    once and scripts in turn: the classes in order, the rest by seed. A class with fewer keeps all of them."""
    rng = random.Random(seed)
    lo, hi = AREA_PX
    out: list[dict] = []
    for cls in sorted(candidates):
        mine = sorted(candidates[cls], key=lambda c: (c["script"], c["k"], c.get("id", "")))
        pool = [c for c in mine if lo <= c["px"] <= hi]
        rng.shuffle(pool)
        taken: list[dict] = []
        seen_frames: set[tuple[str, int]] = set()
        for turn in range(2):  # first one from each script, then whatever is left
            scripts: set[str] = set()
            for c in pool:
                key = (c["script"], c["k"])
                if len(taken) < per and key not in seen_frames and (turn == 1 or c["script"] not in scripts):
                    taken.append(c)
                    seen_frames.add(key)
                    scripts.add(c["script"])
        out += [{**c, "class": cls} for c in taken]
    return out


def write_crops(picks: Sequence[dict], frame_path, out: Path, seed: int) -> list[dict]:
    """Cut the crops (`crop_box`) and write them to `out/crops/crop_0001.png`... under names in a seeded order, and the
    key (a row of `{crop, class, kind, colour, script, k, px}` for each) to `out/crop_key.json`, outside the folder.
    `frame_path(script, k)` is the picture's path. Returns the key."""
    shutil.rmtree(out / "crops", ignore_errors=True)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    order = list(range(len(picks)))
    random.Random(seed).shuffle(order)
    key = []
    for name_no, i in enumerate(order, 1):
        p = picks[i]
        with Image.open(frame_path(p["script"], p["k"])) as im:
            rgb = im.convert("RGB")
            box = crop_box(p["bbox"], (rgb.height, rgb.width))
            rgb.crop(box).save(out / "crops" / f"crop_{name_no:04d}.png")
        key.append({"crop": f"crop_{name_no:04d}.png", "class": p["class"], "kind": p.get("kind"),
                    "colour": p.get("colour"), "script": p["script"], "k": p["k"], "px": p["px"]})
    (out / "crop_key.json").write_text(json.dumps(key, indent=1) + "\n", encoding="utf-8")
    return key


def _section(title: str, items: Sequence[tuple[str, str]]) -> str:
    cells = "".join(f'<figure><img src="{html.escape(src)}" width="640" height="480"><figcaption>{html.escape(cap)}'
                    f"</figcaption></figure>" for src, cap in items)
    return f"<h2>{html.escape(title)}</h2><div class=\"row\">{cells}</div>"


def write_sheet(out: Path, strata: dict[str, list[dict]], frame_path, standing: Sequence[tuple[Path, str]],
                references: Sequence[Path]) -> Path:
    """`out/index.html` and the pictures it shows, each at native size in a file of its own: the strata's frames, the
    standing frames and the references, with captions. Returns the page's path."""
    pics = out / "frames"
    shutil.rmtree(pics, ignore_errors=True)
    pics.mkdir(parents=True, exist_ok=True)
    parts = []
    for name, frames in strata.items():
        items = []
        for n, f in enumerate(frames):
            dest = pics / f"{name.replace(' ', '_')}_{n}.jpg"
            shutil.copy(frame_path(f["script"], f["k"]), dest)
            items.append((f"frames/{dest.name}", f"{name}: script {f['script']}, frame {f['k']}"))
        parts.append(_section(f"The vacuum, {name} ({len(frames)})", items))
    items = []
    for n, (src, cap) in enumerate(standing):
        dest = pics / f"standing_{n}.jpg"
        shutil.copy(src, dest)
        items.append((f"frames/{dest.name}", cap))
    parts.append(_section("Standing camera (release settings)", items))
    items = []
    for n, src in enumerate(references):
        dest = pics / f"reference_{n}{src.suffix.lower()}"
        shutil.copy(src, dest)
        items.append((f"frames/{dest.name}", f"reference {src.name}"))
    parts.append(_section("References", items))
    page = out / "index.html"
    style = "body{font-family:sans-serif}figure{display:inline-block;margin:4px}.row{display:flex;flex-wrap:wrap}"
    page.write_text(f"<!doctype html><meta charset=\"utf-8\"><title>Realism sheet</title><style>{style}</style>"
                    + "".join(parts), encoding="utf-8")
    return page
