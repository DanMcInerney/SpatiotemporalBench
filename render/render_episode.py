# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Render a script's keyframes in Blender: raw float depth, object-index and scene-linear colour arrays, and no JPEG.

    blender --background --factory-startup --python render/render_episode.py -- \
        --root DATASET/scripts/<sid> --private DATASET/private/<sid> --tmp TMPDIR --src REPO/src \
        [--fast] [--frames A:B] [--samples N] [--appearance placeholder|assets --library L.blend --manifest M.json]

Started by `stbench render`; never imported by `stbench`. Writes into --tmp `%06d.depth.npy` (planar camera-z in metres,
float32, background is a large number), `%06d.index.npy` (the object-index pass, float32, 0 = structure or sky),
`%06d.linear.npy` (the Combined pass, float16, H x W x 3, scene-linear in the unit `cycles_setup.UNIT_LUX` names and
clipped at `LINEAR_CLIP`: float16 overflows at 65,504, and anything past 60,000 is white at every exposure the camera
model allows), `instance_map.json` and `render_meta.json`. `stbench render` develops the linear pass into the JPEG with
the robot's camera model (`stbench.camera_model`): Blender's view transform never touches a picture. The data passes
never go through colour management either: all three are read from one float OpenEXR. The camera's near clip is the
robot's (`render_encode.rgb_clip_m` of its depth range's near bound, read from `--root/robot.json`), so a surface that
depth reads is whole in the picture. Exits with status 3 when no GPU is available to Cycles: it never falls back to the
CPU. `--appearance assets` builds carried, light and mover entities and the textured surfaces from the asset library
(`stbench render` has checked it first), lit by `lighting.setup`; `render_meta.json` records the appearance, the
samples, every Cycles and colour setting the scene ended with (`cycles`, from `cycles_setup.settings`), what the
lighting did (`lighting`), and `unit_lux`, `linear_clip` and `clip_start_m`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cycles_setup import UNIT_LUX, configure, settings  # noqa: E402

RGB_SAMPLES, FAST_SAMPLES, ASSET_SAMPLES = 32, 8, 128
LINEAR_CLIP = 60000.0


def parse(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--private", type=Path, required=True)
    ap.add_argument("--tmp", type=Path, required=True)
    ap.add_argument("--src", type=Path, required=True, help="the directory that holds the stbench package")
    ap.add_argument("--frames", default=None, help="A:B, keyframes A <= k < B (default: all)")
    ap.add_argument("--fast", action="store_true", help="fewer RGB samples; depth and ids are unchanged")
    ap.add_argument("--samples", type=int, default=None, help="RGB samples per pixel (overrides --fast)")
    ap.add_argument("--appearance", choices=("placeholder", "assets"), default="placeholder")
    ap.add_argument("--library", type=lambda s: Path(s).resolve(), default=None,
                    help="the asset library .blend (assets appearance)")
    ap.add_argument("--manifest", type=lambda s: Path(s).resolve(), default=None,
                    help="the asset manifest (assets appearance)")
    return ap.parse_args(argv[argv.index("--") + 1:] if "--" in argv else [])


def _exr_format(sc) -> None:
    s = sc.render.image_settings
    if hasattr(s, "media_type"):
        s.media_type = "MULTI_LAYER_IMAGE"
    s.file_format, s.color_depth, s.exr_codec = "OPEN_EXR_MULTILAYER", "32", "ZIP"


def read_passes(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(depth, object index, combined) float32 arrays, (H, W) and (H, W, 3) for the last, from a multilayer float EXR,
    without colour management. The colour channels are picked by name, whatever order the file keeps them in."""
    import OpenImageIO as oiio

    inp = oiio.ImageInput.open(str(path))
    depth = index = combined = None
    i = 0
    while inp.seek_subimage(i, 0):
        spec = inp.spec()
        names = ",".join(spec.channelnames)
        if "Depth" in names:
            depth = inp.read_image(i, 0, 0, 1, oiio.FLOAT)[..., 0]
        elif "Object Index" in names:
            index = inp.read_image(i, 0, 0, 1, oiio.FLOAT)[..., 0]
        elif "Combined" in names:
            rgb = {name.rsplit(".", 1)[-1]: c for c, name in enumerate(spec.channelnames)}
            data = inp.read_image(i, 0, 0, spec.nchannels, oiio.FLOAT)
            combined = data[..., [rgb["R"], rgb["G"], rgb["B"]]]
        i += 1
    inp.close()
    if depth is None or index is None or combined is None:
        raise RuntimeError(f"{path}: the EXR has no Depth, Object Index and Combined parts")
    return (np.ascontiguousarray(depth, dtype=np.float32), np.ascontiguousarray(index, dtype=np.float32),
            np.ascontiguousarray(combined, dtype=np.float32))


def linear_pass(combined: np.ndarray) -> np.ndarray:
    """The Combined pass as a float16 array: a not-a-number reads 0, and nothing is above `LINEAR_CLIP`."""
    return np.minimum(np.nan_to_num(combined, nan=0.0, posinf=LINEAR_CLIP, neginf=0.0), LINEAR_CLIP).astype(np.float16)


def main() -> None:
    args = parse(sys.argv)
    sys.path.insert(0, str(args.src))
    import build_scene

    from stbench import robots
    from stbench.camera import Camera
    from stbench.mapspec import load_map
    from stbench.render_encode import rgb_clip_m
    from stbench.timeline import load

    t_start = time.perf_counter()
    tl = load(args.private / "timeline.json")
    m = load_map(tl.episode.map)
    cam = Camera.load(args.root / "sensors" / "camera.json")
    robot = robots.from_doc(json.loads((args.root / "robot.json").read_text(encoding="utf-8")))
    sc = bpy.context.scene
    assets = None
    if args.appearance == "assets":
        from stbench import asset_manifest

        if args.library is None or args.manifest is None:
            raise SystemExit("STBENCH-ERROR --appearance assets needs --library and --manifest")
        assets = (asset_manifest.load(args.manifest), args.library)
    samples = args.samples or (FAST_SAMPLES if args.fast else ASSET_SAMPLES if assets else RGB_SAMPLES)
    device = configure(sc, cam.width, cam.height, samples, args.appearance)
    hfov = json.loads((args.root / "sensors" / "camera.json").read_text())["hfov_deg"]
    scene = build_scene.build(tl, m, hfov, assets)
    clip_start = rgb_clip_m(robot.depth.near_m)  # build_scene's own near clip is 5 cm: it cut close surfaces open
    scene.camera.data.clip_start = clip_start
    cycles = settings(sc)  # after the build: the lighting sets the exposure and the white balance
    args.tmp.mkdir(parents=True, exist_ok=True)
    (args.tmp / "instance_map.json").write_text(json.dumps({str(k): v for k, v in scene.instance_map.items()}))
    lo, hi = 0, len(tl.ego.keyframes)
    if args.frames:
        a, b = args.frames.split(":")
        lo, hi = int(a or lo), int(b or hi)
    _exr_format(sc)
    frames = []
    for k in range(lo, hi):
        kf = tl.keyframe(k)
        t0 = time.perf_counter()
        scene.apply_time(kf.t)
        scene.apply_camera(kf)
        bpy.ops.render.render(write_still=False)
        t1 = time.perf_counter()
        exr = args.tmp / f"{k:06d}.exr"
        bpy.data.images["Render Result"].save_render(str(exr), scene=sc)
        depth, index, combined = read_passes(exr)
        exr.unlink()
        np.save(args.tmp / f"{k:06d}.depth.npy", depth)
        np.save(args.tmp / f"{k:06d}.index.npy", index)
        np.save(args.tmp / f"{k:06d}.linear.npy", linear_pass(combined))
        t2 = time.perf_counter()
        frames.append({"k": k, "render_s": round(t1 - t0, 4), "total_s": round(t2 - t0, 4)})
        print(f"STBENCH-FRAME {k} render {t1 - t0:.3f}s total {t2 - t0:.3f}s", flush=True)
    meta = {"blender": bpy.app.version_string, "device": device, "samples": samples, "fast": bool(args.fast),
            "width": cam.width, "height": cam.height, "use_gpu_subdivision": False, "seed": 0,
            "appearance": args.appearance, "cycles": cycles, "lighting": scene.lighting,
            "unit_lux": UNIT_LUX, "linear_clip": LINEAR_CLIP, "clip_start_m": clip_start,
            "frames": frames, "script_s": round(time.perf_counter() - t_start, 3), "pid": os.getpid()}
    (args.tmp / "render_meta.json").write_text(json.dumps(meta))


main()
