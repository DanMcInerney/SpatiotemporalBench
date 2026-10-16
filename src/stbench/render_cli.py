"""`stbench render` and `stbench develop`.

`render` runs Blender (path in `STBENCH_BLENDER`, see `blender_run`) on `render/render_episode.py`, which leaves each
keyframe's raw float passes in a temporary folder: depth (planar metres), the object index and the scene-linear Combined
pass (float16, in the unit `render/cycles_setup.UNIT_LUX` names). `encode_frames` turns them into the script's files and
`render` deletes the folder: the robot's camera model (`camera_model`) develops the linear pass into
`sensors/rgb/%06d.jpg` (Blender writes no JPEG), depth is cut to the robot's range and written to
private `depth/%06d.png` (uint16 counts of the unit `sensors/camera.json` names, 0 = none; the agent is fed the camera,
so depth is the scorer's and never under `scripts/`), the object index to private `instances/%06d.png`, both zeroed
where the robot's own body is (`render_encode.self_filter`), plus private `instance_map.json` and `render_log.json`.
It refuses to proceed if Cycles had no GPU. `--appearance placeholder` (the default) is M1's procedural geometry.
`--appearance assets` renders from the asset library and refuses unless that library is verified
(`asset_render.require_library`): there is no fallback to placeholders, so a render labelled photoreal is photoreal.

A render is checked before it is encoded (`render_gate`): a surface that goes dark part-way through a script, which a
Blender run has done twice in 22 drone scripts and not in a second run of the same script, makes `render` run Blender
again, up to `RENDER_ATTEMPTS` runs, and an error at once if two runs darken the same surfaces from the same keyframes
(the scene's doing). The log records `render_attempts`, `dark_first_trip` (what the first run lost, by first keyframe)
and `dark_gate`, the gate's statistic for the render kept.

`--keep-linear DIR` keeps the linear passes in DIR, and `develop` re-develops them: a change to the camera model (or a
robot's `IspSpec`, or an `--isp` override file while the look is being tuned) then costs seconds, not a render.

The log records how the frames were made (the appearance, the samples, the settings, the values the frames were
developed with). `render_binding_problems` reads from it that the script was rendered and that every keyframe was:
`perceive` and the compiler refuse a script with no render log or a partial render. What ties the frames to their
timeline is that a rendered script's `timeline.json` and `sensors/camera.json` are never rewritten with other content
(`episode.write_script`). `stbench perceive` (`perceive_cmd.py`) turns the renders into the private visibility log and
the oracle channel."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np

from . import asset_render, blender_run, camera_model, render_gate
from .asset_manifest import AssetError
from .blender_run import SRC_DIR, BlenderError, blender_path  # noqa: F401  (blender_path is part of this module's API)
from .camera import Camera
from .case import ScriptPaths, depth_before_0_6_0
from .render_encode import cut_depth, depth_to_units, index_to_u16, self_filter, write_png16
from .robots import ROBOTS, IspSpec
from .timeline import Timeline, load

APPEARANCES = ("placeholder", "assets")
GPU_TYPES = ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI")
TMP_NAME = "render_tmp"
LINEAR = "{:06d}.linear.npy"  # a kept linear pass, in `render_tmp` and in a `--keep-linear` folder
RENDER_ATTEMPTS = 3  # a render that went dark (`render_gate`) is made again in a fresh Blender, up to this many runs


class RenderError(Exception):
    """A render failed; the message says where, what is wrong and what to do."""


def parse_frames(text: str | None, n: int) -> tuple[int, int]:
    """"A:B" -> keyframes A <= k < B ("A:" and ":B" are open ended); None means all of them."""
    if not text:
        return 0, n
    a, sep, b = text.partition(":")
    if not sep:
        raise RenderError(f"--frames must look like A:B (half-open), got {text!r}")
    lo, hi = (int(a) if a else 0), (int(b) if b else n)
    if not 0 <= lo < hi <= n:
        raise RenderError(f"--frames {text!r} is outside the script's {n} keyframes (need 0 <= A < B <= {n})")
    return lo, hi


def blender_command(paths: ScriptPaths, tmp: Path, frames: tuple[int, int], fast: bool = False,
                    samples: int | None = None, appearance: str = "placeholder", library: Path | None = None,
                    manifest: Path | None = None) -> list[str]:
    """The Blender command line for one script. Everything the render log records goes through it."""
    args = ["--root", str(paths.root), "--private", str(paths.private), "--tmp", str(tmp), "--src", str(SRC_DIR),
            "--frames", f"{frames[0]}:{frames[1]}", "--appearance", appearance]
    if fast:
        args.append("--fast")
    if samples is not None:
        args += ["--samples", str(samples)]
    if appearance == "assets":
        args += ["--library", str(library), "--manifest", str(manifest)]
    try:
        return blender_run.command("render_episode.py", *args)
    except BlenderError as e:
        raise RenderError(str(e)) from e


def run_blender(cmd: list[str], script_id: str) -> None:
    try:
        blender_run.run(cmd, f"rendering {script_id}")
    except BlenderError as e:
        raise RenderError(str(e)) from e


def check_device(meta: dict) -> None:
    dev = meta.get("device", {})
    if dev.get("type") not in GPU_TYPES or not dev.get("names"):
        raise RenderError(f"Cycles did not render on a GPU (device info: {dev}); refusing to proceed. "
                          "Rendering on the CPU would take about 50x longer.")


# -- the robot's camera, the developed frames ----------------------------------------------------------------------
def load_isp(path: Path, base: IspSpec) -> IspSpec:
    """`base` with the fields of a JSON file replaced (a look-dev candidate: any of `IspSpec`'s fields; the file need
    hold only the ones it changes). A name that is not a field is refused."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = sorted(set(doc) - {f.name for f in fields(IspSpec)})
    if unknown:
        known = ", ".join(f.name for f in fields(IspSpec))
        raise RenderError(f"{path}: {', '.join(unknown)} are not IspSpec fields ({known})")
    if "exposure_ev" in doc:
        doc["exposure_ev"] = tuple(doc["exposure_ev"])
    return replace(base, **doc)


def _isp(tl: Timeline, isp: IspSpec | None) -> IspSpec:
    """The values a script is developed with: `isp` when one is given, else its robot's own."""
    return ROBOTS[tl.episode.robot].camera.isp if isp is None else isp


def _camera(tl: Timeline, cam: Camera, isp: IspSpec | None):
    """The camera the script's robot has now, at the size `sensors/camera.json` says Blender rendered (a script's
    own file: a test may render small), developing with `isp` or its own values."""
    return replace(ROBOTS[tl.episode.robot].camera, width=cam.width, height=cam.height, isp=_isp(tl, isp))


def _develop_frame(paths: ScriptPaths, tl: Timeline, camera, k: int, linear: np.ndarray) -> dict:
    """Develop one frame's linear pass into its JPEG; the frame's log. The noise is seeded by the timeline's own
    identity and `k`, never by the random script id, so a rebuilt script develops to the same bytes."""
    ep = tl.episode
    jpeg, log = camera_model.develop(linear.astype(np.float32), key=(ep.map, ep.seed, ep.robot, ep.job, "camera", k),
                                     camera=camera)
    paths.rgb(k).write_bytes(jpeg)
    return {"k": k, **log}


def _history_in_tmp(tmp: Path, frames: tuple[int, int]) -> render_gate.History:
    """Each large entity's mean luminance in each keyframe, from the raw passes in `tmp` (`render_gate`)."""
    instance_map = json.loads((tmp / "instance_map.json").read_text(encoding="utf-8"))
    history: render_gate.History = {}
    for k in range(*frames):
        means = render_gate.entity_means(np.load(tmp / f"{k:06d}.index.npy"), np.load(tmp / LINEAR.format(k)),
                                         instance_map)
        for entity, mean in means.items():
            history.setdefault(entity, []).append((k, mean))
    return history


def _run_gated(cmd: list[str], tmp: Path, frames: tuple[int, int], script_id: str) -> tuple[dict, dict]:
    """Run Blender into `tmp` until no surface has gone dark (`render_gate`); (its `render_meta`, the gate's record for
    the log: `render_attempts`, `dark_first_trip` and `dark_gate`). A trip is the transient fault unless the next run
    trips at the same keyframes: then the scene itself darkens that surface, and it is an error at once. Fails after
    `RENDER_ATTEMPTS` runs in any case."""
    first_trip: dict[str, int] | None = None
    trips: list[dict[str, int]] = []
    for attempt in range(1, RENDER_ATTEMPTS + 1):
        run_blender(cmd, script_id)
        meta = json.loads((tmp / "render_meta.json").read_text(encoding="utf-8"))
        check_device(meta)
        history = _history_in_tmp(tmp, frames)
        if not (dark := render_gate.dark_surfaces(history)):
            return meta, {"render_attempts": attempt, "dark_first_trip": first_trip,
                          "dark_gate": render_gate.statistic(history)}
        what = f"{script_id}: a surface went dark part-way through the render: {render_gate.describe(dark)}"
        now = render_gate.onsets(dark)
        if now in trips:
            raise RenderError(f"{what}; deterministic: two renders darken the same surfaces from the same keyframes, "
                              "so the scene does it, not Blender: look at those frames")
        if attempt == RENDER_ATTEMPTS:
            raise RenderError(f"{what}; {attempt} renders went dark, at different keyframes each time: the transient "
                              "fault, too often to retry away: look at those frames and the GPU")
        first_trip = first_trip or now
        trips.append(now)
        print(f"render: {what}; rendering the script again in a fresh Blender", file=sys.stderr, flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
    raise AssertionError("unreachable: the last attempt returns or raises")


def encode_frames(paths: ScriptPaths, tmp: Path, frames: tuple[int, int], *, isp: IspSpec | None = None) -> list[dict]:
    """The raw passes in `tmp` -> the script's files, the frame logs back. Per frame: the camera model develops
    `%06d.linear.npy` with the script's camera (`ROBOTS[robot].camera`, the same as its `robot.json`), its `IspSpec`
    replaced by `isp` when one is given (look-dev only), into `sensors/rgb`; depth goes through `cut_depth` with
    `sensors/camera.json`'s near and far, then into that file's unit; the index goes through `index_to_u16`; both pass
    through `self_filter` into their PNGs, both under `private/<sid>/`. Depth and the index never see the camera model
    (D9)."""
    tl, cam = load(paths.timeline), Camera.load(paths.camera)
    camera, logs = _camera(tl, cam, isp), []
    for k in range(*frames):
        logs.append(_develop_frame(paths, tl, camera, k, np.load(tmp / LINEAR.format(k))))
        depth = depth_to_units(cut_depth(np.load(tmp / f"{k:06d}.depth.npy"), cam.near_m, cam.far_m), cam.depth_unit)
        depth, index = self_filter(depth, index_to_u16(np.load(tmp / f"{k:06d}.index.npy")))
        write_png16(paths.depth(k), depth)
        write_png16(paths.instances(k), index)
    return logs


# -- the log ---------------------------------------------------------------------------------------------------------
def _developed_with(tl: Timeline, isp: IspSpec | None, develop: list[dict], linear_kept: Path | None) -> dict:
    values = _isp(tl, isp)
    return {"isp": {**asdict(values), "exposure_ev": list(values.exposure_ev)},
            "isp_override": values != ROBOTS[tl.episode.robot].camera.isp,
            "linear_kept": None if linear_kept is None else str(linear_kept), "develop": develop}


def write_log(paths: ScriptPaths, meta: dict, frames: tuple[int, int], wall_s: float, *, develop: list[dict],
              isp: IspSpec | None = None, linear_kept: Path | None = None, gate: dict | None = None) -> dict:
    """`render_log.json`: Blender's own record of the render, then the unit and the near clip, and how the frames were
    developed (`isp`, whether it is an override of the robot's, where the linear passes were kept, the frame logs) and
    `gate`: what the dark-surface gate saw (`_run_gated`)."""
    per = [f["total_s"] for f in meta["frames"]]
    steady = per[1:] or per
    tl = load(paths.timeline)
    log = {"blender": meta["blender"], "device": meta["device"], "samples": meta["samples"], "fast": meta["fast"],
           "resolution": [meta["width"], meta["height"]], "frames": list(frames), "n_frames": len(per),
           "seconds_per_frame": per, "first_frame_s": per[0], "mean_steady_s": round(statistics.fmean(steady), 4),
           "median_steady_s": round(statistics.median(steady), 4), "wall_s": round(wall_s, 2),
           "use_gpu_subdivision": meta["use_gpu_subdivision"], "seed": meta["seed"],
           "appearance": meta.get("appearance", "placeholder"), "cycles": meta.get("cycles"),
           "lighting": meta.get("lighting"), "unit_lux": meta["unit_lux"], "linear_clip": meta["linear_clip"],
           "clip_start_m": meta["clip_start_m"], **_developed_with(tl, isp, develop, linear_kept), **(gate or {})}
    paths.render_log.write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8")
    return log


def render_binding_problems(paths: ScriptPaths) -> list[str]:
    """What is wrong with the script's render, each problem a sentence that says what to do; [] when nothing is. Pure:
    it reads the log and the timeline and never Blender. Named: no render log (the script has not been rendered); frames
    that are not the whole script (a partial render)."""
    if not paths.render_log.exists():
        return ["no render log: the script has not been rendered (stbench render)"]
    log, tl = json.loads(paths.render_log.read_text(encoding="utf-8")), load(paths.timeline)
    if list(log.get("frames", ())) != [0, tl.n_keyframes]:
        return [f"the render log covers keyframes {log.get('frames')} of {tl.n_keyframes}: a partial render; "
                "render every keyframe"]
    return []


# -- render and develop --------------------------------------------------------------------------------------------
def _drop_perceive_outputs(paths: ScriptPaths) -> None:
    """Delete what `perceive` made: it describes the frames about to be replaced."""
    for p in (paths.visibility, paths.perception_ids, paths.oracle):
        p.unlink(missing_ok=True)


def render(paths: ScriptPaths, fast: bool = False, frames: str | None = None, appearance: str = "placeholder",
           library: Path | None = None, samples: int | None = None, manifest: Path | None = None, *,
           keep_linear: Path | None = None, isp: IspSpec | None = None) -> dict:
    """Render the script (or the keyframes in `frames`) and encode its images. In assets mode the manifest (default: the
    packaged one) and the library (default: the one built for it) must pass `asset_render.require_library`. The script's
    `perceive` outputs are deleted first: they describe the frames being replaced. With `keep_linear` the linear passes
    are moved into that folder before the temporary one is deleted; with `isp` the frames are developed with those
    values instead of the robot's (look-dev: the log says so). Before anything is encoded `render_gate` checks the raw
    passes: a render in which a surface went dark is made again (`_run_gated`: up to `RENDER_ATTEMPTS` runs, an error at
    once when two runs darken the same surface from the same keyframe)."""
    t0 = time.perf_counter()
    if appearance not in APPEARANCES:
        raise RenderError(f"the appearance must be one of {APPEARANCES}, got {appearance!r}")
    tl = load(paths.timeline)
    span = parse_frames(frames, tl.n_keyframes)
    checked = None
    if appearance == "assets":
        try:
            checked = asset_render.require_library(manifest, library)
        except AssetError as e:
            raise RenderError(f"{e}\nnot rendering with assets: there is no fallback to placeholders") from e
    _drop_perceive_outputs(paths)
    if span == (0, tl.n_keyframes):   # a whole render replaces the depth: the old place (pre-0.6.0) is stale and a leak
        shutil.rmtree(depth_before_0_6_0(paths), ignore_errors=True)
    tmp = paths.private / TMP_NAME
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    paths.mkdirs()
    try:
        cmd = blender_command(paths, tmp, span, fast, samples, appearance, checked and checked.blend,
                              checked and checked.manifest_path)
        meta, gate = _run_gated(cmd, tmp, span, paths.id)
        logs = encode_frames(paths, tmp, span, isp=isp)
        imap = json.loads((tmp / "instance_map.json").read_text(encoding="utf-8"))
        paths.instance_map.write_text(json.dumps(imap, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        if keep_linear is not None:
            Path(keep_linear).mkdir(parents=True, exist_ok=True)
            for k in range(*span):
                shutil.move(str(tmp / LINEAR.format(k)), str(Path(keep_linear) / LINEAR.format(k)))
        return write_log(paths, meta, span, time.perf_counter() - t0, develop=logs, isp=isp, linear_kept=keep_linear,
                         gate=gate)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def develop(paths: ScriptPaths, linear: Path, isp: IspSpec | None = None) -> dict:
    """Develop the linear passes in `linear` again, into the script's JPEGs, and rewrite the log's frame logs (and
    `isp`, `isp_override`, `linear_kept`). Depth, the index and everything else the render made stay. Refuses a script
    with no render log, and a `linear` folder that lacks a pass for a rendered keyframe. The `perceive` outputs are
    deleted: they read the JPEGs."""
    tl = load(paths.timeline)
    if not paths.render_log.exists():
        raise RenderError("no render log: the script has not been rendered (stbench render), so there are no frames to "
                          "develop again")
    log = json.loads(paths.render_log.read_text(encoding="utf-8"))
    lo, hi = log["frames"]
    missing = [k for k in range(lo, hi) if not (Path(linear) / LINEAR.format(k)).exists()]
    if missing:
        more = "..." if len(missing) > 5 else ""
        raise RenderError(f"{linear} holds no linear pass for keyframes {missing[:5]}{more}; render with --keep-linear")
    camera = _camera(tl, Camera.load(paths.camera), isp)
    _drop_perceive_outputs(paths)
    logs = [_develop_frame(paths, tl, camera, k, np.load(Path(linear) / LINEAR.format(k))) for k in range(lo, hi)]
    log |= _developed_with(tl, isp, logs, Path(linear))
    paths.render_log.write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8")
    return log


# -- commands ----------------------------------------------------------------------------------------------------
def _isp_override(args: argparse.Namespace, paths: ScriptPaths) -> IspSpec | None:
    """The `--isp` file's values over the script's robot's own, or None without the flag."""
    if args.isp is None:
        return None
    return load_isp(args.isp, ROBOTS[load(paths.timeline).episode.robot].camera.isp)


def _render(args: argparse.Namespace) -> int:
    paths = ScriptPaths.from_dir(args.script_dir)
    try:
        log = render(paths, args.fast, args.frames, args.appearance, args.library, args.samples, args.manifest,
                     keep_linear=args.keep_linear, isp=_isp_override(args, paths))
    except RenderError as e:
        print(f"render: {e}", file=sys.stderr)
        return 2
    dev = ", ".join(log["device"]["names"])
    print(f"{paths.id}: rendered {log['n_frames']} frames ({log['appearance']}) on {log['device']['type']} ({dev}) at "
          f"{log['samples']} spp: {log['mean_steady_s']:.3f} s/frame steady, first {log['first_frame_s']:.2f} s, "
          f"wall {log['wall_s']:.1f} s")
    return 0


def _develop(args: argparse.Namespace) -> int:
    paths = ScriptPaths.from_dir(args.script_dir)
    try:
        log = develop(paths, args.linear, _isp_override(args, paths))
    except RenderError as e:
        print(f"develop: {e}", file=sys.stderr)
        return 2
    evs = [f["ev"] for f in log["develop"]]
    print(f"{paths.id}: developed {len(evs)} frames from {args.linear} (isp override: {log['isp_override']}), median "
          f"EV {statistics.median(evs):+.2f}; run `stbench perceive` again")
    return 0


def add_appearance_flags(parser: argparse.ArgumentParser) -> None:
    """`--appearance` and `--samples`, shared by `render` and `build`."""
    parser.add_argument("--appearance", choices=APPEARANCES, default="placeholder",
                        help="placeholder (M1 geometry) or assets (the photoreal library; refuses without one)")
    parser.add_argument("--samples", type=int, help="RGB samples per pixel (default: 32 placeholder, 128 assets)")


def register(sub) -> None:
    r = sub.add_parser("render", help="render a script in Blender: RGB, depth and instance images")
    r.add_argument("script_dir", type=Path, help="DATASET/scripts/<sid> or DATASET/private/<sid>")
    r.add_argument("--fast", action="store_true", help="fewer RGB samples (depth and ids are unchanged)")
    r.add_argument("--frames", help="keyframes A:B (half-open), default all")
    add_appearance_flags(r)
    r.add_argument("--library", type=Path, help="the asset library .blend (default: the one built for the manifest)")
    r.add_argument("--manifest", type=Path, help="an asset manifest (default: the packaged one)")
    r.add_argument("--keep-linear", type=Path, metavar="DIR", help="keep the linear passes in DIR (`develop` reads)")
    r.add_argument("--isp", type=Path, metavar="FILE", help="develop with these IspSpec fields instead of the robot's "
                                                           "(look-dev; the log says so)")
    r.set_defaults(func=_render)
    d = sub.add_parser("develop", help="develop the kept linear passes into the script's JPEGs again (then: perceive)")
    d.add_argument("script_dir", type=Path, help="DATASET/scripts/<sid> or DATASET/private/<sid>")
    d.add_argument("--linear", type=Path, required=True, metavar="DIR", help="the folder `render --keep-linear` filled")
    d.add_argument("--isp", type=Path, metavar="FILE", help="develop with these IspSpec fields instead of the robot's")
    d.set_defaults(func=_develop)
