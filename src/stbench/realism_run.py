"""`stbench realism probe` (M1's R.2): the measuring renders of one script.

Over the script's own world (`realism_world.rebuild_world`) it makes, into `OUT/<script id>/` (`realism_data.
probe_folder`: `OUT` itself where it is already named by the id), the folder `report --probe OUT` and `sheet --probe
OUT` read:
- `standing/`, a dataset of its own with the standing camera's script (the helper's 1.30 m camera, `--poses` seeded
poses in each room on its free floor), rendered with the assets appearance at `--fast` settings through
`render_cli.render`; and `standing_rgb/`, a script of its first `--rgb` frames, in different rooms, at release
settings, for the sheet (a render log must cover all of its script, so they are a script of their own);
- `index_script/` and `index_standing/`, `%06d.npz` of every frame of the script and of the standing script: `index`
(uint16, the dressing's pieces from `realism_report.DEBUG_BASE` up, entities under it) and `depth` (float16 metres),
rendered at one sample by `render/realism_probe.py` (index and depth only: they are never a dataset's) and
`dressing_index.json` (index -> class);
- `cards.json`, the grey cards (lamps only, daylight only, the worktop) of the script's scene;
- `meta.json`: what was done (the standing script's id, the room of each standing frame).
The Blender runs happen one after another in this process; take the machine's Blender lock first.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import asset_render, blender_run, render_cli
from . import episode as ep
from .case import ScriptPaths
from .mapspec import load_map
from .realism_data import probe_folder
from .realism_world import rebuild_world, standing_timeline
from .seeding import make_rng
from .timeline import Ego, load

TMP = "_index_tmp"


def _index_frames(paths: ScriptPaths, n: int, checked, dest: Path, probe: Path) -> None:
    """`index` and `depth` of the script's first `n` keyframes into `dest/%06d.npz` by `realism_probe.py --task index`,
    and `dressing_index.json` beside `dest`."""
    tmp = probe / TMP
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    cmd = render_cli.blender_command(paths, tmp, (0, n), fast=True, samples=1, appearance="assets",
                                     library=checked.blend, manifest=checked.manifest_path)
    cmd[cmd.index("--python") + 1] = str(blender_run.script("realism_probe.py"))
    render_cli.run_blender([*cmd, "--task", "index"], paths.id)
    try:
        dest.mkdir(parents=True, exist_ok=True)
        for k in range(n):
            index = np.load(tmp / f"{k:06d}.index.npy")
            depth = np.minimum(np.load(tmp / f"{k:06d}.depth.npy"), 60000.0)
            np.savez_compressed(dest / f"{k:06d}.npz", index=index.astype(np.uint16), depth=depth.astype(np.float16))
        shutil.copy(tmp / "dressing_index.json", probe / "dressing_index.json")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def probe(script_dir: str | Path, out: str | Path, poses: int = 30, rgb: int = 4) -> dict:
    """The probe of one script; returns `meta.json`'s content."""
    t0 = time.perf_counter()
    paths = ScriptPaths.from_dir(script_dir)
    tl = load(paths.timeline)
    m = load_map(tl.episode.map)
    world = rebuild_world(tl, m)
    checked = asset_render.require_library(None, None)
    folder = probe_folder(out, paths.id)
    stl, rooms = standing_timeline(tl, m, world, poses, make_rng(m.name, tl.episode.seed, "realism", "standing"))
    standing = ep.write_script(folder / "standing", stl)
    render_cli.render(standing, fast=True, appearance="assets")
    rgb = min(rgb, stl.n_keyframes)
    rgb_id = None
    if rgb:  # the first frames again at release settings, as a script of their own: a log must cover all of its script
        first = replace(stl, ego=Ego([stl.keyframe(k) for k in range(rgb)], [replace(stl.ego.tasks[0], k1=rgb)], []))
        rgb_id = ep.write_script(folder / "standing_rgb", first).id
        render_cli.render(ScriptPaths(folder / "standing_rgb", rgb_id), fast=False, appearance="assets")
    _index_frames(paths, tl.n_keyframes, checked, folder / "index_script", folder)
    _index_frames(standing, stl.n_keyframes, checked, folder / "index_standing", folder)
    blender_run.run(blender_run.command(
        "realism_probe.py", "--task", "cards", "--root", str(paths.root), "--private", str(paths.private), "--src",
        str(blender_run.SRC_DIR), "--manifest", str(checked.manifest_path), "--library", str(checked.blend), "--out",
        str(folder / "cards.json")), f"reading the grey cards of {paths.id}")
    meta = {"script": paths.id, "standing_id": standing.id, "standing_rgb_id": rgb_id,
            "poses_per_room": poses, "rgb": rgb, "rooms": rooms,
            "frames": {"script": tl.n_keyframes, "standing": stl.n_keyframes},
            "wall_s": round(time.perf_counter() - t0, 1)}
    (folder / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return meta
