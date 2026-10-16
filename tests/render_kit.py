"""Fake renders for the render pipeline's free tests: the raw passes a Blender run leaves in `render_tmp`, a stand-in
for the run itself, and a vacuum with a small camera (so that developing a frame takes milliseconds)."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from conftest import script_on_map
from stbench.robots import ROBOTS, Robot

VACUUM = ROBOTS["vacuum"]  # the robot as the repository has it, taken before any test patches the table
META = {"blender": "5.2.1", "device": {"type": "OPTIX", "names": ["GPU"]}, "samples": 8, "fast": True,
        "use_gpu_subdivision": False, "seed": 0, "appearance": "placeholder", "unit_lux": 240.0, "linear_clip": 60000.0,
        "clip_start_m": 0.005}


def small_vacuum(monkeypatch, width: int = 160, height: int = 120, **isp_changes) -> Robot:
    """Replace the vacuum in `ROBOTS` (for this test) by one with a `width` x `height` camera and `isp_changes` made to
    its `IspSpec`; scripts made after this are for that spec."""
    camera = replace(VACUUM.camera, width=width, height=height, isp=replace(VACUUM.camera.isp, **isp_changes))
    robot = replace(VACUUM, camera=camera)
    monkeypatch.setitem(ROBOTS, "vacuum", robot)
    return robot


def vacuum_script(n: int = 3):
    """A one-task vacuum script of `n` keyframes under the apartment's coffee table, facing west, on seed 101's
    world."""
    return script_on_map("apartment_a", 101, "vacuum", [(2.40 + 0.05 * k, 4.40, 0.0, math.pi) for k in range(n)])


def write_passes(tmp: Path, k: int, depth: np.ndarray, index: np.ndarray, linear: np.ndarray) -> None:
    """The three raw passes of keyframe `k`: depth in metres and the object index (float32), the linear pass
    (float16)."""
    tmp.mkdir(parents=True, exist_ok=True)
    np.save(tmp / f"{k:06d}.depth.npy", depth.astype(np.float32))
    np.save(tmp / f"{k:06d}.index.npy", index.astype(np.float32))
    np.save(tmp / f"{k:06d}.linear.npy", linear.astype(np.float16))


def synthetic_passes(width: int, height: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(depth, index, linear) of a frame: a floor at 1 to 3 m, an object (index 2) at 1.5 m, a linear image of a dim
    room with a bright patch."""
    rng = np.random.default_rng(seed)
    depth = np.linspace(1.0, 3.0, height)[:, None] * np.ones((height, width))
    index = np.zeros((height, width))
    index[height // 4:height // 2, width // 4:width // 2] = 2
    depth[index == 2] = 1.5
    linear = np.exp(rng.normal(-3.0, 0.4, (height, width, 3)))
    linear[: height // 8, -width // 8:] = 20.0
    return depth, index, linear


def fake_blender(monkeypatch, run_module, camera_size: tuple[int, int], seed: int = 0) -> None:
    """Make `run_module.run_blender` write the passes and the metadata Blender would (the frames of `--frames`)."""
    width, height = camera_size

    def run(cmd: list[str], script_id: str) -> None:
        tmp = Path(cmd[cmd.index("--tmp") + 1])
        lo, hi = (int(x) for x in cmd[cmd.index("--frames") + 1].split(":"))
        for k in range(lo, hi):
            write_passes(tmp, k, *synthetic_passes(width, height, seed + k))
        (tmp / "instance_map.json").write_text(json.dumps({"1": "E0001", "2": "E0002"}), encoding="utf-8")
        frames = [{"k": k, "render_s": 1.0, "total_s": 1.5} for k in range(lo, hi)]
        meta = {**META, "width": width, "height": height, "frames": frames}
        (tmp / "render_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    monkeypatch.setattr(run_module, "run_blender", run)
