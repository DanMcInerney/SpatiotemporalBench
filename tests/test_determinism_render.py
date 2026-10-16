"""The developing code's and the encoder's output in a second process (`determinism_kit.py`: nothing is recorded, the
two are compared): what they make of fixed inputs. A robot's own values are never in it: the camera here is a literal.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from camera_kit import FULL_ISP, KEY, LITERAL_ISP, literal_camera, synthetic_frame
from determinism_kit import digest
from stbench import camera_model, render_encode

ROOT = Path(__file__).resolve().parents[1]


def render_digest() -> str:
    """The digest of the JPEG and the log of four fixed frames developed under two literal cameras (every stage off,
    every stage on) and four keys, and of the depth and index PNG arrays the encoder makes of fixed passes (cut at the
    vacuum's range, into millimetres, self-filtered)."""
    developed = []
    for seed, isp in ((1, LITERAL_ISP), (2, LITERAL_ISP), (3, LITERAL_ISP), (4, FULL_ISP)):
        jpeg, log = camera_model.develop(synthetic_frame(seed), key=(*KEY[:-1], seed), camera=literal_camera(isp=isp))
        developed.append([jpeg.hex(), log])
    depth = np.array([[0.009, 0.01, 0.5, 2.0], [3.999, 4.0, 4.001, np.nan]], dtype=np.float32)
    index = np.array([[0, 1, 2, render_encode.SELF_ID], [7, render_encode.SELF_ID, 3, 0]], dtype=np.float32)
    counts = render_encode.depth_to_units(render_encode.cut_depth(depth, 0.01, 4.0), "mm")
    counts, ids = render_encode.self_filter(counts, render_encode.index_to_u16(index))
    return digest({"develop": developed, "depth": counts.tolist(), "index": ids.tolist(),
                   "rgb_clip": [render_encode.rgb_clip_m(n) for n in (0.01, 0.2)]})


def test_the_render_digest_is_deterministic():
    """Contract: `develop`'s bytes and logs on four fixed synthetic frames under two literal `CameraSpec`s (not
    `ROBOTS`'), and the encoder's output on fixed depth and index, are the same in two processes (another
    `PYTHONHASHSEED`). Regression caught: a developed frame or an encoded depth that depends on the process it was made
    in."""
    code = "import sys; sys.path.insert(0, 'tests'); import test_determinism_render as t; print(t.render_digest())"
    child = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=True,
                           env={**os.environ, "PYTHONHASHSEED": "12345"})
    assert child.stdout.strip() == render_digest()
