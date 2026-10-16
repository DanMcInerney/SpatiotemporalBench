"""Fixed inputs for the camera model's tests: a literal camera (never `ROBOTS`', whose values the freeze changes), and
synthetic scene-linear frames with the structure of a room (a lit floor, a dark underside, a bright window)."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from stbench import camera_model
from stbench.robots import CameraSpec, IspSpec

KEY = ("apartment_a", 101, "vacuum", "vacuum/1", "camera", 0)  # `(map, seed, robot, job, "camera", k)`

# The vacuum's starting values at M0, written out: a test of the camera model's structure is about these numbers, not
# about whatever the freeze later makes of the robot's own. The stages added after (the tone contrast, the black level,
# the veil, the denoiser) are off here: a test of an older stage reads it with nothing else in the way.
LITERAL_ISP = IspSpec(vignette=0.3, noise_a=6e-4, noise_b=1e-5, meter_sigma=0.35, target=0.18,
                      exposure_ev=(-8.0, 2.0), gain_ev=6.0, wb_clamp=0.1, sharpen=0.25, jpeg_quality=92)
# The same camera with every stage on, values written out (a look-dev candidate of the fix round, not the robot's).
FULL_ISP = replace(LITERAL_ISP, vignette=0.65, noise_a=8e-4, noise_b=5e-6, target=0.09, contrast=1.6,
                   black_level=0.0025, glare=0.01, denoise_luma_px=0.8, denoise_chroma_px=1.0)


def literal_camera(width: int = 64, height: int = 48, isp: IspSpec = LITERAL_ISP) -> CameraSpec:
    return CameraSpec(width, height, 80.0, mount_m=(0.13, 0.0, 0.08), pitch_deg=5.0, pitch_min_deg=5.0,
                      pitch_max_deg=5.0, isp=isp)


def synthetic_frame(seed: int, width: int = 64, height: int = 48) -> np.ndarray:
    """An (H, W, 3) scene-linear float image: a floor (lower half) at 0.05 to 0.4 with a warm cast, an underside (upper
    left) at 0.004 and a window (upper right) at 30, with texture."""
    rng = np.random.default_rng(seed)
    img = np.empty((height, width, 3))
    img[:] = np.array([0.30, 0.22, 0.12]) * np.linspace(0.2, 1.0, height)[:, None, None]
    img[: height // 2] = np.array([0.08, 0.07, 0.06])
    img[: height // 2, : width // 3] = 0.004
    img[: height // 2, 2 * width // 3:] = 30.0
    return img * np.exp(rng.normal(0.0, 0.15, img.shape))


def develop_digest() -> str:
    """The SHA-256 of the bytes `develop` makes of fixed frames under two keys and two cameras (every stage off, every
    stage on), joined (a child process computes it too: `develop` depends on nothing of the process it runs in)."""
    h = hashlib.sha256()
    for seed, key, isp in ((1, KEY, LITERAL_ISP), (2, (*KEY[:-1], 1), LITERAL_ISP), (3, KEY, FULL_ISP)):
        h.update(camera_model.develop(synthetic_frame(seed), key=key, camera=literal_camera(isp=isp))[0])
    return h.hexdigest()
