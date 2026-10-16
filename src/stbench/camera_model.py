"""A robot's camera as a processor: it develops the renderer's scene-linear image into the JPEG the robot would store.

Blender writes the Combined pass in one unit (`render/cycles_setup.UNIT_LUX`: a grey card of albedo 0.18 under 240 lx
reads 0.18) and no JPEG. `develop` takes that image and does what a camera's optics, sensor and image processor do, with
the values of the robot's own `IspSpec` (`robot.json`). One function a stage, in the order a real camera has them:

1. `vignette_gain`: the optics. Light falls off as cos^4 of the angle from the axis, so the corners are darker; the
   falloff is part of what the meter sees. `glare`: the optics also scatter a small share of the light over the frame,
   so a window or a lamp lays a veil over the dark around it.
2. `meter`: a centre-weighted log-average of luminance. A log-average is not dragged down by a small bright patch the
   way a mean is: a tenth of the frame at 200 times the room moves it 0.8 EV, where a mean would move it 4.4 EV.
3. `exposure`: the EV that takes the metered value to the target. The exposure time takes it up to its longest, and the
   analogue gain makes up the rest, so a dim frame is lifted after the noise.
4. `add_noise`: the sensor's noise on the exposed signal, Poisson-Gaussian (variance a x + b) per channel, seeded by
   the frame's key, then `denoise`: what a processor's denoiser leaves of it. It smooths the noise (the luma more
   gently than the colour), so what stays is correlated over a few pixels and coloured less than the sensor made it,
   and it leaves the picture as it is, as an ideal edge-preserving denoiser does: a stated modelling choice, since the
   renderer's image is already clean. A short exposure of a bright scene is clean; the same scene dark and lifted by
   the gain is not.
5. the gain, then `white_balance`: grey-world gains over the middle of the luminance range, each clamped around green
   so that a colour keeps its name whatever the light.
6. `tone_map`: a contrast about mid-grey on the luminance, a shoulder that takes the highlights to white, a gamut
   compression that desaturates what would leave the display's range (so a warm lamp rolls off to white, not to a peach
   disc whose red alone is clipped), a black level (the floor a real camera's darks sit on), then the sRGB transfer;
   `sharpen`: a mild unsharp mask; then 8 bits.
7. `encode_jpeg`: no EXIF, no ICC profile and no comment, so the file carries nothing but the picture.

Depth and the instance pass never come here (D9): they are read from the renderer's own float passes. This module
imports numpy, Pillow and the seeding helper, and no truth module, so the agent side may read it.
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image

from .robots import CameraSpec, IspSpec
from .seeding import make_rng

LUMA = (0.2126, 0.7152, 0.0722)  # Rec. 709 luminance of linear RGB
LUMINANCE_FLOOR = 1e-4  # the meter reads a black pixel as this
WB_PERCENTILES = (5.0, 95.0)  # grey-world averages the pixels between these luminance percentiles
SHARPEN_SIGMA_PX, SHARPEN_TAPS = 1.0, 7  # the unsharp mask's Gaussian blur
CRUSHED_LUMA = 5  # a pixel at or under this 8-bit luma is crushed
# Pillow's subsampling 0 is 4:4:4. At 4:2:0 the encoder averages the colour over 2 x 2 pixels and, with it, the colour
# noise the camera leaves: measured on the vacuum's look-dev frames the colour noise reached a third of the real j7's
# against luma noise (0.2 to 0.3 where the j7's is 0.45 to 0.6), whatever the denoiser left. So the picture keeps its
# colour, and the denoiser (`IspSpec.denoise_chroma_px`) alone sets how coloured the grain is.
JPEG_SUBSAMPLING = 0
# The tone curve (a modelling choice, shared by every robot; a robot's own part is `IspSpec.contrast`).
TONE_PIVOT = 0.18  # the grey card of the linear unit: the one luminance the contrast leaves where it is
SHOULDER_KNEE = 0.75  # the display-linear luminance where the shoulder begins; above it, an exponential approach to 1
# The optics' veil: two Gaussians, each carrying half of the scattered light, with these sigmas as shares of the width.
GLARE_SIGMAS = (0.02, 0.08)
GLARE_DOWNSAMPLE_PX = 4.0  # a veil this wide is smooth: it is blurred on an image averaged down until sigma is this


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec. 709 luminance of the last axis (R, G, B)."""
    return rgb[..., 0] * LUMA[0] + rgb[..., 1] * LUMA[1] + rgb[..., 2] * LUMA[2]


# -- stages ------------------------------------------------------------------------------------------------------
def vignette_gain(width: int, height: int, hfov_deg: float, vignette: float) -> np.ndarray:
    """(H, W) factor 1 - v (1 - cos^4 theta), theta the angle from the optical axis at each pixel's centre, for a
    pinhole with fx = (W / 2) / tan(hfov / 2) and the principal point at the image centre."""
    fx = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    x = np.arange(width) + 0.5 - width / 2
    y = np.arange(height) + 0.5 - height / 2
    cos2 = fx * fx / (fx * fx + x[None, :] ** 2 + y[:, None] ** 2)
    return 1.0 - vignette * (1.0 - cos2 * cos2)


def _upsample(small: np.ndarray, factor: int, height: int, width: int) -> np.ndarray:
    """Bilinear enlargement of an (h, w, C) array by `factor`, cropped to (height, width)."""
    def axis(n_out: int, n_in: int):
        c = np.clip((np.arange(n_out) + 0.5) / factor - 0.5, 0.0, n_in - 1.0)
        i0 = np.floor(c).astype(np.int64)
        return i0, np.minimum(i0 + 1, n_in - 1), c - i0

    y0, y1, ty = axis(height, small.shape[0])
    x0, x1, tx = axis(width, small.shape[1])
    rows = small[y0] * (1 - ty)[:, None, None] + small[y1] * ty[:, None, None]
    return rows[:, x0] * (1 - tx)[None, :, None] + rows[:, x1] * tx[None, :, None]


def wide_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """A Gaussian blur of an (H, W, C) image with `sigma` pixels (edges replicated), for a kernel wider than a few
    pixels: the image is averaged down by sigma / `GLARE_DOWNSAMPLE_PX` (at least 1), blurred there, enlarged again."""
    h, w = img.shape[:2]
    f = max(1, int(sigma // GLARE_DOWNSAMPLE_PX))
    hh, ww = -(-h // f), -(-w // f)
    padded = np.pad(img, ((0, hh * f - h), (0, ww * f - w), (0, 0)), mode="edge")
    small = padded.reshape(hh, f, ww, f, img.shape[2]).mean(axis=(1, 3))
    small = gaussian_blur(small, sigma / f, math.ceil(3 * sigma / f))
    return _upsample(small, f, h, w)


def glare(x: np.ndarray, fraction: float) -> np.ndarray:
    """Veiling glare: (1 - fraction) of each pixel's light stays, and `fraction` of it is scattered over the frame by
    two Gaussians (`GLARE_SIGMAS`, half each) as the optics of a small lens do. Energy is kept, so a flat field is
    unchanged; a window a hundred times the room lifts the dark around it. 0 returns `x` as it is."""
    if fraction == 0:
        return x
    w = x.shape[1]
    scattered = sum(0.5 * wide_blur(x, s * w) for s in GLARE_SIGMAS)
    return (1.0 - fraction) * x + fraction * scattered


def meter(linear: np.ndarray, meter_sigma: float) -> float:
    """exp(sum w log Y / sum w): Y the luminance floored at `LUMINANCE_FLOOR`, w a Gaussian about the image centre with
    sigma `meter_sigma` x the width in pixels."""
    h, w = linear.shape[:2]
    log_y = np.log(np.maximum(luminance(linear), LUMINANCE_FLOOR))
    sigma = meter_sigma * w
    wx = np.exp(-((np.arange(w) + 0.5 - w / 2) ** 2) / (2 * sigma * sigma))
    wy = np.exp(-((np.arange(h) + 0.5 - h / 2) ** 2) / (2 * sigma * sigma))
    weights = wy[:, None] * wx[None, :]
    return float(np.exp((weights * log_y).sum() / weights.sum()))


def exposure(metered: float, isp: IspSpec) -> tuple[float, float, float]:
    """(total EV, exposure-time EV, gain EV): total = log2(target / metered) clamped to [exposure_ev[0],
    exposure_ev[1] + gain_ev]; the exposure time takes up to exposure_ev[1] of it and the gain the rest."""
    total = float(np.clip(math.log2(isp.target / metered), isp.exposure_ev[0], isp.exposure_ev[1] + isp.gain_ev))
    e = min(total, isp.exposure_ev[1])
    return total, e, total - e


def sensor_noise(x: np.ndarray, noise_a: float, noise_b: float, rng: np.random.Generator) -> np.ndarray:
    """The sensor's noise on the exposed signal `x`: N(0, a x + b) per pixel and channel, shot noise and read noise."""
    return rng.standard_normal(x.shape) * np.sqrt(np.maximum(noise_a * x + noise_b, 0.0))


def denoise(noise: np.ndarray, luma_px: float, chroma_px: float) -> np.ndarray:
    """What a denoiser leaves of the sensor's noise (an (H, W, 3) field in linear RGB): its luma (0.2126 R + 0.7152 G +
    0.0722 B) smoothed by a Gaussian of `luma_px` and its colour part (what has no luma) by one of `chroma_px`,
    so the grain left is correlated over a few pixels and colours less than the sensor made it. 0 leaves a part as it
    is. Smoothing the noise and not the picture is a modelling choice: an ideal edge-preserving denoiser."""
    if luma_px == 0 and chroma_px == 0:
        return noise
    y = luminance(noise)[..., None]
    colour = noise - y
    if luma_px:
        y = gaussian_blur(y, luma_px, math.ceil(3 * luma_px))
    if chroma_px:
        colour = gaussian_blur(colour, chroma_px, math.ceil(3 * chroma_px))
    return y + colour


def add_noise(x: np.ndarray, noise_a: float, noise_b: float, rng: np.random.Generator, luma_px: float = 0.0,
              chroma_px: float = 0.0) -> np.ndarray:
    """x plus its sensor noise (`sensor_noise`) as `denoise` leaves it, then at least 0 (the sensor's black)."""
    return np.maximum(x + denoise(sensor_noise(x, noise_a, noise_b, rng), luma_px, chroma_px), 0.0)


def white_balance(x: np.ndarray, wb_clamp: float) -> tuple[np.ndarray, np.ndarray]:
    """Grey-world gains (mean G / mean R, 1, mean G / mean B) over the pixels between the 5th and 95th luminance
    percentiles, each clamped to [1 - wb_clamp, 1 + wb_clamp]. Returns the balanced image and the three gains; a frame
    with no green gets none."""
    lum = luminance(x)
    lo, hi = np.percentile(lum, WB_PERCENTILES)
    mean = x[(lum >= lo) & (lum <= hi)].mean(axis=0)
    gains = np.ones(3)
    if mean[1] > 0:
        with np.errstate(divide="ignore"):
            gains = np.clip(mean[1] / mean, 1.0 - wb_clamp, 1.0 + wb_clamp)
        gains[1] = 1.0
    return x * gains, gains


def srgb_transfer(c: np.ndarray) -> np.ndarray:
    """Linear [0, 1] -> the sRGB encoding."""
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(np.maximum(c, 0.0031308), 1 / 2.4) - 0.055)


def tone_luminance(y: np.ndarray, contrast: float = 1.0) -> np.ndarray:
    """The display-linear luminance (0 to 1) of an exposed luminance `y`: a power `contrast` about `TONE_PIVOT`
    (v = pivot (y / pivot)^contrast, so a grey card keeps its value and a dark thing goes darker and a bright one
    brighter), then a shoulder: v up to `SHOULDER_KNEE` as it is, and above it an exponential approach to 1 with the
    same slope at the knee, so the highlights roll off to white and the curve never falls."""
    v = TONE_PIVOT * (np.maximum(y, 0.0) / TONE_PIVOT) ** contrast
    k = SHOULDER_KNEE
    return np.where(v <= k, v, k + (1.0 - k) * (1.0 - np.exp(-(v - k) / (1.0 - k))))


def tone_map(x: np.ndarray, contrast: float = 1.0, black: float = 0.0) -> np.ndarray:
    """Linear RGB (any positive range) -> display values in [0, 1]: `tone_luminance` on the luminance with the colour
    ratios kept, a gamut compression, a black level, then the sRGB transfer. Where a channel would pass 1 the pixel is
    desaturated towards the grey of its own luminance just far enough to fit, so a bright warm lamp goes white (all
    three channels at the top) and a saturated colour near the top loses chroma, as a processor's highlight handling
    does; the luminance is kept. `black` lifts black to that display-linear value and leaves white at 1."""
    y = np.maximum(luminance(x), 1e-12)
    w = tone_luminance(y, contrast)[..., None]
    rgb = x * (w / y[..., None])
    peak = rgb.max(axis=-1, keepdims=True)
    fit = np.where(peak > 1.0, (1.0 - w) / np.maximum(peak - w, 1e-12), 1.0)
    return srgb_transfer(black + (1.0 - black) * (w + (rgb - w) * fit))


def gaussian_blur(img: np.ndarray, sigma: float, radius: int) -> np.ndarray:
    """A separable Gaussian blur of an (H, W, ...) array, `2 radius + 1` taps (normalised), edges replicated."""
    d = np.arange(-radius, radius + 1)
    k = np.exp(-(d * d) / (2 * sigma * sigma))
    k /= k.sum()
    h, w = img.shape[:2]
    p = np.pad(img, ((radius, radius), (radius, radius)) + ((0, 0),) * (img.ndim - 2), mode="edge")
    rows = sum(k[i] * p[i:i + h] for i in range(len(k)))
    return sum(k[i] * rows[:, i:i + w] for i in range(len(k)))


def sharpen(img: np.ndarray, amount: float) -> np.ndarray:
    """Unsharp mask on an (H, W, 3) image: img + amount (img - blur(img)), the blur a Gaussian of sigma 1 px over 7 taps
    (edges replicated). Not clipped."""
    if amount == 0:
        return img
    return img + amount * (img - gaussian_blur(img, SHARPEN_SIGMA_PX, SHARPEN_TAPS // 2))


def to_8bit(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)


def encode_jpeg(rgb8: np.ndarray, quality: int) -> bytes:
    """A JPEG at `quality` with the JFIF header and nothing else: no EXIF, no ICC profile, no comment. Colour is kept at
    full resolution (4:4:4, `JPEG_SUBSAMPLING`)."""
    buf = io.BytesIO()
    Image.fromarray(rgb8, "RGB").save(buf, format="JPEG", quality=quality, subsampling=JPEG_SUBSAMPLING)
    return buf.getvalue()


# -- the camera --------------------------------------------------------------------------------------------------
def capture(linear: np.ndarray, key: tuple, camera: CameraSpec) -> tuple[np.ndarray, dict]:
    """What the sensor holds: the vignetted, veiled, metered, exposed, noisy and gained signal, in the linear unit and
    before the white balance, and the exposure's log. Negative values (a denoiser's undershoot) read 0."""
    if linear.shape != (camera.height, camera.width, 3):
        raise ValueError(f"the linear image is {linear.shape}, the camera {camera.height} x {camera.width} x 3")
    isp = camera.isp
    x = np.maximum(np.asarray(linear, dtype=np.float64), 0.0)
    x = glare(x * vignette_gain(camera.width, camera.height, camera.hfov_deg, isp.vignette)[..., None], isp.glare)
    metered = meter(x, isp.meter_sigma)
    total, e, g = exposure(metered, isp)
    x = add_noise(x * 2.0**e, isp.noise_a, isp.noise_b, make_rng(*key), isp.denoise_luma_px, isp.denoise_chroma_px)
    return x * 2.0**g, {"ev": total, "exposure_ev": e, "gain_ev": g, "metered": metered}


def develop_8bit(linear: np.ndarray, *, key: tuple, camera: CameraSpec) -> tuple[np.ndarray, dict]:
    """`develop` up to the 8-bit image (H, W, 3): the picture before the JPEG, and the frame's log."""
    x, log = capture(linear, key, camera)
    x, gains = white_balance(x, camera.isp.wb_clamp)
    rgb8 = to_8bit(sharpen(tone_map(x, camera.isp.contrast, camera.isp.black_level), camera.isp.sharpen))
    luma = luminance(rgb8.astype(np.float64))
    log |= {"wb": gains, "clipped": (rgb8 == 255).any(axis=-1).mean(), "crushed": (luma <= CRUSHED_LUMA).mean()}
    return rgb8, {k: [round(float(v), 6) for v in val] if k == "wb" else round(float(val), 6) for k, val in log.items()}


def develop(linear: np.ndarray, *, key: tuple, camera: CameraSpec) -> tuple[bytes, dict]:
    """The JPEG bytes and the frame's log for a scene-linear image (H x W x 3 float, the Combined pass in the linear
    unit), as the robot's camera, `camera`, would store it. `key` seeds the sensor noise: `(map, seed, robot, job,
    "camera", k)` from the timeline, never the random script id, so a rebuilt script develops to the same bytes. The
    log is `{ev, exposure_ev, gain_ev, metered, wb, clipped, crushed}`: the EVs, the metered luminance, the
    white-balance gains, and the shares of pixels with a channel at 255 and with an 8-bit luma of 5 or less."""
    rgb8, log = develop_8bit(linear, key=key, camera=camera)
    return encode_jpeg(rgb8, camera.isp.jpeg_quality), log
