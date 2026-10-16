"""How a picture's noise is measured, in `stbench realism report` (M1's R.2): pure arrays in, numbers
out.

The noise is read from flat patches: 16 x 16 squares of an 8-bit RGB picture whose sixteen 4 x 4 block means of luma
differ by less than `FLAT_LEVELS` levels (a wall, a floor, a sky: nothing in them but the sensor). Per patch:
- `sigma`, the standard deviation of luma, in 8-bit levels, of the pixels from their 4 x 4 block means (the report's
  first noise measure, unchanged since the first look-dev, so the rounds stay comparable);
- `chroma`, the colour noise over the luma noise: the standard deviations of Cb and Cr (Rec.709 weights on the encoded
  values) averaged as a root mean square, over the luma's, of the patch less the plane fitted to it;
- `lag1`, how far the noise reaches: the correlation of that residual with its right and its lower neighbour. White
  noise is 0; a denoiser's smear, 0.4 to 0.7 in the robot's own frames (the pre-screen's reading of the j7).
A patch that holds a pixel of `mask` (the flat fills of a reference, `FILL`) is left out whole: a fill has no noise and
would pull the figure down. Nothing here reads a file."""

from __future__ import annotations

import statistics
from collections.abc import Iterable

import numpy as np

PATCH, FLAT_LEVELS = 16, 2.0
FILL = (140, 138, 140)  # the flat block the robot maker's help pages leave where highlights blew out (exact RGB)
LUMA = np.array([0.2126, 0.7152, 0.0722])  # the Rec.709 weights, on the encoded values: a display's luma
CB, CR = 1.8556, 1.5748  # Rec.709: Cb = (B - Y) / 1.8556, Cr = (R - Y) / 1.5748
BANDS = ((0.0, 0.15), (0.15, 0.35), (0.35, 0.65), (0.65, 1.001))  # mean sRGB luma of a patch, for the tables below


def fill_mask(rgb: np.ndarray) -> np.ndarray:
    """The pixels of an 8-bit RGB picture that are exactly the flat fill, (h, w) bool."""
    return (rgb[..., :3] == np.array(FILL, dtype=rgb.dtype)).all(axis=-1)


def _patches(a: np.ndarray) -> np.ndarray:
    """An (h, w, ...) array cut to (rows, columns, PATCH, PATCH, ...) whole patches."""
    h, w = (a.shape[0] // PATCH) * PATCH, (a.shape[1] // PATCH) * PATCH
    rest = a.shape[2:]
    cut = a[:h, :w].reshape(h // PATCH, PATCH, w // PATCH, PATCH, *rest)
    return np.moveaxis(cut, 2, 1)


def _plane_residual(x: np.ndarray) -> np.ndarray:
    """(..., PATCH, PATCH) less the plane fitted to each patch (its mean and the slopes along both axes)."""
    c = np.arange(PATCH) - (PATCH - 1) / 2
    mean = x.mean(axis=(-2, -1), keepdims=True)
    centred = x - mean
    dy = (centred * c[:, None]).sum(axis=(-2, -1), keepdims=True) / (c ** 2).sum() / PATCH
    dx = (centred * c[None, :]).sum(axis=(-2, -1), keepdims=True) / (c ** 2).sum() / PATCH
    return centred - dy * c[:, None] - dx * c[None, :]


def _lag1(r: np.ndarray) -> np.ndarray:
    """The correlation of a patch's residual with its right and lower neighbours, pooled over (..., PATCH, PATCH)."""
    var = (r ** 2).sum(axis=(-2, -1))
    cov = (r[..., :, :-1] * r[..., :, 1:]).sum(axis=(-2, -1)) + (r[..., :-1, :] * r[..., 1:, :]).sum(axis=(-2, -1))
    pairs = 2 * PATCH * (PATCH - 1)
    return np.where(var > 0, cov / pairs / np.maximum(var / PATCH ** 2, 1e-30), 0.0)


def patch_stats(rgb: np.ndarray, mask: np.ndarray | None = None) -> list[dict]:
    """The flat patches of an 8-bit RGB picture as `{mean, sigma, chroma, lag1}` (see the module's docstring; `mean` is
    the patch's mean luma, 0 to 1). Patches that hold a pixel of `mask` are dropped."""
    rgb = rgb[..., :3].astype(np.float64)
    luma = (rgb / 255.0) @ LUMA  # 0 to 1
    blocks = _patches(luma)
    n = PATCH // 4
    sub = blocks.reshape(*blocks.shape[:2], n, 4, n, 4).mean(axis=(3, 5))  # the sixteen 4 x 4 block means of a patch
    keep = sub.std(axis=(2, 3)) * 255.0 < FLAT_LEVELS
    if mask is not None:
        keep &= ~_patches(mask).any(axis=(2, 3))
    if not keep.any():
        return []
    y = blocks[keep] * 255.0  # (patches, PATCH, PATCH), in 8-bit levels
    resid = y - np.repeat(np.repeat(sub[keep] * 255.0, 4, axis=1), 4, axis=2)
    sigma = resid.std(axis=(1, 2))
    r = _plane_residual(y)
    chans = _patches(rgb)[keep]  # (patches, PATCH, PATCH, 3)
    cb = _plane_residual((chans[..., 2] - y) / CB)
    cr = _plane_residual((chans[..., 0] - y) / CR)
    luma_sd = r.std(axis=(1, 2))
    chroma = np.sqrt((cb.std(axis=(1, 2)) ** 2 + cr.std(axis=(1, 2)) ** 2) / 2) / np.maximum(luma_sd, 1e-9)
    lag1 = _lag1(r)
    return [{"mean": float(m), "sigma": float(s), "chroma": float(c), "lag1": float(k)}
            for m, s, c, k in zip(blocks[keep].mean(axis=(1, 2)), sigma, chroma, lag1, strict=True)]


def table_by_band(patches: Iterable[dict], bands: tuple[tuple[float, float], ...] = BANDS) -> dict[str, dict]:
    """Per luma band of `bands`, the number of patches and the median `sigma`, `chroma` and `lag1` of those in it (None
    for a band with none), as `{"0-0.15": {patches, sigma, chroma, lag1}, ...}`."""
    rows = list(patches)
    out = {}
    for lo, hi in bands:
        mine = [p for p in rows if lo <= p["mean"] < hi]
        out[f"{lo:g}-{min(hi, 1.0):g}"] = {"patches": len(mine)} | {
            k: (float(statistics.median(p[k] for p in mine)) if mine else None) for k in ("sigma", "chroma", "lag1")}
    return out


def overall(patches: Iterable[dict]) -> dict:
    """Every patch in one table row: the number and the medians of `sigma`, `chroma` and `lag1`."""
    return table_by_band(patches, ((0.0, 1.001),))["0-1"]
