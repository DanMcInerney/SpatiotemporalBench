# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The mean luminance of a texture over the part of it a surface wears. Numpy only, no `bpy`, so a free test can run it.

A Google Scanned Object's texture is an atlas: 50 to 72% of its texels are black and no triangle maps to them. The
`stbench_tint` group divides a pixel's luminance by the texture's mean, so a mean taken over the whole image is far
too low and the visible surface shows 2.4 to 4 times its palette colour. The mean that keeps the promise ("the surface
has the palette colour's mean albedo") is the surface's: this module draws points on the surface, reads the texel each
one lands on, and averages them.
"""

from __future__ import annotations

import numpy as np


def surface_mean(lum: np.ndarray, uv: np.ndarray, area: np.ndarray, n: int = 20000, seed: int = 0) -> float:
    """Area-weighted mean of a linear-luminance texture `lum` (h, w) over a surface. `lum[0]` is the row at v = 0, as
    Blender stores image pixels (bottom row first). `uv` (t, 3, 2) are the UV corners of t triangles and `area` (t,)
    their 3-D areas. `n` points are drawn uniformly on the surface (a triangle by area, then a barycentric point in
    it), mapped through the UVs to the nearest texel (u and v repeat, as the image node's default extension does) and
    their luminance averaged. Deterministic in `seed`."""
    lum, uv, area = np.asarray(lum, np.float64), np.asarray(uv, np.float64), np.asarray(area, np.float64)
    if lum.ndim != 2 or uv.ndim != 3 or uv.shape[1:] != (3, 2) or area.shape != uv.shape[:1]:
        raise ValueError(f"lum {lum.shape}, uv {uv.shape} and area {area.shape} do not describe a textured surface")
    total = float(area.sum())
    if not total > 0.0:
        raise ValueError("the surface has no area")
    rng = np.random.default_rng(seed)
    tri = rng.choice(len(area), size=n, p=area / total)
    a, b = np.sqrt(rng.random(n)), rng.random(n)  # uniform in a triangle: weights (1 - a, a (1 - b), a b)
    weights = np.stack([1.0 - a, a * (1.0 - b), a * b], axis=1)
    points = np.einsum("nk,nkd->nd", weights, uv[tri])
    h, w = lum.shape
    col = np.floor(points[:, 0] * w).astype(np.int64) % w
    row = np.floor(points[:, 1] * h).astype(np.int64) % h
    return float(lum[row, col].mean())
