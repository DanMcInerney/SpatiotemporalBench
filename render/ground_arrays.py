# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The numbers of the downtown's worn asphalt: two tileable arrays (`asphalt_arrays`, `macro_array`), numpy only.

Plain Python (no `bpy`), so a test can read them without Blender; `ground_images.py` turns them into images. See there
for what they hold and why they are images. Made from fixed seeds: the same numbers every time, on every machine.
"""

from __future__ import annotations

import numpy as np

TILE_M, TILE_PX = 64.0, 1024
MACRO_M, MACRO_PX = 41.3, 512
PATCH_CELL_M = (6.4, 4.0)  # whole numbers of cells in the 64 m tile: 10 x 16
TRENCH_CELL_M = (32.0, 3.2)  # 2 x 20 cells
BLOCK_M = 32.0
SEED = 20261005
DARK, PALE = np.array((0.052, 0.052, 0.056)), np.array((0.115, 0.113, 0.112))  # linear: the asphalt's two ends
TAR = np.array((0.018, 0.018, 0.02))
CRACK = np.array((0.014, 0.014, 0.016))


def _noise(rng: np.random.Generator, n: int, cells: int) -> np.ndarray:
    """Tileable smooth value noise over an n x n image with `cells` cells a side (0 to 1)."""
    grid = rng.random((cells, cells))
    t = np.arange(n) * cells / n
    i0 = np.floor(t).astype(int)
    f = t - i0
    s = f * f * (3.0 - 2.0 * f)
    i1 = (i0 + 1) % cells
    sx, sy = s[None, :], s[:, None]
    top = grid[i0[:, None], i0[None, :]] * (1 - sx) + grid[i0[:, None], i1[None, :]] * sx
    bottom = grid[i1[:, None], i0[None, :]] * (1 - sx) + grid[i1[:, None], i1[None, :]] * sx
    return top * (1 - sy) + bottom * sy


def _fractal(rng: np.random.Generator, n: int, cells: int, octaves: int, rough: float = 0.5) -> np.ndarray:
    """Tileable fractal noise (0 to 1): `octaves` of `_noise`, each twice the cells and `rough` times the weight."""
    out, weight, total = np.zeros((n, n)), 1.0, 0.0
    for k in range(octaves):
        out += weight * _noise(rng, n, cells * 2 ** k)
        total += weight
        weight *= rough
    return out / total


def _patches(rng: np.random.Generator, n: int, cells: tuple[int, int], share: float, size_u, size_v, edge_px: int
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rectangular repairs in a grid of `cells` (columns, rows) cells over the tile: in `share` of the cells one of
    `size_u` x `size_v` (fractions of the cell) at a random place. Returns (inside, border, tone) per pixel: True inside
    a patch, True on its `edge_px` wide edge, and the patch's random 0..1."""
    cx, cy = cells
    exists, r = rng.random((cy, cx)) < share, rng.random((6, cy, cx))
    w = size_u[0] + (size_u[1] - size_u[0]) * r[0]
    h = size_v[0] + (size_v[1] - size_v[0]) * r[1]
    a = 0.03 + (0.94 - w) * r[2]
    c = 0.03 + (0.94 - h) * r[3]
    ix, iy = (np.arange(n) * cx // n)[None, :], (np.arange(n) * cy // n)[:, None]
    u = ((np.arange(n)[None, :] * cx / n) % 1.0) + 0 * iy
    v = ((np.arange(n)[:, None] * cy / n) % 1.0) + 0 * ix
    ex, ey = edge_px * cx / n, edge_px * cy / n

    def box(pad_u: float, pad_v: float) -> np.ndarray:
        return ((u > a[iy, ix] - pad_u) & (u < a[iy, ix] + w[iy, ix] + pad_u) & (v > c[iy, ix] - pad_v)
                & (v < c[iy, ix] + h[iy, ix] + pad_v) & exists[iy, ix])

    inside = box(0.0, 0.0)
    return inside, box(ex, ey) & ~inside, r[4][iy, ix]


def macro_array() -> np.ndarray:
    """The macro tone (a factor around 1), `MACRO_PX` square: slow tonal variation that breaks the tile's repeat."""
    return 0.78 + 0.44 * _fractal(np.random.default_rng(SEED + 7), MACRO_PX, 5, 3, 0.6)


def asphalt_arrays() -> tuple[np.ndarray, np.ndarray]:
    """(colour (n, n, 3) linear, roughness (n, n)) of the tile."""
    n = TILE_PX
    rng = np.random.default_rng(SEED)
    base = 0.5 * _fractal(rng, n, 6, 2) + 0.3 * _fractal(rng, n, 48, 3, 0.6) + 0.2 * _fractal(rng, n, 256, 2, 0.7)
    t = np.clip((base - 0.30) / 0.40, 0.0, 1.0)[..., None]
    col, rough = DARK + (PALE - DARK) * t, np.full((n, n), 0.9)
    pc = (round(TILE_M / PATCH_CELL_M[0]), round(TILE_M / PATCH_CELL_M[1]))
    edge = 1  # a sealed edge is 5 cm: one pixel here
    layers = [_patches(rng, n, pc, 0.16, (0.25, 0.7), (0.25, 0.7), edge)]
    # trench scars: long thin cells along x, and the same turned a quarter where a 32 m block says so
    tx = (round(TILE_M / TRENCH_CELL_M[0]), round(TILE_M / TRENCH_CELL_M[1]))
    along_x = _patches(rng, n, tx, 0.3, (0.5, 0.95), (0.12, 0.2), edge)
    along_y = [a.T for a in _patches(np.random.default_rng(SEED + 1), n, tx, 0.3, (0.5, 0.95), (0.12, 0.2), edge)]
    metres = np.arange(n) * TILE_M / n
    block = ((metres[None, :] // BLOCK_M + metres[:, None] // BLOCK_M) % 2) > 0
    layers.append(tuple(np.where(block, y, x) for x, y in zip(along_x, along_y, strict=True)))
    for (inside, border, tone), (pale, dark) in zip(layers, ((1.3, 0.65), (1.12, 0.8)), strict=True):  # scars: subtler
        shade = np.where(tone > 0.45, pale, dark)[..., None]
        col = np.where(inside[..., None], col * shade, col)
        col = np.where(border[..., None], TAR, col)
        rough = np.where(inside, 0.78, rough)
        rough = np.where(border, 0.4, rough)
    ridge = np.abs(_fractal(rng, n, 13, 4, 0.6) - 0.6)
    old = np.clip(_fractal(rng, n, 6, 1) * 8.0 - 4.7, 0.0, 1.0)
    crack = 0.7 * np.clip(1.0 - ridge / 0.005, 0.0, 1.0) * old  # a soft line a few pixels wide: no stair-steps
    col = col * (1.0 - crack[..., None]) + CRACK * crack[..., None]
    oil = np.clip(_fractal(rng, n, 13, 4, 0.55) * 9.0 - 5.9, 0.0, 1.0)
    col, rough = col * (1.0 - 0.5 * oil[..., None]), rough * (1.0 - oil) + 0.5 * oil
    return col, rough
