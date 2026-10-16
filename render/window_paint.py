# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The pictures of the downtown's window interiors: the numbers of the window atlas and the shop atlas, numpy only.

Plain Python (no `bpy`), so a test can read them without Blender; `window_atlas.py` turns them into images and says what
they are for. A cell is a pane or a shop bay as seen from the street, painted on the facade's plane: the sky in the
glass over blinds drawn part way down, curtains, an open room, a plainly reflective pane or a lit room, with a frame; or
lit retail, a dim cafe, a shutter, papered glass or a reflective lobby. Made from a fixed seed: the same numbers every
time, on every machine.
"""

from __future__ import annotations

import numpy as np

CELL = 128  # pixels a side of one variant
WINDOWS = (8, 8)  # columns x rows of window variants
SHOPS = (4, 4)
SKY_LOW, SKY_HIGH = np.array((0.030, 0.040, 0.055)), np.array((0.150, 0.190, 0.240))  # linear: the glass at foot, head
ROOMS = np.array(((0.26, 0.23, 0.18), (0.19, 0.23, 0.19), (0.22, 0.22, 0.24), (0.30, 0.20, 0.16), (0.17, 0.19, 0.26)))
CURTAINS = np.array(((0.16, 0.05, 0.05), (0.04, 0.06, 0.14), (0.10, 0.14, 0.08), (0.34, 0.31, 0.24),
                     (0.20, 0.17, 0.14)))
BLINDS = np.array(((0.40, 0.37, 0.31), (0.33, 0.35, 0.37), (0.43, 0.41, 0.35)))
GOODS = np.array(((0.20, 0.13, 0.11), (0.11, 0.14, 0.18), (0.21, 0.18, 0.13), (0.13, 0.16, 0.13), (0.22, 0.22, 0.21),
                  (0.15, 0.14, 0.13)))
FRAME = np.array((0.05, 0.05, 0.055))
SEED = 20261005


def _grid() -> tuple[np.ndarray, np.ndarray]:
    """(wu, wv) of every pixel of a cell: 0 to 1 across and up (row 0 is the foot, as Blender's images are)."""
    c = (np.arange(CELL) + 0.5) / CELL
    return np.meshgrid(c, c)


def _lerp(t: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + (b - a) * t[..., None]


def _paint(img: np.ndarray, mask: np.ndarray, colour: np.ndarray) -> None:
    """Colour the pixels under `mask` (a float 0..1 or bool array) with `colour` (a triple or an image)."""
    m = mask.astype(float)[..., None]
    img[:] = img * (1.0 - m) + np.asarray(colour) * m


def _frame(wu: np.ndarray, wv: np.ndarray, mullion: bool) -> np.ndarray:
    inside = (wu > 0.035) & (wu < 0.965) & (wv > 0.02) & (wv < 0.985)
    mull = (np.abs(wu - 0.5) < 0.015) if mullion else np.zeros_like(wu, dtype=bool)
    return ~inside | mull


def _room(rng: np.random.Generator, wu: np.ndarray, wv: np.ndarray) -> np.ndarray:
    """A room behind the glass: a back wall in one of `ROOMS` (a little off), a lit ceiling, a darker floor, shaded side
    walls and, in some, a desk and a shelf."""
    base = ROOMS[rng.integers(len(ROOMS))] * rng.uniform(0.85, 1.15)
    img = np.broadcast_to(base, wu.shape + (3,)).copy()
    img *= (1.0 + 0.5 * (wv > 0.86) - 0.45 * (wv < 0.14) - 0.12 * ((wu < 0.08) | (wu > 0.92)))[..., None]
    if rng.random() > 0.45:  # a desk or a counter
        x0, width, height = rng.uniform(0.1, 0.6), rng.uniform(0.25, 0.4), rng.uniform(0.14, 0.38)
        _paint(img, (wu > x0) & (wu < x0 + width) & (wv < height), (0.045, 0.04, 0.035))
    if rng.random() > 0.6:  # shelves along a side
        side = wu > 0.7 if rng.random() > 0.5 else wu < 0.3
        _paint(img, side & (wv > 0.3) & (wv < 0.8) & (np.abs((wv * 5.0) % 1.0 - 0.5) < 0.08), (0.06, 0.05, 0.04))
    return img


def _window(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One pane: (colour (n, n, 3), emission (n, n))."""
    wu, wv = _grid()
    kind = rng.random()
    img = _room(rng, wu, wv)
    lit = kind > 0.88
    if lit:
        img = img * 0.5 + np.array((0.46, 0.38, 0.23)) * 0.5
    mirror = 0.72 < kind <= 0.88
    reflect = 0.83 if mirror else 0.28
    img = img * (1.0 - reflect) + _lerp(wv, SKY_LOW, SKY_HIGH) * reflect
    img *= np.clip((1.0 - wv) * 0.45 / 0.12 + 0.55, 0.55, 1.0)[..., None]  # the shadow under the lintel
    if kind < 0.30:  # blinds hang from the head
        drop = rng.uniform(0.25, 0.95)
        colour = BLINDS[rng.integers(len(BLINDS))]
        slats = np.where(((wv * 22.0) % 1.0) < 0.78, 1.0, 0.0)[..., None] * (colour - 0.08) + 0.08
        _paint(img, wv > 1.0 - drop, slats * (0.92 + 0.08 * np.sin(wu * 9.0))[..., None])
    elif kind < 0.52:  # curtains at both sides
        colour = CURTAINS[rng.integers(len(CURTAINS))]
        fold = (0.82 + 0.18 * np.sin(wu * 70.0))[..., None]
        _paint(img, (wu < 0.27) | (wu > 0.73), colour * fold)
    _paint(img, _frame(wu, wv, rng.random() > 0.6), FRAME)
    emission = np.where(lit, 1.0, 0.0) * np.where(_frame(wu, wv, False), 0.0, 1.0)
    return np.clip(img, 0.0, 1.0), emission


def _shop(rng: np.random.Generator, kind: int) -> tuple[np.ndarray, np.ndarray]:
    """One bay of shop glazing: `kind` 0 lit retail with goods, 1 a dim cafe, 2 a rolled-down shutter, 3 papered-over
    glass, 4 a reflective lobby: (colour, emission)."""
    wu, wv = _grid()
    sky = _lerp(wv, SKY_LOW, SKY_HIGH)
    if kind == 0:
        shelves = ((wv * 6.0) % 1.0) < 0.14
        goods = GOODS[rng.integers(len(GOODS), size=(6, 9))]  # six shelves of nine goods
        cells = goods[np.minimum((wv * 6.0).astype(int), 5), np.minimum((wu * 9.0).astype(int), 8)]
        img = cells * 0.8
        _paint(img, shelves, (0.05, 0.04, 0.032))
        _paint(img, (wv < 0.1) | (wv > 0.92), (0.07, 0.06, 0.05))
        img = img * 0.68 + sky * 0.32
        emission = np.ones_like(wu)
    elif kind == 1:
        pendant = (wv > 0.76) & (wv < 0.86) & (((wu * 4.0) % 1.0) < 0.14)
        img = _lerp(wv, np.array((0.05, 0.032, 0.02)), np.array((0.14, 0.10, 0.07)))
        _paint(img, pendant, (0.9, 0.75, 0.45))
        img = img * 0.68 + sky * 0.32
        emission = np.full_like(wu, 0.35)
    elif kind == 2:
        ribs = (0.88 + 0.12 * np.sin(wv * 130.0))[..., None]
        img = np.broadcast_to(np.array((0.16, 0.16, 0.17)), wu.shape + (3,)) * ribs
        emission = np.zeros_like(wu)
    elif kind == 3:
        img = np.broadcast_to(np.array((0.33, 0.32, 0.29)), wu.shape + (3,)).copy()
        _paint(img, np.abs(wv - 0.5) < 0.01, (0.18, 0.18, 0.17))
        emission = np.zeros_like(wu)
    else:
        img = _lerp(wv, np.array((0.03, 0.04, 0.05)), np.array((0.10, 0.12, 0.14)))
        _paint(img, (wv > 0.88) & (wv < 0.96), (0.6, 0.62, 0.6))
        img = img * 0.68 + sky * 0.32
        emission = np.where((wv > 0.88) & (wv < 0.96), 0.0, 0.3)
    frame = _frame(wu, wv, rng.random() > 0.5)
    _paint(img, frame, FRAME)
    return np.clip(img, 0.0, 1.0), emission * np.where(frame, 0.0, 1.0)


def atlas(cols: int, rows: int, cells) -> np.ndarray:
    """An (rows * CELL, cols * CELL, 4) float32 array of `cols` x `rows` cells: `cells(rng, k)` -> (colour, emission)
    for variant k, left to right and bottom to top (row 0 is the foot, as Blender's images are). Alpha is the
    emission."""
    rng = np.random.default_rng(SEED + cols * 100 + rows)
    out = np.zeros((rows * CELL, cols * CELL, 4), dtype=np.float32)
    for k in range(cols * rows):
        colour, emission = cells(rng, k)
        r, c = divmod(k, cols)
        out[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL, :3] = colour
        out[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL, 3] = emission
    return out


def window_array() -> np.ndarray:
    """The window atlas: `WINDOWS` variants of `_window`."""
    return atlas(*WINDOWS, lambda rng, k: _window(rng))


def shop_array() -> np.ndarray:
    """The shop atlas: `SHOPS` variants, the five kinds of `_shop` in turn."""
    return atlas(*SHOPS, lambda rng, k: _shop(rng, k % 5))
