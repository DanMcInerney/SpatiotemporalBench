"""The texture mean behind the colour tint (`render/texel_mean.py`, numpy only): free, model-free, no Blender. The tint
divides a pixel's luminance by this mean, so it decides whether a tinted surface shows its palette colour."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SOURCE = Path(__file__).resolve().parents[1] / "render" / "texel_mean.py"
_spec = importlib.util.spec_from_file_location("texel_mean", _SOURCE)
texel_mean = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(texel_mean)
surface_mean = texel_mean.surface_mean


def test_surface_mean_ignores_texels_no_triangle_maps_to():
    """Contract (VISION: the colour attribute is what the pixels show): the mean the tint divides by is the surface's,
    over the texels its triangles map to, not the image's. A 4 x 4 atlas holds 0.2 in its first column, 0.6 in its
    second and black in the right half; two triangles cover the left half. The surface mean is 0.4 and the whole
    image's is 0.2, so a tint dividing by the image's would show the surface twice as bright. Regression caught: the
    whole-image mean (the Google Scanned Objects atlas bug, whose black background made every scanned kind 2.4 to 4
    times too bright)."""
    lum = np.zeros((4, 4))
    lum[:, 0], lum[:, 1] = 0.2, 0.6
    uv = np.array([[(0, 0), (0.5, 0), (0.5, 1)], [(0, 0), (0.5, 1), (0, 1)]], dtype=float)
    assert abs(lum.mean() - 0.2) < 1e-12
    assert abs(surface_mean(lum, uv, np.array([1.0, 1.0])) - 0.4) < 0.01


def test_surface_mean_weights_triangles_by_their_area():
    """Contract: the mean is area weighted, by the triangles' 3-D areas. Two triangles of equal UV area lie on two
    uniform regions of the atlas, 0.2 on the left and 0.6 on the right; their 3-D areas are 1 and 3, so the surface is
    a quarter of the first and three quarters of the second: 0.25 * 0.2 + 0.75 * 0.6 = 0.5. Regression caught: a
    per-triangle mean or a weight by UV area (both give 0.4), which over-weights the small parts and the stretched
    ones."""
    lum = np.full((4, 4), 0.2)
    lum[:, 2:] = 0.6
    uv = np.array([[(0.05, 0.1), (0.45, 0.1), (0.25, 0.9)], [(0.55, 0.1), (0.95, 0.1), (0.75, 0.9)]], dtype=float)
    assert abs(surface_mean(lum, uv, np.array([1.0, 3.0])) - 0.5) < 0.01


def test_surface_mean_reads_v_from_the_first_row_and_wraps_u():
    """Contract: `lum[0]` is the row at v = 0 (Blender stores image pixels bottom row first) and UVs repeat, so a
    surface keeps the texels it shows. The bottom row (v < 0.5) holds 1.0 on the left and 0.5 on the right, the top row
    holds 0.0; a triangle at v < 0.5 on the left reads 1.0, at v > 0.5 reads 0.0, on the right half 0.5, and one
    shifted a whole tile in u and v reads what it does unshifted. Regression caught: a top-down read of Blender's
    pixels (the surface is measured on the wrong part of the atlas), or a lookup that clips at the image's edge
    instead of repeating."""
    lum = np.array([[1.0, 1.0, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]])
    one = np.array([1.0])
    low = np.array([[(0.05, 0.05), (0.45, 0.05), (0.25, 0.45)]])
    assert surface_mean(lum, low, one) == 1.0
    assert surface_mean(lum, low + [0.0, 0.5], one) == 0.0
    assert surface_mean(lum, low + [0.5, 0.0], one) == 0.5
    assert surface_mean(lum, low + [1.0, 1.0], one) == 1.0


def test_surface_mean_is_deterministic():
    """Contract (V0 section 3: renders reproduce): the same surface gives the same mean on every call, and the seed is
    what decides the draw. Regression caught: an unseeded draw (the library, and every tinted pixel, would differ
    between two builds), or a seed that is ignored."""
    rng = np.random.default_rng(7)
    lum, uv = rng.random((8, 8)), rng.random((5, 3, 2))
    area = rng.random(5) + 0.1
    first = surface_mean(lum, uv, area, n=500, seed=3)
    assert surface_mean(lum, uv, area, n=500, seed=3) == first
    assert surface_mean(lum, uv, area, n=500, seed=4) != first
