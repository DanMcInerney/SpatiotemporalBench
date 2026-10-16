"""The pictures the downtown's shaders read (`render/ground_arrays.py` and `render/window_paint.py`, numpy only: no
Blender needed): the worn asphalt tile and the window and shop atlases. They are painted from fixed seeds into images
because the same picture built from shader nodes cost a street frame about twice the render time; what a test can say
about them without a render is whether they are what the shaders assume (tileable, in the range a road and a lit room
have, deterministic) and whether they hold what the realism plan asks for (a road with repairs and cracks, windows that
are never black voids and are not all alike, shops that glow). How the shaders place them is
`test_facade_windows_blender`'s and the renders'."""

from __future__ import annotations

import numpy as np
import pytest

from dressing_kit import render_module

G = render_module("ground_arrays")
W = render_module("window_paint")
LUMA = np.array((0.2126, 0.7152, 0.0722))


@pytest.fixture(scope="module")
def tile() -> tuple[np.ndarray, np.ndarray]:
    return G.asphalt_arrays()


@pytest.fixture(scope="module")
def windows() -> np.ndarray:
    return W.window_array()


@pytest.fixture(scope="module")
def shops() -> np.ndarray:
    return W.shop_array()


def cells(atlas: np.ndarray, cols: int, rows: int) -> list[np.ndarray]:
    c = W.CELL
    return [atlas[r * c:(r + 1) * c, k * c:(k + 1) * c] for r in range(rows) for k in range(cols)]


def test_the_asphalt_tile_is_a_dark_worn_tileable_road_with_repairs_cracks_and_oil_and_is_the_same_every_time(tile):
    """Contract (plan-realism section 4, "City": asphalt with patching, wear and oil stains; `materials_street.asphalt`
    reads this tile on object coordinates, repeating every 64 m): the tile is 1024 x 1024 pixels of linear colour and
    roughness, made from a fixed seed (a second call gives the same numbers); the colour is dark grey (mean luminance
    0.06 to 0.10, no pixel above 0.3 or below 0.005, and a spread of about 0.02: not flat) as a road is; roughness lies
    from 0.4 (tar) to 0.9 (dry asphalt) and 5% to 30% of the pixels are smoother than 0.8 (patches, scars, oil: here
    14%); and it tiles: across the seam at either edge the colours differ by no more than 1.5 times what two
    neighbouring columns (rows) do inside it. Regression caught: an unseeded tile (the road differs every render), a
    seam a line painted down every 64 m, a road of one grey (no repairs, cracks or oil) or a tile that is a different
    brightness from the old shader's asphalt and so changes every street frame's exposure."""
    col, rough = tile
    again, _ = G.asphalt_arrays()
    assert col.shape == (1024, 1024, 3) and rough.shape == (1024, 1024) and np.array_equal(col, again)
    lum = col @ LUMA
    assert 0.06 <= lum.mean() <= 0.10 and lum.min() >= 0.005 and lum.max() <= 0.3 and 0.01 <= lum.std() <= 0.03
    assert rough.min() >= 0.4 - 1e-9 and rough.max() <= 0.9 + 1e-9 and 0.05 <= (rough < 0.8).mean() <= 0.30
    for axis in (0, 1):
        seam = np.abs(np.take(col, 0, axis=axis) - np.take(col, -1, axis=axis)).mean()
        inner = np.abs(np.diff(col, axis=axis)).mean()
        assert seam <= 1.5 * inner, (axis, seam, inner)


def test_the_macro_tone_is_a_factor_near_one_that_breaks_the_tiles_repeat():
    """Contract (`asphalt_macro`: slow tonal variation over 41.3 m, which with the tile's 64 m repeats every 2,643 m, so
    no repeat shows in the 245 m of scenery): the macro tone is 512 x 512 values around 1 (mean 0.95 to 1.05, between
    0.8 and 1.2), the same every time. Regression caught: a macro tone that darkens or brightens every street (its mean
    far from 1), or one that varies so much that it hides the tile's patches."""
    m = G.macro_array()
    assert m.shape == (512, 512) and np.array_equal(m, G.macro_array())
    assert 0.95 <= m.mean() <= 1.05 and m.min() >= 0.8 and m.max() <= 1.2
    assert G.TILE_M == 64.0 and G.MACRO_M == 41.3


def test_no_window_is_a_black_void_and_the_windows_are_not_all_alike_and_some_are_lit(windows):
    """Contract (plan-realism section 4, "City": windows with interiors, "no black voids", varied per window, seeded;
    the old windows were a dark tint, 0.012 to 0.04 in linear light): the window atlas is 8 x 8 cells of 128 x 128
    pixels in the same fixed-seed numbers every time; every cell's mean luminance is at least 0.07 (the darkest here is
    0.095, the mean 0.17), at least 40 of the 64 cells differ (rounded to a thousandth: blinds, curtains, rooms,
    reflective panes and lit rooms), between 4 and 16 are lit (alpha above 0, the share of a pixel that gives off light:
    here 9) and every cell has its dark frame at its corner (luminance under 0.07). Regression caught: a pane that is
    the old black tint, an atlas of one interior repeated, or lit windows that never glow (an emission of zero) or all
    of them."""
    assert windows.shape == (1024, 1024, 4) and np.array_equal(windows, W.window_array())
    lums, lit = [], 0
    for cell in cells(windows, 8, 8):
        lum = cell[..., :3] @ LUMA
        lums.append(lum.mean())
        lit += cell[..., 3].max() > 0.0
        assert lum[:3, :3].mean() < 0.07 and cell[..., 3].min() >= 0.0 and cell[..., 3].max() <= 1.0
    assert min(lums) >= 0.07 and len(set(np.round(lums, 3))) >= 40 and 4 <= lit <= 16


def test_the_shop_atlas_has_lit_retail_that_glows_a_dim_cafe_shutters_and_papered_glass_and_none_is_black(shops):
    """Contract (plan-realism section 4, "City": storefronts are not black voids): the shop atlas is 4 x 4 cells of the
    five kinds of glazing in turn (lit retail, a dim cafe, a rolled-down shutter, papered-over glass, a reflective
    lobby, the first again for the sixteenth cell); every cell's mean luminance is at least 0.07 (the darkest here
    0.083), retail gives off light (emission 0.9: lit with its goods), the cafe a little (0.3), the lobby a little
    (0.25), and a shutter and papered glass none; the shutter is ribbed (its luminance varies row to row) and the
    papered glass the palest of all (0.28). Regression caught: a shop front painted black (the old shop colours were
    0.01 to 0.09), retail that does not glow, or a shutter that gives off light."""
    assert shops.shape == (512, 512, 4) and np.array_equal(shops, W.shop_array())
    cs = cells(shops, 4, 4)
    lums = [(c[..., :3] @ LUMA).mean() for c in cs]
    emission = [c[..., 3].mean() for c in cs]
    assert min(lums) >= 0.07
    for k, (lum, e) in enumerate(zip(lums, emission, strict=True)):
        kind = k % 5
        want = {0: (0.8, 1.0), 1: (0.2, 0.4), 2: (0.0, 0.0), 3: (0.0, 0.0), 4: (0.15, 0.35)}[kind]
        assert want[0] <= e <= want[1], (k, kind, e)
        assert (lum >= 0.25) == (kind == 3), (k, kind, lum)  # only the papered glass is that pale
    shutter = cs[2][..., :3] @ LUMA
    assert shutter[16:112, 16:112].std(axis=0).mean() > 0.003  # ribs: it varies up the cell
