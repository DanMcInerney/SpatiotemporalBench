"""How `stbench realism report` reads noise (`realism_noise`): flat patches, their luma σ, colour noise and lag-1
correlation, the tables by band, and the flat fills of the references left out. Each is worked by hand on 16 x 16
pictures made here, so a number that reads wrong is caught without a render. Free: arrays, no Blender."""

from __future__ import annotations

import math

import numpy as np
import pytest

from stbench import realism_noise as N


def _checker(a: int, b: int, size: int = 16, run: int = 1) -> np.ndarray:
    """A size x size picture of grey `a` and `b` in a checkerboard of squares `run` pixels across, (h, w, 3) uint8."""
    yy, xx = np.indices((size, size))
    return np.where(((yy // run + xx // run) % 2 == 0)[..., None], a, b).astype(np.uint8).repeat(3, axis=2)


def test_a_flat_patch_has_the_luma_sigma_of_its_pixels_from_their_block_means_and_the_neighbour_correlation():
    """Contract (`patch_stats`; noise is read from flat patches only): of a 16 x 32 picture whose left patch is a
    checkerboard of 124 and 132 (every 4 x 4 block mean is 128, the pixels are 4 levels from it: sigma 4.0, grey so no
    colour noise, and each pixel is the opposite of its right and lower neighbour: lag-1 correlation -1.0, the mean luma
    128 / 255) and whose right patch is a ramp from black to white (its block means differ by 90 levels: not flat), the
    one patch is (0.502, 4.0, 0.0, -1.0). A checkerboard of 2 x 2 squares (also flat): 8 of the 15 horizontal pairs of a
    row lie inside a square (correlation +1) and 7 across (-1), as the vertical ones do: (8 - 7) / 15 = 1/15. Regression
    caught: a textured wall read as noise, the correlation read of the block-mean residual (white noise would then read
    -0.25 and a smear 0.3 too low) or a sigma over the whole picture."""
    img = np.zeros((16, 32, 3), dtype=np.uint8)
    img[:, :16] = _checker(132, 124)
    img[:, 16:] = np.linspace(0, 255, 16).astype(np.uint8)[None, :, None]
    (patch,) = N.patch_stats(img)
    assert patch == {"mean": pytest.approx(128 / 255), "sigma": pytest.approx(4.0), "chroma": pytest.approx(0.0),
                     "lag1": pytest.approx(-1.0)}
    (coarse,) = N.patch_stats(_checker(132, 124, run=2))
    assert coarse["lag1"] == pytest.approx(1 / 15) and coarse["sigma"] == pytest.approx(4.0)


def test_colour_noise_is_cb_and_cr_over_the_luma_noise_of_the_residual():
    """Contract (`patch_stats`' `chroma`): the root mean square of the standard deviations of Cb = (B - Y) / 1.8556 and
    Cr = (R - Y) / 1.5748, over the luma's, of the patch less its plane (Rec.709 on the encoded values). By hand on a
    checkerboard of s = +-1 with R = 128 + 4s, G = 128, B = 128 - 4s: Y = 128 + 0.5616 s (0.2126 * 4 - 0.0722 * 4 =
    0.5616), so B - Y = -4.5616 s and R - Y = 3.4384 s; the chroma noise is sqrt(((4.5616 / 1.8556)^2 + (3.4384 /
    1.5748)^2) / 2) = 2.325, the luma's 0.5616, the ratio 4.14. Regression caught: colour noise measured in R, G, B
    (a grey noise reads as colour) or against the wrong weights."""
    yy, xx = np.indices((16, 16))
    s = np.where((yy + xx) % 2 == 0, 1, -1)
    img = np.stack([128 + 4 * s, np.full((16, 16), 128), 128 - 4 * s], axis=-1).astype(np.uint8)
    (patch,) = N.patch_stats(img)
    expected = math.sqrt(((4.5616 / 1.8556) ** 2 + (3.4384 / 1.5748) ** 2) / 2) / 0.5616
    assert patch["chroma"] == pytest.approx(expected, rel=2e-3)


def test_a_patch_with_a_flat_fill_in_it_is_left_out_whole():
    """Contract (`fill_mask`, `patch_stats`' `mask`): the flat fill of the robot makers' pictures (exactly RGB 140, 138,
    140) is found by exact match, and a patch that holds any of it is dropped: of two flat patches side by side, one
    holding a single pixel of fill (a pixel 140, 138, 141 is not fill), only the other is read. Regression caught: fills
    (no noise, luma 0.54) read as the cleanest patches of the cleanest pictures, which pulled the references' noise
    down."""
    img = np.concatenate([_checker(132, 124), _checker(132, 124)], axis=1)
    img[3, 20] = (140, 138, 140)
    img[5, 21] = (140, 138, 141)
    mask = N.fill_mask(img)
    assert mask.sum() == 1 and mask[3, 20]
    assert len(N.patch_stats(img)) == 2 and len(N.patch_stats(img, mask)) == 1


def test_noise_tables_are_medians_by_luminance_band_and_over_everything():
    """Contract (`table_by_band`, `overall`): per band of mean luma (0-0.15, 0.15-0.35, 0.35-0.65, 0.65-1) the number of
    patches and the median sigma, chroma and lag-1 of those in it, None for a band with none. By hand: patches of mean
    0.1 with sigma 2 and 4 read 3.0 in the darkest band, one at 0.5 with sigma 3 reads 3.0 in the third, one at 0.9 with
    sigma 5 reads 5.0 in the last, the second has none; over all four the median sigma is 3.5. Regression caught: a
    band that takes another's patches, a mean in place of a median, or an empty band read as 0."""
    def patch(mean, sigma, chroma=0.5, lag1=0.0):
        return {"mean": mean, "sigma": sigma, "chroma": chroma, "lag1": lag1}

    rows = [patch(0.1, 2.0, 0.2, 0.1), patch(0.1, 4.0, 0.4, 0.3), patch(0.5, 3.0), patch(0.9, 5.0)]
    table = N.table_by_band(rows)
    assert table["0-0.15"] == {"patches": 2, "sigma": 3.0, "chroma": pytest.approx(0.3), "lag1": pytest.approx(0.2)}
    assert table["0.15-0.35"] == {"patches": 0, "sigma": None, "chroma": None, "lag1": None}
    assert table["0.35-0.65"]["sigma"] == 3.0 and table["0.65-1"]["sigma"] == 5.0
    assert N.overall(rows)["sigma"] == 3.5 and N.overall(rows)["patches"] == 4
