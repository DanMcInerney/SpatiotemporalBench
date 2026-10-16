"""The sky picture of a map's world (`render/hdri_remap.py`: plain numpy, no Blender): a window anywhere in the
apartment shows only the clean stretch of the street panorama, the city's sun disc lands under the sun lamp, and the
sun is clamped. Free: synthetic pictures only, no download."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

RENDER = Path(__file__).resolve().parents[1] / "render"
W, H = 360, 180  # one degree a pixel
BAND = (0.20, 0.60)  # the columns of `potsdamer_platz` with no lettering (lane A's reading of the picture)
HEADING = -18.0  # `look_home.RIG`'s


def _remap():
    spec = importlib.util.spec_from_file_location("hdri_remap", RENDER / "hdri_remap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = _remap()


def u_picture() -> np.ndarray:
    """A picture whose red is the column's u, green its row's v and blue 0: reading it back says where each pixel of the
    world comes from."""
    img = np.zeros((H, W, 3), dtype=np.float32)
    img[..., 0] = ((np.arange(W) + 0.5) / W)[None, :]
    img[..., 1] = ((np.arange(H) + 0.5) / H)[:, None]
    return img


def test_the_apartment_world_only_ever_shows_the_clean_columns_of_the_street_picture():
    """Contract (the coordinator's relay of lane A, 2026-10-05): no direction of the apartment's world shows a column of
    `potsdamer_platz` outside u 0.20 to 0.60 (its "iPhone 7" billboard is at u 0.02-0.09, its station sign at 0.86-0.99
    and a speed-limit sign at 0.70), so no window, wherever the robot looks through it, shows a brand or a place name.
    The unlimited picture is not that (the test is not vacuous), the stretch is whole (no gap, no seam: neighbouring
    columns come from source columns at most a pixel apart), and the north, east and south windows (azimuths +y, +x and
    -y: u 0.25, 0.50 and 0.75 before the turn of -18 degrees) show three different stretches of it, by hand u 0.30,
    0.55 and 0.40 (0.25 + 0.05 is inside the band, 0.50 + 0.05 too, 0.75 + 0.05 = 0.80 folds back at 0.60 to 0.40).
    Regression caught: a lettered column back in a window (a brand on every apartment window), a heading that moves the
    band's edge into the picture's, or a fold that leaves a seam or a hole."""
    shown = R.remap(u_picture(), heading_deg=HEADING, band_u=BAND)
    red = shown[..., 0]
    assert red.min() >= BAND[0] - 0.5 / W and red.max() <= BAND[1] + 0.5 / W
    assert R.remap(u_picture(), heading_deg=HEADING)[..., 0].max() > 0.95  # without the band the whole picture shows
    assert np.abs(np.diff(red[H // 2])).max() <= 1.5 / W  # a fold, not a jump
    for u_out, want in ((0.25, 0.30), (0.50, 0.55), (0.75, 0.40)):
        assert red[H // 2, int(u_out * W)] == pytest.approx(want, abs=1.5 / W)


def test_the_cities_sun_disc_lands_in_the_direction_of_the_sun_lamp_and_the_horizon_and_zenith_stay_put():
    """Contract (the coordinator's relay: "shadows and the visible sun must agree"): the manifest's `kloofendal` has its
    sun disc at azimuth atan2(y, x) = -34 degrees and 48 degrees up; the map's sun lamp is at azimuth 110 from north,
    which is atan2 = 90 - 110 = -20 degrees, 62 degrees up. With the heading `-34 - (-20)` = -14 and the stretch (48,
    62) the bright pixel of a synthetic picture with a disc at (-34, 48) shows at (-20, 62) in a picture of one degree a
    pixel, and the picture's horizon and zenith rows are still the horizon and the zenith. Regression caught: a sun that
    shows 14 degrees below the lamp (a visible sun that disagrees with the shadows), or a turn the wrong way."""
    img = np.full((H, W, 3), 0.1, dtype=np.float32)
    col = round((math.pi - math.radians(-34)) / (2 * math.pi) * W - 0.5)  # u = (pi - atan2(y, x)) / 2 pi
    row = round((48 / 180 + 0.5) * H - 0.5)
    img[row - 1:row + 2, col - 1:col + 2] = 15.0  # the disc, under the clamp so that it survives
    kloofendal = {"sun_atan2_deg": -34.0, "sun_elevation_deg": 48.0}  # `look_downtown.RIG["hdri"]`'s
    aimed = R.aimed_at_sun(kloofendal, azimuth_deg=110.0, elevation_deg=62.0)  # `downtown_a`'s sun
    assert (aimed["heading_deg"], aimed["disc"]) == (-14.0, (48.0, 62.0))
    assert R.aimed_at_sun({"light_strength": 1.0}, 110.0, 62.0) == {"light_strength": 1.0}  # no sun of its own: as is
    shown = R.remap(img, heading_deg=aimed["heading_deg"], disc=aimed["disc"])
    r, c = np.unravel_index(np.argmax(shown[..., 0]), shown.shape[:2])
    azimuth = math.degrees(math.pi - 2 * math.pi * (c + 0.5) / W)
    elevation = ((r + 0.5) / H - 0.5) * 180
    assert azimuth == pytest.approx(-20.0, abs=2.0) and elevation == pytest.approx(62.0, abs=2.5)
    rows = R.remap(u_picture(), disc=(48.0, 62.0))[..., 1][:, 0]
    assert rows[H // 2] == pytest.approx(0.5, abs=1.0 / H) and rows[-1] == pytest.approx(1.0, abs=1.0 / H)


def test_horizontal_illuminance_by_hand():
    """Contract (`horizontal_illuminance`): the sum over a picture's upper hemisphere of luminance x sin(elevation) x
    the pixel's solid angle. By hand: a uniform sky of radiance 0.5 gives pi x 0.5 = 1.5708 (the integral of sin x
    cos over the hemisphere is 1/2, times 2 pi); a picture lit only below the horizon gives 0; and a single ring of
    radiance 2 at the row of elevation 45.5 degrees (the 135th of 180, a degree a row) gives 2 sin(45.5) cos(45.5) x
    (pi / 180) x 2 pi = 2 x 0.499924 x 0.109662 = 0.10965, the ring weighted by its own cosine. Regression caught: a
    sky calibrated by a wrong integral (a missing cos for the pixel's solid angle, or the ground counted as sky),
    which would set every sky's strength wrong by a constant."""
    uniform = np.full((H, W, 3), 0.5, dtype=np.float32)
    assert R.horizontal_illuminance(uniform) == pytest.approx(math.pi * 0.5, rel=5e-4)
    below = np.zeros((H, W, 3), dtype=np.float32)
    below[: H // 2] = 10.0
    assert R.horizontal_illuminance(below) == 0.0
    ring = np.zeros((H, W, 3), dtype=np.float32)
    ring[135] = 2.0
    assert R.horizontal_illuminance(ring) == pytest.approx(0.10965, rel=1e-3)


def test_the_clear_sky_is_the_cie_standard_clear_sky_by_hand():
    """Contract (`clear_sky_lux`, Kittler and Darula 2002, eSim, eq. 10 to 17 and table 2 for type 12) at 38 degrees
    and a luminous turbidity of 2. By hand: sin 38 = 0.615661, so the extraterrestrial horizontal illuminance is
    133.8 x 0.615661 = 82.376 klx; Kasten and Young's air mass is 1 / (0.615661 + 0.50572 x 44.07995^-1.6364) =
    1.6216, av = 1 / (9.9 + 0.043 x 1.6216) = 0.100304, and the sun gives 82.376 exp(-0.100304 x 1.6216 x 2) = 59.50
    klx on a horizontal surface. The sky: sin^4.43 / cos^0.74 = 0.13911, Lz = 0.7 x 3 x 0.13911 + (1.036 x 2 + 0.710)
    x 0.615661 + 0.08 = 2.0849 kcd/m2 over Lz / Dv = (23.00 x 0.13911 + 18.52 x 0.615661) / 82.376 = 0.17727, so
    11.761 klx. A hazier day (turbidity 4) has less sun and more sky: 42.98 and 20.51 klx. Regression caught: a
    clear-sky model implemented with another constant than the one cited (the daylight's ratio to the lamps would
    drift unseen)."""
    sun, sky = R.clear_sky_lux(38.0, 2.0)
    assert sun == pytest.approx(59500.0, rel=1e-3) and sky == pytest.approx(11761.0, rel=1e-3)
    sun, sky = R.clear_sky_lux(38.0, 4.0)
    assert sun == pytest.approx(42978.0, rel=1e-3) and sky == pytest.approx(20509.0, rel=1e-3)


def test_a_suns_disc_is_clamped_so_that_it_does_not_light_the_scene_a_second_time():
    """Contract (`hdri_remap.SUN_CLAMP`): no pixel of the world's picture is brighter than 20, though an
    HDRI's sun disc is a few ten thousand (the sun lamp is the sun), and a pixel below it is not changed. Regression
    caught: the HDRI's own sun (4.9 W/m2 against the lamp's 3.6) doubling the light of every outdoor scene."""
    img = np.full((H, W, 3), 0.5, dtype=np.float32)
    img[90, 180] = 72000.0
    shown = R.remap(img)
    assert shown.max() <= R.SUN_CLAMP + 1e-4 and shown[10, 10, 0] == pytest.approx(0.5)
