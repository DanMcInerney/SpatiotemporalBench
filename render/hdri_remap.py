# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The sky picture a map's world shows: an equirectangular HDRI of the manifest, turned, folded and clamped.

Plain Python and numpy, no `bpy` (a test reads it without Blender); `lighting_sky` turns the result into the world's
environment image. The picture is (H, W, 3) linear floats, its first row the bottom (nadir), as Blender stores pixels,
and its columns run with `u = (pi - atan2(y, x)) / 2 pi` (Cycles' equirectangular mapping: u 0.5 is +x, 0.25 is +y),
its rows with `v = elevation / pi + 0.5`. `remap` builds, for every direction of the world, what the source picture
shows there:
- `clamp`: no pixel brighter than this, so that the sun's disc in an HDRI (a hundred thousand times a sky's radiance)
  does not light the scene a second time next to the sun lamp;
- `heading_deg`: a feature of the picture at azimuth phi shows at phi - heading;
- `band_u`: the picture's columns `(u0, u1)` that may be shown at all. The world is built of that stretch of the picture
  alone, mirrored where it ends and mirrored again, so that no direction shows anything else: the manifest's
  `potsdamer_platz` has a billboard, a station sign and a speed-limit sign outside u 0.20 to 0.60, and a window
  anywhere in the apartment may look anywhere;
- `disc`: `(picture's sun elevation, the sun lamp's elevation)` in degrees. The picture is stretched in elevation so
  that its sun's disc lies in the lamp's direction, the horizon and the zenith staying where they are, so that the sun
  one sees is the sun that casts the shadows. The azimuths are brought together by `heading_deg`.

The sky's strength is set from light, not by eye: `horizontal_illuminance` integrates a picture over its upper
hemisphere, so that a strength can make the sky give a horizontal surface what the clear sky of `clear_sky_lux` gives.
"""

from __future__ import annotations

import math

import numpy as np

LUMA = (0.2126, 0.7152, 0.0722)  # Rec. 709 luminance of linear RGB
# CIE Standard Clear Sky (ISO 15469 / CIE S 011 type 12, "V.4") in absolute units: Kittler and Darula, "CIE General Sky
# Standard Defining Luminance Distributions", eSim 2002, tables 1 and 2 (type 12's A1, A2, B, C, D, E) and eq. 10 to 17
CLEAR_SKY = {"A1": 1.036, "A2": 0.710, "B": 23.00, "C": 4.43, "D": 0.74, "E": 18.52, "extraterrestrial_klx": 133.8}
SUN_CLAMP = 20.0  # an HDRI's pixel is clipped to this radiance: a sky's is about 1, its sun's disc 70,000


def pingpong(x: np.ndarray, scale: float) -> np.ndarray:
    """A triangle wave through 0 at x = 0 and `scale` at x = `scale`, going back down after: Blender's Math PINGPONG."""
    return scale - np.abs(np.mod(x, 2.0 * scale) - scale)


def elevation_lookup(theta: np.ndarray, disc: tuple[float, float] | None) -> np.ndarray:
    """The elevation of the picture to show at elevation `theta` (radians): the identity, or the stretch that takes
    the lamp's elevation to the picture's sun's (`disc`), piecewise linear and convex, so the larger of two lines."""
    if disc is None:
        return theta
    sun, lamp = math.radians(disc[0]), math.radians(disc[1])
    below = theta * (sun / lamp)
    above = sun + (theta - lamp) * ((math.pi / 2 - sun) / (math.pi / 2 - lamp))
    return np.maximum(below, above)


def aimed_at_sun(hdri: dict, azimuth_deg: float, elevation_deg: float) -> dict:
    """`hdri`'s settings with `heading_deg` and `disc` set so that the picture's own sun (`sun_atan2_deg`, the azimuth
    as `atan2(y, x)`, and `sun_elevation_deg`) comes to where a sun lamp shines from: `azimuth_deg` clockwise from
    north (the maps' `sun_rotation_deg`, which is `90 - azimuth` as `atan2(y, x)`) and `elevation_deg` up. Settings
    without a sun of their own are returned as they are."""
    out = dict(hdri)
    if "sun_atan2_deg" in hdri:
        out["heading_deg"] = hdri["sun_atan2_deg"] - (90.0 - azimuth_deg)
        out["disc"] = (hdri["sun_elevation_deg"], elevation_deg)
    return out


def remap(img: np.ndarray, heading_deg: float = 0.0, band_u: tuple[float, float] | None = None,
          disc: tuple[float, float] | None = None, clamp: float = SUN_CLAMP) -> np.ndarray:
    """The picture the world shows (same shape as `img`): see the module docstring. Bilinear, wrapping in u and
    clamped in v."""
    h, w = img.shape[:2]
    u_out = (np.arange(w) + 0.5) / w
    theta = ((np.arange(h) + 0.5) / h - 0.5) * math.pi
    u_src = u_out - heading_deg / 360.0
    if band_u is not None:
        u0, u1 = band_u
        u_src = u0 + pingpong(u_src - u0, u1 - u0)
    v_src = np.clip(elevation_lookup(theta, disc) / math.pi + 0.5, 0.0, 1.0)
    x = u_src * w - 0.5
    y = v_src * h - 0.5
    x0 = np.floor(x).astype(np.int64)
    fx = (x - x0)[None, :, None]
    y0 = np.clip(np.floor(y).astype(np.int64), 0, h - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    fy = (y - np.floor(y))[:, None, None]
    src = np.minimum(img, clamp)
    xa, xb = x0 % w, (x0 + 1) % w
    top = src[y1][:, xa] * (1 - fx) + src[y1][:, xb] * fx
    bottom = src[y0][:, xa] * (1 - fx) + src[y0][:, xb] * fx
    return (bottom * (1 - fy) + top * fy).astype(np.float32)


def horizontal_illuminance(img: np.ndarray) -> float:
    """What a picture gives a horizontal surface: the sum over its upper hemisphere of luminance x sin(elevation) x
    the pixel's solid angle, in the picture's own units (a uniform sky of radiance L gives pi L; a picture lit only
    below the horizon gives 0). `img` is (H, W, 3) linear RGB in the layout of this module, its first row the bottom."""
    h, w = img.shape[:2]
    theta = ((np.arange(h) + 0.5) / h - 0.5) * math.pi  # each row's elevation
    solid_angle = np.cos(theta) * (math.pi / h) * (2.0 * math.pi / w)
    luminance = img[..., 0] * LUMA[0] + img[..., 1] * LUMA[1] + img[..., 2] * LUMA[2]
    return float((luminance.sum(axis=1) * np.maximum(np.sin(theta), 0.0) * solid_angle).sum())


def clear_sky_lux(elevation_deg: float, turbidity: float) -> tuple[float, float]:
    """(the sun, the sky) illuminance in lux on a horizontal surface under the CIE Standard Clear Sky at solar altitude
    `elevation_deg` and luminous turbidity `turbidity` (the number of ideally clean atmospheres the sun shines through:
    2 for a very clear day, 4 for a hazy one). The sun: Pv = 133.8 sin g exp(-av m Tv) klx, with the air mass of Kasten
    and Young, m = 1 / (sin g + 0.50572 (g + 6.07995)^-1.6364), and Clear's av = 1 / (9.9 + 0.043 m). The sky: its
    zenith luminance Lz = 0.7 (Tv + 1) sin^C g / cos^D g + (A1 Tv + A2) sin g + 0.04 Tv kcd/m2, over the ratio
    Lz / Dv = (B sin^C g / cos^D g + E sin g) / Ev, with Ev = 133.8 sin g the extraterrestrial horizontal
    illuminance."""
    k = CLEAR_SKY
    g = math.radians(elevation_deg)
    s, c = math.sin(g), math.cos(g)
    extraterrestrial = k["extraterrestrial_klx"] * s
    air_mass = 1.0 / (s + 0.50572 * (elevation_deg + 6.07995) ** -1.6364)
    extinction = 1.0 / (9.9 + 0.043 * air_mass)
    sun = extraterrestrial * math.exp(-extinction * air_mass * turbidity)
    pattern = s ** k["C"] / c ** k["D"]
    zenith = 0.7 * (turbidity + 1.0) * pattern + (k["A1"] * turbidity + k["A2"]) * s + 0.04 * turbidity
    sky = zenith / ((k["B"] * pattern + k["E"] * s) / extraterrestrial)
    return sun * 1000.0, sky * 1000.0
