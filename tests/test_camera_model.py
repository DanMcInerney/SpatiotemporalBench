"""The camera model (`camera_model.py`: numpy and Pillow, no Blender): each stage by hand, and the structure of the
whole, which no test pins to a tuned curve or an exposure response (recommendations §2.6 item 9): depth and index are
never its input, the white balance keeps catalog colours their names, exposure is monotone, a JPEG carries no
metadata. Free: synthetic frames and a literal camera (`camera_kit`), never the robots' own values."""

from __future__ import annotations

import ast
import io
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from camera_kit import FULL_ISP, KEY, LITERAL_ISP, develop_digest, literal_camera, synthetic_frame
from conftest import apartment
from stbench import camera_model as C
from stbench.catalog import CATALOGS
from stbench.seeding import make_rng

ROOT = Path(__file__).resolve().parents[1]
NO_NOISE = replace(LITERAL_ISP, noise_a=0.0, noise_b=0.0)


def test_develop_is_deterministic():
    """Contract: the same frame and key give the same bytes, in this process and in a fresh one, and another key gives
    other bytes (the noise is seeded by the key and by nothing else). Regression caught: a nondeterministic dataset (an
    unseeded rng, a set or dict order, a process-salted hash), or noise that ignores the key, so every frame of a script
    would share one noise field."""
    frame, cam = synthetic_frame(1), literal_camera()
    first, log = C.develop(frame, key=KEY, camera=cam)
    again, log_again = C.develop(frame.copy(), key=KEY, camera=cam)
    other, _ = C.develop(frame, key=(*KEY[:-1], 1), camera=cam)
    assert first == again and log == log_again and first != other
    code = "import sys; sys.path.insert(0, 'tests'); import camera_kit; print(camera_kit.develop_digest())"
    child = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=True)
    assert child.stdout.strip() == develop_digest()


def test_vignette_by_hand():
    """Contract (`vignette_gain`): at 640 x 480 and 80 degrees with v 0.3, the corner pixel's factor is 0.768268 (by
    hand: fx = 320 / tan 40 = 381.361150; the corner pixel's centre is (319.5, 239.5) px from the principal point, so
    cos^2 theta = fx^2 / (fx^2 + 319.5^2 + 239.5^2) = 0.47703 and cos^4 theta = 0.227561; 1 - 0.3 (1 - 0.227561) =
    0.768268), the centre pixels' at least 0.999997, and the vignette is applied before the meter: a flat field
    meters lower with v 0.3 than with v 0. Regression caught: a falloff stronger or weaker than declared, or one
    applied after metering (so the exposure would not see the dark corners)."""
    gain = C.vignette_gain(640, 480, 80.0, 0.3)
    assert gain.shape == (480, 640)
    assert gain[0, 0] == pytest.approx(0.768268, abs=1e-6) and gain[-1, -1] == pytest.approx(0.768268, abs=1e-6)
    assert gain[239:241, 319:321].min() >= 0.999997
    flat = np.full((48, 64, 3), 0.1)
    _, log = C.capture(flat, KEY, literal_camera(isp=NO_NOISE))
    _, log0 = C.capture(flat, KEY, literal_camera(isp=replace(NO_NOISE, vignette=0.0)))
    assert log["metered"] < log0["metered"] and log0["metered"] == pytest.approx(0.1)


def test_meter_is_a_log_average():
    """Contract (`meter`): with uniform weights (sigma 1e3), a field of 0.05 with 10% of its pixels at 10.0 meters
    exp(0.9 ln 0.05 + 0.1 ln 10) = 0.084932 (a mean would give 1.045, 3.6 EV off); a black field meters the floor
    1e-4; at sigma 0.35 a centred patch moves the meter more than the same patch in a corner. Regression caught: a
    meter dragged down by highlights (a mean, not a log-average), no floor under black, or weights that are not
    centre-heavy."""
    field = np.full((100, 100, 3), 0.05)
    field[:, :10] = 10.0
    assert C.meter(field, 1e3) == pytest.approx(0.084932, abs=1e-5)
    assert C.meter(np.zeros((8, 8, 3)), 0.35) == pytest.approx(1e-4)
    base = np.full((100, 100, 3), 0.05)
    centre, corner = base.copy(), base.copy()
    centre[40:60, 40:60], corner[:20, :20] = 5.0, 5.0
    assert C.meter(centre, 0.35) > C.meter(corner, 0.35) > C.meter(base, 0.35) == pytest.approx(0.05)


def test_exposure_is_monotone_and_clamped():
    """Contract (`exposure`, `capture`): over flat fields from 1e-6 to 1e3 with the noise off, the total EV never
    rises; it is exposure_ev[1] + gain_ev (8) at the dark end and exposure_ev[0] (-8) at the bright end; the gain is
    0 while the exposure time is under its top; and the developed 8-bit mean never falls (a brighter scene is not
    developed darker). Regression caught: exposure that ignores the scene, a gain applied before the exposure is at
    its top, or a brighter scene developed darker."""
    isp, cam = NO_NOISE, literal_camera(isp=NO_NOISE)
    evs, means = [], []
    for x in 10.0 ** np.arange(-6, 3.5, 0.5):
        _, log = C.capture(np.full((48, 64, 3), x), KEY, cam)
        evs.append(log["ev"])
        assert log["gain_ev"] == 0.0 or log["exposure_ev"] == isp.exposure_ev[1]
        means.append(C.develop_8bit(np.full((48, 64, 3), x), key=KEY, camera=cam)[0].mean())
    assert evs[0] == isp.exposure_ev[1] + isp.gain_ev and evs[-1] == isp.exposure_ev[0]
    assert all(b <= a for a, b in zip(evs, evs[1:], strict=False))
    assert all(b >= a - 0.01 for a, b in zip(means, means[1:], strict=False)), means  # flat plateau: 8-bit rounding
    assert means[-1] > means[0]


def test_noise_follows_its_declared_variance():
    """Contract (`add_noise`, `capture`): with a = 6e-4 and b = 1e-5, a flat 0.1 has variance 7.0e-5 (0.1 a + b)
    within 2% over 307,200 pixels, and a field of 0 has the noise of variance b before the floor at 0 (the mean of
    the floored half-normal, sqrt(b / 2 pi) = 1.2616e-3); and noise is added on the exposed signal and before the
    gain: a field 16x darker than the bright one (0.18 / 64 against 0.18), lifted 4 EV by the gain, has relative
    noise 0.36379 before the white balance (sqrt(a 0.01125 + b) / 0.01125 by hand: the exposure time gave it 2 EV and
    the gain the other 4), against 0.06035 for the bright one (sqrt(a 0.18 + b) / 0.18), 6.03 times more, each within
    2%. Regression caught: noise added after the gain (the dark frame as clean as the bright one), or noise that
    ignores the signal."""
    a, b = 6e-4, 1e-5
    flat = np.full((480, 640, 3), 0.1)
    assert C.add_noise(flat, a, b, make_rng("noise", 1)).var() == pytest.approx(0.1 * a + b, rel=0.02)
    zero = C.add_noise(np.zeros((480, 640, 3)), a, b, make_rng("noise", 2))
    assert zero.mean() == pytest.approx(math.sqrt(b / (2 * math.pi)), rel=0.02)
    cam = literal_camera(160, 120, replace(LITERAL_ISP, vignette=0.0))
    bright, _ = C.capture(np.full((120, 160, 3), 0.18), KEY, cam)
    dark, log = C.capture(np.full((120, 160, 3), 0.18 / 64), KEY, cam)
    assert (log["exposure_ev"], log["gain_ev"]) == (2.0, 4.0)
    rel_bright, rel_dark = bright.std() / bright.mean(), dark.std() / dark.mean()
    assert rel_bright == pytest.approx(0.06035, rel=0.02) and rel_dark == pytest.approx(0.36379, rel=0.02)
    assert rel_dark / rel_bright == pytest.approx(6.03, rel=0.02)


def srgb_to_lab(rgb8) -> np.ndarray:
    """CIELAB (D65) of an sRGB colour given as 0-255 values, by the standard formulas."""
    c = np.asarray(rgb8, dtype=float) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = (m @ lin) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def palette() -> dict[str, tuple[float, float, float]]:
    """`render/shapes.PALETTE`'s colour (sRGB, 0 to 1) by key, read from the source (the module imports bpy), as
    `test_mapspec.palette_keys` reads the keys."""
    source = ROOT / "render" / "shapes.py"
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "PALETTE":
            return {k: v[0] for k, v in ast.literal_eval(node.value).items()}
    raise AssertionError(f"no top-level PALETTE in {source}")


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def test_white_balance_is_clamped_and_keeps_colour_names():
    """Contract (`white_balance`, and the stages after it): a field with R/G 2 and B/G 0.4 gets the gains (0.9, 1,
    1.1) at clamp 0.1 (grey world asks for 0.5 and 2.5), and a neutral field (1, 1, 1). And every home catalog colour
    and every colour of the apartment's light furniture (beige and brown among them), as a 16 x 12 patch of its
    palette albedo on a mid-grey surround (0.18) under a light tinted past the clamp, so that the gains sit at either
    edge ((1.2, 1, 1/1.2) and (1/1.2, 1, 1.2): grey world asks for 0.83 and 1.2), develops, through the whole camera,
    nearest in CIELAB (Delta E76) to its own colour among those colours: for a camera with nothing but the starting
    stages (an exposure aim of the grey card itself reads each colour as its palette sRGB) that is the palette sRGB,
    and for the camera with every stage on (an aim a stop lower and a contrast read every colour darker, as the real
    robot's does) it is the colour developed under a neutral light by the same camera. Regression caught: a white
    balance that renames catalog colours (the colour attribute is what the pixels show, VISION), or a tone curve that
    sends a dark or a saturated colour toward another's, or two colours onto one."""
    tinted = np.ones((4, 4, 3)) * np.array([2.0, 1.0, 0.4])
    _, gains = C.white_balance(tinted, 0.1)
    assert gains == pytest.approx([0.9, 1.0, 1.1])
    assert C.white_balance(np.ones((4, 4, 3)), 0.1)[1] == pytest.approx([1.0, 1.0, 1.0])
    colours = {c for kind in CATALOGS["home"].kinds.values() for c in kind.colors}
    colours |= {str(f.color) for f in apartment().furniture if f.mobility == "light"}
    pal = palette()
    assert {"beige", "brown", "black", "white"} <= colours <= pal.keys()
    labs = {name: srgb_to_lab(np.array(pal[name]) * 255) for name in colours}

    def developed(cam, name, tint):
        frame = np.full((48, 64, 3), 0.18)
        frame[18:30, 24:40] = [srgb_to_linear(c) for c in pal[name]]
        rgb8, log = C.develop_8bit(frame * np.array(tint), key=KEY, camera=cam)
        return srgb_to_lab(rgb8[21:27, 28:36].reshape(-1, 3).mean(axis=0)), log["wb"], rgb8[24, 32]

    for cam, reads_palette in ((literal_camera(), True), (literal_camera(isp=FULL_ISP), False)):
        refs = labs if reads_palette else {name: developed(cam, name, (1.0, 1.0, 1.0))[0] for name in colours}
        for tint in ((1.2, 1.0, 1 / 1.2), (1 / 1.2, 1.0, 1.2)):
            for name in sorted(colours):
                lab, wb, pixel = developed(cam, name, tint)
                assert all(0.9 - 1e-6 <= g <= 1.1 + 1e-6 for g in wb)  # the clamp holds, whatever the patch
                if reads_palette:  # and in the plain camera the tint pushes the gains to either edge
                    assert sorted(wb)[0] == pytest.approx(0.9) and sorted(wb)[-1] == pytest.approx(1.1)
                nearest = min(colours, key=lambda other: float(np.linalg.norm(lab - refs[other])))
                assert nearest == name, (name, tint, nearest, pixel)


def test_sharpen_by_hand():
    """Contract (`sharpen`): a flat image is unchanged, and a one-pixel impulse of 1 on 0 becomes
    1 + 0.25 (1 - 0.159241) = 1.210190 before the 8 bits (by hand: the blur is a Gaussian of sigma 1 over 7 taps,
    weights exp(-d^2 / 2) for d from -3 to 3 over their sum 2.505952, so the centre tap is 0.399050 and the
    two-dimensional centre weight 0.399050^2 = 0.159241). Regression caught: a sharpen stronger or weaker than
    declared."""
    flat = np.full((9, 9, 3), 0.4)
    assert np.array_equal(C.sharpen(flat, 0.25), flat)
    impulse = np.zeros((9, 9, 3))
    impulse[4, 4] = 1.0
    out = C.sharpen(impulse, 0.25)
    assert out[4, 4, 0] == pytest.approx(1.210190, abs=1e-6)
    # a neighbour: 0 - 0.25 blur, the blur being the centre tap 0.399050 times the next tap 0.242036
    assert out[4, 5, 0] == pytest.approx(-0.25 * 0.399050 * 0.242036, abs=1e-6)


def test_denoise_leaves_correlated_noise_and_smooths_luma_and_colour_apart():
    """Contract (`denoise`, a stated modelling choice and not a robot's value; `capture` applies it after the exposure):
    the luma of the sensor's noise (0.2126 R + 0.7152 G + 0.0722 B) and its colour part (what has no luma) are
    smoothed by Gaussians of their own sigmas. By hand the Gaussian of sigma 1 over 7 taps has weights 1, 0.606531,
    0.135335, 0.011109 (d = 0 to 3) over their sum 2.505952, whose squares sum to 0.28227 in one dimension and
    0.079679 in two, so white noise falls to its square root, 0.2823, of its standard deviation, and its neighbours
    correlate (lag 1) by sum w_i w_(i+1) / sum w_i^2 = 1.380239 / 1.772636 = 0.7786. With sigmas (1, 0) the luma noise
    is 0.2823 of what it was and the colour part exactly as it was; with (0, 1) the luma is exactly as it was and the
    colour falls to 0.2823; (0, 0) hands the field back; and `capture` takes the sigmas from the camera: the luma
    noise of a flat 0.1 field is 0.2823 of the same field's with the denoiser off. Regression caught: a denoiser that
    leaves white grain (no correlated residue), smooths luma and colour together, or never reaches the frames."""
    noise = np.random.default_rng(2).normal(0.0, 1.0, (300, 300, 3))
    inner = (slice(10, -10), slice(10, -10))
    colour_b = lambda n: n[..., 2] - C.luminance(n)  # noqa: E731
    luma_only = C.denoise(noise, 1.0, 0.0)
    assert C.luminance(luma_only)[inner].std() / C.luminance(noise)[inner].std() == pytest.approx(0.2823, rel=0.03)
    assert colour_b(luma_only) == pytest.approx(colour_b(noise), abs=1e-12)
    y = C.luminance(luma_only)[inner]
    lag1 = ((y[:, 1:] - y.mean()) * (y[:, :-1] - y.mean())).mean() / y.var()
    assert lag1 == pytest.approx(0.7786, abs=0.02)
    colour_only = C.denoise(noise, 0.0, 1.0)
    assert C.luminance(colour_only) == pytest.approx(C.luminance(noise), abs=1e-12)
    assert colour_b(colour_only)[inner].std() / colour_b(noise)[inner].std() == pytest.approx(0.2823, rel=0.03)
    assert np.array_equal(C.denoise(noise, 0.0, 0.0), noise)
    plain = replace(LITERAL_ISP, vignette=0.0, target=0.1)
    cam, cam_denoised = literal_camera(160, 120, plain), literal_camera(160, 120, replace(plain, denoise_luma_px=1.0))
    flat = np.full((120, 160, 3), 0.1)
    raw, _ = C.capture(flat, KEY, cam)
    smooth, _ = C.capture(flat, KEY, cam_denoised)
    ratio = C.luminance(smooth)[10:-10, 10:-10].std() / C.luminance(raw)[10:-10, 10:-10].std()
    assert ratio == pytest.approx(0.2823, rel=0.05)


def test_glare_scatters_a_share_of_the_light_and_keeps_the_energy():
    """Contract (`glare`): `fraction` of the light is scattered over the frame and the rest stays, so the energy is
    kept. A flat field is unchanged (what scatters in is what scatters out); 0 hands the array back; a single bright
    pixel in the middle of a dark frame keeps at least (1 - fraction) of its light and no more than all of it, lights
    a pixel 20 px away, and the total of the frame stays within 0.5%; and a half of the frame at 100 times the other
    lifts the dark half and lowers the bright one. Regression caught: a veil that adds light (a brighter frame for a
    window), a flat field that is tinted, or a glare that leaves the dark around a window as it was."""
    flat = np.full((60, 80, 3), 2.0)
    assert C.glare(flat, 0.05) == pytest.approx(flat, abs=1e-12)
    assert C.glare(flat, 0.0) is flat
    dark = np.zeros((120, 160, 3))
    dark[60, 80] = 1000.0
    veiled = C.glare(dark, 0.05)
    assert 950.0 <= veiled[60, 80, 0] < 1000.0 and veiled[60, 100, 0] > 0.0
    assert veiled.sum() == pytest.approx(dark.sum(), rel=0.005)
    window = np.full((120, 160, 3), 0.1)
    window[:, 80:] = 10.0
    lit = C.glare(window, 0.05)
    assert lit[:, :80].mean() > window[:, :80].mean() and lit[:, 80:].mean() < window[:, 80:].mean()
    assert lit.sum() == pytest.approx(window.sum(), rel=0.005)


def test_jpegs_carry_no_metadata():
    """Contract (`encode_jpeg`, `develop`): a developed JPEG holds the JFIF header and no comment and no APP1 to APP15
    segment (no EXIF, no ICC profile, nothing the renderer or Pillow might stamp), and it decodes at the camera's
    height and width. Regression caught: truth tokens or render times in image metadata."""
    jpeg, _ = C.develop(synthetic_frame(3), key=KEY, camera=literal_camera())
    markers, i = [], 2
    while jpeg[i] == 0xFF and jpeg[i + 1] != 0xDA:
        markers.append(jpeg[i + 1])
        i += 2 + int.from_bytes(jpeg[i + 2:i + 4], "big")
    assert markers[0] == 0xE0 and 0xFE not in markers and not any(0xE1 <= m <= 0xEF for m in markers[1:])
    assert Image.open(io.BytesIO(jpeg)).size == (64, 48)


def test_the_tone_curve_is_monotone():
    """Contract (`tone_map`, structure only: no value is pinned): over greys from 0 to 60,000 (the linear pass's clip),
    at no contrast and at a contrast of 1.6, with and without a black level, the display value never falls (to the
    working precision of the arithmetic), 0 maps to the black level (0 where there is none) and the curve reaches
    255. Regression caught: a curve that folds (a highlight developed darker than a dimmer one) or never reaches
    white."""
    greys = np.concatenate([[0.0], np.geomspace(1e-6, 60000.0, 4000)])
    for contrast, black in ((1.0, 0.0), (1.6, 0.0), (1.6, 0.0025)):
        out = C.tone_map(np.repeat(greys[:, None], 3, axis=1), contrast, black)[:, 0]
        assert out[0] == pytest.approx(float(C.srgb_transfer(np.array(black))), abs=1e-12)
        assert np.all(np.diff(out) >= -1e-12), (contrast, black)
        assert C.to_8bit(out)[-1] == 255


def test_the_tone_curve_by_hand():
    """Contract (`tone_luminance`): the contrast is a power about the grey card (0.18 stays 0.18 at any contrast; 0.36
    at 1.5 is 0.18 x 2^1.5 = 0.509117), v up to the knee (0.75) is as it is, and above it the shoulder is
    k + (1 - k)(1 - exp(-(v - k) / (1 - k))): v 1.75 gives 0.75 + 0.25 (1 - e^-4) = 0.995421, and a huge v gives 1
    and nothing above it. Regression caught: a pivot that moves with the contrast, a shoulder that is not continuous
    at its knee, or one that passes 1."""
    assert C.tone_luminance(np.array([0.18]), 1.6)[0] == pytest.approx(0.18)
    assert C.tone_luminance(np.array([0.36]), 1.5)[0] == pytest.approx(0.509117, abs=1e-6)
    assert C.tone_luminance(np.array([0.75]), 1.0)[0] == pytest.approx(0.75)
    assert C.tone_luminance(np.array([0.75 + 1e-9]), 1.0)[0] == pytest.approx(0.75, abs=1e-8)
    assert C.tone_luminance(np.array([1.75]), 1.0)[0] == pytest.approx(0.995421, abs=1e-6)
    assert C.tone_luminance(np.array([1e9]), 1.0)[0] == 1.0


def test_a_bright_warm_light_rolls_off_to_white_and_a_colour_keeps_its_ratios():
    """Contract (`tone_map`): a pixel that would pass the display's range is desaturated towards the grey of its own
    luminance until it fits, so a warm lamp reaches white in every channel and not a peach disc with its red alone at
    255. By hand, (3.0, 1.5, 0.6) at no contrast has luminance 1.75392, the shoulder takes it to 0.995492 and the colour
    ratios to (1.70272, 0.85136, 0.34054); the red passes 1, so the pixel fits at 0.006374 of its chroma (0.004508 /
    0.707228), which leaves (1.0, 0.994573, 0.991317) and the 8 bits (255, 254, 254). A colour in the middle of the
    range keeps its ratios and takes its luminance from the curve: (0.12, 0.06, 0.03) at contrast 1.5, in
    display-linear light (the sRGB transfer undone), stays 2 : 1 : 0.5 with luminance 0.18 (y / 0.18)^1.5, y = 0.07059.
    Black sits on its floor and white stays white. Regression caught: a lamp developed to a peach disc, or a contrast
    that turns hues."""
    lamp = C.tone_map(np.array([[3.0, 1.5, 0.6]]))
    assert C.to_8bit(lamp)[0].tolist() == [255, 254, 254]
    mid = C.tone_map(np.array([[0.12, 0.06, 0.03]]), 1.5)[0]
    linear = np.where(mid <= 0.04045, mid / 12.92, ((mid + 0.055) / 1.055) ** 2.4)
    assert linear[0] / linear[1] == pytest.approx(2.0, rel=1e-9)
    assert linear[1] / linear[2] == pytest.approx(2.0, rel=1e-9)
    y = 0.2126 * 0.12 + 0.7152 * 0.06 + 0.0722 * 0.03
    assert 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2] == pytest.approx(0.18 * (y / 0.18) ** 1.5)
    assert C.tone_map(np.zeros((1, 3)), 1.6, 0.0025)[0, 0] == pytest.approx(12.92 * 0.0025)
    assert C.tone_map(np.full((1, 3), 1e5), 1.6, 0.0025)[0] == pytest.approx([1.0, 1.0, 1.0], abs=1e-12)
