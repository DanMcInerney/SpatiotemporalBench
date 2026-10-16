"""The colour a render shows a thing in, as a catalog colour word (0.5.0-control; review defect 1; VISION rule 10).

A catalog says a remote is "black"; the picture of it, lit and metered by the robot's camera, may show a silver one. A
question that names a thing by its colour is fair only if the picture bears the word out and no other sighted thing the
picture shows in that colour could be taken for it. The visibility log (`visibility.py`) holds, per sighting,
`rgb_med`: the median sRGB over the thing's instance mask in the delivered JPEG. This module names that colour with
the palette's twelve words and says when it cannot, so `World` (`questions_world`) can screen on what was shown.

The rule, in CIELAB (D65), which separates lightness `L` from chroma `C = hypot(a, b)` and hue `h` as a viewer does:
- **Achromatic** (`C < ACHROMATIC_C`): black below `BLACK_MAX_L`, white from `WHITE_MIN_L`, else grey.
- **Chromatic**: the hue `h` picks a family from `HUE_EDGES` (red, orange, yellow, green, blue, purple, pink). In the
  warm families (red, orange, yellow) a colour under `VIVID_C` chroma is a muted one: brown when darker than
  `LIGHT_WARM_L`, else beige (orange and yellow) or pink (red). Every other chromatic colour keeps its family's word.
- **The neutrals are one group when two things are compared** (`could_be`): a thing that may be taken for any neutral
  word may be taken for all three, so a "white" laptop is not asked beside a "grey" one. A thing's own picture still has
  to show its own word (`held`).
- **A colour is named only when it is steady.** The sample is probed at `MARGIN_DE` (CIELAB units, a little over a just
  noticeable difference) along each Lab axis and each diagonal; if the probes name more than one word the answer is that
  set of words ("no confident class", each word the sample may be taken for), else the one word.

Where the thresholds come from. They follow the palette's labels as the robot's camera renders them, measured over the
sightings of at least 1,024 px in the 0.5.0 development release (46 scripts: 12,100 of carried things, 4,000 of light
ones, 13,000 of clutter), not the textbook values: the camera meters a frame's log-average luminance to 0.18, middle
grey, which is L 49.5, and the rooms are dim, so a white thing's median L is 60, a grey one's 28-34 and a black one's
18, with wide overlaps.
- `WHITE_MIN_L` is that metered mean: a neutral lighter than the frame's average is white. `BLACK_MAX_L` is two stops
  under it (Y 0.04, L 24). Between them is grey.
- The margin leaves a neutral label's overlapping tails unnamed (a "no confident class" outcome) rather than guessing.
- Hue edges are the midpoints of the labels' hue medians (red 30, orange 50, yellow 93, green 141, blue 296, purple 315,
  pink 354 degrees); no palette word lies between green and blue, so a teal at 220 degrees is the margin's.
- `VIVID_C` lies between a brown's chroma (median 11, 95th percentile 15) and a red's or orange's (25th percentile 27
  and 20). `ACHROMATIC_C` is over the neutrals' chroma (95th percentile 4 to 9) and under a beige's (median 13).
The per-label table of what the labels render as is in DECISIONS ("A question names a thing only by a colour its render
shows").

Plain arithmetic over three numbers; it imports nothing of the benchmark, so the agent side may use it.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from functools import lru_cache

ACHROMATIC_C = 8.0  # chroma under which a colour has no hue to name; neutrals render at 0-9, a muted beige at 7-20
BLACK_MAX_L = 24.0  # lightness under which a neutral is black (black labels render at a median L of 18, grey at 28-34)
WHITE_MIN_L = 50.0  # from which a neutral is white: the metered mean (grey's 75th percentile 39, white's 25th 42)
VIVID_C = 22.0  # chroma under which a warm colour is muted: a brown's 95th percentile is 15, a red's 25th is 27
LIGHT_WARM_L = 38.0  # a muted warm colour is brown under this lightness, beige (or pink, if red) from it
MARGIN_DE = 3.0  # the probe radius: a colour counts only if every colour this near it has the same word
# Lab hue angle (degrees) at which each family starts, in order; pink wraps through 0 up to red's start
HUE_EDGES = ((12.0, "red"), (40.0, "orange"), (72.0, "yellow"), (118.0, "green"), (220.0, "blue"), (305.0, "purple"),
             (332.0, "pink"))
WORDS = frozenset({"black", "grey", "white", "beige", "brown", "red", "orange", "yellow", "green", "blue", "purple",
                   "pink"})  # the palette: every colour word the catalogs give a carried thing
# The three neutral words are one group when two things are compared for a look-alike (`could_be`): a person can not
# reliably tell white, grey, silver and black apart on a thing of several materials under varied light, and a median
# over a mask mixes a body with its screen, lid or roof (the 0.5.0 log review's laptops).
NEUTRALS = frozenset({"black", "grey", "white"})

_WHITE = (0.95047, 1.0, 1.08883)  # the D65 white point
_XYZ = ((0.4124564, 0.3575761, 0.1804375), (0.2126729, 0.7151522, 0.0721750), (0.0193339, 0.1191920, 0.9503041))
_ROOT3 = math.sqrt(3)
# the centre, the six Lab axes and the eight diagonals, each at radius 1 (scaled by MARGIN_DE)
_PROBES = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0),
           (0.0, 0.0, -1.0), *((x / _ROOT3, y / _ROOT3, z / _ROOT3) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)))


def _f(t: float) -> float:
    return t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116


def lab(rgb: Sequence[float]) -> tuple[float, float, float]:
    """CIELAB (D65) of an sRGB colour in 0..255 (the delivered JPEG's encoding)."""
    lin = [c / 255 / 12.92 if c / 255 <= 0.04045 else ((c / 255 + 0.055) / 1.055) ** 2.4 for c in rgb]
    x, y, z = (_f(sum(m * c for m, c in zip(row, lin, strict=True)) / w) for row, w in zip(_XYZ, _WHITE, strict=True))
    return 116 * y - 16, 500 * (x - y), 200 * (y - z)


def name(light: float, a: float, b: float) -> str:
    """The word of one Lab colour, by the rule in the module docstring (no margin)."""
    chroma = math.hypot(a, b)
    if chroma < ACHROMATIC_C:
        return "black" if light < BLACK_MAX_L else "white" if light >= WHITE_MIN_L else "grey"
    hue = math.degrees(math.atan2(b, a)) % 360
    family = next((word for edge, word in reversed(HUE_EDGES) if hue >= edge), "pink")  # under the first edge: pink
    if family in ("red", "orange", "yellow") and chroma < VIVID_C:
        return "brown" if light < LIGHT_WARM_L else "pink" if family == "red" else "beige"
    return family


@lru_cache(maxsize=65536)
def _classify(rgb: tuple[int, ...]) -> frozenset[str]:
    light, a, b = lab(rgb)
    return frozenset(name(light + MARGIN_DE * dl, a + MARGIN_DE * da, b + MARGIN_DE * db) for dl, da, db in _PROBES)


def classify(rgb: Sequence[float]) -> frozenset[str]:
    """The words a sRGB colour (0..255 each) may be taken for: one word when every colour within `MARGIN_DE` of it has
    that word (a confident class), else the words of the colours around it (no confident class: the sample is near a
    boundary, and each word is one a viewer might say)."""
    if len(rgb) != 3:
        raise ValueError(f"a colour is three channels, not {len(rgb)}")
    return _classify(tuple(int(round(c)) for c in rgb))


def label_reads(label: object) -> frozenset[str]:
    """The words a thing LABELLED `label` may be taken for when no render says otherwise (an older root's visibility
    log, or a thing never recognisable): the label, and for a neutral all three neutrals. Any other label stands alone,
    as does one outside the palette."""
    return NEUTRALS if label in NEUTRALS else frozenset({label})


def could_be(words: frozenset[str], colour: object) -> bool:
    """Whether a thing that may be taken for `words` (`taken_for` of its frames, or `label_reads`) could be taken for
    something of the colour word `colour`: it is one of the words, or both are neutral (black, grey and white are one
    group). A chromatic colour is hidden only by its own word or a boundary next to it."""
    return colour in words or (colour in NEUTRALS and not words.isdisjoint(NEUTRALS))


def held(classes: Iterable[frozenset[str]]) -> str | None:
    """The colour the picture mostly shows a thing in: the confident class (one word) that the most sightings show,
    alone. A sighting near a boundary (several words) votes for none, so a few of them cannot veto a thing the other
    frames show in one colour. None when no sighting is confident or two words tie. A thing of several colours seen from
    several sides (a red cabin under a white roof) is held by the colour most of its frames show."""
    votes = Counter(next(iter(c)) for c in classes if len(c) == 1).most_common(2)
    if not votes or (len(votes) == 2 and votes[0][1] == votes[1][1]):
        return None
    return votes[0][0]


def taken_for(classes: Iterable[frozenset[str]]) -> frozenset[str]:
    """Every word any of the sightings' classes allows: what the thing may be taken for, seen across its frames."""
    return frozenset().union(*classes)
