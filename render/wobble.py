# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The scalar functions that wrinkle procedural surfaces: smooth, deterministic and bounded.

Pure Python (no `bpy`, no `mathutils`), so a test can import it without Blender. Nothing here is random and nothing
reads the clock: `wave` is a sum of three sines of its arguments, so the same arguments give the same value in every
run, which is what keeps a film's creases, a sack's lumps and a basket's ribs the same mesh in every build (VISION: the
same world in, the same scene out). `smoothstep` and `bump` shape a feature: a step, and a ridge or a groove.
"""

from __future__ import annotations

import math


def wave(a: float, b: float = 0.0, c: float = 0.0, seed: float = 0.0) -> float:
    """A smooth deterministic value in about [-1, 1] that varies over a few units of each argument: three sines at
    incommensurate frequencies. The same arguments always give the same value."""
    return (math.sin(1.7 * a + 2.3 * b + 0.9 * c + seed) + math.sin(3.1 * b - 1.3 * a + 1.9 * c + 2.0 * seed + 1.0)
            + math.sin(2.7 * c + 0.7 * a - 1.1 * b + 3.0 * seed + 2.0)) / 3.0


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = min(1.0, max(0.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def bump(x: float, half_width: float) -> float:
    """A smooth bump of height 1 at x = 0 that falls to 0 at +-half_width (a raised ridge, a groove when negated)."""
    t = abs(x) / half_width
    return 0.0 if t >= 1.0 else 0.5 * (1.0 + math.cos(math.pi * t))
