# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The barcode of a beam label: Code 39, as the bars to draw.

Plain Python (no `bpy`), so a free test can check the table. Code 39 has A-Z, 0-9, "-" and the "*" that starts and
stops a symbol; a character is nine elements, bar and space in turn from a bar, each wide ("w", three modules) or
narrow ("n", one), and every character here has two wide bars and one wide space. `dressing_warehouse_kit.Mesh.barcode`
draws the bars as thin boxes.
"""

from __future__ import annotations

PATTERNS = (
    "nnnwwnwnn wnnwnnnnw nnwwnnnnw wnwwnnnnn nnnwwnnnw wnnwwnnnn nnwwwnnnn nnnwnnwnw wnnwnnwnn nnwwnnwnn "  # 0-9
    "wnnnnwnnw nnwnnwnnw wnwnnwnnn nnnnwwnnw wnnnwwnnn nnwnwwnnn nnnnnwwnw wnnnnwwnn nnwnnwwnn nnnnwwwnn "  # A-J
    "wnnnnnnww nnwnnnnww wnwnnnnwn nnnnwnnww wnnnwnnwn nnwnwnnwn nnnnnnwww wnnnnnwwn nnwnnnwwn nnnnwnwwn "  # K-T
    "wwnnnnnnw nwwnnnnnw wwwnnnnnn nwnnwnnnw wwnnwnnnn nwwnwnnnn "  # U-Z
    "nwnnnnwnw nwnnwnwnn"  # "-" and "*"
)
CODE39 = dict(zip("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-*", PATTERNS.split(), strict=True))


def code39_bars(string: str) -> list[tuple[float, float]]:
    """The bars of a Code 39 symbol of `string` as (start, width) in modules (narrow = 1, wide = 3), characters one
    narrow space apart, framed by the start and stop stars."""
    out, at = [], 0.0
    for ch in f"*{string}*":
        for i, e in enumerate(CODE39[ch]):
            w = 3.0 if e == "w" else 1.0
            if i % 2 == 0:
                out.append((at, w))
            at += w
        at += 1.0
    return out
