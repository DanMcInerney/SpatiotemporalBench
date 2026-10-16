# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The downtown's window interiors, painted once into small images: an atlas of window panes and one of shop glazings.

Runs inside Blender's own Python and is never imported by `stbench`. No file is read or written: the atlases are made
here with numpy from a fixed seed (the same pixels every time, on every machine) into `bpy.data.images`, 8-bit sRGB, so
the facade shader (`facade_shaders`) picks a variant per window and reads it with one image lookup. That costs a
fraction of building the same picture out of hundreds of math nodes in the shader, which the first version did (a facade
frame took 70% longer to render than before the windows had interiors).

A cell of the window atlas is a pane as seen from the street, painted on the facade's plane (a cheap interior card with
no parallax): the sky in the glass (lighter towards its head), over blinds drawn part way down, curtains at the sides,
an open room (a pale back wall, a lit ceiling, a darker floor, sometimes a desk), a plainly reflective pane or a lit
room; a dark frame, a mullion in some, a shadow under the lintel. A cell of the shop atlas is the glazing of one bay:
lit retail with goods on shelves, a dim cafe with pendant lamps, a rolled-down shutter, papered-over glass or a
reflective lobby. The alpha channel is the share of the pixel that gives off light (a lit room, a lit shop).
"""

from __future__ import annotations

import bpy
import numpy as np
import window_paint
from window_paint import SHOPS, WINDOWS  # noqa: F401  (the shader reads both)


def _image(name: str, rgba: np.ndarray) -> bpy.types.Image:
    """An 8-bit sRGB image of linear `rgba` (alpha is data: emission): a quarter of the memory of a float image and the
    GPU's own texture filtering and decoding, which made a frame 20% faster than float pixels."""
    rgba = rgba.copy()
    c = rgba[..., :3]
    rgba[..., :3] = np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.maximum(c, 0.0031308) ** (1 / 2.4) - 0.055)
    img = bpy.data.images.new(name, rgba.shape[1], rgba.shape[0], alpha=True, float_buffer=False)
    img.colorspace_settings.name = "sRGB"
    img.alpha_mode = "CHANNEL_PACKED"
    img.pixels.foreach_set(rgba.ravel())
    img.update()
    return img


def window_atlas() -> bpy.types.Image:
    """The window atlas (`WINDOWS` variants), made on first use and found again by name."""
    return bpy.data.images.get("window_atlas") or _image("window_atlas", window_paint.window_array())


def shop_atlas() -> bpy.types.Image:
    """The shop atlas (`SHOPS` variants), made on first use and found again by name."""
    return bpy.data.images.get("shop_atlas") or _image("shop_atlas", window_paint.shop_array())
