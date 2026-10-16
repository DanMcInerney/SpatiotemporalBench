# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The downtown's worn asphalt, painted once into two tileable images.

Runs inside Blender's own Python and is never imported by `stbench`. No file is read or written: the images are made
here with numpy from fixed seeds (the same pixels every time, on every machine) into `bpy.data.images`, 8-bit sRGB, and
the asphalt shader (`materials_street.asphalt`) reads them with two lookups on object coordinates. The first version
built the same picture out of 140 shader nodes (six noise textures and a score of cell patterns), and a street frame
took longer to render than before the road had patches; the ground is half of every frame, so a cheap shader matters
more here than anywhere.

`asphalt_tile` is a 64 m square at 6.25 cm a pixel: a soft mottling at three scales, rectangular patches of repair in
`PATCH_CELL_M` cells (a seventh of them: fresh and dark, or aged and pale, with a tar-sealed edge), long trench scars
along x or along y (32 m blocks, a street runs one way or the other), cracks along a contour of noise that only the old
areas show and oil stains that are darker and glossier. Its alpha is the roughness. `asphalt_macro` is a 41.3 m square
of slow tonal variation that breaks the tile's repeat (the two periods share no factor). The fine grain the eye sees
close up is the shader's one noise.
"""

from __future__ import annotations

import bpy
import ground_arrays
import numpy as np
from ground_arrays import MACRO_M, TILE_M  # noqa: F401  (the shader reads both)


def _image(name: str, rgba: np.ndarray) -> bpy.types.Image:
    """An 8-bit sRGB image of linear `rgba` (alpha is data: roughness, or 1): the colours of a dark road keep their
    precision in sRGB, and a byte image is a quarter of the memory of a float one and filtered by the GPU itself."""
    rgba = rgba.astype(np.float32)
    c = rgba[..., :3]
    rgba[..., :3] = np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.maximum(c, 0.0031308) ** (1 / 2.4) - 0.055)
    img = bpy.data.images.new(name, rgba.shape[1], rgba.shape[0], alpha=True, float_buffer=False)
    img.colorspace_settings.name = "sRGB"
    img.alpha_mode = "CHANNEL_PACKED"
    img.pixels.foreach_set(rgba.ravel())
    img.update()
    return img


def asphalt_tile() -> bpy.types.Image:
    """The tile (colour in RGB, roughness in alpha), made on first use and found again by name."""
    img = bpy.data.images.get("asphalt_tile")
    if img is None:
        col, rough = ground_arrays.asphalt_arrays()
        img = _image("asphalt_tile", np.dstack([col, rough]))
    return img


def asphalt_macro() -> bpy.types.Image:
    """The macro tone (a factor around 1, in every channel), made on first use and found again by name."""
    img = bpy.data.images.get("asphalt_macro")
    if img is None:
        tone = ground_arrays.macro_array()
        img = _image("asphalt_macro", np.dstack([tone, tone, tone, np.ones_like(tone)]))
    return img
