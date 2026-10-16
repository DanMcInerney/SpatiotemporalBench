"""What the resolution tests share (0.9.0, `resolution_at`): `raised`, the compiler's inputs for a script with the
camera mounted `CAMERA_Z` (1.30 m, the helper's) above the base instead of the vacuum's 0.08 m, so that a pixel at a few
metres is a centimetre or two and the floor of the resolution (`credit_kernel.MIN_RHO_M`, 1 cm) is not what a test
reads. Only the delivered pose rows' camera height changes; the poses, the headings and the pictures are the script's.

The tests' expected values are literals worked out by hand: one pixel of the camera at a floor point at distance d from
the camera's floor point, the camera h above the plane the point lies on, is
hypot((d^2 + h^2) / (h fy), sqrt(d^2 + h^2) / fx) with fx = fy = 381.36115 (every robot's camera), at least 0.01."""

from __future__ import annotations

from dataclasses import replace

from stbench.questions_world import Inputs

CAMERA_Z = 1.30


def raised(inp: Inputs, z: float = CAMERA_Z) -> Inputs:
    """`inp` with every pose row's camera `z` metres up (its x and y as delivered)."""
    rows = [{**row, "camera": {**row["camera"], "pos": [*row["camera"]["pos"][:2], z]}} for row in inp.poses]
    return replace(inp, poses=rows)
