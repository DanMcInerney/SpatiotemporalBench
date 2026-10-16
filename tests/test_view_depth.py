"""The depth picture `stbench view` draws for a `detections` stream: a refusal of an empty range. (The gray rule itself
is tested with the stream in test_view_frames.py.)"""

from __future__ import annotations

import numpy as np
import pytest

from stbench.camera import Camera
from stbench.view_depth import depth_view

CAM = dict(width=3, height=1, fx=1.0, fy=1.0, cx=1.0, cy=0.0, depth_unit="mm")


@pytest.mark.parametrize("near, far", [(5.0, 5.0), (5.0, 2.0)])
def test_depth_view_refuses_an_empty_range(near, far):
    """Contract: a camera whose far bound is not beyond its near bound has no gray rule to draw by, so the picture is
    refused with a ValueError that names the range. Regression caught: a divide by zero turned into a picture of NaNs
    cast to 0 (a black frame that looks like a sensor reading nothing)."""
    with pytest.raises(ValueError, match="range is empty"):
        depth_view(np.ones((1, 3), dtype=np.uint16), Camera(**CAM, near_m=near, far_m=far))
