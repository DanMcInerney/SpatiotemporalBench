"""Every family's `needs` against the channels a run asks it on (0.5.0-control: the headline channel sends no depth)."""

from __future__ import annotations

import pytest

from stbench.questions_text import FAMILIES
from stbench.stream_protocol import DEFAULT_CHANNEL, GATE_CHANNEL, provides


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_a_family_needs_only_what_the_camera_channel_provides(name):
    """Contract: `FamilySpec.needs` is what a system needs to answer (the gold may read more, the depth), so every
    registered family's `needs` lies within what the headline channel provides (`rgb_pose`: the picture, the pose and
    the status, `see` computed from the picture) and within what the gate channel provides (`detections`), whatever
    its gold reads. The stream asks a question only when its `needs` are within the channel's, so a family that names
    the depth is never asked on `rgb_pose`. Regression caught: a family left with "depth" in `needs` after the headline
    channel dropped the depth image, which loses every one of its questions on the camera-only runs (the AMR's
    `out_of_view`: 185 questions on release-b) and shows nowhere but in a run."""
    needs = set(FAMILIES[name].needs)
    assert needs <= provides(DEFAULT_CHANNEL), sorted(needs - provides(DEFAULT_CHANNEL))
    assert needs <= provides(GATE_CHANNEL), sorted(needs - provides(GATE_CHANNEL))
