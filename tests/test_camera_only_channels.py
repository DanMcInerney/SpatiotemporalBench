"""The agent has a camera and no depth sensor (0.6.0): `rgbd_pose` is not a channel, and no command accepts it."""

from __future__ import annotations

import pytest

from stbench import cli
from stbench.stream_protocol import CHANNELS, DEFAULT_CHANNEL, GATE_CHANNEL, provides


def test_only_the_gate_channel_sends_depth_and_the_agents_channel_is_the_camera():
    """Contract (VISION rule 9, 0.6.0): the channels are `rgb_pose`, which the agents run on and which sends no depth,
    and `detections`, the instruments' gate channel, which does; `provides` says the same of what a question may need.
    Regression caught: a channel that hands an agent the recorded depth image (the camera-only robot gets a depth
    sensor back), or the default moving off the camera."""
    assert DEFAULT_CHANNEL == "rgb_pose" and GATE_CHANNEL == "detections"
    assert set(CHANNELS) == {"rgb_pose", "detections"}
    assert [name for name, sends in CHANNELS.items() if "depth" in sends] == [GATE_CHANNEL]
    assert "depth" not in provides("rgb_pose")


@pytest.mark.parametrize("argv", [
    ["run", "all_unknown", "somewhere", "--channel", "rgbd_pose"],
    ["eval", "all_unknown", "--data", "somewhere", "--out", "nowhere", "--channel", "rgbd_pose"],
])
def test_a_run_asking_for_rgbd_pose_is_refused_before_anything_runs(argv, capsys):
    """Contract: `stbench run` and `stbench eval` refuse `--channel rgbd_pose` with exit 2 and a message naming the
    channels there are, before reading a dataset. Regression caught: the depth image reaching an agent through a
    command that still lists the retired channel."""
    with pytest.raises(SystemExit) as refused:
        cli.main(argv)
    assert refused.value.code == 2
    err = capsys.readouterr().err
    assert "rgbd_pose" in err and "rgb_pose" in err and "detections" in err
