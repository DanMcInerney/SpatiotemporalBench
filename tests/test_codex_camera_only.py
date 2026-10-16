"""The reference agent on the camera alone (`codex_system.py` on `rgb_pose`, the default channel;
"what if we just didn't even include a depth image?"), over the fake command (`codex_fake`): one camera image sent
with each frame, as an image block of the reply to the model's pull, and the system prompt that states the camera's
mount and has no word of depth. The scenario of `test_codex_system.py` is the same flow on the gate channel. No model,
no network.
"""

from __future__ import annotations

from codex_kit import ROBOT, Run, jpeg_hash, pulled
from stbench.codex_log import read_log
from stbench.stream_protocol import DEFAULT_CHANNEL

BRIEF = "Your assignment is to clean up."
HANGING = {**ROBOT, "camera": {**ROBOT["camera"], "mount_m": [0.0, 0.0, -0.1]}}


def run_in(tmp_path, name: str, **opts) -> Run:
    """A `Run` in its own new folder under `tmp_path`."""
    (tmp_path / name).mkdir()
    return Run(tmp_path / name, **opts)


def test_a_frame_on_the_default_channel_sends_one_camera_image_in_the_reply_and_no_depth(tmp_path):
    """Contract (the agent is handed the camera's images and no depth): on the default channel
    (`rgb_pose`) each frame is one reply to the model's pull with two blocks, the header text and the frame's camera
    JPEG as an image block, byte for byte (the fake names it by hash) and nothing else; the stream's inbox holds no
    depth file, though a depth path may be in the step; the text names no image (the image follows the header in the
    same reply); and the journal logs only JPEGs, as hashes. Regression caught: a depth image sent on the camera-alone
    channel (a sensor the benchmark withholds), an image for the wrong frame, or a header that still says how many
    images are attached."""
    with Run(tmp_path) as run:
        assert run.channel == DEFAULT_CHANNEL == "rgb_pose"
        run.system.task(0, "2026-03-02T08:00:00", 0.0, BRIEF)
        run.stream([0, 1])
        assert list((run.work / "inbox" / "depth").iterdir()) == []
        run.stream([2])
        run.serve()  # the last frame is held until the next message
        (live,) = run.fake.execs()
    frames = pulled(live)
    assert [c["blocks"] for c in frames] == [["text", "image"]] * 3
    assert [c["images"] for c in frames] == [[jpeg_hash(k)] for k in (0, 1, 2)]
    assert all("image" not in c["text"].lower() and "\n" not in c["text"] and "depth" not in c["text"].lower()
               for c in frames)
    logged = [e["images"] for e in read_log(run.log) if e.get("event") == "served"]
    assert [[image.split(" ")[0] for image in images] for images in logged] == [[], ["<jpeg"], ["<jpeg"], ["<jpeg"]]


def test_the_first_message_states_the_cameras_mount_from_the_robot_file_and_has_no_depth_paragraph(tmp_path):
    """Contract (the review's defects 3 and 4): the first exec's message, the system prompt and the
    first task, tells the camera as the robot knows it, its size and field of view from `camera.json` and its mount from
    the `robot.json` the harness put in `public/` (the kit's: 0.13 m ahead of the base and 0.08 m above it; a camera
    that hangs 0.10 m below it for a flier), and on `rgb_pose` has no word of depth anywhere in it. Regression caught:
    a mount that never reaches the model in a real run (only in the renderer's unit test), a robot file read from the
    wrong place, or the depth paragraph sent to an agent that is sent no depth."""
    with run_in(tmp_path, "ground") as run:
        run.system.task(0, "2026-03-02T08:00:00", 0.0, BRIEF)
        (first,) = run.fake.execs()
    assert ("The camera image is 640 x 480 pixels with a horizontal field of view of 80 degrees. "
            "The camera is mounted 0.13 m ahead of the base and 0.08 m above it.") in first["prompt"]
    assert "depth" not in first["prompt"].lower()
    with run_in(tmp_path, "flier", robot=HANGING) as flier:
        flier.system.task(0, "2026-03-02T08:00:00", 0.0, BRIEF)
        assert "The camera is mounted 0.10 m below the base." in flier.fake.execs()[0]["prompt"]
