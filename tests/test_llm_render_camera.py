"""The reference agent on the camera alone (`llm_render.py`; 2026-10-10: "what if we just didn't even include a
depth image? ... the agent has to guess the distance"; 0.6.0: the robot has a camera and no depth sensor): the camera
stated as the robot knows it, the offset frame and its units, no word of depth, one image a frame. The expected texts
are hand-written literals with made-up numbers (no robot of the library).
"""

from __future__ import annotations

import copy

import pytest

from codex_kit import CREDIT
from stbench.llm_render import frame_message, mount_sentence, system_prompt
from test_llm_render import BODY, CAMERA, EXPECTED_FRAMES, STEPS, public

GROUND = {"id": "made_up_ground", "flies": False, "body": BODY, "camera": {"mount_m": [0.21, 0.0, 0.64]}}
HANGING = {"id": "made_up_flier", "flies": True, "body": {**BODY, "shape": "disc", "width_m": 0.5, "climb_m": 0.0},
           "camera": {"mount_m": [0.0, 0.0, -0.15]}}


def test_the_mount_sentence_is_built_from_the_numbers_of_the_robot_file():
    """Contract (the review's defect 3: the camera's mount never reached the agent): `robot.json`'s `camera.mount_m`
    ([forward, left, up] from the base, metres) becomes one sentence, each non-zero axis a distance to two decimals and
    a direction by its sign, the first "of the base" and the others "of it": a camera ahead of and above the base, one
    that hangs below it, one behind and to the side, and one at the base; an offset under half a centimetre is no
    offset; anything but three numbers is refused. Hand sentences for made-up numbers. Regression caught: a sign read
    the wrong way (a hanging camera said to be above the base puts every distance estimate a drone makes off by twice
    the hang), an axis dropped, a constant instead of the file's numbers, or a bad `robot.json` passing as a camera
    at the base."""
    assert mount_sentence([0.21, 0.0, 0.64]) == "The camera is mounted 0.21 m ahead of the base and 0.64 m above it."
    assert mount_sentence([0.0, 0.0, -0.15]) == "The camera is mounted 0.15 m below the base."
    assert mount_sentence([-0.3, 0.07, -0.15]) == (
        "The camera is mounted 0.30 m behind the base, 0.07 m to the left of it and 0.15 m below it.")
    assert mount_sentence([0, -0.2, 1]) == "The camera is mounted 0.20 m to the right of the base and 1.00 m above it."
    assert mount_sentence([0.004, -0.002, 0.0]) == "The camera is mounted at the base."
    for bad in (None, "0.1 0 0", [0.1, 0.0], [0.1, 0.0, 0.2, 0.3], [0.1, "0", 0.2], [True, 0.0, 0.2], {"x": 1}):
        with pytest.raises(ValueError, match=r"mount_m is \[forward, left, up\]"):
            mount_sentence(bad)


@pytest.mark.parametrize("channel", ["rgb_pose", "detections"])
def test_the_system_prompt_states_the_camera_from_the_files_of_the_robot_on_every_channel(tmp_path, channel):
    """Contract (the review's defect 3; VISION rule 8: a robot is a profile): the system prompt gives the camera as
    the robot knows it: the image size and horizontal field of view of `camera.json`, then, right after, the mount
    sentence from `robot.json`, once, on each channel the agent reads. Other numbers in the files give other words, a
    hanging camera is said to hang, nothing names the robot, and a `robot.json` with no mount is an error rather than a
    prompt without one. Regression caught: a mount that reaches only some channels, a camera sentence from a constant,
    the robot's id in the prompt (a special case in disguise), or a prompt silently missing the mount."""
    ground = system_prompt(public(tmp_path, robot=GROUND), channel)
    lens = "The camera image is 640 x 480 pixels with a horizontal field of view of 80 degrees. "
    assert (lens + "The camera is mounted 0.21 m ahead of the base and 0.64 m above it.") in ground
    assert ground.count("The camera is mounted") == 1 and "made_up" not in ground
    hanging = system_prompt(public(tmp_path / "h", {**CAMERA, "width": 320, "height": 240}, robot=HANGING), channel)
    assert "320 x 240 pixels" in hanging and "The camera is mounted 0.15 m below the base." in hanging
    assert "m ahead of the base" not in hanging and "above it" not in hanging and "made_up" not in hanging
    with pytest.raises(ValueError, match="mount_m"):
        system_prompt(public(tmp_path / "n", robot={"id": "made_up", "camera": {"width": 640}}), channel)


def test_the_prompt_says_the_camera_is_the_only_eye_and_has_no_word_of_depth_on_the_camera_channel(tmp_path):
    """Contract (FD-0.6.0 §1: camera only; 2026-10-10: no depth image for the agent): the system prompt on the
    default channel `rgb_pose` says the camera is the robot's only eye on the world and that any distance the agent
    gives is its own estimate, once, right after the mount and body sentences; it has no depth paragraph and no mention
    of depth, a depth image, gray levels or a range sensor's black, and says a frame is its header line and its camera
    image. Regression caught: the depth paragraph (or its "away is white") coming back in a prompt for an
    agent that is sent no depth, which would send it looking for an image it does not get, or a prompt that leaves the
    agent believing a sensor measures distance for it."""
    prompt = system_prompt(public(tmp_path), "rgb_pose", asks_during=True, **CREDIT)
    assert "depth" not in prompt.lower() and "gray" not in prompt.lower() and "Black" not in prompt
    assert ("The camera is mounted 0.21 m ahead of the base and 0.64 m above it. The robot's body is a box 50 cm long "
            "and 40 cm wide and 30 cm tall, centred on the base. It drives over anything up to 1.5 cm high and passes "
            "under anything 32 cm or more above the floor. The camera is the robot's only eye on the world: nothing "
            "measures a distance for you, so any distance you give is your own estimate.") in prompt
    assert prompt.count("only eye") == 1
    assert "- A frame is its header line and its camera image. The header gives its number k and clock time," in prompt


def test_the_prompt_states_the_offset_frame_and_its_units_beside_the_offset_shape(tmp_path):
    """Contract (FD-0.6.0 §4: the frame is named; plan §3 `PROMPT_SHAPES["offset"]`): the prompt tells, right after the
    map frame, that an offset is not in the map frame but metres from the camera along the floor, ahead along
    the robot's heading (whichever way the camera is pitched) and left to its left, with the height of a thing no part
    of it, and its answer-shape line says the same in its own words (a ahead, l to its left, negative behind or to the
    right). Both are on every channel the agent reads. Regression caught: an offset answered in map axes or in metres
    along the camera's tilted axis (signs and units guessed by the model: ahead and left swapped, a height folded in; a
    drone's camera pitched straight down has no "way it faces" along the floor), or the shape without its frame."""
    want = ("An offset answer is not in the map frame: it is metres from the camera, along the floor, with ahead along "
            "the robot's heading (the heading in each frame's header, whichever way the camera is pitched) and left to "
            "its left; a thing's height is not part of it.")
    for channel in ("rgb_pose", "detections"):
        prompt = system_prompt(public(tmp_path / channel), channel)
        assert "a camera pitch is positive looking up. " + want + " The camera image is" in prompt
        assert ('- offset: {"type": "offset", "ahead_m": a, "left_m": l} (metres from the camera along the floor: a '
                "ahead of it along the robot's heading, l to its left; negative is behind it or to its right)"
                ) in prompt


def test_an_rgb_pose_frame_is_its_header_and_returns_only_the_camera_image_to_send():
    """Contract (2026-10-10): a frame on `rgb_pose` is its header line, and the one image to send is the path of
    the step's `rgb`, whether or not the step also carries a depth path (no channel sends one to the agent, and the
    renderer would not send one if it did). A step with no camera image is an error. Regression caught: a depth image
    sent on the camera-alone channel (the agent would again be handed a sensor the benchmark withholds), or an image
    line left in the text."""
    got = [frame_message(step, "rgb_pose") for step in STEPS]
    assert got == [(EXPECTED_FRAMES[0], "inbox/rgb/000012.jpg"), (EXPECTED_FRAMES[1], "inbox/rgb/000013.jpg")]
    bare = [{k: v for k, v in step.items() if k != "depth"} for step in copy.deepcopy(STEPS)]
    assert [frame_message(step, "rgb_pose") for step in bare] == got   # the same with no depth key at all
    with pytest.raises(ValueError, match="step k12 has no camera image: the reference agent reads it on 'rgb_pose'"):
        frame_message({k: v for k, v in bare[0].items() if k != "rgb"}, "rgb_pose")
    assert "depth" not in "".join(text for text, _ in got)
