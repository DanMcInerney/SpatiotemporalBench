"""The reference agent's renderer on the `detections` channel (`llm_render.py`), the perfect-perception diagnostic: a
frame's detections as text lines in place of its camera image, and the headline channel's messages unchanged by it.

The expected texts are hand-written literals. A change to any of them is a change to what the model reads
(`test_determinism_llm.py` checks only that the bytes repeat across processes).
"""

from __future__ import annotations

import copy
import re

import pytest

from stbench import leak
from stbench.llm_render import detection_line, frame_message, system_prompt
from test_llm_render import EXPECTED_FRAMES, EXPECTED_IMAGES, STEPS, public

MUG = {"d": "9f3a01bc", "kind": "mug", "attributes": {"size": "small", "color": "red"},
       "bbox_px": [120, 200, 180.5, 260], "center_cam": [0.1, -0.3, 1.5], "center_map": [2.0, 1.0, 0.9],
       "extent_m": [0.1, 0.1, 0.12]}
DOOR = {"d": "01aa77ee", "kind": "door", "attributes": {}, "bbox_px": [300, 10, 420, 470],
        "center_cam": [0.5, -0.001, 2.5], "center_map": [3.1, 1.2, 1.0], "extent_m": [0.9, 0.05, 2.0],
        "state": {"open": True}}
# What the harness would never show: an entity id behind a detection, a key a row should not carry, a private path.
TRUTH = {"identity": {"9f3a01bc": "E0042", "01aa77ee": "E0007"}, "entity": "E0042", "region": "R3",
         "gold": "C:/data/private/case1/gold.jsonl", "truth": "it is E0042"}

# STEPS as `detections` delivers them: the frames' headers as on `rgb_pose`, no `rgb`, and each frame's detections
# (the second sees nothing), with the privileged `identity` and a row that carries more than the channel gives.
SEEN = [{k: v for k, v in STEPS[0].items() if k != "rgb"} | {"detections": [{**MUG, **TRUTH}, DOOR], **TRUTH},
        {k: v for k, v in STEPS[1].items() if k != "rgb"} | {"detections": [], **TRUTH}]
EXPECTED_DETECTIONS = [
    EXPECTED_FRAMES[0] + "\n"
    "Detections in frame k12 (2):\n"
    "  - mug (color red, size small): map (2.00, 1.00, 0.90), camera (0.10, -0.30, 1.50), "
    "size 0.10 x 0.10 x 0.12 m, box (120, 200) to (180.5, 260) px\n"
    "  - door: map (3.10, 1.20, 1.00), camera (0.50, 0.00, 2.50), size 0.90 x 0.05 x 2.00 m, "
    "box (300, 10) to (420, 470) px, open true",
    EXPECTED_FRAMES[1] + "\nDetections in frame k13: none"]
# `system_prompt(..., "rgb_pose")` up to its answer shapes: the role and the protocol, and nothing of memory.
RGB_PROMPT_HEAD = (
    "You are the brain of a robot that is doing a job. You are told the job, then shown what the robot senses as it "
    "works, and you are asked questions about it.\n\n"
    "Coordinates are metres in the map frame: x points east, y north and z up. A heading is in degrees, "
    "counter-clockwise from +x; a camera pitch is positive looking up. An offset answer is not in the map frame: it "
    "is metres from the camera, along the floor, with ahead along the robot's heading (the heading in each frame's "
    "header, whichever way the camera is pitched) and left to its left; a thing's height is not part of it. The "
    "camera image is 640 x 480 pixels with a horizontal field of view of 80 degrees. The camera is mounted 0.21 m "
    "ahead of the base and 0.64 m above it. The robot's body is a box 50 cm long and 40 cm wide and 30 cm tall, "
    "centred on the base. It drives over anything up to 1.5 cm high and passes under anything 32 cm or more above "
    "the floor. The camera is the robot's only eye on the world: nothing measures a distance for you, so any "
    "distance you give is your own estimate.\n\n"
    "How this works:\n"
    "- You receive what the robot senses by calling next_frame. Each call returns the next thing, in time order: a "
    "task message, one frame, the questions of an ask, or word that the job is over. Call it again when you are ready "
    "for the next; nothing advances until you do.\n"
    "- A task message tells you the clock time a task begins at, how long the robot went unobserved since the last "
    "one, and the assignment.\n"
    "- A frame is its header line and its camera image. The header gives its number k and clock time, where the "
    "robot's base is, the way it heads, how it is moving (its own velocity in the map frame, in metres a second, from "
    "its controller), the pitch of its camera, and its status. Things in the image carry no labels or positions.\n"
    "- Questions come at some points during the job and after it, numbered Q1, Q2, ... A frame may come with "
    "questions about it, which follow the frame in the same reply. Answer each, using only what the robot has "
    "sensed so far, by calling submit_answer once for each question with its label and its answer, before you call "
    "next_frame again.\n"
    "- When next_frame says the job is over, reply with one short line.\n\n"
    "Answers are JSON objects, one shape for each kind of answer:\n")


def test_on_rgb_pose_the_prompt_and_a_frame_are_these_bytes_whatever_the_steps_carry(tmp_path):
    """Contract (the detections arm must not change the headline agent's inputs; DECISIONS 2026-10-10: the prompt says
    only the role and the protocol): on `rgb_pose` the system prompt (whole, up to the answer shapes; for a job that
    asks during the job, whose prompt speaks of questions during it: `asks_during`) and a frame (`EXPECTED_FRAMES`,
    with its image to send) are exactly these texts, whatever the steps also carry (detections, an identity).
    Regression caught: the arm's wording leaking into the picture prompt (a changed sentence), a sentence on memory or
    on what to keep coming back, or a frame that renders a step's detections because it carries some."""
    asking = system_prompt(public(tmp_path), "rgb_pose", asks_during=True)
    assert asking.startswith(RGB_PROMPT_HEAD + "- position: ")
    carrying = [step | {"detections": [MUG], **TRUTH} for step in STEPS]
    assert [frame_message(step, "rgb_pose") for step in carrying] == list(
        zip(EXPECTED_FRAMES, EXPECTED_IMAGES, strict=True))


def test_on_detections_a_frame_is_the_header_and_the_detections_as_text_with_no_image():
    """Contract (the diagnostic: perfect perception, in words): a frame on `detections` is the same header line as on
    `rgb_pose`, a line saying how many detections it holds (or none) and one line a detection (kind, its attributes
    sorted, its place in the map and the camera frame, its size, its box in pixels and, for a door, `open true`), and
    nothing after it (no cue); there is no image to send, an empty frame says `none`, and only the header starts a line
    with `[k` (the leak scan's frame rule). Regression caught: a frame that still names an image it never sends, a
    detection line that starts like a header (the scan would read it as a frame), or a frame with no detections left
    out (the agent could not tell nothing seen from nothing sent)."""
    got = [frame_message(step, "detections") for step in SEEN]
    assert [text for text, _ in got] == EXPECTED_DETECTIONS
    assert [image for _, image in got] == [None, None]
    assert [re.findall(r"^\[k(\d+) ", text, re.M) for text, _ in got] == [["12"], ["13"]]
    assert not [text for text, _ in got if "image" in text.lower()]


def test_a_detection_message_shows_what_the_channel_gives_and_nothing_else():
    """Contract (VISION rules 3, 6; the channel's own fields only): the message shows each detection's kind,
    attributes, boxes, centres and size, a door's state, and nothing of the step or the row beyond them: not the
    detection's key `d` (fresh for each detection, it names nothing), a step's `identity`, a key such as `entity`,
    `gold` or `truth` a row should not carry, an entity id, a region id or a private path. Regression caught: a
    renderer that dumps the row or the step, which would hand the model an entity id or a path into the truth."""
    text = "\n".join(frame_message(step, "detections")[0] for step in SEEN)
    for hidden in (MUG["d"], DOOR["d"], "E0042", "E0007", "R3", "private", "gold", "truth", "entity", "identity",
                   "C:/data"):
        assert hidden not in text, hidden
    assert [rule for rule, rx in leak.RULES.items() if rx.search(text)] == []
    shown = detection_line({**MUG, **TRUTH})
    assert shown == detection_line(MUG) == (
        "  - mug (color red, size small): map (2.00, 1.00, 0.90), camera (0.10, -0.30, 1.50), "
        "size 0.10 x 0.10 x 0.12 m, box (120, 200) to (180.5, 260) px")
    assert detection_line(DOOR).endswith("camera (0.50, 0.00, 2.50), size 0.90 x 0.05 x 2.00 m, "
                                         "box (300, 10) to (420, 470) px, open true")  # -0.001 is never "-0.00"


def test_the_channel_decides_how_a_frame_is_rendered_and_a_step_missing_its_input_is_an_error():
    """Contract: how a frame is rendered is the channel's, not the step's keys: a step with a picture but no detections
    is refused on `detections` and one with detections but no picture on a picture channel, a detection lacking a field
    and a name that is no channel are errors before any exec, and an empty detection list is a valid frame. Regression
    caught: a silent fallback from one reading to the other (an arm that is said to run on detections but is shown
    pictures, or the reverse), or a half-built line sent to the model."""
    with pytest.raises(ValueError, match="step k12 has no detections"):
        frame_message(STEPS[0], "detections")
    with pytest.raises(ValueError, match="step k12 has no camera image"):
        frame_message(SEEN[0], "rgb_pose")
    with pytest.raises(ValueError, match="lacks center_map"):
        broken = {k: v for k, v in MUG.items() if k != "center_map"}
        frame_message(copy.deepcopy(SEEN[0]) | {"detections": [broken]}, "detections")
    with pytest.raises(ValueError, match="unknown channel 'rgb'"):
        frame_message(SEEN[0], "rgb")
    assert frame_message(SEEN[1], "detections")[0].endswith("Detections in frame k13: none")


def test_on_detections_the_prompt_has_no_images_and_says_how_detections_read(tmp_path):
    """Contract (the diagnostic's inputs are told, not guessed): the system prompt on `detections` says the images are
    not sent and what a detection line holds (the camera frame's axes, the units, that nothing says which thing is
    which), has no mention of attached images or of depth at all (0.6.0: the robot has a camera and no depth sensor,
    and the old "depth images are not sent to you" sentence is gone with the paragraph it answered), and still holds the
    role, the frame of reference, the closing line and every answer shape. Regression caught: the prompt telling the
    model of images it is never sent, or a depth sentence kept on the one channel that was not sent a depth image."""
    prompt = system_prompt(public(tmp_path), "detections")
    for text in ("You are the brain of a robot that is doing a job.",
                 "You are not sent the images, only the detections in them, with boxes in those pixels.",
                 "- A frame is its header line and the detections found in it. The header gives its number k",
                 "(x right, y down and z forward from the camera, in metres)",
                 "It does not say which thing is which: one thing seen in two frames is two detections",
                 "- When next_frame says the job is over, reply with one short line.",
                 "Answers are JSON objects, one shape for each kind of answer:", "unknown"):
        assert text in prompt, text
    for text in ("attached", "Images", "grayscale", "Black"):
        assert text not in prompt, text
    assert "depth" not in prompt.lower()
