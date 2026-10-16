"""The reference agent's renderer: what the model is told, and nothing it may not be told (`llm_render.py`).

The expected texts are hand-written literals. A change to any of them is a change to what the model reads
(`test_determinism_llm.py` checks only that the bytes repeat across processes).
"""

from __future__ import annotations

import copy
import json
import re

import pytest

from stbench import mapspec, stream_protocol
from stbench.agent_tools import unanswered
from stbench.answers import PROMPT_SHAPES, TYPES, Malformed, parse_answer
from stbench.llm_render import (
    END_MESSAGE,
    RESUME_MESSAGE,
    UNKNOWN_RULE,
    ask_message,
    frame_message,
    labelled,
    system_prompt,
    task_message,
)

CAMERA = {"width": 640, "height": 480, "hfov_deg": 80.0, "fx": 381.3, "fy": 381.3, "cx": 320.0, "cy": 240.0,
          "depth_unit": "mm", "near_m": 0.01, "far_m": 4.0}
BODY = {"shape": "rect", "length_m": 0.5, "width_m": 0.4, "height_m": 0.3, "climb_m": 0.015, "clearance_m": 0.05,
        "under_margin_m": 0.04}   # a made-up body: no robot of the library
ROBOT = {"id": "made_up", "flies": False, "body": BODY,
         "camera": {"width": 640, "height": 480, "hfov_deg": 80.0, "mount_m": [0.21, 0.0, 0.64]}}
STATUS = {"phase": "cleaning", "task": 0, "battery_pct": 97, "brush": True}


def public(tmp_path, camera=CAMERA, robot=ROBOT, **extra):
    """A `public/` folder with `camera.json`, `robot.json` and whatever else is written into it (`name=text`)."""
    folder = tmp_path / "public"
    folder.mkdir(parents=True)
    (folder / "camera.json").write_text(json.dumps(camera), encoding="utf-8")
    (folder / "robot.json").write_text(json.dumps(robot), encoding="utf-8")
    for name, text in extra.items():
        (folder / name).write_text(text, encoding="utf-8")
    return folder


def step(k, clock, pos, yaw, pitch, status=STATUS, **extra):
    return {"k": k, "clock": clock, "pose": {"pos": pos, "rot": [1.0, 0.0, 0.0, 0.0], "yaw_deg": yaw},
            "camera": {"pos": [9.0, 9.0, 9.0], "rot": [1.0, 0.0, 0.0, 0.0], "pitch_deg": pitch}, "status": status,
            "rgb": f"inbox/rgb/{k:06d}.jpg", "depth": f"inbox/depth/{k:06d}.png", **extra}


# Frame 12 is the plain case, at rest; frame 13 has a base at x = -0.001 (never "-0.00"), a heading of -0.4
# (printed to 0.1 deg, as the gold reads it), a velocity of (-0.0, 0.625) m/s (printed to the millimetre a second as
# delivered, never "-0.000"), a pitch of -15.2 (-15 deg) and a status whose keys come in no order (phase and task
# first, the rest sorted, a float without ".0").
# The clocks are delivered ones, to the tenth of a second (`YYYY-MM-DDTHH:MM:SS.s`), printed by the header as given.
STEPS = [step(12, "2026-03-02T08:00:12.0", [5.6, 5.05, 0.0], -90.0, 5.0, velocity=[0.0, 0.0]),
         step(13, "2026-03-02T08:00:13.5", [-0.001, 4.8, 0.0], -0.4, -15.2,
              {"tether_m": 1.5, "brush": False, "task": 1, "phase": "transit", "battery_pct": 96},
              velocity=[-0.0, 0.625])]
EXPECTED_FRAMES = [
    "[k12 2026-03-02T08:00:12.0] base at (5.60, 5.05, 0.00), heading -90.0 deg, velocity (0.000, 0.000) m/s, "
    "camera pitch 5 deg; status: phase cleaning, task 0, battery_pct 97, brush true",
    "[k13 2026-03-02T08:00:13.5] base at (0.00, 4.80, 0.00), heading -0.4 deg, velocity (0.000, 0.625) m/s, "
    "camera pitch -15 deg; status: phase transit, task 1, battery_pct 96, brush false, tether_m 1.5"]
# The path of each step's camera image, which the caller sends as the image block after the text (a step's depth path,
# which no channel sends to an agent, is never returned).
EXPECTED_IMAGES = ["inbox/rgb/000012.jpg", "inbox/rgb/000013.jpg"]


def test_the_system_prompt_holds_the_role_the_frame_the_camera_and_every_answer_shape(tmp_path):
    """Contract (plan §4.9, §2.6; DECISIONS 2026-10-10): the system prompt is the role and the protocol: "the brain of a
    robot that is doing a job", the frame of reference (metres, x east, y north, z up; headings counter-clockwise from
    +x; a pitch positive up), the camera as `public/camera.json` gives it (size, field of view), how tasks, frames and
    asks come (by calling `next_frame`; FD-questions-0.10 §1.5: a frame may come with questions about it, which follow
    the frame in the same reply, to be answered with `submit_answer` before `next_frame` is called again), every answer
    type's shape from `answers.PROMPT_SHAPES` (all of `answers.TYPES`, then `unknown`) and the unknown rule, and it says
    nothing of the brief, which arrives in the task message (`test_llm_render_camera.py` has the camera alone and the
    mount). A prompt with other camera numbers says other numbers. Regression caught: an answer type the model is
    never told how to give (a family asks `cells` and the prompt shows only six shapes), a model not told that
    questions can come inside a frame's reply, a camera read from a constant instead of the file, or a brief in the
    system prompt (it would be sent once, not with each task)."""
    prompt = system_prompt(public(tmp_path), "rgb_pose")
    assert prompt.startswith("You are the brain of a robot that is doing a job. ")
    for text in ("Coordinates are metres in the map frame: x points east, y north and z up. A heading is in degrees, "
                 "counter-clockwise from +x; a camera pitch is positive looking up.",
                 "The camera image is 640 x 480 pixels with a horizontal field of view of 80 degrees.",
                 "- You receive what the robot senses by calling next_frame. Each call returns the next thing, in time "
                 "order: a task message, one frame, the questions of an ask, or word that the job is over. Call it "
                 "again when you are ready for the next; nothing advances until you do.",
                 "- A frame is its header line and its camera image. The header gives its number k and clock time, "
                 "where the robot's base is, the way it heads, how it is moving (its own velocity in the map frame, "
                 "in metres a second, from its controller), the pitch of its camera, and its status. Things in the "
                 "image carry no labels or positions.",
                 "numbered Q1, Q2, ... A frame may come with questions about it, which follow the frame in the same "
                 "reply. Answer each, using only what the robot has sensed so far, by calling submit_answer once for "
                 "each question with its label and its answer, before you call next_frame again.",
                 "- When next_frame says the job is over, reply with one short line.", UNKNOWN_RULE):
        assert text in prompt, text
    assert [line for line in prompt.splitlines() if line.startswith("- ") and line[2:].split(":")[0] in
            {*TYPES, "unknown"}] == [f"- {kind}: {PROMPT_SHAPES[kind]}" for kind in (*TYPES, "unknown")]
    assert "assignment is" not in prompt and "Task 0" not in prompt
    small = {**CAMERA, "width": 320, "height": 240, "hfov_deg": 90.5}
    other = system_prompt(public(tmp_path / "b", small), "rgb_pose")
    assert "320 x 240 pixels with a horizontal field of view of 90.5 degrees" in other
    assert "640 x 480" not in other


def test_the_system_prompt_is_built_from_the_camera_and_robot_files_and_the_channel_alone(tmp_path, monkeypatch):
    """Contract (plan §4.9: no floor plan, no room names, no mission file; map D1, HA-H16): the prompt reads
    `public/camera.json` and `public/robot.json` and nothing else of the working folder, so a floor plan or a mission
    file left beside them never reaches the model, on every channel the agent reads (`rgb_pose` and
    `detections`); a name that is no channel (the retired `rgb`) and a channel that sends neither images nor detections
    are refused. Regression caught: the retired prompt's mission and floor-plan sections coming back, or a prompt
    written for a channel whose frames carry nothing the agent reads (the model would be told of images or detections
    it never gets)."""
    monkeypatch.setitem(stream_protocol.CHANNELS, "pose_only", frozenset({"depth", "pose", "status"}))
    folder = public(tmp_path, **{"mission.md": "Clean the zzmission.", "floorplan.json": json.dumps(
        {"rooms": [{"name": "zzroom", "polygon": [[0, 0], [1, 0], [1, 1]], "level": 0}]})})
    for channel in ("rgb_pose", "detections"):
        prompt = system_prompt(folder, channel)
        assert "zz" not in prompt and "floor plan" not in prompt.lower()
    for channel, why in (("rgb", "unknown channel 'rgb'"), ("pose_only", "sends neither")):
        with pytest.raises(ValueError, match=why):
            system_prompt(folder, channel)


@pytest.mark.parametrize("channel", ["rgb_pose", "detections"])
def test_the_prompt_speaks_of_decision_points_only_for_a_job_that_asks_during_the_job(tmp_path, channel):
    """Contract (0.5.0's pilot fix round; 0.6.0: look points): the system prompt says questions come "at some points
    during the job and after it" only when the job asks during the job (`asks_during`, which the harness reads from the
    job's `decisions` and `looks`); otherwise it says they come "after the job", and the two prompts differ in that
    clause alone, on every channel the agent reads. Without the flag the prompt makes no promise of questions during
    the job. Regression caught: a prompt that promises questions during the job to a job that asks none, or one that
    keeps promising them at "decision points" to a job whose questions come at look points (the agent waits for a
    question at a doorway)."""
    asking = system_prompt(public(tmp_path), channel, asks_during=True)
    quiet = system_prompt(public(tmp_path / "q"), channel, asks_during=False)
    bullet = "- Questions come {}, numbered Q1, Q2, ... A frame may come with questions about it, which"
    assert bullet.format("at some points during the job and after it") in asking
    assert bullet.format("after the job") in quiet
    assert "decision point" not in quiet and "during the job" not in quiet
    assert asking.replace("at some points during the job and after it", "after the job") == quiet
    assert system_prompt(public(tmp_path / "d"), channel) == quiet


@pytest.mark.parametrize("channel", ["rgb_pose", "detections"])
def test_the_prompt_speaks_of_questions_after_the_job_only_for_a_job_that_asks_any(tmp_path, channel):
    """Contract (final review, nit): the system prompt says questions come "at some points during the job and after it"
    when the job asks both ways, "at some points during the job" alone when it asks only at look points and has no end
    family (`asks_end` False: the drone), and "after the job" when it asks none during. Regression caught: the drone
    told that questions come after the job when none does."""
    bullet = "- Questions come {}, numbered Q1, Q2, ... A frame may come with questions about it, which"
    both = system_prompt(public(tmp_path), channel, asks_during=True, asks_end=True)
    during_only = system_prompt(public(tmp_path / "d"), channel, asks_during=True, asks_end=False)
    assert bullet.format("at some points during the job and after it") in both
    assert bullet.format("at some points during the job") in during_only
    assert "after it" not in during_only and "after the job" not in during_only
    assert both.replace("at some points during the job and after it", "at some points during the job") == during_only


def test_a_task_message_gives_its_number_its_clock_the_gap_and_the_brief(tmp_path):
    """Contract (plan §2.4): a task message names the task (case-relative), the clock it begins at, the unobserved gap
    since the last task ended (not said for the first, whose gap is 0) and the brief as the harness composed it, and
    nothing after it (no cue to update a memory). Regression caught: a gap the model is never told, so it cannot tell a
    minute from an afternoon between two tasks, a brief altered on the way, or a cue that coaches the agent to keep
    notes coming back."""
    assert task_message(0, "2026-03-02T08:00:00.0", 0.0, "Your assignment is to clean.") == (
        "Task 0 begins at 2026-03-02T08:00:00.0.\nYour assignment is to clean.")
    assert task_message(1, "2026-03-02T08:20:31.4", 612.4, "Next: dock.") == (
        "Task 1 begins at 2026-03-02T08:20:31.4.\n"
        "The robot went unobserved for 612 s since the last task's final frame.\nNext: dock.")


def test_a_frame_message_is_exactly_the_header_line_and_the_path_of_the_image_to_send():
    """Contract (2026-10-10: the agent is handed the time-ordered, timestamped images of the robot's
    sensors): a frame is its header line (case `k`, clock, the base's x, y, z, the base's heading, the base's own
    velocity (final review S3: `velocity (vx, vy) m/s` to the millimetre a second as delivered, left out of a step that
    has none), the camera's pitch and the status, `phase` and `task` first and the other keys sorted) and nothing after
    it (no title, no line naming an image, no cue); the image to send is the step's `rgb` path, returned and never
    shown. The positions have two decimals and no "-0.00", the velocity three and no "-0.000"; no room is named.
    Regression caught: a velocity the step carries but the agent is not shown (it would difference two poses by eye),
    one shown for a step that has none, "-0.000" from a delivered -0.0, a header that shows the camera's position or
    the base's pitch (the model would mix the two poses), a status in the robot's own order (a changed text from one
    robot to the next), the old batch title or image lines coming back, or a closing line that tells the agent what
    to do."""
    got = [frame_message(each, "rgb_pose") for each in STEPS]
    assert got == list(zip(EXPECTED_FRAMES, EXPECTED_IMAGES, strict=True))
    assert all("\n" not in text and "Image" not in text and "Frames" not in text for text, _ in got)
    bare = {k: v for k, v in STEPS[1].items() if k != "velocity"}
    assert frame_message(bare, "rgb_pose")[0] == EXPECTED_FRAMES[1].replace("velocity (0.000, 0.625) m/s, ", "")


def test_a_frame_message_reads_only_the_fields_it_names():
    """Contract (VISION rules 3, 6): the text is built from the step's `k`, `clock`, `pose` position and yaw,
    `velocity`, `camera` pitch and `status`, and its `rgb` path is returned to send without being shown; the privileged
    `identity`, a detection list, the quaternions and the camera's own position never reach the text. Regression
    caught: a renderer
    that dumps the whole step, which would hand the model entity ids or the oracle's detections."""
    full = copy.deepcopy(STEPS)
    for i, each in enumerate(full):
        each.update(identity={"deadbeef": "E0042"}, detections=[{"kind": "mug", "center_map": [1, 2, 0.5]}], t=1.5 * i)
    assert [frame_message(each, "rgb_pose") for each in full] == list(
        zip(EXPECTED_FRAMES, EXPECTED_IMAGES, strict=True))
    assert not [text for text in EXPECTED_FRAMES if "inbox" in text or "E0042" in text or "mug" in text]


def test_a_frame_needs_its_picture_on_a_camera_channel():
    """Contract: a frame on `rgb_pose` has a camera image to send, so a step with no `rgb` is an error and not a header
    for an image that never goes. Regression caught: a frame whose image is silently missing (the model would be told of
    a camera frame it cannot look at)."""
    with pytest.raises(ValueError, match="step k12 has no camera image"):
        frame_message({k: v for k, v in STEPS[0].items() if k != "rgb"}, "rgb_pose")


QUESTIONS = [{"qid": "7f3a9c01e5d24b68", "text": "How many red mugs did you see?", "answer_type": "count",
              "template": "where_last", "slots": {"kind": "mug", "room": "scullery"}, "needs": ["rgb"], "scope": "now"},
             {"qid": "0f1e2d3c4b5a6978", "text": "Where is the lamp now?", "answer_type": "position"}]


def test_an_ask_message_labels_its_questions_and_shows_each_text_type_and_shape():
    """Contract (plan §4.9, §2.6; map HA-H14): an ask is one message that says when it is and what it may rely on, then
    for each question in the order given its label Q1..Qn, its text, its answer type and that type's shape from
    `answers.PROMPT_SHAPES`, then the unknown rule; an end ask says it is after the job. Only `text` and `answer_type`
    are read: no qid, template, slots or scope reaches the model, whatever the question carries. Regression caught: a
    qid in the prompt (the leak scan would flag it, and the model could index the dataset by it), a question shown
    with no shape for its type, or the retired "Answer each from your memory" back in the ask (the ask says only to
    answer from what the robot has sensed, and to call `submit_answer` for each label)."""
    questions = labelled(QUESTIONS)
    assert list(questions) == ["Q1", "Q2"] and questions["Q2"] is QUESTIONS[1]
    during = ask_message("during", {"k": 31, "clock": "2026-03-02T08:00:46.5"}, questions)
    assert during == (
        "Questions during the job, at the robot's latest frame (k31, 2026-03-02T08:00:46.5). Answer each, using only "
        "what the robot has sensed up to that frame. Call submit_answer once for each question, with its label and "
        "your answer.\n\n"
        "Q1: How many red mugs did you see?\n"
        'Answer type: count. answer = {"type": "count", "value": N} (N is a whole number, 0 or more)\n\n'
        "Q2: Where is the lamp now?\n"
        'Answer type: position. answer = {"type": "position", "xy": [x, y]} (metres, map frame; "xyz": [x, y, z] also '
        "works)\n\n"
        'Answer only if you are more likely right than wrong; otherwise answer {"type": "unknown"}. '
        "A wrong answer costs more than unknown.")
    for hidden in ("7f3a9c01e5d24b68", "0f1e2d3c4b5a6978", "where_last", "scullery", "needs", "scope", "mug\""):
        assert hidden not in during
    end = ask_message("end", {"k": 97, "clock": "2026-03-02T08:02:25.5"}, questions)
    assert end.startswith("Questions after the job, at the robot's last frame (k97, 2026-03-02T08:02:25.5). ")


def test_an_ask_shows_the_shape_of_every_answer_type_and_refuses_one_it_cannot():
    """Contract: a question of any type in `answers.TYPES` is shown with that type's own shape, so no family can ask
    what the prompt cannot show; a type outside the list and a `when` that is neither during nor end are errors before
    any exec. Regression caught: a type missing from the shapes producing a message with no shape, or a KeyError deep
    in a paid run."""
    for kind in TYPES:
        message = ask_message("end", {"k": 1, "clock": "c"}, labelled([{"qid": "x", "text": "?", "answer_type": kind}]))
        assert f"Answer type: {kind}. answer = {PROMPT_SHAPES[kind]}" in message and UNKNOWN_RULE in message
    with pytest.raises(ValueError, match="answer_type"):
        ask_message("end", {"k": 1, "clock": "c"}, labelled([{"qid": "x", "text": "?", "answer_type": "polygon"}]))
    with pytest.raises(ValueError, match="during or end"):
        ask_message("now", {"k": 1, "clock": "c"}, {})


def test_every_answer_type_has_a_shape_that_names_the_fields_the_parser_reads():
    """Contract: the shape the model is shown for each answer type names, as keys, every field the strict parser reads
    for it (`type`, and `xy`, `clock`, `value`, `items`, `change`, `points`, `deg_left`), and the example answer built
    from those names is accepted for that type. Regression caught: a shape that drifts from the parser (a field
    renamed in `answers.py` only), which would score every such answer malformed while the prompt still teaches the
    old name."""
    valid = {
        "position": {"type": "position", "xy": [1.0, 2.0]},
        "offset": {"type": "offset", "ahead_m": 3.5, "left_m": -1.0},
        "time": {"type": "time", "clock": "2026-03-02T10:00:00"},
        "yesno": {"type": "yesno", "value": True},
        "count": {"type": "count", "value": 3},
        "number": {"type": "number", "value": 2.5},
        "set": {"type": "set", "items": [{"xy": [1.0, 2.0], "change": "moved"}]},
        "positions": {"type": "positions", "items": [{"xy": [1.0, 2.0]}]},
        "cells": {"type": "cells", "items": [[2, 4]]},
        "route": {"type": "route", "points": [[1.0, 2.0]]},
        "heading": {"type": "heading", "deg_left": -35.0},
        "unknown": {"type": "unknown"},
    }
    assert set(valid) == {*TYPES, "unknown"}  # one example for every type the prompt shows
    for kind, answer in valid.items():
        assert not isinstance(parse_answer(answer, kind if kind != "unknown" else "count"), Malformed)
        names = {"type", *answer, *(k for item in answer.get("items", []) if isinstance(item, dict) for k in item)}
        assert f'"type": "{kind}"' in PROMPT_SHAPES[kind]
        assert all(f'"{name}"' in PROMPT_SHAPES[kind] for name in names), kind


def test_the_fixed_lines_the_model_is_sent_besides_the_items():
    """Contract: three lines are fixed words, the reply to `next_frame` once the job is over, the line a thread whose
    turn ended early is resumed with, and the error a premature `next_frame` gets (the labels left, then the nudge, as a
    tool error). Regression caught: a reworded end that the prompt's "when next_frame says the job is over" no longer
    describes, a resume line that coaches, or a nudge that does not say which question it is about."""
    assert END_MESSAGE == "The job is over: there are no more frames."
    assert RESUME_MESSAGE == "Call next_frame to continue."
    assert unanswered(["Q1", "Q3"]) == (
        "error: you have not answered Q1, Q3. Call submit_answer now; use unknown if you cannot know.")


def test_the_prompt_names_no_room(tmp_path):
    """Contract (plan §4.9; D1, map HA-H16): the system prompt, a task, a frame and an ask message hold no room name
    or kind of any bundled map and no floor plan: where things are is answered as points. The brief is the harness's, so
    the test gives a neutral one. Regression caught: the retired floor-plan section or a room-labelled frame header
    coming back, which would hand the agent a layout the dataset does not give a system."""
    questions = labelled([{"qid": "x", "text": "Where is the thing?", "answer_type": "position"}])
    messages = [system_prompt(public(tmp_path), "rgb_pose"),
                task_message(0, "2026-03-02T08:00:00", 0.0, "Clean up."),
                frame_message(STEPS[0], "rgb_pose")[0], ask_message("end", {"k": 13, "clock": "c"}, questions),
                unanswered(["Q1"]), END_MESSAGE, RESUME_MESSAGE]
    names = {word.lower() for name in mapspec.bundled_maps() for r in mapspec.load_map(name).regions
             for word in (r.name, r.kind)}
    assert {"kitchen", "living room", "high street", "west aisle"} <= names  # the scan finds the rooms, so no vacuity
    for text in messages:
        assert [n for n in names if re.search(rf"\b{re.escape(n)}\b", text.lower())] == []
        assert "floor plan" not in text.lower() and "floorplan" not in text.lower()
