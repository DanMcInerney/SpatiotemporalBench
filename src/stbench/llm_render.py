"""Steps and questions to messages for the reference agent: the system prompt, a task, a frame, an ask.

Everything is built from what a system is sent (`stream_protocol`): `public/camera.json` and `public/robot.json`, the
brief of a task, the step of a frame and an ask's questions as `ask_view` gives them. There is no floor plan and no
room name (a system answers where as points), no mission file, and nothing of the dataset. A step's `identity`, a
question's `template` and `slots`, and every other key are never read, so none can reach the model (of a step only its
`rgb` path is returned, for the caller to attach, and never shown); of a detection only the fields of `DETECTION_KEYS`
and `OPTIONAL_KEYS` and its door `state` are read, so its key `d` and anything else a row carries cannot. Every
function is a pure function of its input: the same steps and questions give the same bytes. Nothing here branches on
a model or a robot, and the one branch on the channel is what the channel sends (pictures, or detections as text).

The robot has a camera and no depth sensor (0.6.0): the prompt says the camera is its only eye on the world, and no
message carries a depth image or a word about one. The prompt and the messages are part of the measurement
(PRIOR-ART §11), fixed for a release.
The prompt tells of questions during the job only for a job that asks any (`system_prompt`'s `asks_during`: the job has
decision-point questions or look points) and of questions after it only for a job that asks any (`asks_end`: its
`ends`), so it never promises a question that does not come. It shows the shape of each kind of answer the job asks and
no other (`system_prompt`'s `kinds`: `brief.answer_types`, from the job's families, so both arms of the leak check get
the same bytes), and `unknown`; an ask whose type the prompt did not show is refused (`ask_message`'s `shown`). It
states how an answer is credited (0.9.0: `llm_credit`, which owns the targets and the credit section, a titled list of
plain bullets): the four quarters against the job's own clearance C and re-find radius R, the harmful side's halved
margin, a list per item (`lists`: the job asks `task_things`) and a heading's regret (`headings`: the job asks
`open_heading`), and the matching radius of the job's own place (`system_prompt`'s `match_m`, from `JobSpec.match_m`),
only when it is given the radius. The three numbers come together or none. Its bullets about one kind of answer or one
family are told to the job that asks it and to no other (`system_prompt`'s `kinds` and `families`:
`brief.answer_types`, `brief.asked_families`), as the shapes are.

The text-only option (`text_only`, a zero-shot check that the questions are not answerable from the text of the stream
alone) sends no image at all: the prompt says so, and a frame is its header line alone. It is an option and not a
channel, since the stream is the `rgb_pose` one, and the run is stamped with it (`cmd_codex`).

The camera is stated as the robot knows it (a robot's brain knows its own camera): the image size and
horizontal field of view from `camera.json`, and the mount from `robot.json`'s `camera.mount_m` (`mount_sentence`), the
camera's place relative to the base whose position every header gives. The pitch is in each header. So is the body
(0.8.0, `body_sentence`, from `robot.json`'s `body` and `flies`): its shape and size, its height and that the base is
its middle, and for a robot that drives what it drives over and the height it passes under, since a question of how
far the body could move has no answer without them. The credit section defines, for a job that asks `clear_path`, the
corridor (`CORRIDOR_TARGET`: ahead, back and a turn in place first) and, for a job that asks it, the heading
(`HEADING_TARGET`).

The agent pulls what the robot senses with `next_frame` (`agent_tools`, served by `codex_feed`), and each call returns
one item in stream order: a task message, one frame, an ask, a frame that comes with the questions about it, or word
that the job is over (`END_MESSAGE`). A frame is its header line as a text block and its camera image as the image block
after it (the caller sends the image, read from the path `frame_message` returns). Nothing follows the frame but the
questions that come with it, when any do (their text is `ask_message`'s, unchanged, as a block after the image): the
item carries no cue and no instruction about what to keep (the agent remembers with its own context):

  [k12 2026-03-02T08:00:18.0] base at (5.60, 5.05, 0.00), heading -90.0 deg, velocity (0.000, -0.250) m/s, camera
  pitch 5 deg; status: phase ...

With `text_only` a frame is its header line alone, with no image. A header line starts a line with `[k<k> `
(`llm_scan.HEADER` reads it, so no other text starts a line so). `k` is the case's own number. The position and heading
are the robot's base's, the velocity its own (the step's `velocity`, the controller's odometry: the base's displacement
over the previous frame of its task, in the map frame, `(0.000, 0.000)` at a task's first frame), the pitch the
camera's; the status lists `phase` and `task` first, then the robot's other keys in sorted order.
The camera's position is not shown: the system prompt says where it sits on the base. A job that states a subgoal
(what the robot is doing now, `JobSpec.subgoals`) has "; subgoal: <text>" at the end of the header line, on one line
(white space inside the text is collapsed, so the text cannot start a header of its own); a step with none (a job that
states none, or the neutral brief) has the header as above.

On a channel that sends detections and no pictures (`detections`, what the pixels cost) a frame is the
same text with no image: after the frame's header line come its detections, one indented line each (so no line of
them starts a header), in the order the harness delivered them, with every field the channel gives a system except the
key `d`, which is fresh for each detection and names nothing:

  [k12 2026-03-02T08:00:18.0] base at (5.60, 5.05, 0.00), heading -90.0 deg, velocity (0.000, -0.250) m/s, camera ...
  Detections in frame k12 (2):
    - mug (color red, size small): map (2.00, 1.00, 0.90), camera (0.10, -0.30, 1.50), size 0.10 x 0.10 x 0.12 m, ...
    - door: map (3.10, 1.20, 1.00), camera (...), size 0.90 x 0.05 x 2.00 m, box (...) px, open true

(or "Detections in frame k12: none".)

An ask is one item that labels its questions Q1..Qn, in the order given, and shows for each the question's text, its
answer type and that type's shape (`answers.PROMPT_SHAPES`); it comes alone or after the frame it is about, in the same
reply (`codex_system`'s held frame), and its text is the same either way. A question's qid is never shown: the system
maps the labels back (`labelled`). A turn that ends before the job does is resumed with `RESUME_MESSAGE`, one fixed
line.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

from .answers import PROMPT_SHAPES, TYPES
from .grid import band_of
from .llm_credit import check_kinds, credit_sentence, length
from .stream_protocol import CHANNELS

RGB = "rgb"  # the step key of a frame's one image, the camera's
END_MESSAGE = "The job is over: there are no more frames."  # the reply to next_frame once the stream has ended
RESUME_MESSAGE = "Call next_frame to continue."  # what a thread whose turn ended early is resumed with
DETECTION_KEYS = ("kind", "attributes", "bbox_px", "center_cam", "center_map", "extent_m")  # the fields a line shows
OPTIONAL_KEYS = ("floor_hull",)  # shown when a detection has it, `none` when it is null (0.7.0)
UNKNOWN_RULE = ('Answer only if you are more likely right than wrong; otherwise answer {"type": "unknown"}. '
                'A wrong answer costs more than unknown.')
WHEN_ASKED = {(True, True): "at some points during the job and after it",
              (True, False): "at some points during the job",
              (False, True): "after the job", (False, False): "after the job"}  # by `asks_during`, `asks_end`
FIRST_KEYS = ("phase", "task")  # a status line lists these first, then the robot's other keys in sorted order
_ROLE = ("You are the brain of a robot that is doing a job. You are told the job, then shown what the robot senses "
         "as it works, and you are asked questions about it.")
_FRAME = ("Coordinates are metres in the map frame: x points east, y north and z up. A heading is in degrees, "
          "counter-clockwise from +x; a camera pitch is positive looking up.")
_EYE = ("The camera is the robot's only eye on the world: nothing measures a distance for you, so any distance you "
        "give is your own estimate.")
_OFFSET = ("An offset answer is not in the map frame: it is metres from the camera, along the floor, with ahead along "
           "the robot's heading (the heading in each frame's header, whichever way the camera is pitched) and left to "
           "its left; a thing's height is not part of it.")
MOUNT_WORDS = (("ahead of", "behind"), ("to the left of", "to the right of"), ("above", "below"))  # mount x, y, z


def _num(v: float) -> str:
    """Two decimals, finer than the smallest rung (5 cm, `rules.REL_FLOORS_M`); never "-0.00"."""
    s = f"{v:.2f}"
    return "0.00" if s == "-0.00" else s


def _speed(v: float) -> str:
    """A velocity component, three decimals as delivered (`case.VELOCITY_DECIMALS`); never "-0.000"."""
    s = f"{v:.3f}"
    return "0.000" if s == "-0.000" else s


def _value(v) -> str:
    """A status value as one word: `true` or `false`, a number without a needless ".0", a string as it is."""
    if isinstance(v, bool):
        return "true" if v else "false"
    return f"{v:g}" if isinstance(v, float) else str(v)


_DETECTED = ("the camera's detector made out in that frame: its kind, its colour and size where it has them, the "
             "middle of the thing in the map frame and in the camera frame (x right, y down and z forward from the "
             "camera, in metres), its size in metres, its box in the image in pixels, where it gives it the floor hull "
             "of the thing (`floor hull`: the outline, the convex hull, of what the frame shows of the thing, dropped "
             "onto the floor, as (x, y) map points in metres, counter-clockwise; none when the thing is not clearly "
             "visible in the frame), and for a door whether it is open. The detector is exact about what it lists, but "
             "lists no thing too small or too far to make out. It does not say which thing is which: one thing seen in "
             "two frames is two detections, and knowing it again is for you.")


def _reads(channel: str) -> str:
    """What the reference agent reads on `channel`: "pictures" when it sends the camera image, else "detections" when
    it sends those. ValueError for a name that is no channel and for a channel that sends neither."""
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel {channel!r}; the channels are {', '.join(CHANNELS)}")
    for sends, reads in (("rgb", "pictures"), ("detections", "detections")):
        if sends in CHANNELS[channel]:
            return reads
    raise ValueError(f"the reference agent reads camera images or detections, and channel {channel!r} sends neither")


def mount_sentence(mount: object) -> str:
    """Where the camera sits on the base, from `robot.json`'s `camera.mount_m`: [forward, left, up] in metres, relative
    to the base whose position every frame header gives (the camera's own position is never sent). Each non-zero axis is
    a distance and a direction, with the first "of the base" and the others "of it": 0.38 m ahead of the base and
    0.50 m above it; a camera that hangs under a base that flies is 0.10 m below the base; one at the base is "at the
    base". Two decimals as the headers have them. ValueError unless `mount` is a list of three numbers."""
    if not (isinstance(mount, list | tuple) and len(mount) == 3
            and all(isinstance(c, int | float) and not isinstance(c, bool) for c in mount)):
        raise ValueError(f"camera.mount_m is [forward, left, up] in metres, three numbers, not {mount!r}")
    parts = [f"{_num(abs(c))} m {words[c < 0]}" for c, words in zip(mount, MOUNT_WORDS, strict=True)
             if _num(abs(c)) != "0.00"]
    if not parts:
        return "The camera is mounted at the base."
    placed = [f"{parts[0]} the base", *(f"{part} it" for part in parts[1:])]
    where = f"{', '.join(placed[:-1])} and {placed[-1]}" if len(placed) > 1 else placed[0]
    return f"The camera is mounted {where}."


def body_sentence(robot: object) -> str:
    """The robot's body as it knows it, from `robot.json`'s `flies` and `body` (0.8.0: without it "how far could your
    body go" has no answer): its shape and size, its height, and that the base is the middle of the body. A robot that
    drives adds what it drives over (`climb_m`) and the least height above the floor that it passes under, which is the
    top of the band `grid.band` reads obstacles in (`height_m` plus half `under_margin_m`); a flier states neither.
    Lengths as a person says them (`_length`): "a disc 34 cm across and 9 cm tall", "a box 80 cm long and 58 cm wide".
    ValueError for a `robot.json` with no body, a shape that is no disc or box, a disc whose two sizes differ and a size
    that is no number above 0 (a climb or margin: not below 0)."""
    body, flies = (robot.get("body"), robot.get("flies")) if isinstance(robot, dict) else (None, None)
    if not isinstance(body, dict) or not isinstance(flies, bool):
        raise ValueError("robot.json needs `flies` and a `body` (shape, length_m, width_m, height_m, climb_m, "
                         "under_margin_m)")
    size = {}
    for key, least in (("length_m", False), ("width_m", False), ("height_m", False), ("climb_m", True),
                       ("under_margin_m", True)):   # least: a size that may be 0
        value = body.get(key)
        if (isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value)
                or value < 0 or (value == 0 and not least)):
            raise ValueError(f"body.{key} is not a valid size: {value!r}")
        size[key] = value
    if body.get("shape") == "disc" and size["length_m"] != size["width_m"]:
        raise ValueError(f"a disc body has one size, got length_m {size['length_m']!r} and width_m {size['width_m']!r}")
    if body.get("shape") == "disc":
        form = f"a disc {length(size['length_m'])} across"
    elif body.get("shape") == "rect":
        form = f"a box {length(size['length_m'])} long and {length(size['width_m'])} wide"
    else:
        raise ValueError(f"body.shape is disc or rect, got {body.get('shape')!r}")
    text = f"The robot's body is {form} and {length(size['height_m'])} tall, centred on the base."
    if flies:
        return text
    top = band_of(size["climb_m"], size["height_m"], size["under_margin_m"])[1]   # `grid.band`'s top
    return (f"{text} It drives over anything up to {length(size['climb_m'])} high and passes under anything "
            f"{length(top)} or more above the floor.")


def listed(kinds: Sequence[str] | None) -> tuple[str, ...]:
    """The answer types the prompt shows a shape of, in `PROMPT_SHAPES`' order with `unknown` last: `kinds` (the types
    the job asks: `brief.answer_types`) and `unknown`, or every type for None (no job known). ValueError for a name that
    is no answer type (`unknown` is always listed, so it is not one to name)."""
    check_kinds(kinds)
    return tuple(kind for kind in PROMPT_SHAPES if kinds is None or kind in kinds or kind == "unknown")


def _answers(kinds: Sequence[str] | None) -> str:
    """The prompt's last section: the shape of each answer type listed (`listed`), one line each, `unknown` last, and
    the unknown rule."""
    lines = [f"- {kind}: {PROMPT_SHAPES[kind]}" for kind in listed(kinds)]
    return "\n".join(["Answers are JSON objects, one shape for each kind of answer:", *lines, UNKNOWN_RULE])


def system_prompt(public: Path, channel: str, *, asks_during: bool = False, asks_end: bool = True,
                  match_m: float | None = None, clearance_m: float | None = None, reacquire_m: float | None = None,
                  lists: bool = False, headings: bool = False, hazards: Sequence[str] = (),
                  kinds: Sequence[str] | None = None, families: Sequence[str] | None = None,
                  turns: bool = True, text_only: bool = False) -> str:
    """The prompt for one case, from `public/camera.json` and `public/robot.json`, the channel and whether the job asks
    during the job at all (`asks_during`: the harness reads it from the job's `decisions` and `looks`) and after it
    (`asks_end`: from its `ends`): the role, the frame of reference (the map frame, and the offset frame an offset
    answer is in), the camera as the robot knows it (its image size and horizontal field of view, and where it is
    mounted on the base: `mount_sentence`) and that it is the robot's only eye on the world, its body (`body_sentence`),
    how tasks, frames and asks come (by calling `next_frame`), the shape of each answer type the job asks and of
    `unknown`, and the unknown rule, and, only when the job's three numbers are given (the harness reads `match_m`,
    `clearance_m` and `reacquire_m` from `JobSpec`), how an answer is credited (`credit_sentence`, with a list's
    bullets when the job asks lists, the omission of its critical person or cord that scores -3 for the list families
    `hazards` names, and a heading's bullets when it asks headings: `brief.asks_lists`, `list_hazards`,
    `asks_headings`); with none there is no such section. `kinds` are the answer types the job asks
    (`brief.answer_types`, read from the job and never from the run's `during` arm), so the prompt shows no shape the
    job never asks, and `families` the families it asks (`brief.asked_families`), so the credit section tells the
    rules of those kinds and families alone (`credit_sentence`); None for either, a caller that knows no job, shows
    every type (`listed`) and every such bullet.
    Nothing of memory: the agent remembers with its own context, and is told neither what to keep nor that its
    conversation may be shortened. Questions are said to come at some points during the job when `asks_during` and after
    it when `asks_end` (a job that asks none after it, the drone, is told of the first alone), else after the job. The
    brief arrives in the first task message, never here. On a channel of pictures each frame comes with its camera
    image, and there is no word of any other sensor; with `text_only` no image is sent and the prompt says so, a frame
    being its header line alone. On one of detections the frames come as the detections found in them. Only the "How
    this works" bullets speak of how things arrive: by `next_frame`, one item at a time, and nothing advances until the
    agent calls it again. ValueError for `text_only` on a channel of detections (it has no image to withhold), for
    some but not all of the credit's three numbers and for a name in `kinds` that is no answer type."""
    reads = _reads(channel)
    credit = (match_m, clearance_m, reacquire_m)
    if any(v is not None for v in credit) and any(v is None for v in credit):
        raise ValueError("the credit sentence needs match_m, clearance_m and reacquire_m together, or none of them")
    if text_only and reads != "pictures":
        raise ValueError(f"text_only withholds the camera images, and channel {channel!r} sends none")
    images = reads == "pictures" and not text_only
    camera = json.loads((Path(public) / "camera.json").read_text(encoding="utf-8"))
    robot = json.loads((Path(public) / "robot.json").read_text(encoding="utf-8"))
    lens = (f"The camera image is {camera['width']} x {camera['height']} pixels with a horizontal field of view of "
            f"{camera['hfov_deg']:g} degrees. {mount_sentence((robot.get('camera') or {}).get('mount_m'))} "
            f"{body_sentence(robot)}")
    if reads == "detections":
        lens += " You are not sent the images, only the detections in them, with boxes in those pixels."
    else:
        lens += f" {_EYE}"
        if text_only:
            lens += " In this run you are not sent the images: a frame is its header line alone."
    frame = ("A frame is its header line and its camera image." if images else
             "A frame is its header line alone, with no image." if text_only else
             "A frame is its header line and the detections found in it.")
    how = [
        "How this works:",
        "- You receive what the robot senses by calling next_frame. Each call returns the next thing, in time order: "
        "a task message, one frame, the questions of an ask, or word that the job is over. Call it again when you are "
        "ready for the next; nothing advances until you do.",
        "- A task message tells you the clock time a task begins at, how long the robot went unobserved since the "
        "last one, and the assignment.",
        f"- {frame} The header gives its number k and clock time, where the robot's base is, the way it heads, how it "
        "is moving (its own velocity in the map frame, in metres a second, from its controller), the pitch of its "
        "camera, and its status."
        + (" Things in the image carry no labels or positions." if images else "" if text_only else
           f" Below it comes one line for each thing {_DETECTED}"),
        f"- Questions come {WHEN_ASKED[asks_during, asks_end]}, numbered Q1, Q2, ... A frame may come with questions "
        "about it, which follow the frame in the same reply. Answer each, using only what the robot has sensed so "
        "far, by calling submit_answer once for each question with its label and its answer, before you call "
        "next_frame again.",
        "- When next_frame says the job is over, reply with one short line.",
    ]
    return "\n\n".join([_ROLE, f"{_FRAME} {_OFFSET} {lens}", "\n".join(how), _answers(kinds),
                        *([credit_sentence(match_m, clearance_m, reacquire_m, lists=lists, headings=headings,
                                           hazards=hazards, kinds=kinds, families=families, turns=turns)]
                          if match_m is not None else [])])


def task_message(index: int, clock: str, gap_s: float, brief: str) -> str:
    """The message of a task: its number and start clock, the unobserved time since the last task ended (none is said
    for the first) and the brief."""
    lines = [f"Task {index} begins at {clock}."]
    if gap_s > 0:
        lines.append(f"The robot went unobserved for {gap_s:.0f} s since the last task's final frame.")
    return "\n".join([*lines, brief])


def header(step: dict) -> str:
    """A frame's header line: its number and clock, the base's position and heading, its own velocity when the step
    has one (`velocity (vx, vy) m/s`, as delivered, to the millimetre a second: the map frame's x and y), the camera's
    pitch, the status and, when the step has one, "; subgoal: <text>" (on one line: white space in the text is
    collapsed, so no line of it can start a header)."""
    x, y, z = step["pose"]["pos"]
    status = step["status"]
    keys = [*(k for k in FIRST_KEYS if k in status), *sorted(k for k in status if k not in FIRST_KEYS)]
    subgoal = " ".join(str(step["subgoal"]).split()) if step.get("subgoal") else ""
    moving = f"velocity ({_speed(step['velocity'][0])}, {_speed(step['velocity'][1])}) m/s, " if step.get(
        "velocity") else ""
    line = (f"[k{step['k']} {step['clock']}] base at ({_num(x)}, {_num(y)}, {_num(z)}), "
            f"heading {round(step['pose']['yaw_deg'], 1) + 0.0:.1f} deg, {moving}"   # + 0.0: no "-0.0"
            f"camera pitch {round(step['camera']['pitch_deg'])} deg; "
            f"status: {', '.join(f'{k} {_value(status[k])}' for k in keys)}")
    return f"{line}; subgoal: {subgoal}" if subgoal else line


def _xyz(point: list[float]) -> str:
    return f"({', '.join(_num(c) for c in point)})"


def detection_line(detection: dict) -> str:
    """One detection as an indented line (so none starts a header): its kind and attributes, then its place in the map
    frame and in the camera frame, its size, its box in the image, its floor hull (`OPTIONAL_KEYS`: a list of map-frame
    (x, y) points to the millimetre, counter-clockwise, shown when the detection has the key, as `none` when it is
    null) and, for a door, `open true` or `open false`. Only `DETECTION_KEYS`, `OPTIONAL_KEYS` and the door `state` are
    read, never the key `d`.
    ValueError for a detection lacking one of `DETECTION_KEYS`."""
    if lacking := [key for key in DETECTION_KEYS if key not in detection]:
        raise ValueError(f"a detection lacks {', '.join(lacking)}")
    attributes, (u0, v0, u1, v1) = detection["attributes"], detection["bbox_px"]
    named = ", ".join(f"{key} {_value(attributes[key])}" for key in sorted(attributes))
    named = f" ({named})" if named else ""
    parts = [f"map {_xyz(detection['center_map'])}", f"camera {_xyz(detection['center_cam'])}",
             f"size {' x '.join(_num(c) for c in detection['extent_m'])} m",
             f"box ({u0:g}, {v0:g}) to ({u1:g}, {v1:g}) px",
             *(["floor hull none" if detection["floor_hull"] is None else
                f"floor hull [{', '.join(f'({x:.3f}, {y:.3f})' for x, y in detection['floor_hull'])}]"]
               if "floor_hull" in detection else []),
             *(f"{key} {_value(value)}" for key, value in sorted(detection.get("state", {}).items()))]
    return f"  - {detection['kind']}{named}: {', '.join(parts)}"


def frame_message(step: dict, channel: str, *, text_only: bool = False) -> tuple[str, str | None]:
    """(the text of a frame, the path of its camera image or None). On a channel of pictures the text is the header
    line and the step must give its camera image's path, as delivered (relative to the working folder), which the caller
    sends as the image block that follows the text; on a channel of detections there is no image and the header line is
    followed by the frame's detections (`Detections in frame k<k> (n):` and a line each, or `: none`). With `text_only`
    there is no image either, whatever the step carries: a frame is its header line alone (a step needs no `rgb`). A
    step on `detections` needs its list, which may be empty. ValueError for `text_only` on `detections`."""
    reads = _reads(channel)
    if text_only and reads != "pictures":
        raise ValueError(f"text_only withholds the camera images, and channel {channel!r} sends none")
    if reads == "detections":
        if not isinstance(step.get("detections"), list):
            raise ValueError(f"step k{step['k']} has no detections: the reference agent reads them on {channel!r}")
        found = step["detections"]
        lines = [header(step), f"Detections in frame k{step['k']}" + (f" ({len(found)}):" if found else ": none"),
                 *map(detection_line, found)]
        return "\n".join(lines), None
    if text_only:
        return header(step), None
    if not step.get(RGB):
        raise ValueError(f"step k{step['k']} has no camera image: the reference agent reads it on {channel!r}")
    return header(step), step[RGB]


def labelled(questions: list[dict]) -> dict[str, dict]:
    """The questions of an ask by label, Q1..Qn in the order given: the only names the model ever sees for them."""
    return {f"Q{n}": q for n, q in enumerate(questions, 1)}


def ask_message(when: str, now: dict, questions: dict[str, dict], *, shown: Sequence[str] | None = None) -> str:
    """The message of an ask (`questions` is `labelled`'s): when it is, how to answer, and each question's label, text,
    answer type and that type's shape. Only `text` and `answer_type` of a question are read: no qid, template, slots or
    time. `shown` are the types the system prompt showed a shape of (`listed`); ValueError for a question of another
    type, since the agent was never told that type's form in the prompt (None: no prompt to match)."""
    if when not in ("during", "end"):
        raise ValueError(f"an ask is during or end, got {when!r}")
    if shown is not None and (unshown := sorted({q["answer_type"] for q in questions.values()} - set(shown))):
        raise ValueError(f"the system prompt shows no shape for the answer type {', '.join(unshown)} of this ask")
    where = ("during the job, at the robot's latest frame" if when == "during"
             else "after the job, at the robot's last frame")
    lines = [f"Questions {where} (k{now['k']}, {now['clock']}). Answer each, using only what the robot has sensed up "
             "to that frame. Call submit_answer once for each question, with its label and your answer."]
    for label, q in questions.items():
        kind = q["answer_type"]
        if kind not in TYPES:
            raise ValueError(f"answer_type must be one of {list(TYPES)}, got {kind!r}")
        lines += ["", f"{label}: {q['text']}", f"Answer type: {kind}. answer = {PROMPT_SHAPES[kind]}"]
    return "\n".join([*lines, "", UNKNOWN_RULE])
