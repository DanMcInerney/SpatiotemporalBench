"""The pointing tool of the reference agent: `locate` turns a pixel of a frame of the current batch into a map point.

A channel that sends rgb, depth and pose gives the model each frame's camera image and header line, and delivers the
frame's depth image (planar depth, 16-bit, in the unit `camera.json` names: millimetres unless the map says another),
the camera's pose (the step's `camera` block: its position and rotation in the map frame) and, in `camera.json`, the
intrinsics. Reading where a thing is in an image is perception; turning that pixel and the camera's pose into map
metres is arithmetic the channel's own inputs fix, and the pilot (release 0.1) showed it is the part a model does
badly: the right object, 0.2-1.2 m off. So the tool does that arithmetic and nothing else: it reads the frame's depth at
the pixel (in metres by the camera's unit, `Camera.depth_scale`), back-projects it with the intrinsics
(`camera.unproject`), and moves it into the map frame by the camera's pose (`camera.cam_to_map`), the same two
functions the visibility log and the perception module use. It tells the model nothing the channel does not already
hand it, and it finds nothing: the model still decides what is at which pixel.

The pose it reads is the camera's, never the step's `pose`, which is the robot's base: the camera sits on a mount
(0.13 m ahead of the vacuum's centre) and pitches, so the base's pose would misplace every point.

It works for the frames of the batch being observed only. The harness empties the inbox at the next batch, so the depth
of an earlier frame is gone; the model locates while it sees, and keeps the result in its memory. An ask follows the
batch that ends at its step, with that batch's inbox, so the model can locate in that batch alone, with the images it
already holds.

A pixel (u, v) is a pixel of the frame's image: u counts columns from the left edge, v rows from the top, and the pixel
is the one whose column is `floor(u)` and row `floor(v)`; its centre is the point used (`camera.py`: pixel column j
spans u in [j, j + 1)). The reply is the map position (x, y, z) of the surface seen there, in metres to the centimetre,
and its distance from the camera. That is the visible surface, so it is nearer the camera than the centre of a thing. A
refusal is a string `"error: ..."` the model can recover from; nothing here raises for a request.

Unused: the reference agent estimates positions from what it is sent and is offered no pointing tool
(DECISIONS, 2026-10-10, "The reference agent remembers with its own context"). The module and its tests stay;
nothing in the package imports it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .camera import Camera, cam_to_map, unproject
from .render_encode import read_png16

LOCATE_NAME = "locate"
LOCATE = {"type": "function", "function": {
    "name": LOCATE_NAME,
    "description": (
        "Turn a point in a frame's camera image into a map position. Give the frame's number k (the number in its "
        "header line, [k117 ...]) and the pixel (u, v) of the point in that frame's image: u counts pixels from the "
        "left edge, v from the top. The reply is the map position (x, y, z) in metres of the surface seen at that "
        "pixel, from the robot's depth sensor and the camera's pose, and its distance from the camera. That is a "
        "point on the visible surface, nearer the robot than the middle of a thing. It works only for the frames of "
        "the batch you are looking at now: the depth of earlier frames is gone, so locate what you may need while "
        "you see it and write the result to your memory."),
    "parameters": {"type": "object", "properties": {
        "k": {"type": "integer", "minimum": 0, "description": "The frame's number, from its header line."},
        "u": {"type": "number", "description": "The pixel's column in the frame's image, from the left edge."},
        "v": {"type": "number", "description": "The pixel's row in the frame's image, from the top."}},
        "required": ["k", "u", "v"]}}}


class _Refused(Exception):
    """A request the tool will not carry out; the message goes back to the model as `error: <message>`."""


def batch_of(steps: list[dict]) -> dict[int, dict]:
    """What `locate` needs of a batch of steps, by keyframe: the inbox-relative path of its depth image and the camera's
    pose (the step's `camera` block, not its `pose`, which is the base). A step with no depth path or no camera
    rotation is left out."""
    return {step["k"]: {"depth": step["depth"], "pos": list(step["camera"]["pos"]), "rot": list(step["camera"]["rot"])}
            for step in steps if step.get("depth") and "rot" in step.get("camera", {})}


def _arguments(arguments) -> dict:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except (ValueError, RecursionError) as exc:
            raise _Refused(f"the arguments are not valid JSON ({exc})") from None
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise _Refused("the arguments must be a JSON object")
    properties = LOCATE["function"]["parameters"]["properties"]
    for key in arguments:
        if key not in properties:
            raise _Refused(f"unexpected argument: {key} ({LOCATE_NAME} takes {', '.join(properties)})")
    for key in LOCATE["function"]["parameters"]["required"]:
        if key not in arguments:
            raise _Refused(f"missing required argument: {key}")
    return arguments


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise _Refused(f"{name} must be a number")
    return float(value)


def _metres(value: float) -> str:
    text = f"{value:.2f}"
    return "0.00" if text == "-0.00" else text


def locate(arguments, batch: dict[int, dict], workdir: Path, cam: Camera) -> str:
    """Run one `locate` call over the current `batch` (`batch_of`), whose depth images are under `workdir`: the reply
    (the frame, the pixel, the map position and the distance), or `error: <what is wrong>`."""
    try:
        args = _arguments(arguments)
        k = _number(args["k"], "k")
        if not k.is_integer() or k < 0:
            raise _Refused("k must be an integer of at least 0")
        k = int(k)
        u, v = _number(args["u"], "u"), _number(args["v"], "v")
        if k not in batch:
            if not batch:
                raise _Refused(f"no frame k{k}: there is no batch of frames to locate in")
            span = f"frame k{min(batch)}" if len(batch) == 1 else f"frames k{min(batch)} to k{max(batch)}"
            raise _Refused(f"no frame k{k} in this batch: you can locate in {span} only (the depth of earlier frames "
                           "is gone)")
        if not (0 <= u < cam.width and 0 <= v < cam.height):
            raise _Refused(f"pixel ({u:g}, {v:g}) is outside the {cam.width} x {cam.height} image: u must be at least "
                           f"0 and below {cam.width}, and v at least 0 and below {cam.height}")
        column, row = math.floor(u), math.floor(v)
        try:
            depth = read_png16(Path(workdir) / batch[k]["depth"])
        except (OSError, ValueError):
            raise _Refused(f"the depth image of frame k{k} could not be read") from None
        if depth.shape != (cam.height, cam.width):
            raise _Refused(f"the depth image of frame k{k} is not {cam.width} x {cam.height}")
        counts = int(depth[row, column])  # in the camera's depth unit
        if counts == 0:
            raise _Refused(f"no depth at pixel ({u:g}, {v:g}) of frame k{k}: it is background, sky or out of the "
                           "sensor's range; try a pixel on the surface of the thing")
        metres = counts * cam.depth_scale
        x, y, z = cam_to_map(unproject(column + 0.5, row + 0.5, metres, cam), batch[k]["pos"], batch[k]["rot"])
        return (f"frame k{k}, pixel ({u:g}, {v:g}): map position ({_metres(x)}, {_metres(y)}, {_metres(z)}) m, "
                f"{_metres(metres)} m from the camera")
    except _Refused as exc:
        return f"error: {exc}"
