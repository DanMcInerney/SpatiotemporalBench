"""What a robot is: its body, speeds, camera, depth range, status vocabulary and keyframe policy.

The four robots are plain data (`ROBOTS`); everything derived from one (the public `robot.json`, `sensors/camera.json`,
the camera's pose, the rules a status and a body must keep) is a function here, so the generator, the validator, the
renderer, the compiler and a system under test read one definition. A system parses the robot it is sent with
`from_doc`. This module imports only `geometry`, so baselines and the agent may import it.

The camera model (`IspSpec`) is part of the robot, so tuning one robot's camera moves no other robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from .geometry import Quat, q, quat_yaw_pitch

Xyz = tuple[float, float, float]


@dataclass(frozen=True)
class Body:
    shape: str              # "disc" or "rect"
    length_m: float         # along the base's x axis (forward); a disc's diameter
    width_m: float          # along the base's y axis (left); a disc's diameter
    height_m: float         # the top of the body above the base origin
    climb_m: float          # the highest floor step it drives over (0 for a flier)
    clearance_m: float      # the margin its planner keeps from obstacles, beyond the body
    under_margin_m: float   # a ground robot passes under a piece whose under_m >= height_m + under_margin_m


@dataclass(frozen=True)
class Tool:                 # a part of the robot a family reads: the vacuum's brush
    name: str
    shape: str              # "disc" (turning in place sweeps nothing new) or "rect"
    size_m: tuple[float, float]
    offset_m: tuple[float, float]   # its centre in the base frame


@dataclass(frozen=True)
class IspSpec:              # the camera's processor: camera_model.develop applies it to the linear render
    vignette: float         # corner falloff: x (1 - vignette x (1 - cos^4 theta)), theta from the optical axis
    noise_a: float          # sensor noise on the exposed signal, before the gain and the denoiser: Poisson-Gaussian,
    noise_b: float          #   variance noise_a x + noise_b per channel, in the render's linear unit
    meter_sigma: float      # centre-weighted metering of log luminance: Gaussian weights, sigma as a share of the width
    target: float           # the metered log-average's target (0.18)
    exposure_ev: tuple[float, float]   # the exposure time's range in EV from the linear unit; it scales the signal
                                       #   before the sensor noise, so a short exposure of a bright scene stays clean
    gain_ev: float          # the analogue gain, 0 to this many EV, after the noise, once the exposure is at its top
    wb_clamp: float         # grey-world gains within [1 - wb_clamp, 1 + wb_clamp] of green
    sharpen: float          # unsharp-mask amount after the tone map; 0 for none
    jpeg_quality: int
    # the optics, the denoiser and the tone curve of a camera with a real lens and processor (R.1's fix round); each
    # default is the stage off, so a robot that names none of them has a plain camera
    contrast: float = 1.0           # the tone curve's contrast about mid-grey (1 = none): its log-log slope
    black_level: float = 0.0        # the display-linear value of black: the sensor's offset and the lens's flare
    glare: float = 0.0              # the share of the light the optics scatter over the frame as a veil (0 = none)
    denoise_luma_px: float = 0.0    # the denoiser leaves the sensor's luma noise correlated over this sigma, in pixels
    denoise_chroma_px: float = 0.0  # ... and its colour noise over this sigma (0 = the noise as the sensor made it)


@dataclass(frozen=True)
class CameraSpec:
    width: int
    height: int
    hfov_deg: float         # fx = fy = (width / 2) / tan(hfov / 2); cx = width / 2; cy = height / 2
    mount_m: tuple[float, float, float]   # the camera centre in the base frame (x forward, y left, z up)
    pitch_deg: float        # the default pitch, positive up
    pitch_min_deg: float    # the gimbal's limits; both equal pitch_deg for a fixed mount
    pitch_max_deg: float
    isp: IspSpec


@dataclass(frozen=True)
class DepthSpec:
    unit: str               # "mm", "cm" or "m" (camera.DEPTH_UNITS)
    near_m: float           # nearer surfaces read 0
    far_m: float            # farther surfaces read 0; far_m <= 65,535 x unit


@dataclass(frozen=True)
class StatusSpec:
    phases: tuple[str, ...]           # the closed phase vocabulary
    keys: dict[str, str]              # every other status key -> "int", "bool" or "float"
    start: dict[str, object]          # each task's first step: "phase" -> a phase; a key -> (lo, hi) or a value


@dataclass(frozen=True)
class FramePolicy:
    move_m: float           # a keyframe once the base has moved this far since the last one
    turn_deg: float         # or once the heading or the camera pitch has turned this far
    still_s: float          # or once this long has passed with neither
    min_dt_s: float         # never two keyframes of one task closer in time than this


@dataclass(frozen=True)
class Robot:
    id: str                 # "vacuum", "helper", "amr", "drone"
    name: str               # the brief's words: "robot vacuum"
    flies: bool
    body: Body
    speeds_mps: dict[str, tuple[float, float]]   # per phase: (min, max)
    turn_dps: float         # in-place turn rate, degrees a second
    camera: CameraSpec
    depth: DepthSpec
    status: StatusSpec
    frames: FramePolicy
    tools: tuple[Tool, ...] = ()
    sources: tuple[str, ...] = ()       # URLs and "modelling choice: ..." notes


# The four robots. Every number is frozen for M0; a later change, with a source, goes through the coordinator. Each
# robot holds its own `IspSpec`, so tuning one moves no other (R.1 tunes the vacuum's).
ROBOTS: dict[str, Robot] = {
    "vacuum": Robot(
        id="vacuum", name="robot vacuum", flies=False,
        body=Body("disc", length_m=0.34, width_m=0.34, height_m=0.09, climb_m=0.02, clearance_m=0.02,
                  under_margin_m=0.01),
        speeds_mps={"cleaning": (0.28, 0.31), "transit": (0.30, 0.35), "undocking": (0.10, 0.15),
                    "docking": (0.10, 0.15)},
        turn_dps=45.0,
        camera=CameraSpec(
            640, 480, 80.0, mount_m=(0.13, 0.0, 0.08), pitch_deg=5.0, pitch_min_deg=5.0, pitch_max_deg=5.0,
            # frozen for M1 (the apartment's look was approved on 2026-10-07): fitted to real Roomba j7 frames
            isp=IspSpec(vignette=0.65, noise_a=8e-4, noise_b=5e-6, meter_sigma=0.35, target=0.09,
                        exposure_ev=(-8.0, 2.0), gain_ev=6.0, wb_clamp=0.1, sharpen=0.25, jpeg_quality=92,
                        contrast=1.6, black_level=0.0025, glare=0.01, denoise_luma_px=0.8, denoise_chroma_px=1.0)),
        depth=DepthSpec("mm", near_m=0.01, far_m=4.0),
        status=StatusSpec(
            phases=("docked", "undocking", "cleaning", "transit", "docking", "paused"),
            keys={"battery_pct": "int", "brush": "bool"},
            start={"phase": "docked", "battery_pct": (90, 100), "brush": False}),
        frames=FramePolicy(move_m=1.0, turn_deg=45.0, still_s=5.0, min_dt_s=1.0),
        tools=(Tool("brush", "disc", size_m=(0.25, 0.25), offset_m=(0.0, 0.0)),),
        sources=("Roomba 338-353 mm across, 86-94 mm high, 0.306 m/s default (PRIOR-ART §7; US 7,388,343)",
                 "iRobot recharge and resume (design §4)",
                 "modelling choice: the camera mount, a 4 m depth range (short-range front 3D sensing), the brush "
                 "disc"),
    ),
    "helper": Robot(
        id="helper", name="home helper robot", flies=False,
        body=Body("rect", length_m=0.34, width_m=0.33, height_m=1.41, climb_m=0.01, clearance_m=0.10,
                  under_margin_m=0.05),
        speeds_mps={"navigating": (0.5, 0.8), "carrying": (0.4, 0.6), "grasping": (0.1, 0.2), "placing": (0.1, 0.2)},
        turn_dps=45.0,
        camera=CameraSpec(
            640, 480, 80.0, mount_m=(0.0, 0.0, 1.30), pitch_deg=-15.0, pitch_min_deg=-60.0, pitch_max_deg=20.0,
            isp=IspSpec(vignette=0.3, noise_a=6e-4, noise_b=1e-5, meter_sigma=0.35, target=0.18,
                        exposure_ev=(-8.0, 2.0), gain_ev=6.0, wb_clamp=0.1, sharpen=0.25, jpeg_quality=92)),
        depth=DepthSpec("mm", near_m=0.2, far_m=10.0),
        status=StatusSpec(
            phases=("navigating", "grasping", "carrying", "placing", "waiting"),
            keys={"battery_pct": "int"},
            start={"phase": "waiting", "battery_pct": (80, 100)}),
        frames=FramePolicy(move_m=0.75, turn_deg=30.0, still_s=5.0, min_dt_s=1.0),
        sources=("Stretch 3 class: 33 × 34 × 141 cm with a pan-tilt head camera (hello-robot.com)",
                 "speeds: design §4 (the M3 lane cites a platform)"),
    ),
    "amr": Robot(
        id="amr", name="tote robot", flies=False,
        body=Body("rect", length_m=0.80, width_m=0.58, height_m=0.65, climb_m=0.01, clearance_m=0.10,
                  under_margin_m=0.05),
        speeds_mps={"transit": (0.8, 1.2), "approach": (0.2, 0.2), "undocking": (0.2, 0.3)},
        turn_dps=45.0,
        camera=CameraSpec(
            640, 480, 80.0, mount_m=(0.38, 0.0, 0.50), pitch_deg=-10.0, pitch_min_deg=-10.0, pitch_max_deg=-10.0,
            isp=IspSpec(vignette=0.3, noise_a=6e-4, noise_b=1e-5, meter_sigma=0.35, target=0.18,
                        exposure_ev=(-8.0, 2.0), gain_ev=6.0, wb_clamp=0.1, sharpen=0.25, jpeg_quality=92)),
        depth=DepthSpec("mm", near_m=0.2, far_m=10.0),
        status=StatusSpec(
            phases=("transit", "approach", "at_station", "undocking", "waiting"),
            keys={"battery_pct": "int", "loaded": "bool"},
            start={"phase": "waiting", "battery_pct": (60, 100), "loaded": False}),
        frames=FramePolicy(move_m=1.0, turn_deg=30.0, still_s=5.0, min_dt_s=1.0),
        sources=("MiR250: 800 × 580 × 300 mm, 2.0 m/s maximum (mobile-industrial-robots.com specifications)",
                 "0.2 m/s near docking (design §4)",
                 "modelling choice: a tote module to 0.65 m, the camera"),
    ),
    "drone": Robot(
        id="drone", name="delivery drone", flies=True,
        body=Body("disc", length_m=1.30, width_m=1.30, height_m=0.30, climb_m=0.0, clearance_m=3.0,
                  under_margin_m=0.0),
        speeds_mps={"cruise": (29.0, 29.0), "approach": (8.0, 12.0), "descend": (1.0, 2.0), "lower": (0.5, 1.0),
                    "climb": (3.0, 5.0)},
        turn_dps=30.0,
        camera=CameraSpec(
            640, 480, 80.0, mount_m=(0.0, 0.0, -0.10), pitch_deg=-35.0, pitch_min_deg=-90.0, pitch_max_deg=-20.0,
            isp=IspSpec(vignette=0.3, noise_a=6e-4, noise_b=1e-5, meter_sigma=0.35, target=0.18,
                        exposure_ev=(-8.0, 2.0), gain_ev=6.0, wb_clamp=0.1, sharpen=0.25, jpeg_quality=92)),
        depth=DepthSpec("cm", near_m=0.5, far_m=600.0),
        status=StatusSpec(
            phases=("preflight", "takeoff", "climb", "cruise", "approach", "inspect", "descend", "lower", "release",
                    "climb_out", "return", "land"),
            keys={"battery_pct": "int", "tether_m": "float"},
            start={"phase": "preflight", "battery_pct": (95, 100), "tether_m": 0.0}),
        frames=FramePolicy(move_m=3.0, turn_deg=30.0, still_s=3.0, min_dt_s=1.0),
        sources=("Wing: cruise about 45 m at about 29 m/s, lowering from about 7 m (wing.com/safety; design §4)",
                 "modelling choice: body, gimbal"),
    ),
}

# Every JSON key `robot_doc` can hold, for the leak allowlist. Fail-closed: the field names are written out by hand (a
# field added to a record but not here is refused by the leak scan, never allowed by accident), plus every phase and
# status key of every robot, which appear as keys of `speeds_mps`, `status.keys` and `status.start`. `schema` is no
# longer written; it stays so that the `robot.json` of an earlier build still passes the scan.
_FIELD_KEYS = frozenset({
    "schema", "id", "name", "flies", "body", "shape", "length_m", "width_m", "height_m", "climb_m", "clearance_m",
    "under_margin_m", "speeds_mps", "turn_dps", "camera", "width", "height", "hfov_deg", "mount_m", "pitch_deg",
    "pitch_min_deg", "pitch_max_deg", "isp", "vignette", "noise_a", "noise_b", "meter_sigma", "target", "exposure_ev",
    "gain_ev", "wb_clamp", "sharpen", "jpeg_quality", "contrast", "black_level", "glare", "denoise_luma_px",
    "denoise_chroma_px",
    "depth", "unit", "near_m", "far_m", "status", "phases", "keys",
    "start", "phase", "frames", "move_m", "turn_deg", "still_s", "min_dt_s", "tools", "size_m", "offset_m", "sources",
})
DOC_KEYS: frozenset[str] = (_FIELD_KEYS | {p for r in ROBOTS.values() for p in r.status.phases}
                            | {k for r in ROBOTS.values() for k in r.status.keys})


# -- the public documents ----------------------------------------------------------------------------------------
def _plain(x: Any) -> Any:
    """A record as JSON holds it: dataclasses as dicts (in field order), tuples as lists."""
    if is_dataclass(x) and not isinstance(x, type):
        return {f.name: _plain(getattr(x, f.name)) for f in fields(x)}
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, tuple | list):
        return [_plain(v) for v in x]
    return x


def robot_doc(r: Robot) -> dict:
    """The public `robot.json`: every field, nested records as dicts and tuples as lists."""
    return _plain(r)


def from_doc(d: dict) -> Robot:
    """`robot_doc`'s inverse (tuples rebuilt, so `from_doc(robot_doc(r)) == r`). It reads only the keys it names: the
    `robot.json` of an earlier build may carry a `schema`. A system parses the robot of its `start` message with it."""
    cam, isp, st = d["camera"], d["camera"]["isp"], d["status"]
    return Robot(
        id=d["id"], name=d["name"], flies=d["flies"], body=Body(**d["body"]),
        speeds_mps={k: tuple(v) for k, v in d["speeds_mps"].items()}, turn_dps=d["turn_dps"],
        camera=CameraSpec(**{**cam, "mount_m": tuple(cam["mount_m"]),
                             "isp": IspSpec(**{**isp, "exposure_ev": tuple(isp["exposure_ev"])})}),
        depth=DepthSpec(**d["depth"]),
        status=StatusSpec(tuple(st["phases"]), dict(st["keys"]),
                          {k: tuple(v) if isinstance(v, list) else v for k, v in st["start"].items()}),
        frames=FramePolicy(**d["frames"]),
        tools=tuple(Tool(t["name"], t["shape"], tuple(t["size_m"]), tuple(t["offset_m"])) for t in d["tools"]),
        sources=tuple(d["sources"]),
    )


def _focal_px(c: CameraSpec) -> float:
    return (c.width / 2) / math.tan(math.radians(c.hfov_deg / 2))


def camera_doc(r: Robot) -> dict:
    """`sensors/camera.json`: the pinhole intrinsics (square pixels, the principal point at the centre) and the depth
    unit and range."""
    c = r.camera
    f = round(_focal_px(c), 6)
    return {"width": c.width, "height": c.height, "hfov_deg": c.hfov_deg, "fx": f, "fy": f, "cx": c.width / 2,
            "cy": c.height / 2, "depth_unit": r.depth.unit, "near_m": r.depth.near_m, "far_m": r.depth.far_m}


# -- rules a robot's own numbers decide --------------------------------------------------------------------------
def camera_pose(r: Robot, base: Xyz, yaw: float, pitch: float) -> tuple[Xyz, Quat]:
    """The camera's pose for a base at `base` heading `yaw` (radians) with the camera pitched `pitch` (radians, up
    positive): its mount turned by the heading and added to the base, and the body rotation of yaw then pitch."""
    mx, my, mz = r.camera.mount_m
    c, s = math.cos(yaw), math.sin(yaw)
    pos = (q(base[0] + c * mx - s * my), q(base[1] + s * mx + c * my), q(base[2] + mz))
    return pos, quat_yaw_pitch(yaw, pitch)


def passes_under(r: Robot, under_m: float) -> bool:
    """A ground robot fits under a piece whose open underside is at least its height plus its margin."""
    return not r.flies and under_m >= r.body.height_m + r.body.under_margin_m - 1e-9


def over_m(r: Robot, base_z: float) -> float | None:
    """The height a flier clears at base height base_z: base_z - r.body.clearance_m; None for a ground robot."""
    return base_z - r.body.clearance_m if r.flies else None


def flies_over(over: float | None, top_m: float) -> bool:
    """The one rule at the height a flier clears (`over_m`): it passes over a thing exactly when the thing's top is
    below it, and a thing whose top is at it or above obstructs. A ground robot (`over` None) passes over nothing."""
    return over is not None and top_m < over


def blind_m(r: Robot) -> float:
    """The floor distance from the body's front edge to where the lowest image row meets the floor, for a ground robot
    at its default pitch: mount_z / tan(vfov / 2 - pitch) - (length_m / 2 - mount_x). Inf when that row never meets
    it."""
    if r.flies:
        raise ValueError(f"{r.id} flies: it has no floor in view to be blind to")
    c = r.camera
    dip = math.atan((c.height / 2) / _focal_px(c)) - math.radians(c.pitch_deg)
    if dip <= 0:
        return math.inf
    return c.mount_m[2] / math.tan(dip) - (r.body.length_m / 2 - c.mount_m[0])


_KINDS = {
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "float": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "bool": lambda v: isinstance(v, bool),
}


def check_status(r: Robot, status: dict, *, first: bool) -> list[str]:
    """Problems with one step's status, `{"phase": ..., <the robot's other keys>}` (the task index is not in it): an
    unknown phase, a key missing, undeclared or of the wrong type. With `first`, also that it is in the start set: the
    start phase, and each key within its (lo, hi) (inclusive) or equal to its value."""
    spec, problems = r.status, []
    phase = status.get("phase")
    if phase not in spec.phases:
        problems.append(f"unknown phase {phase!r}; {r.id} has {', '.join(spec.phases)}")
    for key in spec.keys:
        if key not in status:
            problems.append(f"status key {key!r} is missing")
    for key, value in status.items():
        if key == "phase":
            continue
        if key not in spec.keys:
            problems.append(f"status key {key!r} is not one of {r.id}'s: {', '.join(spec.keys)}")
        elif not _KINDS[spec.keys[key]](value):
            problems.append(f"status key {key!r} is {value!r}; it must be {spec.keys[key]}")
        elif first and not _in_start(spec.start[key], value):
            problems.append(f"status key {key!r} starts a task at {value!r}, outside the start set {spec.start[key]}")
    if first and phase != spec.start["phase"]:
        problems.append(f"a task starts in phase {spec.start['phase']!r}, not {phase!r}")
    return problems


def _in_start(want: Any, value: Any) -> bool:
    return want[0] <= value <= want[1] if isinstance(want, tuple) else value == want
