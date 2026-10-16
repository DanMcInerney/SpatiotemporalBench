"""The ego records of the world timeline: a robot's keyframes, tasks and decision points.

A keyframe holds both the camera and the base that carries it. `pos` and `rot` are the CAMERA's, derived from the base
by `robots.camera_pose` and checked by `validate`, so the renderer and the pose rows read them as they always did;
`base`, `yaw` (the base's heading, counter-clockwise from +x) and `cam_pitch` (positive up) are what the robot did. A
family about where the robot was reads `base`, never `pos`. Tasks are half-open keyframe ranges [k0, k1) that cover the
keyframes without a gap; a decision point is the last step before the robot commits to a manoeuvre, and may carry the
`response` the job gave after it (what the robot then did, which follows the world: "go", "abort", "skip",
"elsewhere"). The key is written only when set, so an ego without responses is byte for byte what it was.
"""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import Quat, q


@dataclass(frozen=True)
class Keyframe:
    k: int
    t: float
    pos: tuple[float, float, float]  # the camera
    rot: Quat  # the camera: qz(yaw) * qy(-pitch)
    base: tuple[float, float, float]  # the base origin
    yaw: float  # the base heading, radians
    cam_pitch: float  # the camera's pitch, radians, positive up
    task: int  # an index into Ego.tasks
    phase: str
    status: dict[str, object]  # the robot's other status keys: battery_pct, brush, ...


@dataclass(frozen=True)
class TaskPlan:
    kind: str  # one of the job's task kinds: "clean_room"
    params: dict[str, object]  # world-independent words the brief may use: {"room": "living room"}


@dataclass(frozen=True)
class Task:
    index: int
    k0: int  # keyframes [k0, k1)
    k1: int
    plan: TaskPlan
    targets: tuple[str, ...]  # private: the region or entity ids the job acts on


@dataclass(frozen=True)
class DecisionPoint:
    k: int  # the last step before the robot commits to the manoeuvre
    kind: str  # in the job's closed vocabulary
    response: str | None = None  # what the robot did after the ask, for a job that answers one; None writes no key


@dataclass(frozen=True)
class Ego:
    keyframes: list[Keyframe]
    tasks: list[Task]
    decisions: list[DecisionPoint]


def _quantised(x: object) -> object:
    """A status or parameter value as the file holds it: floats at 6 decimals (bools and ints as they are)."""
    return q(x) if isinstance(x, float) else x


def ego_dict(ego: Ego) -> dict:
    """The ego as the canonical file holds it: fixed key order, quantised floats, status and params by key."""
    return {
        "keyframes": [{"k": kf.k, "t": q(kf.t), "pos": [q(v) for v in kf.pos], "rot": [q(v) for v in kf.rot],
                       "base": [q(v) for v in kf.base], "yaw": q(kf.yaw), "cam_pitch": q(kf.cam_pitch),
                       "task": kf.task, "phase": kf.phase,
                       "status": {k: _quantised(v) for k, v in sorted(kf.status.items())}} for kf in ego.keyframes],
        "tasks": [{"index": t.index, "k0": t.k0, "k1": t.k1,
                   "plan": {"kind": t.plan.kind,
                            "params": {k: _quantised(v) for k, v in sorted(t.plan.params.items())}},
                   "targets": list(t.targets)} for t in ego.tasks],
        "decisions": [{"k": d.k, "kind": d.kind, **({} if d.response is None else {"response": d.response})}
                      for d in ego.decisions],
    }


def ego_from_dict(d: dict) -> Ego:
    """`ego_dict`'s inverse."""
    return Ego(
        [Keyframe(int(k["k"]), float(k["t"]), tuple(float(v) for v in k["pos"]), tuple(float(v) for v in k["rot"]),
                  tuple(float(v) for v in k["base"]), float(k["yaw"]), float(k["cam_pitch"]), int(k["task"]),
                  k["phase"], dict(k["status"])) for k in d["keyframes"]],
        [Task(int(t["index"]), int(t["k0"]), int(t["k1"]), TaskPlan(t["plan"]["kind"], dict(t["plan"]["params"])),
              tuple(t["targets"])) for t in d["tasks"]],
        [DecisionPoint(int(x["k"]), x["kind"], x.get("response")) for x in d["decisions"]],
    )
