"""Motion primitives to keyframes under a robot's frame policy (`robots.FramePolicy`).

A job describes what its robot does as a list of primitives (straight `Move`s, in-place `Turn`s, `Tilt`s of the camera,
`Hold`s and `Mark`s); `frames` samples them. A frame is required:
- at the start;
- at the end of every `Turn` and `Tilt`, so a vertex of the path is a keyframe;
- at every `Mark`, a decision point or whatever else the job needs a frame at;
- at the end of every phase's run of primitives, so a phase's last moment resolves (an event names it).
Otherwise a frame comes when the base has travelled `move_m` since the last one, or its heading or its camera pitch has
turned `turn_deg` (each counted apart), or `still_s` seconds have passed, whichever is first. Required frames at one
instant merge into one (a `Mark` after a `Turn`); two that cannot merge (another phase, or two decisions) are `min_dt_s`
apart, the robot holding still until then, as real robots pause. So the straight lines between consecutive frames are
the path, and every vertex is a frame: M2's `uncovered` reads the brush's swath from them. Nothing here reads a
timeline or names a robot or a phase.

A flier's primitives carry the base's height (`z`) and the camera's pitch (`pitch`, radians, positive up; None: the
job's one fixed pitch, as a ground robot's): travel counts the 3-D length, so a vertical climb is a `Move` of no length
on the ground and is not empty, and a frame carries both. Every default is a ground robot's, so a list of primitives
that names neither samples to the frames it always did.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .geometry import Point, distance, q, wrap_angle
from .robots import FramePolicy

EPS = 1e-9


@dataclass(frozen=True)
class Move:
    """A straight run from `a` to `b` at `speed` m/s. `yaw` is the heading: the direction of travel, or its opposite
    when the robot reverses."""

    a: Point
    b: Point
    yaw: float
    speed: float
    phase: str
    z: tuple[float, float] = (0.0, 0.0)  # the base's height at `a` and at `b`
    pitch: float | None = None  # the camera's pitch all along


@dataclass(frozen=True)
class Turn:
    """A turn in place at `at`, the short way from `yaw0` to `yaw1`, at `rate_dps` degrees a second."""

    at: Point
    yaw0: float
    yaw1: float
    rate_dps: float
    phase: str
    z: float = 0.0
    pitch: float | None = None


@dataclass(frozen=True)
class Tilt:
    """The camera turning in place at `at` (heading `yaw`, base height `z`) from `pitch0` to `pitch1` radians at
    `rate_dps` degrees a second: a gimbal."""

    at: Point
    yaw: float
    z: float
    pitch0: float
    pitch1: float
    rate_dps: float
    phase: str


@dataclass(frozen=True)
class Hold:
    """Standing still at `at` for `seconds`."""

    at: Point
    yaw: float
    seconds: float
    phase: str
    z: float = 0.0
    pitch: float | None = None


@dataclass(frozen=True)
class Mark:
    """A required frame at the current pose, in the phase of the primitive before it."""

    decision: str | None = None


@dataclass(frozen=True)
class Frame:
    t: float
    xy: Point
    yaw: float
    phase: str
    decision: str | None
    z: float = 0.0  # the base's height
    pitch: float | None = None  # the camera's pitch, radians; None: the job's fixed pitch


Prim = Move | Turn | Tilt | Hold | Mark


class _Sampler:
    """The robot's state while the primitives run, and the frames so far."""

    def __init__(self, policy: FramePolicy, t0: float):
        self.policy = policy
        self.t, self.xy, self.yaw, self.phase = t0, (0.0, 0.0), 0.0, ""
        self.z, self.pitch = 0.0, None
        self.out: list[Frame] = []
        self.travel = self.turned = self.tilted = 0.0  # since the last frame: metres, radians of heading, of pitch

    def _frame(self, decision: str | None = None) -> None:
        pitch = None if self.pitch is None else q(self.pitch)
        self.out.append(Frame(q(self.t), (q(self.xy[0]), q(self.xy[1])), q(wrap_angle(self.yaw)), self.phase, decision,
                              q(self.z), pitch))
        self.travel = self.turned = self.tilted = 0.0

    def require(self, decision: str | None = None) -> None:
        """A required frame now: merged into the last frame when that is the same instant and phase (and holds no
        other decision); else at least `min_dt_s` after it, the robot holding still until then."""
        last = self.out[-1] if self.out else None
        if last is not None:
            gap = q(self.t) - last.t
            if gap < EPS and last.phase == self.phase and (decision is None or last.decision in (None, decision)):
                self.out[-1] = replace(last, decision=decision or last.decision)
                return
            if gap < self.policy.min_dt_s - EPS:
                self.t = last.t + self.policy.min_dt_s
        self._frame(decision)

    def _due(self, speed: float, turn_rate: float, tilt_rate: float) -> float:
        """Seconds until the next threshold frame at this `speed` (m/s), `turn_rate` and `tilt_rate` (rad/s): the first
        threshold reached, and never before `min_dt_s` after the last frame."""
        p, last = self.policy, self.out[-1].t
        waits = [p.still_s - (self.t - last)]
        if speed > 0:
            waits.append((p.move_m - self.travel) / speed)
        if turn_rate > 0:
            waits.append((math.radians(p.turn_deg) - self.turned) / turn_rate)
        if tilt_rate > 0:
            waits.append((math.radians(p.turn_deg) - self.tilted) / tilt_rate)
        return max(min(waits), last + p.min_dt_s - self.t, 0.0)

    def run(self, seconds: float, speed: float, turn_rate: float, advance, tilt_rate: float = 0.0) -> None:
        """Let `seconds` pass at this `speed`, `turn_rate` and `tilt_rate`; `advance(dt)` moves the state by dt. A
        threshold frame falls wherever the policy says."""
        left = seconds
        while (wait := self._due(speed, turn_rate, tilt_rate)) <= left + EPS:
            advance(wait)
            left -= wait
            self._frame()
        advance(max(left, 0.0))


def frames(prims: list[Prim], policy: FramePolicy, t0: float = 0.0) -> list[Frame]:
    """The keyframes of the primitives, starting at time `t0`, as the module docstring says. The first primitive must
    be a `Move`, `Turn`, `Tilt` or `Hold` (a `Mark` has no phase before it)."""
    s = _Sampler(policy, t0)
    work = [p for p in prims if not _empty(p)]
    for i, prim in enumerate(work):
        if isinstance(prim, Mark):
            if not s.out:
                raise ValueError("a Mark needs a primitive before it")
            s.require(prim.decision)
            continue
        s.xy, s.yaw, s.phase = _start(prim), _start_yaw(prim), prim.phase
        s.z, s.pitch = _start_z(prim), _start_pitch(prim)
        if not s.out:
            s.require()
        _drive(s, prim)
        if isinstance(prim, Turn | Tilt):
            s.require()
        following = next((p for p in work[i + 1:] if not isinstance(p, Mark)), None)
        if following is None or following.phase != prim.phase:
            s.require()
    return s.out


def _length(prim: Move) -> float:
    """A move's length, in 3-D when it climbs: exactly its ground distance when it does not."""
    ground, dz = distance(prim.a, prim.b), prim.z[1] - prim.z[0]
    return ground if dz == 0.0 else math.hypot(ground, dz)


def _empty(prim: Prim) -> bool:
    """A primitive that does nothing: a move of no length, a turn or a tilt of no angle."""
    if isinstance(prim, Move):
        return _length(prim) < EPS
    if isinstance(prim, Turn):
        return abs(wrap_angle(prim.yaw1 - prim.yaw0)) < EPS
    if isinstance(prim, Tilt):
        return abs(prim.pitch1 - prim.pitch0) < EPS
    return False


def _start(prim: Move | Turn | Tilt | Hold) -> Point:
    return prim.a if isinstance(prim, Move) else prim.at


def _start_yaw(prim: Move | Turn | Tilt | Hold) -> float:
    return prim.yaw0 if isinstance(prim, Turn) else prim.yaw


def _start_z(prim: Move | Turn | Tilt | Hold) -> float:
    return prim.z[0] if isinstance(prim, Move) else prim.z


def _start_pitch(prim: Move | Turn | Tilt | Hold) -> float | None:
    return prim.pitch0 if isinstance(prim, Tilt) else prim.pitch


def _drive(s: _Sampler, prim: Move | Turn | Tilt | Hold) -> None:
    """Run one primitive from its start state, ending exactly on its end state (no drift from the steps)."""
    if isinstance(prim, Move):
        length = _length(prim)
        ux, uy = (prim.b[0] - prim.a[0]) / length, (prim.b[1] - prim.a[1]) / length
        uz = (prim.z[1] - prim.z[0]) / length

        def advance(dt: float) -> None:
            s.t += dt
            s.xy = (s.xy[0] + ux * prim.speed * dt, s.xy[1] + uy * prim.speed * dt)
            s.z += uz * prim.speed * dt
            s.travel += prim.speed * dt

        s.run(length / prim.speed, prim.speed, 0.0, advance)
        s.xy, s.z = prim.b, prim.z[1]
    elif isinstance(prim, Tilt):
        rate = math.radians(prim.rate_dps)
        delta = prim.pitch1 - prim.pitch0

        def tilt(dt: float) -> None:
            s.t += dt
            s.pitch += math.copysign(rate * dt, delta)
            s.tilted += rate * dt

        s.run(abs(delta) / rate, 0.0, 0.0, tilt, tilt_rate=rate)
        s.pitch = prim.pitch1
    elif isinstance(prim, Turn):
        rate = math.radians(prim.rate_dps)
        delta = wrap_angle(prim.yaw1 - prim.yaw0)

        def advance(dt: float) -> None:
            s.t += dt
            s.yaw += math.copysign(rate * dt, delta)
            s.turned += rate * dt

        s.run(abs(delta) / rate, 0.0, rate, advance)
        s.yaw = prim.yaw1
    else:
        s.run(prim.seconds, 0.0, 0.0, lambda dt: setattr(s, "t", s.t + dt))
