"""What the robot can know: the rules shared by the question compiler and the scripted baselines (V0 §7).

One owner per concept: the thresholds of "moved", "stale" and "in view", the derivation of a room's change set from
where things were last seen in two visits, the Perceive set's widening of a large thing's tolerance, and the relative
rungs of a distance asked of the camera (`REL_SHARES`, `REL_FLOORS_M`, `relative_ladder`), the clearance rungs of a
job's own margin (`clearance_ladder`, 0.7.0), the widening kept for a thing whose top was hidden (`hidden_widening`)
and, for the graded credit (0.9.0, `credit_kernel`), the camera's one-pixel resolution at a floor point
(`resolution_m`) and the tolerances a gold row of that credit holds (`graded_tolerances`), with, for the harm families
(0.10.0), the catastrophe rule a row adds to them (`CATASTROPHES`, `with_catastrophe`).
The compiler feeds it truth-derived sightings (entities); a baseline feeds it its own tracks. Everything here is
geometry over agent-facing data (camera, pose, planar depth), so a system under test may use it too.

The seen threshold (0.07% of the image, 215 px at 640x480) is owned here (`seen_px`): the oracle channel detects from
it and the in-view rule requires it (`visibility` imports it). The compiler's sightings use `gold_seen_px`, 25% higher,
so a gold never hinges on where between 161 and 269 px a detector draws the line.

Every threshold takes the entity's kind: `seen_table.BOUNDS` may give a kind its own seen bound (hi) and glimpse bound
(lo, `glimpse_px`: under it a glimpse counts as unseen, not as the dead band); a kind without an entry, and `kind=None`,
uses the global ones (`SEEN_FRACTION`, `GLIMPSE_FRACTION`). With the identity table (no entry) every kind has
the global bounds, so passing the kind changes nothing.

Planar depth is used at a coarse grain (`coarse_depth`: the nearest depth in each 4x4 block, 160x120), because that is
what a system can afford to keep in its own state for the "in view" test; the compiler and the baselines both use it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

from . import seen_table
from .camera import CV_FROM_BODY, Camera, rot_matrix
from .credit_kernel import CATASTROPHE_PERSON, CATASTROPHE_THING, MIN_RHO_M
from .nearest_part import CONTACT_M

SEEN_FRACTION = 0.0007  # of the image: 0.07%; the seen bound (hi) of a kind with no entry in `seen_table.BOUNDS`
GLIMPSE_FRACTION = 0.0  # the glimpse bound (lo) of such a kind: 0, so any pixel is a glimpse (`glimpse_px` is >= 1)
GOLD_SEEN_SCALE = 1.25  # a sighting decides a gold only from this multiple of seen_px
DEPTH_BLOCK = 4  # side of the pixel block `coarse_depth` reduces to its nearest depth
# Camera z (m) within which a point counts as in view. The far bound is 150 m, past the distance at which the largest
# carried kind still covers its seen bound (a box truck covers 0.75 of it out to 116 m, a car out to 69 m) and past the
# city's longest in-map distance (139 m), so the cover test and the depth image decide and the bound never does. At
# 8 m (the apartment's) and then 30 m a removal the robot could see from further was neither an event nor a
# don't-care region, so a correct report of it was charged. The kept maps cannot change: a camera and a point inside a
# map are at most its 3D diagonal apart, 27.3 m in the warehouse and 11.2 m in the apartment, under 30 m and 150 m both
# (all 150 kept episodes recompiled identical).
Z_RANGE = (0.3, 150.0)
DEPTH_TOL_M = 0.1  # the depth at its pixel may be this much nearer than the point
MOVED_YES_M = 0.45  # positions at least this far apart: moved
MOVED_NO_M = 0.15  # positions closer than this: the same place
MATCH_M = 0.5  # tracking radius, set matching radius and the radius of a don't-care region
STALE_M = 1.0  # a later sighting this far (plus the kind's widening) from a thing's place makes that place stale
WIDEN_FREE_M = 0.25  # the Perceive set's widening: half a thing's length beyond this is widened for (`widening`)
# The relative rungs of a distance asked of the camera (0.6.0, PA lesson 1): an error is judged against how far the
# thing is, so a rung is a share of the range r (the slant distance from the camera to the thing, metres) with a floor
# for what is near. A rung met is a third of the credit, so the full credit takes 10% of r (at least 5 cm), two thirds
# 25% (at least 10 cm) and one third 50% (at least 20 cm). The reference agent's prompt reads these numbers.
REL_SHARES = (0.10, 0.25, 0.50)
REL_FLOORS_M = (0.05, 0.10, 0.20)
Xy = tuple[float, float]
Xyz = tuple[float, float, float]
Event = tuple[str, Xy]


def _fractions(kind: str | None) -> tuple[float, float]:
    """(lo, hi) of the kind as fractions of the image: its entry in `seen_table.BOUNDS`, else the global bounds."""
    return seen_table.BOUNDS.get(kind, (GLIMPSE_FRACTION, SEEN_FRACTION))


def seen_px(cam: Camera, kind: str | None = None) -> int:
    """Pixels an entity of the kind needs to count as seen: 0.07% of the image, 215 at 640x480, unless the kind has its
    own bound in the seen table."""
    return round(_fractions(kind)[1] * cam.width * cam.height)


def glimpse_px(cam: Camera, kind: str | None = None) -> int:
    """Pixels an entity of the kind needs for a glimpse to count at all: under this it is unseen, from it up to the
    sure-seen bound it is the dead band. At least 1, so a kind with no bound of its own counts every pixel."""
    return max(1, round(_fractions(kind)[0] * cam.width * cam.height))


def gold_seen_px(cam: Camera, kind: str | None = None) -> int:
    """Pixels a sighting needs before the compiler lets it decide a gold: 1.25 x the kind's seen_px, 269 at 640x480
    for a kind on the global bound. From glimpse_px up to this the compiler treats a sighting as the dead band (the
    question is skipped; in `changed` its place is a don't-care region) although the oracle channel detects from
    seen_px. So every gold holds for a system that detects at any threshold up to 1.25 x the kind's bound."""
    return math.ceil(GOLD_SEEN_SCALE * seen_px(cam, kind))


def min_cover_px(cam: Camera, kind: str | None = None) -> int:
    """The pixel count a thing of the kind must be expected to cover to be "in view": its seen threshold."""
    return seen_px(cam, kind)


def relative_ladder(r_m: float, widen_m: float = 0.0) -> list[float]:
    """The rungs of a distance credit for a thing `r_m` metres from the camera: for each of `REL_SHARES` and
    `REL_FLOORS_M`, `max(floor, share * r_m) + widen_m` (a large thing's `widening` is added to every rung), to the
    micrometre. Strictly increasing, whatever `r_m` (the shares and the floors both rise), so it is a valid
    `tolerances.ladder`. ValueError for an `r_m` that is not a finite number of at least 0 or a negative widening."""
    if isinstance(r_m, bool) or not isinstance(r_m, int | float) or not math.isfinite(r_m) or r_m < 0:
        raise ValueError(f"r_m must be a finite number of at least 0, got {r_m!r}")
    if widen_m < 0:
        raise ValueError(f"widen_m must be at least 0, got {widen_m!r}")
    return [round(max(floor, share * r_m) + widen_m, 6) for share, floor in zip(REL_SHARES, REL_FLOORS_M, strict=True)]


def clearance_ladder(c_m: float) -> list[float]:
    """The rungs of the clearance credit (0.7.0): `[c/2, c, 2c]` for a job's clearance `c_m` metres, to the micrometre.
    Absolute, unlike `relative_ladder`: they do not widen with range, because a margin a controller keeps does not.
    The same shape as `relative_ladder` (three strictly increasing rungs, a third of the credit each), so a gold row
    carries them as `tolerances.ladder` and the scorer is unchanged. ValueError for a `c_m` that is not a finite number
    above 0."""
    if isinstance(c_m, bool) or not isinstance(c_m, int | float) or not math.isfinite(c_m) or c_m <= 0:
        raise ValueError(f"c_m must be a finite number above 0, got {c_m!r}")
    return [round(c_m * f, 6) for f in (0.5, 1.0, 2.0)]


def resolution_m(cam_pos: Sequence[float], xy: Sequence[float], base_z: float, fx: float, fy: float) -> float | None:
    """The one-pixel error of a point on the floor plane z = `base_z` at `xy`, seen from a camera at `cam_pos` with
    focal lengths `fx` and `fy` pixels: radially `(d^2 + h^2) / (h fy)` and laterally `sqrt(d^2 + h^2) / fx`, combined
    as their hypotenuse, at least `credit_kernel.MIN_RHO_M` (an answer's natural precision and the pose's rounding); d
    is the floor distance and h = |camera z - base z|. None for a camera within `nearest_part.CONTACT_M` of the plane
    (the ray runs along it: no bound). The `rho` of a gold row's `tolerances.resolution_m`."""
    d, h = math.hypot(xy[0] - cam_pos[0], xy[1] - cam_pos[1]), abs(cam_pos[2] - base_z)
    if h <= CONTACT_M:
        return None
    slant2 = d * d + h * h
    return max(math.hypot(slant2 / (h * fy), math.sqrt(slant2) / fx), MIN_RHO_M)


HARMS = ("over", "radial", None)   # the errors on the harmful side: a number too long, an offset too far, none


def _positive(v: object, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v) or v <= 0:
        raise ValueError(f"{what} must be a finite number above 0, got {v!r}")
    return float(v)


def graded_tolerances(rho: float, need: float, harm: str | None, threshold: float | None = None) -> dict:
    """The `tolerances` of a gold row scored by the graded credit (0.9.0, `credit_kernel`), to the micrometre:
    `resolution_m` (`rho`, at least `MIN_RHO_M`), `need_m`, `harm` (`HARMS`) and, for a number, `threshold_m` (the
    body's length plus the need: the least clearance there is room to move on); an offset and a position pass no
    threshold. ValueError for a `rho` or `need` that is not a finite number above 0, a `harm` outside `HARMS` or a
    `threshold` that is not above 0."""
    if harm not in HARMS:
        raise ValueError(f"harm must be one of {list(HARMS)}, got {harm!r}")
    out = {"resolution_m": round(max(_positive(rho, "rho"), MIN_RHO_M), 6), "need_m": round(_positive(need, "need"), 6),
           "harm": harm}
    if threshold is not None:
        out["threshold_m"] = round(_positive(threshold, "threshold"), 6)
    return out


class Catastrophe(NamedTuple):
    """A harm family's catastrophe rule (`records.TOLERANCES`' `catastrophe`): `rule` (`records.CATASTROPHE_RULES`),
    `weight` (`credit_kernel`'s) and whether the row's `threshold_m` goes into it."""

    rule: str
    weight: int
    threshold: bool


# The harm families that carry a catastrophe rule from the compile, with the rule each takes (0.10.0, FD-questions-0.10
# §4.3), so that no family module is edited: `clear_path` (a corridor said longer than it is, by more than the margin,
# where the true corridor is shorter than the body's length plus C and the answer reaches it: the robot would drive into
# something it had seen), `closest_person` (a person said to be clear of the body who came closer than it has room for)
# and `person_mem` (a person placed farther away than they are). A family that sets its own `catastrophe` (the new
# families do) is not in this table and is left as it is.
# `clear_path` takes the row's `threshold_m`, the body's length plus C, as `closest_person` does (final review S5): an
# overstatement of a corridor with room to spare is not a collision (a drone corridor's median gold is 24 m against a
# margin of 2.4 m, and a 30 m answer to it hits nothing), so a long corridor overstated is merely a wrong answer, which
# the graded credit already charges; only a corridor shorter than the room the body needs, said to be one it can drive,
# is a catastrophe. It was a flat -2 for any overstatement past max(C, 2 rho).
CATASTROPHES = {"clear_path": Catastrophe("over", CATASTROPHE_THING, True),
                "closest_person": Catastrophe("over", CATASTROPHE_PERSON, True),
                "person_mem": Catastrophe("radial", CATASTROPHE_PERSON, False)}
CATASTROPHE_GOLDS = {"over": "number", "radial": "offset"}   # the gold type a table rule is about
CAP_EPS = 1e-9


def catastrophe_tolerance(rule: str, weight: int, margin: float, *, threshold: float | None = None,
                          critical: Sequence[Sequence[float]] | None = None) -> dict:
    """The `catastrophe` tolerance of a gold row (`records.TOLERANCES`), to the micrometre: `weight` (2 or 3: the
    kernel's `CATASTROPHE_THING` or `CATASTROPHE_PERSON`), `rule`, `margin_m` (a harmful error must exceed it), and for
    `over` an optional `threshold_m`, for `miss` the `critical` points. ValueError for a `margin` or `threshold` that is
    not a finite number above 0."""
    out = {"weight": weight, "rule": rule, "margin_m": round(_positive(margin, "margin"), 6)}
    if threshold is not None:
        out["threshold_m"] = round(_positive(threshold, "threshold"), 6)
    if critical is not None:
        out["critical"] = [[round(float(x), 6), round(float(y), 6)] for x, y in critical]
    return out


def with_catastrophe(family: str, gold: dict, tolerances: dict, cap: float | None = None) -> dict:
    """`tolerances` of a gold row with the family's catastrophe rule from `CATASTROPHES` added, or as they are when the
    family has none in the table, the row already holds a `catastrophe`, the gold is `unknown` or the gold stands at
    `cap` (a corridor or a distance asked up to it: nothing nearer was seen, so nothing was wrongly cleared). The
    margin is `max(C, 2 rho)` from the row's `need_m` and `resolution_m`: a harmful error larger than the clearance, or
    than twice what one pixel resolves there, whichever is more. ValueError for a gold of another type than the rule's
    (`over` is about a number, `radial` an offset)."""
    entry = CATASTROPHES.get(family)
    if entry is None or "catastrophe" in tolerances or gold["type"] == "unknown":
        return tolerances
    if gold["type"] != CATASTROPHE_GOLDS[entry.rule]:
        raise ValueError(f"the {family} catastrophe rule {entry.rule!r} is about a {CATASTROPHE_GOLDS[entry.rule]} "
                         f"gold, got a {gold['type']}")
    if cap is not None and gold["type"] == "number" and gold["value"] >= cap - CAP_EPS:
        return tolerances
    margin = max(tolerances["need_m"], 2 * tolerances["resolution_m"])
    threshold = tolerances["threshold_m"] if entry.threshold else None
    return {**tolerances, "catastrophe": catastrophe_tolerance(entry.rule, entry.weight, margin, threshold=threshold)}


def widening(size: Sequence[float]) -> float:
    """The Perceive set's widening `w` for a thing of this box size (x, y, height), metres: `max(0, max(sx, sy) / 2 -
    WIDEN_FREE_M)`, to the micrometre so float noise never reaches a gold. The compiler
    adds it to a position's rungs and a set's matching radius; it is public, so a memory-less guess reads it too
    (`table_perceive.where_went_guess`)."""
    return round(max(0.0, max(size[0], size[1]) / 2 - WIDEN_FREE_M), 6)


def hidden_widening(size: Sequence[float], cam_z: float, top_z: float) -> float:
    """The widening of a thing whose top was hidden at the sighting (0.7.0): `widening(size)` when its top, at height
    `top_z` metres, was above the camera at height `cam_z` (the camera saw only the sides, so the middle the answer
    names sits behind them), else 0 (a camera at or above the top sees its top face, so the middle is in view). Both
    heights are on the same axis (world z)."""
    return widening(size) if top_z > cam_z else 0.0


def cross_section_m2(size: Sequence[float]) -> float:
    """A nominal silhouette area for a box: the geometric mean of its three face areas, (abc)^(2/3)."""
    return float(size[0] * size[1] * size[2]) ** (2 / 3)


def would_cover_px(size: Sequence[float], z: np.ndarray | float, cam: Camera) -> np.ndarray | float:
    """Pixels the nominal silhouette covers at camera depth z."""
    return cam.fx * cam.fy * cross_section_m2(size) / np.square(z)


class Viewer:
    """Which places a keyframe had in view. `poses` is {k: (pos, rot)}; `depth_of(k)` gives the planar depth image
    (camera units, 0 = none) or None when there is none, in which case nothing counts as in view."""

    def __init__(self, cam: Camera, poses: dict[int, tuple[Sequence[float], Sequence[float]]],
                 depth_of: Callable[[int], np.ndarray | None], cover_scale: float = 1.0,
                 depth_tol: float = DEPTH_TOL_M):
        """`cover_scale` and `depth_tol` perturb the two thresholds (the compiler checks that a `changed` event
        survives +-25% and +-0.05 m); the depth image may be coarse (any size that divides the camera's)."""
        self.cam, self.depth_of = cam, depth_of
        self.cover_scale, self.depth_tol = cover_scale, depth_tol
        self._row = {k: i for i, k in enumerate(sorted(poses))}
        ordered = sorted(poses)
        self._pos = np.array([poses[k][0] for k in ordered], dtype=float).reshape(-1, 3)
        self._m = np.array([CV_FROM_BODY @ rot_matrix(poses[k][1]).T for k in ordered]).reshape(-1, 3, 3)

    def in_view(self, ks: Iterable[int], centre: Sequence[float], size: Sequence[float],
                kind: str | None = None) -> bool:
        """True if at some keyframe in `ks` the point projects into the image with camera z in Z_RANGE, nothing nearer
        than z - depth_tol blocks it, and a thing of this size would cover at least min_cover_px(kind) there."""
        rows = [self._row[k] for k in ks if k in self._row]
        if not rows:
            return False
        cam, ordered = self.cam, sorted(self._row)
        cp = np.einsum("nij,nj->ni", self._m[rows], np.asarray(centre, dtype=float) - self._pos[rows])
        z = cp[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u, v = cam.fx * cp[:, 0] / z + cam.cx, cam.fy * cp[:, 1] / z + cam.cy
            ok = (z > Z_RANGE[0]) & (z < Z_RANGE[1]) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
            ok &= would_cover_px(size, z, cam) >= min_cover_px(cam, kind) * self.cover_scale
        for i in np.flatnonzero(ok):
            depth = self.depth_of(ordered[rows[i]])
            if depth is None:
                continue
            by, bx = cam.height // depth.shape[0], cam.width // depth.shape[1]
            d = float(depth[int(v[i]) // by, int(u[i]) // bx]) * cam.depth_scale
            if d == 0.0 or d >= z[i] - self.depth_tol:
                return True
        return False


@dataclass(frozen=True)
class Item:
    """One thing across two visits: its bounding-box centre where it was last seen in visit i (`a`) and in visit j
    (`b`), None when it was not seen then. `unsure` lists places where it was only glimpsed (dead band) or where the
    sighting record contradicts the depth; an item with any yields no event, only don't-care regions. `kind` picks
    the seen bounds of its "in view" tests (None: the global ones)."""

    key: str
    size: Xyz
    a: Xyz | None = None
    b: Xyz | None = None
    unsure: tuple[Xy, ...] = ()
    kind: str | None = None


def _xy(p: Xyz | Xy) -> Xy:
    return (round(float(p[0]), 3), round(float(p[1]), 3))


def room_events(room: str, room_of: Callable[[Xy], str | None], items: Iterable[Item],
                in_view_i: Callable[[Xyz, Xyz, str | None], bool], in_view_j: Callable[[Xyz, Xyz, str | None], bool]
                ) -> tuple[list[Event], list[Xy]]:
    """The changes in `room` between visit i and visit j, and the places whose events cannot be told.

    - moved: seen in both visits inside the room, at least MOVED_YES_M apart; reported where it is in visit j.
    - removed: seen in visit i inside the room and then seen outside it, or not seen in visit j while its place was in
      view (`in_view_j(centre, size, kind)`); reported at the old place.
    - added: the converse, reported at the new place.
    - two sightings MOVED_NO_M to MOVED_YES_M apart, and unsure items: no event, a don't-care region at each of
      their places inside the room.
    """
    events: list[Event] = []
    dont_care: list[Xy] = []
    for it in items:
        places = [p for p in (it.a, it.b) if p is not None]
        if it.unsure:
            dont_care += [_xy(p) for p in (*places, *it.unsure) if room_of(_xy(p)) == room]
            continue
        a_in = it.a is not None and room_of(_xy(it.a)) == room
        b_in = it.b is not None and room_of(_xy(it.b)) == room
        if it.a is not None and it.b is not None:
            d = math.dist(_xy(it.a), _xy(it.b))
            if d < MOVED_NO_M or not (a_in or b_in):
                continue
            if d < MOVED_YES_M:
                dont_care += [_xy(p) for p in (it.a, it.b)]
            elif a_in and b_in:
                events.append(("moved", _xy(it.b)))
            elif a_in:
                events.append(("removed", _xy(it.a)))
            else:
                events.append(("added", _xy(it.b)))
        elif it.a is not None and a_in and in_view_j(it.a, it.size, it.kind):
            events.append(("removed", _xy(it.a)))
        elif it.b is not None and b_in and in_view_i(it.b, it.size, it.kind):
            events.append(("added", _xy(it.b)))
    return sorted(events), dont_care


# -- depth images ------------------------------------------------------------------------------------------------
def coarse_depth(depth: np.ndarray, block: int = DEPTH_BLOCK) -> np.ndarray:
    """Reduce a depth image to the nearest depth in each block x block pixels (0, no depth, is ignored unless the whole
    block has none)."""
    h, w = depth.shape
    blocks = depth[: h - h % block, : w - w % block].astype(np.uint32).reshape(h // block, block, w // block, block)
    near = np.where(blocks == 0, np.uint32(1 << 30), blocks).min(axis=(1, 3))
    return np.where(near == 1 << 30, 0, near).astype(np.uint16)


@lru_cache(maxsize=512)
def _depth(path: str, mtime_ns: int) -> np.ndarray:
    with Image.open(path) as im:
        return coarse_depth(np.asarray(im))


def load_depth(path: str | Path) -> np.ndarray | None:
    """A 16-bit planar depth PNG as a coarse (h/4, w/4) array in camera units (the unit of the episode's `camera.json`:
    mm unless the map names another), or None if there is no file."""
    p = Path(path)
    return _depth(str(p), p.stat().st_mtime_ns) if p.exists() else None
