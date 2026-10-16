"""`path_log`'s answers to `ego_now` and `ego_mem` (a module of `baselines_path.PATH_MODULES`; the gold is
`questions_ego`).

It reads no picture, so it never knows what is in view or what it saw. What a public layout offers it is that things sit
at fixed spots (a staging lane, a shelf, a roll cage at one end of an aisle) where other scripts saw things of the kind.
Each prior record of the template and kind has an offset gold, which its base, heading and the robot's camera mount
(`robots.camera_pose`) put back on the map; learning there, where the golds of different scripts are comparable, the
candidates are the dense places of those points (`places.dense_places`, under the support floor `blind.supported`) that
its camera faced (`places.Viewpoints`).
- `ego_mem` (a thing it saw): a place faced at any logged step up to the ask.
- `ego_now` (a thing in view now): a place in view at the ask step alone.
A place is a cluster of the nearest parts other scripts' cameras were asked for (the golds are the point of the thing
nearest each camera, so a fixed thing's golds lie along its near faces; they are one place within the gold's last rung,
2C of the job's clearance C, `_radius`) and the rule is: answer the cluster's point nearest the asking camera, put in
that camera's frame. That is the nearest point of the thing's footprint as far as the golds have shown it, which keeps
the probe as strong as the layout allows. The place kept is judged on that rule (`learned_place`): each record is
answered by the cluster's point nearest the record's own camera with the record's script left out, and the place stays
only if the held-out gain over abstaining, from a bootstrap over the scripts (`blind.gain_low`), is above 0, the round
every learned answer takes; a fixed point would be judged on a different answer from the one returned. A record is
judged only where the rule would answer it: when its own cameras (the record's `cameras`, `_seen`) had the place in
view, as the asking cameras must, so a kind that stands at one spot in half the scripts and at another in the rest keeps
each spot, and a record about the other spot does not count against it.
With no place kept it answers from the pool of other scripts' records of the template at the first level of
`answer_keys.levels("path_log", ...)` that holds enough of them (the phase, the pose, the subgoal), by `learned_answer`
(like every learned answer): a best response is kept only where its held-out gain over abstaining, from a bootstrap
over the scripts left out, has its interval above 0. A constant offset in the camera's frame rarely does, and a pool
whose mean merely beat abstaining on a few scripts answered at a loss (-0.12 on the drone's `ego_now`), so
that is mostly `unknown`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .answer_candidates import UNKNOWN
from .blind import TIE, gain_low, learned_answer, pool_at, supported
from .ego_frame import from_offset, to_offset
from .places import Viewpoints, dense_places, gold_points, match_radius
from .robots import camera_pose
from .score import penalised

if TYPE_CHECKING:
    from .baselines_path import PathAnswerer, PathLog


def _camera_xy(pl: PathLog, base: tuple[float, float], yaw_deg: float) -> tuple[float, float]:
    """The floor point under the camera of a base at `base` heading `yaw_deg` (the mount turned by the heading)."""
    return camera_pose(pl.model, (base[0], base[1], 0.0), math.radians(yaw_deg), 0.0)[0][:2]


def pooled(pl: PathLog, rec: dict) -> dict:
    """The pool's best response alone (`blind.learned_answer`)."""
    return learned_answer(pool_at("path_log", rec, pl.prior))


def _radius(gold: dict) -> float | None:
    """The radius at which a thing's golds are one place (and a place answers one): the gold's own `match_m` where it
    carries one, else twice the need C of its graded credit (`tolerances.need_m`; before the graded credit its ladder's
    last rung, 2C of the job's clearance C); None for neither. Ego golds carry no `match_m`, so the job's clearance
    sets it, the same rule for every job."""
    tol = gold.get("tolerances") or {}
    need = tol.get("need_m")
    return tol.get("match_m", 2 * need if need is not None else (tol.get("ladder") or [None])[-1])


def on_the_map(pl: PathLog, template: str, kind: str) -> list[dict]:
    """The prior's records of `template` questions about `kind`, each offset gold put on the map as a `position` gold
    (its rungs kept, its `match_m` the radius of `_radius`); an `unknown` gold stays. Empty below the support floor: a
    thin prior is one world's accident."""
    found = [p for p in pl.prior if p["template"] == template and p["slots"]["kind"] == kind]
    if not supported(len(found), len({p["script"] for p in found})):
        return []
    out = []
    for p in found:
        g = p["gold"]["gold"]
        if g["type"] == "offset":
            xy = from_offset((g["ahead_m"], g["left_m"]), _camera_xy(pl, p["base"], p["yaw"]), p["yaw"])
            radius = _radius(p["gold"])
            tol = {**(p["gold"].get("tolerances") or {}), **({} if radius is None else {"match_m": radius})}
            p = {**p, "gold": {**p["gold"], "gold": {"type": "position", "xy": list(xy)}, "metric": "xy",
                               "tolerances": tol}}
        out.append(p)
    return out


def _points(records: list[dict]) -> list[tuple[str, tuple[float, float]]]:
    """(script, (x, y)) of every position gold among the records."""
    return [(r["script"], p[:2]) for r in records for p in gold_points([r["gold"]])]


def _nearest(points: list[tuple[float, float]], camera: tuple[float, float]) -> dict:
    """The point nearest `camera` among `points` (the smaller x, then y, on a tie), as a `position` answer; `unknown`
    for none."""
    if not points:
        return dict(UNKNOWN)
    x, y = min(points, key=lambda p: (math.dist(p, camera), p))
    return {"type": "position", "xy": [x, y]}


def _seen(template: str, cameras: list[dict]) -> list[dict]:
    """The camera poses a place must be in view of for the rule of `template` to fire: the last one for `ego_now` (in
    view at the ask), all of them for `ego_mem` (faced at some logged step up to the ask)."""
    return cameras[-1:] if template == "ego_now" else cameras


def _views(pl: PathLog, template: str, records: list[dict]) -> list[Viewpoints]:
    """Where each record's own cameras looked (`_seen`), kept for the system's life (the prior does not change)."""
    return [pl.cached(f"ego_view:{template}:{id(r['cameras'])}", lambda r=r: Viewpoints(_seen(template, r["cameras"]),
                                                                                      pl.cam)) for r in records]


def _gains(pl: PathLog, records: list[dict], views: list[Viewpoints], place: tuple[float, ...],
           held_out: bool) -> dict[str, list[float]]:
    """Each script's gains over abstaining, row by row, under the rule `path_log` answers by: the point nearest a
    record's own camera among the golds within the matching radius of the `place`; with `held_out` the golds of the
    record's own script are left out, so the rule is read on a script it did not learn from (`unknown`, a gain of 0,
    when no gold is left). Only the records the rule would answer are scored: those whose own cameras had the place in
    view (`views`), as the asking camera must have it. A record of the same kind at another spot, where the rule is
    silent, counts for no place that its cameras did not face."""
    radius = match_radius([r["gold"] for r in records])
    near = [(s, p) for s, p in _points(records) if math.dist(p, place[:2]) <= radius + 1e-9]
    out: dict[str, list[float]] = {}
    for r, view in zip(records, views, strict=True):
        if not view.sees(place):
            continue
        camera = _camera_xy(pl, r["base"], r["yaw"])
        answer = _nearest([p for s, p in near if not held_out or s != r["script"]], camera)
        out.setdefault(r["script"], []).append(penalised(answer, r["gold"]) - penalised(UNKNOWN, r["gold"]))
    return out


def learned_place(pl: PathLog, records: list[dict], views: list[Viewpoints],
                  faced: list[tuple[float, ...]]) -> tuple[float, ...] | None:
    """The faced dense place whose rule (`_gains`) gains most over abstaining on the records it would answer, kept only
    where the rule beats abstaining on the scripts it left out beyond what the draw of scripts explains: the lower end
    of the 95% interval of its held-out gain over `unknown`, from a bootstrap over the scripts that have a row it would
    answer (`blind.gain_low`), is above 0, the round every learned answer takes. It is the rule that is answered that is
    tested, not a fixed point. A prior that is `in_pool` keeps its best place as it is, scored without leaving a script
    out. None where no record's cameras faced any place."""
    in_pool = getattr(pl.prior, "in_pool", False)
    best, best_gain, kept = None, -math.inf, {}
    for place in faced:
        gains = _gains(pl, records, views, place, held_out=not in_pool)
        if (total := sum(map(sum, gains.values()))) > best_gain + TIE and gains:
            best, best_gain, kept = place, total, gains
    if best is None:
        return None
    if not in_pool and gain_low([sum(g) for g in kept.values()], [len(g) for g in kept.values()]) <= TIE:
        return None
    return best


def answer(pl: PathLog, question: dict, rec: dict, template: str) -> dict:
    """The dense place of the kind that the cameras faced (`_seen`) and whose rule wins over the prior's records on the
    map (`learned_place`), answered as the point of its cluster nearest the ask's camera, as an offset from that camera;
    else the pool's answer."""
    records = on_the_map(pl, template, question["slots"]["kind"])
    if records:
        cameras = [s["camera"] for s in pl.log.steps_until(max(pl.log.steps))]
        view = Viewpoints(_seen(template, cameras), pl.cam)
        faced = [p for p in dense_places([r["gold"] for r in records]) if view.sees(p)]
        if faced and (place := learned_place(pl, records, _views(pl, template, records), faced)) is not None:
            radius = match_radius([r["gold"] for r in records])
            camera = _camera_xy(pl, rec["base"], rec["yaw"])
            found = _nearest([p for _, p in _points(records) if math.dist(p, place[:2]) <= radius + 1e-9], camera)
            ahead, left = to_offset(found["xy"], camera, rec["yaw"])
            return {"type": "offset", "ahead_m": round(ahead, 4), "left_m": round(left, 4)}
    return pooled(pl, rec)


def ego_now(pl: PathLog, question: dict, rec: dict) -> dict:
    """A place of the kind in view at the ask step."""
    return answer(pl, question, rec, "ego_now")


def ego_mem(pl: PathLog, question: dict, rec: dict) -> dict:
    """A place of the kind faced at some logged step."""
    return answer(pl, question, rec, "ego_mem")


PATH_ANSWERERS: dict[str, PathAnswerer] = {"ego_now": ego_now, "ego_mem": ego_mem}
