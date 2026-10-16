"""`path_log`'s answer to `task_things` (a module of `baselines_path.PATH_MODULES`; `questions_task` is the gold).

`path_log` reads no picture, so it never saw a thing. What it holds is its own log, which gives the task's anchor
exactly as the gold builds it (`task_predicate.anchor_of` over the task's logged steps), and the public map, which names
the places a thing could stand on the floor: the surfaces of the map's pieces that lie at floor height (a rug, a mat, a
free bay of a rack, `FLOOR_SURFACE_M` or under). Its claim is the centres of those within the predicate's reach, listed
as the places of the things. The claim is an extra candidate of the best response over the pool of other scripts'
golds (`answer_keys.levels("path_log", ...)`: the template and slots, then coarser), which keeps it only when the rule
beats abstaining on the scripts left out beyond the draw of scripts (`blind.learned_answer`), and the answer with no
pool is `unknown`. The recurring-spots rule is held to the same bar (`spots_held_out`). The claim only
tries surface centres: it names no lane or clutter anchor, so a family whose items stand off the map's surfaces is not
tested by it beyond the pool.

A predicate that names `item_kinds` (0.11.0, the vacuum's list of what its brush could catch) needs no rule here: the
pool is other scripts' golds, which hold only the items of those kinds (`questions_task`), so the recurring-spots rule
and the learned answer already list no other kind, and a kind the predicate makes don't-care (the bath mat, the
clothes pile) is in no gold's items. A claim at a kind the predicate does not name is a false place to the gold, as it
is to a model.
"""

from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING

from .answer_candidates import UNKNOWN
from .blind import TIE, gain_low, held_out_mean, learned_answer, pool_at, supported
from .geometry import Rect
from .mapspec import surface_world_rect
from .rules import MATCH_M
from .score import penalised
from .task_predicate import Anchor, TaskThings, anchor_of, reaches

if TYPE_CHECKING:
    from .baselines_path import PathAnswerer, PathLog

FLOOR_SURFACE_M = 0.05   # a surface this high or lower is at floor level
POINT_M = 0.05           # the side of the footprint a spot is judged by: a place, not a thing


def map_claim(pl: PathLog, slots: dict) -> dict | None:
    """The centres of the map's floor-level surfaces within the predicate's reach of the task's anchor, as a `positions`
    answer (4 decimals, sorted); None when the system has no map or no predicate, the task has no anchor, or no surface
    is within reach."""
    pred, m = pl.task_things, pl.map
    if pred is None or m is None:
        return None
    if (anchor := anchor_of(pred, pl.log.task_steps(slots["task"]), m)) is None:
        return None
    rects = (surface_world_rect(f.center, f.yaw, s) for f in m.furniture for s in f.surfaces if s.z <= FLOOR_SURFACE_M)
    places = sorted((round(r.cx, 4), round(r.cy, 4)) for r in rects if reaches(pred, anchor, r))
    return {"type": "positions", "items": [{"xy": list(p)} for p in places]} if places else None


def spots_claim(pred: TaskThings, anchor: Anchor, records: list[dict]) -> dict:
    """The places the records' golds put items, within the predicate's reach of `anchor`, those within `MATCH_M` of one
    another merged into their mean (greedily, from the lowest x): a `positions` answer (4 decimals, sorted), empty
    when no place is within reach. A record's gold that is no `positions` list holds none."""
    near = sorted(tuple(i["xy"]) for r in records if r["gold"]["gold"]["type"] == "positions"
                  for i in r["gold"]["gold"]["items"] if reaches(pred, anchor, Rect(i["xy"][0], i["xy"][1], POINT_M,
                                                                                    POINT_M, 0.0)))
    merged: list[list[tuple]] = []
    for p in near:
        for group in merged:
            if math.dist(p, group[0]) <= MATCH_M:
                group.append(p)
                break
        else:
            merged.append([p])
    centres = sorted((round(statistics.fmean(x for x, _ in g), 4), round(statistics.fmean(y for _, y in g), 4))
                     for g in merged)
    return {"type": "positions", "items": [{"xy": list(c)} for c in centres]}


def spot_records(pl: PathLog) -> list[dict]:
    """The prior's `task_things` records (other scripts', on this layout and job), or none under the support floor."""
    found = [p for p in pl.prior if p["template"] == "task_things"]
    return found if supported(len(found), len({p["script"] for p in found})) else []


def spots_held_out(pred: TaskThings, records: list[dict]) -> tuple[float, float]:
    """The spots rule on the scripts left out: `(mean, gain_low)`, each record answered from the other scripts' records
    alone with its own task's anchor (`unknown` for a record with none). `mean` is its mean penalised score over the
    records; `gain_low` the lower end of the 95% interval of its gain over abstaining, from a bootstrap over the scripts
    (`blind.gain_low`), the bar `blind.learned_answer` sets every learned answer."""
    gains: dict[str, list[float]] = {}
    scores = []
    for r in records:
        others = [o for o in records if o["script"] != r["script"]]
        answer = UNKNOWN if r.get("anchor") is None else spots_claim(pred, r["anchor"], others)
        scores.append(penalised(answer, r["gold"]))
        gains.setdefault(r["script"], []).append(scores[-1] - penalised(UNKNOWN, r["gold"]))
    return statistics.fmean(scores), gain_low([sum(g) for g in gains.values()], [len(g) for g in gains.values()])


def worth(pool: list[dict], learned: dict, extra: list[dict], records: list[dict]) -> float:
    """The mean a rule must beat to be answered over `learned`, the pool's learned answer: abstaining's over `records`
    when `learned` abstains; else its score over the pool, held out (`blind.held_out_mean`), or in-sample under an
    `in_pool` prior, whose pool keeps its own best response."""
    if learned["type"] == UNKNOWN["type"]:
        return statistics.fmean(penalised(UNKNOWN, r["gold"]) for r in records)
    if getattr(pool, "in_pool", False):
        return statistics.fmean(penalised(learned, p["gold"]) for p in pool)
    return held_out_mean(pool, extra)


def task_things(pl: PathLog, question: dict, rec: dict) -> dict:
    """The better of the learned answer over the pool of the ask's key, with the map's claim as an extra candidate
    (`unknown` with no pool), and the recurring-spots rule, which stands when its gain over abstaining on the scripts
    left out is sure (an interval over scripts above 0, as `blind.learned_answer` asks of the learned answer) and its
    mean there beats the learned answer's (`worth`). A pool of an `in_pool` prior is asked the second only."""
    claim = map_claim(pl, question["slots"])
    extra = [] if claim is None else [claim]
    pool = pool_at("path_log", rec, pl.prior)
    learned = learned_answer(pool, extra)
    pred, m = pl.task_things, pl.map
    records = spot_records(pl)
    if pred is None or m is None or not records:
        return learned
    if (anchor := anchor_of(pred, pl.log.task_steps(question["slots"]["task"]), m)) is None:
        return learned
    mean, low = pl.cached("task_things_spots", lambda: spots_held_out(pred, records))
    sure = getattr(pool, "in_pool", False) or low > TIE
    return spots_claim(pred, anchor, records) if sure and mean > worth(pool, learned, extra, records) + TIE else learned


PATH_ANSWERERS: dict[str, PathAnswerer] = {"task_things": task_things}
