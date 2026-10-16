"""The built worlds that several world tests share: the initial world of a (map, seed), the same world after one gap of
the world processes, and the queries over them. A module of its own so that no test file imports another's helpers
(`test_generate.py` and `test_dressing_zones.py` import these)."""

from __future__ import annotations

import json
from collections import defaultdict
from functools import cache
from importlib import resources

import numpy as np

from stbench import timeline as T
from stbench.changes import add_or_remove, move_carried, shift_light_furniture
from stbench.generate import build_world, to_timeline
from stbench.mapspec import load_map
from stbench.seeding import make_rng
from stbench.timeline import Ego, EpisodeInfo

GAP = (300.0, 7200.0)  # an explicit gap, seconds: the world processes act strictly inside it, 30 s from its ends


def info(m, seed: int) -> EpisodeInfo:
    return EpisodeInfo(m.name, seed, "vacuum", "vacuum/1", "2026-03-02T08:00:00")


def world_timeline(m, w, seed: int) -> T.Timeline:
    """The world as a timeline with no ego (the world processes and the item draws need none)."""
    return to_timeline(m, w, Ego([], [], []), info(m, seed))


@cache
def initial_world(map_name: str, seed: int) -> T.Timeline:
    """The map's initial world, cached per worker process (timelines are never mutated)."""
    m = load_map(map_name)
    return world_timeline(m, build_world(m, seed), seed)


@cache
def after_gap(map_name: str, seed: int) -> T.Timeline:
    """The initial world after one gap of `GAP` in which light furniture shifts, carried items move and a delivery
    arrives or an item leaves (the processes of `changes`, drawn from their own key), cached per worker process."""
    m = load_map(map_name)
    w, rng, (lo, hi) = build_world(m, seed), make_rng(map_name, seed, "gap test"), GAP
    shift_light_furniture(w, rng, lo, hi, np.empty((0, 2)))
    occupied = w.occupied()
    touched = move_carried(w, rng, lo, hi, occupied)
    add_or_remove(w, rng, lo, hi, touched, occupied)
    return world_timeline(m, w, seed)


def initial_carried(tl):
    return [e for e in tl.entities if e.mobility == "carried" and tl.placement_of(e.id, 0.0) is not None]


def changes(tl):
    """Every change after t=0 as (kind, entity, t, before, after): move and shift (a new placement after an old one),
    add, remove."""
    by = defaultdict(list)
    for p in tl.placements:
        by[p.entity].append(p)
    out = []
    for eid, ps in by.items():
        ps.sort(key=lambda p: p.t0)
        mob = tl.entity(eid).mobility
        for prev, nxt in zip(ps, ps[1:], strict=False):
            out.append(("shift" if mob == "light" else "move", eid, nxt.t0, prev, nxt))
        if ps[0].t0 > 0:
            out.append(("add", eid, ps[0].t0, None, ps[0]))
        if ps[-1].t1 is not None:
            out.append(("remove", eid, ps[-1].t1, ps[-1], None))
    return out


def manifest_models():
    """kind -> asset id of the packaged manifest's one non-alternate model, read from the raw JSON."""
    raw = json.loads(resources.files("stbench.assets").joinpath("manifest.json").read_text(encoding="utf-8"))
    return {a["kind"]: a["id"] for a in raw["assets"] if a["role"] == "model" and not a.get("alternate_for")}


def manifest_model_of(kind: str, colour: str | None):
    """The packaged manifest's non-alternate model of a kind in a colour, read from the raw JSON: the only one, or of
    several the one that lists the colour in `for_colours`, else the one that lists none."""
    raw = json.loads(resources.files("stbench.assets").joinpath("manifest.json").read_text(encoding="utf-8"))
    found = [a for a in raw["assets"] if a["role"] == "model" and a["kind"] == kind and not a.get("alternate_for")]
    if len(found) > 1:
        found = ([a for a in found if colour in a.get("for_colours", ())]
                 or [a for a in found if not a.get("for_colours")])
    (model,) = found
    return model["id"]
