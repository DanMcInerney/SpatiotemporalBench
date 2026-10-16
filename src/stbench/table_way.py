"""The table systems' answer to `way_to` (a module of `table_answers.ANSWER_MODULES`; the gold: `questions_way`).

A table system holds what the gold reads: the free-space states its mapper made of every depth image it was sent, the
delivered poses of every step, and the tracks it built of the detections, each with the floor hull and middle of every
recognisable sighting. It calls the one rule the gold calls, `route_field.route`, over the latest cells of every logged
step up to the ask, the poses of the steps (by task), the ask's delivered base and heading, and the middle of the thing
at the latest recognisable sighting of the track the description names (the track with the latest hulled sighting, as
`table_ego.ego_mem` takes it), `slots["reach_m"]` and `slots["room_m"]` as the gold does. It answers the best heading
(`route_field.best_index`), `unknown` when no track has a recognisable sighting or no route exists through floor seen or
driven. So `oracle` and `memory_table` give the gold's heading, and `current_view`, whose store holds the latest frame
alone, has no track of a thing out of that frame and gives `unknown`: the memoryless answer the `memory` scope is held
to. Nothing here reads a truth module or a picture."""

from __future__ import annotations

import math

from . import clear_profile, route_field
from .answer_candidates import UNKNOWN
from .table import Answerer, Store


def way_to(store: Store, slots: dict, ctx: dict) -> dict:
    """The best heading of the outcomes `route_field.route` gives over the store's cells and poses up to the ask, toward
    the thing the description names at its latest recognisable sighting; `unknown` when it names none or no route is
    seen."""
    hulled = [t for t in store.matching(slots["kind"], slots["attributes"]) if t.hulls]
    if not hulled:
        return dict(UNKNOWN)
    track = max(hulled, key=lambda t: max(t.hulls))
    thing = tuple(round(v, 4) for v in track.hulls[max(track.hulls)][0])
    now = ctx["now"]
    way = route_field.route(store.latest_states(now["k"]), route_field.runs_of(store.steps_until(now["k"])),
                            ctx["robot"], now["pose"]["pos"][:2], math.radians(now["pose"]["yaw_deg"]), thing,
                            slots["reach_m"], slots["reach_m"], slots["room_m"])
    if way is None:
        return dict(UNKNOWN)
    return {"type": "heading", "deg_left": float(clear_profile.deg_of(way.best, route_field.STEP_DEG))}


ANSWERERS: dict[str, Answerer] = {"way_to": way_to}
