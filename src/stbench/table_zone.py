"""The table systems' answers to `zone_clear`, `clearance_then` and `slots_empty` (a module of
`table_answers.ANSWER_MODULES`; `questions_zone` is the gold).

A zone is a place of the public map (`zones`: a piece's footprint, an interior opening's way or the window of the
surface the base faces; inset), found from the map and the robot's pose, and its state is `zones.zone_state` over the
fine cells the mapper observed with the zone's own ground and band (`Store.zone_cells`, only the cells inside the job's
zones). A table system holds all of it, and calls the functions the gold calls (`zones.zone_of`, `zone_state`,
`coverage`, `centroid`), so `oracle` and `memory_table` give the gold. The rule of a question is found by its words
among the rules the mapper was given (`Mapper.zone_rules` and `slot_rules`, the job's `zones` and `slot_zones`:
`slots["zone"]` is a rule's `words`, `slots["places"]` a slot rule's `piece_words`); a system built without rules
answers `unknown`, and one given rules without the map raises. A zone is chosen from the delivered base and its heading
(`pose.yaw_deg`), which only a `zones.SPOT` zone reads.
- `zone_clear`: the zone at the ask's delivered pose; the state over the zone cells of the steps just observed
  (`ctx["latest"]`) and nothing else, as the gold reads its look window: it asks about right now, so a zone those steps
  did not cover is `unknown` (a blocker among them is `no`, 80% of the zone's cells seen free is `yes`). `clear` is yes,
  `blocked` no, else `unknown`.
- `clearance_then`: the step the event's wording resolves to over the steps of the task the question names
  (`events.resolve`), the zone at that step's pose, the state over the steps of the window that held it
  (`Log.window_of`), up to and including it. A zone with a cell in the last observed steps is answered anyway (the gold
  does not ask it). With no event table (`current_view`) the moment is the last step the store holds of the task named:
  a memoryless system answers about where it stands, and one that holds no message of the task named (`task_steps` is
  empty) answers `unknown`.
- `slots_empty`: each zone of the rule is judged over every logged step; the centroids of the passed zones (a cell of it
  observed) that are clear and not observed in the last observed steps.
`current_view`'s store holds the last frame alone, so it gets `zone_clear` as `memory_table` does (the frame is all
either reads) and the end families from the end frame only.

A window here is what one `observe` call carried. The gold's is the compiler's look window, so a caller that observes
one in a call (a hand case) gets the gold; the harness sends one frame a call, so under it a window is one frame and
these three families, which no job has asked since 0.6.0, no longer reproduce their golds.
"""

from __future__ import annotations

import math

from . import events as events_mod
from . import zones
from .answer_candidates import UNKNOWN
from .mapspec import MapSpec
from .table import Answerer, Store, ZoneBand
from .zones import Zone, ZoneRule

YES_NO = {"clear": {"type": "yesno", "value": True}, "blocked": {"type": "yesno", "value": False}}


def rule_of(store: Store, words: str, *, plural: bool = False) -> ZoneRule | None:
    """The rule the question names: of the mapper's `zones` and `slot_zones` the one whose `words` equal `words` (words
    are unique within a job, `questions_zone.check_rules`), or, with `plural`, of the slot rules the one whose
    `piece_words` do. None when the mapper holds none: a system built without the job's rules."""
    mapper = store.mapper
    if mapper is None:
        return None
    rules = mapper.slot_rules if plural else (*mapper.zone_rules, *mapper.slot_rules)
    return next((r for r in rules if (r.piece_words if plural else r.words) == words), None)


def _map(ctx: dict, family: str) -> MapSpec:
    if ctx["map"] is None:
        raise ValueError(f"{family} names a zone of the map: the table needs the case layout's public map (the option "
                         f"'map')")
    return ctx["map"]


def _answer(states: dict[tuple[int, int], tuple[str, int]], zone: Zone) -> dict:
    """yes when `zones.zone_state` says clear, no when blocked, else `unknown`."""
    return dict(YES_NO.get(zones.zone_state(states, zone)[0], UNKNOWN))


def _zone_at(ctx: dict, family: str, rule: ZoneRule, step: dict) -> Zone | None:
    """The zone of the rule for the robot at the delivered pose of `step`: the gold's `zones.zone_of` with the base and
    the heading (`yaw_deg`) the stream delivered."""
    pose = step["pose"]
    return zones.zone_of(_map(ctx, family), rule, pose["pos"][:2], math.radians(pose["yaw_deg"]))


def zone_clear(store: Store, slots: dict, ctx: dict) -> dict:
    """Is the zone at the base clear right now: the state over the steps just observed, and no earlier step."""
    if (rule := rule_of(store, slots["zone"])) is None:
        return dict(UNKNOWN)
    if (zone := _zone_at(ctx, "zone_clear", rule, ctx["now"])) is None:
        return dict(UNKNOWN)
    return _answer(store.zone_states(ctx["latest"], zone), zone)


def clearance_then(store: Store, slots: dict, ctx: dict) -> dict:
    """Was the zone at the named moment clear: the state over the steps of the moment's window up to the moment."""
    if (rule := rule_of(store, slots["zone"])) is None:
        return dict(UNKNOWN)
    steps = store.task_steps(slots.get("task"))
    if ctx["events"] is None:
        k = steps[-1]["k"] if steps else None
    elif slots["event"] not in ctx["events"]:
        raise ValueError(f"the job's events do not name {slots['event']!r}; they are {sorted(ctx['events'])}")
    else:
        k = events_mod.resolve(ctx["events"][slots["event"]], steps)
    if k is None:
        return dict(UNKNOWN)
    if (zone := _zone_at(ctx, "clearance_then", rule, store.steps[k])) is None:
        return dict(UNKNOWN)
    return _answer(store.zone_states(store.window_of(k), zone, until=k), zone)


def slots_empty(store: Store, slots: dict, ctx: dict) -> dict:
    """The centroids of the zones of the kind that were passed, are clear as last seen and are not in the last
    observed steps."""
    if (rule := rule_of(store, slots["places"], plural=True)) is None:
        return dict(UNKNOWN)
    bands: dict[ZoneBand, tuple[dict, dict]] = {}   # a band's states over every step and the last observed ones, once
    items = []
    for zone in zones.zones_of(_map(ctx, "slots_empty"), rule):
        if (band := (zone.base_z, zone.band)) not in bands:
            bands[band] = (store.zone_states(store.steps, zone), store.zone_states(ctx["latest"], zone))
        window, last = bands[band]
        if (zones.coverage(window, zone) > 0 and zones.coverage(last, zone) == 0
                and zones.zone_state(window, zone)[0] == "clear"):
            items.append({"xy": list(zones.centroid(zone))})
    return {"type": "positions", "items": items}


ANSWERERS: dict[str, Answerer] = {"zone_clear": zone_clear, "clearance_then": clearance_then,
                                  "slots_empty": slots_empty}
