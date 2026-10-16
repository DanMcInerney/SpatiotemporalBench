"""Answering questions from a memory table by their template and slots (never their wording): one answerer a family.

The same code answers for every table-based baseline; they differ in how the store was fed (identity tracks or
association, one frame or the whole stream) and in `ctx`. An answer the memory cannot support is `unknown`; a template
no answerer is registered for raises, naming it, rather than answering `unknown` silently (map SG-H22).

`ctx` is what an ask adds to the store: `now`, the step the ask is about (the log's entry at `now["k"]`); `latest`, the
`k`s of the last observe (the latest frame); `events`, the job's `{wording: EventRule}` or None; `robot`, the
`robots.Robot`; `map`, the case layout's public `MapSpec` (None when the system was built without one: an answerer that
names a room needs it and says so). A question that names a task carries it as `slots["task"]` (the case index, None for
the window's only task) or `slots["tasks"]` (a list); `table_log.Log.task_steps` finds its steps.

The registry. `path_ahead` and `object_where` are built in; every other family is answered by a module of
`ANSWER_MODULES` (`table_<group>`, paired with `questions_text.FAMILY_MODULES`) that defines `ANSWERERS: dict[str,
Answerer]`, found by `answer_registry.collect`: a module not written yet is skipped, a family answered twice raises. A
module imports `table`, `table_log`, `grid`, `blind`, `places`, `answer_*` and neutral modules, never this one.
"""

from __future__ import annotations

import math

from . import answer_registry, grid
from . import events as events_mod
from .answer_candidates import UNKNOWN
from .table import Answerer, Store

ANSWER_MODULES = ("table_self", "table_space", "table_cover", "table_view", "table_changed", "table_things",
                  "table_route", "table_zone", "table_adjacent", "table_change", "table_ego", "table_task",
                  "table_clear", "table_people", "table_heading", "table_approach", "table_hazard", "table_way")


def path_ahead(store: Store, slots: dict, ctx: dict) -> dict:
    """`grid.path_ahead` over the latest frame's cells only, at the ask's delivered pose: the function the gold calls
    on the same depth. An answer about "now" from older frames would be a guess about space that may have changed
    (design §8.1)."""
    pose = ctx["now"]["pose"]
    states = store.states_of(ctx["latest"])
    return grid.path_ahead(states, ctx["robot"], pose["pos"][:2], math.radians(pose["yaw_deg"]), slots["cap"])[0]


def object_where(store: Store, slots: dict, ctx: dict) -> dict:
    """Where the thing was seen last: the xy (4 decimals) of the latest sighting of the tracks that match the
    description. A question that names an event (the slot `event`: an anchored form, as `object_where` asked before
    0.5.0-control) reads only the sightings at or before the step its wording resolves to over the steps of the task
    the question names (`slots["task"]`, the last task when None; `events.resolve` over the event table; a wording that
    never occurs, or a task the stream never had, gives `unknown`). A question with no event ("where did you last
    see"; `object_where`'s form now), or a system with no event table (`current_view`), reads every sighting in the
    store: up to the window's end. `unknown` when nothing matches."""
    until = math.inf
    if ctx["events"] is not None and slots.get("event") is not None:
        if slots["event"] not in ctx["events"]:
            raise ValueError(f"the job's events do not name {slots['event']!r}; they are {sorted(ctx['events'])}")
        until = events_mod.resolve(ctx["events"][slots["event"]], store.task_steps(slots.get("task")))
        if until is None:
            return dict(UNKNOWN)
    seen = [s for t in store.matching(slots["kind"], slots["attributes"]) for s in t.sightings if s[0] <= until]
    if not seen:
        return dict(UNKNOWN)
    last = max(seen, key=lambda s: s[0])
    return {"type": "position", "xy": [round(last[1], 4), round(last[2], 4)]}


BUILT_IN: dict[str, Answerer] = {"path_ahead": path_ahead, "object_where": object_where}


def registry(modules: tuple[str, ...] = ANSWER_MODULES) -> dict[str, Answerer]:
    """The built-in answerers and those of the modules among `modules` that exist, family by family."""
    return {**BUILT_IN, **answer_registry.collect(modules, "ANSWERERS", taken=BUILT_IN)}


ANSWERERS: dict[str, Answerer] = registry()


def answer(store: Store, question: dict, ctx: dict) -> dict:
    """The answer to `question` (its `template` and `slots`) from the store. KeyError naming the template when no
    answerer is registered for it."""
    template = question["template"]
    if template not in ANSWERERS:
        raise KeyError(f"no answerer for the template {template!r}; the answerers are {sorted(ANSWERERS)}")
    return ANSWERERS[template](store, question["slots"], ctx)
