"""The keys of the prior systems: the one home of the quantisers, the pose key and the prior levels (plan §4.6).

`blind`, `path_log` and `current_view_prior` answer a question from a pool of other scripts' records, and the pool is
found by a key. The harness side (`suite_priors`, which writes the records) and every prior system (which looks them up)
both import this module, so a key can never differ by one rounding between the two: a prior whose keys never match falls
back to the template and passes its shortcut gate for nothing.

A record is what both sides can build for one asked question (`RECORD_KEYS`); `levels` keys it from exact to loose, each
list ending at `(template,)`. This module is neutral baseline-side code: it imports `grid` and `answers`, and no module
that reads truth.

A structure family (`STRUCTURE_TEMPLATES`: the templates whose spec declares `map_reader` as its reference, 0.8.0) asks
about the fixed structure of a place, so a pose key would let a gated system learn the layout from other scripts: the
levels of `current_view_prior` have none on such a template and `path_log`'s are the bare template alone: it answers
such a family from its own log, never from a pool (`pooled`). `map_reader`, the reference that is meant to hold the
layout, keys on the pose.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from itertools import pairwise

from .answers import (
    Cells,
    Count,
    Heading,
    Malformed,
    Number,
    Offset,
    Position,
    Positions,
    SetAnswer,
    Time,
    Unknown,
    YesNo,
    parse_answer,
)
from .grid import cell

# A prior record, as `suite_priors` writes it and the prior systems read it: the asked question's template and slots,
# `when`, the delivered clock, the phase and the base pose (xy, and yaw in degrees as delivered) of its step, the room
# words its briefs named (`rooms_named`), `current_view`'s answer there (`nm`, or None), the other end questions of its
# ask ([{template, slots}]; [] during), the `task_predicate.Anchor` of its task (a `task_things` question's, built from
# the poses the robot was sent; None for any other question), and its gold row. No level reads the clock:
# `blind.learned_answer` reads a time gold against it. `subgoal` is the text the step of the ask carried (None: the job
# states none, or the step was sent without it): only a during question's `path_log` level reads it. `anchor` is read
# by `path_log`'s `task_things` rule alone, never by a level. `cameras` is the camera pose rows (`{pos, rot, ...}`) of
# the steps the asked system had been sent up to the ask, the ask's step last: `path_log`'s ego rule reads which places
# such a record's cameras faced, never a level. `own` is `path_log`'s own-log answer to a question of a template with
# an own-log rule (`clear_path`, `open_heading`; None for any other), built from the poses the robot was sent
# (`baselines_path_own.own_answer`): `baselines_path_own.kept` scores it, never a level.
RECORD_KEYS = ("script", "template", "slots", "when", "clock", "phase", "base", "yaw", "subgoal", "rooms", "nm",
               "siblings", "anchor", "cameras", "own", "gold")
# The templates whose spec declares `map_reader` (`gates.MAP_READER`) as its reference: a baseline may not import the
# specs (they are truth side), so `tests/test_answer_keys_structure.py` ties this set to them.
STRUCTURE_TEMPLATES = frozenset({"clear_path", "open_heading"})
SLOT_DROP_ORDER = ("task", "tasks", "attributes", "event", "events", "kind")   # `blind`'s loosening, in this order
CLOCK_SLOTS = ("clock", "clocks")                    # dropped from every key: clocks never repeat across cases
NUMBER_EDGES = (0.25, 0.5, 1.0, 2.0)
COUNT_BUCKETS = (0, 2, 5, 10)                        # 0, 1-2, 3-5, 6-10, 11+
HEADING_BINS = 8                                     # 45 degrees each, centred on the compass points
EPS = 1e-9                                           # a value this close to an edge or the cap is on it
NO_ANSWER = "none"                                   # `nm` None: current_view gave no answer there


def _number_bin(v: float, cap: float | None) -> str:
    if cap is not None and abs(v - cap) <= EPS:
        return "cap"
    below = [e for e in NUMBER_EDGES if v + EPS < e]
    if len(below) == len(NUMBER_EDGES):
        return f"<{NUMBER_EDGES[0]:g}"
    if not below:
        return f">={NUMBER_EDGES[-1]:g}"
    upper = below[0]
    return f"{NUMBER_EDGES[NUMBER_EDGES.index(upper) - 1]:g}-{upper:g}"


def _count_bucket(n: int) -> str:
    if n <= COUNT_BUCKETS[0]:
        return str(COUNT_BUCKETS[0])
    for lo, hi in pairwise(COUNT_BUCKETS):
        if n <= hi:
            return f"{lo + 1}-{hi}"
    return f"{COUNT_BUCKETS[-1] + 1}+"


def quantise(answer: dict, cap: float | None = None) -> str:
    """The key of an answer (a JSON object), fixed before data:
    - number: its bin of `NUMBER_EDGES` (`<0.25`, `0.25-0.5`, `0.5-1`, `1-2`, `>=2`), or `cap` when `cap` is given and
      the value is within 1e-9 of it;
    - position: its answer-grid cell, `i,j` (`grid.cell`; the height is ignored);
    - offset: its 0.5 m cell in the camera's frame, `i,j` = floor(ahead / 0.5), floor(left / 0.5) (`grid.cell`);
    - cells, positions, set: a count bucket of its items (`0`, `1-2`, `3-5`, `6-10`, `11+`); count: its value's bucket;
    - heading: its 45 degree bin, `h0` to `h7`, centred on the robot's heading and counted to its left (`h0` ahead, `h2`
      to its left, `h4` behind, `h6` to its right);
    - yesno: `yes` or `no`; time: its `HH:MM`; unknown: `unknown`.
    A route, a malformed answer or anything else raises ValueError."""
    match parse_answer(answer):
        case Number(v):
            return _number_bin(v, cap)
        case Position() as p:
            return "{},{}".format(*cell(*p.xy))
        case Offset(ahead_m, left_m):
            return "{},{}".format(*cell(ahead_m, left_m))
        case Cells(items) | SetAnswer(items) | Positions(items):
            return _count_bucket(len(items))
        case Heading(deg_left):
            return f"h{round(deg_left / (360 / HEADING_BINS)) % HEADING_BINS}"
        case Count(v):
            return _count_bucket(v)
        case YesNo(v):
            return "yes" if v else "no"
        case Time(clock):
            return clock.strftime("%H:%M")
        case Unknown():
            return "unknown"
        case Malformed(reason):
            raise ValueError(f"cannot key a malformed answer: {reason}")
    raise ValueError(f"cannot key an answer of type {answer.get('type')!r}")


def pose_key(base_xy: tuple[float, float], yaw_deg: float) -> str:
    """`i,j,h`: the 0.5 m answer-grid cell of the base (`grid.cell`) and the 45 degree heading bin of its yaw, h =
    floor(((yaw + 22.5) mod 360) / 45), so h 0 is east and the bins run counter-clockwise."""
    i, j = cell(*base_xy)
    return f"{i},{j},{math.floor(((yaw_deg + 22.5) % 360) / (360 / HEADING_BINS))}"


def rooms_named(briefs: list[str], kinds: Iterable[str]) -> tuple[str, ...]:
    """The region kinds that occur in any brief as whole words (any case), sorted: the one function both `path_log` and
    `suite_priors` call, so a record and a lookup name the same rooms."""
    text = "\n".join(briefs)
    named = {k for k in kinds if re.search(rf"(?<!\w){re.escape(k)}(?!\w)", text, re.IGNORECASE)}
    return tuple(sorted(named))


def _key(slots: dict) -> str:
    return json.dumps(slots, sort_keys=True)


def _without(slots: dict, name: str) -> dict:
    """`slots` without `name`, and without `name` in every slot whose value is a description (a dict with a `kind`: the
    things a pair family names); a description left empty is dropped."""
    out = {}
    for key, value in slots.items():
        if key == name:
            continue
        if isinstance(value, dict) and "kind" in value:
            value = {k: v for k, v in value.items() if k != name}
            if not value:
                continue
        out[key] = value
    return out


def _blind(rec: dict) -> list[tuple]:
    """The template with all its slots (`CLOCK_SLOTS` dropped), then with each name of `SLOT_DROP_ORDER` dropped in turn
    from the slots and from the descriptions nested in them (a step that changes nothing adds no level; one that leaves
    no slot adds none either, the template alone ends the list), then the template. A task index goes first: it says
    nothing of where a thing is and splits every pool by the window's size; a pair of descriptions recurs in no other
    script, and the pair of their kinds is the level that does. The moments go before the kind (`event`, then `events`
    of a pair of moments): a pair of wordings recurs rarely, and the kind alone is the level that finds a record."""
    template = rec["template"]
    slots = {k: v for k, v in rec["slots"].items() if k not in CLOCK_SLOTS}
    out = [(template, _key(slots))]
    for name in SLOT_DROP_ORDER:
        if (looser := _without(slots, name)) != slots:
            slots = looser
            if slots:
                out.append((template, _key(slots)))
    return [*out, (template,)]


def _posed(rec: dict) -> list[tuple]:
    """`path_log`'s levels with the pose key, which `map_reader` keys on whatever the template."""
    template = rec["template"]
    if rec["when"] == "during":
        pose = pose_key(rec["base"], rec["yaw"])
        # The subgoal level is always there (its text None for a job that states none, or a step sent without it), so
        # the levels of a record and of a lookup that differ in the subgoal alone still line up by depth (`pool_at`).
        return [(template, rec["phase"], pose, rec.get("subgoal")), (template, rec["phase"], pose),
                (template, rec["phase"]), (template,)]
    blind, rooms = _blind(rec), tuple(rec["rooms"])
    return [*(level + (rooms,) for level in blind), *blind]


def _path_log(rec: dict) -> list[tuple]:
    template = rec["template"]
    if rec["when"] == "during" and template in STRUCTURE_TEMPLATES:   # no pose, and no pool: its own log (`pooled`)
        return [(template,)]
    return _posed(rec)


def _current_view_prior(rec: dict) -> list[tuple]:
    template, phase = rec["template"], rec["phase"]
    nm = rec["nm"]
    q = NO_ANSWER if nm is None else quantise(nm, rec["slots"].get("cap"))
    if template in STRUCTURE_TEMPLATES:   # no pose: it would learn the layout
        return [(template, phase, q), (template, q), (template,)]
    return [(template, phase, pose_key(rec["base"], rec["yaw"]), q), (template, phase, q), (template, q), (template,)]


_LEVELS = {"blind": _blind, "path_log": _path_log, "current_view_prior": _current_view_prior,
           "map_reader": _posed}


def pooled(system: str, template: str) -> bool:
    """Whether `system` answers `template` from a pool of other scripts' records, so that its levels have something to
    find. `path_log` does not on a structure family: it answers from the robot's own log alone, and a phase or pose
    pool would be a shortcut of the layout inside a gated system (`baselines_path_clear`), so a reach rate of 0 there is
    the design."""
    return not (system == "path_log" and template in STRUCTURE_TEMPLATES)


def levels(system: str, rec: dict) -> list[tuple]:
    """The keys of a record, exact to loose, each list ending at `(template,)`. A key's slots are `json.dumps(slots,
    sort_keys=True)` after dropping `CLOCK_SLOTS`.
    - `blind`: the template and all slots, then each name of `SLOT_DROP_ORDER` dropped in turn from the slots and from
      the descriptions nested in them (`object_where`: kind and attributes; kind; `same_place`: kind, attributes, events
      and tasks; kind, attributes and events; kind and events; kind; `order`: the exact pair, the pair of kinds), then
      the template.
    - `path_log`: a during question keys on template, phase, pose key and subgoal (the text of its ask step; None when
      the step has none), then template, phase and pose key, then template and phase, then the template; an
      end question on each of `blind`'s levels with its rooms appended, then `blind`'s levels, so `blind`'s levels are
      all among `path_log`'s and it is never coarser. A job whose subgoal told a coin's outcome would let the first
      level beat the floor, and `path_log`'s shortcut gate would fail it.
    - `current_view_prior`: template, phase, pose key and `quantise(nm, cap)`; template, phase and the quantised answer;
      template and the quantised answer; the template.
    - On a template of `STRUCTURE_TEMPLATES` neither has a pose key: `path_log` keys a during question on the template
      alone (it pools nothing there: `pooled`); `current_view_prior` on template, phase and the quantised answer, then
      template and the quantised answer, then the template.
    - `map_reader`: `path_log`'s levels with the pose key, whatever the template.
    Another system name raises ValueError."""
    if system not in _LEVELS:
        raise ValueError(f"levels are defined for {list(_LEVELS)}, not {system!r}")
    return _LEVELS[system](rec)
