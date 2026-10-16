"""The prior keys (plan §4.6, recommendations §2.3 item 1): the quantisers, the pose key, the rooms a brief names and
the levels of each prior system. Expected keys are written out by hand in each test."""

from __future__ import annotations

import json

import pytest

from stbench import answer_keys
from stbench.answer_keys import levels, pose_key, quantise, rooms_named

OBJECT_WHERE = {"template": "object_where", "when": "end",
                "slots": {"kind": "mug", "attributes": {"color": "red"}, "event": "left"}, "rooms": ("bedroom",)}
ALL = '{"attributes": {"color": "red"}, "event": "left", "kind": "mug"}'
NO_ATTRIBUTES = '{"event": "left", "kind": "mug"}'
KIND_ONLY = '{"kind": "mug"}'


def number(v):
    return {"type": "number", "value": v}


def position(*xy):
    return {"type": "position", **({"xyz": list(xy)} if len(xy) == 3 else {"xy": list(xy)})}


def test_quantisers_by_hand():
    """Contract (plan §4.6): each type has one key, fixed before data. A number is its bin of 0.25, 0.5, 1 and 2 (an
    edge opens the bin above it: 0.25 is `0.25-0.5`; a value within 1e-9 of an edge is on it, so ten steps of 0.1 are
    1 and not 0.9999999999999999), or `cap` when the cap is given and the value is within 1e-9 of it; a position its
    0.5 m cell `i,j` (negative cells by floor, the height ignored); cells, positions and set a count bucket of their
    items (cells distinct), a count its own bucket (0, 1-2, 3-5, 6-10, 11+); yes/no `yes` or `no`; a time its
    `HH:MM`; unknown `unknown`; a route, a malformed answer or a failed one cannot be keyed.

    Regression caught: priors whose keys never repeat (an answer keyed by its raw float or its seconds), a key that
    flips with float noise at a bin edge, or a route silently keyed as something else."""
    bins = [(0.0, "<0.25"), (0.1, "<0.25"), (-1.0, "<0.25"), (0.25, "0.25-0.5"), (0.4999, "0.25-0.5"), (0.5, "0.5-1"),
            (0.99, "0.5-1"), (1.0, "1-2"), (sum([0.1] * 10), "1-2"), (1.99, "1-2"), (2.0, ">=2"), (7.5, ">=2")]
    for value, key in bins:
        assert quantise(number(value)) == key, value
    assert quantise(number(3.0), cap=3.0) == "cap" and quantise(number(2.9999999999), cap=3.0) == "cap"
    assert quantise(number(2.5), cap=3.0) == ">=2" and quantise(number(3.0)) == ">=2"
    assert quantise(position(0.49, 0.5)) == "0,1" and quantise(position(-0.1, -0.6)) == "-1,-2"
    assert quantise(position(1.2, 0.3, 9.0)) == "2,0"
    for n, key in ((0, "0"), (1, "1-2"), (2, "1-2"), (3, "3-5"), (5, "3-5"), (6, "6-10"), (10, "6-10"), (11, "11+"),
                   (500, "11+")):
        assert quantise({"type": "count", "value": n}) == key, n
    cells = {"type": "cells", "items": [[0, 0], [0, 0], [1, 0], [2, 0]]}  # three distinct
    assert quantise(cells) == "3-5" and quantise({"type": "cells", "items": []}) == "0"
    assert quantise({"type": "positions", "items": [{"xy": [i, 0]} for i in range(4)]}) == "3-5"
    assert quantise({"type": "set", "items": [{"xy": [i, 0], "change": "added"} for i in range(12)]}) == "11+"
    assert quantise({"type": "yesno", "value": True}) == "yes" and quantise({"type": "yesno", "value": False}) == "no"
    assert quantise({"type": "time", "clock": "2026-05-14T08:37:42"}) == "08:37"
    assert quantise({"type": "unknown"}) == "unknown"
    for unkeyable in ({"type": "route", "points": [[0, 0]]}, {"type": "number"}, {"type": "failed", "reason": "x"}):
        with pytest.raises(ValueError, match="cannot key"):
            quantise(unkeyable)


def test_pose_key_by_hand():
    """Contract (plan §4.6): the pose key is the base's 0.5 m cell and the 45 degree heading bin
    h = floor(((yaw + 22.5) mod 360) / 45), the bins centred on the compass points. (0.49, 0.5) at 22.4 degrees is
    `0,1,0` and at 22.5 `0,1,1`; the bin wraps (-22.5 and 337.5 both `0`, 359 `0`), 90 is 2, 180 and -180 are 4.
    Regression caught: a heading bin that starts at 0 instead of straddling it, or one that does not wrap, so two
    sides key one pose differently."""
    assert pose_key((0.49, 0.5), 22.4) == "0,1,0" and pose_key((0.49, 0.5), 22.5) == "0,1,1"
    assert [pose_key((0, 0), yaw)[-1] for yaw in (-22.5, 337.5, 359.0, 90.0, 180.0, -180.0)] == list("000244")
    assert pose_key([-0.1, 1.2], 0.0) == "-1,2,0"


def test_rooms_named_by_hand():
    """Contract (plan §4.6): the region kinds that occur in any brief as whole words, in any case, sorted: one function
    for `path_log` and for the harness's records. `rooms_named(["Its current assignment is to clean the bedroom."],
    ["bedroom", "bathroom", "hall"])` is `("bedroom",)`; `hallway` does not name `hall`; a name of two words is one
    name; kinds named in two briefs come out once, sorted; no brief names none. Regression caught: a record and a lookup
    naming different rooms from one brief, or a substring match crediting `hall` for `hallway`."""
    kinds = ["bedroom", "bathroom", "hall", "living room"]
    assert rooms_named(["Its current assignment is to clean the bedroom."], kinds[:3]) == ("bedroom",)
    assert rooms_named(["Clean the hallway."], kinds) == ()
    assert rooms_named(["Bedroom first.", "Then the living room, then the hall.", "The bedroom again."],
                       kinds) == ("bedroom", "hall", "living room")
    assert rooms_named([], kinds) == ()


def test_levels_by_hand():
    """Contract (plan §4.6): `levels` keys a record from exact to loose, each list ending at `(template,)`, a key's
    slots being `json.dumps(slots, sort_keys=True)` with clocks dropped.
    - `blind`: all slots, then `attributes` dropped, then `event`, then the template: for `object_where` kind,
      attributes and event; kind and event; kind; template. A slot the question lacks is skipped (no attributes: the
      first drop is a no-op), and a question of no slots has the template only after its empty key.
    - `path_log`, during: template, phase, pose key and subgoal (None when the record has none: the level is always
      there, so a lookup sent without a subgoal lines up by depth with records that have one); template, phase and pose
      key; template and phase; template. End: each `blind` level with the rooms appended, then `blind`'s own levels, so
      every `blind` level is among them.
    - `current_view_prior`: template, phase, pose key and the quantised `nm` (the cap read from the slots); template,
      phase and `nm`; template and `nm`; template. An `nm` of None keys as `none`.
    Another system's name is refused.

    Regression caught: a `path_log` coarser than `blind` (so its shortcut gate passes for a pool it never reached), two
    sides keying one record differently (a clock, a list against a tuple), or levels that do not end at the template."""
    blind = levels("blind", OBJECT_WHERE)
    assert blind == [("object_where", ALL), ("object_where", NO_ATTRIBUTES), ("object_where", KIND_ONLY),
                     ("object_where",)]
    lacking = {"template": "object_where", "slots": {"kind": "mug", "event": "left"}}
    assert levels("blind", lacking) == [("object_where", NO_ATTRIBUTES), ("object_where", KIND_ONLY),
                                        ("object_where",)]
    assert levels("blind", {"template": "path_ahead", "slots": {"cap": 3.0}}) == [("path_ahead", '{"cap": 3.0}'),
                                                                                  ("path_ahead",)]
    assert levels("blind", {"template": "t", "slots": {}}) == [("t", "{}"), ("t",)]
    path = levels("path_log", OBJECT_WHERE)
    rooms = ("bedroom",)
    assert path == [("object_where", ALL, rooms), ("object_where", NO_ATTRIBUTES, rooms),
                    ("object_where", KIND_ONLY, rooms), ("object_where", rooms), *blind]
    assert set(blind) <= set(path) and path[-1] == ("object_where",)
    during = {"template": "path_ahead", "when": "during", "slots": {"cap": 3.0}, "phase": "undocking",
              "base": (0.49, 0.5), "yaw": 22.4, "rooms": ()}
    assert levels("path_log", during) == [("path_ahead", "undocking", "0,1,0", None),
                                          ("path_ahead", "undocking", "0,1,0"), ("path_ahead", "undocking"),
                                          ("path_ahead",)]
    prior = {**during, "nm": number(1.4)}
    assert levels("current_view_prior", prior) == [("path_ahead", "undocking", "0,1,0", "1-2"),
                                                   ("path_ahead", "undocking", "1-2"), ("path_ahead", "1-2"),
                                                   ("path_ahead",)]
    assert levels("current_view_prior", {**prior, "nm": number(3.0)})[1] == ("path_ahead", "undocking", "cap")
    assert levels("current_view_prior", {**prior, "nm": {"type": "unknown"}})[2] == ("path_ahead", "unknown")
    assert levels("current_view_prior", {**prior, "nm": None})[2] == ("path_ahead", "none")
    for system in ("oracle", "current_view", "other"):
        with pytest.raises(ValueError, match="levels are defined for"):
            levels(system, prior)


def test_clocks_and_json_round_trips_never_change_a_key():
    """Contract: clock slots (`clock`, `clocks`) are in no key, and a record read back from JSON (lists for tuples) keys
    as the record that was written, for every system. Regression caught: a clock that never repeats across cases making
    every key unique, or a prior record that misses its own lookup after a round trip through a file."""
    clocked = {**OBJECT_WHERE, "slots": {**OBJECT_WHERE["slots"], "clock": "08:30", "clocks": ["08:30", "08:40"]}}
    assert levels("blind", clocked) == levels("blind", OBJECT_WHERE)
    assert levels("path_log", clocked) == levels("path_log", OBJECT_WHERE)
    during = {"template": "path_ahead", "when": "during", "slots": {"cap": 3.0}, "phase": "cruising",
              "base": (1.2, 0.3), "yaw": -90.0, "rooms": ("bedroom",), "nm": number(0.75)}
    listed = {**during, "base": [1.2, 0.3], "rooms": ["bedroom"]}
    for system, rec in (("blind", OBJECT_WHERE), ("path_log", OBJECT_WHERE), ("path_log", during),
                        ("current_view_prior", during)):
        assert levels(system, {**rec, "base": list(rec.get("base", ())), "rooms": list(rec["rooms"])}) == levels(
            system, rec), system
    assert levels("current_view_prior", listed) == levels("current_view_prior", during)


def test_levels_read_only_what_the_record_holds():
    """Contract: `levels` reads only keys of `answer_keys.RECORD_KEYS`, and only the keys its system needs, so the
    lookup side can build a record from what a system sees at an ask (no gold, no siblings) and the writing side holds
    the same keys. Regression caught: a level that reads a field one side cannot build (the gold, a sibling question),
    so the two sides could never agree."""
    class Spy(dict):
        read: set = set()

        def __getitem__(self, key):
            Spy.read.add(key)
            return super().__getitem__(key)

    during = {"template": "t", "when": "during", "slots": {"cap": 3.0}, "phase": "p", "base": (0, 0), "yaw": 0.0,
              "rooms": (), "nm": number(1.0)}
    for system, rec in (("blind", OBJECT_WHERE), ("path_log", OBJECT_WHERE), ("path_log", during),
                        ("current_view_prior", during)):
        Spy.read = set()
        levels(system, Spy(rec))
        assert Spy.read <= set(answer_keys.RECORD_KEYS) and Spy.read, system
    assert answer_keys.RECORD_KEYS == ("script", "template", "slots", "when", "clock", "phase", "base", "yaw",
                                       "subgoal", "rooms", "nm", "siblings", "anchor", "cameras", "own", "gold")
    assert answer_keys.SLOT_DROP_ORDER == ("task", "tasks", "attributes", "event", "events", "kind")


def shoe(color: str) -> dict:
    return {"kind": "shoe", "attributes": {"color": color}}


def test_blind_loosens_nested_descriptions():
    """Contract (the review's M4, `answer_keys._blind`): the loosening names each step in turn and drops it from the
    slots and from every slot whose value is a description (a dict with a `kind`), a description left empty goes, and a
    step that changes nothing adds no level; the task slots go first (`SLOT_DROP_ORDER`). Hand records. An `order`
    record naming a red and a blue shoe: the exact pair, then (attributes dropped inside both) the pair of kinds, then
    (kinds dropped, both descriptions empty, no slot left) the template: three levels, where the exact pair alone used
    to stand between the record and the template. A 3-task `object_where` of the red bag in task 2: M1's exact key
    second (the task index dropped: it says nothing of where a thing is and splits each pool by the window's size), and
    M1's levels after it; a one-task window's `task` of None is dropped the same way. A `changed` record naming `tasks`
    [0, 2] drops them at their step. A clock is in no key. `path_log`'s end levels extend `blind`'s with the rooms, so
    they follow. Regression caught: pair families keyed on exact description pairs, so `blind` and `path_log` never
    reach the kind level (the review's M4) and a prior is empty for a family it exists for."""
    pair = {"template": "order", "when": "end", "rooms": ("hall",),
            "slots": {"first": shoe("red"), "second": shoe("blue")}}
    exact = json.dumps(pair["slots"], sort_keys=True)
    kinds = '{"first": {"kind": "shoe"}, "second": {"kind": "shoe"}}'
    assert levels("blind", pair) == [("order", exact), ("order", kinds), ("order",)]
    where = {"template": "object_where", "when": "end", "rooms": (),
             "slots": {"kind": "bag", "attributes": {"color": "red"}, "event": "left", "task": 2}}
    with_task = '{"attributes": {"color": "red"}, "event": "left", "kind": "bag", "task": 2}'
    m1_key = '{"attributes": {"color": "red"}, "event": "left", "kind": "bag"}'
    assert levels("blind", where) == [("object_where", with_task), ("object_where", m1_key),
                                      ("object_where", '{"event": "left", "kind": "bag"}'),
                                      ("object_where", '{"kind": "bag"}'), ("object_where",)]
    one_task = {**where, "slots": {**where["slots"], "task": None}}
    assert levels("blind", one_task)[1:] == levels("blind", where)[1:]
    changed = {"template": "changed", "when": "end", "slots": {"room": "hall", "tasks": [0, 2]}}
    assert levels("blind", changed) == [("changed", '{"room": "hall", "tasks": [0, 2]}'),
                                        ("changed", '{"room": "hall"}'), ("changed",)]
    clocked = {**pair, "slots": {**pair["slots"], "clock": "2026-03-02T08:00:00"}}
    assert levels("blind", clocked) == levels("blind", pair)
    assert levels("path_log", pair) == [("order", exact, ("hall",)), ("order", kinds, ("hall",)),
                                        ("order", ("hall",)), ("order", exact), ("order", kinds), ("order",)]
    assert levels("path_log", where)[1] == ("object_where", m1_key, ())


def test_blind_loosens_a_pair_of_moments_before_the_kind():
    """Contract (FD-0.5.0-impl §2, `SLOT_DROP_ORDER`): the change families name two moments in `events` (and a task for
    each in `tasks`), and `blind` drops them, in order, before the kind: the task slots, the attributes, then the
    moments, leaving the kind alone before the template. Hand records of `same_place`, a red mug at two moments of tasks
    0 and 1: the exact key; (tasks dropped) kind, attributes and events; (attributes dropped) kind and events; (events
    dropped) kind alone; the template. Another script's red mug at other moments of other tasks shares the kind-only
    level, so a record is found; with the moments kept to the end it would share none above the template. Regression
    caught: a `blind` keyed on a pair of wordings that no other script repeats, which never finds a record and passes
    its shortcut gate for nothing."""
    def same_place(events, tasks):
        return {"template": "same_place", "when": "end", "rooms": (),
                "slots": {"kind": "mug", "attributes": {"color": "red"}, "events": events, "tasks": tasks}}

    moments = '["when you left the dock", "when you finished"]'
    here = same_place(["when you left the dock", "when you finished"], [0, 1])
    other = same_place(["when you started", "when you docked"], [2, 4])
    assert levels("blind", here) == [
        ("same_place", '{"attributes": {"color": "red"}, "events": ' + moments + ', "kind": "mug", "tasks": [0, 1]}'),
        ("same_place", '{"attributes": {"color": "red"}, "events": ' + moments + ', "kind": "mug"}'),
        ("same_place", '{"events": ' + moments + ', "kind": "mug"}'),
        ("same_place", KIND_ONLY), ("same_place",)]
    assert set(levels("blind", here)) & set(levels("blind", other)) == {("same_place", KIND_ONLY), ("same_place",)}
    assert answer_keys.SLOT_DROP_ORDER.index("events") < answer_keys.SLOT_DROP_ORDER.index("kind")
