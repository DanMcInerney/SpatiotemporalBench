"""`generate.generate_script` and `generate.light_starts`: a script is a robot's job over the built world, with clutter
and the light furniture's starts drawn from keys of their own. Positions are the map's (`maps/apartment_a.json`)."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from unittest import mock

import pytest

from process_kit import guard_off
from stbench import SEED_KEY, clutter, generate
from stbench import timeline as T
from stbench.dressing_boxes import WALL_PROUD_MAX_M
from stbench.generate import build_world, generate_script, light_starts
from stbench.geometry import Rect, rects_overlap, wrap_angle
from stbench.mapspec import load_map
from stbench.seeding import make_rng
from stbench.validate import door_leaf_rect, validate
from world_kit import world_timeline

# Light furniture at its map place: id -> (x, y, yaw). Chairs face their table; the stool and the basket are free.
MAP_PLACES = {"L01": (1.114, 1.9, 0.0), "L02": (3.0865, 1.9, math.pi), "L03": (3.6, 4.9, math.pi),
              "L04": (7.9585, 3.85, 0.0), "L05": (1.6, 2.52, 0.0), "L06": (7.8, 1.95, 0.0)}
SEEDS = range(1, 11)


def digest(seed: int) -> str:
    return T.digest(generate_script("apartment_a", seed, "vacuum", "vacuum/1", 1))


def test_scripts_are_deterministic():
    """Contract: the same inputs give a byte-identical timeline, in this process and in another with another
    `PYTHONHASHSEED`, and another seed gives another one. Regression caught: an unseeded draw or a set's iteration order
    in the world, the clutter, the light starts or the path, which gives every run another script and every recorded
    digest another value."""
    out = subprocess.run([sys.executable, __file__, "103"], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONHASHSEED": "12345"}).stdout.strip()
    assert digest(103) == digest(103) == out
    assert digest(103) != digest(104)


def test_a_script_is_the_world_then_the_clutter_then_the_light_starts_then_the_job():
    """Contract (plan §4.3): `generate_script` builds the world, places clutter from the key (SEED_KEY, "clutter"),
    draws the light starts, then plans and writes the job's path: in that order, so the path meets what lies on the
    floor. A job of another robot is refused. Regression caught: the path planned before the clutter, a clutter drawn
    from another key (which redraws every world), or a helper's script on the vacuum's job."""
    calls = []
    first_draw = []

    def place(world, m, rng):
        calls.append("clutter")
        first_draw.append(float(rng.random()))

    real_starts = generate.light_starts

    def starts(*args, **kwargs):
        calls.append("light starts")
        return real_starts(*args, **kwargs)

    with mock.patch.object(clutter, "place", place), mock.patch.object(generate, "light_starts", starts):
        tl = generate_script("apartment_a", 3, "vacuum", "vacuum/1", 1)
    assert calls == ["clutter", "light starts"]
    assert first_draw == [float(make_rng("apartment_a", 3, SEED_KEY, "clutter").random())]
    ep = tl.episode
    assert (ep.map, ep.seed, ep.robot, ep.job) == ("apartment_a", 3, "vacuum", "vacuum/1")
    assert tl.ego.tasks[0].plan.kind == "clean_room"
    with pytest.raises(ValueError, match="helper"):
        generate_script("apartment_a", 3, "helper", "vacuum/1", 1)


def starts(seed: int):
    """The map's world at `seed` with its light furniture's starts: the world, the start of each piece as (x, y,
    yaw)."""
    m = load_map("apartment_a")
    w = build_world(m, seed)
    light_starts(w, m, make_rng(m.name, seed, SEED_KEY, "light_start"), frozenset({"dock"}))
    tl = world_timeline(m, w, seed)
    pieces = [e.id for e in tl.entities if e.mobility == "light"]  # in the map's order: L01 to L06
    return w, {mid: (*tl.placement_of(eid, 0.0).pos[:2], tl.placement_of(eid, 0.0).yaw)
               for mid, eid in zip(MAP_PLACES, pieces, strict=True)}


def test_light_furniture_starts_vary():
    """Contract (plan §4.3, design §3.1: nothing perfectly aligned): over seeds 1-10 each piece of light furniture
    starts in another place from its map place in at least 3 seeds, in places that differ from seed to seed (a piece
    hemmed in, as the desk chair is by the things on the rug, keeps its map place when no draw is valid; the stool,
    whose map place is 0.025 m off the kitchen's wall y = 2.75 and which keeps 0.12 m off walls, keeps it in every
    seed); chairs are pulled out 0 to 0.6 m against their facing direction (on that line)
    and turned within 15 degrees, the stool and the basket start 0.3 to 1.0 m from their map places; every start is
    valid under `validate`. Regression caught: chairs at one spot in every script (which a `blind` baseline would
    learn), a chair pushed into its table, or a start that collides with something."""
    m, worlds = load_map("apartment_a"), {seed: starts(seed) for seed in SEEDS}
    for eid, (x0, y0, yaw0) in MAP_PLACES.items():
        moved = []
        for seed in SEEDS:
            x, y, yaw = worlds[seed][1][eid]
            dx, dy = x - x0, y - y0
            if math.hypot(dx, dy) < 1e-4:
                continue  # it kept its map place
            moved.append((round(x, 3), round(y, 3)))
            if eid in ("L01", "L02", "L03", "L04"):  # the chairs
                along, across = -(dx * math.cos(yaw0) + dy * math.sin(yaw0)), -dx * math.sin(yaw0) + dy * math.cos(yaw0)
                assert -1e-4 <= along <= 0.6 + 1e-4 and abs(across) < 1e-4, (eid, along, across)
                assert abs(wrap_angle(yaw - yaw0)) <= math.radians(15) + 1e-4, eid
            else:
                assert 0.3 - 1e-4 <= math.hypot(dx, dy) <= 1.0 + 1e-4, eid
        if eid == "L05":  # the stool: its map place is 0.025 m off the kitchen's wall, so every start is too near one
            assert moved == [], moved
            continue
        assert len(moved) >= 3 and len(set(moved)) >= len(moved) - 1, (eid, moved)
    for seed, (w, _) in worlds.items():
        assert validate(world_timeline(m, w, seed), m) == [], seed


def test_a_start_keeps_clear_of_the_docks_apron_the_floor_and_the_doors():
    """Contract (plan §4.3): a start must be clear of the apron of an anchor (the dock's 1.2 m in front and 0.5 m each
    side), of floor items and clutter, of the keep-out around an interior opening, and in its own room. Hand cases, 0.4
    m squares in the hall: one at (5.3, 5.05) is in the dock's apron (x 4.65 to 5.97, y 4.41 to 5.69) and refused when
    the dock is an anchor, fine when none is; one at (5.4, 2.6) is in the open, 0.78 m from the nearest door (the
    keep-out is half a door's width and 0.3 m: 0.725 m); one at (5.1, 3.9) is 0.3 m from the living room's door D2 at
    (4.6, 3.9) and refused; a floor item put at (5.4, 2.6) refuses the open one. Regression caught: furniture put in a
    robot's way home, or on a toy, or across a door."""
    m, w = load_map("apartment_a"), build_world(load_map("apartment_a"), 1)
    old, none, dock = Rect(5.6, 2.6, 0.4, 0.4, 0.0), frozenset(), frozenset({"dock"})
    apron, open_floor = Rect(5.3, 5.05, 0.4, 0.4, 0.0), Rect(5.4, 2.6, 0.4, 0.4, 0.0)
    in_door = Rect(5.1, 3.9, 0.4, 0.4, 0.0)
    assert not generate.start_valid(w, m, "x", old, apron, dock) and generate.start_valid(w, m, "x", old, apron, none)
    assert generate.start_valid(w, m, "x", old, open_floor, dock)
    assert not generate.start_valid(w, m, "x", old, in_door, dock)
    w.clutter["C1"] = (Rect(5.4, 2.6, 0.1, 0.1, 0.0), 0.0, 0.05)
    assert not generate.start_valid(w, m, "x", old, open_floor, dock)


def test_an_arch_keeps_its_gap_clear_and_does_not_freeze_the_chair_beside_it():
    """Contract (deviation from plan §4.3's `door_keepout` of every opening, which circles an arch 0.9 m round its
    centre): a start keeps 0.3 m clear of an arch's gap, a band 1.2 m wide and 0.7 m across the wall. Hand cases in the
    kitchen, whose arch O1 is at (3.6, 2.8) on the wall y = 2.8: a 0.4 m square at (3.6, 2.4) is in the band (y 2.45 to
    3.15) and refused, one at (3.4, 1.4) is clear (D1's open leaf, at y 1.47, starts at x 3.745); and the dining chair
    L02 (3.0865, 1.9), 0.74 m from the arch's centre and so inside the circle, can be pulled out 0.1 m east.
    Regression caught: a chair that no draw can move, in every script, because a map place lies inside an arch's
    circle."""
    m, w = load_map("apartment_a"), build_world(load_map("apartment_a"), 1)
    old = Rect(3.0, 1.9, 0.4, 0.4, 0.0)
    assert not generate.start_valid(w, m, "x", old, Rect(3.6, 2.4, 0.4, 0.4, 0.0), frozenset())
    assert generate.start_valid(w, m, "x", old, Rect(3.4, 1.4, 0.4, 0.4, 0.0), frozenset())
    chair = [e for e in w.entities if e.mobility == "light"][1]  # L02
    at = w.light_rects[chair.id]
    assert generate.start_valid(w, m, chair.id, at, at._replace(cx=at.cx + 0.1), frozenset(), chair.under_m)


def test_a_piece_with_no_valid_start_keeps_its_map_place():
    """Contract (plan §4.3): after 60 draws without a valid start a piece stays where the map put it. Regression
    caught: a piece moved to an invalid place when the draws run out, or a placement left half-written."""
    m = load_map("apartment_a")
    w = build_world(m, 1)
    before = w.ledger.finish()
    with mock.patch.object(generate, "start_valid", lambda *args: False):
        light_starts(w, m, make_rng("any"), frozenset())
    assert w.ledger.finish() == before


DOCK = frozenset({"dock"})
LEAF_SEEDS = (21, 37, 39, 41)  # seeds whose starts met an open leaf (D3's, by the basket; D4's, by a chair) before R1
WALL_SEEDS = (2, 19, 54, 56)  # seeds whose desk chair L03 started within 0.12 m of a wall before R2


def moved_pieces(seed: int, redraws: int = 0):
    """The light pieces of `apartment_a` at `seed` that stand away from their map places after the starts and after
    `redraws` more draws of each from its map place (a gap's process, one second later each time; the floor guard off),
    as (map, world, piece id, footprint) for the start and every successful re-draw."""
    m, w = load_map("apartment_a"), build_world(load_map("apartment_a"), seed)
    light_starts(w, m, make_rng(m.name, seed, SEED_KEY, "light_start"), DOCK)
    rng, out = make_rng("moved_pieces", seed), []
    out += [(eid, r) for eid, r in w.light_rects.items() if r != generate.map_place(w, m, eid)]
    for t in range(redraws):
        for eid in list(w.light_rects):
            if generate.light_start(w, m, rng, eid, DOCK, 100.0 + t):
                out.append((eid, w.light_rects[eid]))
    return m, out


def test_light_furniture_keeps_clear_of_every_open_door_leaf(monkeypatch):
    """Contract (R-home R1): a piece of light furniture keeps `FURNITURE_MARGIN` (0.05 m) clear of the open leaf of
    every interior door, since doors open and close between tasks and a leaf must not strike a piece; the keep-out
    circle (0.725 m from a door's centre) is not enough, the open leaf's tip being 0.95 m out. Hand case: D1 (4.6, 1.9)
    opens west into the kitchen along y = 1.47 from x = 4.6 to 3.75; a 0.2 m square at (3.8, 1.35) is 0.83 m from the
    door's centre, outside the circle, and on the leaf: refused, and accepted once the leaf rule is taken away; one at
    (3.8, 1.0) is 0.35 m from the leaf and fine. Then, on seeds whose starts met a leaf before (21, 41: the basket
    through D3's; 37: a chair through D4's; 39), no piece the start moves, and none of 5 gap re-draws of every piece
    from its map place (`light_start` after the start, as a gap does), meets an open leaf. Regression caught: a leaf
    that strikes a chair or the laundry basket when the door opens in a gap."""
    guard_off(monkeypatch)
    m, w, kitchen = load_map("apartment_a"), build_world(load_map("apartment_a"), 1), Rect(3.0, 1.0, 0.4, 0.4, 0.0)
    on_leaf = Rect(3.8, 1.35, 0.2, 0.2, 0.0)
    assert math.hypot(4.6 - 3.9, 1.9 - 1.45) > 0.725  # the corner of the square nearest the door: outside the keep-out
    assert not generate.start_valid(w, m, "x", kitchen, on_leaf, frozenset())
    assert generate.start_valid(w, m, "x", kitchen, Rect(3.8, 1.0, 0.2, 0.2, 0.0), frozenset())
    with mock.patch.object(generate, "door_leaf_rect", lambda *args: Rect(-99.0, -99.0, 0.0, 0.0, 0.0)):
        assert generate.start_valid(w, m, "x", kitchen, on_leaf, frozenset())
    leaves = [door_leaf_rect(m, o, True, 0.04) for o in m.toggling_doors()]
    assert len(leaves) == 4
    seen = 0
    for seed in LEAF_SEEDS:
        for eid, rect in moved_pieces(seed, redraws=5)[1]:
            assert not any(rects_overlap(rect, leaf, 0.05 - 1e-6) for leaf in leaves), (seed, eid, rect)
            seen += 1
    assert seen >= 100  # pieces did move: the loop is not vacuous


def test_light_furniture_keeps_off_the_walls_by_the_dressings_margin(monkeypatch):
    """Contract (R-home R2): a piece of light furniture keeps at least `dressing_boxes.WALL_PROUD_MAX_M` (0.12 m) from
    every wall face, where the dressing guard assumes furniture stays (a radiator 9 cm deep stands behind it). Hand
    case: the hall's west wall face is x = 4.65; a 0.4 m square at x 4.73-5.13, y 2.7-3.1 is 0.08 m off it, away from
    every door and from the dock: refused, and accepted once the wall rule is taken away; at x 4.80-5.20, 0.15 m off,
    fine. Then, on seeds whose desk chair started within 0.12 m of a wall before (2, 19, 54, 56), no piece the start
    moves, and none of 5 gap re-draws of every piece from its map place, comes within 0.12 m of a wall. (A piece that
    keeps its map place keeps what the map gave it: the stool's is 0.025 m off the kitchen's wall.) Regression caught:
    a chair against a radiator, at box level, in the dressing guard's blind spot."""
    guard_off(monkeypatch)
    m, w, hall = load_map("apartment_a"), build_world(load_map("apartment_a"), 1), Rect(5.4, 2.9, 0.4, 0.4, 0.0)
    near = Rect(4.93, 2.9, 0.4, 0.4, 0.0)
    assert not generate.start_valid(w, m, "x", hall, near, frozenset())
    assert generate.start_valid(w, m, "x", hall, Rect(5.0, 2.9, 0.4, 0.4, 0.0), frozenset())
    monkeypatch.setattr(generate, "WALL_PROUD_MAX_M", 0.0)
    assert generate.start_valid(w, m, "x", hall, near, frozenset())
    monkeypatch.undo()
    guard_off(monkeypatch)
    assert WALL_PROUD_MAX_M == 0.12
    seen = 0
    for seed in WALL_SEEDS:
        m, moved = moved_pieces(seed, redraws=5)
        for eid, rect in moved:
            assert not any(rects_overlap(rect, wall, WALL_PROUD_MAX_M - 1e-6) for wall in m.wall_pieces()), (seed, eid)
            seen += 1
    assert seen >= 100


if __name__ == "__main__":  # the second process of `test_scripts_are_deterministic`
    print(digest(int(sys.argv[1])))
