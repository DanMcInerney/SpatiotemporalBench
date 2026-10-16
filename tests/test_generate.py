"""Generator contracts (V0 §4): the world `build_world` makes (determinism, lookalikes, placement priors, doors,
under-furniture) and the world processes of `changes` over an explicit gap. The ego, the tours and the tiers are gone:
a script's path is its job's (`job_*.py`)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict
from importlib import resources

import pytest

from conftest import apartment, script_on_map
from stbench import timeline as T
from stbench.catalog import KINDS
from stbench.generate import build_world
from stbench.geometry import distance
from stbench.mapspec import parse_map
from stbench.validate import validate
from validate_kit import generic_rules_only  # noqa: F401  (a fixture)
from world_kit import GAP, after_gap, changes, info, initial_carried, initial_world, manifest_model_of, world_timeline

SEEDS = range(1, 9)
LIVING_ROOM_FLOOR = [(2.4, 3.4, 0.0, 0.0), (2.8, 3.4, 0.0, 0.0), (3.2, 3.4, 0.0, 0.0)]  # open floor, x 2.4-3.2


def test_same_inputs_give_identical_json_bytes():
    """Determinism: building the world twice from the same (map, seed) yields byte-identical JSON, so the timeline hash
    is stable. A regression (unseeded randomness, set iteration order) changes the bytes."""
    m = apartment()
    a, b = world_timeline(m, build_world(m, 3), 3), world_timeline(m, build_world(m, 3), 3)
    assert T.dumps(a) == T.dumps(b)
    assert T.digest(a) == T.digest(b)


@pytest.mark.parametrize("which", ["initial_world", "after_gap"])
def test_hash_is_stable_across_processes(which):
    """Contract: the seeding uses a stable hash, not Python's salted hash(), and nothing iterates a set into the output:
    a subprocess with another PYTHONHASHSEED agrees, for the initial world and for the world after a gap. Regression
    caught: seeding through `hash()` or iterating a set into the world, which gives every run another world and every
    recorded world hash another value."""
    out = subprocess.run([sys.executable, __file__, which, "apartment_a", "3"], capture_output=True, text=True,
                         check=True, env={**os.environ, "PYTHONHASHSEED": "12345"}).stdout.strip()
    assert out == T.digest(globals()[which]("apartment_a", 3))


def test_different_seeds_give_different_worlds():
    """Contract: two seeds differ in their world. Regression caught: a generator that ignores its seed, so that every
    script of a map starts from one world."""
    assert initial_world("apartment_a", 1).world_hash != initial_world("apartment_a", 2).world_hash


@pytest.mark.usefixtures("generic_rules_only")
def test_generated_worlds_are_valid(apt):
    """Every built world passes validate(): no overlapping placements, items on real supports, no collisions,
    everything inside the map, with a hand-made ego on the living room's open floor and with no ego, before and after a
    gap's changes. This is the generator's contract with V0 §3. Regression caught: a generator change that puts an item
    on nothing, inside a piece of furniture or through a wall, which `validate` (and so every script built on the world)
    would then refuse."""
    for seed in SEEDS:
        assert validate(script_on_map("apartment_a", seed, "vacuum", LIVING_ROOM_FLOOR), apt) == [], seed
        assert validate(after_gap("apartment_a", seed), apt) == [], seed


def spec_groups(tl):
    """The initial carried entities grouped by (kind, attributes)."""
    groups = defaultdict(list)
    for e in initial_carried(tl):
        groups[(e.kind, tuple(sorted(e.attributes.items())))].append(e)
    return groups


def test_item_count_and_mandatory_lookalikes():
    """Contract: 30-40 carried items at the start, with at least two groups that differ in exactly one attribute (a
    memory system must tell lookalikes apart). The identical sets have their own test. Regression caught: a sampler
    that stops drawing lookalikes (every item told apart by its kind alone) or draws a world of fewer than 30 items."""
    for seed in SEEDS:
        items = initial_carried(initial_world("apartment_a", seed))
        assert 30 <= len(items) <= 40
        keys = list(spec_groups(initial_world("apartment_a", seed)))
        edges = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]
                 if a[0] == b[0] and sum(x != y for x, y in zip(a[1], b[1], strict=True)) == 1
                 and [k for k, _ in a[1]] == [k for k, _ in b[1]]]
        parent = {k: k for k in keys}

        def find(x, parent=parent):
            while parent[x] != x:
                x = parent[x]
            return x

        for a, b in edges:
            parent[find(a)] = find(b)
        groups = defaultdict(set)
        for a, b in edges:
            groups[find(a)] |= {a, b}
        assert len(groups) >= 2, seed


def test_identical_sets_are_two_or_three_groups_of_two_to_four_of_the_set_kinds():
    """Contract (M2 plan 4.1, the identity major): a world starts with 2-3 identical sets, each of 2-4 items of one of
    mug, glass, bowl, plate or towel, a kind at most once, and no other description repeats (identical items exist
    only as sets). Regression caught: sets back to 1-2 groups of any kind, which left count questions about identical
    items at 3% (8 of 232) because scattered copies were rarely in view together."""
    for seed in range(1, 41):
        groups = spec_groups(initial_world("apartment_a", seed))
        sets = {k: v for k, v in groups.items() if len(v) > 1}
        assert 2 <= len(sets) <= 3, (seed, [(k[0], len(v)) for k, v in sets.items()])
        assert all(2 <= len(v) <= 4 for v in sets.values()), seed
        kinds = [k[0] for k in sets]
        assert set(kinds) <= {"mug", "glass", "bowl", "plate", "towel"} and len(set(kinds)) == len(kinds), seed


def test_each_set_starts_on_one_surface_when_it_fits():
    """Contract (M2 plan 4.1): the members of a set start together on one surface (same support piece, same surface
    height), which is what lets count's 'in view together' rule keep identical-item questions. On the apartment every
    set fits some surface, so none may be split. Regression caught: sets placed item by item over the whole flat."""
    for seed in range(1, 41):
        tl = initial_world("apartment_a", seed)
        for key, members in spec_groups(tl).items():
            if len(members) > 1:
                places = {(p.on, p.pos[2]) for p in (tl.placement_of(e.id, 0.0) for e in members)}
                assert len(places) == 1, (seed, key, places)


def test_entities_carry_their_kinds_manifest_asset_and_identicals_share_it():
    """Contract (M2 plan 4.2): every carried item (an item added in a gap too) and every light piece names the
    manifest's one model of its kind in `Entity.asset`; furniture and doors name none; identical items therefore share
    an asset, and the asset survives the JSON round trip. Regression caught: a mixed-asset identical group, an added
    delivery box left on placeholder geometry, or an asset on a door."""
    added = 0
    for seed in range(1, 13):
        for tl in (initial_world("apartment_a", seed), after_gap("apartment_a", seed)):
            for e in tl.entities:
                if e.mobility in ("carried", "light"):
                    assert e.asset == manifest_model_of(e.kind, e.attributes.get("color")), (seed, e)
                else:
                    assert e.asset is None, (seed, e)
            added += sum(tl.placement_of(e.id, 0.0) is None for e in tl.entities if e.mobility == "carried")
            assert all(e.asset == manifest_model_of(e.kind, e.attributes.get("color"))
                       for e in T.loads(T.dumps(tl)).entities if e.mobility == "carried")
    assert added >= 1  # an item that appears in a gap is covered


def test_door_entities_carry_hinge_and_swing():
    """Contract (map TV-H22): `build_world` copies each opening's hinge and swing into its door entity, opens every
    interior door at time 0 and keeps the entrance shut. Hand values, from the map: D0 (the entrance) hinge b swing
    left and shut; D1 and D2 hinge a, left; D3 and D4 hinge b, right; the four interior doors open, each by one state
    for the whole script. Regression caught: an open leaf that only map-reading code can find, or closed doors the
    vacuum drives through (the map's `initial_open` is false for every door)."""
    m = apartment()
    w = build_world(m, 1)
    tl = world_timeline(m, w, 1)
    want = {"D0": ("b", "left", False), "D1": ("a", "left", True), "D2": ("a", "left", True),
            "D3": ("b", "right", True), "D4": ("b", "right", True)}
    assert set(w.door_entity) == set(want)
    for door, (hinge, swing, is_open) in want.items():
        e = tl.entity(w.door_entity[door])
        assert (e.kind, e.attributes["hinge"], e.attributes["swing"]) == ("door", hinge, swing), door
        states = [s for s in tl.states if s.entity == e.id]
        assert [(s.key, s.t0, s.t1, s.value) for s in states] == [("open", 0.0, None, is_open)], door


def test_pieces_carry_their_under_m_and_regions_their_kind():
    """Contract (plan §2.2): the map's optional `furniture[i].under_m` and `regions[i].kind` reach the timeline: a piece
    with an open underside has it in its entity (0 for every other piece), and a region has its map kind, else its
    name. Hand case: the dining table given an under_m of 0.31 and the hall the kind 'front hall' in a copy of the
    apartment's file; every other piece keeps the under_m the file gives it. Regression caught: an underside that stays
    in the map, so the planner and `validate` read a solid table."""
    raw = json.loads(resources.files("stbench.maps").joinpath("apartment_a.json").read_bytes())
    next(f for f in raw["furniture"] if f["name"] == "dining table")["under_m"] = 0.31
    raw["regions"][0]["kind"] = "front hall"
    m = parse_map(raw)
    tl = world_timeline(m, build_world(m, 1), 1)
    pieces = [e for e in tl.entities if e.mobility in ("fixed", "light") and e.kind != "door"]
    want = {f.name: f.under_m for f in apartment().furniture}
    want["dining table"] = 0.31
    assert {e.attributes["name"]: e.under_m for e in pieces} == want
    assert [r.kind for r in tl.regions] == ["front hall", "living room", "kitchen", "bedroom", "bathroom"]
    assert [r.name for r in tl.regions] == ["hall", "living room", "kitchen", "bedroom", "bathroom"]


def test_every_item_carries_a_colour_of_its_kind_and_nothing_else():
    """Contract (the probe's predeclared consequence, DECISIONS "Probe run 1": the size attribute is dropped, size
    accuracy .389 over n 36 against .80; VISION rule 10): every carried entity of every generated world, those added
    in a gap included, has the one attribute "color", a colour its kind has in the catalog and the kind's own box, and
    every kind is one the catalog has. Regression caught: a size attribute back in the sampler or in the delivery box,
    a colour the catalog does not list, a box scaled for a variant, or a dropped kind generated."""
    seen_kinds: set[str] = set()
    for seed in range(1, 13):
        for tl in (initial_world("apartment_a", seed), after_gap("apartment_a", seed)):
            for e in (e for e in tl.entities if e.mobility == "carried"):
                assert set(e.attributes) == {"color"}, (seed, e)
                assert e.attributes["color"] in KINDS[e.kind].colors, (seed, e)
                assert e.size == KINDS[e.kind].size, (seed, e)
                seen_kinds.add(e.kind)
    assert seen_kinds <= set(KINDS) and len(seen_kinds) >= 12  # no dropped kind, and the worlds are varied


def test_about_seventy_percent_of_items_start_on_typical_receptacles():
    """Contract: priors must not be trivially exploitable: about 70% of items sit on a receptacle kind typical for them
    and about 30% do not (a blind 'where do mugs go' guess fails a third of the time). Regression caught: a placement
    that is always typical (the prior alone answers every where question) or never typical (the prior says nothing)."""
    typical = total = 0
    for seed in SEEDS:
        tl = initial_world("apartment_a", seed)
        for e in initial_carried(tl):
            support = tl.entity(tl.placement_of(e.id, 0.0).on)
            typical += bool(set(KINDS[e.kind].receptacles) & set(support.attributes["receptacles"]))
            total += 1
    assert 0.6 <= typical / total <= 0.8


def test_items_rest_at_the_height_of_a_map_surface(apt):
    """Contract: each item's base is at the height of one of its support's surfaces on the map (counter 0.9, table
    0.75...), looked up from the map file independently of the generator. Regression caught: an item floating above its
    support or sunk into it, which no camera would see standing on anything."""
    tl = initial_world("apartment_a", 2)
    heights = defaultdict(set)
    for f in apt.furniture:
        for s in f.surfaces:
            heights[f.name].add(s.z)
    for e in initial_carried(tl):
        support = tl.entity(tl.placement_of(e.id, 0.0).on)
        z = tl.placement_of(e.id, 0.0).pos[2]
        assert any(abs(z - h) < 1e-6 for h in heights[support.attributes["name"]]), (e.kind, support.attributes)


# -- the world processes of `changes`, over an explicit gap ------------------------------------------------------------
def test_change_counts_are_in_bounds():
    """Contract: per gap, carried moves around Poisson(2) per room (about 6-10 in total), 0-2 furniture shifts, about
    one add or remove. Every world has some change; averages sit in sane bands (3RScan-like sparsity is not claimed).
    Regression caught: a gap process that stops changing the world (nothing to ask about) or changes it wholesale
    (every item moved, so nothing stays to remember)."""
    moves, shifts, addrm = [], [], []
    for seed in range(1, 25):
        ch = changes(after_gap("apartment_a", seed))
        assert ch, seed
        moves.append(sum(c[0] == "move" for c in ch))
        shifts.append(sum(c[0] == "shift" for c in ch))
        addrm.append(sum(c[0] in ("add", "remove") for c in ch))
    n = len(moves)
    assert max(shifts) <= 2 and max(addrm) <= 2 and max(moves) <= 20
    assert 4.0 <= sum(moves) / n <= 11.0
    assert 0.4 <= sum(shifts) / n <= 1.6
    assert 0.5 <= sum(addrm) / n <= 1.5


def test_every_change_happens_inside_the_gap():
    """Contract: changes are timed while the robot is away: at least 30 s after the gap began and before it ended, so
    the ego never watches one happen. Regression caught: a change timed inside a task, which the robot would see happen
    (a 'moved while you were away' gold that its own camera contradicts)."""
    lo, hi = GAP
    for seed in range(1, 9):
        for kind, eid, t, *_ in changes(after_gap("apartment_a", seed)):
            assert lo + 30 <= t <= hi - 30, (seed, kind, eid, t)


def test_moves_and_shifts_displace_by_at_least_0_6_m():
    """Contract: a moved item or a shifted piece lands at least 0.6 m from where it was (so 'moved' is unambiguous).
    Regression caught: a move of a few centimetres, which no frame could tell from the item standing still, so a 'moved
    between' gold would have no visible answer."""
    seen = 0
    for seed in range(1, 17):
        for kind, eid, _, prev, nxt in changes(after_gap("apartment_a", seed)):
            if kind in ("move", "shift"):
                seen += 1
                assert distance(prev.pos[:2], nxt.pos[:2]) >= 0.6 - 1e-6, (seed, kind, eid)
    assert seen > 20


def test_shifted_light_furniture_moves_at_most_one_metre():
    """Contract: shifts are 0.6-1.0 m (V0: 0.3-1.0, raised to a 0.6 minimum). Regression caught: a light piece that
    jumps farther than a metre in one gap, beyond the shifts the contract names."""
    shifts = 0
    for seed in range(1, 17):
        for kind, _, _, prev, nxt in changes(after_gap("apartment_a", seed)):
            if kind == "shift":
                shifts += 1
                assert distance(prev.pos[:2], nxt.pos[:2]) <= 1.0 + 1e-6
    assert shifts > 5  # the loop saw some


def test_added_items_never_match_an_existing_description():
    """Contract: a delivery box that appears must not duplicate any existing item's kind and attributes: identical items
    exist only from the start, so an 'added' item is always the unique answer to its description. Regression caught: an
    added item identical to one already there, which would give a question about 'the new mug' two answers."""
    added = 0
    for seed in range(1, 25):
        tl = after_gap("apartment_a", seed)
        for kind, eid, _, _, _ in changes(tl):
            if kind == "add":
                added += 1
                e = tl.entity(eid)
                twins = [o for o in tl.entities if o.id != eid and o.kind == e.kind and o.attributes == e.attributes]
                assert twins == [], (seed, e)
    assert added >= 3


def test_to_timeline_keeps_the_info_and_the_maps_regions():
    """Contract: `to_timeline` is the one assembly of a script: the info it is given, the map's regions (with their
    kind), the world's entities and the ledger's placements and states. Regression caught: a region that loses its kind,
    or an info replaced on the way."""
    m = apartment()
    tl = initial_world("apartment_a", 1)
    assert tl.episode == info(m, 1)
    assert [(r.id, r.name, r.kind) for r in tl.regions] == [(r.id, r.name, r.kind) for r in m.regions]
    assert tl.placements and tl.states


if __name__ == "__main__":  # the second process of `test_hash_is_stable_across_processes`
    print(T.digest(globals()[sys.argv[1]](sys.argv[2], int(sys.argv[3]))))
