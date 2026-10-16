"""Placement contracts for identical sets (M2 plan 4.1): what is sampled, and how a set is put on surfaces."""

from __future__ import annotations

from collections import Counter

from stbench.catalog import KINDS
from stbench.geometry import Rect, corners, rects_overlap, to_local
from stbench.placement import (
    ITEM_MARGIN,
    ItemSpec,
    Slot,
    _siblings,
    item_rect,
    occupied_entry,
    place_set,
    sample_plan,
)
from stbench.seeding import make_rng

MUG = ItemSpec("mug", {"color": "red"})
PLATE = ItemSpec("plate", {"color": "white"})  # the plate's box is 0.222 m square
SET_KINDS = {"mug", "glass", "bowl", "plate", "towel"}  # written out: a change to the catalog's list must be seen


def surface(piece: str, sid: str, kind: str, rect: Rect, z: float = 0.75) -> Slot:
    return Slot(piece, sid, kind, rect, z, None)


def key(spec: ItemSpec) -> tuple:
    return spec.key


def test_a_plan_draws_two_or_three_identical_sets_of_two_to_four_from_the_set_kinds():
    """Contract (plan 4.1, the identity major): 2-3 identical sets of 2-4 items, each of a different kind from the five
    kinds people own in sets; every member of a set is the same (kind, attributes); no other item repeats a set's or
    another item's description; 30-40 items in all.
    Regression caught: sets back to 1-2 groups of any kind (scattered copies, rarely in view together), a set of one,
    or a 'single' that silently duplicates a set so a description matches more items than the plan says."""
    for seed in range(1, 41):
        plan = sample_plan(make_rng("plan", seed))
        assert 2 <= len(plan.sets) <= 3, seed
        assert all(2 <= len(s) <= 4 for s in plan.sets), seed
        assert all(len({m.key for m in s}) == 1 for s in plan.sets), seed
        kinds = [s[0].kind for s in plan.sets]
        assert set(kinds) <= SET_KINDS and len(set(kinds)) == len(kinds), (seed, kinds)
        others = Counter(key(o) for o in plan.others)
        assert max(others.values()) == 1, seed  # no repeats among the rest
        assert not set(others) & {s[0].key for s in plan.sets}, seed
        assert 30 <= len(plan.items) <= 40, seed


def test_one_colour_kinds_make_no_attribute_group():
    """Contract (V0 section 4: a lookalike group differs in exactly one attribute, and a description is ambiguous only
    inside an identical set or a lookalike group): a kind with one colour (the plant) has no sibling, and
    a sibling of another kind has the spec's kind and a colour that is neither the spec's nor one in use (None when too
    few are left). Over seeds 1-40 no world holds two items of a one-colour kind. Regression caught: a sibling with the
    same description as its spec (an unmarked identical pair, so a question about it has no unique answer), or a
    one-colour kind drawn twice."""
    for kind in (k.name for k in KINDS.values() if len(k.colors) == 1):
        assert _siblings(make_rng("one", 1), ItemSpec(kind, {"color": KINDS[kind].colors[0]}), 1, set()) is None, kind
    spec = ItemSpec("remote", {"color": "black"})
    used = {ItemSpec("remote", {"color": "white"}).key, spec.key}
    for seed in range(20):
        assert _siblings(make_rng("two", seed), spec, 2, used) is None  # one colour left, two wanted
        (sib,) = _siblings(make_rng("one", seed), spec, 1, used)
        assert sib.kind == "remote" and sib.attributes == {"color": "grey"}, seed
    for seed in range(1, 41):
        kinds = Counter(i.kind for i in sample_plan(make_rng("plan", seed)).items)
        assert all(kinds[k.name] <= 1 for k in KINDS.values() if len(k.colors) == 1), seed


def inside_surface(slot: Slot, r: Rect) -> bool:
    """Every corner of the footprint, in the surface's own frame, is within its half extents."""
    local = [to_local(slot.rect, *c) for c in corners(r)]
    return all(abs(x) <= slot.rect.sx / 2 and abs(y) <= slot.rect.sy / 2 for x, y in local)


def test_a_set_that_fits_sits_in_one_row_on_one_surface():
    """Contract (plan 4.1): the members of a set sit side by side in one row on one surface, all the same way round,
    inside the surface and `ITEM_MARGIN` apart (so count's 'in view together' holds). A 1.0 x 0.5 table holds four 0.10
    mugs in a row along its long side (0.46 m): one coordinate across the table, four along it, each footprint, from its
    own corners, inside. Regression caught: members scattered or stacked, overlapping, or hanging over an edge."""
    table = surface("F1", "S1", "dining table", Rect(2.0, 3.0, 1.0, 0.5, 0.0))
    for seed in range(20):
        spots = place_set(make_rng("row", seed), (MUG,) * 4, [table], [])
        assert len(spots) == 4 and all(s is not None and s.slot is table for s in spots), seed
        rects = [item_rect(MUG, s.pos[0], s.pos[1], s.yaw) for s in spots]
        assert {s.pos[2] for s in spots} == {0.75} and len({round(s.yaw, 9) for s in spots}) == 1, seed
        assert all(inside_surface(table, r) for r in rects), seed
        assert not any(rects_overlap(a, b, ITEM_MARGIN) for i, a in enumerate(rects) for b in rects[i + 1:]), seed
        across = [to_local(table.rect, r.cx, r.cy)[1] for r in rects]
        along = [to_local(table.rect, r.cx, r.cy)[0] for r in rects]
        assert max(across) - min(across) < 1e-9 and len({round(x, 6) for x in along}) == 4, seed


def test_a_set_keeps_clear_of_items_already_on_the_surface():
    """Contract (plan 4.1, validity): a set never lands on what is there. A 0.3 m item stands at the left end of the
    1.0 x 0.5 table (local x -0.5..-0.2); the four mugs go beside it, inside the table, `ITEM_MARGIN` clear of it and of
    each other. Regression caught: a block laid out without looking at `occupied`, which `validate` rejects as a
    collision."""
    table = surface("F1", "S1", "dining table", Rect(2.0, 3.0, 1.0, 0.5, 0.0))
    neighbour = (Rect(1.65, 3.0, 0.3, 0.3, 0.0), 0.75, 0.95)
    for seed in range(20):
        spots = place_set(make_rng("clear", seed), (MUG,) * 4, [table], [neighbour])
        assert all(s is not None for s in spots), seed
        rects = [item_rect(MUG, s.pos[0], s.pos[1], s.yaw) for s in spots]
        assert all(inside_surface(table, r) and not rects_overlap(r, neighbour[0], ITEM_MARGIN) for r in rects), seed
        assert not any(rects_overlap(a, b, ITEM_MARGIN) for i, a in enumerate(rects) for b in rects[i + 1:]), seed


def test_a_set_that_fits_no_surface_spreads_over_the_surfaces_of_one_piece():
    """Contract (plan 4.1): when no single surface holds the set, its members spread over the surfaces of one piece of
    furniture. Hand case: three 0.222 m plates; every surface is 0.3 x 0.3 (one plate each, two would need 0.48);
    the shelf unit F2 has three of them, the lone tray F3 one. The members land on F2's three surfaces, one each.
    Regression caught: a set that splits over several pieces, or never splits and is dropped."""
    unit = [surface("F2", f"S{i}", "shelf", Rect(5.0, 1.0, 0.3, 0.3, 0.0), 0.4 + 0.5 * i) for i in range(3)]
    tray = surface("F3", "S9", "shelf", Rect(8.0, 1.0, 0.3, 0.3, 0.0))
    for seed in range(10):
        spots = place_set(make_rng("spread", seed), (PLATE,) * 3, [*unit, tray], [])
        assert all(s is not None for s in spots), seed
        assert {s.slot.furniture_entity for s in spots} == {"F2"}, seed
        assert sorted(s.slot.surface_id for s in spots) == ["S0", "S1", "S2"], seed


def test_a_set_with_no_piece_to_spread_over_places_each_member_where_it_fits():
    """Contract (plan 4.1): last, each member goes wherever there is room, and a member with no room is None (the
    generator leaves it out) rather than an error. Three 0.3 x 0.3 single-surface pieces take one plate each; a
    world of 0.1 x 0.1 surfaces takes none.
    Regression caught: a set that cannot be kept together raising or dropping the members that do fit."""
    singles = [surface(f"F{i}", "S", "shelf", Rect(2.0 * i, 0.0, 0.3, 0.3, 0.0)) for i in range(3)]
    spots = place_set(make_rng("single", 1), (PLATE,) * 3, singles, [])
    assert sorted(s.slot.furniture_entity for s in spots) == ["F0", "F1", "F2"]
    tiny = [surface(f"F{i}", "S", "shelf", Rect(2.0 * i, 0.0, 0.1, 0.1, 0.0)) for i in range(3)]
    assert place_set(make_rng("single", 2), (PLATE,) * 3, tiny, []) == [None, None, None]


def test_place_set_leaves_the_occupied_list_to_its_caller():
    """Contract: `place_set` returns spots and does not record them, as `find_spot` does not; the caller adds what it
    keeps. Regression caught: a set that marks its own footprints, so the caller's later entries double-count and an
    item that was dropped still blocks its place."""
    table = surface("F1", "S1", "dining table", Rect(2.0, 3.0, 1.0, 0.5, 0.0))
    occupied = [occupied_entry(MUG, (2.4, 3.0, 0.75), 0.0)]
    place_set(make_rng("pure", 1), (MUG,) * 3, [table], occupied)
    assert len(occupied) == 1


def test_sets_follow_the_typical_receptacle_rule_like_single_items():
    """Contract (priors must not be exploitable): a set goes on a typical receptacle kind 70% of the time, as a single
    item does, not always. Two surfaces of equal size, one a kitchen counter (typical for a mug) and one a bed;
    over 300 draws 60-80% land on the counter.
    Regression caught: sets always placed on typical surfaces, which lifts the typical share of all items above the
    band a blind 'where do mugs go' guess must not exploit."""
    counter = surface("F1", "S1", "kitchen counter", Rect(0.0, 0.0, 1.0, 0.5, 0.0), 0.9)
    bed = surface("F2", "S2", "bed", Rect(5.0, 0.0, 1.0, 0.5, 0.0), 0.5)
    on_counter = sum(place_set(make_rng("typ", i), (MUG,) * 3, [counter, bed], [])[0].slot is counter
                     for i in range(300))
    assert 180 <= on_counter <= 240
