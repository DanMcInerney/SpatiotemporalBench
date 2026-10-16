"""The zone rule (`zones`): a piece's footprint, inset, on its ground, and what free-space cells say of it, by hand on
`zone_kit`'s map (two bays, a drop zone on a 0.15 m plaza, four staging lanes) and the tote robot's ray-cast camera."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

import zone_kit as zk
from space_kit import box
from stbench import zones
from stbench.geometry import corners
from stbench.zones import ZoneRule

CELLS = zones.zone_cells(zk.B1)
PLAZA_VIEW = {"base": (3.0, 5.5), "yaw": math.pi / 2}  # the drop zone ahead: all 576 cells in the camera's floor view
SLAB, MARKING = box(1.0, 5.0, 6.5, 10.5, 0.0, 0.15), box(1.8, 4.2, 7.3, 9.7, 0.15, 0.155)


def occupied_in(zone, states) -> set:
    return {c for c in zones.zone_cells(zone) if states.get(c, ("",))[0] == "occupied"}


def lump(height: float, centre=(1.5, 1.5), mobility: str = "fixed"):
    return replace(zk.piece("X1", "lump", "table", centre, (0.4, 0.4, height)), mobility=mobility)


def test_zone_state_by_hand():
    """Contract: a bay is blocked when anything over 0.3 m stands in it, clear when 80% of its cells were seen free and
    nothing blocked it, else unknown, from the cells a ray-cast frame observed with the zone's own band. B1's zone is
    x 0.75..2.25, y 0.45..2.55: 16 x 22 = 352 cells. Seen from (3.9, 1.5) facing west:
    - empty: all 352 cells free (evidence the frame, keyframe 0), clear;
    - a 0.5 m box in it (x 1.2..1.6, y 1.2..1.8): blocked, by the frame;
    - a 0.2 m box there: clear (its points are under the 0.3 m the band starts at);
    - 211 of the 352 cells seen (59.9%): unknown, with the keyframe that saw them;
    - a 0.5 m box whose face stands on the bay's north edge, 0.05 m outside the zone (x 1.03..1.97, y 2.6..3.1): its
      face points have y 2.5995..2.6, in the sub-cells centred 2.59 (outside the zone), whose fine cells v = 25 are
      the zone's top row (centre 2.55, on the edge, so inside): the ten cells u = 10..19 are occupied and none
      blocks, 342 of 352 are free: clear.
    Regression caught: a zone judged on another band or polygon, a thing beside a zone blocking it (the corridor's
    rule: a fine cell's centre decides nothing, its obstacle sub-cells do), or an unseen zone called clear."""
    assert len(CELLS) == 352
    empty = zk.seen(zk.B1)
    assert zones.zone_state(empty, zk.B1) == ("clear", [0]) and zones.coverage(empty, zk.B1) == 1.0
    assert zones.zone_state(zk.seen(zk.B1, [box(1.2, 1.6, 1.2, 1.8, 0.0, 0.5)]), zk.B1) == ("blocked", [0])
    assert zones.zone_state(zk.seen(zk.B1, [box(1.2, 1.6, 1.2, 1.8, 0.0, 0.2)]), zk.B1) == ("clear", [0])
    partial = {c: ("free", 4) for c in CELLS[:211]}
    assert zones.zone_state(partial, zk.B1) == ("unknown", [4])
    beside = zk.seen(zk.B1, [box(1.03, 1.97, 2.6, 3.1, 0.0, 0.5)])
    assert occupied_in(zk.B1, beside) == {(u, 25) for u in range(10, 20)}
    assert zones.zone_state(beside, zk.B1) == ("clear", [0])


def test_a_station_never_blocks_its_own_bay():
    """Contract (the review's B2): a station that only touches its bay never blocks it. A shelving box (x 0.2..0.7,
    y 0.4..2.6, 0.9 m tall) against B1's west face, ray-cast at millimetre depth from 3.2 m in front of the face
    (the base (3.9, 1.5), the whole bay in view), puts its face points in the sub-cells centred 0.69 and 0.71, in the
    fine cells u = 6 and u = 7 (the zone's west column u = 7, 22 cells, has its centre 0.75 on the zone's edge, so
    it holds the station and none blocks): the zone, inset 0.05 m to x 0.75, is clear; a 0.5 m box standing 0.1 m
    inside that edge (x 0.85..1.35) blocks it. Without the inset (the piece's own footprint, x 0.7..2.3) the same
    station is read as an occupant and the bay is blocked. Regression caught: a fixed piece read as an occupant, which
    made `zone_clear` `no` at every docking of the tote robot."""
    station = box(0.2, 0.7, 0.4, 2.6, 0.0, 0.9)
    touching = zk.seen(zk.B1, [station])
    assert len(occupied_in(zk.B1, touching)) == 22 and zones.zone_state(touching, zk.B1) == ("clear", [0])
    standing = zk.seen(zk.B1, [station, box(0.85, 1.35, 1.2, 1.8, 0.0, 0.5)])
    assert zones.zone_state(standing, zk.B1) == ("blocked", [0])
    footprint = replace(zk.B1, polygon=tuple(corners(zk.PIECES[0].footprint())))
    assert zones.zone_state(touching, footprint) == ("blocked", [0])


def test_the_ground_is_the_low_piece_under_it():
    """Contract: a zone lies on the top of the tallest other fixed piece of at most 0.2 m whose footprint holds the
    piece's centre, else on the floor, and is judged from that height with the band (0.3 m, 3.5 m) above it. Hand
    values: the drop zone (0.155 m) on the 0.15 m plaza has the ground 0.15; a bay has 0; a 0.3 m piece under a bay's
    centre is no ground (0), a 0.18 m one is (0.18) and the taller of two low ones (0.1, 0.18) is; a 0.18 m piece that
    does not hold the centre, or a light one that does, is none. Ray-cast from (3.0, 5.5) facing north, over the plaza
    slab and the drop zone's marking (all 576 cells in view): empty and with a 0.2 m box standing on the slab (top
    0.35 m, 0.2 m above its ground) it is clear, with a 0.45 m box (top 0.6 m) blocked; judged from the street's level
    the 0.2 m box is blocked (its top is 0.35 m up). Regression caught: a marking on a plaza judged from the street."""
    drop, bay = zk.PIECES[3], zk.PIECES[0]
    assert (zones.ground_z(zk.MAP, drop), zones.ground_z(zk.MAP, bay)) == (0.15, 0.0)
    assert (zk.DROP_ZONE.base_z, zk.DROP_ZONE.band, zk.B1.base_z) == (0.15, (0.3, 3.5), 0.0)

    def ground(*extra):
        return zones.ground_z(replace(zk.MAP, furniture=(*zk.MAP.furniture, *extra)), bay)

    assert ground(lump(0.3)) == 0.0 and ground(lump(0.18)) == 0.18 and ground(lump(0.1), lump(0.18)) == 0.18
    assert ground(lump(0.18, centre=(3.0, 1.5))) == 0.0 and ground(lump(0.18, mobility="light")) == 0.0
    scene = [SLAB, MARKING]
    assert zones.zone_state(zk.seen(zk.DROP_ZONE, scene, **PLAZA_VIEW), zk.DROP_ZONE) == ("clear", [0])
    low = [*scene, box(2.5, 3.0, 8.2, 8.7, 0.15, 0.35)]
    assert zones.zone_state(zk.seen(zk.DROP_ZONE, low, **PLAZA_VIEW), zk.DROP_ZONE) == ("clear", [0])
    high = [*scene, box(2.5, 3.0, 8.2, 8.7, 0.15, 0.60)]
    assert zones.zone_state(zk.seen(zk.DROP_ZONE, high, **PLAZA_VIEW), zk.DROP_ZONE) == ("blocked", [0])
    street = replace(zk.DROP_ZONE, base_z=0.0)
    assert zones.zone_state(zk.seen(street, low, **PLAZA_VIEW), street) == ("blocked", [0])


def test_a_zone_is_the_footprint_grown_and_inset_by_hand():
    """Contract: a zone's polygon is its piece's footprint grown by the rule's margin less `EDGE_M` (0.05 m),
    counter-clockwise, and its cells are the fine cells whose centres lie in it, a centre on an edge (to 1e-9) in, at
    any yaw. Hand values: B1 (centre (1.5, 1.5), 1.6 x 2.2 m) is x 0.75..2.25, y 0.45..2.55, whose cell centres run
    x 0.75 (u = 7) to 2.25 (u = 22) and y 0.45 (v = 4) to 2.55 (v = 25); B2 (centre (1.5, 4.5), turned by pi, whose
    corners carry 1e-16 of noise) has the same size and u, with v = 34..55; a margin of 0.2 m grows it by 0.15 m
    (x 0.55..2.45, y 0.25..2.75: u 5..24, v 2..27); a piece smaller than the inset has no zone. Regression caught: a
    polygon that is the footprint (the station touching a bay then blocks it), an edge row lost to float noise on a
    turned piece, or the margin added to the whole footprint."""
    assert [tuple(round(v, 9) for v in p) for p in zk.B1.polygon] == [(2.25, 2.55), (0.75, 2.55), (0.75, 0.45),
                                                                     (2.25, 0.45)]
    assert zones.zone_cells(zk.B1) == [(u, v) for u in range(7, 23) for v in range(4, 26)]
    assert zones.zone_cells(zk.B2) == [(u, v) for u in range(7, 23) for v in range(34, 56)]
    wide = zones.zones_of(zk.MAP, ZoneRule("the bay", "bay", margin_m=0.2))[0]
    assert zones.zone_cells(wide) == [(u, v) for u in range(5, 25) for v in range(2, 28)]
    assert zones.zones_of(zk.MAP, ZoneRule("the bay", "bay", margin_m=0.2))[0].piece == "F01"
    assert [z.piece for z in zones.zones_of(zk.MAP, zk.LANES)] == ["F05", "F06", "F07", "F08"]
    assert zones.centroid(zk.L2) == (12.5, 2.0)
    with pytest.raises(ValueError, match="F01"):
        zones.zones_of(zk.MAP, ZoneRule("the bay", "bay", margin_m=-0.8))


def test_the_zone_is_the_nearest_piece_in_map_order():
    """Contract: the zone is the piece of the rule's kind whose centre is nearest the point (a table system chooses it
    from the public map and its own pose), the first in map order on a tie to 1e-9, None when the map has no piece of
    the kind. Hand values: (1.5, 2.0) is 0.5 from B1 and 2.5 from B2; (1.5, 3.9) is 0.9 from B2; (1.5, 3.0) is 1.5
    from both, B1 first in map order, and with the map reversed B2; 1e-10 past the middle is still a tie, 1e-6 is B2.
    Regression caught: the lowest id or the last of a tie chosen, or a zone found by anything but the map and a
    pose."""
    def near(xy, m=zk.MAP):
        return zones.zone_of(m, zk.BAY, xy).piece

    assert (near((1.5, 2.0)), near((1.5, 3.9)), near((1.5, 3.0))) == ("F01", "F02", "F01")
    assert near((1.5, 3.0 + 1e-10)) == "F01" and near((1.5, 3.0 + 1e-6)) == "F02"
    assert near((1.5, 3.0), replace(zk.MAP, furniture=zk.MAP.furniture[::-1])) == "F02"
    assert zones.zone_of(zk.MAP, ZoneRule("the dock", "dock"), (1.5, 1.5)) is None


def test_a_zone_is_in_view_from_80_percent_of_its_cells():
    """Contract: a zone is in view when at least `ZONE_SHARE` (80%) of its cells were observed in any state, and clear
    when at least 80% were free and none blocks. Hand values for B1's 352 cells (80% is 281.6): 282 observed is in
    view (0.801), 281 is not (0.798); 282 free of 352 is clear, 281 free is unknown; coverage counts an occupied cell
    that blocks nothing; a zone nobody observed has coverage 0. Regression caught: a threshold that moves by a cell
    without anyone seeing it, which would change which wording `zone_clear` asks."""
    def at(n, state="free"):
        return {c: (state, 3) for c in CELLS[:n]}

    assert zones.in_view(at(282), zk.B1) and not zones.in_view(at(281), zk.B1)
    assert zones.coverage(at(282), zk.B1) == 282 / 352 and zones.coverage({}, zk.B1) == 0.0
    assert zones.zone_state(at(282), zk.B1)[0] == "clear" and zones.zone_state(at(281), zk.B1)[0] == "unknown"
    assert zones.zone_state({}, zk.B1) == ("unknown", [])
    bare_edge = {**at(281), CELLS[281]: ("occupied", 3)}  # a bare "occupied" holds all 25 sub-cells: in the zone
    assert zones.zone_state(bare_edge, zk.B1)[0] == "blocked" and zones.in_view(bare_edge, zk.B1)


def test_the_evidence_is_what_decided_the_state():
    """Contract: the evidence is the sorted distinct keyframes of the blocking cells when blocked, else of every
    observed zone cell. Hand values: 300 cells last seen at keyframe 3 and 52 at 5 give [3, 5] for clear; a blocker
    last seen at 7 among them gives [7] (blocked, whatever else was seen); a cell outside the zone seen at 9 adds
    nothing. Regression caught: evidence that names frames the verdict did not read, or leaves out one it did."""
    mixed = {**{c: ("free", 3) for c in CELLS[:300]}, **{c: ("free", 5) for c in CELLS[300:]}, (0, 0): ("free", 9)}
    assert zones.zone_state(mixed, zk.B1) == ("clear", [3, 5])
    assert zones.zone_state({**mixed, CELLS[10]: ("occupied", 7)}, zk.B1) == ("blocked", [7])


def test_band_cells_key_the_cells_by_ground_and_band():
    """Contract: `band_cells` lists the fine cells of every zone of the rules by the (ground, band) they are observed
    with, so a zone band's states can keep only those. Hand values: the bays (ground 0, band 0.3..3.5) hold both bays'
    352 cells each; the drop zone (0.15, 0.3..3.5) its 576; a rule asking 0.5 m has a key of its own with the same bays'
    cells. Regression caught: a cache that keeps whole drone frames, or a band that reads another's cells."""
    taller = ZoneRule("the tall bay", "bay", height_m=0.5)
    keep = zones.band_cells(zk.MAP, [zk.BAY, zk.DROP, taller])
    bays = frozenset(zones.zone_cells(zk.B1)) | frozenset(zones.zone_cells(zk.B2))
    assert keep == {(0.0, (0.3, 3.5)): bays, (0.15, (0.3, 3.5)): frozenset(zones.zone_cells(zk.DROP_ZONE)),
                    (0.0, (0.5, 3.5)): bays}
    assert len(keep[(0.0, (0.3, 3.5))]) == 704 and len(keep[(0.15, (0.3, 3.5))]) == 576
