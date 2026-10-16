"""`adjacency` (the rule `adjacent` and its table systems share), by hand on `adjacent_kit`'s six-room apartment."""

from __future__ import annotations

from dataclasses import replace

import adjacent_kit as ak
from stbench import mapspec
from stbench.adjacency import (
    connector,
    crossed,
    first_crossings,
    inside_cells,
    interior_point,
    joins,
    regions_along,
    toured,
)
from stbench.mapspec import Wall

HALL, COR, KIT, STUDY, BED, BATH = "R1", "R2", "R3", "R4", "R5", "R6"


def pair(a: str, b: str) -> frozenset[str]:
    return frozenset((a, b))


def test_joins_by_hand():
    """Contract: rooms are joined by a door, an arch, or an open shared edge of at least 0.8 m, and by nothing else. On
    the kit map (itself a legal map, `check_map` clean) hall-corridor and corridor-kitchen (doors), kitchen-study (an
    arch) and corridor-bedroom (a shared edge of 1.0 m, x = 6.1, y 2..3) are joined; hall-study (a wall between, a
    0.1 m gap between their polygons), kitchen-bedroom (a wall), bedroom-bathroom (a shared edge of 0.5 m) and the
    entrance (outside to the hall) are not, and the bathroom joins nothing. Regression caught: a region graph joining
    rooms no robot passes between (a wall, a short edge, the entrance) or leaving out the open-plan edge."""
    m = ak.kit_map()
    assert mapspec.check_map(m) == []
    assert joins(m) == {pair(HALL, COR), pair(COR, KIT), pair(KIT, STUDY), pair(COR, BED)}


def test_a_wall_on_an_open_edge_decides():
    """Contract: an open edge joins only where a run of 0.8 m or more is clear of wall pieces. A wall along the
    corridor-bedroom edge's lower half (x = 6.1, y 2.0..2.5, pieces reaching 0.05 m past each end) leaves 0.45 m
    clear: not joined; a stub covering 0.1 m of it (y 2.0..2.05, the piece to 2.1) leaves 0.9 m: joined; a wall whose
    face lies against the whole edge, 1 mm onto it (x 6.099..6.199, y -0.05..3.25), stands on none of it.
    Regression caught: a wall drawn across an open plan ignored, or a doorjamb stub (or a flush wall) taken to close a
    passage."""
    m = ak.kit_map()

    def with_wall(a, b, x=6.1):
        return replace(m, walls=(*m.walls, Wall("W9", (x, a), (x, b), 0.1, 2.5, False)))

    assert pair(COR, BED) not in joins(with_wall(2.0, 2.5))
    assert pair(COR, BED) in joins(with_wall(2.0, 2.05))
    assert pair(COR, BED) in joins(with_wall(0.0, 3.2, x=6.149))


def test_the_edge_minimum_is_the_narrowest_door():
    """Contract: a shared edge joins from 0.8 m, the narrowest door `check_map` allows: the bedroom moved up so that
    its edge with the corridor is 0.8 m (y 2.2..3.0) joins them, and 0.75 m (y 2.25..3.0) does not. Regression
    caught: a bound off by a tenth, closing the passage every shorter door would have kept open or opening a gap no
    robot fits."""
    m = ak.kit_map()

    def bedroom_from(y0):
        bed = replace(m.region(BED), polygon=((6.1, y0), (9.1, y0), (9.1, 5.0), (6.1, 5.0)))
        return replace(m, regions=tuple(bed if r.id == BED else r for r in m.regions))

    assert pair(COR, BED) in joins(bedroom_from(2.2))
    assert pair(COR, BED) not in joins(bedroom_from(2.25))


def test_stacked_floors_share_no_edge():
    """Contract: regions of different levels are never joined by an edge, whatever their outlines: a loft on the
    storey above the corridor, with the corridor's own polygon, shares every edge of it and joins nothing. Regression
    caught: a two-storey building read as one open plan."""
    m = ak.kit_map()
    above = replace(m.region(COR), id="R7", name="loft", kind="loft", level=1)
    assert pair(COR, "R7") not in joins(replace(m, regions=(*m.regions, above)))
    assert pair(COR, "R7") in joins(replace(m, regions=(*m.regions, replace(above, level=0))))


def test_the_connector_is_one_room_or_none():
    """Contract: `connector(a, b)` is the one region joined to both when a and b are different and not joined to each
    other. On the kit map kitchen and bedroom (both joined to the corridor, and to nothing else in common) give the
    corridor; hall and bedroom the corridor; corridor and study the kitchen. Hall and study (a wall between: no
    region joined to both), the joined hall and corridor, a room and itself and the bathroom (joined to nothing) give
    None; and with a door cut between hall and study, corridor and study are joined to both the hall and the kitchen,
    and kitchen and hall to both the corridor and the study: two connectors, None, while hall and bedroom still have
    the corridor alone; and with a door between kitchen and bedroom (a triangle with the corridor) the joined kitchen
    and corridor, though both joined to the bedroom, have none. Regression caught: "the room that connects" when two
    do, or a joined pair given a third room."""
    m = ak.kit_map()
    assert connector(m, KIT, BED) == COR and connector(m, HALL, BED) == COR and connector(m, COR, STUDY) == KIT
    assert connector(m, HALL, STUDY) is None and connector(m, HALL, COR) is None and connector(m, KIT, KIT) is None
    assert connector(m, BATH, KIT) is None and connector(m, BED, BATH) is None
    both = ak.kit_map(hall_study_door=True)
    assert joins(both) == joins(m) | {pair(HALL, STUDY)}
    assert connector(both, COR, STUDY) is None and connector(both, KIT, HALL) is None
    assert connector(both, HALL, BED) == COR
    triangle = ak.kit_map(kitchen_bedroom_door=True)
    assert connector(triangle, KIT, COR) is None and connector(triangle, KIT, BED) is None
    assert connector(triangle, HALL, KIT) == COR and connector(triangle, STUDY, BED) == KIT


def test_crossings_by_hand():
    """Contract: a path's rooms are its base positions' regions in order, a position in no room (a doorway, a wall)
    dropped and a repeat collapsed; its crossings are the consecutive pairs that are joined, each with the index of
    the position that first enters the second room. On the kit map the walk corridor (4.6, 1.5), the door D2, kitchen,
    D2, corridor, corridor, bedroom, the gap between kitchen and bedroom (6.05, 4.0) and the hall (a jump from the
    bedroom, which no opening joins to it) is corridor, kitchen, corridor, bedroom, hall; it crosses corridor-kitchen
    (first at index 2) and corridor-bedroom (first at 6) and not bedroom-hall; and a walk that never leaves one room
    crosses nothing. Regression caught: a doorway read as a room, a jump between keyframes counted as a crossing
    of rooms no opening joins, or a first crossing indexed at the room it left."""
    m = ak.kit_map()
    walk = [ak.COR, ak.D2, ak.KIT, ak.D2, ak.COR, ak.COR_E, ak.BED, (6.05, 4.0), ak.HALL]
    assert regions_along(m, walk) == [COR, KIT, COR, BED, HALL]
    assert crossed(m, walk) == {pair(COR, KIT), pair(COR, BED)}
    assert first_crossings(m, walk) == {pair(COR, KIT): 2, pair(COR, BED): 6}
    assert regions_along(m, [ak.HALL, ak.D1, ak.COR]) == [HALL, COR]
    assert first_crossings(m, [ak.HALL, ak.D1, ak.COR]) == {pair(HALL, COR): 2}
    assert crossed(m, [ak.HALL, (1.0, 1.0), ak.HALL]) == frozenset() and regions_along(m, [ak.D1, ak.D2]) == []


def test_an_interior_point_keeps_its_margin():
    """Contract: `interior_point` is a polygon's centroid when it lies at least the margin inside, else None: the
    corridor (3.1..6.1 x 0..3) gives (4.6, 1.5), a 0.3 m wide strip with a 0.2 m margin gives None, and an L shape's
    centroid outside the shape (a concave region) gives None. Regression caught: a gold point on a wall, which the
    `region` credit would score 0."""
    assert interior_point(ak.kit_map().region(COR).polygon, 0.2) == (4.6, 1.5)
    assert interior_point([(0.0, 0.0), (3.0, 0.0), (3.0, 0.3), (0.0, 0.3)], 0.2) is None
    ell = [(0.0, 0.0), (4.0, 0.0), (4.0, 0.5), (0.5, 0.5), (0.5, 4.0), (0.0, 4.0)]
    assert interior_point(ell, 0.0) is None


def test_a_toured_room_is_half_its_cells_in_any_state():
    """Contract: a room is toured when at least half of its inside fine cells are among the observed cells, in any
    state (`changed` asks the same rule through this function): the kitchen (870 inside cells) is toured at 435
    observed cells, occupied or free, and not at 434. Regression caught: a bound off by one cell, or a room of which
    only the free cells count."""
    kitchen, cells = ak.kit_map().region(KIT), ak.room_cells(KIT)
    assert len(inside_cells(kitchen)) == 870 and set(inside_cells(kitchen)) <= set(cells)
    assert toured(kitchen, dict.fromkeys(cells[:435], ("occupied", 3)))
    assert not toured(kitchen, dict.fromkeys(cells[:434], ("free", 3)))
