"""The household's grime (`render/dressing_home_grime.py`, M1's R.2): where the dust, hairs, crumbs and scuffs of the
apartment lie. Free: the layout is plain Python over the map; the pieces are `test_dressing_layout_home`'s."""

from __future__ import annotations

import itertools
import math
from collections import Counter

from test_dressing_layout_home import GRIME, _dressing, _map, _of, _rect


def test_the_grime_is_flush_where_a_floor_is_dirty_and_nowhere_else():
    """Contract (`dressing_home_grime`; the guard's flush floor and swept bodies are `test_dressing_zones`'s, door
    swings and overlaps `test_dressing`'s and the test above): every piece of grime lies on the floor and is under 1 cm
    thick (the helper climbs 1 cm, the vacuum 2: nothing here is an obstacle); a dust line is 3.5 cm deep, 0.2 to 1.4 m
    long and lies against the foot of a baseboard, in front of it; hairs are at most 15 (3 in the hall, 3 in the living
    room, 2 in the kitchen, 3 in the bedroom, 4 in the bathroom: the table in `dressing_home_grime`) and at least 12,
    crumbs are in the kitchen and the living room only, 10 and 3 clusters at most, each by a counter, the stove, the
    sink unit or a table; a scuff is 45 x 10 cm in front of a door. Nothing lies on a rug or the mat (F14, F22, F35):
    under the pile a plate would be hidden, on it in the volume things rest in. Regression caught: dirt a robot
    drives into, dust floating off the wall, crumbs in the bedroom, or grime that vanishes under a rug."""
    m = _map()
    grime = [d for d in _dressing() if d.cls in GRIME]
    assert grime and all(d.pos[2] == 0.0 and d.size[2] < 0.01 for d in grime)
    baseboards = _of("baseboard")
    dust = _of("dust_line")
    assert dust and all(d.size[0] == 0.035 and 0.2 <= d.size[1] <= 1.4 for d in dust)
    for d in dust:
        x0, y0, _z0, x1, y1, _z1 = d.box
        along_x = x1 - x0 > y1 - y0
        lo, hi = (x0, x1) if along_x else (y0, y1)
        near, far = (y0, y1) if along_x else (x0, x1)
        assert any(b[0 if along_x else 1] - 1e-3 <= lo and hi <= b[3 if along_x else 4] + 1e-3
                   and (abs(near - b[4 if along_x else 3]) < 2e-3 or abs(far - b[1 if along_x else 0]) < 2e-3)
                   for b in [bb.box for bb in baseboards]), d.pos
    rooms = Counter(m.region_of(d.pos[:2]) for d in _of("hair"))
    by_room = {r.name: r.id for r in m.regions}
    caps = {"hall": 3, "living room": 3, "kitchen": 2, "bedroom": 3, "bathroom": 4}
    assert 12 <= sum(rooms.values()) <= 15 and all(rooms[by_room[n]] <= c for n, c in caps.items())
    crumbs = Counter(m.region_of(d.pos[:2]) for d in _of("crumbs"))
    assert set(crumbs) <= {by_room["kitchen"], by_room["living room"]}
    assert crumbs[by_room["kitchen"]] <= 10 and crumbs[by_room["living room"]] <= 3 and sum(crumbs.values()) >= 10
    by = [f for f in m.furniture if f.kind in ("counter", "sink_unit", "stove", "dining_table", "coffee_table")]
    for d in _of("crumbs"):
        assert any(rect_distance_to_point(_rect(f), d.pos[:2]) <= 0.7 for f in by), d.pos
    assert all(d.size == (0.45, 0.1, 0.0015) for d in _of("scuff"))
    mats = [f for f in m.furniture if f.kind in ("rug", "floor_mat")]
    assert {f.id for f in mats} == {"F14", "F22", "F35"}
    for d, f in itertools.product(grime, mats):
        x0, y0, x1, y1 = _rect(f)
        assert not (d.box[0] < x1 and x0 < d.box[3] and d.box[1] < y1 and y0 < d.box[4]), (d.cls, d.pos, f.id)


def rect_distance_to_point(rect: tuple[float, float, float, float], p: tuple[float, float]) -> float:
    """The distance of a point from a rectangle (xmin, ymin, xmax, ymax), 0 inside it."""
    return math.hypot(max(rect[0] - p[0], 0.0, p[0] - rect[2]), max(rect[1] - p[1], 0.0, p[1] - rect[3]))
