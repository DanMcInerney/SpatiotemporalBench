# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The warehouse's look, by region kind (`look.py`): data only, no `bpy`.

The dock carries no `kind` (its name is its kind) and the three aisles are of kind "aisle", whatever their names: a
job reads rack aisles by kind, and so does the render, so another warehouse's aisles need no row of their own. A
fulfilment hall's `pick area`, `pack area` and `staging area` (M4, step W) are the same slab under the same lamps. Its
floors have their own finish, the slab's, so `FLOOR_FINISH` is `None`, and its lamps are one high-bay fixture
(`fixture_lumens`) wherever they hang, so the rig has no lux by kind.
"""

KINDS = frozenset({"dock", "aisle", "pick area", "pack area", "staging area"})
PLACEHOLDER_FLOOR = {
    "dock": "concrete", "aisle": "concrete",
    "pick area": "concrete", "pack area": "concrete", "staging area": "concrete",
}
FLOOR_FINISH = None
DRESSING = "warehouse"
RIG = {
    "balance_k": 5000.0,
    # EN 12464-1 asks at least 150 lx on gangway floors and 200 lx on rack faces, in neutral white: lights over the
    # aisle give a rack face about 0.8 of what the floor gets, so 17,000 lm (a 130 W LED high-bay) meets the
    # faces on average (measured 200 lx; 115 to 250 lx from the lowest level to the highest) and the floor has
    # more than it needs (250 lx, at least 170)
    "lux": None,
    "cct_k": {"dock": 5000.0, "aisle": 5000.0, "pick area": 5000.0, "pack area": 5000.0, "staging area": 5000.0},
    "fixture_lumens": 17000.0,
    # the six skylights (3% of the roof) give the floor about 40 lx of daylight (27,000 lm over 344 m2 at a
    # utilisation of 0.5): a fifth of the lamps', where a real overcast day would give several times that
    "emitter": {"shape": "DISK", "size_m": 0.36, "spread_deg": 180.0, "z_max": 5.80, "drop_m": 0.002},
    "skylight": {"lumens": 27000.0, "cct_k": 5500.0}, "sun": None, "hdri": None, "sky": 0.0, "portals": False,
}
