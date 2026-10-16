# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The residential catalog's look, by region kind (`look.py`): data only, no `bpy`.

A street of detached houses and, held out, the courtyards of apartment blocks: outdoors, in the open, lit by the sun
and the sky alone like the downtown (the same picture of a sky, through the manifest's `skies["residential"]`). A region
is a `street` (carriageway, kerbs, sidewalks and verges), a `garden` (a plot's front and back lawns, its house and its
drive), a `car park`, a `green` (a common with a playground), a `passage` (a footpath between two plots) or a
`courtyard` (the design's own word for the held-out layout's yards; no public map uses it, so the vocabulary says
nothing of that layout).

The placeholder appearance has no grass (its floors are the downtown's two procedural ones: asphalt for the roads,
paving for everything a person walks on); the assets appearance takes each kind's floor from the manifest's `floors`
(grass for `garden` and `green`). Outdoors nothing is worn (`wear.apply` leaves an outdoor map as it is), so
`FLOOR_FINISH` is `None`; there are no ceiling lamps, so the rig has no lux and no colour temperature by kind.

`GROUND` and `SCENERY` are the two keys `build_scene` reads beyond the tables (a look without them gets the downtown's
asphalt and `scenery.py`): the material of the plane under and round the map, and the scenery module beyond its edge,
`scenery_<SCENERY>.py`.
"""

KINDS = frozenset({"street", "garden", "car park", "green", "passage", "courtyard"})
PLACEHOLDER_FLOOR = {"street": "asphalt", "car park": "asphalt", "garden": "paving", "green": "paving",
                     "passage": "paving", "courtyard": "paving"}
FLOOR_FINISH = None
DRESSING = "residential"
GROUND = "lawn"
SCENERY = "residential"
RIG = {  # the downtown's, value for value: the same sky, the same sun, no lamps
    "balance_k": None, "lux": None, "cct_k": None, "emitter": None, "sun": None,
    "hdri": {"camera_strength": 1.0, "light_strength": 5.2, "light_desaturation": 0.5,
             "sun_atan2_deg": -34.0, "sun_elevation_deg": 48.0},
    "sky": None,
    "portals": False,
}
