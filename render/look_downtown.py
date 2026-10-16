# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The downtown's look, by region kind (`look.py`): data only, no `bpy`.

The downtown's regions carry no `kind`, so a kind is a region's name. Its floors are the street's own (asphalt and
paving, never worn: `wear` leaves an outdoor map as it is), so `FLOOR_FINISH` is `None`; it has no ceiling lamps, so the
rig has no lux and no colour temperature by kind: the sun and the sky are all its light.
"""

KINDS = frozenset({"high street", "west street", "east street", "square"})
PLACEHOLDER_FLOOR = {"high street": "asphalt", "west street": "asphalt", "east street": "asphalt",
                     "square": "paving"}
FLOOR_FINISH = None
DRESSING = "downtown"
RIG = {
    "balance_k": None, "lux": None, "cct_k": None, "emitter": None, "sun": None,
    # the sky's light is what the procedural sky's was (5.2 against the Nishita sky's 0.55: both light a horizontal
    # surface with 5.2 W/m2, measured); the picture is turned and stretched so that its own sun (at atan2(y, x) of
    # -34 degrees and 48 degrees up, clamped to a small bright patch) lies where the map's sun lamp is. The
    # desaturation is `lighting_sky.SKY_LIGHT_DESAT` (a `bpy` module, which this file may not import: a test pins that
    # the two are one number)
    "hdri": {"camera_strength": 1.0, "light_strength": 5.2, "light_desaturation": 0.5,
             "sun_atan2_deg": -34.0, "sun_elevation_deg": 48.0},
    "sky": None,
    "portals": False,
}
