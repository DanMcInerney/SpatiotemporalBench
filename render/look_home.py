# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The household's look, by region kind (`look.py`): data only, no `bpy`.

Every per-kind table has exactly `KINDS` as keys. `PLACEHOLDER_FLOOR` is the floor of the placeholder appearance
(`build_scene`), `FLOOR_FINISH` the finish the photoreal floor is worn with (`wear`), `WALL_PAINT` the multiplier of the
wall colour (`wear_surfaces`), `ROLE` what the room is for, which decides how it is dressed (`dressing_home_plan`) and
`RIG` the light (`lighting`). `DRESSING` names the layout module (`dressing_layout_<DRESSING>`).
"""

# The apartment's five kinds, then the larger homes' five: a dining room and a study dress as living rooms, a utility
# room as a kitchen, a corridor as a hall and a cloakroom as a bathroom, each with its own paint and light
KINDS = frozenset({"hall", "living room", "kitchen", "bedroom", "bathroom",
                   "dining room", "utility room", "study", "corridor", "cloakroom"})
PLACEHOLDER_FLOOR = {"hall": "tile", "kitchen": "tile", "bathroom": "tile_blue", "living room": "wood",
                     "bedroom": "carpet", "dining room": "wood", "utility room": "tile", "study": "carpet",
                     "corridor": "tile", "cloakroom": "tile"}
FLOOR_FINISH = {"hall": "tile", "kitchen": "tile", "bathroom": "tile", "living room": "wood", "bedroom": "carpet",
                "dining room": "wood", "utility room": "tile", "study": "carpet", "corridor": "tile",
                "cloakroom": "tile"}
# Paint by kind: the linear multiplier of the wall colour (a warm off-white). Some colour, in pale shades.
WALL_PAINT = {"hall": (1.0, 0.96, 0.88), "living room": (0.88, 0.95, 0.90), "kitchen": (1.0, 0.93, 0.72),
              "bedroom": (0.78, 0.85, 1.0), "bathroom": (0.72, 0.92, 0.95), "dining room": (0.95, 0.90, 0.82),
              "utility room": (0.92, 0.95, 0.92), "study": (0.88, 0.92, 0.96), "corridor": (1.0, 0.96, 0.88),
              "cloakroom": (0.72, 0.92, 0.95)}
ROLE = {"hall": "hall", "living room": "living", "kitchen": "kitchen", "bedroom": "bedroom", "bathroom": "bathroom",
        "dining room": "living", "utility room": "kitchen", "study": "living", "corridor": "hall",
        "cloakroom": "bathroom"}
DRESSING = "home"
RIG = {
    "balance_k": 3250.0,
    # work-plane lux by what the room is (PennState EGEE 102: living areas 100-300, kitchen tasks 300-750) and
    # the colour temperature of its lamp (PacLights: 2700-3000 K in living areas, 3000-3500 K kitchens and baths).
    # The hall is asked for 15% over the band's 100 lx floor: with 100 asked its lamps measured 96 lx (a long
    # narrow room, more wall than the lumen method's utilisation allows for)
    "lux": {"living room": 150.0, "bedroom": 120.0, "hall": 115.0, "kitchen": 300.0, "bathroom": 200.0,
            "dining room": 150.0, "utility room": 200.0, "study": 200.0, "corridor": 115.0, "cloakroom": 200.0},
    "cct_k": {"living room": 3000.0, "bedroom": 3000.0, "hall": 3000.0, "kitchen": 3500.0, "bathroom": 3500.0,
              "dining room": 3000.0, "utility room": 3500.0, "study": 3000.0, "corridor": 3000.0,
              "cloakroom": 3500.0},
    "emitter": {"shape": "SQUARE", "size_m": 0.36, "spread_deg": 180.0, "z_max": 2.40, "drop_m": 0.002},
    # a kitchen worktop wants 300 to 750 lx and the ceiling lamp alone, with the wall cabinets over the counters
    # shading them, gives about 230: a 4 cm LED strip under each wall cabinet, a quarter of its depth in from the
    # front, gives the worktop the rest (measured; `lighting_probe`). A strip's output is an LED tape's
    "under_cabinet": {"lumens_per_m": 500.0, "width_m": 0.04, "drop_m": 0.002, "front_share": 0.25},
    # the sun and the sky at 38 degrees up, from the CIE Standard Clear Sky at a luminous turbidity of 2
    # (`hdri_remap.clear_sky_lux(38.0, 2.0)`, rounded): 59,501 lx on a horizontal surface from the sun and 11,762 lx
    # from the sky; the sun at azimuth 141, south-east (10:00 solar time at 45 N near an equinox)
    "sun": {"elevation_deg": 38.0, "rotation_deg": 141.0, "lux": 59500.0},
    # the street is only ever the picture's columns 0.20 to 0.60: the rest holds a billboard, a station sign and a
    # speed limit, which a window anywhere in the apartment must never show; the picture is turned 18 degrees so
    # that the north, east and south windows (u 0.25, 0.50 and 0.75 unturned) show three different stretches of it
    "hdri": {"sky_lux": 11760.0, "heading_deg": -18.0, "band_u": (0.20, 0.60)},
    "sky": 0.3, "portals": True,
}
