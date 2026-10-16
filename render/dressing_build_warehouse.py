# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The warehouse's dressing builders, by name: the `BUILDERS` that `dressing.py` calls
for each recipe `dressing_layout_warehouse` emits.

Runs inside Blender's own Python and is never imported by `stbench`. A recipe `proc:<builder>[:<text>]` calls
`BUILDERS[builder](d, ctx, text)`, which returns the piece's one object, drawn in the Dressing's frame (x out of the
wall or rack face, y across, z up) to its size and to nothing beyond its box. An `asset:<id>` recipe whose model the
library lacks calls the builder of the class's name instead: those stand-ins are `fire_alarm`, `security_camera` and
`power_panel` (`cable_run` is all procedural conduit since the Poly Haven cable kit was dropped). The shell is
`dressing_warehouse_roof` and `dressing_warehouse_walls`, the docks `dressing_warehouse_dock`, the racks
`dressing_warehouse_racks`, the fittings `dressing_warehouse_fittings`. The table
is a literal so that a free test can read its names from the source.
"""

from __future__ import annotations

import dressing_warehouse_dock as dock
import dressing_warehouse_fittings as fittings
import dressing_warehouse_racks as racks
import dressing_warehouse_roof as roof
import dressing_warehouse_walls as walls

BUILDERS = {
    "steel_column": walls.steel_column,
    "roof_truss": roof.roof_truss,
    "purlin": roof.purlin,
    "roof_deck": roof.roof_deck,
    "skylight": roof.skylight,
    "air_duct": roof.air_duct,
    "cable_tray": roof.cable_tray,
    "sprinkler_pipe": roof.sprinkler_pipe,
    "sprinkler_head": roof.sprinkler_head,
    "high_bay_light": roof.high_bay_light,
    "wall_cladding": walls.wall_cladding,
    "wall_dado": walls.wall_dado,
    "safety_band": walls.safety_band,
    "conduit": walls.conduit,
    "dock_seal_pad": dock.dock_seal_pad,
    "dock_seal_head": dock.dock_seal_head,
    "dock_bumper": dock.dock_bumper,
    "dock_leveler": dock.dock_leveler,
    "dock_light": dock.dock_light,
    "door_number": dock.door_number,
    "upright_protector": racks.upright_protector,
    "end_guard": racks.end_guard,
    "beam_label": racks.beam_label,
    "aisle_sign": racks.aisle_sign,
    "fire_extinguisher": fittings.fire_extinguisher,
    "power_panel": fittings.power_panel,
    "utility_enclosure": fittings.utility_enclosure,
    "fire_alarm": fittings.fire_alarm,
    "security_camera": fittings.security_camera,
}
