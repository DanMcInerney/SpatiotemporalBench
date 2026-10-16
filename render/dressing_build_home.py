# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The household's dressing builders, by the name of the `proc:<builder>` recipe the layout gives a piece
(`dressing_layout_home`) and, for the four asset classes, by the class name the stand-in of a missing model is found
under (`dressing.draw`): `BUILDERS[name](d, ctx, text) -> [object]`.

Runs inside Blender's own Python and is never imported by `stbench`. The builders live in five modules by what they
draw, `dressing_build_trim` (trim and small fittings), `dressing_build_fixtures` (radiator, television, kitchen and
bathroom fixtures), `dressing_build_soft` (curtains and blinds), `dressing_build_standins` (frames, clock, lamps and
fan) and `dressing_build_grime` (dust, hair, crumbs and scuffs on the floor); this table is the one list, a literal
one so that a test can read its keys without Blender.
"""

from __future__ import annotations

import dressing_build_fixtures as fixtures
import dressing_build_grime as grime
import dressing_build_soft as soft
import dressing_build_standins as standins
import dressing_build_trim as trim

BUILDERS = {
    "baseboard": trim.baseboard, "door_casing": trim.door_casing, "window_sill": trim.window_sill,
    "outlet": trim.outlet, "light_switch": trim.light_switch, "thermostat": trim.thermostat,
    "smoke_detector": trim.smoke_detector, "radiator": fixtures.radiator, "curtain": soft.curtain,
    "roller_blind": soft.roller_blind, "wall_tv": fixtures.wall_tv, "upper_cabinet": fixtures.upper_cabinet,
    "range_hood": fixtures.range_hood, "backsplash": fixtures.backsplash, "bathroom_cabinet": fixtures.bathroom_cabinet,
    "toilet_roll_holder": fixtures.toilet_roll_holder, "shower_curtain": soft.shower_curtain,
    "picture_frame": standins.picture_frame, "wall_clock": standins.wall_clock, "ceiling_lamp": standins.ceiling_lamp,
    "ceiling_fan": standins.ceiling_fan, "dust_line": grime.dust_line, "hair": grime.hair, "crumbs": grime.crumbs,
    "scuff": grime.scuff,
}
