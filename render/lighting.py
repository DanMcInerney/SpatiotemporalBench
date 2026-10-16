# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The light of the photoreal appearance: lamps with real power and colour,
daylight through the windows at its physical intensity, and a fixed white balance for each catalog.

Runs inside Blender's own Python and is never imported by `stbench`. `setup(sc, m, spec, app)` is called by
`build_scene.build` for `--appearance assets`, after the structure and the dressing exist; it makes the sky, the sun and
every lamp (`lighting_sky` draws the sky) and sets the colour management. The placeholder appearance never gets here.

Units. Cycles' watts are radiometric: a Lambertian area light of P watts emits P watts (measured by
`tests/test_render_lighting_blender.py`). A render unit of irradiance is `UNIT_LUX / pi` lux, so that a grey card
(albedo 0.18) lit with `UNIT_LUX` lux reads 0.18, in every catalog (`cycles_setup.UNIT_LUX`, the one owner: 240). It is
no longer any camera's exposure: the camera model (`stbench.camera_model`) meters each frame, so the light is the light
a place really has and the range it makes (a sunlit floor reads about 75) is the camera's to take. A lamp of `lm` lumens
is `lm * pi / UNIT_LUX` watts. Its `lm` follows from the lux its room should have by the lumen method
`lm = lux * area / (LUMEN_UTILISATION * lamps in the room)`, and `tests/test_render_lighting_blender.py` measures the
lux the rooms really have with grey cards (`lighting_probe`).

Daylight (home) is at its physical intensity, from the CIE Standard Clear Sky (`hdri_remap.clear_sky_lux`: Kittler and
Darula 2002, ISO 15469 / CIE S 011 type 12) with a luminous turbidity of 2 at the sun's 38 degrees: the sun gives a
horizontal floor 59,500 lx and the sky 11,760 lx. The sun is at azimuth 141 degrees (south-east), 38 degrees up: 10:00
local solar time at 45 N near an equinox (NOAA solar calculator), inside the scripts' 07:00 to 10:00 clocks. The sky is
the manifest's HDRI at one strength for the camera and the light, set so that its horizontal illuminance, with its sun
disc clamped, is the sky's 11,760 lx (`lighting_sky`). There is no glass to take the usual fifth of the daylight, as the
windows are open holes (a stated simplification: the camera's exposure absorbs it).

White balance. A lamp has its colour temperature (3000 K in a living room), and the camera of a catalog is balanced to
one temperature, `balance_k`, so that the lamps come out near neutral and the catalog colours hold (VISION: the
colour attribute is what the pixels show). The balance is a gain per channel on each source's colour (`balanced`), not
Blender's view-transform white balance: that one adapts in a cone space and turns saturated colours by 10 to 15 degrees
even where the light is neutral (measured, DECISIONS "Realism Phase 1, light"). With a lamp 250 K off the balance every
palette colour stays within a Delta E of 10 of itself (the nearest two palette colours are 27 apart). Daylight is
neutral: the sun is white and the sky is the HDRI as it is (outdoors its light is paler than the sky one sees, as it
always was). Windows are open holes (no glass: depth through one is "none", as the RGB shows) with a Cycles portal in
each, so that the sky lights the room instead of being sampled blindly through a hole.
"""

from __future__ import annotations

import math
from pathlib import Path

import bpy
import hdri_remap
import lighting_sky
import look
from cycles_setup import UNIT_LUX
from mathutils import Matrix, Vector

LUMEN_UTILISATION = 0.75  # small rooms, white ceiling, pale walls: the share of a lamp's lumens on the work plane
LUMA = (0.2126, 0.7152, 0.0722)
# The rig of a catalog (`MapSpec.catalog`), its white balance, lamps and daylight, is `look_<catalog>.RIG`.


def render_watts(lumens: float) -> float:
    """Blender watts of a Lambertian lamp of `lumens`, in the linear unit: a grey card reads 0.18 at `UNIT_LUX` lux."""
    return lumens * math.pi / UNIT_LUX


def region_area(polygon) -> float:
    pts = [tuple(p) for p in polygon]
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1], strict=True))) / 2


def kelvin_rgb(kelvin: float) -> tuple[float, float, float]:
    """Blender's linear colour of a light of `kelvin` (luminance 1), read back from a scratch light."""
    ld = bpy.data.lights.new("kelvin", "AREA")
    ld.use_temperature, ld.temperature = True, kelvin
    rgb = tuple(ld.temperature_color)
    bpy.data.lights.remove(ld)
    return rgb


def balanced(kelvin: float, balance_k: float) -> tuple[float, float, float]:
    """The factor per channel on the colour of a source of `kelvin` so that it reads as `balance_k` would make it
    (a source at the balance is white), with luminance 1 for the source's own colour times it, so that the lumens
    stay lumens."""
    src, ref = kelvin_rgb(kelvin), kelvin_rgb(balance_k)
    inv = [1.0 / c for c in ref]
    luma = sum(w * s * i for w, s, i in zip(LUMA, src, inv, strict=True))
    return tuple(i / luma for i in inv)


def _facing(ob, direction: Vector) -> None:
    """Turn an area light so that it emits along `direction` (it shines down its local -z), its local y up."""
    z = -direction.normalized()
    up = Vector((0, 0, 1)) if abs(z.z) < 0.999 else Vector((0, 1, 0))
    x = up.cross(z).normalized()
    ob.rotation_euler = Matrix((x, z.cross(x), z)).transposed().to_euler()


def _area_light(sc, name: str, cfg: dict, pos, watts: float, cct_k: float, balance_k: float | None):
    ld = bpy.data.lights.new(name, "AREA")
    ld.shape, ld.size, ld.energy, ld.spread = cfg["shape"], cfg["size_m"], watts, math.radians(cfg["spread_deg"])
    ld.use_temperature, ld.temperature = True, cct_k  # the lamp's own colour, and the camera's balance on top of it
    if balance_k is not None:
        ld.color = balanced(cct_k, balance_k)
    ob = bpy.data.objects.new(name, ld)
    sc.collection.objects.link(ob)
    ob.location = pos
    _facing(ob, Vector((0, 0, -1)))
    return ob


def per_kind(m, rig: dict, key: str, region) -> float:
    """The rig's `key` (`lux` or `cct_k`) for a region, by its kind. There is no default: a catalog whose lamps need no
    such table declares it `None`, and a kind the table lacks is a `look.LookError`, not a lamp of some usual lux."""
    table, kind = rig[key], look.kind_of(region)
    if table is None or kind not in table:
        raise look.LookError(f"the {m.catalog} rig has no {key} for region kind {kind!r}; "
                             + ("it declares none" if table is None else f"its kinds are {sorted(table)}"))
    return table[kind]


def lamps(sc, m, rig: dict) -> list[dict]:
    """One area light per map ceiling light, just under the lens or diffuser of the fixture the dressing builds there
    (at most `z_max`), with the power and colour temperature of the room it lights."""
    out, per_room, e = [], {}, rig["emitter"]
    for lt in m.ceiling_lights:
        per_room[lt.region] = per_room.get(lt.region, 0) + 1
    for i, lt in enumerate(m.ceiling_lights):
        room = m.region(lt.region)
        if "fixture_lumens" in rig:
            lumens = rig["fixture_lumens"]
        else:
            lux = per_kind(m, rig, "lux", room)
            lumens = lux * region_area(room.polygon) / (LUMEN_UTILISATION * per_room[lt.region])
        cct = per_kind(m, rig, "cct_k", room)
        watts = render_watts(lumens)
        x, y, z = lt.pos
        pos = (x, y, min(z, e["z_max"]) - e["drop_m"])
        _area_light(sc, f"ceiling_light_{i}", e, pos, watts, cct, rig["balance_k"])
        out.append({"name": f"ceiling_light_{i}", "room": room.name, "lumens": round(lumens), "watts": round(watts, 2),
                    "cct_k": cct, "pos": [x, y, round(pos[2], 3)]})
    return out


def under_cabinet_lights(sc, m, rig: dict) -> list[dict]:
    """A strip light under each wall cabinet the dressing built (`dressing_upper_cabinet_*`, from its bounding box): as
    long as the cabinet, `width_m` across, `drop_m` under its underside, `front_share` of its depth in from the side
    that faces the room's middle, with `lumens_per_m` of the colour temperature of its room. The room a cabinet is in is
    the one at its middle."""
    cfg, out = rig.get("under_cabinet"), []
    cabinets = sorted((o for o in bpy.data.objects if o.name.startswith("dressing_upper_cabinet_")),
                      key=lambda o: o.name)
    if cabinets and cfg:
        bpy.context.view_layer.update()  # the objects' world matrices, which a script has not made yet
    for i, ob in enumerate(cabinets if cfg else ()):
        corners = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
        lo, hi = (Vector(f(v[k] for v in corners) for k in range(3)) for f in (min, max))
        centre, extent = (lo + hi) / 2, hi - lo
        along_x = extent.x >= extent.y
        length, depth = (extent.x, extent.y) if along_x else (extent.y, extent.x)
        room_id = m.region_of((centre.x, centre.y))
        if room_id is None:  # a cabinet in no room has no room to light
            continue
        region = m.region(room_id)
        poly = [tuple(p) for p in region.polygon]
        middle = (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))
        shift = math.copysign(cfg["front_share"] * depth, (middle[1] - centre.y) if along_x else (middle[0] - centre.x))
        pos = (centre.x, centre.y + shift, lo.z - cfg["drop_m"]) if along_x else (
            centre.x + shift, centre.y, lo.z - cfg["drop_m"])
        lumens = cfg["lumens_per_m"] * length
        cct = per_kind(m, rig, "cct_k", region)
        light = _area_light(sc, f"under_cabinet_light_{i}", {"shape": "RECTANGLE", "size_m": length if along_x
                                                              else cfg["width_m"], "spread_deg": 180.0},
                            pos, render_watts(lumens), cct, rig["balance_k"])
        light.data.size_y = cfg["width_m"] if along_x else length
        out.append({"name": light.name, "room": region.name, "lumens": round(lumens), "cct_k": cct,
                    "pos": [round(v, 3) for v in pos], "length_m": round(length, 3)})
    return out


def portals(sc, m) -> int:
    """A Cycles portal in each window: an area light the size of the opening, on the wall's plane, facing into the
    building (towards the map's centre), which only guides the sampling of the sky."""
    xmin, ymin, xmax, ymax = m.bounds
    centre = Vector(((xmin + xmax) / 2, (ymin + ymax) / 2, 0.0))
    for x in m.windows:
        w = m.wall(x.wall)
        cx, cy = x.center
        inward = Vector((-w.direction[1], w.direction[0], 0.0))
        if (centre - Vector((cx, cy, 0.0))).dot(inward) < 0:
            inward = -inward
        ld = bpy.data.lights.new(f"portal_{x.id}", "AREA")
        ld.shape, ld.size, ld.size_y = "RECTANGLE", x.width, x.height
        ld.cycles.is_portal = True
        ob = bpy.data.objects.new(f"portal_{x.id}", ld)
        sc.collection.objects.link(ob)
        ob.location = (cx, cy, x.sill + x.height / 2)
        _facing(ob, inward)
    return len(m.windows)


def daylight_panels(rig: dict) -> int:
    """Skylights: slot 0 of every `dressing_skylight_*` object (the pane; slot 1 is its frame) becomes a daylight panel,
    `skylight.lumens` in all, shared by the panes' undersides. The pane is the light, so it is an emission and nothing
    else. Returns how many panes there are."""
    sk = rig.get("skylight")
    panels = [ob for ob in bpy.data.objects if ob.name.startswith("dressing_skylight_")]
    if not sk or not panels:
        return 0
    area = sum(ob.dimensions.x * ob.dimensions.y for ob in panels)
    mat = bpy.data.materials.new("daylight_panel")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    emit, out = nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
    emit.inputs["Color"].default_value = (*balanced(sk["cct_k"], rig["balance_k"]), 1.0)
    emit.inputs["Strength"].default_value = render_watts(sk["lumens"]) / (math.pi * area)
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    for ob in panels:
        ob.material_slots[0].material = mat
    return len(panels)


def _hdri_file(app, catalog: str) -> Path | None:
    """The catalog's HDRI in the asset cache (None: the manifest names none, or there is no asset library)."""
    if app is None:
        return None
    from stbench import asset_manifest  # not at the top: a script puts `stbench` on the path only when it starts

    path = asset_manifest.hdri_path(app.m, catalog)
    return path if path is not None and Path(path).exists() else None


def setup(sc, m, spec: dict, app) -> dict:
    """Light the scene of an `assets` render and return what was done (it goes into `render_meta.json`). `spec` is the
    map's raw dict; `app` the asset `Appearance`."""
    rig = look.module(m.catalog).RIG
    sun = rig["sun"] or ({"elevation_deg": m.outdoor.sun_elevation_deg, "rotation_deg": m.outdoor.sun_rotation_deg,
                          "strength": m.outdoor.sun_strength, "sky": m.outdoor.sky_strength} if m.outdoor else None)
    record: dict = {"rig": m.catalog, "unit_lux": UNIT_LUX, "balance_k": rig["balance_k"]}
    hdri_path = _hdri_file(app, m.catalog) if rig["hdri"] else None
    if sun is None:  # a closed building: nothing sees a sky, so the world is black and costs no samples
        lighting_sky.build(sc, {"elevation_deg": 90.0, "rotation_deg": 0.0, "strength": 0.0, "sky": 0.0})
        record["world"] = "none"
    else:
        strength = sun["strength"] if "strength" in sun else \
            sun["lux"] * math.pi / UNIT_LUX / math.sin(math.radians(sun["elevation_deg"]))
        cfg = {"elevation_deg": sun["elevation_deg"], "rotation_deg": sun["rotation_deg"], "strength": strength,
               "sky": rig["sky"] if rig["sky"] is not None else sun["sky"]}
        hdri = hdri_remap.aimed_at_sun(rig["hdri"] or {}, cfg["rotation_deg"], cfg["elevation_deg"])
        record["world"] = lighting_sky.build(sc, cfg, m.outdoor, hdri_path, hdri)
        record["sun"] = {"elevation_deg": cfg["elevation_deg"], "rotation_deg": cfg["rotation_deg"],
                         "watts_per_m2": round(strength, 3), "lux": sun.get("lux")}
        record["hdri"] = hdri_path.name if hdri_path else None
        if hdri_path and "sky_lux" in hdri:  # the one strength that gives a horizontal surface the sky's lux
            record["sky"] = {"lux": hdri["sky_lux"], "strength": round(hdri["camera_strength"], 3)}
    if rig["emitter"] and m.ceiling_lights:
        record["lamps"] = lamps(sc, m, rig)
    record["under_cabinet"] = under_cabinet_lights(sc, m, rig)
    if rig["portals"]:
        record["portals"] = portals(sc, m)
    record["skylights"] = daylight_panels(rig)
    vs = sc.view_settings
    vs.exposure, vs.use_white_balance = 0.0, False  # the balance is in the sources' colours (see the docstring)
    return record
