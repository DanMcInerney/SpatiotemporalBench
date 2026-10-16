# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Map spec + timeline -> a Blender scene, and per-time updates of that scene.

Runs inside Blender's own Python; `stbench` never imports this file. Blender's world is the map frame: metres, x east,
y north, z up. The structure (walls with openings cut out, floors, ceilings, window frames, fixtures, ground) is built
once and carries `pass_index` 0. Every timeline entity is one object whose `pass_index` is `instance_map[...]`'s key;
`apply_time` moves, shows or hides each one for a time t, and `apply_camera` sets the camera from an ego keyframe.
Appearance is `placeholder` (M1: procedural placeholders, plain palette materials) or `assets` (`asset_entities`:
carried, light, mover and clutter entities from the asset library, textured surfaces; the structure keeps its M1
geometry). Clutter, which has dozens of kinds, is a box of its size and colour in the placeholder appearance
(`items.clutter_box`), so `build` never raises on a kind.
A mover is an ordinary entity to `apply_time`: its placement at each keyframe's time sets where it stands. In the assets
appearance `build` also builds the map's render-only dressing (`dressing.build`, never an entity), wears the fixed
surfaces (`wear.apply`) and lights the scene (`lighting.setup`: lamps inside the dressing's fixtures, portals in the
windows, the sky and the sun; no ground plane under an indoor map and no `fixture_i` panels); the placeholder
appearance keeps its ground, panels, sky and sun exactly as they were.
An outdoor map (the downtown) has no ceilings, underlay or ceiling lights: it has the map's sun and a procedural sky,
its walls drawn as building facades (`facades`), asphalt and paving floors, the city beyond its open street ends as
static scenery (`scenery`, never an entity) and a camera that sees 1,000 m. A catalog's look may name another ground
under it and another scenery module (`GROUND`, `SCENERY`: the suburb's lawn and `scenery_residential`); the downtown's
look names neither, and its scene is as it was. An indoor map builds exactly as it always did. """

from __future__ import annotations

import importlib.util
import json
import math

import bpy
import facades
import floor_look
import furniture
import furniture_city
import furniture_residential
import furniture_warehouse
import items
import lighting
import lighting_sky
import look
import materials
import materials_city
import shapes
import wear
from asset_entities import Appearance
from mathutils import Matrix
from shapes import Part, box, emission_material, entity_mesh, link, material, polygon_mesh

CLIP_END_M = {False: 200.0, True: 1000.0}  # by `outdoor`: the farthest the camera sees
OUTDOOR_GROUND_M = 150.0  # the ground reaches this far past the scenery (`scenery_layout.REACH_M` past the map)
CITY_FLOORS = ("asphalt", "paving")  # the downtown's: one material per kind, shared by the regions that name it
BY_POSE = {**furniture_warehouse.BUILDERS, **furniture_city.BUILDERS}  # drawn from the surfaces of their own pose
# camera body frame (x forward, y left, z up) -> Blender camera frame (x right, y up, -z forward), as columns
BODY_FROM_BLENDER_CAM = Matrix(((0, 0, -1), (-1, 0, 0), (0, 1, 0)))
ITEM_MOBILITIES = ("carried", "mover")  # drawn by `items.BUILDERS`; light and fixed furniture by `furniture.BUILDERS`
ASSET_MOBILITIES = ("carried", "light", "mover", "clutter")  # entities that come from the asset library in assets mode


# -- structure ---------------------------------------------------------------------------------------------------
def _wall_boxes(m) -> list[Part]:
    """Solid boxes of every wall with door, arch and window openings removed (a lintel stays above doors)."""
    parts: list[Part] = []
    for w in m.walls:
        if w.facade:  # a building's face: `facades` draws it
            continue
        holes = [(o.at - o.width / 2, o.at + o.width / 2, 0.0, o.height) for o in m.openings if o.wall == w.id]
        holes += [(x.at - x.width / 2, x.at + x.width / 2, x.sill, x.sill + x.height) for x in m.windows
                  if x.wall == w.id]
        t = w.thickness
        cuts = sorted({-t / 2, w.length + t / 2, *(h[0] for h in holes), *(h[1] for h in holes)})
        for a, b in zip(cuts, cuts[1:], strict=False):
            if b - a < 1e-6:
                continue
            spans = sorted((z0, z1) for (h0, h1, z0, z1) in holes if h0 <= a + 1e-9 and h1 >= b - 1e-9)
            z, solids = 0.0, []
            for z0, z1 in spans:
                if z0 > z:
                    solids.append((z, z0))
                z = max(z, z1)
            if z < w.height:
                solids.append((z, w.height))
            cx, cy = w.point_at((a + b) / 2)
            for z0, z1 in solids:
                parts.append(Part("box", (cx, cy, (z0 + z1) / 2), (b - a, t, z1 - z0), (0, 0, w.yaw), "wall"))
    return parts


def _window_frames(m) -> list[Part]:
    """A frame of four bars around each window opening and a mullion; no glass (M1)."""
    parts: list[Part] = []
    for x in m.windows:
        w = m.wall(x.wall)
        bar, depth = 0.05, w.thickness + 0.02
        z0, z1 = x.sill, x.sill + x.height
        cx, cy = x.center
        specs = [(0.0, z0 + bar / 2, x.width, bar), (0.0, z1 - bar / 2, x.width, bar),
                 (-x.width / 2 + bar / 2, (z0 + z1) / 2, bar, x.height),
                 (x.width / 2 - bar / 2, (z0 + z1) / 2, bar, x.height),
                 (0.0, (z0 + z1) / 2, bar * 0.6, x.height)]
        ux, uy = w.direction
        for along, zc, length, height in specs:
            parts.append(Part("box", (cx + ux * along, cy + uy * along, zc), (length, depth, height), (0, 0, w.yaw),
                              "frame"))
    return parts


def _floor_material(kind: str) -> bpy.types.Material:
    if kind == "concrete":  # the procedural slab of `materials`, the same in both appearances
        return materials.concrete_material()
    if kind in CITY_FLOORS:  # the downtown's street and paving, the same in both appearances
        return getattr(materials_city, kind)()
    mat = bpy.data.materials.new(f"floor_{kind}")
    mat.use_nodes = True
    nt = mat.node_tree
    p = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    p.inputs["Roughness"].default_value = 0.35 if kind.startswith("tile") else 0.6
    if kind.startswith("tile"):
        tex = nt.nodes.new("ShaderNodeTexChecker")
        tex.inputs["Scale"].default_value = 2.0  # 0.5 m tiles
        a, b = ((0.62, 0.62, 0.60), (0.48, 0.48, 0.47)) if kind == "tile" else ((0.45, 0.60, 0.68), (0.35, 0.50, 0.60))
        tex.inputs["Color1"].default_value = (*(c ** 2.2 for c in a), 1)
        tex.inputs["Color2"].default_value = (*(c ** 2.2 for c in b), 1)
        nt.links.new(coord.outputs["Object"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Color"], p.inputs["Base Color"])
    elif kind == "wood":
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (1.0, 5.0, 1.0)
        tex = nt.nodes.new("ShaderNodeTexWave")
        tex.bands_direction = "Y"
        tex.inputs["Scale"].default_value = 2.5
        tex.inputs["Distortion"].default_value = 2.0
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (0.30, 0.17, 0.07, 1)
        ramp.color_ramp.elements[1].color = (0.45, 0.27, 0.12, 1)
        nt.links.new(coord.outputs["Object"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], p.inputs["Base Color"])
    else:  # carpet
        tex = nt.nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = 40.0
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (0.40, 0.31, 0.22, 1)
        ramp.color_ramp.elements[1].color = (0.50, 0.40, 0.29, 1)
        nt.links.new(coord.outputs["Object"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], p.inputs["Base Color"])
        p.inputs["Roughness"].default_value = 0.95
    return mat


def _scenery_module(catalog: str):
    """The module that draws the scenery beyond an outdoor map's edge: `scenery_<SCENERY>` where the catalog's look
    names one, else the downtown's `scenery`. Imported here, not at the top: it reads `stbench`, which a script puts on
    the path only when it starts."""
    name = getattr(look.module(catalog), "SCENERY", None)
    return importlib.import_module("scenery" if name is None else f"scenery_{name}")


def _ground_material(catalog: str, road: bpy.types.Material) -> bpy.types.Material:
    """The material of the plane under and round an outdoor map: the shader the catalog's look names (`GROUND`, one of
    `materials.SHADERS`), else the downtown's asphalt (`road`)."""
    name = getattr(look.module(catalog), "GROUND", "asphalt")
    return road if name == "asphalt" else materials.shader_material(name)


def _build_structure(m, app=None) -> None:
    resolve = app.resolver() if app else None
    if parts := _wall_boxes(m) + _window_frames(m):
        link("walls", entity_mesh("walls", parts, resolve))
    xmin, ymin, xmax, ymax = m.bounds
    kinds: dict[str, bpy.types.Material] = {}
    city_floors: dict[str, bpy.types.Material] = {}
    if m.outdoor is None:
        under = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        link("underlay", polygon_mesh("underlay", under, -0.002, material("underlay")))
        if app is None:  # the photoreal sky has no ground under it: depth through a window is "none", as the RGB shows
            link("ground", polygon_mesh("ground", [(-300, -300), (300, -300), (300, 300), (-300, 300)], -0.05,
                                        material("ground")))
    for r in m.regions:
        poly = [tuple(p) for p in r.polygon]
        region_kind = look.kind_of(r)
        kind = look.get(m.catalog, "PLACEHOLDER_FLOOR", region_kind)  # what the placeholder floor is made of
        floor = app.floor_material(region_kind) if app else None
        if floor is None:
            floor = city_floors[kind] if kind in city_floors else kinds.setdefault(kind, _floor_material(kind))
        if kind in CITY_FLOORS:
            city_floors.setdefault(kind, floor)
        ob = link(f"floor_{r.id}", polygon_mesh(f"floor_{r.id}", poly, r.z0, floor))
        if floor_look.is_displaced(floor):  # the tile and wood floors have true relief, diced near the camera
            floor_look.subdivide(ob)
        if m.outdoor is None:
            link(f"ceiling_{r.id}", polygon_mesh(f"ceiling_{r.id}", poly, r.z1, material("ceiling", resolve)))
    if m.outdoor is not None:
        scene_mod = _scenery_module(m.catalog)
        road = city_floors.get("asphalt") or materials_city.asphalt()
        ground = _ground_material(m.catalog, road)
        g = scene_mod.layout.REACH_M + OUTDOOR_GROUND_M
        link("ground", polygon_mesh("ground", [(xmin - g, ymin - g), (xmax + g, ymin - g), (xmax + g, ymax + g),
                                               (xmin - g, ymax + g)], -0.05, ground))
        mats = facades.Materials()
        facades.build(m, mats)
        scene_mod.build(m, ground, mats)
    if app is not None:  # the photoreal lamps are `lighting`'s, each inside the dressing's lamp
        return
    glow = emission_material("fixture_glow", 2.0) if m.ceiling_lights else None  # outdoors nothing uses it
    for i, lt in enumerate(m.ceiling_lights):
        x, y, z = lt.pos
        panel = entity_mesh(f"fixture_{i}", [box(x - 0.3, x + 0.3, y - 0.3, y + 0.3, z + 0.03, z + 0.05, "fixture")])
        panel.materials.clear()
        panel.materials.append(glow)
        link(f"fixture_{i}", panel)
        ld = bpy.data.lights.new(f"ceiling_light_{i}", "AREA")
        ld.shape, ld.size, ld.energy = "SQUARE", 0.6, lt.watts  # the map's brightness (absent: the apartment's 45 W)
        ob = bpy.data.objects.new(f"ceiling_light_{i}", ld)
        bpy.context.scene.collection.objects.link(ob)
        ob.location = (x, y, z)  # an area light emits towards -z


def _build_sky(sc, outdoor=None) -> None:
    """The procedural sky and the sun of the placeholder appearance and of the asset preview: the indoor map's (light
    through its windows) or, outdoors, the map's own (`outdoor`). The photoreal appearance lights its scenes with
    `lighting.setup`."""
    cfg = lighting_sky.SUN if outdoor is None else {"elevation_deg": outdoor.sun_elevation_deg,
                                                    "rotation_deg": outdoor.sun_rotation_deg,
                                                    "strength": outdoor.sun_strength, "sky": outdoor.sky_strength}
    lighting_sky.build(sc, cfg, outdoor)


# -- doors -------------------------------------------------------------------------------------------------------
class Door:
    """A door leaf: an object whose origin is the hinge, turning a quarter between closed and open."""

    def __init__(self, ob, hinge, closed_dir, open_dir, toward_opening):
        self.ob, self.hinge, self.toward = ob, hinge, toward_opening
        self.closed_dir, self.open_dir = closed_dir, open_dir

    def set_open(self, is_open: bool) -> None:
        d = self.open_dir if is_open else self.closed_dir
        self.ob.rotation_euler = (0.0, 0.0, math.atan2(d[1], d[0]))
        off = 0.02 if is_open else 0.0  # an open leaf sits flush against the jamb face
        self.ob.location = (self.hinge[0] + self.toward[0] * off, self.hinge[1] + self.toward[1] * off, 0.0)


def _door_parts(width: float, thickness: float, height: float, mat: str) -> list[Part]:
    """The leaf with its hinge at the local origin and its free edge at +x."""
    return [box(0, width, -thickness / 2, thickness / 2, 0, height, mat),
            Part("sphere", (width - 0.07, thickness / 2 + 0.03, 1.0), (0.06, 0.06, 0.06), (0, 0, 0), "steel"),
            Part("sphere", (width - 0.07, -thickness / 2 - 0.03, 1.0), (0.06, 0.06, 0.06), (0, 0, 0), "steel")]


# -- the scene ---------------------------------------------------------------------------------------------------
class Scene:
    def __init__(self, tl, m):
        self.tl, self.m = tl, m
        self.objects: dict[str, bpy.types.Object] = {}
        self.doors: dict[str, Door] = {}
        self.instance_map: dict[int, str] = {}
        self.camera: bpy.types.Object | None = None
        self.lighting: dict | None = None  # what `lighting.setup` did (assets appearance), for the render log

    def apply_time(self, t: float) -> None:
        """Show each entity where its placement at t puts it (or hide it); set door leaves from their open state."""
        for eid, ob in self.objects.items():
            p = self.tl.placement_of(eid, t)
            ob.hide_render = ob.hide_viewport = p is None
            if p is None:
                continue
            if eid in self.doors:
                self.doors[eid].set_open(bool(self.tl.state_of(eid, t, "open")))
            else:
                ob.location = p.pos
                ob.rotation_euler = (0.0, 0.0, p.yaw)

    def apply_camera(self, kf) -> None:
        """Blender cameras look down their local -z with +y up; the keyframe's body frame has x forward, z up."""
        r_body = Matrix(_quat_matrix(kf.rot))
        cam_rot = r_body @ BODY_FROM_BLENDER_CAM
        mat = Matrix.Translation(kf.pos) @ cam_rot.to_4x4()
        self.camera.matrix_world = mat


def _quat_matrix(rot):
    from mathutils import Quaternion
    return Quaternion(rot).to_matrix()


def _surface_heights(m) -> dict[str, tuple[float, ...]]:
    return {f.name: tuple(sorted(s.z for s in f.surfaces)) for f in m.furniture if f.surfaces}


def _surfaces_at(m, kind: str, pos) -> tuple[tuple[float, float, float, float, float], ...]:
    """The receptacle surfaces `(cx, cy, sx, sy, z)` of the map furniture of this kind standing at `pos`: a piece is
    found by where it stands, as a door is, because six racks share one name and differ only in which bay-levels have a
    surface. Raises LookupError for a piece the map does not have there."""
    for f in m.furniture:
        if f.kind == kind and math.dist(f.center, pos[:2]) < 1e-3:
            return tuple((*s.rect, s.z) for s in f.surfaces)
    raise LookupError(f"the map {m.name} has no {kind} at {pos[:2]}")


def _raw_spec(m) -> dict:
    """The map's own file as a dict, which is what the dressing layouts and the lamps read."""
    from stbench.map_files import map_bytes  # not at the top: a script puts `stbench` on the path only when it starts

    return json.loads(map_bytes(m.name).decode("utf-8"))


def _dress(spec: dict, m, app: Appearance) -> None:
    """Build the map's render-only dressing (`dressing.build`, lane BH's dispatcher) once, after the structure."""
    if importlib.util.find_spec("dressing") is None:  # not written yet: the scene is the bare structure
        return
    import dressing

    dressing.build(spec, m.catalog, dressing.Ctx(app))


def build(tl, m, hfov_deg: float = 80.0, assets=None) -> Scene:
    """Build everything for this episode. `assets` is None (placeholders) or `(manifest, library .blend path)`.
    Raises ValueError if an entity kind has no placeholder geometry (clutter has none to lack), or, in assets mode, if a
    carried, light, mover or clutter entity has no asset or does not fit its asset."""
    sc = bpy.context.scene
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.node_groups):
        for x in list(coll):
            coll.remove(x)
    shapes.reset()
    app = None
    if assets is not None:
        wanted = [e.asset for e in tl.entities if e.mobility in ASSET_MOBILITIES]
        if None in wanted:
            raise ValueError("assets mode needs an asset on every carried, light, mover and clutter entity; this "
                             "timeline was generated without them")
        app = Appearance(assets[0], assets[1], wanted)
    spec = _raw_spec(m) if app is not None else None
    exact = m.catalog in wear.LOOK_CATALOGS  # the household's furniture gets each part's own faces (`shapes._add_part`)
    materials.use_map(spec if m.outdoor is None else None)  # the warehouse slab carries its floor markings (assets)
    _build_structure(m, app)
    if app is None:
        _build_sky(sc, m.outdoor)
    scene = Scene(tl, m)
    zs = _surface_heights(m)
    city = furniture_city.Surfaces()
    residential = furniture_residential.Surfaces(city)
    openings = {(round(o.center[0], 3), round(o.center[1], 3)): o for o in m.doors()}
    for i, e in enumerate(tl.entities):
        idx = i + 1
        scene.instance_map[idx] = e.id
        if e.kind == "door":
            p0 = tl.placement_of(e.id, 0.0) or next(p for p in tl.placements if p.entity == e.id)
            o = openings[(round(p0.pos[0], 3), round(p0.pos[1], 3))]
            mat = "door" if o.interior else "door_entrance"
            me = entity_mesh(f"E_{idx}", _door_parts(e.size[0], e.size[1], e.size[2], mat),
                             app.resolver() if app else None, exact)
            ob = link(f"door_{o.id}", me, idx)
            hinge, free = m.door_leaf(o, False)
            ux, uy = (free[0] - hinge[0]) / o.width, (free[1] - hinge[1]) / o.width
            _, tip = m.door_leaf(o, True)
            ox, oy = (tip[0] - hinge[0]) / o.width, (tip[1] - hinge[1]) / o.width
            scene.doors[e.id] = Door(ob, hinge, (ux, uy), (ox, oy), (ux, uy))
        elif app is not None and e.mobility in ASSET_MOBILITIES:
            colour = str(e.attributes.get("color", "grey"))
            ob = app.make_object(e.asset, e.kind, colour, e.size, f"{e.kind}_{idx}", idx)
        elif e.mobility == "clutter":
            color = str(e.attributes.get("color", "grey"))
            ob = link(f"{e.kind}_{idx}", entity_mesh(f"E_{idx}", items.clutter_box(tuple(e.size), color)), idx)
        else:
            color = str(e.attributes.get("color", "grey"))
            if e.mobility in ITEM_MOBILITIES:
                table = items.BUILDERS
            else:
                table = {**furniture.BUILDERS, **furniture_warehouse.BUILDERS, **furniture_residential.BUILDERS}
            if e.kind not in table:
                raise ValueError(f"no placeholder geometry for {e.mobility} kind {e.kind!r}; known: {sorted(table)}")
            if e.mobility in ITEM_MOBILITIES:
                parts = table[e.kind](tuple(e.size), color)
            elif e.kind in BY_POSE:
                p0 = tl.placement_of(e.id, 0.0) or next(p for p in tl.placements if p.entity == e.id)
                parts = table[e.kind](tuple(e.size), color, _surfaces_at(m, e.kind, p0.pos))
            else:
                parts = table[e.kind](tuple(e.size), color, zs.get(str(e.attributes.get("name")), ()))
            resolve = app.resolver(e.kind, color) if app else None
            if e.kind in furniture_city.BUILDERS:  # the downtown's ground pieces: their own procedural surfaces
                resolve = city.resolver(resolve)
            elif e.kind in furniture_residential.BUILDERS:  # the suburb's: lawn, roofs, walls (over the city's, too)
                resolve = residential.resolver(resolve)
            ob = link(f"{e.kind}_{idx}", entity_mesh(f"E_{idx}", parts, resolve, exact), idx)
        scene.objects[e.id] = ob
    if app is not None:  # the dressing round the entities, the wear of the fixed surfaces, then the light on all of it
        from stbench import asset_manifest  # not at the top: a script puts `stbench` on the path only when it starts

        _dress(spec, m, app)
        wear.apply(scene, m, lambda set_id: asset_manifest.texture_dir(app.m, set_id))
        scene.lighting = lighting.setup(sc, m, spec, app)
        wear.light_lenses(scene.lighting.get("lamps", []))  # each lamp's lens shows its own lamp's light
    cd = bpy.data.cameras.new("cam")
    cd.sensor_fit = "HORIZONTAL"
    cd.dof.use_dof = False  # a pinhole: `camera.json` and `locate` are
    cd.angle = math.radians(hfov_deg)
    cd.clip_start, cd.clip_end = 0.05, CLIP_END_M[m.outdoor is not None]
    scene.camera = bpy.data.objects.new("cam", cd)
    sc.collection.objects.link(scene.camera)
    sc.camera = scene.camera
    return scene
