# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Wear and imperfection for the structure and the fixed furniture.

Runs inside Blender's own Python and is never imported by `stbench`. `apply(scene, m, texture_dir)` takes a built scene
and gives what a building and its fixed furniture are made of a lived-in finish; `worn(material, kind)` is one
material's finish.
- walls: smooth painted plaster, per room, with wear only where it gathers (`wear_surfaces`); ceilings: a faint patch;
  the dressing's trim and the doors, a doorway's threshold, and what lies under a table or a sofa have finishes
  there too;
- floors (wood, carpet, tile, by the region's name): wear paths, scuffs and blotchy dirt, in the roughness and a few per
  cent of the colour, the grout of tiles grubby and a dust band round whatever stands on the floor;
- fixed furniture: a little roughness and tone variation and dirt at the foot; a pallet rack's paint is chipped (a few
  per cent of it, more near the floor where forklifts knock it) to dark bare steel;
- rugs: a pattern (a fringe, a border, accent stripes, a lattice and a medallion) in tones of the rug's own colour. The
  tinted carpet is multiplied by a pattern whose mean is 1 and the accents are a paler tint of the same colour, so the
  rug's colour attribute stays its dominant colour;
- carpet and rugs also have a pile (`floor_look.pile`: sheen and a bump of fibre tips and tufts).

Every effect is procedural (noise, ramps and math on object coordinates: no file, no UVs, no download) and subtle: a
change of a few per cent in the colour, never a shape a perception model could take for an object. Where the manifest's
ambientCG mask sets are on disk (`load_masks`: SurfaceImperfections003, Smear005, Scratches004, Fingerprints002, their
Opacity maps, a grey mask each) the walls take their grime streaks from SurfaceImperfections, the floors their scuffs
from Smear and the furniture its roughness from Fingerprints, box-projected at real-world scale, in place of the
noise that stands in for them.

The finish goes on a *copy* of each material, put in the mesh of the one object it belongs to, so no material another
object shares changes, and it is only ever given to structure and to fixed furniture: a carried item, a piece of light
furniture or a mover (every tracked thing) keeps exactly its materials, so the catalog's colours do not move.
Deterministic: no noise has a seed of its own, and the copies are made in a fixed order.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import bpy
import floor_look
import furniture_city
import look
import wear_surfaces as W
from cycles_setup import UNIT_LUX
from materials import TINT_GROUP, find_map, load_image
from materials_city import lin
from shader_nodes import G, principled

PILE_KINDS = ("carpet", "rug")  # the finishes that take `floor_look.pile`
SKIP_FURNITURE = ("staging_lane", "bay")  # painted outlines, not bodies
DIRT = W.DIRT  # what dust and grime multiply a surface by
GROUT_DIRT = lin((0.52, 0.47, 0.41))  # the grime in a tile's joints
BARE_STEEL = lin((0.30, 0.29, 0.29))  # under a chip of paint
PALE = lin((0.72, 0.69, 0.64))  # an accent of a rug is its colour mixed towards this (a pale wool, albedo 0.6-0.7)
# the flat colour of the tint, not the texture's picture: paint, and a rug whose library texture is a bath mat's grid
# (the rug's pattern is `_rug`'s)
FLAT_BASE = ("wall", "ceiling", "door", "rug")
# a library bump's strength times this, by finish: paint, laminate and doors are smooth to the eye from 5 to 30 cm; a
# loop
# pile is fibre (`floor_look.pile`), not the bubbles of the library's displacement map
BUMP_SCALE = {"wall": 0.06, "ceiling": 0.06, "door": 0.08, "smooth": 0.03, "furniture": 0.25, "fabric": 0.6,
              "mat": 0.25, "carpet": 0.2, "rug": 0.2}
SURFACE_OF = {"door": "door", "vanity": "smooth", "toilet": "smooth", "bathtub": "smooth", "sofa": "fabric",
              "bed": "fabric", "floor_mat": "mat"}  # entity kind -> finish; any other fixed piece is "furniture"
FACTORIES = {"rug": lambda size: _rug(size), "door": W.door, "cover": W.cover}  # finishes that need the piece's size
UNDER_KINDS = ("raw", "cover")  # palette keys of what lies under furniture (`shapes.PALETTE`): same-named finishes
TRIM_CLASSES = ("baseboard", "door_casing", "window_sill")  # dressing classes that get the trim's finish
# dressing piece (name prefix) -> (radius across, radius up, strength) of the grease round a wall fitting
SPOT_PIECES = {"dressing_outlet": (0.09, 0.12, 0.30), "dressing_light_switch": (0.08, 0.13, 0.36),
               "dressing_thermostat": (0.08, 0.10, 0.20)}
# a tile set -> (its height map's value off the grout, on it, the grime in the joints, the dust band round fixtures, the
# tile's own tint: the bathroom's are pale blue, the hall's and kitchen's a warm grey)
TILE_GROUT = {"acg_Tiles139": (0.40, 0.25, 0.60, 0.45, lin((1.0, 0.97, 0.92))),
              "acg_Tiles107": (0.64, 0.50, 0.75, 0.80, lin((0.86, 0.92, 0.96)))}
AO_KEPT = 0.25  # the share of a carpet's baked ambient occlusion kept: each loop has a dark gap, which reads as bubbles
RUG_TONE = 0.85  # a rug's albedo is its pattern's times this (a pale wool is 0.6-0.7)
HALO_M = 0.07  # the dust band round a fixture's foot
LOOK_CATALOGS = ("home",)  # the catalogs that take the floor-level look of this round (the warehouse's is as it was)
_cache: dict[tuple, bpy.types.Material] = {}
_masks: dict[str, bpy.types.Image] = {}
MASK_SETS = {"imperfections": "acg_SurfaceImperfections003", "smear": "acg_Smear005",
             "scratches": "acg_Scratches004", "fingerprints": "acg_Fingerprints002"}
MASK_TILE_M = {"imperfections": 1.5, "smear": 1.5, "scratches": 0.6, "fingerprints": 0.5}  # one image covers this much
MASK_RANGE = {"imperfections": (0.07, 0.26), "smear": (0.15, 0.6), "scratches": (0.03, 0.2),
              "fingerprints": (0.1, 0.5)}  # the greys that count as none and as all of the mask's effect


def use_masks(masks: dict[str, bpy.types.Image]) -> None:
    """The mask images (`MASK_SETS`' keys) the finishes read in place of noise; none by default."""
    _masks.clear()
    _masks.update(masks)


def load_masks(folder_of: Callable[[str], Path]) -> dict[str, bpy.types.Image]:
    """The `MASK_SETS` images: `folder_of(set id)` is the folder a set's maps are unpacked to (`asset_manifest.
    texture_dir`); a set whose folder or Opacity map is missing is left out, and its effect stays procedural."""
    out = {}
    for key, set_id in MASK_SETS.items():
        try:
            path = find_map(folder_of(set_id), "Opacity")
        except Exception:  # a set the manifest does not list, or that the cache lacks: its effect stays procedural
            path = None
        if path is not None:
            out[key] = load_image(path, srgb=False)
    return out


def _mask(g: G, key: str, obj):
    """The scalar 0 to 1 of a mask image at `obj` (box-projected, `MASK_TILE_M` metres a tile, its `MASK_RANGE`
    stretched to 0..1), or None where the image is not there."""
    img = _masks.get(key)
    if img is None:
        return None
    tile, (lo, hi) = MASK_TILE_M[key], MASK_RANGE[key]
    tex = g.node("ShaderNodeTexImage", {"Vector": g.scaled(obj, 1 / tile, 1 / tile, 1 / tile)}, projection="BOX",
                 projection_blend=0.2)
    tex.image = img
    return g.smooth(g.node("ShaderNodeRGBToBW", {"Color": tex.outputs["Color"]}).outputs["Val"], lo, hi)


# -- taking a material apart -----------------------------------------------------------------------------------
def _source(nt: bpy.types.NodeTree, socket):
    """What feeds an input socket, with the link taken off it, or the socket's own constant (a tuple or a float)."""
    if socket.links:
        src = socket.links[0].from_socket
        nt.links.remove(socket.links[0])
        return src
    v = socket.default_value
    return tuple(v) if hasattr(v, "__len__") else float(v)


def _vary(g: G, rough, noise, amount: float):
    """A roughness (a constant or a socket) moved up and down by `amount` / 2 along a noise."""
    return g.math("ADD", rough, g.math("MULTIPLY_ADD", noise, amount, -amount / 2), clamp=True)


# -- the finishes: each takes the colour and roughness of a material and the object-coordinates vector, and returns
# the colour and the roughness of the worn one -----------------------------------------------------------------
def _wall(g: G, base, rough, obj, ctx=None):
    z = g.xyz(obj)[2]
    streaks = _mask(g, "imperfections", obj)  # drips and smudges, long in z
    gathered = (g.math("MULTIPLY_ADD", g.noise(obj, 5.0, 3.0), 1.2, 0.2) if streaks is None
                else g.math("MULTIPLY_ADD", streaks, 1.0, 0.15))
    foot = g.math("MULTIPLY", g.smooth(z, 0.55, 0.0), gathered)
    scuff = g.math("MULTIPLY", g.smooth(g.noise(g.scaled(obj, 1.0, 1.0, 0.35), 14.0, 2.0, 0.6), 0.64, 0.76),
                   g.smooth(z, 1.5, 1.0))
    blotch = g.noise(g.shifted(obj, 3.0, 7.0, 1.0), 0.5, 3.0)  # broad patches of the plaster, in colour and gloss
    col = g.tint(g.math("MULTIPLY", foot, 0.28), base, DIRT)
    if streaks is not None:  # the same smudges, faint, up the whole wall
        col = g.tint(g.math("MULTIPLY", streaks, 0.05), col, DIRT)
    col = g.tint(g.math("MULTIPLY", scuff, 0.16), col, lin((0.62, 0.60, 0.57)))
    return g.shade(col, g.math("MULTIPLY_ADD", blotch, 0.08, 0.96)), _vary(g, rough, blotch, 0.10)


def _ceiling(g: G, base, rough, obj, ctx=None):
    col = g.shade(base, g.math("MULTIPLY_ADD", g.noise(g.shifted(obj, 5.0, 1.0, 3.0), 0.4, 3.0), 0.06, 0.97))
    return col, rough


# kind -> (path lightening, path amount, grime amount, rough up on paths, rough up on scuffs)
FLOORS = {"wood": (1.10, 0.25, 0.10, 0.14, 0.10), "carpet": (0.90, 0.30, 0.16, -0.05, 0.0),
          "tile": (1.04, 0.15, 0.12, 0.10, 0.12)}


def _floor(kind: str):
    lighten, path_amt, grime_amt, rough_path, rough_scuff = FLOORS[kind]

    def finish(g: G, base, rough, obj, ctx=None):
        smear = _mask(g, "smear", obj)  # long scuff marks, across x
        scuffs = g.noise(g.scaled(obj, 1.0, 0.3, 1.0), 38.0, 2.0, 0.6) if smear is None else smear
        paths = g.smooth(g.noise(g.scaled(obj, 0.35, 0.9, 1.0), 1.6, 3.0, 0.55), 0.50, 0.72)
        scuff = g.smooth(scuffs, 0.58, 0.72) if smear is None else smear
        grime = g.smooth(g.noise(g.shifted(obj, 9.0, 2.0, 0.0), 3.0, 3.0, 0.6), 0.50, 0.78)
        col = g.tint(g.math("MULTIPLY", paths, path_amt), base, lin((lighten, lighten, lighten)))
        col = g.tint(g.math("MULTIPLY", grime, grime_amt * 2.0), col, DIRT)
        col = g.tint(g.math("MULTIPLY", scuff, 0.10), col, lin((0.8, 0.78, 0.74)))
        if kind == "tile" and ctx is not None and ctx.name in TILE_GROUT:
            col = _grubby_tile(g, col, obj, ctx, TILE_GROUT[ctx.name])
        rough = g.math("ADD", g.math("ADD", rough, g.math("MULTIPLY", paths, rough_path)),
                       g.math("MULTIPLY", scuff, rough_scuff), clamp=True)
        return col, _vary(g, rough, scuffs, 0.08 if smear is None else 0.3)

    return finish


def _grubby_tile(g: G, col, obj, ctx, spec):
    """Grime in the joints of a tile floor (where its height map is low) and a band of dust and spilt water round
    whatever stands on it: the tile's white is no cleaner than a real bathroom's, whose grout is grey-brown."""
    off, on, joint_amt, halo_amt, paint = spec
    col = g.tint(1.0, col, paint)
    dirt = g.smooth(g.noise(g.shifted(obj, 5.0, 3.0, 0.0), 6.0, 3.0, 0.6), 0.25, 0.7)
    if ctx.height is not None:
        joints = g.smooth(g.node("ShaderNodeRGBToBW", {"Color": ctx.height}).outputs["Val"], off, on)
        col = g.tint(g.math("MULTIPLY", joints, g.math("MULTIPLY_ADD", dirt, 0.5 * joint_amt, 0.5 * joint_amt)), col,
                     GROUT_DIRT)
    band = W.halo(g, obj, HALO_M)
    return g.tint(g.math("MULTIPLY", band, g.math("MULTIPLY_ADD", dirt, 0.5 * halo_amt, 0.5 * halo_amt)), col, DIRT)


def _furniture(g: G, base, rough, obj, ctx=None):
    z = g.xyz(obj)[2]
    n = g.noise(g.shifted(obj, 2.0, 4.0, 6.0), 6.0, 3.0)
    col = g.shade(base, g.math("MULTIPLY_ADD", n, 0.05, 0.975))
    col = g.tint(g.math("MULTIPLY", g.smooth(z, 0.12, 0.0), 0.35), col, DIRT)
    prints = _mask(g, "fingerprints", obj)  # smudges in the gloss
    gloss = 0.12 if prints is None else 0.12 if _home else 0.2  # a home's smudges are fainter: at 6 cm they are clouds
    return col, _vary(g, rough, n if prints is None else prints, gloss)


def _paint(g: G, base, rough, obj, ctx=None):
    """Chipped paint on a pallet rack: dark bare steel in a few per cent of it, more in the 0.5 m above the floor."""
    z = g.xyz(obj)[2]
    knocked = g.math("MULTIPLY", g.smooth(z, 0.5, 0.0), 0.07)
    lo, hi = g.math("SUBTRACT", 0.64, knocked), g.math("SUBTRACT", 0.72, knocked)
    chips = g.smooth(g.noise(obj, 22.0, 5.0, 0.65), lo, hi)
    if (scratches := _mask(g, "scratches", obj)) is not None:  # long scratches bare the steel too
        chips = g.math("MAXIMUM", chips, g.math("MULTIPLY", scratches, 0.8))
    col = g.mix(chips, g.tint(g.math("MULTIPLY", g.smooth(z, 0.4, 0.0), 0.3), base, DIRT), BARE_STEEL)
    return col, g.math("ADD", g.math("MULTIPLY", chips, 0.4), _vary(g, rough, g.noise(obj, 16.0, 3.0), 0.12),
                       clamp=True)


def _harlequin(g: G, x, y, pitch: float):
    """0 or 1 in a lattice of diamonds `pitch` across (a harlequin check), edges softened to a few millimetres."""
    halves = [g.smooth(g.triangle(v, pitch), 0.48, 0.52) for v in (g.math("ADD", x, y), g.math("SUBTRACT", x, y))]
    return g.math("ABSOLUTE", g.math("SUBTRACT", *halves))


def _rug(size: tuple[float, float]):
    """A rug's pattern, from the distance `d` to its edge (hx, hy are its half sizes): a fringe, a plain border, a pale
    stripe, a band of small diamonds, a second stripe, then a field of large diamonds with a medallion's rings. Every
    tone factor has a mean of 1 (the diamonds swing +-), so the tinted carpet keeps its colour overall."""
    hx, hy = size[0] / 2, size[1] / 2

    def finish(g: G, base, rough, obj, ctx=None):
        x, y, _ = g.xyz(obj)
        d = g.math("MINIMUM", g.math("SUBTRACT", hx, g.math("ABSOLUTE", x)),
                   g.math("SUBTRACT", hy, g.math("ABSOLUTE", y)))
        small, large = _harlequin(g, x, y, 0.10), _harlequin(g, x, y, 0.32)
        radius = g.math("SQRT", g.math("ADD", g.math("POWER", g.math("DIVIDE", x, hx), 2.0),
                                       g.math("POWER", g.math("DIVIDE", y, hy), 2.0)))
        rings = g.math("MULTIPLY", g.math("MULTIPLY_ADD", g.sine(radius, 0.26), 0.20, -0.10),
                       g.smooth(radius, 0.75, 0.55))
        tone = g.math("ADD", 1.0, g.math("MULTIPLY", g.between(d, -1.0, 0.035), -0.18))
        band = g.math("MULTIPLY", g.between(d, 0.14, 0.30), g.math("MULTIPLY_ADD", small, 0.28, -0.14))
        tone = g.math("ADD", tone, band)
        tone = g.math("ADD", tone, g.math("MULTIPLY", g.smooth(d, 0.31, 0.34),
                                          g.math("ADD", g.math("MULTIPLY_ADD", large, 0.18, -0.09), rings)))
        accent = g.math("MAXIMUM", g.between(d, 0.10, 0.125), g.between(d, 0.30, 0.325))
        fleck = g.noise(obj, 260.0, 2.0, 0.6)  # the fibres of the wool: a flat tint has no variation of its own
        coloured = g.shade(base, g.math("MULTIPLY", tone, g.math("MULTIPLY_ADD", fleck, 0.18, 0.91)))
        pale = g.mix(0.45, coloured, PALE)  # a paler tint of the same colour
        return g.shade(g.mix(g.math("MULTIPLY", accent, 0.4), coloured, pale), RUG_TONE if _home else 1.0), rough

    return finish


FINISHES = {"wall": _wall, "ceiling": _ceiling, "wood": _floor("wood"), "carpet": _floor("carpet"),
            "tile": _floor("tile"), "furniture": _furniture, "paint": _paint}
# the household's (`LOOK_CATALOGS`): smooth painted plaster, and the finishes of the surfaces it adds
HOME_FINISHES = {**FINISHES, "wall": W.wall, "ceiling": W.ceiling, "smooth": _furniture, "fabric": _furniture,
                 "mat": _furniture, "trim": W.trim, "threshold": W.threshold, "raw": W.raw}
W.mask_fn = _mask
_home = False  # whether the scene being dressed is of a catalog in `LOOK_CATALOGS`


@dataclass(frozen=True)
class Ctx:
    """What a finish may ask of the material it dresses: its name and the socket feeding its bump (its height map)."""

    name: str
    height: object | None


def _alive(mat) -> bool:
    try:
        return bpy.data.materials.get(mat.name) is mat
    except ReferenceError:
        return False


def _tint_target(mat: bpy.types.Material) -> tuple | None:
    """The colour a tinted material was given (the palette colour of a painted surface), None if it has no tint."""
    node = mat.node_tree.nodes.get(TINT_GROUP)
    return None if node is None else tuple(node.inputs["Target"].default_value)


def worn(mat: bpy.types.Material, kind: str, size: tuple[float, float] | None = None) -> bpy.types.Material:
    """A worn copy of `mat` (`kind`: `FINISHES`, or one of `FACTORIES` with the piece's `size` (depth, width)), made
    once per material and kind. The copy keeps the material's custom properties and images. In a household scene
    (`apply`) its library bump is cut by `BUMP_SCALE` and a painted finish (`FLAT_BASE`) takes the flat colour the
    material was tinted to."""
    key = (mat.name, kind, size, tuple(sorted(_masks)), _home)
    if key in _cache and _alive(_cache[key]):
        return _cache[key]
    finish = FACTORIES[kind](size) if kind in FACTORIES else (HOME_FINISHES if _home else FINISHES)[kind]
    out = mat.copy()
    out.name = f"{mat.name}+{kind}"
    nt, p = out.node_tree, principled(out)
    g = G(nt)
    bump = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBump" and n.inputs["Height"].links), None)
    height = None if bump is None else bump.inputs["Height"].links[0].from_socket
    obj = g.node("ShaderNodeTexCoord").outputs["Object"]
    base = _source(nt, p.inputs["Base Color"])
    if _home and kind in FLAT_BASE and (target := _tint_target(out)) is not None:
        base = target
    if _home:
        W.scale_bump(out, BUMP_SCALE.get(kind, 1.0))
        if kind in PILE_KINDS:
            W.soften_ao(out, AO_KEPT)
    col, rough = finish(g, base, _source(nt, p.inputs["Roughness"]), obj, Ctx(mat.name, height))
    g.put(p, "Base Color", col)
    g.put(p, "Roughness", rough)
    if kind in PILE_KINDS:
        floor_look.pile(g, p, obj)
    _cache[key] = out
    return out


# -- a scene -------------------------------------------------------------------------------------------------------
def _dress(ob: bpy.types.Object, kind: str, size=None, painted: tuple[str, ...] = ()) -> None:
    """Replace the materials of an object's mesh with worn copies of `kind`; where `painted` names palette colours, a
    slot of one of them gets the chipped-paint finish and the others the plain furniture one, and a slot of what lies
    under furniture (`UNDER_KINDS`) the finish of its own name."""
    me = ob.data
    for i, mat in enumerate(list(me.materials)):
        if mat is not None:
            name = mat.name.partition("*")[0]
            use = "paint" if name in painted else name if name in UNDER_KINDS else kind
            me.materials[i] = worn(mat, use, size if use in FACTORIES else None)  # size only where a finish needs it


def _class_of(name: str) -> str | None:
    """The dressing class an object's name says (`dressing_<class>_<n>`, with `_<k>` for a second object of a piece)."""
    hit = re.fullmatch(r"dressing_(.+?)_\d+(?:_\d+)?", name)
    return hit.group(1) if hit else None


def scene_marks(m):
    """The rooms (kind, box), the wall spots that wear marks and the floor footprints dust gathers round, as
    `wear_surfaces.use_scene` takes them: from the map (the rooms, the doors' jambs, the corners of tall furniture that
    stand within 12 cm of a wall, the footprints of what stands whole on the floor) and from the dressing in the scene
    (the sockets, switches and thermostats), so the marks are where the fittings are."""
    import numpy as np

    from stbench.geometry import dist_to_rect  # not at the top: a script puts `stbench` on the path only when it starts

    rooms = [(look.kind_of(r), (min(p[0] for p in r.polygon), min(p[1] for p in r.polygon),
                                max(p[0] for p in r.polygon), max(p[1] for p in r.polygon))) for r in m.regions]
    spots = []
    for ob in bpy.data.objects:
        for prefix, (across, up, strength) in SPOT_PIECES.items():
            if ob.name.startswith(prefix):
                spots.append((ob.location.x, ob.location.y, ob.location.z + 0.06, across, across, up, strength))
    for o in (o for o in m.openings if o.type == "door"):
        ux, uy = math.cos(o.yaw), math.sin(o.yaw)
        for side in (-1, 1):
            x, y = o.center[0] + side * ux * (o.width / 2 + 0.03), o.center[1] + side * uy * (o.width / 2 + 0.03)
            spots += [(x, y, 1.0, 0.12, 0.12, 0.22, 0.24), (x, y, 0.15, 0.14, 0.14, 0.22, 0.34)]
    walls = m.wall_pieces()
    footprints = []
    for f in (f for f in m.furniture if f.mobility == "fixed" and f.blocks and f.size[2] >= 0.2):
        hx, hy, c, s = f.size[0] / 2, f.size[1] / 2, math.cos(f.yaw), math.sin(f.yaw)
        if f.under_m == 0.0:
            footprints.append((f.center[0], f.center[1], hx, hy, f.yaw))
        if f.size[2] < 0.4:
            continue
        for lx, ly in ((hx, hy), (hx, -hy), (-hx, hy), (-hx, -hy)):
            x, y = f.center[0] + c * lx - s * ly, f.center[1] + s * lx + c * ly
            if min(float(dist_to_rect(np.array([x]), np.array([y]), w)[0]) for w in walls) < 0.12:
                spots.append((x, y, 0.22, 0.12, 0.12, 0.20, 0.26))
    return rooms, spots, footprints


def use_catalog(catalog: str) -> None:
    """Dress what follows with this round's floor-level look (`LOOK_CATALOGS`) or, for any other catalog, as it was."""
    global _home
    _home = catalog in LOOK_CATALOGS


def apply(scene, m, texture_dir: Callable[[str], Path] | None = None) -> None:
    """Give a built scene's walls, ceilings, floors, fixed furniture, trim, threshold and undersides their finish. An
    outdoor map (the city) has its own and is left as it is. Never touches an entity that is not fixed: those are the
    tracked things. `texture_dir` (as `load_masks` takes it) lets the finishes read the manifest's mask sets; without
    it, or where a set is missing, the noise stands in. The household look is on only while this runs."""
    if m.outdoor is not None:
        return
    use_catalog(m.catalog)
    try:
        _apply(scene, m, texture_dir)
    finally:
        use_catalog("")
        W.use_scene([], [], [])


def _apply(scene, m, texture_dir: Callable[[str], Path] | None) -> None:
    if texture_dir is not None:
        use_masks(load_masks(texture_dir))
    W.use_scene(*(scene_marks(m) if _home else ([], [], [])))
    finish = {"walls": "wall", **({"underlay": "threshold"} if _home else {})}
    worn_floors = look.module(m.catalog).FLOOR_FINISH is not None  # None: the catalog's floors have their own finish
    for r in m.regions:
        finish[f"ceiling_{r.id}"] = "ceiling"
        if worn_floors:
            finish[f"floor_{r.id}"] = look.get(m.catalog, "FLOOR_FINISH", look.kind_of(r))
    for name, kind in sorted(finish.items()):
        if (ob := bpy.data.objects.get(name)) is not None:
            _dress(ob, kind)
    dressing = bpy.data.collections.get("dressing")
    for ob in () if dressing is None or not _home else sorted(dressing.objects, key=lambda o: o.name):
        if _class_of(ob.name) in TRIM_CLASSES:
            _dress(ob, "trim")
    for eid, ob in scene.objects.items():
        e = scene.tl.entity(eid)
        if e.mobility != "fixed" or e.kind in furniture_city.BUILDERS or e.kind in SKIP_FURNITURE:
            continue
        size = (float(e.size[0]), float(e.size[1]))
        if e.kind == "rug":
            _dress(ob, "rug", size)
        elif e.kind == "pallet_rack":  # the upright paint chips; the cartons and pallets of its stock are scenery
            _dress(ob, "furniture", painted=(str(e.attributes.get("color")), "orange", "grey"))
        else:
            _dress(ob, SURFACE_OF.get(e.kind, "furniture") if _home else "furniture", size)


# -- the lamps' lenses ---------------------------------------------------------------------------------------------
LENS_MATERIAL = "dress:led_emitter_warm"  # `materials_dressing.PREFIX` and the name of the home's lens
LENS_AREA_M2 = math.pi * 0.198 ** 2  # a flush lamp's lens seen from below (`dressing_build_standins._lamp_flush`)
LENS_REACH_M = 0.05  # a lamp's lens is centred on its light, to this much


def light_lenses(lamps: list[dict]) -> None:
    """Give the lens of each flush ceiling lamp the luminance of its own lamp, as a physical lens has it. `lamps` is the
    rig's record (`lighting.setup`'s `lamps`: name, lumens, pos): a lamp of `lumens` lumens is `lumens * pi / UNIT_LUX`
    render watts, and a Lambertian emitter of that power over the lens (`LENS_AREA_M2`) has the radiance, a mesh
    emission's strength, `lumens / (UNIT_LUX * LENS_AREA_M2)`: 21 for the hall's 610 lm, 165 for the kitchen's 4,860.
    The lens is an emission only the camera sees (`materials_dressing._camera_only`), so no light is counted twice. The
    lamp's own copy of the material goes on its object's slot, the mesh being shared by every lamp of the size."""
    dressing = bpy.data.collections.get("dressing")
    lenses = [] if dressing is None else [ob for ob in dressing.objects
                                          if _class_of(ob.name) == "ceiling_lamp" and ob.dimensions.z < 0.5]
    for lamp in lamps:
        for ob in (ob for ob in lenses if math.hypot(ob.location.x - lamp["pos"][0],
                                                     ob.location.y - lamp["pos"][1]) < LENS_REACH_M):
            for slot in (s for s in ob.material_slots if s.material is not None and s.material.name == LENS_MATERIAL):
                mat = slot.material.copy()
                mat.name = f"{LENS_MATERIAL}@{lamp['name']}"
                strength = principled(mat).inputs["Emission Strength"].links[0].from_node  # the camera-only product
                strength.inputs[1].default_value = lamp["lumens"] / (UNIT_LUX * LENS_AREA_M2)
                slot.link, slot.material = "OBJECT", mat
