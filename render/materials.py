# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Photoreal materials: ambientCG node trees, the `stbench_tint` group, and the procedural shaders.

Runs inside Blender's own Python and is never imported by `stbench`.
- An ambientCG material (colour, roughness, metalness, ambient occlusion and a bump from the displacement map) is
  projected as a box on object coordinates at real-world scale, `tile_m` metres per tile, so it needs no UVs.
- `stbench_tint` makes a textured surface show a catalog colour. It computes `target * clamp(lum(texture) / mean, 0,
  4)`, with `mean` the mean luminance, measured once from the texture's pixels, of the texels the surface uses
  (`texel_mean`; a scanned object's atlas holds texels no triangle maps to, which a whole-image mean would count): the
  surface then has the palette colour's mean albedo and keeps the texture's variation (VISION: the colour attribute is
  what the pixels show).
- The downtown's shaders (asphalt, paving, car paint, glass, lamps, rubble) are `materials_city`'s.
- `concrete` is the warehouse floor (`materials_floor`): a pale, mottled slab with saw-cut joints every
  `CONCRETE_JOINT_M`, worn, and with the map's yellow floor markings and tyre marks when `use_map` has been given a
  warehouse map. It is built from noise and math nodes on object coordinates, so it needs no file and no download (a
  floor mesh sits at the map's origin).
- `named` is the dressing's library (`materials_dressing`: plaster, trim, plastics, metals, paints, cloth, tile, lamps)
  plus every name of `SHADERS`, built once per scene; `use_map` tells the materials which map the scene is for.
- Glass is frosted (roughness 0.45) so that what lies behind it is not recognisable, which agrees with the ID pass.
  Half of its light is scattered at the surface (transmission 0.5) so that a tinted glass keeps its colour: a ray that
  crosses a hollow glass's four surfaces takes the tint four times, so a fully transmissive blue glass rendered as a
  dark navy (the palette blue to the fourth power) and a green one as olive.
"""

from __future__ import annotations

from pathlib import Path

import bpy
import floor_marks
import materials_city
import materials_dressing
import materials_floor
import materials_residential
import numpy as np
from shader_nodes import G
from texel_mean import surface_mean

TINT_GROUP = "stbench_tint"
MEAN_PROP = "stbench_mean_lum"  # custom property on a library material: its base colour's mean luminance
SURFACE_SAMPLES = 20000  # points drawn on a surface to measure its texture's mean (texel_mean.surface_mean)
SURFACE_SEED = 0
DEFAULT_TILE_M = 1.0  # metres per tile where the manifest has no tile_m yet
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
LUMA = np.array([0.2126, 0.7152, 0.0722])
MAX_BOOST = 4.0  # a pixel may be at most this much brighter than the texture's mean
BUMP = {"strength": 0.5, "distance": 0.004}
FROSTED_TRANSMISSION = 0.5  # the rest of a frosted glass's light is diffuse, in its base colour (see the docstring)
SHADERS = ("frosted_glass", "dark_glass", "rubber", "plastic", "stretch_film", "leaf", "soil", "concrete", "asphalt",
           "paving",
           # the downtown's (`materials_city`): car paint, glass, tyres, trim, lamps, rubble and the vehicles' fixed
           # colours
           "paint", "roof_panel", "car_glass", "tyre", "metal", "trim_dark", "liner", "matte_white", "matte_grey",
           "matte_red", "matte_yellow", "wood", "rubble_grey", "rubble_brick", "rubble_dirt", "headlight", "taillight",
           "amber_lamp", "sign_lamp",
           # the suburb's (`materials_residential`): the ground under and round it
           "lawn")
DRESSING_NAMES = materials_dressing.NAMES  # the dressing's named materials (`named`)
_named: dict[str, bpy.types.Material] = {}
_map: dict | None = None  # the raw spec of the map the scene is built for (`use_map`)
CONCRETE_JOINT_M = materials_floor.JOINT_M  # the saw-cut joints of the slab form a grid of this pitch
CONCRETE_GREYS = materials_floor.CONCRETE_GREYS  # linear: the darker and lighter ends of the mottling


# -- the dressing's materials ------------------------------------------------------------------------------------
def use_map(spec: dict | None) -> None:
    """Tell the materials which map the scene is for: its raw spec as `src/stbench/maps/<map>.json` holds it, or None
    for none (the default: plain materials, as they were built before the dressing). `build_scene` calls it once,
    before it builds the structure; it forgets the named materials, which a scene rebuild frees."""
    global _map
    _map = spec
    _named.clear()


def _alive(mat: bpy.types.Material) -> bool:
    try:
        return bpy.data.materials.get(mat.name) is mat
    except ReferenceError:  # freed by a scene rebuild
        return False


def named(name: str) -> bpy.types.Material:
    """The dressing's material `name` (`DRESSING_NAMES`, or one of `SHADERS`), built once per scene and shared by
    everything that asks for it. Raises KeyError for an unknown name."""
    mat = _named.get(name)
    if mat is None or not _alive(mat):
        # the warehouse's, the city's and the household's fixed shaders are named too
        mat = _named[name] = shader_material(name) if name in SHADERS else materials_dressing.build(name)
    return mat


def dressing_material(name: str) -> bpy.types.Material | None:
    """`named(name)`, or None for a name this library does not have (the caller then makes its own)."""
    return named(name) if name in DRESSING_NAMES or name in SHADERS else None


# -- helpers -----------------------------------------------------------------------------------------------------
def principled(mat: bpy.types.Material) -> bpy.types.Node:
    return next(n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")


def new_material(name: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    return mat


def find_map(folder: Path, name: str) -> Path | None:
    """The image in `folder` (searched recursively) named `<anything>_<name>.jpg|png`, case-insensitively; the first in
    sorted order, so a build is deterministic."""
    hits = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
                  and p.stem.lower().endswith("_" + name.lower()))
    return hits[0] if hits else None


def load_image(path: Path, srgb: bool) -> bpy.types.Image:
    """The image at an absolute path, as colour (sRGB) or data (Non-Color). A file Blender cannot read is an error."""
    img = bpy.data.images.load(str(Path(path).resolve()), check_existing=True)
    if img.size[0] == 0 or img.size[1] == 0:
        raise RuntimeError(f"Blender could not read the image {path}")
    img.colorspace_settings.name = "sRGB" if srgb else "Non-Color"
    return img


def srgb_to_linear(a: np.ndarray) -> np.ndarray:
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def luminance_pixels(img: bpy.types.Image) -> np.ndarray:
    """An image's luminance in linear light, (height, width), the first row at the bottom as Blender stores pixels (an
    sRGB image is decoded first)."""
    w, h = img.size
    buf = np.empty(w * h * img.channels, dtype=np.float32)
    img.pixels.foreach_get(buf)
    rgb = buf.reshape(-1, img.channels)[:, :3].astype(np.float64)
    if img.colorspace_settings.name == "sRGB" and not img.is_float:
        rgb = srgb_to_linear(rgb)
    return (rgb @ LUMA).reshape(h, w)


def mean_luminance(img: bpy.types.Image) -> float:
    """Mean luminance over every texel of an image, those no surface uses included (an atlas's black background)."""
    return float(luminance_pixels(img).mean())


def _upstream_image(socket) -> bpy.types.Image | None:
    """The first image texture upstream of an input socket (breadth first, in socket order)."""
    todo, seen = [socket], set()
    while todo:
        s = todo.pop(0)
        for link in s.links:
            node = link.from_node
            if node.name in seen:
                continue
            seen.add(node.name)
            if node.bl_idname == "ShaderNodeTexImage" and node.image is not None:
                return node.image
            todo += list(node.inputs)
    return None


def base_image(mat: bpy.types.Material) -> bpy.types.Image | None:
    """The image that feeds the material's base colour, None for a constant colour."""
    return _upstream_image(principled(mat).inputs["Base Color"])


def base_constant_luminance(mat: bpy.types.Material) -> float:
    return float(np.asarray(principled(mat).inputs["Base Color"].default_value[:3]) @ LUMA)


def base_mean_luminance(mat: bpy.types.Material) -> float:
    """Mean luminance of what feeds the material's base colour: its texture's pixels (the whole image, for a surface
    that projects a tileable texture), else the constant colour."""
    img = base_image(mat)
    return base_constant_luminance(mat) if img is None else mean_luminance(img)


def surface_mean_luminance(mat: bpy.types.Material, triangles: tuple[np.ndarray, np.ndarray] | None
                           ) -> tuple[float, float | None]:
    """(the mean luminance of what feeds the material's base colour over the surface that wears it, the same over its
    whole image). `triangles` is the surface's UV corners (t, 3, 2) and 3-D areas (t,). A constant colour has no
    texture: the first is its luminance and the second None. A surface without UVs (None) or without area projects its
    texture, or shows nothing, and takes the whole image's mean, so both numbers are then equal."""
    img = base_image(mat)
    if img is None:
        return base_constant_luminance(mat), None
    lum = luminance_pixels(img)
    whole = float(lum.mean())
    if triangles is None or not float(triangles[1].sum()) > 0.0:
        return whole, whole
    return surface_mean(lum, *triangles, n=SURFACE_SAMPLES, seed=SURFACE_SEED), whole


# -- the tint group ----------------------------------------------------------------------------------------------
def tint_group() -> bpy.types.ShaderNodeTree:
    """The `stbench_tint` node group (inputs Color, Target, Mean; output Tinted), created on first use."""
    if TINT_GROUP in bpy.data.node_groups:
        return bpy.data.node_groups[TINT_GROUP]
    g = bpy.data.node_groups.new(TINT_GROUP, "ShaderNodeTree")
    iface = g.interface
    for name, kind, default in (("Color", "NodeSocketColor", (0.5, 0.5, 0.5, 1.0)),
                                ("Target", "NodeSocketColor", (1.0, 1.0, 1.0, 1.0)), ("Mean", "NodeSocketFloat", 0.5)):
        iface.new_socket(name, in_out="INPUT", socket_type=kind).default_value = default
    iface.new_socket("Tinted", in_out="OUTPUT", socket_type="NodeSocketColor")
    n = g.nodes
    inp, out = n.new("NodeGroupInput"), n.new("NodeGroupOutput")
    lum = n.new("ShaderNodeRGBToBW")
    ratio = n.new("ShaderNodeMath")
    ratio.operation = "DIVIDE"
    low = n.new("ShaderNodeMath")
    low.operation, low.inputs[1].default_value = "MAXIMUM", 0.0
    high = n.new("ShaderNodeMath")
    high.operation, high.inputs[1].default_value = "MINIMUM", MAX_BOOST
    scale = n.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    cap = n.new("ShaderNodeVectorMath")
    cap.operation = "MINIMUM"
    cap.inputs[1].default_value = (1.0, 1.0, 1.0)
    link = g.links.new
    link(inp.outputs["Color"], lum.inputs["Color"])
    link(lum.outputs["Val"], ratio.inputs[0])
    link(inp.outputs["Mean"], ratio.inputs[1])
    link(ratio.outputs[0], low.inputs[0])
    link(low.outputs[0], high.inputs[0])
    link(inp.outputs["Target"], scale.inputs[0])
    link(high.outputs[0], scale.inputs["Scale"])
    link(scale.outputs["Vector"], cap.inputs[0])
    link(cap.outputs["Vector"], out.inputs["Tinted"])
    return g


def add_tint(mat: bpy.types.Material, mean: float) -> bpy.types.Node:
    """Put a tint group in front of the material's base colour (its texture, or its constant colour)."""
    nt, p = mat.node_tree, principled(mat)
    g = nt.nodes.new("ShaderNodeGroup")
    g.node_tree, g.name = tint_group(), TINT_GROUP
    g.inputs["Mean"].default_value = max(mean, 1e-4)
    base = p.inputs["Base Color"]
    if base.links:
        source = base.links[0].from_socket
        nt.links.remove(base.links[0])
        nt.links.new(source, g.inputs["Color"])
    else:
        g.inputs["Color"].default_value = base.default_value
    nt.links.new(g.outputs["Tinted"], base)
    return g


def set_tint_target(mat: bpy.types.Material, linear_rgb: tuple[float, float, float]) -> None:
    node = mat.node_tree.nodes.get(TINT_GROUP)
    if node is None:
        raise KeyError(f"material {mat.name!r} has no {TINT_GROUP} node")
    node.inputs["Target"].default_value = (*linear_rgb, 1.0)


def darkened(mat: bpy.types.Material, factor: float) -> bpy.types.Material:
    """A copy of a material whose base colour is multiplied by `factor` (the palette's "oak*0.8")."""
    out = mat.copy()
    out.name = f"{mat.name}*{factor:g}"
    nt, p = out.node_tree, principled(out)
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type, mix.blend_type = "RGBA", "MULTIPLY"
    mix.inputs["Factor"].default_value = 1.0
    mix.inputs["B"].default_value = (factor, factor, factor, 1.0)
    base = p.inputs["Base Color"]
    if base.links:
        source = base.links[0].from_socket
        nt.links.remove(base.links[0])
        nt.links.new(source, mix.inputs["A"])
    else:
        mix.inputs["A"].default_value = base.default_value
    nt.links.new(mix.outputs["Result"], base)
    return out


# -- ambientCG ---------------------------------------------------------------------------------------------------
def ambientcg_material(asset_id: str, folder: Path, tile_m: list | None) -> bpy.types.Material:
    """The material for an unpacked ambientCG zip in `folder`; a missing colour map is an error, the others are
    optional. Records the base colour's mean luminance in `MEAN_PROP`."""
    tx, ty = tile_m or (DEFAULT_TILE_M, DEFAULT_TILE_M)
    mat = new_material(asset_id)
    nt, p = mat.node_tree, principled(mat)
    coord, mapping = nt.nodes.new("ShaderNodeTexCoord"), nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1 / tx, 1 / ty, 1 / ty)
    nt.links.new(coord.outputs["Object"], mapping.inputs["Vector"])

    def texture(name: str, srgb: bool):
        path = find_map(folder, name)
        if path is None:
            return None
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image, node.projection, node.projection_blend = load_image(path, srgb), "BOX", 0.2
        nt.links.new(mapping.outputs["Vector"], node.inputs["Vector"])
        return node

    colour = texture("Color", True)
    if colour is None:
        raise RuntimeError(f"{asset_id}: no *_Color image under {folder}")
    source = colour.outputs["Color"]
    if (ao := texture("AmbientOcclusion", False)) is not None:
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type, mix.blend_type = "RGBA", "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        nt.links.new(colour.outputs["Color"], mix.inputs["A"])
        nt.links.new(ao.outputs["Color"], mix.inputs["B"])
        source = mix.outputs["Result"]
    nt.links.new(source, p.inputs["Base Color"])
    for name, socket in (("Roughness", "Roughness"), ("Metalness", "Metallic")):
        if (node := texture(name, False)) is not None:
            nt.links.new(node.outputs["Color"], p.inputs[socket])
    if (disp := texture("Displacement", False)) is not None:
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = BUMP["strength"]
        bump.inputs["Distance"].default_value = BUMP["distance"]
        nt.links.new(disp.outputs["Color"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], p.inputs["Normal"])
    mat[MEAN_PROP] = base_mean_luminance(mat)
    return mat


# -- procedural shaders ------------------------------------------------------------------------------------------
def concrete_material() -> bpy.types.Material:
    """A warehouse slab (`materials_floor.slab`): mottled concrete with saw-cut joints every `CONCRETE_JOINT_M`, worn,
    and with the map's floor markings and tyre marks when `use_map` gave it a warehouse map (`floor_marks.layout`)."""
    mat = materials_floor.slab(floor_marks.layout(_map) if _map is not None else None)
    mat[MEAN_PROP] = float(np.asarray([(a + b) / 2 for a, b in zip(*CONCRETE_GREYS, strict=True)]) @ LUMA)
    return mat


def shader_material(name: str) -> bpy.types.Material:
    """One of `SHADERS`, the manifest's `shader:<name>` materials."""
    if name not in SHADERS:
        raise KeyError(f"unknown shader {name!r}; known: {list(SHADERS)}")
    if name == "concrete":
        return concrete_material()
    if name == "lawn":  # the suburb's ground (`look_residential.GROUND`)
        mat = materials_residential.lawn()
        mat[MEAN_PROP] = base_mean_luminance(mat)
        return mat
    if name in materials_city.NAMES:  # the downtown's: street surfaces, vehicle paint and glass, lamps, rubble
        mat = materials_city.shader(name)
        mat[MEAN_PROP] = base_mean_luminance(mat)
        return mat
    mat = new_material(f"shader:{name}")
    p = principled(mat)
    if name == "frosted_glass":  # tinted by its base colour; rough enough that nothing behind it is recognisable
        p.inputs["Base Color"].default_value = (0.9, 0.95, 0.95, 1.0)
        p.inputs["Roughness"].default_value = 0.45
        p.inputs["IOR"].default_value = 1.45
        p.inputs["Transmission Weight"].default_value = FROSTED_TRANSMISSION
    elif name == "dark_glass":  # a screen: glossy, opaque and nearly black
        p.inputs["Base Color"].default_value = (0.008, 0.01, 0.016, 1.0)
        p.inputs["Roughness"].default_value = 0.08
    elif name == "plastic":  # a moulded housing: smooth and semi-gloss, tinted by its base colour
        p.inputs["Base Color"].default_value = (0.9, 0.9, 0.9, 1.0)
        p.inputs["Roughness"].default_value = 0.38
    elif name == "stretch_film":  # a pallet's wrap: `plastic` (opaque, tinted by its base colour), but glossy, wrinkled
        p.inputs["Base Color"].default_value = (0.9, 0.9, 0.9, 1.0)
        p.inputs["Roughness"].default_value = 0.15
        p.inputs["Specular IOR Level"].default_value = 0.6
        g = G(mat.node_tree)
        wrinkles = g.noise(g.node("ShaderNodeTexCoord").outputs["Object"], 45.0, 3.0, 0.6)  # fractions of a mm
        g.put(p, "Normal", g.bump(wrinkles, 0.2, 0.0004))
    elif name == "leaf":  # a succulent's leaves: flat dark green, slightly waxy (the pot carries the colour)
        p.inputs["Base Color"].default_value = (0.045, 0.2, 0.06, 1.0)
        p.inputs["Roughness"].default_value = 0.45
    elif name == "soil":
        p.inputs["Base Color"].default_value = (0.05, 0.034, 0.022, 1.0)
        p.inputs["Roughness"].default_value = 0.95
    else:
        p.inputs["Base Color"].default_value = (0.04, 0.04, 0.045, 1.0)
        p.inputs["Roughness"].default_value = 0.8
    mat[MEAN_PROP] = base_mean_luminance(mat)
    return mat
