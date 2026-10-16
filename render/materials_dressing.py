# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The named PBR materials of the dressing: what the trim, fittings, fixtures and
building structure of a map are made of.

Runs inside Blender's own Python and is never imported by `stbench`. `build(name)` makes one of `NAMES`;
`materials.named` is the cached entry point the dressing builders use. Every material is procedural: noise, ramps and
math on object coordinates, so it needs no file, no UVs and no download (the city's shaders and the warehouse's
concrete are built the same way), and it is the same in every run. A piece's object frame is the dressing's: x out
of its face, y across, z up, so a ribbed or pleated material reads its pattern off `y` and `z`.

Colours are given as sRGB and converted. Nothing here is a tracked kind's colour recipe: the dressing's validity (it
never looks like a tracked thing of its catalog) is its registry's (`stbench.scenery`) and its layout's. A material
whose name says a colour is a paint of that colour on a part that is not a thing to track (a sprinkler pipe, a dock
bumper, a fire extinguisher).
"""

from __future__ import annotations

from collections.abc import Callable

import bpy
from materials_city import lin
from shader_nodes import G, new_material, principled

PREFIX = "dress:"
Builder = Callable[[G, bpy.types.Node, bpy.types.NodeSocket], None]
BUILDERS: dict[str, Builder] = {}


def _material(fn: Builder) -> Builder:
    BUILDERS[fn.__name__] = fn
    return fn


def _tone(g: G, obj, scale: float, amount: float, detail: float = 3.0):
    """A scalar around 1 (1 +- `amount` / 2) that varies `scale` times a metre: a paint's or a plaster's patchiness."""
    return g.math("MULTIPLY_ADD", g.noise(obj, scale, detail, 0.5), amount, 1.0 - amount / 2)


def _paint(g: G, p, obj, rgb, rough: float, patch: float = 0.06, orange_peel: float = 0.1) -> None:
    """A painted surface: the colour with faint large patches, a semi-gloss and a fine orange-peel bump."""
    g.put(p, "Base Color", g.vmath("SCALE", lin(rgb), scale=_tone(g, obj, 0.8, patch)))
    g.put(p, "Roughness", g.math("MULTIPLY_ADD", g.noise(obj, 25.0, 3.0), 0.12, rough - 0.06))
    g.put(p, "Normal", g.bump(g.noise(obj, 220.0, 2.0, 0.6), orange_peel, 0.0006))


def _metal(g: G, p, obj, rgb, rough: float, spread: float = 0.1, streak: float = 0.0, metallic: float = 0.7) -> None:
    """A bare metal: its F0 colour, roughness varying by `spread` and, if `streak`, brushed along z. A building's metal
    reflects a dark room, so a fully metallic surface would render near black; `metallic` below 1 keeps it a grey plate
    with a sheen (a real one is dusty and oxidised, which is the same thing)."""
    g.put(p, "Base Color", lin(rgb))
    g.put(p, "Metallic", metallic)
    brushed = g.noise(g.scaled(obj, 1.0, 1.0, 60.0), 80.0, 2.0, 0.6) if streak else g.noise(obj, 30.0, 3.0)
    g.put(p, "Roughness", g.math("MULTIPLY_ADD", brushed, spread, rough - spread / 2))
    if streak:
        g.put(p, "Normal", g.bump(brushed, streak, 0.0004))


# -- walls and trim ------------------------------------------------------------------------------------------------
@_material
def plaster_wall(g, p, obj):
    """Painted plaster: warm off-white, faint broad patches and a fine grain, matt."""
    grain = g.math("MULTIPLY_ADD", g.noise(obj, 60.0, 5.0, 0.6), 0.06, 0.97)
    tone = g.math("MULTIPLY", _tone(g, obj, 0.7, 0.10), grain)
    g.put(p, "Base Color", g.vmath("SCALE", lin((0.88, 0.86, 0.81)), scale=tone))
    g.put(p, "Roughness", 0.9)
    g.put(p, "Normal", g.bump(g.noise(obj, 40.0, 4.0, 0.6), 0.15, 0.002))


@_material
def painted_wood_trim(g, p, obj):
    """Satin white paint on wood (baseboards, casings, sills), a faint grain in the bump."""
    _paint(g, p, obj, (0.93, 0.92, 0.89), 0.38, patch=0.03, orange_peel=0.05)
    g.put(p, "Normal", g.bump(g.noise(g.scaled(obj, 1.0, 1.0, 14.0), 50.0, 3.0, 0.6), 0.12, 0.0008))


@_material
def white_laminate(g, p, obj):
    """Glossy white cabinet laminate."""
    g.put(p, "Base Color", g.vmath("SCALE", lin((0.92, 0.92, 0.90)), scale=_tone(g, obj, 1.0, 0.03)))
    g.put(p, "Roughness", 0.3)


@_material
def concrete_wall(g, p, obj):
    """Cast concrete on a wall (a dado): pale, mottled, a pitted fine bump and no joints."""
    pits = g.math("MULTIPLY_ADD", g.noise(obj, 90.0, 6.0), 0.18, 0.91)
    tone = g.math("MULTIPLY", _tone(g, obj, 0.5, 0.22, 4.0), pits)
    g.put(p, "Base Color", g.vmath("SCALE", lin((0.60, 0.60, 0.58)), scale=tone))
    g.put(p, "Roughness", 0.78)
    g.put(p, "Normal", g.bump(g.noise(obj, 120.0, 6.0, 0.7), 0.35, 0.0015))


@_material
def concrete_floor(g, p, obj):
    """Floor-grade concrete on a piece (a plinth, a leveler's pit, a pad): the slab's two greys mottled, rough, no
    joints. The warehouse's own floor is `materials.concrete_material`."""
    patches, grain = g.noise(obj, 0.45, 4.0), g.noise(obj, 90.0, 6.0)
    mix = g.math("ADD", g.math("MULTIPLY", patches, 0.7), g.math("MULTIPLY", grain, 0.3))
    g.put(p, "Base Color", g.ramp(mix, [(0.3, (0.19, 0.19, 0.185)), (0.7, (0.30, 0.295, 0.285))]))
    g.put(p, "Roughness", 0.55)
    g.put(p, "Normal", g.bump(g.noise(obj, 90.0, 6.0), 0.2, 0.002))


# -- plastics and rubber -------------------------------------------------------------------------------------------
@_material
def white_plastic(g, p, obj):
    """A moulded housing (a switch plate, a detector): off-white, semi-gloss."""
    g.put(p, "Base Color", g.vmath("SCALE", lin((0.92, 0.92, 0.90)), scale=_tone(g, obj, 2.0, 0.03)))
    g.put(p, "Roughness", g.math("MULTIPLY_ADD", g.noise(obj, 40.0, 2.0), 0.08, 0.31))


@_material
def black_plastic(g, p, obj):
    """A bezel or a housing: near black, semi-gloss, albedo about 0.03 as real black plastic."""
    g.put(p, "Base Color", lin((0.17, 0.17, 0.18)))
    g.put(p, "Roughness", g.math("MULTIPLY_ADD", g.noise(obj, 40.0, 2.0), 0.08, 0.36))


@_material
def rubber_black(g, p, obj):
    """Rubber (a bumper, a dock seal, a cable sheath): black, rough, a fine pitted bump."""
    g.put(p, "Base Color", lin((0.07, 0.07, 0.075)))
    g.put(p, "Roughness", g.math("MULTIPLY_ADD", g.noise(obj, 60.0, 3.0), 0.15, 0.78))
    g.put(p, "Normal", g.bump(g.noise(obj, 150.0, 4.0, 0.7), 0.3, 0.001))


# -- metals --------------------------------------------------------------------------------------------------------
@_material
def brushed_steel(g, p, obj):
    """Stainless or brushed steel: a pale F0 colour, roughness 0.3 to 0.4 in streaks."""
    _metal(g, p, obj, (0.70, 0.70, 0.72), 0.36, spread=0.12, streak=0.25)


@_material
def chrome(g, p, obj):
    """Polished chrome: nearly a mirror of a dim room, roughness 0.12."""
    _metal(g, p, obj, (0.78, 0.79, 0.80), 0.12, spread=0.05, metallic=0.9)


@_material
def brass(g, p, obj):
    """Warm brass: a hook, a number plate."""
    _metal(g, p, obj, (0.80, 0.60, 0.30), 0.3, spread=0.1)


@_material
def galvanised_steel(g, p, obj):
    """Hot-dip galvanised steel: a zinc spangle of small crystals, each a little lighter or darker, roughness 0.4."""
    cells = g.node("ShaderNodeTexVoronoi", {"Vector": obj, "Scale": 55.0, "Randomness": 1.0}).outputs["Color"]
    value = g.math("MULTIPLY_ADD", g.node("ShaderNodeSeparateColor", {"Color": cells}).outputs["Red"], 0.22, 0.89)
    g.put(p, "Base Color", g.vmath("SCALE", lin((0.64, 0.66, 0.68)), scale=value))
    g.put(p, "Metallic", 0.6)
    g.put(p, "Roughness", g.math("MULTIPLY_ADD", g.noise(obj, 20.0, 3.0), 0.12, 0.34))


@_material
def corrugated_steel(g, p, obj):
    """Insulated wall cladding or roof deck: painted light grey-white, trapezoidal ribs 10 cm apart across y."""
    height = g.smooth(g.triangle(g.xyz(obj)[1], 0.10), 0.1, 0.55)
    g.put(p, "Base Color", g.vmath("SCALE", lin((0.80, 0.81, 0.80)), scale=_tone(g, obj, 0.9, 0.05)))
    g.put(p, "Roughness", 0.48)
    g.put(p, "Normal", g.bump(height, 0.9, 0.012))


@_material
def diamond_plate(g, p, obj):
    """Tread plate (a dock leveler): worn steel, half metal, with a diamond lattice of raised studs, 2.2 cm pitch."""
    x, y, _ = g.xyz(obj)
    u, v = g.math("ADD", x, y), g.math("SUBTRACT", x, y)
    stud = g.math("MULTIPLY", g.sine(u, 0.031), g.sine(v, 0.031))
    g.put(p, "Base Color", lin((0.60, 0.62, 0.64)))
    g.put(p, "Metallic", 0.5)
    g.put(p, "Roughness", g.math("MULTIPLY_ADD", g.noise(obj, 30.0, 3.0), 0.15, 0.38))
    g.put(p, "Normal", g.bump(g.smooth(stud, 0.35, 0.9), 0.7, 0.0025))


# -- painted metal -------------------------------------------------------------------------------------------------
@_material
def painted_metal_red(g, p, obj):
    """Fire-equipment red: the sprinkler pipes, an extinguisher."""
    _paint(g, p, obj, (0.60, 0.05, 0.04), 0.42)


@_material
def painted_metal_grey(g, p, obj):
    """Grey enamel: ducts, cable trays, enclosures."""
    _paint(g, p, obj, (0.45, 0.46, 0.48), 0.5)


@_material
def painted_metal_white(g, p, obj):
    """White enamel: a radiator, a pipe, a bracket."""
    _paint(g, p, obj, (0.90, 0.90, 0.88), 0.4, patch=0.03)


@_material
def safety_yellow(g, p, obj):
    """OSHA safety yellow on steel, a little dulled by a working life (guards, bollards, a wall band)."""
    _paint(g, p, obj, (0.88, 0.68, 0.06), 0.45)


# -- cloth, paper, ceramic, glass ----------------------------------------------------------------------------------
@_material
def curtain_fabric(g, p, obj):
    """A muted linen: pleats across y in the normal (so a flat panel hangs in folds), a woven bump, a soft sheen."""
    x, y, z = g.xyz(obj)
    drift = g.math("MULTIPLY", g.noise(obj, 3.0, 1.0), 0.05)
    fold = g.sine(g.math("ADD", y, drift), 0.16)
    weave = g.math("MULTIPLY", g.sine(y, 0.0022), g.sine(z, 0.0022))
    slub = g.noise(g.scaled(obj, 1.0, 1.0, 25.0), 120.0, 3.0, 0.6)
    tone = g.math("MULTIPLY", g.math("MULTIPLY_ADD", fold, 0.22, 0.85), g.math("MULTIPLY_ADD", slub, 0.12, 0.94))
    g.put(p, "Base Color", g.vmath("SCALE", lin((0.74, 0.71, 0.65)), scale=tone))
    g.put(p, "Roughness", 0.85)
    g.put(p, "Sheen Weight", 0.4)
    g.put(p, "Normal", g.bump(g.math("ADD", g.math("MULTIPLY", fold, 0.9), g.math("MULTIPLY", weave, 0.1)), 0.8, 0.02))


@_material
def label_paper(g, p, obj):
    """A printed label: white paper, a slight fibre texture."""
    g.put(p, "Base Color", g.vmath("SCALE", lin((0.90, 0.90, 0.87)), scale=_tone(g, obj, 8.0, 0.04, 2.0)))
    g.put(p, "Roughness", 0.6)
    g.put(p, "Normal", g.bump(g.noise(g.scaled(obj, 1.0, 40.0, 40.0), 60.0, 2.0), 0.08, 0.0005))


@_material
def cardboard(g, p, obj):
    """Corrugated cardboard: kraft brown with patches, a 1 cm flute print-through across y and a fibre grain."""
    flute = g.triangle(g.xyz(obj)[1], 0.011)
    tone = g.math("MULTIPLY", _tone(g, obj, 4.0, 0.16), g.math("MULTIPLY_ADD", flute, 0.04, 0.98))
    g.put(p, "Base Color", g.vmath("SCALE", lin((0.66, 0.53, 0.37)), scale=tone))
    g.put(p, "Roughness", 0.88)
    g.put(p, "Normal", g.bump(g.math("ADD", g.math("MULTIPLY", flute, 0.6), g.noise(obj, 250.0, 2.0)), 0.5, 0.0008))


@_material
def ceramic_tile(g, p, obj):
    """Glazed 15 x 7.5 cm subway tiles in a half-brick bond, each a little different, with grey grout (off y, z)."""
    x, y, z = g.xyz(obj)
    plane = g.combine(y, z, 0.0)
    brick = g.node("ShaderNodeTexBrick", {"Vector": plane, "Scale": 1.0, "Mortar Size": 0.004, "Mortar Smooth": 0.1,
                                           "Bias": 0.0, "Brick Width": 0.15, "Row Height": 0.075,
                                           "Color1": lin((0.90, 0.90, 0.88)), "Color2": lin((0.87, 0.88, 0.87)),
                                           "Mortar": lin((0.60, 0.58, 0.54))}, offset=0.5, offset_frequency=2,
                   squash=1.0, squash_frequency=2)
    tile = brick.outputs["Fac"]  # 1 on a tile, 0 in the mortar
    g.put(p, "Base Color", brick.outputs["Color"])
    g.put(p, "Roughness", g.math("MULTIPLY_ADD", tile, -0.6, 0.78))  # glaze 0.18, grout 0.78
    g.put(p, "Coat Weight", 0.2)
    g.put(p, "Normal", g.bump(tile, 0.6, 0.0012))


@_material
def glass_panel(g, p, obj):
    """Frosted glass (a cabinet door, a shower screen, a picture glass): rough enough that what lies behind is not
    recognisable, which agrees with the ID pass; half of its light is scattered in its own colour."""
    g.put(p, "Base Color", lin((0.90, 0.95, 0.95)))
    g.put(p, "Roughness", 0.45)
    g.put(p, "IOR", 1.45)
    g.put(p, "Transmission Weight", 0.5)


# -- lights --------------------------------------------------------------------------------------------------------
def _camera_only(g: G, strength: float):
    """An emission strength that only the camera sees: the lens shows lit and casts nothing, because the scene's real
    lights (`lighting.py`) are the only light and a visible lens must not count the same light twice."""
    return g.math("MULTIPLY", g.node("ShaderNodeLightPath").outputs["Is Camera Ray"], strength)


def _glow(g: G, p, kelvin: float, strength: float) -> None:
    """A lit diffuser: a white base and the glow of a black body of `kelvin`, seen by the camera alone."""
    g.put(p, "Base Color", lin((0.80, 0.80, 0.78)))
    g.put(p, "Roughness", 0.5)
    g.put(p, "Emission Color", g.node("ShaderNodeBlackbody", {"Temperature": kelvin}).outputs["Color"])
    g.put(p, "Emission Strength", _camera_only(g, strength))


@_material
def led_emitter(g, p, obj):
    """A neutral-white (4000 K) LED lens."""
    _glow(g, p, 4000.0, 3.0)


HOME_LENS_STRENGTH = 55.0
# The home's lamp lens is lit with its own lamp's light, as a physical lens is. A lamp of `lm` lumens is `lm * pi /
# UNIT_LUX` render watts (`lighting.render_watts`, UNIT_LUX 240) and a Lambertian emitter of that power over a lens of
# area A has the radiance, which is a mesh emission's strength, lm / (UNIT_LUX * A). The flush lamps
# (`dressing_build_standins._lamp_flush`) have a lens of radius 0.198 m, A = 0.123 m2, and run 610 lm (the hall) to
# 4,860 (the kitchen): strengths 21 to 165, which `wear.light_lenses` sets for each lamp from the rig's own record. A
# material alone cannot know its lamp, so its default is the median lamp's, 1,600 lm (the bedroom's and bathroom's): 54,
# here 55. The 3.0 it had was a lamp of 90 lm: its disc rendered peach and never read as lit.


@_material
def led_emitter_warm(g, p, obj):
    """A warm-white (3000 K) lamp or fixture, the home's, lit as the median lamp is (`HOME_LENS_STRENGTH`)."""
    _glow(g, p, 3000.0, HOME_LENS_STRENGTH)


@_material
def led_emitter_cool(g, p, obj):
    """A cool-white (5000 K) high-bay lens, the warehouse's."""
    _glow(g, p, 5000.0, 4.0)


@_material
def skylight_panel(g, p, obj):
    """A translucent roof light: milky and glowing with daylight, to the camera alone."""
    g.put(p, "Base Color", lin((0.90, 0.92, 0.90)))
    g.put(p, "Roughness", 0.5)
    g.put(p, "Emission Color", g.node("ShaderNodeBlackbody", {"Temperature": 6500.0}).outputs["Color"])
    g.put(p, "Emission Strength", _camera_only(g, 2.0))


NAMES = tuple(BUILDERS)


def build(name: str) -> bpy.types.Material:
    """The material `name` of `NAMES`, new each call (`materials.named` caches it)."""
    if name not in BUILDERS:
        raise KeyError(f"unknown dressing material {name!r}; known: {list(NAMES)}")
    m = new_material(PREFIX + name)
    g, p = G(m.node_tree), principled(m)
    BUILDERS[name](g, p, g.node("ShaderNodeTexCoord").outputs["Object"])
    return m
