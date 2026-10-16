# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Draw a map's render-only dressing: trim, fittings, fixtures and structure that no
timeline holds.

Runs inside Blender's own Python and is never imported by `stbench`. `build(spec, catalog, ctx)` takes the map spec
dict, the catalog name and a `Ctx`; it asks `dressing_layout_<DRESSING>.layout(spec)` what the map has (`DRESSING` is
the catalog's `look_<catalog>.DRESSING`, `None` for a catalog that has no dressing; a catalog with no look module is an
error) and makes one object set per `Dressing`, all `pass_index` 0 (in a realism probe each piece
has its own from `DEBUG_INDEX_FROM` up, mapped to its class in `INDEX`) in one "dressing" collection, named
`dressing_<cls>_<n>` (`n` counts the pieces of a class from 0; a second object of one piece adds `_<k>`). It never
becomes an entity: nothing here is in `instance_map`.

How a piece is drawn follows its recipe. `proc:<builder>[:<text>]` calls `dressing_build_<DRESSING>.BUILDERS[builder]`,
`fn(d, ctx, text) -> [objects]`, which builds the piece in its own frame (origin the centre of its base, x facing out of
the wall, y across, z up, extents `d.size`) and returns unlinked objects with an identity transform; this module places
them at `d.pos` turned by `d.yaw_deg`. `asset:<id>` takes the model `ctx.asset(id)` as a linked duplicate (the mesh is
shared) fitted to the box: size is (depth, width, height) and a Poly Haven model is width x depth x height facing -y, so
the model is turned a quarter to face the piece's x, scaled axis by axis to the box and set on its base centre;
where the library has no such model the builder of the same name as the class draws a procedural stand-in, so a
scene builds with no download. Identical pieces (same recipe, size and text) share one mesh. A scene that holds a film
(dust, a scuff: `dressing_materials.decal`) takes depth and index from the first surface whatever its alpha, so they do
not depend on the sample count.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import asset_dressing  # noqa: E402
import dressing_materials  # noqa: E402
import look  # noqa: E402
from dressing_spec import Dressing  # noqa: E402

COLLECTION = "dressing"
DEBUG_INDEX_FROM: int | None = None  # a probe sets it (`realism_probe`): each piece then has its own index from here up
INDEX: dict[int, str] = {}  # what `build` gave them: a piece's index -> its class, the probe's `dressing_index.json`


class DressingError(LookupError):
    """A recipe nobody can draw: the catalog has no builders module, or the builder named is not in it."""


class Ctx:
    """What a builder may ask of the render: materials by name and models by id.

    `app` is the scene's `asset_entities.Appearance` (None for placeholders: then no model is found and every asset
    piece is its procedural stand-in). `material(name)` is the render library's `materials.dressing_material(name)`
    (lane M: named PBR materials, called when the function exists) or `named(name)` where one is given, else this
    package's `dressing_materials.FALLBACKS`; one material is made per name. `asset(id)` is `asset_dressing`'s object
    holding the model's mesh (lane A: the library's model of that id, unlinked, at scale 1), or None when the library
    does not have it; `build` calls `release` when it is done, which removes those objects (the pieces keep the
    meshes)."""

    def __init__(self, app=None, *, named=None):
        self.named = named
        self._models = None if app is None else asset_dressing.DressingAssets(app.m, app.library)
        self._materials: dict[str, bpy.types.Material] = {}
        self._assets: dict[str, bpy.types.Object | None] = {}

    def material(self, name: str) -> bpy.types.Material:
        if name not in self._materials:
            lookup = self.named
            if lookup is None:
                import materials  # not at the top: this module must import where `materials` is not on the path

                lookup = getattr(materials, "dressing_material", None)
            found = lookup(name) if lookup else None
            if found is None:
                if name not in dressing_materials.FALLBACKS:
                    raise KeyError(f"no dressing material {name!r}; known: {sorted(dressing_materials.FALLBACKS)}")
                found = dressing_materials.FALLBACKS[name]()
            self._materials[name] = found
        return self._materials[name]

    def asset(self, asset_id: str) -> bpy.types.Object | None:
        if asset_id not in self._assets:
            self._assets[asset_id] = None if self._models is None else self._models.asset(asset_id)
        return self._assets[asset_id]

    def release(self) -> None:
        """Remove the template objects `asset` made, so no loose object is left in the file."""
        for ob in self._assets.values():
            if ob is not None:
                bpy.data.objects.remove(ob)
        self._assets.clear()


# -- one piece ---------------------------------------------------------------------------------------------------
def extents(mesh: bpy.types.Mesh) -> tuple[Vector, Vector]:
    """The lowest and highest corner of a mesh's vertices."""
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    return Vector(co.min(axis=0).tolist()), Vector(co.max(axis=0).tolist())


def fitted(d: Dressing, template: bpy.types.Object, name: str) -> bpy.types.Object:
    """A linked duplicate of the template's mesh in the Dressing's box. The model's width (x) runs across the piece (y),
    its depth (y) out of the wall (x) and it faces -y, so a quarter turn after `yaw_deg` makes it face the piece's x."""
    lo, hi = extents(template.data)
    native = hi - lo
    depth, width, height = d.size
    scale = Vector((width / native.x, depth / native.y, height / native.z))
    turn = Matrix.Rotation(math.radians(d.yaw_deg + 90.0), 3, "Z")
    base = Vector(((lo.x + hi.x) / 2 * scale.x, (lo.y + hi.y) / 2 * scale.y, lo.z * scale.z))
    ob = bpy.data.objects.new(name, template.data)
    ob.scale = scale
    ob.rotation_euler = (0.0, 0.0, math.radians(d.yaw_deg + 90.0))
    ob.location = Vector(d.pos) - turn @ base
    return ob


def builders_of(catalog: str) -> dict:
    """The `BUILDERS` of `dressing_build_<DRESSING>`, or a `DressingError` that says which module is missing."""
    suffix = look.module(catalog).DRESSING
    if suffix is None:
        raise DressingError(f"the {catalog} catalog declares no dressing (look_{catalog}.DRESSING is None)")
    module = f"dressing_build_{suffix}"
    if importlib.util.find_spec(module) is None:
        raise DressingError(f"the {catalog} catalog has a dressing layout but no {module}.py with its BUILDERS")
    return importlib.import_module(module).BUILDERS


def layout_of(catalog: str, spec: dict) -> list[Dressing]:
    """What the catalog's layout (`dressing_layout_<DRESSING>`, named by its look module) puts in the map. A catalog
    that declares `DRESSING = None` has no dressing; one with no look module raises (`look.LookError`)."""
    suffix = look.module(catalog).DRESSING
    if suffix is None:
        return []
    return importlib.import_module(f"dressing_layout_{suffix}").layout(spec)


def parse(recipe: str) -> tuple[str, str, str | None]:
    """`proc:<builder>[:<text>]` or `asset:<id>` as (kind, builder or id, text)."""
    kind, _, rest = recipe.partition(":")
    if kind == "proc":
        name, _, text = rest.partition(":")
        return kind, name, text or None
    if kind == "asset" and rest:
        return kind, rest, None
    raise DressingError(f"recipe {recipe!r} is neither proc:<builder>[:<text>] nor asset:<id>")


def draw(d: Dressing, ctx: Ctx, builders: dict) -> list[bpy.types.Object]:
    """The objects a procedural builder makes for a piece (a missing asset's stand-in is the builder of the class's
    name), in the piece's own frame."""
    kind, key, text = parse(d.recipe)
    builder = key if kind == "proc" else d.cls
    if builder not in builders:
        raise DressingError(f"{d.cls}: no builder {builder!r} for recipe {d.recipe!r}; known: {sorted(builders)}")
    return builders[builder](d, ctx, text)


def collection() -> bpy.types.Collection:
    """The scene's "dressing" collection, empty and linked (a build starts it afresh)."""
    coll = bpy.data.collections.get(COLLECTION) or bpy.data.collections.new(COLLECTION)
    for ob in list(coll.objects):
        bpy.data.objects.remove(ob)
    if coll.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(coll)
    return coll


def build(spec: dict, catalog: str, ctx: Ctx) -> list[bpy.types.Object]:
    """Every phase 1 dressing object of the map, in layout order, as objects in the "dressing" collection."""
    pieces = layout_of(catalog, spec)
    if not pieces:
        return []
    builders = builders_of(catalog)
    coll, out = collection(), []
    counts: dict[str, int] = {}
    meshes: dict[tuple, list[bpy.types.Mesh]] = {}  # per (recipe, size, class): what an identical piece shares
    INDEX.clear()
    for i, d in enumerate(pieces):
        index = 0 if DEBUG_INDEX_FROM is None else DEBUG_INDEX_FROM + i  # 0 always, but in a probe
        if index:
            INDEX[index] = d.cls
        n = counts[d.cls] = counts.get(d.cls, -1) + 1
        name = f"dressing_{d.cls}_{n}"
        kind, key, _text = parse(d.recipe)
        template = ctx.asset(key) if kind == "asset" else None
        if template is not None:
            objs = [fitted(d, template, name)]
        else:
            sig = (d.recipe, d.size, d.cls)
            if sig in meshes:
                objs = [bpy.data.objects.new(name, me) for me in meshes[sig]]
            else:
                objs = draw(d, ctx, builders)
                meshes[sig] = [ob.data for ob in objs]
                for ob in objs:
                    ob.data.name = name
            for ob in objs:
                ob.location, ob.rotation_euler = d.pos, (0.0, 0.0, math.radians(d.yaw_deg))
        for k, ob in enumerate(objs):
            ob.name = f"{name}_{k}" if k else name
            ob.pass_index = index
            coll.objects.link(ob)
        out += objs
    ctx.release()
    if any(m is not None and dressing_materials.FILM in m for ob in out for m in ob.data.materials):
        # A film is clear in places. By default the depth and index passes take a surface only where its alpha is at
        # least 0.5, which depends on where in the pixel each sample falls, so depth would change with the sample count;
        # at 0 the first surface is the one they record, as for any opaque one (the film is at most 3 mm thick).
        bpy.context.scene.view_layers[0].pass_alpha_threshold = 0.0
    return out
