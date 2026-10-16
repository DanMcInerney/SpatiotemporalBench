# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Entities and surfaces from the asset library, for `--appearance assets`.

Runs inside Blender's own Python and is never imported by `stbench`. `Appearance` appends what an episode needs from
the library (`bpy.data.libraries.load(link=False)`, well under a second) and then
- makes one object per carried, light, mover or clutter entity that shares its asset's mesh, with `pass_index` on that
  one object, and per-object material slots: one tinted copy per (asset, slot, colour), shared by every identical
  entity, so identical items are identical in every pixel statistic;
- resolves the palette keys of the M1 structure and fixed furniture (`shapes.material`'s `resolve` hook) to textured
  materials: the manifest's `palette_materials`, walls and ceilings, doors, and a few kinds' surfaces, tinted where the
  key is a colour; and gives floors their materials (a library material, or a procedural shader such as the warehouse's
  concrete).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import bpy
import floor_look
import materials as M
import numpy as np
from shapes import PALETTE, srgb_to_linear

# Fixed-furniture kinds whose colour-keyed parts use a tinted texture (the manifest's `uses` hints: rugs, floor mat,
# sofa, bed linen, sanitaryware). The colour attribute still sets the colour.
KIND_SURFACES = {"sofa": "acg_Fabric030", "rug": "acg_Carpet001", "floor_mat": "acg_Carpet001",
                 "bed": "acg_Fabric036", "toilet": "acg_Porcelain001", "vanity": "acg_Porcelain001",
                 "bathtub": "acg_Porcelain001"}
# Palette keys that are painted textures: key -> "walls" (the manifest's wall material) or a palette material key.
PAINTED_KEYS = {"wall": "walls", "ceiling": "walls", "door": "oak", "door_entrance": "oak"}
# Kinds whose woven surface shows less chroma and a darker tone than the palette's colour (a rug is dyed wool, not a
# signal): kind -> (the share of the colour's chroma kept, the tone). The palette blue of the living-room rug was a
# saturated screen blue in every frame; this makes it a dusty denim (still the palette's blue by hue).
MUTED = {"rug": (0.5, 0.5), "floor_mat": (0.65, 0.8)}
ANISOTROPY_SLACK = 1e-6
Resolver = Callable[[str], "bpy.types.Material | None"]


def linear(colour: str, factor: float = 1.0, lift: float = 0.0) -> tuple[float, float, float]:
    """A palette colour name as linear RGB, scaled by `factor` (clipped at 1) after moving it the fraction `lift` of the
    way to white in sRGB (the lighter tint of the cart's second set of umbrella gores)."""
    if colour not in PALETTE:
        raise KeyError(f"colour attribute {colour!r} has no palette entry; known: {sorted(PALETTE)}")
    return tuple(min(1.0, srgb_to_linear(min(1.0, (c + (1.0 - c) * lift) * factor))) for c in PALETTE[colour][0])


def muted(rgb: tuple[float, float, float], chroma: float, tone: float = 1.0) -> tuple[float, float, float]:
    """A linear colour with its chroma cut to the share `chroma` of what it was (moved towards the grey of its own
    luminance: the hue stays) and then scaled by `tone` (its brightness)."""
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return tuple((lum + (c - lum) * chroma) * tone for c in rgb)


Box = tuple[float, float, float]


def squash(asset, native: Box, size: Box) -> float:
    """How much fitting the asset's native box to an entity's box distorts it: `max(s) / min(s) - 1` for a `per_axis`
    fit, 0 for the others (`uniform` keeps the proportions, `exact` is built to the box)."""
    s = [want / have for want, have in zip(size, native, strict=True)]
    return max(s) / min(s) - 1.0 if (asset.fit or {}).get("mode") == "per_axis" else 0.0


def fit_scale(asset, native: Box, size: Box, strict: bool = True) -> Box:
    """The object scale that fits the asset's native box to an entity's box: per axis (`per_axis`, `exact`) or by
    height alone (`uniform`). A `per_axis` asset may not be squashed more than its `fit.max_anisotropy` (V0: the
    catalog size is what renders, so a box that disagrees with the mesh is an error, not a quiet distortion). The
    asset preview passes `strict=False` to see an asset whose catalog box has not been refitted yet."""
    s = tuple(want / have for want, have in zip(size, native, strict=True))
    if (asset.fit or {}).get("mode") == "uniform":
        return (s[2], s[2], s[2])
    worst, limit = squash(asset, native, size), float((asset.fit or {}).get("max_anisotropy", 0.0))
    if strict and worst > limit + ANISOTROPY_SLACK:
        raise ValueError(f"{asset.id}: fitting its {[round(n, 3) for n in native]} m box to {list(size)} squashes it "
                         f"by {worst:.2f}, more than the manifest's {limit}; the catalog size must follow the asset")
    return s


class Appearance:
    def __init__(self, manifest, library: Path, asset_ids):
        self.m, self.library = manifest, Path(library)
        self.ids = sorted(set(asset_ids))
        self.meshes: dict[str, bpy.types.Mesh] = {}
        self.materials: dict[str, bpy.types.Material] = {}
        self.native: dict[str, Box] = {}
        self._tinted: dict[tuple, bpy.types.Material] = {}
        self._dark: dict[tuple, bpy.types.Material] = {}
        self._shaders: dict[str, bpy.types.Material] = {}
        self._append()

    def _append(self) -> None:
        shared = {a.id for a in self.m.assets if a.role == "material"}
        with bpy.data.libraries.load(str(self.library), link=False) as (src, dst):
            missing = [i for i in self.ids if f"M:{i}" not in src.meshes]
            if missing:
                raise LookupError(f"the library {self.library.name} has no mesh for {missing}; rebuild it")
            dst.meshes = [f"M:{i}" for i in self.ids]
            dst.materials = [n for n in src.materials if n in shared or n.split(":", 1)[0] in self.ids]
            dst.node_groups = [n for n in src.node_groups if n == M.TINT_GROUP]
        self.meshes = {i: me for i, me in zip(self.ids, dst.meshes, strict=True)}
        self.materials = {mat.name: mat for mat in dst.materials}
        for i, me in self.meshes.items():
            co = np.empty(len(me.vertices) * 3)
            me.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3)
            self.native[i] = tuple(float(x) for x in co.max(axis=0) - co.min(axis=0))

    # -- entities --------------------------------------------------------------------------------------------
    def _slot_material(self, a, base: bpy.types.Material, colour: str) -> bpy.types.Material:
        tint = a.tint or {}
        slot = base.name.split(":", 1)[1]
        wanted = tint.get("materials")
        if tint.get("mode", "none") == "none" or wanted is None or not ("*" in wanted or slot in wanted):
            return base
        if tint["mode"] == "swap" and colour in self.m.palette_materials:
            return self.materials[self.m.palette_materials[colour]]
        key = (a.id, slot, colour)
        if key not in self._tinted:
            mat = base.copy()
            mat.name = f"{base.name}@{colour}"
            M.set_tint_target(mat, linear(colour, lift=float((tint.get("lighten") or {}).get(slot, 0.0))))
            self._tinted[key] = mat
        return self._tinted[key]

    def make_object(self, asset_id: str, kind: str, colour: str, size, name: str, pass_index: int,
                    strict: bool = True) -> bpy.types.Object:
        """One object for an entity: the asset's shared mesh, scaled to the entity's box (`strict` refuses a squash past
        the manifest's limit), its slots' materials chosen by the colour attribute."""
        a = self.m.asset(asset_id)
        if a.kind != kind:
            raise ValueError(f"entity of kind {kind!r} has asset {asset_id!r}, which is a {a.kind!r}")
        me = self.meshes[asset_id]
        ob = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(ob)
        ob.pass_index = pass_index
        ob.scale = fit_scale(a, self.native[asset_id], tuple(size), strict)
        for i, base in enumerate(me.materials):
            ob.material_slots[i].link = "OBJECT"
            ob.material_slots[i].material = self._slot_material(a, base, colour)
        return ob

    # -- surfaces --------------------------------------------------------------------------------------------
    def floor_material(self, kind: str) -> bpy.types.Material | None:
        """The floor of a region kind (`look.kind_of`): a library material, or a procedural shader (`shader:concrete`)
        built once and shared by every region that names it. None where the manifest names no floor for the kind (the
        placeholder floor stays): the shipped manifest names one for every kind of every catalog
        (`tests/test_render_looks.py`), and a kind no catalog has never gets this far (`build_scene` raises first)."""
        ref = self.m.floors.get(kind)
        if not ref:
            return None
        if ref.startswith("shader:"):
            if ref not in self._shaders:
                self._shaders[ref] = M.shader_material(ref.split(":", 1)[1])
            return self._shaders[ref]
        mat = self.materials[ref]
        if ref in floor_look.RELIEF_M:  # a tile or wood floor: true relief where a bump reads flat (`floor_look`)
            floor_look.displace(mat, floor_look.RELIEF_M[ref])
        return mat

    def _shared(self, asset_id: str | None) -> bpy.types.Material | None:
        return self.materials.get(asset_id) if asset_id else None

    def _textured(self, asset_id: str, factor: float) -> bpy.types.Material | None:
        base = self._shared(asset_id)
        if base is None or factor == 1.0:
            return base
        if (asset_id, factor) not in self._dark:
            self._dark[(asset_id, factor)] = M.darkened(base, factor)
        return self._dark[(asset_id, factor)]

    def _painted(self, asset_id: str | None, colour: str, factor: float, mute: tuple[float, float] | None = None
                 ) -> bpy.types.Material | None:
        """A texture tinted to a palette colour: the texture's variation at the colour's mean albedo, the colour
        muted (`mute`: the share of its chroma kept and its tone, `muted`) where it is a dyed weave."""
        base = self._shared(asset_id)
        if base is None:
            return None
        key = (asset_id, colour, factor, mute)
        if key not in self._tinted:
            mat = base.copy()
            mat.name = f"{base.name}@{colour}*{factor:g}" + ("~muted" if mute else "")
            M.add_tint(mat, base[M.MEAN_PROP])
            M.set_tint_target(mat, muted(linear(colour, factor), *mute) if mute else linear(colour, factor))
            self._tinted[key] = mat
        return self._tinted[key]

    def resolver(self, kind: str | None = None, colour: str | None = None) -> Resolver:
        """`key -> Material | None` for a part's palette key ("oak", "grey*0.8"): textured where the manifest says so,
        None where the plain palette material stays."""
        pal = self.m.palette_materials

        def resolve(key: str):
            base, _, f = key.partition("*")
            factor = float(f) if f else 1.0
            if kind in KIND_SURFACES and base == colour:
                return self._painted(KIND_SURFACES[kind], base, factor, MUTED.get(kind))
            if base in pal:
                return self._textured(pal[base], factor)
            if base in PAINTED_KEYS:
                target = PAINTED_KEYS[base]
                return self._painted(self.m.walls if target == "walls" else pal.get(target), base, factor)
            return None

        return resolve
