# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The dressing models of the asset library, by id, for `--appearance assets` (`asset:<id>` recipes).

Runs inside Blender's own Python and is never imported by `stbench`. `DressingAssets(manifest, library).asset(id)` is
the `asset` of `dressing.Ctx`: a new object that shares the library's one mesh of that model (appended on first use,
well under a second), at scale 1, `pass_index` 0 (dressing is never an entity), not linked to any collection, or None
where the library holds no such model (the builder of the same class then draws its procedural stand-in, so a scene
builds without downloads). The mesh is the model as Poly Haven publishes it: standing on z = 0, centred on x and y, its
width along x, its depth along y and its height along z, the front facing -y; `native(id)` is that box. Fitting it to
a `Dressing`'s box and facing it is the caller's.
"""

from __future__ import annotations

from pathlib import Path

import bpy
import numpy as np

ROLE = "dressing"


class DressingAssets:
    def __init__(self, manifest, library: Path):
        self.m, self.library = manifest, Path(library)
        wanted = {a.id for a in manifest.assets if a.role == ROLE}
        with bpy.data.libraries.load(str(self.library), link=False) as (src, _):  # only the names: nothing is appended
            self.in_library = {n[2:] for n in src.meshes if n.startswith("M:")} & wanted
        self._meshes: dict[str, bpy.types.Mesh] = {}

    def has(self, asset_id: str) -> bool:
        return asset_id in self.in_library

    def _mesh(self, asset_id: str) -> bpy.types.Mesh:
        if asset_id not in self._meshes:
            with bpy.data.libraries.load(str(self.library), link=False) as (src, dst):
                dst.meshes = [f"M:{asset_id}"]
                dst.materials = [n for n in src.materials if n.split(":", 1)[0] == asset_id]
            self._meshes[asset_id] = dst.meshes[0]
        return self._meshes[asset_id]

    def native(self, asset_id: str) -> tuple[float, float, float]:
        """The model's box in metres, (width, depth, height)."""
        me = self._mesh(asset_id)
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        return tuple(float(x) for x in co.max(axis=0) - co.min(axis=0))

    def asset(self, asset_id: str) -> bpy.types.Object | None:
        if not self.has(asset_id):
            return None
        ob = bpy.data.objects.new(f"dressing_asset_{asset_id}", self._mesh(asset_id))
        ob.pass_index = 0
        return ob
