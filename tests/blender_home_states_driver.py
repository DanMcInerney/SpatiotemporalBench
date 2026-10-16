# SpatiotemporalBench: Blender-side driver of a test.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Runs inside Blender, for `test_home_states_blender.py`: builds the household's scene of a script (the placeholder
drawings of its entities, its doors and, with `dressing.build`, all of the apartment's render-only dressing) and writes
what the test compares as one JSON document: the world box of every dressing object, of every door's leaf slab turned
open and turned shut, and of every entity that has more than one placement, at each time asked for. Not a
pytest module (Blender's Python has no pytest). No render, no GPU and nothing downloaded.

    blender --background --factory-startup --python tests/blender_home_states_driver.py -- \
        --render DIR --src DIR --private SCRIPT_PRIVATE --times 0,7200 --out RESULT.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
for flag in ("--render", "--src", "--private", "--out"):
    ap.add_argument(flag, type=Path, required=True)
ap.add_argument("--times", required=True)
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
sys.path[:0] = [str(args.render), str(args.src)]

import bpy  # noqa: E402
import build_scene  # noqa: E402
import dressing  # noqa: E402
import materials  # noqa: E402
import numpy as np  # noqa: E402
import shapes  # noqa: E402

from stbench.mapspec import load_map  # noqa: E402
from stbench.timeline import load  # noqa: E402


def world_box(ob) -> list[float]:
    """The box of an object's mesh in the map frame (xmin, ymin, zmin, xmax, ymax, zmax), to a tenth of a millimetre."""
    co = np.empty(len(ob.data.vertices) * 3)
    ob.data.vertices.foreach_get("co", co)
    w = np.array(ob.matrix_world)
    pts = co.reshape(-1, 3) @ w[:3, :3].T + w[:3, 3]
    return [round(float(v), 4) for v in (*pts.min(axis=0), *pts.max(axis=0))]


def leaf_box(ob, size) -> list[float]:
    """The box of a door's leaf slab in the map frame: the entity's size (width, thickness, height) laid out as the
    leaf is in its own frame (hinge at the origin, the free edge at +x) and carried by the object's transform. The
    mesh's own box is left out: its two knobs, 6 cm out of each face, would swell the box of the whole leaf."""
    corners = np.array([(x, y, z) for x in (0.0, size[0]) for y in (-size[1] / 2, size[1] / 2) for z in (0.0, size[2])])
    w = np.array(ob.matrix_world)
    pts = corners @ w[:3, :3].T + w[:3, 3]
    return [round(float(v), 4) for v in (*pts.min(axis=0), *pts.max(axis=0))]


def settled() -> None:
    bpy.context.view_layer.update()  # the objects' world matrices, after their transforms were set


def main() -> None:
    tl = load(args.private / "timeline.json")
    m = load_map(tl.episode.map)
    spec = json.loads((args.src / "stbench" / "maps" / f"{m.name}.json").read_text(encoding="utf-8"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    shapes.reset()
    materials.use_map(spec)
    scene = build_scene.build(tl, m, 80.0, None)
    pieces = dressing.build(spec, m.catalog, dressing.Ctx())
    settled()
    out: dict = {"dressing": [], "doors": [], "entities": {}}
    for ob in pieces:
        hit = re.fullmatch(r"dressing_(.+?)_\d+(?:_\d+)?", ob.name)
        out["dressing"].append({"name": ob.name, "cls": hit.group(1) if hit else None, "box": world_box(ob)})
    by_position = {(round(o.center[0], 3), round(o.center[1], 3)): o for o in m.doors()}
    for eid, door in scene.doors.items():
        first = tl.placement_of(eid, 0.0) or tl.placements_of(eid)[0]
        opening = by_position[(round(first.pos[0], 3), round(first.pos[1], 3))]
        row = {"id": opening.id, "entity": eid, "interior": bool(opening.interior)}
        for key, is_open in (("open", True), ("shut", False)):
            door.set_open(is_open)
            settled()
            row[key] = leaf_box(door.ob, tl.entity(eid).size)
        out["doors"].append(row)
    moved = [e.id for e in tl.entities if len(tl.placements_of(e.id)) > 1]
    for t in (float(v) for v in args.times.split(",")):
        scene.apply_time(t)
        settled()
        for eid in moved:
            ob = scene.objects[eid]
            out["entities"].setdefault(eid, {"kind": tl.entity(eid).kind, "boxes": {}})["boxes"][str(t)] = (
                None if ob.hide_render else world_box(ob))
    args.out.write_text(json.dumps(out), encoding="utf-8")
    print(f"STBENCH-STATES {len(out['dressing'])} dressing objects, {len(out['doors'])} doors, {len(moved)} moved",
          flush=True)


main()
