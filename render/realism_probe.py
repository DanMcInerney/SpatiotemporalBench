# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The Blender side of `stbench realism probe` (M1's R.2): what the measures need that a render does
not write. Runs inside Blender's own Python and is never imported by `stbench`.

blender --background --factory-startup --python render/realism_probe.py -- --task index|cards ARGS

- `index`: ARGS are `render_episode.py`'s. The script's frames are rendered by `render_episode` itself, in this process,
after `dressing.DEBUG_INDEX_FROM` is set: every piece of dressing has an object index of its own from 1000 up (today
all of it is 0), so the index pass says which piece each pixel is. Entities stay under 1000 (a world holds 160-250) and
under `render_encode.SELF_ID`. `dressing_index.json` (index -> class) is written beside the frames. These are
measuring frames, never into a dataset: depth and index are what they are for, at one sample.
- `cards`: ARGS are `--root --private --src --manifest --library --out`. Builds the script's scene with the assets
appearance and reads grey cards (`lighting_probe`: albedo 0.18, flat, a 0.1 m square) at 0.75 m on a 3 x 3 grid in each
room and on every kitchen worktop, with the lamps alone and with the daylight alone, on the CPU. `--out` is
`cards.json`: `{"unit_lux", "lamps": [...], "daylight": [...], "worktop": [...]}`, each card with its lux and its cast
(red and blue over green, in linear light).
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DEBUG_BASE = 1000  # the first dressing piece's pass index (`stbench.realism_report.DEBUG_BASE`)
WORKTOP = "kitchen counter"  # the surface kind of a worktop
CARD_Z_M = 0.75  # a card's height in a room


def take_task(argv: list[str]) -> tuple[str, list[str]]:
    """`--task X` taken out of the command line: (X, the rest), for the script that is run after it."""
    i = argv.index("--task")
    return argv[i + 1], argv[:i] + argv[i + 2:]


def index() -> None:
    import dressing

    dressing.DEBUG_INDEX_FROM = DEBUG_BASE
    runpy.run_path(str(HERE / "render_episode.py"), run_name="__main__")
    tmp = Path(sys.argv[sys.argv.index("--tmp") + 1])
    named = {str(k): v for k, v in dressing.INDEX.items()}
    (tmp / "dressing_index.json").write_text(json.dumps(named), encoding="utf-8")


def cards(argv: list[str]) -> None:
    import bpy
    import build_scene
    import cycles_setup
    import lighting_probe

    ap = argparse.ArgumentParser()
    for flag in ("--root", "--private", "--src", "--manifest", "--library", "--out"):
        ap.add_argument(flag, type=Path, required=True)
    args = ap.parse_args(argv[argv.index("--") + 1:])
    sys.path.insert(0, str(args.src))
    from stbench import asset_manifest
    from stbench.mapspec import load_map
    from stbench.timeline import load

    tl = load(args.private / "timeline.json")
    m = load_map(tl.episode.map)
    hfov = json.loads((args.root / "sensors" / "camera.json").read_text(encoding="utf-8"))["hfov_deg"]
    scene = build_scene.build(tl, m, hfov, (asset_manifest.load(args.manifest), args.library))
    scene.apply_time(0.0)
    sc = bpy.context.scene
    rooms = lighting_probe.card_points(m, CARD_Z_M)
    out = {"unit_lux": cycles_setup.UNIT_LUX,
           "lamps": lighting_probe.measure(sc, rooms, sources="lamps"),
           "daylight": lighting_probe.measure(sc, rooms, sources="daylight"),
           "worktop": lighting_probe.measure(sc, lighting_probe.surface_points(m, WORKTOP), sources="lamps")}
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")


def main() -> None:
    task, rest = take_task(sys.argv)
    sys.argv = rest
    if task == "index":
        index()
    elif task == "cards":
        cards(rest)
    else:
        raise SystemExit(f"STBENCH-ERROR unknown task {task!r}: index or cards")


main()
