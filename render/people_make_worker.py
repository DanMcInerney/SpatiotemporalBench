# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""The worker's body, made once: the resident's body and pose in work clothes and work boots, exported as a `.glb`.

    BLENDER_USER_RESOURCES=SCRATCH BLENDER_USER_CONFIG=SCRATCH/config blender --background --factory-startup \
        --python render/people_make_worker.py -- --out <cache>/worker/worker.glb --extension <MPFB zip> \
        --pack <system assets zip>

Run by hand, never by `stbench`: the file is the manifest's `worker` asset, and its source `makehuman_worker` (kind
`generated`) names what made it (`generator`: this recipe and its inputs, each with its licence and url).
The resident's recipe, `people_make.py`, is imported and used as it is, so its output does not move. This file gives its
own parts (the clothes `male_worksuit01`, bib trousers over a work shirt, and the shoes `shoes03`, closed black work
boots, both from the CC0 MakeHuman system assets pack), and leaves the human's body, rig and pose, the bake, the stand
and the export to the resident's. The two recipes take the same two inputs with the same refusal:
`BLENDER_USER_RESOURCES` must be set (the machine's own profile is never touched). It opens no network connection.
Prints `STBENCH-PERSON <json>` with the measured box.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import people_make  # noqa: E402  (unchanged: the worker's body, rig and pose are the resident's)

# What the worker is made of (every asset is CC0, from the system assets pack): the resident's body, hair and eyes with
# work clothes and closed shoes.
PARTS = {"skin": "young_caucasian_male/young_caucasian_male.mhmat", "eyes": "high-poly/high-poly.mhclo",
         "eyebrows": "eyebrow001/eyebrow001.mhclo", "eyelashes": "eyelashes01/eyelashes01.mhclo",
         "hair": "short02/short02.mhclo", "clothes": "male_worksuit01/male_worksuit01.mhclo",
         "shoes": "shoes03/shoes03.mhclo"}
OBJECT_OF = {"Human": "body", "Human.high-poly": "eyes", "Human.eyebrow001": "eyebrows",
             "Human.eyelashes01": "eyelashes", "Human.short02": "hair", "Human.male_worksuit01": "clothes",
             "Human.shoes03": "shoes"}


# -- the person ----------------------------------------------------------------------------------------------------
def build_human():
    """The resident's human (its phenotype and rig) with the worker's parts, and one subdivision level; returns
    (body, rig)."""
    humans = importlib.import_module(f"{people_make.EXTENSION}.services.humanservice").HumanService
    info = humans._create_default_human_info_dict()
    info["phenotype"] = people_make.PHENOTYPE
    info.update(rig="default", eyes=PARTS["eyes"], eyebrows=PARTS["eyebrows"], eyelashes=PARTS["eyelashes"],
                hair=PARTS["hair"], clothes=[PARTS["clothes"], PARTS["shoes"]], skin_mhmat=PARTS["skin"],
                skin_material_type="MAKESKIN", clothes_material_type="MAKESKIN", eyes_material_type="MAKESKIN")
    settings = humans.get_default_deserialization_settings()
    settings["subdiv_levels"] = people_make.SUBDIV
    body = humans.deserialize_from_dict(info, settings)
    rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    return body, rig


def main() -> None:
    args = people_make.parse(sys.argv)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    people_make.install(args)
    bpy.ops.preferences.addon_enable(module=people_make.EXTENSION)
    body, rig = build_human()
    people_make.pose(rig)
    people_make.OBJECT_OF = OBJECT_OF  # `bake` names each baked object's part from this table, read when it runs
    baked = people_make.bake(rig)
    box = people_make.stand(baked)
    people_make.export(baked, args.out)
    report = {"bbox_m": box, "blender": ".".join(map(str, bpy.app.version)), "parts": sorted(o.name for o in baked)}
    print(f"STBENCH-PERSON {json.dumps(report, sort_keys=True)}", flush=True)


if __name__ == "__main__":  # Blender runs `--python` scripts as __main__
    try:
        main()
    except people_make.RecipeError as e:
        print(f"STBENCH-ERROR {e}", flush=True)
        raise SystemExit(4) from e
