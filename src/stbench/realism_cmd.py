"""`stbench realism`: the tools that measure how a look reads (M1's R.2).

stbench realism probe SCRIPT_DIR --out DIR [--poses 30] [--rgb 4]      the measuring renders of one script
stbench realism report DATASET --probe DIR [--linear DIR] [--refs DIR] [--crop-answers FILE] [--crop-key FILE]
                       [--recog-lum X] --out DIR                       realism.json and realism.md
stbench realism sheet DATASET --probe DIR [--refs DIR] --out DIR [--seed N]   index.html, the crops, the key, the
prompts

`probe` is `realism_run`'s; `report` and `sheet` read a dataset's scripts and the probe folders and refuse, rather than
measure, a script whose renders are not whole (`render_cli.render_binding_problems`). The probe of a
script is `realism_data.probe_folder(DIR, <script id>)` for all three commands: `probe --out DIR` writes `DIR/<id>`
(or `DIR` itself where it is already named by the id) and `--probe DIR` reads the same place; a script with no probe
there is named in the report, not passed over. Reading only: report and sheet render nothing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import job_registry
from . import realism_crops as K
from . import realism_data as D
from . import realism_report as R
from .camera import Camera
from .catalog import CATALOGS
from .contact_sheet import (
    DOCK_PHASES,
    STRATA,
    centre_median,
    draw,
    pick_crops,
    stratum,
    write_crops,
    write_sheet,
)
from .dressing_boxes import open_floor
from .geometry import Rect, point_in_rect
from .mapspec import MapSpec, load_map
from .realism_rows import build_rows
from .render_cli import render_binding_problems
from .render_encode import read_png16
from .scenery import tracked_colours

NOISE_FRAMES = 12  # frames of a script whose flat patches are read


def free_floor_m2(m: MapSpec) -> float:
    """The floor of a map's rooms less its walls and blocking footprints, in m2 (the apartment's is about 39)."""
    return sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in open_floor(m, 0.0))


def askable(m: MapSpec) -> dict[str, list[str]]:
    """Every kind a question can name in the map's catalog, with its tracked colours: the catalog's kinds, the light
    furniture the map holds (the colours it has there) and the movers' kinds."""
    cat = CATALOGS[m.catalog]
    out = {k: sorted(tracked_colours(m.catalog, k)) for k in cat.kinds}
    for f in m.furniture:
        if f.mobility == "light":
            out.setdefault(f.kind, [])
            out[f.kind] = sorted({*out[f.kind], f.color})
    for k, mover in cat.mover_kinds.items():
        out.setdefault(k, list(mover.colors) if hasattr(mover, "colors") else [])
    return out


def _on_floor(tl, eid: str) -> bool:
    p = tl.placement_of(eid, 0.0)
    return p is not None and p.on is None and p.pos[2] < 1e-9


def _spaced(n: int, take: int) -> list[int]:
    """`take` or so of the keyframes 0..n-1, evenly spaced."""
    return list(range(n))[::max(1, n // take)]


def _script(paths, probe: Path | None, linear: Path | None) -> dict:
    """Everything the report reads of one script: its numbers, or the reasons it is skipped."""
    if problems := render_binding_problems(paths):
        return {"skipped": problems}
    tl = D.timeline_of(paths)
    m = load_map(tl.episode.map)
    cam_ks = list(range(tl.n_keyframes))
    log = D.read_log(paths)
    out = {"id": paths.id, "map": m.name, "free_m2": free_floor_m2(m),
           "counts": R.entity_counts((e.mobility, _on_floor(tl, e.id)) for e in tl.entities),
           "frames": D.exposure_frames(paths, tl, m), "sightings": D.sightings(paths, tl),
           "lamps": log.get("lighting", {}).get("lamps", []),
           "daylight": log.get("lighting", {}).get("sun", {}).get("elevation_deg", 0.0) > 0.0,
           "patches": D.noise_patches(paths, _spaced(len(cam_ks), NOISE_FRAMES)),
           "relevance": sorted({kind for t in tl.ego.tasks
                                for kind in job_registry.get(tl.episode.job).relevance(t.plan).kinds})}
    if linear is not None and (linear / paths.id).is_dir():
        out["stops"] = D.linear_stops(linear / paths.id, _spaced(len(cam_ks), NOISE_FRAMES))
    folder = None if probe is None else D.probe_folder(probe, paths.id)
    if folder is not None and (folder / "meta.json").exists():
        cam = Camera.load(paths.camera)
        standing = D.standing_paths(folder)
        if problems := render_binding_problems(standing):
            out["probe_skipped"] = problems
        else:
            out["standing_objects"] = D.probe_objects(folder, "index_standing", Camera.load(standing.camera), R.TRIM)
        out["script_objects"] = D.probe_objects(folder, "index_script", cam, R.TRIM)
        out["cards"] = D.probe_cards(folder)
        if linear is not None and (linear / paths.id).is_dir():
            out["windows"] = D.window_range(linear / paths.id, folder, _spaced(len(cam_ks), D.WINDOW_FRAMES))
    elif folder is not None:
        out["no_probe"] = str(folder)
    return out


def report(dataset: str, probe: str | None, out: str, linear: str | None = None, refs: str | None = None,
           crop_answers: str | None = None, crop_key: str | None = None, recog_lum: float | None = None) -> dict:
    """Write `realism.json` and `realism.md` into `out`; returns the document."""
    from .recognise import RECOG_LUM

    scripts, skipped = [], {}
    for paths in D.script_paths(dataset):
        s = _script(paths, None if probe is None else Path(probe), None if linear is None else Path(linear))
        if "skipped" in s:
            skipped[paths.id] = s["skipped"]
        else:
            scripts.append(s)
    if not scripts:
        raise SystemExit(f"realism: every script of {dataset} has its render missing or partial: {skipped}")
    crops = None
    if crop_answers:
        key_path = Path(crop_key) if crop_key else Path(crop_answers).with_name("crop_key.json")
        answers = json.loads(Path(crop_answers).read_text(encoding="utf-8"))
        m = load_map(scripts[0]["map"])
        crops = K.crop_scores(answers, json.loads(key_path.read_text(encoding="utf-8")), askable(m))
    rows = build_rows(scripts, None if refs is None else D.reference_stats(refs), RECOG_LUM if recog_lum is None else
                      recog_lum, crops)
    no_probe = {s["id"]: s["no_probe"] for s in scripts if "no_probe" in s}
    doc = {"rows": [r.to_dict() for r in rows], "scripts": [s["id"] for s in scripts], "skipped": skipped,
           "no_probe": no_probe, "recog_lum": RECOG_LUM if recog_lum is None else recog_lum}
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "realism.json").write_text(json.dumps(doc, indent=1, default=float) + "\n", encoding="utf-8")
    note = f"\nSkipped (render missing or partial): {skipped}\n" if skipped else ""
    note += f"\nNo probe folder (`stbench realism probe` writes DIR/<script id>): {no_probe}\n" if no_probe else ""
    (dest / "realism.md").write_text(R.render_md(rows) + note, encoding="utf-8")
    return doc


# -- the sheet -----------------------------------------------------------------------------------------------------
def _frames_of(paths, tl, m: MapSpec) -> list[dict]:
    """Every keyframe of a script with its stratum (`contact_sheet.stratum`; depth is read only where it decides) and
    its pose, for the draw to pass over a view it has."""
    under = [f.footprint() for f in m.furniture if f.mobility == "fixed" and f.blocks and f.under_m > 0]
    solids: list[Rect] = [*m.wall_pieces(), *(f.footprint() for f in m.furniture if f.mobility == "fixed" and f.blocks)]
    cam = Camera.load(paths.camera)
    unit = 1e-3 if cam.depth_unit == "mm" else 1.0
    out = []
    for k in range(tl.n_keyframes):
        kf = tl.keyframe(k)
        decided = kf.phase in DOCK_PHASES or any(point_in_rect(r, kf.base[0], kf.base[1]) for r in under)
        centre = None if decided else centre_median(read_png16(paths.depth(k)).astype(float) * unit)
        out.append({"script": paths.id, "k": k, "stratum": stratum(kf.phase, kf.base[:2], under, solids, centre),
                    "pose": (m.name, kf.base[0], kf.base[1], kf.yaw)})
    return out


def crop_candidates(paths, tl, probe: Path | None, ask: dict) -> dict[str, list[dict]]:
    """Sightings of 1,024 to 2,048 px by class: clutter and askable entities from the visibility log, dressing pieces
    from the probe's index of the script's own frames. A class is `group:name` (clutter, askable or dressing)."""
    out: dict[str, list[dict]] = {}
    for line in paths.visibility.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for e in row["entities"]:
            ent = tl.entity(e["id"])
            group = "clutter" if ent.mobility == "clutter" else "askable" if ent.kind in ask else None
            if group:
                out.setdefault(f"{group}:{ent.kind}", []).append(
                    {"script": paths.id, "k": row["k"], "px": e["px"], "bbox": e["bbox"], "id": e["id"],
                     "kind": ent.kind if group == "askable" else None, "colour": ent.attributes.get("color")})
    if probe is not None and (D.probe_folder(probe, paths.id) / "meta.json").exists():
        folder = D.probe_folder(probe, paths.id)
        classes = D.dressing_index(folder)
        for p in sorted((folder / "index_script").glob("*.npz")):
            index = D.read_frame(p)[0]
            for pid, n in D.pixel_counts(index).items():
                if pid in classes and 1024 <= n <= 2048:
                    vs, us = np.nonzero(index == pid)
                    out.setdefault(f"dressing:{classes[pid]}", []).append(
                        {"script": paths.id, "k": int(p.stem), "px": n, "id": str(pid),
                         "bbox": [int(us.min()), int(vs.min()), int(us.max()) + 1, int(vs.max()) + 1],
                         "kind": None, "colour": None})
    return out


def sheet(dataset: str, probe: str | None, refs: str | None, out: str, seed: int = 0) -> dict:
    """`index.html` with the strata's frames, the standing frames and the references, and the crops with their key and
    the crop check's two fixed prompts (`crop_prompts.json`); returns what was written."""
    frames, candidates, standing = [], {}, []
    paths_of = {p.id: p for p in D.script_paths(dataset)}
    ask = None
    for sid, paths in paths_of.items():
        if render_binding_problems(paths):
            print(f"realism: skipping {sid}: its render is missing or partial", file=sys.stderr)
            continue
        tl = D.timeline_of(paths)
        m = load_map(tl.episode.map)
        ask = ask or askable(m)
        frames += _frames_of(paths, tl, m)
        for cls, found in crop_candidates(paths, tl, None if probe is None else Path(probe), ask).items():
            candidates.setdefault(cls, []).extend(found)
        if probe is not None and (D.probe_folder(probe, sid) / "meta.json").exists():
            folder = D.probe_folder(probe, sid)
            meta = D.probe_meta(folder)
            sp = D.standing_rgb_paths(folder)
            standing += [] if sp is None else [(sp.rgb(k), f"standing camera, {meta['rooms'][k]}, script {sid}, "
                                                            f"frame {k}") for k in range(meta["rgb"])]
        elif probe is not None:
            print(f"realism: no probe of {sid} under {D.probe_folder(probe, sid)}", file=sys.stderr)
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    pages = write_sheet(dest, draw(frames, seed), lambda s, k: paths_of[s].rgb(k), standing,
                        sorted(p for p in Path(refs).iterdir() if p.suffix.lower() in D.IMAGES) if refs else [])
    key = write_crops(pick_crops(candidates, seed), lambda s, k: paths_of[s].rgb(k), dest, seed)
    (dest / "crop_prompts.json").write_text(json.dumps(K.prompts(sorted(ask or {})), indent=1) + "\n", encoding="utf-8")
    return {"page": str(pages), "crops": len(key), "classes": sorted({r["class"] for r in key}),
            "strata": {name: sum(1 for f in frames if f["stratum"] == name) for name in STRATA}}


# -- the command ---------------------------------------------------------------------------------------------------
def _probe(args: argparse.Namespace) -> int:
    from .realism_run import probe

    meta = probe(args.script_dir, args.out, args.poses, args.rgb)
    print(f"probed {meta['script']}: {meta['frames']['standing']} standing frames ({meta['rgb']} at release settings), "
          f"{meta['frames']['script']} script frames indexed, {meta['wall_s']} s")
    return 0


def _report(args: argparse.Namespace) -> int:
    doc = report(args.dataset, args.probe, args.out, args.linear, args.refs, args.crop_answers, args.crop_key,
                 args.recog_lum)
    print(f"wrote {Path(args.out) / 'realism.md'} ({len(doc['rows'])} measures, {len(doc['scripts'])} scripts)")
    return 0


def _sheet(args: argparse.Namespace) -> int:
    made = sheet(args.dataset, args.probe, args.refs, args.out, args.seed)
    print(f"wrote {made['page']} and {made['crops']} crops; strata {made['strata']}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("realism", help="measure how a look reads: probe, report, sheet")
    s = p.add_subparsers(dest="realism_command", required=True)
    a = s.add_parser("probe", help="the measuring renders of one script (the Blender lock first)")
    a.add_argument("script_dir")
    a.add_argument("--out", required=True, help="the folder of probes: DIR/<script id> is written (DIR itself if it is "
                                                 "named by the id); `report --probe` and `sheet --probe` read the same")
    a.add_argument("--poses", type=int, default=30, help="standing poses a room")
    a.add_argument("--rgb", type=int, default=4, help="standing frames also rendered at release settings")
    a.set_defaults(func=_probe)
    b = s.add_parser("report", help="realism.json and realism.md of a dataset and its probes")
    b.add_argument("dataset")
    b.add_argument("--probe", help="the folder of probes (see `probe --out`)")
    b.add_argument("--linear", help="a folder of kept linear passes, one folder a script id")
    b.add_argument("--refs", help="a folder of reference photographs (its sources.json says which are robot frames)")
    b.add_argument("--crop-answers", help="the reviewers' answers: {crop: {'forced': '<colour> <kind>', 'free': "
                                          "'<colour> <object>'}} (a plain string is a free answer)")
    b.add_argument("--crop-key", help="the key the sheet wrote (default: crop_key.json beside the answers)")
    b.add_argument("--recog-lum", type=float, default=None)
    b.add_argument("--out", required=True)
    b.set_defaults(func=_report)
    c = s.add_parser("sheet", help="the contact sheet, the strata's frames and the crops")
    c.add_argument("dataset")
    c.add_argument("--probe", help="the folder of probes (see `probe --out`)")
    c.add_argument("--refs")
    c.add_argument("--out", required=True)
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(func=_sheet)
