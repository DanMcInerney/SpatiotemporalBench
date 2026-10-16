"""Contact sheets of the photoreal assets, for a person to look at before any episode is rendered (`assets preview`).

One sheet per kind (every catalog's, and the light furniture of every map): a column per catalog colour, a row per
view (four, a quarter turn apart), so a recoloured scan, a squashed box or a mirrored pair is easy to see. A
`dressing.png` shows every dressing model from the front, two three-quarter views and the back; `surfaces.png` every
textured surface on a plane and on a sphere; `textures.png` the maps of every `texture` set (Pillow alone: a decal is
shown over grey with its opacity). Blender renders the tiles (`render/preview_assets.py`, from a job list this module
writes: the entities exactly as an episode would build them); Pillow labels and assembles them here. Writes
`<kind>.png`, `dressing.png`, `surfaces.png`, `textures.png` and `index.json` into the output folder.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw

from . import blender_run
from .asset_manifest import PROCEDURAL, AssetError, Manifest, load, texture_dir
from .blender_run import SRC_DIR, BlenderError
from .catalog import CATALOGS
from .mapspec import bundled_maps, load_map

VIEWS = 4
TILE = (192, 144)
LABEL_H = 14
DRESSING_VIEWS = ("front", "three-quarter left", "three-quarter right", "back")
MAP_TILE = 128  # a texture map's tile, in pixels
GREY = (128, 128, 128)  # what a decal's opacity shows through


def preview_jobs(m: Manifest, kinds: list[str] | None = None) -> dict:
    """What to render: per kind (carried, light furniture and mover) its asset, catalog colours, size and, for a kind of
    an outdoor map's catalog, that map's sun and sky; and every material's id and tile size. The kinds are those of
    every catalog, with the colours of each catalog it appears in (the box, which two share, once), and the light
    furniture of every bundled map in the colours the map has it."""
    wanted: dict[str, tuple[list[str], tuple[float, float, float]]] = {}

    def add(name: str, colours, size) -> None:
        have, _ = wanted.setdefault(name, ([], size))
        have += [c for c in colours if c not in have]

    outdoor: dict[str, dict] = {}  # the sun and sky of each catalog's outdoor map (a kind is lit like its episode)
    for name in bundled_maps():
        if (spec := load_map(name)).outdoor is not None:
            outdoor[spec.catalog] = asdict(spec.outdoor)
    lit: dict[str, dict] = {}
    for cname, catalog in CATALOGS.items():
        for k in (*catalog.kinds.values(), *catalog.mover_kinds.values()):
            add(k.name, k.colors, k.size)
            if cname in outdoor:
                lit[k.name] = outdoor[cname]
    for name in bundled_maps():
        for f in load_map(name).furniture:
            if f.mobility == "light":
                add(f.kind, [f.color], f.size)
    clutter = _clutter_items(m, kinds)
    if kinds and (unknown := sorted(set(kinds) - set(wanted) - {c["kind"] for c in clutter})):
        raise AssetError(f"no such kind(s): {', '.join(unknown)}; known: {', '.join(wanted)}")
    items = [_item(m, name, colours, size, lit.get(name)) for name, (colours, size) in wanted.items()
             if not kinds or name in kinds]
    items = [i for group in items for i in group] + clutter
    return {"items": items, "surfaces": [{"id": a.id, "tile_m": a.tile_m} for a in m.assets if a.role == "material"],
            "dressing": [{"id": a.id, "box": a.native_bbox_m, "uses": list(a.uses)} for a in m.assets
                         if a.role == "dressing"]}


def _item(m: Manifest, kind: str, colours: list[str], size, outdoor) -> list[dict]:
    """The sheets of a kind: one, or where it has several models (one per colour, `Asset.for_colours`) a sheet per
    model, named by it, with the colours it is drawn in."""
    by_model: dict[str, list[str]] = {}
    for colour in colours:
        by_model.setdefault(m.model_for(kind, colour).id, []).append(colour)
    return [{"kind": kind, "asset": asset_id, "colours": drawn, "size": list(size), "note": _note(m.asset(asset_id)),
             "outdoor": outdoor, **({"name": asset_id} if len(by_model) > 1 else {})}
            for asset_id, drawn in by_model.items()]


def _clutter_items(m: Manifest, kinds: list[str] | None) -> list[dict]:
    """A sheet per clutter model (a kind has several), named by the model: an untinted one in its own colour, a tinted
    one in every colour its kind lists (white, grey and black where no catalog has the kind), at its measured box and
    logo review."""
    colours = {k.name: k.colors for c in CATALOGS.values() for k in c.clutter}
    out = []
    for a in m.assets:
        if a.role != "clutter" or (kinds and a.kind not in kinds):
            continue
        shown = [a.colour] if a.colour else list(colours.get(a.kind, ("white", "grey", "black")))
        note = ", ".join(n for n in (_note(a), f"logo {a.logo_review}" if a.logo_review != "none" else "") if n)
        out.append({"kind": a.kind, "name": a.id, "asset": a.id, "colours": shown,
                    "size": list(a.native_bbox_m or (0.1, 0.1, 0.1)), "note": note, "outdoor": None})
    return out


def _note(asset) -> str:
    """What is still open in the manifest for this asset, for the sheet's title."""
    tint, open_ = asset.tint or {}, []
    if tint.get("mode") in ("luminance", "swap") and tint.get("materials") is None:
        open_.append("tint.materials not set")
    if asset.native_bbox_m is None and asset.source != PROCEDURAL:
        open_.append("native_bbox_m not recorded")
    return ", ".join(open_)


def _sheet(title: str, columns: list[str], tiles: list[list[Path]], tile: tuple[int, int],
           rows: list[str] | None = None) -> Image.Image:
    """`tiles[row][column]` under a title row and a row of column labels; `rows` labels each row's first tile."""
    w, h = tile
    sheet = Image.new("RGB", (w * len(columns), 2 * LABEL_H + h * len(tiles)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    draw.text((2, 1), title, fill=(255, 255, 255))
    for c, label in enumerate(columns):
        draw.text((c * w + 2, LABEL_H + 1), label, fill=(200, 200, 120))
    for r, row in enumerate(tiles):
        for c, path in enumerate(row):
            with Image.open(path) as im:
                sheet.paste(im.convert("RGB"), (c * w, 2 * LABEL_H + r * h))
        if rows:  # on a dark strip: the tiles' skies are light
            top = 2 * LABEL_H + r * h
            draw.rectangle((0, top, w * len(columns), top + LABEL_H - 2), fill=(24, 24, 24))
            draw.text((2, top + 1), rows[r], fill=(255, 255, 255))
    return sheet


def _surfaces_sheet(ids: list[str], tiles: Path, tile: tuple[int, int]) -> Image.Image:
    """Every surface on a plane and, where its tile exists, on a sphere, three surfaces to a row."""
    w, h = tile
    per_row = 3
    cells = [(i, tiles / "_surfaces" / f"{i}.png", tiles / "_surfaces" / f"{i}__sphere.png") for i in ids]
    rows = [cells[r:r + per_row] for r in range(0, len(cells), per_row)]
    sheet = Image.new("RGB", (2 * per_row * w, len(rows) * (h + LABEL_H)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for r, row in enumerate(rows):
        for c, (name, plane, ball) in enumerate(row):
            for k, path in enumerate(p for p in (plane, ball) if p.is_file()):
                with Image.open(path) as im:
                    sheet.paste(im.convert("RGB"), ((2 * c + k) * w, r * (h + LABEL_H) + LABEL_H))
            draw.text((2 * c * w + 2, r * (h + LABEL_H) + 1), name, fill=(200, 200, 120))
    return sheet


def compose(jobs: dict, tiles: Path, out: Path, tile: tuple[int, int]) -> list[Path]:
    fit_file = tiles / "fit.json"
    fit = json.loads(fit_file.read_text(encoding="utf-8")) if fit_file.is_file() else {}
    dressing = jobs.get("dressing", [])
    written = []
    index = {"kinds": {}, "surfaces": [s["id"] for s in jobs["surfaces"]], "dressing": [d["id"] for d in dressing]}
    for item in jobs["items"]:
        name = item.get("name", item["kind"])  # a clutter kind has several models: each is a sheet of its own
        rows = [[tiles / name / f"{c}__{v}.png" for c in item["colours"]] for v in range(VIEWS)]
        path = out / f"{name}.png"
        title = f"{item['kind']} ({item['asset']})"
        title += f"  box squash {100 * fit[name]:.0f}%" if name in fit else ""
        title += f"  [{item['note']}]" if item.get("note") else ""
        _sheet(title, item["colours"], rows, tile).save(path)
        written.append(path)
        index["kinds"][name] = {"asset": item["asset"], "colours": item["colours"], "views": VIEWS,
                                "squash": fit.get(name), "note": item.get("note", "")}
    if dressing:
        rows = [[tiles / "_dressing" / f"{d['id']}__{v}.png" for v in range(len(DRESSING_VIEWS))] for d in dressing]
        labels = [f"{d['id']}  {' x '.join(f'{100 * x:.1f}' for x in d['box'])} cm" if d.get("box") else d["id"]
                  for d in dressing]
        path = out / "dressing.png"
        _sheet("dressing models (width x depth x height; front toward -y)", list(DRESSING_VIEWS), rows, tile,
               labels).save(path)
        written.append(path)
    if jobs["surfaces"]:
        path = out / "surfaces.png"
        _surfaces_sheet([s["id"] for s in jobs["surfaces"]], tiles, tile).save(path)
        written.append(path)
    (out / "index.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    return [*written, out / "index.json"]


# -- texture sets ------------------------------------------------------------------------------------------------
def _map(folder: Path, name: str) -> Path | None:
    """The `<anything>_<name>.jpg|png` of an unpacked set, the first in sorted order (as `materials.find_map`)."""
    hits = sorted(p for p in folder.glob("*") if p.suffix.lower() in (".jpg", ".png") and p.stem.endswith("_" + name))
    return hits[0] if hits else None


def texture_sheet(m: Manifest, cache: Path | None = None) -> Image.Image | None:
    """The maps of every `texture` set in a row (Color, Opacity, Roughness) and the colour over grey through its opacity
    (what a decal or a grime layer adds to a surface); None when the manifest holds none."""
    sets = [a for a in m.assets if a.role == "texture"]
    if not sets:
        return None
    names = ("Color", "Opacity", "Roughness")
    side = MAP_TILE
    sheet = Image.new("RGB", (side * 4, (side + LABEL_H) * len(sets)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for r, a in enumerate(sets):
        y = r * (side + LABEL_H)
        folder = texture_dir(m, a.id, cache)
        draw.text((2, y + 1), a.id + "  Color | Opacity | Roughness | Color over grey by Opacity", fill=(200, 200, 120))
        found = {n: _map(folder, n) for n in names}
        images = {n: Image.open(p).convert("RGB").resize((side, side)) for n, p in found.items() if p}
        for c, n in enumerate(names):
            if n in images:
                sheet.paste(images[n], (c * side, y + LABEL_H))
        if "Color" in images:
            alpha = images["Opacity"].convert("L") if "Opacity" in images else Image.new("L", (side, side), 255)
            sheet.paste(Image.composite(images["Color"], Image.new("RGB", (side, side), GREY), alpha),
                        (3 * side, y + LABEL_H))
    return sheet


def preview(blend: Path, manifest: Path, out: Path, kinds: list[str] | None = None, samples: int = 32,
            tile: tuple[int, int] = TILE) -> list[Path]:
    """Render and assemble the sheets for an already checked library; the files written."""
    m = load(manifest)
    jobs = preview_jobs(m, kinds)
    out = Path(out)
    tiles = out / "tiles"
    shutil.rmtree(tiles, ignore_errors=True)
    tiles.mkdir(parents=True)
    (out / "jobs.json").write_text(json.dumps(jobs, indent=1) + "\n", encoding="utf-8")
    cmd = blender_run.command("preview_assets.py", "--manifest", str(manifest), "--library", str(blend), "--jobs",
                              str(out / "jobs.json"), "--out", str(tiles), "--src", str(SRC_DIR), "--samples",
                              str(samples), "--tile", f"{tile[0]}x{tile[1]}")
    try:
        blender_run.run(cmd, "rendering the asset preview")
    except BlenderError as e:
        raise AssetError(str(e)) from e
    written = compose(jobs, tiles, out, tile)
    if (textures := texture_sheet(m)) is not None:
        textures.save(out / "textures.png")
        written.insert(-1, out / "textures.png")
    shutil.rmtree(tiles, ignore_errors=True)
    (out / "jobs.json").unlink()
    return written
