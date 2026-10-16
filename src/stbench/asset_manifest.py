"""The asset manifest: every third-party file and procedural builder the photoreal renderer may use, with its source and
licence (VISION rule 11).

`src/stbench/assets/manifest.json` ships with the package. This module loads and checks it and owns the cache layout.
It is pure: no network (that is `asset_fetch`) and no Blender. The licence allowlist lives here, in code, not in the
data.

An asset has one of six roles:
- `model`: an entity's mesh, one per queried kind (carried, light furniture and mover), or, where a kind has several,
  one per colour: each lists its `for_colours`, so a description (colour, kind) always has the same model (a toy is a
  duck in yellow and a bear in pink; `Manifest.model_for(kind, colour)`);
- `clutter`: a mesh of world content that no question names (`catalog.ClutterKind`): several a kind, each with its own
  measured box (the entity's size) and, untinted, its own `colour` (a tinted one takes the entity's). `logo_review` says
  what a person saw of a brand on the asset's preview when it was added: `none`, `minor` or `dominant` (CC BY licenses
  no trademark, so a dominant one is never used). Every CC BY asset carries one;
- `material`: an ambientCG PBR set the library builds into a Blender material, named by the asset id;
- `dressing`: a model placed by an `asset:<id>` recipe of a map's dressing (`render/dressing_spec.py`): it is never an
  entity, so it has no kind or mobility, and its native size (width, depth, height, front toward -y: a glTF as
  Poly Haven publishes it, not turned to face +x) is what the layout's box is fitted to;
- `hdri`: an environment map, one file, named for a catalog by the manifest's `skies` (`hdri_path`);
- `texture`: a set of loose maps a material or a decal reads from the cache (`texture_dir`; grime and wear masks,
  road-marking decals): the library builds nothing from it, and it is fetched and credited like the rest.

A source is `hosted` or `generated` (a recipe makes its file, nothing is downloaded; `asset_generated`).

`transform` brings an imported model to Z up, front toward +x: `unit_scale` (to metres), `rotate_deg` (applied after the
import) and `up`. `up` is the axis the Blender importer is told the file's up is, and `validate` holds it to the format:
an OBJ carries no up axis, so each OBJ entry says `"Y"` or `"Z"` (measured per asset); a glTF is Y up by its spec and
Blender's glTF importer converts that to Z up, so `up` is not an input there and a glTF entry says `"Z"`, the result
after import, never the spec's `"Y"`. After import every model is Z up whatever the file said.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import MISSING, fields
from functools import cache
from importlib import resources
from pathlib import Path
from urllib.parse import urlsplit

from .asset_coverage import (  # noqa: F401  (the queried_* names are re-exported)
    BOX_SLACK_M,
    clutter_coverage_problems,
    coverage_problems,
    queried_clutter,
    queried_colours,
    queried_kinds,
)
from .asset_generated import GENERATED, parts_problems, source_problems  # noqa: F401  (GENERATED is re-exported)
from .asset_types import Asset, AssetError, AssetFile, Manifest, Source
from .catalog import CATALOGS

LICENCES = frozenset({"CC0-1.0", "CC-BY-4.0", "BSD-3-Clause"})
CODE_LICENCES = frozenset({"GPL-3.0-or-later"})  # only for the source that is our own code
PROCEDURAL = "procedural"
MAX_ANISOTROPY = 0.25  # no asset may be squashed more than this to fit its catalog box
ROLES = ("model", "material", "dressing", "hdri", "texture", "clutter")
IMPORTED = ("model", "dressing", "clutter")  # the roles Blender builds a mesh for (an import, or a procedural builder)
LOGO_REVIEWS = ("none", "minor", "dominant")
CLUTTER_MOBILITY = "clutter"  # the `mobility` of a clutter model: the entity's, `timeline.MOBILITY`
MOBILITIES, FORMATS, HDRI_FORMATS = ("carried", "light", "mover"), ("gltf", "obj"), ("hdr", "exr")
UP_AXES = {"gltf": ("Z",), "obj": ("Y", "Z")}  # `transform.up` allowed per format (see the module docstring)
FIT_MODES, TINT_MODES = ("per_axis", "uniform", "exact"), ("luminance", "swap", "none")
EXTRACT_DIR = "x"  # a zip's members land in <cache>/<asset id>/x/
TOP_LEVEL = {"sources", "assets", "palette_materials", "floors", "walls", "skies"}
_ID = re.compile(r"[A-Za-z0-9_]+")
REPO = Path(__file__).resolve().parents[2]


# -- loading -----------------------------------------------------------------------------------------------------
def _build(cls, raw, where: str, **converted):
    """`cls` from a JSON object: unknown or missing fields are errors, so a typo cannot silently drop a field."""
    if not isinstance(raw, dict):
        raise AssetError(f"{where}: expected an object")
    names = {f.name for f in fields(cls)}
    if unknown := sorted(set(raw) - names):
        raise AssetError(f"{where}: unknown field(s) {unknown}; known: {sorted(names)}")
    required = [f.name for f in fields(cls) if f.default is MISSING and f.name not in converted]
    if missing := [n for n in required if n not in raw]:
        raise AssetError(f"{where}: missing field(s) {missing}")
    return cls(**{**raw, **converted})


def _safe_path(path) -> bool:
    return (isinstance(path, str) and path != "" and "\\" not in path and ":" not in path
            and all(part not in ("", ".", "..") for part in path.split("/")))


def _asset(raw, i: int) -> Asset:
    if not isinstance(raw, dict):
        raise AssetError(f"asset {i}: expected an object")
    if "id" in raw and not _safe_path(raw["id"]):  # the id names the asset's cache folder
        raise AssetError(f"asset {i}: id {raw['id']!r} must be relative, no '..', ':' or backslash")
    where, files = f"asset {raw.get('id', i)}", []
    for j, f in enumerate(raw.get("files", ())):
        files.append(_build(AssetFile, f, f"{where} file {j}"))
        if not _safe_path(files[-1].path):
            raise AssetError(f"{where} file {j}: path {files[-1].path!r} must be relative, no '..', ':' or backslash")
    return _build(Asset, raw, where, files=tuple(files), uses=tuple(raw.get("uses", ())),
                  drop_slots=tuple(raw.get("drop_slots", ())), rest_pose=tuple(raw.get("rest_pose", ())),
                  for_colours=tuple(raw.get("for_colours", ())), parts=tuple(raw.get("parts", ())))


def parse(raw: dict, sha256: str = "") -> Manifest:
    """A manifest from its parsed JSON. Structure errors raise `AssetError`; `validate` reports the rest."""
    if not isinstance(raw, dict):
        raise AssetError("the manifest must be an object")
    if unknown := sorted(set(raw) - TOP_LEVEL):
        raise AssetError(f"manifest: unknown top-level field(s) {unknown}")
    sources = {k: _build(Source, v, f"source {k}", hosts=tuple(v.get("hosts", ())) if isinstance(v, dict) else ())
               for k, v in raw.get("sources", {}).items()}
    assets = tuple(_asset(a, i) for i, a in enumerate(raw.get("assets", ())))
    return Manifest(sources, assets, dict(raw.get("palette_materials", {})), dict(raw.get("floors", {})),
                    raw.get("walls"), dict(raw.get("skies", {})), sha256)


def packaged_path() -> Path:
    return Path(str(resources.files("stbench.assets").joinpath("manifest.json")))


def load(path: str | Path | None = None) -> Manifest:
    """The manifest at `path`, or the packaged one."""
    target = packaged_path() if path is None else Path(path)
    try:
        data = target.read_bytes()
        return parse(json.loads(data.decode("utf-8")), hashlib.sha256(data).hexdigest())
    except (OSError, ValueError) as e:
        raise AssetError(f"cannot read the asset manifest {target}: {e}") from e


@cache
def packaged() -> Manifest:
    return load()


def model_for(kind: str, colour: str | None = None) -> Asset:
    return packaged().model_for(kind, colour)


def material(asset_id: str) -> Asset:
    return packaged().material(asset_id)


# -- validation --------------------------------------------------------------------------------------------------
def _num(v) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _allowed(source: str) -> frozenset[str]:
    return LICENCES | CODE_LICENCES if source == PROCEDURAL else LICENCES


def _source_problems(key: str, s: Source) -> list[str]:
    out = []
    if s.licence not in _allowed(key):
        out.append(f"source {key}: licence {s.licence!r} is not on the allowlist {sorted(_allowed(key))}")
    out += source_problems(key, s, LICENCES, CODE_LICENCES)
    if key != PROCEDURAL:
        if not s.hosts and s.kind != GENERATED:
            out.append(f"source {key}: no hosts")
        if not s.licence_url:
            out.append(f"source {key}: no licence_url")
        if s.licence.startswith("CC-BY") and not s.attribution:
            out.append(f"source {key}: a {s.licence} source needs an attribution line")
    return out


def _file_problems(a: Asset, f: AssetFile, hosts: tuple[str, ...], generated: bool = False) -> list[str]:
    where, out = f"{a.id} {f.path}", []
    url = urlsplit(f.url)
    if generated:
        if f.url:
            out.append(f"{where}: a generated file has no url (its recipe makes it)")
    elif url.scheme != "https" or url.hostname not in hosts:
        out.append(f"{where}: url {f.url!r} is not https on one of the source's hosts {list(hosts)}")
    return out


def _fit_problems(a: Asset) -> list[str]:
    """`fit` and the measured box: what a mesh's native size is fitted to (an entity's catalog box, a dressing's)."""
    out, fit = [], a.fit or {}
    if fit.get("mode") not in FIT_MODES:
        out.append(f"{a.id}: fit.mode must be one of {FIT_MODES}")
    if fit.get("mode") == "per_axis":
        limit = fit.get("max_anisotropy")
        if not (_num(limit) and 0 <= limit <= MAX_ANISOTROPY):
            out.append(f"{a.id}: fit.max_anisotropy must be a number from 0 to {MAX_ANISOTROPY}")
    if a.source == PROCEDURAL and fit.get("mode") != "exact":
        out.append(f"{a.id}: procedural geometry is built to the box, so fit.mode must be 'exact'")
    box = a.native_bbox_m
    if box is not None and not (len(box) == 3 and all(_num(v) and v > 0 for v in box)):
        out.append(f"{a.id}: native_bbox_m must be null or three positive numbers")
    return out


def _tint_problems(a: Asset) -> list[str]:
    out, tint = [], a.tint or {}
    if tint.get("mode") not in TINT_MODES or not isinstance(tint.get("materials"), list | None):
        out.append(f"{a.id}: tint needs a mode from {TINT_MODES} and materials as a list (null until measured)")
    lighten = tint.get("lighten")  # optional: a tinted slot shown this fraction of the way to white (umbrella gores)
    if lighten is not None and not (isinstance(lighten, dict) and set(lighten) <= set(tint.get("materials") or ())
                                    and all(_num(v) and 0 <= v <= 1 for v in lighten.values())):
        out.append(f"{a.id}: tint.lighten must map slots of tint.materials to a fraction from 0 to 1")
    return out


def _model_problems(a: Asset) -> list[str]:
    out = _fit_problems(a) + _tint_problems(a)
    if a.mobility not in MOBILITIES:
        out.append(f"{a.id}: mobility {a.mobility!r} must be one of {MOBILITIES}")
    if not a.kind:
        out.append(f"{a.id}: a model needs a kind")
    if a.compose is not None and not (a.compose.get("mode") == "pair_mirror" and _num(a.compose.get("gap_m"))):
        out.append(f"{a.id}: compose must be null or {{mode: pair_mirror, gap_m: number}}")
    if not all(isinstance(c, str) and c for c in a.for_colours) or len(set(a.for_colours)) != len(a.for_colours):
        out.append(f"{a.id}: for_colours must be distinct colour names")
    if a.for_colours and a.alternate_for:
        out.append(f"{a.id}: an alternate serves no colour of its own (it is never used unless asked for)")
    return out + parts_problems(a, LICENCES)


def _clutter_problems(a: Asset) -> list[str]:
    """A clutter model is the mesh of one clutter kind, built to its own box: its entity's size is the measured box, so
    it is fitted `exact` (a scan that is not a real size is rescaled by `transform.unit_scale`, so the box is the real
    one). It shows one colour: an untinted model declares its own `colour`, a tinted one takes the entity's and declares
    none. It has a logo review, and a pair, a mirrored half or an alternate is no clutter model."""
    out = _fit_problems(a) + _tint_problems(a)
    if (a.fit or {}).get("mode") != "exact":
        out.append(f"{a.id}: a clutter model is fitted exact (rescale it with transform.unit_scale)")
    if a.mobility != CLUTTER_MOBILITY:
        out.append(f"{a.id}: a clutter model's mobility is {CLUTTER_MOBILITY!r}, not {a.mobility!r}")
    if not a.kind:
        out.append(f"{a.id}: a clutter model needs a kind")
    if ((a.tint or {}).get("mode") in ("luminance", "swap")) == bool(a.colour):
        out.append(f"{a.id}: a tinted clutter model takes the entity's colour and an untinted one declares its own "
                   "`colour`, never both or neither")
    if a.logo_review not in LOGO_REVIEWS:
        out.append(f"{a.id}: logo_review must be one of {LOGO_REVIEWS}, set from the asset's preview")
    entity = {"compose": a.compose, "alternate_for": a.alternate_for}
    return out + [f"{a.id}: a clutter model has no {name}" for name, v in entity.items() if v]


def _dressing_problems(a: Asset) -> list[str]:
    """A dressing model is never an entity (validity guard): no kind, no mobility, no
    colour attribute to tint it by, no pair, no alternate; it is fetched, so a source and a glTF are needed."""
    out = _fit_problems(a)
    if a.source == PROCEDURAL:
        out.append(f"{a.id}: dressing is a fetched model; procedural dressing is built by the layout's builders")
    entity = {"kind": a.kind, "mobility": a.mobility, "tint": a.tint, "compose": a.compose,
              "alternate_for": a.alternate_for}
    out += [f"{a.id}: dressing is never an entity, so {name} must be absent" for name, v in entity.items() if v]
    if a.format != "gltf":
        out.append(f"{a.id}: dressing is a glTF (the Poly Haven models)")
    return out


def _edit_problems(a: Asset) -> list[str]:
    """`drop_slots` and `rest_pose` are the only edits to the mesh a source published: a dressing model's, and for
    `drop_slots` a clutter model's too (the glass of a picture frame, which the library would leave opaque)."""
    allowed = {"drop_slots": ("dressing", "clutter"), "rest_pose": ("dressing",)}
    out = [f"{a.id}: {name} is for a {' or '.join(roles)} model" for name, roles in allowed.items()
           if getattr(a, name) and a.role not in roles]
    return out + [f"{a.id}: {name} must be names (text)" for name in allowed
                  if not all(isinstance(n, str) and n for n in getattr(a, name))]


def _hdri_problems(a: Asset) -> list[str]:
    out = []
    if not (len(a.files) == 1 and a.format in HDRI_FORMATS and a.entry == a.files[0].path):
        out.append(f"{a.id}: an hdri is one file, its entry, in a format from {HDRI_FORMATS}")
    return out + [f"{a.id}: an hdri has no {n}" for n in ("kind", "mobility", "transform", "fit", "tint")
                  if getattr(a, n)]


def _download_problems(a: Asset, hosts: tuple[str, ...], generated: bool = False) -> list[str]:
    out = [] if a.files and a.page else [f"{a.id}: needs files and a page"]
    out += [p for f in a.files for p in _file_problems(a, f, hosts, generated)]
    if len({f.path for f in a.files}) != len(a.files):
        out.append(f"{a.id}: duplicate file paths")
    bad = a.format not in FORMATS or a.entry not in {f.path for f in a.files} or not isinstance(a.transform, dict)
    if a.role in IMPORTED and (bad or not a.transform):
        out.append(f"{a.id}: a {a.role} needs a format in {FORMATS}, an entry that is one of its files and a "
                   "transform")
    elif a.role in IMPORTED and a.transform.get("up") not in UP_AXES[a.format]:
        out.append(f"{a.id}: transform.up {a.transform.get('up')!r} must be one of {UP_AXES[a.format]} for a "
                   f"{a.format} file")
    if a.role in ("material", "texture") and not all(f.extract for f in a.files):
        out.append(f"{a.id}: a {a.role} is an ambientCG zip, unpacked (extract: true)")
    return out


def _asset_problems(a: Asset, src: Source | None, by_id: dict[str, Asset]) -> list[str]:
    if not _ID.fullmatch(a.id):
        return [f"{a.id!r}: ids are letters, digits and underscores"]
    if src is None:
        return [f"{a.id}: unknown source {a.source!r}"]
    out = []
    if a.licence not in _allowed(a.source):
        out.append(f"{a.id}: licence {a.licence!r} is not on the allowlist {sorted(_allowed(a.source))}")
    if a.licence != src.licence:
        out.append(f"{a.id}: licence {a.licence!r} differs from its source's {src.licence!r}")
    if a.role not in ROLES:
        return [*out, f"{a.id}: role {a.role!r} must be one of {ROLES}"]
    if a.for_colours and a.role != "model":
        out.append(f"{a.id}: for_colours is for a model, not a {a.role}")
    if a.parts and a.role != "model":
        out.append(f"{a.id}: parts is for a model, not a {a.role}")
    if a.licence.startswith("CC-BY") and not (a.title and a.changes):
        out.append(f"{a.id}: a {a.licence} asset needs a title and a note of what we changed")
    if a.licence.startswith("CC-BY") and a.logo_review is None:
        out.append(f"{a.id}: a {a.licence} asset needs a logo_review (CC BY licenses no trademark)")
    if a.logo_review == "dominant" and not a.alternate_for:
        out.append(f"{a.id}: its logo dominates the preview, so the asset is never used; drop it")
    if a.source == PROCEDURAL:
        if not a.builder or a.files:
            out.append(f"{a.id}: procedural assets need a builder and download nothing")
    else:
        out += _download_problems(a, src.hosts, src.kind == GENERATED)
    checks = {"model": _model_problems, "dressing": _dressing_problems, "hdri": _hdri_problems,
              "clutter": _clutter_problems}
    out += _edit_problems(a)
    if a.role in checks:
        out += checks[a.role](a)
    elif a.tile_m is not None and not (len(a.tile_m) == 2 and all(_num(v) and v > 0 for v in a.tile_m)):
        out.append(f"{a.id}: tile_m must be null or two positive numbers")
    if a.alternate_for is not None:
        main = by_id.get(a.alternate_for)
        if main is None or main.role != "model" or main.kind != a.kind or main.alternate_for:
            out.append(f"{a.id}: alternate_for must name a primary model of the same kind")
    return out


def _reference_problems(m: Manifest, by_id: dict[str, Asset]) -> list[str]:
    refs = [("palette_materials", k, v) for k, v in m.palette_materials.items()]
    refs += [("floors", k, v) for k, v in m.floors.items() if not (isinstance(v, str) and v.startswith("shader:"))]
    refs += [("walls", "walls", m.walls)] if m.walls else []
    refs += [(a.id, k, v) for a in m.assets for k, v in (a.materials or {}).items() if not v.startswith("shader:")]
    out = [f"{where}: {key} refers to {ref!r}, which is not a material in the manifest"
           for where, key, ref in refs if ref not in by_id or by_id[ref].role != "material"]
    for catalog, ref in m.skies.items():
        if catalog not in CATALOGS:
            out.append(f"skies: {catalog!r} is not a catalog; known: {sorted(CATALOGS)}")
        if ref not in by_id or by_id[ref].role != "hdri":
            out.append(f"skies: {catalog} refers to {ref!r}, which is not an hdri in the manifest")
    return out


def validate(m: Manifest, catalog_kinds=None, light_kinds=(), mover_kinds=(), clutter=None, colours=None) -> list[str]:
    """Problems found (an empty list is a pass): licences, hosts, references and, when the kinds are given,
    exactly one model per carried kind, per light-furniture kind and per mover kind (`queried_kinds` for the packaged
    manifest; where `colours` says which colours a kind has, `queried_colours`, a kind may have several models if each
    colour has exactly one), and, when `clutter` lists clutter kinds (`queried_clutter`), their models."""
    by_id = {a.id: a for a in m.assets}
    out = [p for k, s in m.sources.items() for p in _source_problems(k, s)]
    names = [a.id.lower() for a in m.assets]  # ids name cache folders, which Windows compares without case
    out += [f"{n}: duplicate id" for n in sorted({n for n in names if names.count(n) > 1})]
    out += [p for a in m.assets for p in _asset_problems(a, m.sources.get(a.source), by_id)]
    out += _reference_problems(m, by_id)
    if catalog_kinds is not None:
        out += coverage_problems(m, catalog_kinds, light_kinds, mover_kinds, colours)
    if clutter is not None:
        out += clutter_coverage_problems(m, clutter)
    return out


# -- the cache ---------------------------------------------------------------------------------------------------
def is_dataset_root(path: Path) -> bool:
    """A dataset root holds the agent-facing tree (`scripts/`, or `episodes/` in the releases before the redesign) and
    `private/`."""
    return ((path / "scripts").is_dir() or (path / "episodes").is_dir()) and (path / "private").is_dir()


def cache_dir() -> Path:
    """Where downloaded assets live: `STBENCH_ASSET_CACHE`, else `~/.cache/stbench/assets` on every platform. Not
    %LOCALAPPDATA%: Windows redirects a packaged app's AppData writes (the Claude desktop app's sessions among them)
    into a private folder, so a shell outside it would find another, empty cache. Refuses a path inside a git work
    tree, this checkout or a dataset, so that assets never land in git or beside agent-facing files."""
    env = os.environ.get("STBENCH_ASSET_CACHE")
    path = Path(env) if env else Path.home() / ".cache" / "stbench" / "assets"
    path = path.expanduser().resolve()
    for above in (path, *path.parents):
        in_checkout = above == REPO and (REPO / "pyproject.toml").is_file()
        if (above / ".git").exists() or in_checkout or is_dataset_root(above):
            raise AssetError(f"the asset cache {path} is inside {above}, a repository or a dataset; "
                             "set STBENCH_ASSET_CACHE to a folder outside both")
    return path


def asset_dir(asset: Asset, cache: Path | None = None) -> Path:
    return (cache or cache_dir()) / asset.id


def file_path(asset: Asset, f: AssetFile, cache: Path | None = None) -> Path:
    return asset_dir(asset, cache).joinpath(*f.path.split("/"))


def extract_dir(asset: Asset, cache: Path | None = None) -> Path:
    return asset_dir(asset, cache) / EXTRACT_DIR


def hdri_path(m: Manifest, catalog: str, cache: Path | None = None) -> Path | None:
    """The cached file of a catalog's sky, or None where the manifest names none (the procedural sky stays). It is not
    checked here: every assets render passes `asset_render.require_library`, which checks that the cache holds it."""
    sky = m.sky_for(catalog)
    return file_path(sky, sky.files[0], cache) if sky else None


def texture_dir(m: Manifest, asset_id: str, cache: Path | None = None) -> Path:
    """The folder a `texture` set's maps are unpacked to (`<id>_1K-JPG_Roughness.jpg` and so on; `materials.find_map`
    finds one by its suffix). A `material` set lies in the same place, for a layer that reads another map of it."""
    found = m.asset(asset_id)
    if found.role not in ("texture", "material"):
        raise AssetError(f"{asset_id!r} is a {found.role}, not a texture set")
    return extract_dir(found, cache)
