"""The manifest's records: `Source`, `AssetFile`, `Asset`, `Manifest` and `AssetError`. Plain data and lookups, nothing
else: loading, validation and the cache layout are `asset_manifest`'s, which re-exports these names, so
`asset_manifest.Asset` is this class.
"""

from __future__ import annotations

from dataclasses import dataclass


class AssetError(Exception):
    """A manifest, cache or download problem; the message says what is wrong and what to do."""


@dataclass(frozen=True, eq=False)
class Source:
    name: str
    licence: str
    hosts: tuple[str, ...]
    licence_url: str | None = None
    attribution: str | None = None  # the credit line: CC BY requires it, a CC0 source may ask
    checked: str | None = None  # the date the licence page was read
    kind: str = "hosted"  # "hosted": files downloaded from `hosts`; "generated": made by a recipe, nothing downloaded
    generator: dict | None = None  # generated: the recipe's path and its inputs (name, kind, licence, url)


@dataclass(frozen=True, eq=False)
class AssetFile:
    path: str  # relative to the asset's cache folder
    url: str = ""  # empty for the file of a generated source: the recipe makes it
    extract: bool = False  # a zip, unpacked to <asset folder>/x/


@dataclass(frozen=True, eq=False)
class Asset:
    id: str  # also the timeline's `entity.asset` value
    source: str
    role: str  # "model" | "material" | "dressing" | "hdri" | "texture" | "clutter"
    licence: str
    title: str = ""
    page: str | None = None
    changes: str | None = None  # what we did to it: CC BY asks for an indication of changes
    kind: str | None = None
    mobility: str | None = None  # "carried" | "light" | "mover"; "clutter" for a clutter model
    colour: str | None = None  # clutter: the palette key an untinted model shows (a tinted one takes the entity's)
    for_colours: tuple[str, ...] = ()  # model: the colours it is drawn in, where a kind has several models (none: all)
    logo_review: str | None = None  # clutter: "none" | "minor" | "dominant", from the asset's preview
    files: tuple[AssetFile, ...] = ()
    estimate_bytes: int | None = None  # the download's size, shown before anything is fetched
    entry: str | None = None  # the file Blender imports
    format: str | None = None
    transform: dict | None = None  # unit_scale, up, rotate_deg: brings the asset to Z up, front toward +x
    pivot: str | None = None
    native_bbox_m: list | None = None  # measured by the library build and committed
    fit: dict | None = None
    tint: dict | None = None
    compose: dict | None = None
    alternate_for: str | None = None  # never fetched unless asked for
    builder: str | None = None  # procedural geometry
    materials: dict | None = None
    tile_m: list | None = None
    uses: tuple[str, ...] = ()
    drop_slots: tuple[str, ...] = ()  # dressing or clutter: slots whose faces the library removes (glass left opaque)
    rest_pose: tuple[str, ...] = ()  # dressing: glTF nodes whose rotation is reset (a door the file leaves open)
    parts: tuple[dict, ...] = ()  # model: what a generated mesh is made of, each part with its own licence

    def download_size(self) -> int:
        """Bytes to download, as the manifest estimates (0 for an asset it gives no estimate for)."""
        return self.estimate_bytes or 0


@dataclass(frozen=True, eq=False)
class Manifest:
    sources: dict[str, Source]
    assets: tuple[Asset, ...]
    palette_materials: dict[str, str]  # palette key ("oak") -> material asset id
    floors: dict[str, str]  # region name -> material asset id, or "shader:<name>" for a procedural floor (concrete)
    walls: str | None
    skies: dict[str, str]  # catalog name -> hdri asset id: the sky of a catalog's maps (none: the procedural sky)
    sha256: str  # of the manifest file bytes: only the name of the asset library built from it (a cache key)

    def asset(self, asset_id: str) -> Asset:
        found = next((a for a in self.assets if a.id == asset_id), None)
        if found is None:
            raise AssetError(f"no asset {asset_id!r} in the manifest")
        return found

    def models_of(self, kind: str) -> list[Asset]:
        """The primary models of a kind, in manifest order (an alternate is never one of them)."""
        return [a for a in self.assets if a.role == "model" and a.kind == kind and not a.alternate_for]

    def model_for(self, kind: str, colour: str | None = None) -> Asset:
        """The one model for a kind in a colour (an alternate is never it). A kind with one model has it in every
        colour. A kind with several gives each its `for_colours`, so a description (colour, kind) always has the same
        model: the models that list the colour, else the ones that list none; exactly one must be left."""
        models = self.models_of(kind)
        if len(models) > 1:
            models = [a for a in models if colour in a.for_colours] or [a for a in models if not a.for_colours]
        if len(models) != 1:
            raise AssetError(f"{len(models)} models for kind {kind!r}" + (f" in {colour!r}" if colour else "")
                             + " in the manifest, need exactly one")
        return models[0]

    def clutter_models(self, kind: str) -> list[Asset]:
        """The models of a clutter kind (role `clutter`), in manifest order: several a kind, each with its own box."""
        return [a for a in self.assets if a.role == "clutter" and a.kind == kind]

    def material(self, asset_id: str) -> Asset:
        return self.of_role(asset_id, "material")

    def of_role(self, asset_id: str, role: str) -> Asset:
        found = self.asset(asset_id)
        if found.role != role:
            raise AssetError(f"{asset_id!r} is a {found.role}, not a {role}")
        return found

    def sky_for(self, catalog: str) -> Asset | None:
        """The HDRI of a catalog's maps, or None where the manifest names none (the procedural sky stays)."""
        ref = self.skies.get(catalog)
        return self.of_role(ref, "hdri") if ref else None
