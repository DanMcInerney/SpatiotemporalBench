"""Where map files are found: the maps shipped in `stbench/maps` and the private ones in the folders `STBENCH_MAPS`
names (standard library only, so the loader, the renderer and the tests all read it).

A held-out layout's map never ships: it lives in the held-out dataset root, at `private/maps/<name>.json`, outside the
repository and the wheel (`src/stbench/maps/README.md`, "Private maps"). `map_bytes` is the one place that turns a map's
name into its file, and the only module that reads the package's map folder.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

MAPS_ENV = "STBENCH_MAPS"  # folders of private maps, os.pathsep-separated, searched before the bundled maps
HELD_OUT = ("home_large", "cross_dock", "courtyards")  # the held-out layouts (design §3, D3): never shipped


def map_folders() -> tuple[Path, ...]:
    """The folders STBENCH_MAPS names, in order, empty entries skipped and a folder named twice kept once. ValueError
    for one that is not a folder: a typo must not fall back to the bundled maps silently."""
    folders: list[Path] = []
    for entry in os.environ.get(MAPS_ENV, "").split(os.pathsep):
        if not entry:
            continue
        folder = Path(entry)
        if not folder.is_dir():
            raise ValueError(f"{MAPS_ENV} names {entry!r}, which is not a folder")
        if all(not folder.samefile(seen) for seen in folders):
            folders.append(folder)
    return tuple(folders)


def bundled_maps() -> tuple[str, ...]:
    """The names of the maps shipped in `stbench/maps` (`load_map` takes any of them), sorted."""
    return tuple(sorted(p.name.removesuffix(".json") for p in resources.files("stbench.maps").iterdir()
                        if p.name.endswith(".json")))


def private_maps() -> tuple[str, ...]:
    """The names of the maps in the STBENCH_MAPS folders (the stems of their `*.json`), sorted."""
    return tuple(sorted({p.name.removesuffix(".json") for folder in map_folders() for p in folder.glob("*.json")}))


def is_map_path(name_or_path: str | Path) -> bool:
    """Whether `map_bytes` reads this as a file path (a `.json`, or an existing file) rather than as a map's name."""
    p = Path(str(name_or_path))
    return p.suffix == ".json" or p.is_file()


def map_bytes(name_or_path: str | Path) -> bytes:
    """A map file's bytes. A path (a `.json`, or an existing file) is read as given. A name is looked for as
    `<name>.json` in each STBENCH_MAPS folder, then among the bundled maps. ValueError when the name is found in more
    than one place; FileNotFoundError, naming STBENCH_MAPS, when it is nowhere."""
    if is_map_path(name_or_path):
        return Path(str(name_or_path)).read_bytes()
    name = str(name_or_path)
    found = [folder / f"{name}.json" for folder in map_folders() if (folder / f"{name}.json").is_file()]
    shipped = resources.files("stbench.maps").joinpath(f"{name}.json")
    if shipped.is_file():
        found.append(shipped)
    if len(found) > 1:
        raise ValueError(f"the map {name!r} is found in more than one place ({', '.join(map(str, found))}): "
                         "one name is one map")
    if not found:
        raise FileNotFoundError(f"no map named {name!r}: it is not among the bundled maps ({', '.join(bundled_maps())})"
                                f" nor in a folder {MAPS_ENV} names ({os.environ.get(MAPS_ENV) or 'unset'})")
    return found[0].read_bytes()
