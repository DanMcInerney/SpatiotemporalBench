"""The private map path (M4.0): a held-out map lives in the held-out dataset root, never in the package or the
repository, and every loader finds it by name through `STBENCH_MAPS`. A private map here is a renamed copy of
`apartment_a`, written to a temp folder the variable names (set with `monkeypatch` only, so the tests stay
parallel-safe)."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from stbench import cli
from stbench import episode as ep
from stbench.generate import generate_script
from stbench.map_files import HELD_OUT, MAPS_ENV, bundled_maps, map_bytes, map_folders, private_maps
from stbench.mapspec import load_map
from test_architecture import MODULES, RENDER, ROOT, SRC, modules_under, string_constants

PRIVATE = "apartment_private_copy"


def write_map(folder: Path, name: str = PRIVATE, file_name: str | None = None) -> bytes:
    """A copy of `apartment_a` named `name`, written to `folder/<file_name or name>.json`; its bytes."""
    doc = json.loads(map_bytes("apartment_a"))
    doc["name"] = name
    data = json.dumps(doc, indent=1).encode("utf-8")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{file_name or name}.json").write_bytes(data)
    return data


def test_a_private_map_loads_by_name_through_stbench_maps(tmp_path, monkeypatch):
    """Contract (plan §5.3, M4.0): with `STBENCH_MAPS` naming a folder, `load_map` finds a map in it by name, and
    `private_maps` lists it (empty entries of the variable are skipped); without the
    variable the name is `FileNotFoundError` naming `STBENCH_MAPS`; a variable naming a folder that does not exist is a
    `ValueError`, even for a bundled map. Regression caught: a held-out map that loads only by path (so nothing after
    `generate` could find it again), or a typo in the variable that silently loads nothing and falls back to the
    bundled maps."""
    write_map(tmp_path / "private")
    monkeypatch.setenv(MAPS_ENV, "")
    with pytest.raises(FileNotFoundError, match=MAPS_ENV):
        load_map(PRIVATE)
    assert private_maps() == () and map_folders() == ()
    monkeypatch.setenv(MAPS_ENV, os.pathsep.join(["", str(tmp_path / "private"), ""]))
    m = load_map(PRIVATE)
    assert m.name == PRIVATE
    assert private_maps() == (PRIVATE,) and map_folders() == (tmp_path / "private",)
    assert load_map("apartment_a").name == "apartment_a"  # a bundled map is still found beside it
    monkeypatch.setenv(MAPS_ENV, str(tmp_path / "typo"))
    for call in (private_maps, lambda: load_map("apartment_a")):
        with pytest.raises(ValueError, match="not a folder"):
            call()


def test_one_name_one_map(tmp_path, monkeypatch):
    """Contract (M4.0): a map is found by one name and the name is its own, so a timeline's map is the one it was made
    on: refused with `ValueError` are a private `apartment_a.json` beside the bundled one, one name in two private
    folders, and a file `x.json` whose `name` is `y` loaded by the name `x`. Read as a path the file is still read as
    given, and one folder named twice in the variable is one place, not two. Regression caught: a private map that
    shadows a bundled one or another private one (two maps under one name, so the map a timeline names and the file
    the renderer reads can differ), or a map file renamed on disk that loads under the wrong name."""
    shadow, first, second = tmp_path / "shadow", tmp_path / "first", tmp_path / "second"
    write_map(shadow, "apartment_a")
    write_map(first)
    write_map(second)
    write_map(second, "y", file_name="x")
    monkeypatch.setenv(MAPS_ENV, str(shadow))
    with pytest.raises(ValueError, match="more than one place"):
        load_map("apartment_a")
    monkeypatch.setenv(MAPS_ENV, os.pathsep.join([str(first), str(first)]))
    assert load_map(PRIVATE).name == PRIVATE  # one folder named twice is one place
    monkeypatch.setenv(MAPS_ENV, os.pathsep.join([str(first), str(second)]))
    with pytest.raises(ValueError, match="more than one place"):
        load_map(PRIVATE)
    monkeypatch.setenv(MAPS_ENV, str(second))
    with pytest.raises(ValueError, match="'y'"):
        load_map("x")
    assert load_map(second / "x.json").name == "y"  # by path the file is read as given


def test_a_script_on_a_private_map_generates_and_validates(tmp_path, monkeypatch):
    """Contract (M4.0): with `STBENCH_MAPS` set, a script generated on a private map and written by `write_script`
    passes `stbench validate`, which loads the map by the name its timeline records and runs the leak scan with that
    name. Regression caught: a loader that reads only the bundled maps (the generator's, or
    `validate`'s), so a held-out script could be made and never checked."""
    write_map(tmp_path / "private")
    monkeypatch.setenv(MAPS_ENV, str(tmp_path / "private"))
    tl = generate_script(PRIVATE, 1, "vacuum", "vacuum/1", 1)
    assert tl.episode.map == PRIVATE
    paths = ep.write_script(tmp_path / "data", tl)
    assert cli.main(["validate", str(paths.root)]) == 0
    monkeypatch.delenv(MAPS_ENV)
    with pytest.raises(FileNotFoundError, match=MAPS_ENV):  # the same script cannot be validated without the map
        cli.main(["validate", str(paths.root)])


def test_no_held_out_map_ships_or_is_committed():
    """Contract (design §3, D3; recommendations §2.6 item 3): the package holds no held-out layout, and no tracked
    file of the repository is one. `bundled_maps` and `HELD_OUT` share nothing; every `*.json` under `src/stbench` that
    has a map's `name`, `bounds` and `regions` is a bundled map (the wheel is that folder as it stands), and the scan
    finds every bundled map; no file `git ls-files` lists is a map named in `HELD_OUT` (the repository and the sdist
    are published; skipped outside a git checkout). Regression caught: a held-out map copied into the package, or
    committed as a fixture or an example."""
    assert set(bundled_maps()).isdisjoint(HELD_OUT)

    def named_maps(paths: list[Path]) -> dict[Path, str]:
        found = {}
        for path in paths:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(doc, dict) and {"name", "bounds", "regions"} <= set(doc):
                found[path] = str(doc.get("name"))
        return found

    in_package = named_maps(sorted(SRC.rglob("*.json")))
    assert sorted(p.stem for p in in_package) == list(bundled_maps())  # every bundled map found, none else in the wheel
    assert all(p.stem == name for p, name in in_package.items())
    try:
        listed = subprocess.run(["git", "ls-files", "-z", "*.json"], cwd=ROOT, capture_output=True, text=True,
                                check=True).stdout.split("\0")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    tracked = named_maps([ROOT / rel for rel in listed if rel])
    assert ROOT / "src" / "stbench" / "maps" / "apartment_a.json" in tracked  # the scan sees the bundled maps
    assert [(p.relative_to(ROOT).as_posix(), n) for p, n in tracked.items() if n in HELD_OUT] == []


def joins_maps_onto_a_package(source: str) -> bool:
    """True when a module calls `files(...)` (`importlib.resources`) and holds the string constant `maps`: the form
    `resources.files("stbench").joinpath("maps", ...)`, which names the package folder in two constants."""
    tree = ast.parse(source)
    names = [getattr(n.func, "id", None) or getattr(n.func, "attr", None) for n in ast.walk(tree)
             if isinstance(n, ast.Call)]
    calls_files = "files" in names
    return calls_files and "maps" in string_constants(source)


def test_map_files_are_read_in_one_place():
    """Contract (M4.0): the package's map folder is read by `map_files.py` only: no other module of `src/stbench` or
    `render/` holds the package name `stbench.maps` as a string constant (docstrings aside), nor joins `maps` onto a
    package with `resources.files(...)`, so every loader resolves a name through `STBENCH_MAPS`. Regression caught: a
    loader that reads the package directly and so never finds a private map, as `build_scene._raw_spec` did. The scan
    sees those two forms, not every spelling (a `Path(__file__).parent / "maps"` passes it); for the renderer,
    `test_private_map_blender.py` loads a private map through `_raw_spec` itself."""
    pattern = re.compile(r"stbench[./]maps")
    modules = {f"src/stbench/{n}": p for n, p in MODULES.items()} | {f"render/{n}": p
                                                                      for n, p in modules_under(RENDER).items()}
    holders = sorted(n for n, p in modules.items()
                     if any(pattern.search(c) for c in string_constants(p.read_text(encoding="utf-8"))))
    assert "src/stbench/map_files" in holders  # the scan finds the one reader, so the rule below is not vacuous
    assert holders == ["src/stbench/map_files"]
    joiners = sorted(n for n, p in modules.items() if joins_maps_onto_a_package(p.read_text(encoding="utf-8")))
    assert joiners == []
    # the second scan sees the form it is for, and not the package data or a dataset folder that happen to be nearby
    assert joins_maps_onto_a_package('from importlib import resources\n'
                                     'x = resources.files("stbench").joinpath("maps", "apartment_a.json")\n')
    assert joins_maps_onto_a_package('from importlib.resources import files\nx = files(__package__) / "maps"\n')
    assert not joins_maps_onto_a_package('x = resources.files("stbench").joinpath("prices.json")\n')
    assert not joins_maps_onto_a_package('x = root / "private" / "maps"\n')
