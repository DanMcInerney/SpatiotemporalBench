"""Packaging and licences (Release 0.1, plan R4): what the wheel carries, and the three licence texts."""

from __future__ import annotations

import re
import subprocess
import tomllib
from importlib import resources
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
# Package files that the code reads at run time: the sandbox's shim (copied into the container), the map and the asset
# manifest, the export's dataset licence and card, the price table and the run viewer's page (`stbench view`). A wheel
# without one fails at the first command that needs it.
PACKAGE_FILES = ["shim.py", "maps/apartment_a.json", "assets/manifest.json", "release/CC-BY-4.0.txt",
                 "release/dataset-card.md", "prices.json", "viewer/page.html", "viewer/page.css", "viewer/util.js",
                 "viewer/log.js", "viewer/page.js"]


def test_the_wheel_is_the_package_folder_and_carries_every_file_the_code_reads():
    """Contract (plan R4): the wheel is `src/stbench` as it stands, with no `include` or `exclude` that could drop a
    file, and no `render/` (GPL, Blender-only); the shim, the map, the asset manifest, the dataset licence and the
    card sit inside it and are reachable by `importlib.resources`. Regression caught: the shim moved out of the
    package, an `exclude` added to the build, or package data placed beside the package, so that an installed
    `stbench run --local`, `eval` or `export` fails where a checkout works."""
    wheel = PYPROJECT["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel == {"packages": ["src/stbench"]}
    package = resources.files("stbench")
    for rel in PACKAGE_FILES:
        node = package.joinpath(*rel.split("/"))
        assert node.is_file(), rel
        assert (ROOT / "src" / "stbench" / rel).is_file(), rel


def test_the_sdist_holds_the_mit_code_and_not_the_gpl_scripts_the_tests_or_the_data():
    """Contract (review S5, final review S9): the sdist is `src`, the docs, the examples and the top-level files the
    README names, chosen by `only-include`, so none of the GPL-3.0 Blender code (`render/`) and none of
    `data/` can ride along under the package's `MIT` metadata, and no `tests/` that would fail from an sdist (they read
    `render/` and the repository's documents; CI runs them from the checkout). Without the table hatch packs every file
    git tracks (over 300 files, `render/` among them). `docs/figures` (frames of the CC BY 4.0 dataset,
    shown by the README through an absolute URL) is excluded on purpose, so no dataset content rides along under the
    `MIT` metadata. Regression caught: the table dropped or widened, a GPL folder or `tests` added to it, the figures
    folder let into the sdist. The built listing was checked by hand (the review's report)."""
    sdist = PYPROJECT["tool"]["hatch"]["build"]["targets"]["sdist"]
    only = set(sdist["only-include"])
    assert {"src", "docs", "examples", "LICENSE", "README.md"} <= only
    assert not {"render", "tools", "data", "episodes", "private", "runs", "tests"} & only
    assert "docs/figures" in sdist["exclude"]
    assert all((ROOT / name).exists() for name in only)


def test_no_package_file_is_ignored_by_git():
    """Contract: the wheel builder skips what `.gitignore` ignores, so the files above must not match it (the file
    ignores `data/` and `episodes/`, folders a package file could be put in by mistake). Regression caught: a new
    ignore line that silently drops package data from the wheel. Skipped outside a git checkout."""
    paths = [f"src/stbench/{rel}" for rel in PACKAGE_FILES]
    try:
        done = subprocess.run(["git", "-C", str(ROOT), "check-ignore", *paths], capture_output=True, text=True,
                              timeout=60)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git is not available")
    if done.returncode == 128:
        pytest.skip("not a git checkout")
    assert done.stdout == "" and done.returncode == 1


def test_the_licence_files_are_the_canonical_texts():
    """Contract (DECISIONS, Release 0.1): the code is MIT with the authors' line, `render/` is GPL-3.0 (the FSF text),
    the dataset is CC BY 4.0 (the Creative Commons legal code, shipped inside the package for `stbench export`), and
    the package metadata says MIT. Regression caught: a licence file edited or truncated, the wrong holder line, or
    metadata that disagrees with the file."""
    mit = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert mit.startswith("MIT License\n\nCopyright (c) 2026 the SpatiotemporalBench authors\n\n"
                          "Permission is hereby granted, free of charge")
    assert mit.rstrip().endswith("ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS "
                                 "IN THE SOFTWARE.")
    gpl = (ROOT / "render" / "LICENSE").read_text(encoding="utf-8")
    assert gpl.lstrip().startswith("GNU GENERAL PUBLIC LICENSE\n                       Version 3, 29 June 2007")
    assert "END OF TERMS AND CONDITIONS" in gpl and len(gpl.splitlines()) > 600
    cc = resources.files("stbench").joinpath("release", "CC-BY-4.0.txt").read_text(encoding="utf-8")
    assert cc.startswith("Attribution 4.0 International")
    assert "Creative Commons Attribution 4.0 International Public License" in cc and len(cc.splitlines()) > 350
    assert PYPROJECT["project"]["license"] == "MIT" and PYPROJECT["project"]["license-files"] == ["LICENSE"]


def test_every_blender_script_carries_the_gpl_header():
    """Contract: a script that uses Blender's Python API is GPL-3.0-or-later, wherever it lives (`render/`, a test
    driver that runs inside Blender), and says so in its first lines; the rest of
    the repository is MIT. The scripts are `render/*.py` and every other `.py` of the repository's own folders that
    imports `bpy`, `bmesh` or `mathutils`. Regression caught: a new Blender script without the notice, which would
    leave it under the repository's MIT licence (a test driver did)."""
    imports = re.compile(r"^\s*(?:import|from)\s+(?:bpy|bmesh|mathutils)\b", re.MULTILINE)
    folders = ("src", "render", "tools", "tests", "examples")
    own = [p for folder in folders for p in sorted((ROOT / folder).rglob("*.py")) if "__pycache__" not in p.parts]
    importing = (p for p in own if imports.search(p.read_text(encoding="utf-8")))
    scripts = sorted({*(ROOT / "render").glob("*.py"), *importing})
    assert len(scripts) > 10 and {p.parent.name for p in scripts} >= {"render", "tests"}

    def notice(f: Path) -> str:
        head = " ".join(f.read_text(encoding="utf-8").splitlines()[:6])  # the notice wraps over comment lines
        return " ".join(head.replace("#", " ").split())

    assert [f.relative_to(ROOT).as_posix() for f in scripts if "GNU General Public License" not in notice(f)] == []
