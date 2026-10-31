"""Starting Blender: where it is, which version it is, and running one of the `render/` scripts in it.

`render_cli` (episodes) and `asset_render` (the asset library and its preview) both start Blender this way, so the path
(`STBENCH_BLENDER`), the command line and the error summary have one owner. A Python exception in a script makes
Blender exit with status 1 here (`--python-exit-code`); by default it would exit 0 with the traceback on stderr.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from functools import cache
from pathlib import Path

# Where the installer puts Blender when it is on no PATH; Linux installs put `blender` on the PATH.
DEFAULT_BLENDER = {"win32": "C:/Program Files/Blender Foundation/Blender 5.2/blender.exe",
                   "darwin": "/Applications/Blender.app/Contents/MacOS/Blender"}.get(sys.platform, "blender")
SRC_DIR = Path(__file__).resolve().parents[1]  # the directory that holds the stbench package
RENDER_DIR = Path(__file__).resolve().parents[2] / "render"


class BlenderError(Exception):
    """Blender is missing, or a script run in it failed; the message says where, what is wrong and what to do."""


def blender_path() -> str:
    """`STBENCH_BLENDER`, else the `blender` on the PATH, else where this system's installer puts it."""
    return os.environ.get("STBENCH_BLENDER") or shutil.which("blender") or DEFAULT_BLENDER


def script(name: str) -> Path:
    """A script in `render/` of the repository checkout this package was installed from."""
    path = RENDER_DIR / name
    if not path.exists():
        raise BlenderError(f"render script not found at {path}; stbench render needs the repository checkout "
                           "(an editable install), not an installed wheel")
    return path


def command(script_name: str, *args: str) -> list[str]:
    """`blender --background --factory-startup --python <script> -- <args>`."""
    return [blender_path(), "--background", "--factory-startup", "--python-exit-code", "1", "--python",
            str(script(script_name)), "--", *args]


def _require(exe: str) -> None:
    if not Path(exe).exists():
        raise BlenderError(f"Blender not found at {exe}; set STBENCH_BLENDER to the Blender executable")


@cache
def _version_of(exe: str) -> str:
    out = subprocess.run([exe, "--background", "--version"], capture_output=True, text=True, encoding="utf-8",
                         errors="replace", check=False).stdout
    found = re.search(r"Blender (\d+\.\d+\.\d+)", out)
    if not found:
        raise BlenderError(f"cannot read the version from `{exe} --version`: {out[:200]!r}")
    return found.group(1)


def blender_version() -> str:
    """"5.2.1": the version of the Blender at `blender_path()`, which names the asset library built with it."""
    exe = blender_path()
    _require(exe)
    return _version_of(exe)


def run(cmd: list[str], what: str) -> subprocess.CompletedProcess:
    """Run a Blender command line to completion; a non-zero exit is a `BlenderError` that quotes the script's errors."""
    _require(cmd[0])
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if proc.returncode != 0:
        lines = proc.stdout.splitlines() + proc.stderr.splitlines()
        starts = [i for i, ln in enumerate(lines) if ln.startswith("Traceback")]
        errors = [ln for ln in lines if "STBENCH-ERROR" in ln or "Error" in ln or "Traceback" in ln]
        shown = lines[starts[-1]:][:16] if starts else (errors or proc.stdout.splitlines())[-12:]
        raise BlenderError(f"Blender exited with status {proc.returncode} while {what}:\n" + "\n".join(shown))
    return proc
