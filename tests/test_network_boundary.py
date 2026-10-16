"""One owner for network access: only `asset_fetch` opens a connection. The scan reads every module under `src/stbench`
and `render/`, subpackages included, so a module in a new folder is checked like a flat one."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "stbench"
RENDER = ROOT / "render"
ALLOWED = {"asset_fetch"}  # the downloader
# The standard-library network modules, and the common third-party clients (none is a dependency; one added would be
# network access hidden behind a package).
NETWORK = {"urllib", "http", "socket", "ssl", "urllib3", "requests", "httpx", "aiohttp", "ftplib", "smtplib",
           "poplib", "imaplib", "telnetlib", "xmlrpc", "socketserver"}
PURE = {"urllib.parse"}  # URL string handling: it opens nothing, and the manifest checks hosts with it


def network_imports(source: str) -> list[str]:
    """The network modules a source file imports (`import urllib.request`, `from http import client`, `import ssl as s`)
    or loads by a literal name (`__import__("socket")`, `importlib.import_module("ssl")`). `urllib.parse` is not one."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.split(".")[0] in NETWORK and a.name not in PURE]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").split(".")[0] in NETWORK:
            if node.module not in PURE and not (node.module == "urllib" and {a.name for a in node.names} == {"parse"}):
                found.append(node.module)
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            func = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            name = node.args[0].value
            if func in ("__import__", "import_module") and isinstance(name, str) and name.split(".")[0] in NETWORK:
                found.append(name)
    return found


def test_scanner_finds_every_form_of_a_network_import():
    """Contract: the scan behind the rule below sees `import urllib.request`, `import http.client as h`,
    `from urllib import request`, `from socket import *`, `import ssl`, a literal `__import__` and
    `importlib.import_module`, and ignores URL parsing (`urllib.parse`), a relative import and a module that only has
    a network word in its name. Regression caught: a scanner that misses a form, so the rule below passes vacuously."""
    src = "\n".join([
        "import urllib.request", "import http.client as h", "from urllib import request", "from socket import *",
        "import ssl", "x = __import__('socket')", "y = importlib.import_module('http.client')",
        "from urllib.parse import urlparse", "from urllib import parse", "import urllib.parse",
        "from .urllib_like import z", "import urllib_like", "import json", "z = importlib.import_module(name)"])
    assert network_imports(src) == ["urllib.request", "http.client", "urllib", "socket", "ssl", "socket", "http.client"]


def importers_under(root: Path, prefix: str = "") -> dict[str, list[str]]:
    """The modules below `root` that import a network module, each keyed by `prefix` and its path from `root` without
    `.py` (`asset_fetch`, `jobs/x`), so two modules with one stem in different folders stay two keys."""
    return {prefix + f.relative_to(root).with_suffix("").as_posix(): hits
            for f in sorted(root.rglob("*.py")) if "__pycache__" not in f.parts
            if (hits := network_imports(f.read_text(encoding="utf-8")))}


def test_the_network_scan_sees_subpackages(tmp_path):
    """Contract: the scan reads a module in a subfolder like a flat one and keys it by its path from the root. On a
    temporary tree, `pkg/sub/deep.py` (`import socket`) is reported as `sub/deep` with `["socket"]`, and `pkg/flat.py`
    (`import json`) is not reported. Regression caught: a network import hidden in a subfolder (`jobs/x.py`) because
    the scan globbed the folder flat (map X-H10)."""
    (tmp_path / "pkg" / "sub").mkdir(parents=True)
    (tmp_path / "pkg" / "flat.py").write_text("import json\n", encoding="utf-8")
    (tmp_path / "pkg" / "sub" / "deep.py").write_text("import socket\n", encoding="utf-8")
    assert importers_under(tmp_path / "pkg") == {"sub/deep": ["socket"]}


def test_two_modules_with_one_stem_do_not_hide_each_other(tmp_path):
    """Contract: importers are keyed by path, not by file stem. On a temporary tree, `x.py` (`import socket`) and
    `sub/x.py` (`import ssl`) are two keys, `x` and `sub/x`. Regression caught: a stem-keyed dict where the second
    `x.py` overwrites the first, so an offender in one folder passes because another module shares its name."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "x.py").write_text("import socket\n", encoding="utf-8")
    (tmp_path / "sub" / "x.py").write_text("import ssl\n", encoding="utf-8")
    assert importers_under(tmp_path) == {"x": ["socket"], "sub/x": ["ssl"]}


def test_only_asset_fetch_touches_the_network():
    """Contract (AGENTS: nothing special-cased, VISION rule 6, spend and download control): the package opens a network
    connection in exactly one module, the asset downloader, with an injected transport and a test that never
    uses the real one; the Blender scripts open none (their keys start with `render/`, so a `render/asset_fetch.py`
    would not borrow the downloader's exemption). Regression caught: a stray `urllib` or `socket` in, say,
    `build.py`, a baseline, a sandbox helper, a module of a new subpackage or a render script, an unreviewed way out
    for a download or a paid call."""
    importers = {**importers_under(SRC), **importers_under(RENDER, "render/")}
    assert importers["asset_fetch"], "the downloader imports no network module: the scan finds nothing"
    assert {name: hits for name, hits in importers.items() if name not in ALLOWED} == {}
