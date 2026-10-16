# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""How a catalog's places look to the render, by what a region is: one `look_<catalog>.py` a catalog.

Plain Python (no `bpy`, no `stbench`), so a free test loads it by path as `tests/dressing_kit.py` loads the layouts. A
region has a *kind* (`Region.kind`: the map's optional `kind`, else the region's name) drawn from its catalog's closed
vocabulary, `look_<catalog>.KINDS`, and every render table that depends on what a place is (its floor, its finish, the
paint on its walls, the light it needs, how it is dressed) is a dict in that catalog's module with exactly those kinds
as keys. A catalog's realism lane owns its module and adds a kind or a table there, never in a shared file.

Nothing here falls back: a catalog with no module, a table the module does not hold and a kind the table has no row for
each raise `LookError`, naming what is known, so that a map whose rooms nothing knows fails where it is built instead of
rendering with a stand-in floor, a default lamp or no dressing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


class LookError(LookupError):
    """A catalog, table or region kind the render has no look for."""


def catalogs() -> tuple[str, ...]:
    """The catalogs that have a look module (`look_<catalog>.py` beside this file)."""
    return tuple(sorted(p.stem.removeprefix("look_") for p in HERE.glob("look_*.py")))


def module(catalog: str):
    """`look_<catalog>`, or a `LookError` naming the catalog and the ones there are. It is loaded from its file beside
    this one, so it is found whatever `sys.path` holds, and it is the module `import look_<catalog>` gives."""
    if catalog not in catalogs():
        raise LookError(f"the render has no look module for catalog {catalog!r} (render/look_{catalog}.py); "
                        f"the catalogs with one are {list(catalogs())}")
    name = f"look_{catalog}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
        loaded = sys.modules[name] = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(loaded)
        except BaseException:  # a half-run module must not be what the next call finds
            del sys.modules[name]
            raise
    return sys.modules[name]


def kinds(catalog: str) -> frozenset[str]:
    """The catalog's closed vocabulary of region kinds."""
    return module(catalog).KINDS


def kind_of(region) -> str:
    """The kind of a region: a parsed map's `Region.kind` or, for a region as the map spec dict holds it, its `kind`,
    else its name (the same rule `mapspec` applies when it parses)."""
    if isinstance(region, dict):
        return str(region.get("kind", region["name"]))
    return region.kind or region.name


def get(catalog: str, table: str, kind: str):
    """`look_<catalog>.<table>[kind]`; a `LookError` for a catalog with no look, a catalog that holds no such table
    (or declares it `None`: it has no use for one) and a kind the table has no row for."""
    mod = module(catalog)
    rows = getattr(mod, table, None)
    if rows is None:
        raise LookError(f"look_{catalog} holds no {table} table (a catalog that needs none declares it None)")
    if kind not in rows:
        raise LookError(f"the {catalog} catalog's {table} has no row for region kind {kind!r}; "
                        f"its vocabulary is {sorted(mod.KINDS)}")
    return rows[kind]
