"""The dressing of every held-out map present (M4-integration call 8): the render-only objects its catalog's layout lays
out (`render/dressing_layout_<catalog>.py`) are registered classes of the catalog, in their phases, stand only where the
validity guard allows and are the same every run. Held-out maps come from `STBENCH_MAPS` as in `test_heldout_maps.py`,
which this module shares its parameters with; no layout fact is stated."""

from __future__ import annotations

import copy

import pytest

from dressing_kit import layout_of, layout_path, raw_map
from stbench.dressing_zones import DressingGuard
from stbench.scenery import layout_problems
from test_heldout_maps import maps_present


@pytest.fixture(scope="module", params=maps_present())
def laid_out(request):
    """(the raw map, its catalog, its dressing), once a map a module; a catalog with no layout module yet skips."""
    raw = raw_map(request.param)
    catalog = raw.get("catalog", "home")
    if not layout_path(catalog).exists():
        pytest.skip(f"render/dressing_layout_{catalog}.py is not written yet")
    return raw, catalog, layout_of(catalog, raw)


def test_every_held_out_map_dresses_cleanly(laid_out):
    """Contract: its catalog's layout emits only classes registered for the catalog, in
    the phase the registry gives them (`scenery.layout_problems`, less the lines saying a class is never emitted: that
    every class appears is the bundled maps' rule, not a layout's); `DressingGuard` finds nothing wrong with any box
    (inside an allowed place, in no forbidden volume, so dressing only ever occludes); and two runs give the same
    layout. Regression caught: dressing in the vacuum's swept body or a narrow passage, showing depth no entity
    explains; an unregistered class; a layout that depends on more than the map."""
    raw, catalog, layout = laid_out
    assert layout, f"the {catalog} layout emits nothing"
    problems = layout_problems(catalog, [(d.cls, d.phase) for d in layout])
    assert [p for p in problems if not p.endswith("that the layout never emits")] == []
    guard = DressingGuard(raw, catalog)
    bad = [f"{d.cls} at {tuple(round(v, 2) for v in d.pos)} yaw {d.yaw_deg:g}: {guard.problems(d.box, d.cls)}"
           for d in layout if guard.problems(d.box, d.cls)]
    assert bad == []
    assert layout_of(catalog, copy.deepcopy(raw)) == layout
