"""Which builders the household's dressing has (`render/dressing_build_home.py`), read
without Blender: the module's `BUILDERS` table is a literal, so its keys and the functions it names are read from the
source. How the builders draw is `test_dressing_build_blender.py`'s; which classes there are is the registry's
(`stbench.scenery.DRESSING`) and which pieces a map has is the layout's. Free."""

from __future__ import annotations

import ast
from functools import cache
from pathlib import Path

from dressing_kit import RENDER, layout_of, raw_map
from stbench.scenery import dressing_of

TABLE = RENDER / "dressing_build_home.py"


@cache
def table() -> dict[str, tuple[str, str]]:
    """`BUILDERS` as written: recipe name -> (module the builder is imported from, its function), the module resolved
    through the table's own `import dressing_build_x as alias` lines."""
    tree = ast.parse(TABLE.read_text(encoding="utf-8"))
    modules = {a.asname: a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names}
    (assign,) = [n for n in tree.body if isinstance(n, ast.Assign) and [t.id for t in n.targets] == ["BUILDERS"]]
    return {k.value: (modules[v.value.id], v.attr) for k, v in zip(assign.value.keys, assign.value.values, strict=True)}


def test_every_class_of_the_household_registry_has_one_builder_and_no_builder_is_for_another_class():
    """Contract (the dispatcher `render/dressing.py` looks a recipe's builder up by name in `BUILDERS`, and a missing
    asset's stand-in by its class name): the table's keys are exactly the registry's phase 1 classes of the home
    catalog, the 21 that draw procedurally (the grime's four among them) and the four (`picture_frame`, `wall_clock`,
    `ceiling_lamp`, `ceiling_fan`) whose Poly Haven model has a procedural stand-in. Regression caught: a class added
    to the registry or the layout that nothing draws (the scene would stop at the first such piece), a builder nothing
    calls, or a stand-in missing for an asset class (a scene could not build without the download)."""
    classes = {c.name for c in dressing_of("home", 1)}
    assert len(classes) == 25
    assert set(table()) == classes


def test_every_recipe_the_apartments_layout_emits_names_a_builder_or_a_class_with_a_stand_in():
    """Contract (`dressing.draw`): a `proc:<name>` recipe needs `BUILDERS[name]` and an `asset:<id>` recipe needs
    `BUILDERS[class]` for the day the library lacks the model. Checked against what the layout makes of apartment_a (245
    pieces, 21 procedural classes and 4 asset ones). Regression caught: a layout change that renames a recipe without
    its builder."""
    pieces = layout_of("home", raw_map("apartment_a"))
    assert {d.recipe.split(":")[0] for d in pieces} == {"proc", "asset"}
    for d in pieces:
        kind, _, rest = d.recipe.partition(":")
        assert (rest.split(":")[0] if kind == "proc" else d.cls) in table(), d.recipe


def test_each_builder_is_the_function_of_its_own_name_and_exists_in_the_module_it_is_imported_from():
    """Contract (one owner per concept, and a table a reader can trust): `BUILDERS["outlet"]` is the function `outlet`,
    defined in the module the table imports it from. Regression caught: a key wired to the wrong function (a switch
    drawn where an outlet is asked for) or to one that was renamed away."""
    defined = {}
    for module in {m for m, _ in table().values()}:
        tree = ast.parse((RENDER / f"{module}.py").read_text(encoding="utf-8"))
        defined[module] = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name, (module, function) in table().items():
        assert function == name and function in defined[module], (name, module, function)
    assert all(Path(RENDER / f"{m}.py").exists() for m, _ in table().values())
