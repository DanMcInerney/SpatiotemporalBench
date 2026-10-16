"""Fixed scenery that looks like a tracked kind (`scenery.py`): per tracked kind the scenery resembles, the colours it
looks like. Worlds are not needed: the contracts are about the data, the catalogs' and the render's source (parsed,
since it imports Blender's Python)."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

from stbench.catalog import CATALOGS
from stbench.scenery import DRESSING, WAREHOUSE_STOCK_LOOKALIKES, Scenery, lookalike_problems, tracked_colours

FURNITURE = Path(__file__).resolve().parents[1] / "render" / "furniture_warehouse.py"


def _carton_colours() -> set[str]:
    """The colour names of the render's rack cartons, read from `render/furniture_warehouse.py` without importing it:
    the palette key `CARTON` ("beige*0.8") and the string constants of the alternate shade chosen for `mat` in `_stack`
    (the stock's other layers), each cut at its `*` brightness factor."""
    tree = ast.parse(FURNITURE.read_text(encoding="utf-8"))
    carton = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                  and any(getattr(t, "id", None) == "CARTON" for t in n.targets))
    keys = {ast.literal_eval(carton)}
    stack = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_stack")
    mats = [n.value for n in ast.walk(stack) if isinstance(n, ast.Assign)
            and any(getattr(t, "id", None) == "mat" for t in n.targets) and isinstance(n.value, ast.IfExp)]
    assert mats, "_stack no longer picks the carton's shade with `mat = A if ... else B`: update this reader"
    keys |= {n.value for m in mats for n in ast.walk(m) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    return {key.split("*")[0] for key in keys}


def test_the_declared_look_alike_colours_include_the_colour_the_render_draws_the_rack_cartons_in():
    """Contract (VISION rule 10: what a question is about is what the pictures show): the scenery declaration describes
    the render, so the rack cartons' colour, read from `render/furniture_warehouse.py` (`CARTON` and the alternate shade
    of the layers beside it), is one of the colours the `rack_stock` class declares for the box and for the pallet.
    Regression caught: the render's cartons recoloured (to white, say, which a tracked box has) without the
    declaration, so a tracked kind looks like scenery, or a declaration that never matched the render."""
    colours = _carton_colours()
    assert "beige" in colours  # the reader finds the render's cartons
    declared = dict(DRESSING["rack_stock"].resembles)
    for kind in ("box", "pallet"):
        assert colours <= set(declared[kind]), (kind, colours)


def test_a_look_alike_a_tracked_thing_can_be_or_of_a_kind_the_catalog_lacks_is_reported(monkeypatch):
    """Contract: `lookalike_problems` is the check `scenery` offers a catalog author, and `tracked_colours` is what it
    compares against: a kind's colours plus, for the delivery kind alone, the delivery colours. The real entry has no
    problem. With a brown box added to the warehouse catalog (the Release 0.2 regression), a beige delivery colour (the
    delivery kind is the pallet, so it clashes with the pallet's beige but not the box's), an entry with a kind the
    catalog lacks (a mug) and one with no colour, it reports exactly those. Regression caught: a check that compares
    only a kind's own colours (so a delivered beige pallet passes), one that compares a delivery colour to every kind,
    one that accepts a kind-colour clash, or one that crashes on an unknown kind."""
    real = Scenery("warehouse", WAREHOUSE_STOCK_LOOKALIKES)
    assert lookalike_problems(real) == []
    warehouse = CATALOGS["warehouse"]
    box = warehouse.kinds["box"]
    assert tracked_colours("warehouse", "box") == {"white", "blue", "red", "green"}  # no brown, no beige
    monkeypatch.setitem(warehouse.kinds, "box", dataclasses.replace(box, colors=(*box.colors, "brown")))
    assert lookalike_problems(real) == ["(box, brown) is a tracked thing of the warehouse catalog, not scenery"]
    monkeypatch.setitem(warehouse.kinds, "box", box)
    monkeypatch.setitem(CATALOGS, "warehouse", dataclasses.replace(warehouse, delivery_colors=("white", "beige")))
    assert {"white", "beige"} <= tracked_colours("warehouse", "pallet")
    assert "beige" not in tracked_colours("warehouse", "box")
    assert lookalike_problems(real) == ["(pallet, beige) is a tracked thing of the warehouse catalog, not scenery"]
    bad = dataclasses.replace(real, lookalikes=(("mug", ("beige",)), ("tote", ())))
    problems = lookalike_problems(bad)
    assert len(problems) == 2
    assert "'mug' is not a kind" in problems[0] and "'tote' has no look-alike colour" in problems[1]
