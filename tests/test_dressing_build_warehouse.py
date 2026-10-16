"""The warehouse's dressing builders, the parts that need no Blender (`render/dressing_build_warehouse.py`,
`render/dressing_code39.py`): that every recipe the layout emits has a builder, and the barcode's table. What the
builders draw is `test_dressing_build_warehouse_blender.py`'s. `render/` is not a package and `bpy` is not importable
here, so the builders' table is read from the source and the barcode module (plain Python) by name."""

from __future__ import annotations

import ast

import pytest

from dressing_kit import RENDER, layout_of, raw_map, render_module

CODE39 = render_module("dressing_code39")


@pytest.fixture(scope="module")
def layout() -> list:
    return layout_of("warehouse", raw_map("warehouse_a"))


def builder_names() -> set[str]:
    """The keys of the dict literal `BUILDERS` in `dressing_build_warehouse.py`, read from the source."""
    tree = ast.parse((RENDER / "dressing_build_warehouse.py").read_text(encoding="utf-8"))
    table = next(n.value for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "BUILDERS")
    return {k.value for k in table.keys}


def called(d) -> str:
    """The builder `dressing.draw` calls for a Dressing: the recipe's `proc:` name, or the class's own name when the
    recipe is an asset (the stand-in used while the library lacks the model)."""
    kind, _, rest = d.recipe.partition(":")
    return rest.split(":")[0] if kind == "proc" else d.cls


def test_every_recipe_the_layout_emits_has_a_builder_and_every_builder_is_used(layout):
    """Contract (the dispatcher's, `dressing.draw`: `proc:<name>` calls `BUILDERS[name]`, an `asset:<id>` the library
    lacks calls `BUILDERS[<class>]`): each Dressing of `warehouse_a`'s layout resolves to a builder of
    `dressing_build_warehouse.BUILDERS`, and each builder is called by some Dressing, so a name that nothing emits is a
    typo or dead code. Regression caught: a layout recipe with no builder (the scene build raises after the structure
    is built, in the middle of a render), a builder misspelled in the table, and a new asset class whose stand-in was
    forgotten."""
    names = builder_names()
    assert not {(d.cls, d.recipe) for d in layout if called(d) not in names}
    assert not names - {called(d) for d in layout}


def test_the_code39_table_is_a_real_code39_table():
    """Contract (the beam labels' barcode is Code 39, ISO/IEC 16388: the start and stop `*`, A-Z, 0-9 and "-"): each
    character is nine elements, bar and space in turn from a bar, three of them wide, two of those wide bars and one a
    wide space; no two characters have one pattern; `0` is `nnnwwnwnn` and the star `nwnnwnwnn` (the standard's table).
    Regression caught: a typo in the 38 patterns of the table (a label whose bars decode as another code, or not at
    all)."""
    table = CODE39.CODE39
    assert len(table) == 38 and len(set(table.values())) == 38
    for ch, pattern in table.items():
        assert len(pattern) == 9 and set(pattern) == {"n", "w"}, ch
        wide = [i for i, e in enumerate(pattern) if e == "w"]
        assert len(wide) == 3 and sum(i % 2 == 0 for i in wide) == 2, ch
    assert table["0"] == "nnnwwnwnn" and table["*"] == "nwnnwnwnn" and table["-"] == "nwnnnnwnw"


@pytest.mark.parametrize("code", ["A-01-02", "C-08-03", "1"])
def test_a_barcode_is_five_bars_a_character_framed_by_stars(code):
    """Contract (Code 39 as above): the symbol of a string is the string between two stars, each character 15 modules
    wide (six narrow elements and three wide, a wide being 3 modules) and one module of space between characters, so n
    characters end at module 16 (n + 2) - 1, with five bars each; the first bar is one module wide and starts at 0 (the
    star starts with a narrow bar). Regression caught: bars drawn without the stars, with a gap missing between
    characters (a label that does not scan) or the symbol drawn from its other end."""
    bars = CODE39.code39_bars(code)
    n = len(code) + 2
    assert len(bars) == 5 * n
    assert bars[0] == (0.0, 1.0) and bars[-1][0] + bars[-1][1] == 16 * n - 1
    assert [w for _, w in bars[:5]] == [1.0, 1.0, 3.0, 3.0, 1.0]  # the star: n w n n w n w n n, bars at 0, 2, 4, 6, 8
