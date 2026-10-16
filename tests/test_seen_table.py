"""The per-kind seen table (`stbench.seen_table`): its checks, and that every place that builds `Sightings` passes the
kind.

The lookups themselves are tested where they live: test_rules.py (thresholds, sightings, in-view), test_visibility.py
(status), test_perception.py (the oracle channel) and test_table.py (the baseline).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from conftest import apartment
from stbench import seen_table
from stbench.catalog import KINDS

SRC = Path(__file__).resolve().parents[1] / "src" / "stbench"


def test_the_packaged_table_is_valid_and_names_only_kinds_the_compiler_describes():
    """Contract: every entry is a (lo, hi) pair of fractions with 0 <= lo <= hi and a usable hi, for a carried kind of
    the catalog or a light kind of the map (fixed furniture and doors are not queried, so they keep the global bounds).
    Regression caught: a kind misspelt in the table (it would silently keep the global bounds), or a pixel count
    written where a fraction belongs."""
    light = {f.kind for f in apartment().furniture if f.mobility == "light"}
    assert seen_table.problems(seen_table.BOUNDS) == []
    assert set(seen_table.BOUNDS) <= set(KINDS) | light


def test_problems_names_each_malformed_entry():
    """Contract: `problems` flags lo above hi, a negative lo, a hi that is zero or above 1% of the image (a pixel count
    written as a fraction), a bound that is not a number, a pair of the wrong length and a nameless kind; a good entry,
    with lo = 0 (every pixel is a glimpse) or lo = hi, passes. Regression caught: a table with a typo accepted."""
    good = {"mug": (0.0001, 0.0007), "glass": (0.0, 0.0007), "plate": (0.0005, 0.0005)}
    assert seen_table.problems(good) == []
    bad = {"a": (0.001, 0.0005), "b": (-0.1, 0.0007), "c": (0.0, 0.0), "d": (0.0, 215), "e": (0.0, "x"),
           "f": (0.1,), "": (0.0, 0.0007)}
    got = seen_table.problems(bad)
    assert len(got) == 7 and [p.split(":")[0] for p in got] == ["kind '' is not a name", "a", "b", "c", "d", "e", "f"]


def sightings_without_the_kind(source: str) -> list[int]:
    """Lines of `Sightings(...)` and `Sightings.load(...)` calls in the source that pass no `kind_of` (not a third
    positional argument and not the keyword)."""
    out = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        built = (isinstance(f, ast.Name) and f.id == "Sightings") or (
            isinstance(f, ast.Attribute) and f.attr == "load" and isinstance(f.value, ast.Name)
            and f.value.id == "Sightings")
        if built and len(node.args) < 3 and not any(k.arg == "kind_of" for k in node.keywords):
            out.append(node.lineno)
    return out


def test_the_call_scanner_finds_sightings_built_without_the_kind():
    """Contract: the scanner behind the rule below flags the constructor and `Sightings.load` with two arguments and
    accepts a third positional or the `kind_of` keyword. Regression caught: a scanner that misses a form, so the rule
    below passes vacuously."""
    src = ("Sightings(rows, cam)\nSightings.load(path, cam)\nSightings(rows, cam, kinds)\n"
           "Sightings(rows, cam, kind_of=k)\nSightings.load(path, cam, kinds)\nSightings.load(path, cam, kind_of=k)\n"
           "other(rows, cam)\n")
    assert sightings_without_the_kind(src) == [1, 2]


def test_a_table_with_data_needs_every_sightings_in_the_package_to_pass_the_kind():
    """Contract (M2 lane E1): once `seen_table.BOUNDS` holds an entry, every place that builds `Sightings` (in the
    package, and in `tests/synth.py`) passes `kind_of`, or that call classifies by the global bounds and the golds
    ignore the table. Skipped with the identity table, where there is nothing to wire. When this fails after the table
    gets data, wire `kinds_of(timeline)` into the named call. Regression caught: a table of bounds that never reaches
    the golds, with every test of the lookups still green."""
    if not seen_table.BOUNDS:
        pytest.skip("the table is empty, so nothing needs the kind")
    files = [f for f in sorted(SRC.glob("*.py")) if f.name != "sightings.py"] + [Path(__file__).with_name("synth.py")]
    unwired = {f"{f.name}:{n}" for f in files for n in sightings_without_the_kind(f.read_text(encoding="utf-8"))}
    assert unwired == set(), f"Sightings built without kind_of while the seen table has data: {sorted(unwired)}"
