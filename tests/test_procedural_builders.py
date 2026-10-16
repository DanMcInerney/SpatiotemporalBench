"""The warehouse's and the household's procedural builders, checked without Blender (`render/` is not a package, so it
is put on the path; the builders import `bpy` and are only read as source here). What needs Blender (extents, colour
shares, determinism of the meshes) is in `test_procedural_detail_blender.py`."""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "render"))  # its modules import each other by bare name

import wobble  # noqa: E402

RENDER = Path(__file__).resolve().parents[1] / "render"
# the modules that build the detailed procedural kinds of the warehouse and the household, and the kind each top-level
# builder is for: (module, function) per kind
BUILDER_OF = {"pallet": ("procedural_pallet.py", "pallet"), "tote": ("procedural_goods.py", "tote"),
              "drum": ("procedural_goods.py", "drum"), "sack": ("procedural_goods.py", "sack"),
              "bin": ("procedural_goods.py", "bin_"), "pallet_jack": ("procedural_jack.py", "pallet_jack"),
              "roll_cage": ("procedural_equipment.py", "roll_cage"),
              "tote_robot": ("procedural_equipment.py", "tote_robot"),
              "forklift": ("procedural_forklift.py", "forklift")}
MODULES = ("procedural_pallet.py", "procedural_goods.py", "procedural_jack.py", "procedural_equipment.py",
           "procedural_forklift.py", "procedural_home.py", "partskit.py", "wobble.py")


def literal(path: Path, name: str):
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"no top-level {name} in {path}")


def test_the_wobble_that_wrinkles_surfaces_is_deterministic_and_bounded():
    """Contract (the dressing's and the scene's determinism rule: the same world in, the same scene out; the film's
    creases, the sack's lumps and the basket's ribs are made of `wobble`): `wave` gives the same value for the same
    arguments, stays within [-1, 1] and does vary; `bump` is 1 at its centre, 0 from its half width on and never
    negative; `smoothstep` runs from 0 to 1 and rises. Regression caught: a wobble drawn from `random` or the clock,
    which makes two builds of one pallet two different meshes, or a bump that goes negative and cuts a hole where it
    should raise a ridge."""
    grid = [(a * 0.37, b * 0.53, c * 0.71) for a in range(-8, 9) for b in range(-8, 9) for c in range(-3, 4)]
    values = [wobble.wave(*p) for p in grid]
    assert values == [wobble.wave(*p) for p in grid]
    assert all(-1.0 <= v <= 1.0 for v in values) and max(values) - min(values) > 1.0
    assert wobble.bump(0.0, 0.02) == 1.0 and wobble.bump(0.02, 0.02) == 0.0 and wobble.bump(-0.05, 0.02) == 0.0
    assert all(0.0 <= wobble.bump(x / 100, 0.03) <= 1.0 for x in range(-10, 11))
    assert wobble.smoothstep(0.0, 1.0, -1.0) == 0.0 and wobble.smoothstep(0.0, 1.0, 2.0) == 1.0
    assert wobble.smoothstep(0.0, 1.0, 0.25) < wobble.smoothstep(0.0, 1.0, 0.75)
    assert math.isclose(wobble.smoothstep(2.0, 0.0, 1.0), 0.5)  # the edges may be given high to low (a hole's soft rim)


def test_no_procedural_builder_draws_on_chance_or_the_clock():
    """Contract (the scene's determinism rule: seed only from the map name or ids, never `random` or the time): the
    modules that build the detailed procedural kinds import neither `random`, `time`, `datetime` nor `secrets`, and
    call no `uuid`. Regression caught: a 'natural-looking' jitter added to a wrap, a sack or a leaf with `random`, which
    would make every library build a different mesh and every scene digest unstable."""
    forbidden = {"random", "time", "datetime", "secrets", "uuid"}
    for name in MODULES:
        tree = ast.parse((RENDER / name).read_text(encoding="utf-8"))
        imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert imported & forbidden == set(), name


def test_each_builder_reads_exactly_the_slots_its_kind_declares():
    """Contract (plan section 3.3: the manifest's materials are the builder's slots): a warehouse builder that takes its
    slot dict reads the names `procedural.SLOTS` declares for its kind and no others, so `slot["wheel"]` for
    `slot["wheels"]` is found here and not as a KeyError in the middle of a library build, and a declared slot that the
    builder forgot (a drum with no rings) is found as well. The scan sees real accesses (it is not empty). Regression
    caught: a renamed, mistyped or dropped slot in one of the split builder modules, which only a Blender run
    noticed."""
    slots = literal(RENDER / "procedural.py", "SLOTS")
    for kind, (module, function) in BUILDER_OF.items():
        tree = ast.parse((RENDER / module).read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        asked = {n.slice.value for n in ast.walk(fn) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                 and n.value.id == "slot" and isinstance(n.slice, ast.Constant)}
        assert asked, kind
        assert asked == set(slots[kind]), (kind, sorted(asked ^ set(slots[kind])))
