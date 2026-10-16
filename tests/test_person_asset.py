"""The people's bodies as assets: the resident (lane M2-P2), one barefoot, clothed, free-licensed person, an ordinary
mover model made by a recipe instead of downloaded, and its placeholder; and the worker (lane M3-people, step 0), the
second, shod body of the warehouse's pickers and the downtown's pedestrians, made by a second recipe. Free: no Blender,
no network, nothing downloaded; the synthetic manifests come from `asset_synth`."""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

from asset_synth import FakeNet, asset, raw_manifest
from stbench import asset_fetch as F
from stbench import asset_manifest as A
from stbench import asset_render as AR
from stbench.asset_coverage import PEOPLE_KINDS
from stbench.catalog import CATALOGS

REPO = Path(__file__).resolve().parents[1]
FOOTWEAR = re.compile(r"shoe|sock|slipper|sandal|boot|sneaker|trainer|flip.?flop|footwear", re.IGNORECASE)
GLB = b"a body a recipe made"
RECIPE = "render/people_make.py"
WORKER_RECIPE = "render/people_make_worker.py"
PLACEHOLDER = "PLACEHOLDER"  # what the worker's manifest row says of its box until its recipe has run


def _licensed(part: dict) -> bool:
    return part["licence"] == "CC0-1.0" or (part["licence"] == "CC-BY-4.0" and all(
        part.get(k) for k in ("title", "author", "changes")))


def test_the_person_asset_is_licensed_and_barefoot():
    """Contract (VISION rule 11, plan section 5.1 P): the `resident` is a `person` model (the home's body), a mover,
    CC0, made by a generated source that has no host and names its recipe and every input (a name, a kind, a licence and
    a url); every part it is made of is CC0 or CC BY with its title, author and changes;
    no part's name or kind is a shoe, sock or slipper (nothing on its feet can be read as the tracked `shoes`, the
    plainest look-alike at floor level); its box is a standing adult's. Regression caught: a non-free part, a resident
    in shoes that the vacuum's camera reads as the shoes it is asked about, or a generated source that names no recipe
    or no inputs."""
    m = A.packaged()
    resident = m.asset("resident")
    source = m.sources[resident.source]
    assert (resident.id, resident.role, resident.mobility) == ("resident", "model", "mover")
    assert resident.licence == "CC0-1.0"
    assert resident.tint["mode"] == "none"  # drawn in its own materials: the mover path's colour is ignored
    assert resident.parts and all(_licensed(p) for p in resident.parts)
    assert [p for p in resident.parts if FOOTWEAR.search(p["name"]) or FOOTWEAR.search(p["kind"])] == []
    assert FOOTWEAR.search("toigo_ankle_boots_male") and FOOTWEAR.search("shoes01")  # the guard is not vacuous
    length, width, height = resident.native_bbox_m
    assert 1.5 <= height <= 2.0 and 0.2 <= length <= 0.7 and 0.3 <= width <= 0.8
    assert (source.kind, source.hosts, source.licence) == ("generated", (), "CC0-1.0")
    generator = source.generator
    assert generator["recipe"] == RECIPE and (REPO / RECIPE).is_file()
    assert all(i["url"].startswith("https://") and i["licence"] for i in generator["inputs"])
    assert {i["kind"] for i in generator["inputs"]} == {"tool", "data"}


def test_the_placeholder_builds_a_person(monkeypatch):
    """Contract (plan section 5.1 P): `items.BUILDERS` has a `person`, a plain box of the entity's size in a neutral
    colour, whatever colour `build_scene` passes (a person has no colour attribute), so a placeholder render of a home
    script with a resident builds. `items` imports `bpy` through `shapes`, so a stand-in for `shapes` records the box.
    Regression caught: a placeholder render that raises on a resident ('no placeholder geometry for mover kind
    person'), or a person drawn in the colour of a thing it is not."""
    drawn = []
    stand_in = types.ModuleType("shapes")
    for name in ("Part", "ball", "cyl", "ring", "wedge"):
        setattr(stand_in, name, lambda *a, **k: ("other", a))
    stand_in.box = lambda *a: drawn.append(a) or ("box", a)
    monkeypatch.setitem(sys.modules, "shapes", stand_in)
    spec = importlib.util.spec_from_file_location("items_under_test", REPO / "render" / "items.py")
    items = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(items)
    assert "person" in items.BUILDERS
    parts = items.BUILDERS["person"]((0.30, 0.50, 1.70), "red")
    assert parts == [("box", (-0.15, 0.15, -0.25, 0.25, 0, 1.70, "grey"))] and len(drawn) == 1


# -- the generated source: policed like a downloaded one ---------------------------------------------------------------
def _part(name="shirt", **extra) -> dict:
    return {"name": name, "kind": "clothes", "licence": "CC0-1.0", "source": "a pack", **extra}


def with_resident(raw: dict) -> dict:
    """The synthetic manifest plus a generated source and a `person` model made by it (tiny bytes, no real recipe)."""
    raw["sources"]["gen"] = {
        "name": "Body Makers", "kind": "generated", "licence": "CC0-1.0", "licence_url": "https://gen.example.org/l",
        "hosts": [], "generator": {"recipe": RECIPE, "inputs": [
            {"name": "plugin", "kind": "tool", "licence": "GPL-3.0-or-later", "url": "https://gen.example.org/plugin.zip"},
            {"name": "pack", "kind": "data", "licence": "CC0-1.0", "url": "https://gen.example.org/pack.zip"}]}}
    raw["assets"].append({
        "id": "resident", "source": "gen", "role": "model", "kind": "person", "mobility": "mover",
        "title": "A person", "page": "https://gen.example.org/p", "licence": "CC0-1.0",
        "files": [{"path": "resident.glb"}], "entry": "resident.glb",
        "format": "gltf", "transform": {"unit_scale": 1.0, "up": "Z", "rotate_deg": [0, 0, 0]}, "pivot": "base_centre",
        "native_bbox_m": [0.3, 0.5, 1.7], "fit": {"mode": "per_axis", "max_anisotropy": 0.05},
        "tint": {"mode": "none", "materials": []}, "compose": None, "parts": [_part()]})
    return raw


def problems(edit=None, kinds=None) -> list[str]:
    raw = with_resident(copy.deepcopy(raw_manifest()))
    if edit:
        edit(raw)
    return A.validate(A.parse(raw), *(kinds or (None,)))


def _gen(raw): return raw["sources"]["gen"]
def _resident(raw): return asset(raw, "resident")
def _drop(key): return lambda raw: _gen(raw)["generator"].pop(key)
def _set(path, value): return lambda raw: _gen(raw)["generator"]["inputs"][path[0]].update({path[1]: value})
def _mutate(edit_gen=None, edit_res=None):
    return lambda raw: ((edit_gen or (lambda g: None))(_gen(raw)), (edit_res or (lambda a: None))(_resident(raw)))


@pytest.mark.parametrize("edit, wanted", [
    (_mutate(lambda g: g.update(hosts=["gen.example.org"])), "has no hosts"),
    (_mutate(lambda g: g.pop("generator")), "needs one"),
    (_mutate(lambda g: g.update(kind="hosted")), "belongs to a generated source"),
    (_drop("recipe"), "generator.recipe"),
    (_drop("inputs"), "generator.inputs"),
    (_set((1, "url"), "http://gen.example.org/pack.zip"), "inputs[1].url"),
    (_set((1, "licence"), "CC-BY-NC-4.0"), "inputs[1].licence"),
    (_set((1, "licence"), "GPL-3.0-or-later"), "inputs[1].licence"),  # a datum may not be GPL: only a tool may
    (_set((0, "kind"), "script"), "inputs[0].kind"),
    (_mutate(edit_res=lambda a: a["files"][0].update(url="https://gen.example.org/resident.glb")), "has no url"),
    (_mutate(edit_res=lambda a: a["parts"].append(_part("hat", licence="CC-BY-NC-4.0"))), "hat"),
    (_mutate(edit_res=lambda a: a["parts"].append(_part("hat", licence="CC-BY-4.0", title="Hat"))), "an author"),
    (_mutate(edit_res=lambda a: a["parts"].append({"name": "x"})), "needs a name, a kind and a source"),
])
def test_a_generated_source_is_policed_as_strictly_as_a_downloaded_one(edit, wanted):
    """Contract (VISION rule 11, plan section 5.1 P): the baseline (a generated source with a tool and a datum input, a
    person model made by it) validates; a generated source with a host, no generator, no recipe or inputs, an input
    with no https url, a non-free or mislabelled input (a GPL datum), a file with a url, or a part with a non-free
    licence or a CC BY credit missing is reported. Regression caught: a body whose provenance is not named, or whose
    part is not free, passing the manifest check and reaching a render."""
    assert problems() == []
    found = problems(edit)
    assert any(wanted in p for p in found), found


def test_a_generated_file_is_never_downloaded_and_names_its_recipe_when_missing_from_the_cache(tmp_path):
    """Contract (plan section 5.1 P): `asset_fetch` never plans a download for a generated file (it has no host), and
    the library gate's cache check (`asset_render.cache_problems`) names the recipe to run when the file is missing
    from the cache and accepts it once it is there. Regression caught: a fetch that tries to open a url that is not
    there, or a missing body reported as a bare 'missing' with no way to make it."""
    m = A.parse(with_resident(copy.deepcopy(raw_manifest())))
    net = FakeNet()
    resident = m.asset("resident")
    assert [t.asset.id for t in F.plan(m, cache=tmp_path)] != []  # the other assets are still to download
    assert all(t.asset.id != "resident" for t in F.plan(m, ["resident", "mug_a"], cache=tmp_path))
    F.fetch([t for t in F.plan(m, ["resident"], cache=tmp_path)], net)
    assert net.calls == []

    def lines():
        return [line for line in AR.cache_problems(m, tmp_path) if line.startswith("resident ")]

    assert len(lines()) == 1 and "missing" in lines()[0] and RECIPE in lines()[0]
    path = A.file_path(resident, resident.files[0], tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(GLB)
    assert lines() == []


# -- coverage: a person is a model that no catalog lists ---------------------------------------------------------------
KINDS = (("mug", "glass"), (), ())  # the synthetic manifest's carried, light and mover kinds


def test_a_person_kind_may_have_several_bodies_but_never_a_prop_or_a_colours():
    """Contract (the coordinator's M2-P2 call, widened by M3-people step 0: the home's resident and the warehouse's and
    downtown's worker are two models of kind `person`, and a people module names its own by id): a `person` model is
    accepted though no catalog lists it as a mover kind (people are an overlay, `people.py`), alone or beside a second
    body, as a mover made for no colour; a person filed as a carried item or made for a colour is reported, a second
    body too; any other kind that no catalog lists is still reported. Regression caught: the packaged manifest failing
    its own coverage check with two bodies, the allowance quietly accepting a model nobody queries (a teapot), or a
    second body filed as a prop (a person is no clutter or carried thing)."""
    assert problems(kinds=KINDS) == []
    copy_of = lambda raw, **extra: raw["assets"].append({**copy.deepcopy(_resident(raw)), **extra})  # noqa: E731
    teapot = lambda raw: copy_of(raw, id="teapot", kind="teapot")  # noqa: E731
    two = lambda raw: copy_of(raw, id="worker")  # noqa: E731
    carried = lambda raw: _resident(raw).update(mobility="carried")  # noqa: E731
    coloured = lambda raw: _resident(raw).update(for_colours=["red"])  # noqa: E731
    second_carried = lambda raw: copy_of(raw, id="worker", mobility="carried")  # noqa: E731
    second_coloured = lambda raw: copy_of(raw, id="worker", for_colours=["red"])  # noqa: E731
    assert problems(two, KINDS) == []
    assert any("'teapot' is not a queried kind" in p for p in problems(teapot, KINDS))
    for edit in (carried, coloured, second_carried, second_coloured):
        assert any("kind person: needs mover models, each for no colour" in p for p in problems(edit, KINDS)), edit


# -- the worker: the second, shod body (M3-people step 0) --------------------------------------------------------------
def _recipe_parts() -> dict[str, str]:
    """The `PARTS` table of the worker's recipe, read from its source (the recipe imports `bpy`: it is not imported)."""
    tree = ast.parse((REPO / WORKER_RECIPE).read_text(encoding="utf-8"))
    table = next(n.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id == "PARTS")
    return ast.literal_eval(table)


def test_the_worker_is_licensed_and_shod():
    """Contract (VISION rule 11, M3-people step 0): the `worker` is a second `person` model beside the resident, a mover
    for no colour, CC0, made by its own generated source (no host) that names its recipe and the same two inputs as the
    resident's; every part is CC0 or CC BY with its credit; it wears shoes and work clothes, and its parts are exactly
    the ones its recipe asks of the pack. Regression caught: a non-free part, a barefoot picker (the second body's whole
    point), or a manifest listing parts the recipe no longer builds."""
    m = A.packaged()
    worker, resident = m.asset("worker"), m.asset("resident")
    assert (worker.role, worker.kind, worker.mobility, worker.for_colours) == ("model", "person", "mover", ())
    assert worker.licence == "CC0-1.0" and worker.tint["mode"] == "none"
    assert worker.parts and all(_licensed(p) for p in worker.parts)
    assert {p["kind"] for p in worker.parts} >= {"body", "skin", "eyes", "hair", "clothes", "shoes"}
    shoes = [p for p in worker.parts if p["kind"] == "shoes"]
    assert len(shoes) == 1 and FOOTWEAR.search(shoes[0]["name"])  # the resident has none (its own test)
    recipe_parts = {path.split("/")[0] for path in _recipe_parts().values()}
    assert {p["name"] for p in worker.parts} - {"base mesh hm08 and targets"} == recipe_parts
    source, home = m.sources[worker.source], m.sources[resident.source]
    assert worker.source != resident.source
    assert (source.kind, source.hosts, source.licence) == ("generated", (), "CC0-1.0")
    assert source.generator["recipe"] == WORKER_RECIPE and (REPO / WORKER_RECIPE).is_file()
    assert source.generator["inputs"] == home.generator["inputs"]  # one MPFB and one pack make both bodies


def test_the_worker_is_built_and_measured():
    """Contract (M3-people step 0): once `render/people_make_worker.py` has run, the manifest holds its measured box, a
    standing adult's, with no PLACEHOLDER note left in the row. Skipped while the row says PLACEHOLDER (the box is an
    estimate until the recipe has run under the Blender lock). Regression caught: the estimate shipping as the measured
    box, so that a worker is drawn stretched."""
    worker = A.packaged().asset("worker")
    if PLACEHOLDER in worker.changes:
        pytest.skip("the worker's file is not built yet: its box is an estimate (M3-people step 0)")
    length, width, height = worker.native_bbox_m
    assert 1.5 <= height <= 2.0 and 0.2 <= length <= 0.7 and 0.3 <= width <= 0.8


def test_every_people_module_names_a_person_model():
    """Contract (M3-people step 0: a person kind has several bodies, each people module names its own by id): every
    `people_<catalog>` module that exists declares a KIND of the people kinds and a MODEL that is a `person` model of
    the manifest. Regression caught: a module naming a body that does not exist, so that its entities are drawn by
    nothing or the render raises."""
    bodies = {a.id for a in A.packaged().models_of("person")}
    modules = [importlib.import_module(f"stbench.people_{c}") for c in sorted(CATALOGS)
               if importlib.util.find_spec(f"stbench.people_{c}")]
    assert "stbench.people_home" in [mod.__name__ for mod in modules]  # the loop is not vacuous
    for mod in modules:
        assert mod.KIND in PEOPLE_KINDS and mod.MODEL in bodies, mod.__name__
