"""The generic validity of every held-out map present (M4-integration call 8, D3). A held-out layout's map never enters
the repository, so these tests find the maps of `map_files.HELD_OUT` among `map_files.private_maps()` (the folders
`STBENCH_MAPS` names, as the environment gives it: no test sets it) and skip, naming why, when none is there. They
state no layout fact: only what any map the generator, the dressing and the renderer accept must be, and that its
catalog's M4 job runs on it. The sibling modules `test_heldout_dressing.py`, `test_heldout_names.py` and
`test_heldout_blender.py` check the dressing, the names and the Blender scene the same way."""

from __future__ import annotations

import math

import pytest

from dressing_kit import render_module
from stbench import cli, job_registry, map_files
from stbench.case import read_index
from stbench.episode import write_script
from stbench.generate import generate_script
from stbench.mapspec import check_map, load_map

# the job each catalog's held-out layouts run (M4-integration call 1: one robot a layout)
JOBS = {"home": "vacuum/2", "warehouse": "amr/1", "residential": "drone/1"}
SEED = 101


def held_out_present() -> list[str]:
    return [name for name in map_files.private_maps() if name in map_files.HELD_OUT]


def maps_present() -> list:
    """One pytest parameter a held-out map present; a single skipped one, naming why, when there is none."""
    names = held_out_present()
    if names:
        return [pytest.param(name, id=name) for name in names]
    return [pytest.param(None, id="none", marks=pytest.mark.skip(
        reason=f"no held-out map is present: {map_files.MAPS_ENV} names none of {', '.join(map_files.HELD_OUT)}"))]


@pytest.fixture(scope="module", params=maps_present())
def held_out(request):
    """The loaded held-out map, once a module."""
    return load_map(request.param)


@pytest.fixture(scope="module")
def one_task_script(held_out):
    """A one-task script of the map's catalog's job, once a module (a map whose job is not registered yet skips)."""
    job_id = JOBS.get(held_out.catalog)
    if job_id is None:
        pytest.skip(f"the {held_out.catalog} catalog has no M4 job named in this module")
    try:
        job = job_registry.get(job_id)
    except KeyError:
        pytest.skip(f"the job {job_id} of the {held_out.catalog} catalog is not registered yet")
    return generate_script(held_out.name, SEED, job.robot, job_id, 1)


def test_every_held_out_map_is_valid_and_rectilinear(held_out):
    """Contract (call 4, call 8): `check_map` reports nothing; every region is a 4-vertex axis-aligned rectangle, every
    wall is axis-aligned and every blocking fixed or light piece stands at a quarter turn (what the dressing's boxes,
    the clutter and the generator need: `dressing_boxes.rooms` and `square`); every region's kind is in the vocabulary
    of the map's catalog (`look.kinds`), so the render has a floor, a paint and a light for it. Regression caught: a
    held-out map that the generator or the render refuses only after its scripts exist, or that is dressed and lit by
    a stand-in."""
    m = held_out
    assert check_map(m) == []
    for r in m.regions:
        xs, ys = sorted({round(p[0], 9) for p in r.polygon}), sorted({round(p[1], 9) for p in r.polygon})
        assert len(r.polygon) == 4 and len(xs) == 2 and len(ys) == 2, f"{r.id} is not an axis-aligned rectangle"
    assert all(w.a[0] == w.b[0] or w.a[1] == w.b[1] for w in m.walls)
    turned = [f.id for f in m.furniture if f.blocks and abs(math.sin(2 * f.yaw)) > 1e-6]
    assert turned == []
    assert {r.kind for r in m.regions} <= render_module("look").kinds(m.catalog)


def test_every_held_out_map_runs_its_job(held_out, one_task_script, tmp_path):
    """Contract (call 9): a one-task script of the map's catalog's job (home `vacuum/2`, warehouse `amr/1`,
    residential `drone/1`) written by `write_script` passes `stbench validate` (the map is found through the variable,
    its SHA-256 checked, the leak scan run with its name), and its index entry says `heldout`. Regression caught: a
    job that cannot run on a held-out map, or a held-out script indexed `dev`."""
    paths = write_script(tmp_path, one_task_script)
    assert cli.main(["validate", str(paths.root)]) == 0
    assert read_index(tmp_path)[paths.id]["split"] == "heldout"
