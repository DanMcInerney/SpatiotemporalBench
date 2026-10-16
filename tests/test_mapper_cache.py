"""`table.Mapper`'s memory of the steps it has read: a step any mapper of the process has read is not worked out
again, and only a step that is the same in everything its cells depend on is found."""

from __future__ import annotations

import pytest

import zone_kit as zk
from builders import keyframe
from space_kit import box, render_depth
from stbench import table
from stbench.render_encode import write_png16
from stbench.robots import ROBOTS
from stbench.table import Mapper
from synth import camera

VACUUM = ROBOTS["vacuum"]
CAM = camera()
FRAME = keyframe("vacuum", 0, 0.0, (-0.17, 0.0, 0.0), 0.0, task=0, phase="docked")
SCENE = [box(1.43, 1.73, -0.15, 0.15, 0.0, 0.3)]


@pytest.fixture(autouse=True)
def empty_cache():
    """The cache is the process's: each test starts with it empty and leaves it so."""
    table.SEEN.clear()
    yield
    table.SEEN.clear()


@pytest.fixture
def reads(monkeypatch):
    """How many times a mapper worked a step out (not found in `SEEN`)."""
    count = {"n": 0}
    work = Mapper._read

    def counted(self, *args):
        count["n"] += 1
        return work(self, *args)

    monkeypatch.setattr(Mapper, "_read", counted)
    return count


def image(folder, boxes=SCENE, name="d.png"):
    folder.mkdir(parents=True, exist_ok=True)
    write_png16(folder / name, render_depth(CAM, FRAME.pos, FRAME.rot, boxes))


def step(floor_z=0.0, pos=None):
    return {"k": 0, "depth": "d.png", "pose": {"pos": [-0.17, 0.0, floor_z], "rot": [1.0, 0.0, 0.0, 0.0]},
            "camera": {"pos": list(FRAME.pos if pos is None else pos), "rot": list(FRAME.rot)}}


def test_a_step_one_mapper_read_is_found_by_the_next(tmp_path, reads):
    """Contract: two mappers of one configuration given a step with the same image (bytes, in different folders), camera
    pose and floor work it out once; the second gets what the first did, and the cells are the obstacle's four occupied
    cells in front of the floor (a hand case: a 0.3 m box with its face at x = 1.43). Regression caught: every
    baseline over a case reading each depth image again (the cost of `stbench check`), or a cache that returns another
    step's cells."""
    image(tmp_path / "a")
    image(tmp_path / "b")
    first = Mapper(VACUUM, CAM, tmp_path / "a").observe(step())
    second = Mapper(VACUUM, CAM, tmp_path / "b").observe(step())
    assert reads["n"] == 1 and second is first
    assert {c for c, s in first.items() if s == "occupied"} == {(14, v) for v in range(-2, 2)}


def test_a_step_is_found_only_when_everything_its_cells_depend_on_is_the_same(tmp_path, reads):
    """Contract: the image's bytes, the camera's position, the floor the band is measured from and the mapper's
    configuration (its zones) each decide the cells; a step that differs from a read one in any of them is read again.
    Hand case: the box moved back 0.2 m (another image at the same path), the camera 0.1 m higher, the floor 0.1 m lower
    (every floor point is then over the band: no free cell), and a mapper with the job's zones each read once more,
    4 reads after the first, and the floor 0.1 m lower observes fewer cells than the floor at 0. Regression caught:
    cells served from a step that is not this one (a key without the floor, the pose or the configuration)."""
    image(tmp_path)
    mapper = Mapper(VACUUM, CAM, tmp_path)
    base = mapper.observe(step())
    assert reads["n"] == 1 and mapper.observe(step()) is base
    image(tmp_path, [box(1.63, 1.93, -0.15, 0.15, 0.0, 0.3)])
    moved = mapper.observe(step())
    assert reads["n"] == 2 and moved is not base and {c for c, s in moved.items() if s == "occupied"} != {
        c for c, s in base.items() if s == "occupied"}
    image(tmp_path)
    assert mapper.observe(step()) is base and reads["n"] == 2   # the first image's bytes again: found
    higher = mapper.observe(step(pos=(FRAME.pos[0], FRAME.pos[1], FRAME.pos[2] + 0.1)))
    lower = mapper.observe(step(floor_z=-0.1))
    zoned = Mapper(VACUUM, CAM, tmp_path, zk.MAP, [zk.DROP, zk.BAY], [zk.LANES]).observe(step())
    assert reads["n"] == 5 and higher is not base and zoned is not base
    assert len(lower) < len(base) and not any(s == "free" for s in lower.values())
