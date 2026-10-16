"""The record layer's output in a second process (`determinism_kit.py`: nothing is recorded, the two are compared).

`outputs` digests the record writers on a fixture: the bytes of a fixture timeline, a `case.json` (its start clock
drawn from a fixed case id), a script's and a case's index entry, the files of a written script (`robot.json`,
`camera.json`, `pose.jsonl`, `timeline.json`), and the oracle's output (the detection rows and key maps
`perception.build` makes of a fixed script: each keyframe's key draw, shuffle and fields). The fixture is built on a
frozen test robot, so tuning the real vacuum (an `IspSpec` value, a phase name) moves none
of them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

from determinism_kit import digest
from stbench import case as case_mod
from stbench import perception, robots
from stbench import timeline as T
from stbench.camera import Camera
from stbench.case import ScriptPaths, write_case
from stbench.episode import pose_rows, write_script
from stbench.robots import Body, CameraSpec, DepthSpec, FramePolicy, IspSpec, Robot, StatusSpec, Tool
from synth import HOME, Synth

CASE_ID = "0123456789abcdef"  # the id the fixture's case claims, so its start clock (drawn from the id) is fixed
FROZEN = Robot(  # the fixture's vacuum: every number written out, so a tune of the real one moves nothing here
    id="vacuum", name="robot vacuum", flies=False,
    body=Body("disc", length_m=0.34, width_m=0.34, height_m=0.09, climb_m=0.02, clearance_m=0.02,
              under_margin_m=0.01),
    speeds_mps={"cleaning": (0.28, 0.31), "transit": (0.30, 0.35), "undocking": (0.10, 0.15),
                "docking": (0.10, 0.15)},
    turn_dps=45.0,
    camera=CameraSpec(
        640, 480, 80.0, mount_m=(0.13, 0.0, 0.08), pitch_deg=5.0, pitch_min_deg=5.0, pitch_max_deg=5.0,
        isp=IspSpec(vignette=0.3, noise_a=6e-4, noise_b=1e-5, meter_sigma=0.35, target=0.18,
                    exposure_ev=(-8.0, 2.0), gain_ev=6.0, wb_clamp=0.1, sharpen=0.25, jpeg_quality=92)),
    depth=DepthSpec("mm", near_m=0.01, far_m=4.0),
    status=StatusSpec(
        phases=("docked", "undocking", "cleaning", "transit", "docking", "paused"),
        keys={"battery_pct": "int", "brush": "bool"},
        start={"phase": "docked", "battery_pct": (90, 100), "brush": False}),
    frames=FramePolicy(move_m=1.0, turn_deg=45.0, still_s=5.0, min_dt_s=1.0),
    tools=(Tool("brush", "disc", size_m=(0.25, 0.25), offset_m=(0.0, 0.0)),),
    sources=("a test fixture, no citation",),
)


def fixture_script(dataset: Path) -> tuple[ScriptPaths, T.Timeline]:
    """Two tasks in the kitchen, a decision point in the first, written as a script by the frozen vacuum (the caller
    holds it in `robots.ROBOTS`)."""
    tl = Synth(tasks=[[(HOME.rooms[0], 4)], [(HOME.rooms[0], 3)]], decisions=[(2, "before_doorway")]).timeline()
    return write_script(dataset, tl), tl


def oracle_digest() -> str:
    """The digest of the oracle's output: the detection rows and key maps (`perception.build`) of a fixed script on the
    frozen vacuum: six keyframes of the kitchen, a mug seen in all of them, a book in the last four and a plant under
    the seen bound in none. Each keyframe's keys and shuffle are drawn from its own generator, so the digest holds the
    draw (the inputs it is seeded from, the 8-hex keys, the order) and every field a detection carries, its clock
    included. The pixel counts sit far from any kind's seen bound, so a seen-bound tune moves nothing."""
    with mock.patch.dict(robots.ROBOTS, {"vacuum": FROZEN}):
        synth = Synth(tasks=[[(HOME.rooms[0], 6)]])
        synth.item("E01", "mug", {"color": "red"}, [(0, None, (2.0, 1.0, 0.9))])
        synth.item("E02", "book", {"color": "blue"}, [(0, None, (3.0, 1.5, 0.9))])
        synth.item("E03", "plant", {"color": "green"}, [(0, None, (1.0, 0.5, 0.0))])
        tl = synth.timeline()
        boxes = {"E01": [100, 100, 140, 140], "E02": [300, 200, 360, 250], "E03": [20, 20, 25, 25]}
        sees = {"E01": (range(6), 5000), "E02": (range(2, 6), 5000), "E03": (range(6), 5)}
        vis = [{"k": k, "entities": [{"id": e, "px": px, "bbox": boxes[e]} for e, (ks, px) in sees.items() if k in ks]}
               for k in range(6)]
        rows, ids = perception.build(tl, vis, pose_rows(tl), Camera.load(robots.camera_doc(FROZEN)))
    return digest({"oracle": rows, "ids": ids})


def outputs(dataset: Path) -> dict[str, str]:
    """The digest of each record writer's output on the fixture, with the random script id left out (it names nothing)
    and the case's id fixed, and of the oracle's."""
    with mock.patch.dict(robots.ROBOTS, {"vacuum": FROZEN}), mock.patch.object(
            case_mod, "claim_id", lambda dataset, entry: CASE_ID):
        paths, tl = fixture_script(dataset)
        case = write_case(dataset, paths.id, 1, 1, "short")
        record = json.loads(case.record.read_text(encoding="utf-8"))
        entries = case_mod.read_index(dataset)
        files = {name: path.read_text(encoding="utf-8")
                 for name, path in (("timeline.json", paths.timeline), ("robot.json", paths.robot),
                                    ("camera.json", paths.camera), ("pose.jsonl", paths.pose))}
        return {
            "timeline": digest(T.dumps(tl)),
            "script": digest(files),
            "case": digest({k: v for k, v in record.items() if k not in ("case", "script")}),
            "index_script": digest(entries[paths.id]),
            "index_case": digest({k: v for k, v in entries[case.id].items() if k != "script"}),
            "oracle": oracle_digest(),
        }


def test_record_outputs_are_deterministic(tmp_path):
    """Contract: a fixture timeline's bytes, a `case.json` (start clock included, from a fixed id), a written script, a
    script's and a case's index entry, and the oracle's detection rows on a fixed script are the same in a second
    process (another `PYTHONHASHSEED`), and the six differ. Regression caught: a record that depends on dict or set
    order or on the process (a clock drawn another way, a detection key drawn from other inputs)."""
    here = outputs(tmp_path / "here")
    out = subprocess.run([sys.executable, __file__, str(tmp_path / "there")], capture_output=True, text=True,
                         check=True, env={**os.environ, "PYTHONHASHSEED": "12345"}).stdout
    assert json.loads(out) == here
    assert len(here) == 6 and len(set(here.values())) == 6  # six different outputs, none an empty digest


if __name__ == "__main__":  # the second process of the test above
    print(json.dumps(outputs(Path(sys.argv[1]))))
