"""Helpers for the tests of the scripted systems: compiled cases to run them on, steps and detections to feed them by
hand, and prior records with their gold rows.

`compiled_case` writes `space_kit.things_run()` as a case of a fixed id (so its draws and clock repeat) and compiles it
with `vacuum/1` as it would be were `path_ahead` asked at `before_doorway` (`JOB`): one `path_ahead` (2.2, from keyframe
21) and two `object_where` questions. The registered job asks no during family in M1, so the during-ask path of every
system is exercised on this one. `play`
runs a system class over a case through `stream.run_case` and keeps the instance, `scores` gives each question's
penalised score against the case's gold. `started` builds a system and sends it `start`, `step`/`det` make the messages
a hand-fed system gets, and `record` makes a prior record (`answer_keys.RECORD_KEYS`) with a gold row.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from unittest import mock

from space_kit import SpaceScript, things_run
from stbench import case as case_mod
from stbench import job_registry, questions, robots, score
from stbench.case import CasePaths
from stbench.geometry import quat_z
from stbench.stream import CaseRun, run_case
from stbench.stream_protocol import System

CID = "0123456789abcdef"
# each draws other things for its end questions; the first two ask two `object_where` questions (one `unknown`,
# one answerable), the third two answerable ones
IDS = (CID, "3333333333333333", "1111111111111111")
ROBOT = robots.ROBOTS["vacuum"]
EVENTS = dict(job_registry.get("vacuum/1").events)   # the registered job's, which `compiled_case` compiles with
JOB = replace(job_registry.get("vacuum/1"), decisions={"before_doorway": ("path_ahead",), "room_done": ()})
START = "2026-03-02T08:00:00"


def compiled_case(dataset: Path, cid: str = CID, script: SpaceScript | None = None) -> CasePaths:
    """`things_run()` (or `script`) as a case with the id `cid`, its questions and gold compiled."""
    with mock.patch.object(case_mod, "claim_id", lambda dataset, entry: cid):
        case = (script or things_run()).write_case(dataset)
    with mock.patch.object(job_registry, "get", lambda job_id: JOB):
        questions.write(case)
    return case


def rows(case: CasePaths) -> tuple[list[dict], list[dict]]:
    """The case's question rows and gold rows, in file order."""
    return tuple([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                 for path in (case.questions, case.gold))


def play(case: CasePaths, cls: type[System], channel: str = "detections", *, stream_from: CasePaths | None = None,
         ask_map: dict[int, int] | None = None, **opts) -> tuple[CaseRun, System]:
    """`run_case` of `case` on `cls(**opts)`: the run and the system instance (for its store or log)."""
    built: list[System] = []

    def make(state, work):
        built.append(cls(state, work, **opts))
        return built[0]

    return run_case(make, case, channel=channel, stream_from=stream_from, ask_map=ask_map), built[0]


def scores(case: CasePaths, run: CaseRun) -> dict[str, float]:
    """The penalised score of each answer of `run` against the case's gold, by qid."""
    return {g["qid"]: score.penalised(run.answers[g["qid"]], g) for g in rows(case)[1]}


# -- messages to a system by hand ---------------------------------------------------------------------------------
def started(cls: type[System], tmp_path: Path, channel: str = "detections", **opts) -> System:
    """`cls` built over a working folder whose `public/camera.json` is the vacuum's, and sent `start`."""
    (tmp_path / "public").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "public" / "camera.json").write_text(json.dumps(robots.camera_doc(ROBOT)), encoding="utf-8")
    system = cls(tmp_path / "state", tmp_path, **opts)
    system.start("0123456789abcdef", robots.robot_doc(ROBOT), channel)
    return system


def step(k: int, x: float, y: float, yaw_deg: float = 0.0, phase: str = "cleaning", **extra) -> dict:
    """A delivered step of the vacuum: its base at (x, y) on the floor heading `yaw_deg`, its camera derived from the
    base, one second a step; `extra` adds keys (`detections`, `identity`, `rgb`, `depth`)."""
    yaw = math.radians(yaw_deg)
    cam_pos, cam_rot = robots.camera_pose(ROBOT, (x, y, 0.0), yaw, math.radians(ROBOT.camera.pitch_deg))
    return {"k": k, "clock": f"2026-03-02T08:00:{k:02d}",
            "pose": {"pos": [x, y, 0.0], "rot": list(quat_z(yaw)), "yaw_deg": yaw_deg},
            "camera": {"pos": list(cam_pos), "rot": list(cam_rot), "pitch_deg": ROBOT.camera.pitch_deg},
            "status": {"phase": phase, "task": 0, "battery_pct": 90, "brush": True}, **extra}


def det(key: str, kind: str, color: str, x: float, y: float, z: float = 0.1, hull=None) -> dict:
    """A detection at the map point (x, y, z), carrying its `center_map` and, when `hull` is given, its `floor_hull`
    (a list of [x, y], as the oracle channel sends of a recognisable detection)."""
    return {"d": key, "kind": kind, "attributes": {"color": color}, "bbox_px": [0, 0, 40, 40],
            "center_cam": [0.0, 0.0, 1.0], "center_map": [x, y, z], "extent_m": [0.2, 0.2, 0.2],
            "floor_hull": hull}


def question(template: str, slots: dict, qid: str = "q1", answer_type: str = "position") -> dict:
    """A question as the structured view gives it."""
    return {"qid": qid, "text": "", "answer_type": answer_type, "template": template, "slots": slots}


def where(kind: str, color: str, event: str, qid: str = "q1") -> dict:
    return question("object_where", {"kind": kind, "attributes": {"color": color}, "event": event}, qid)


def ahead(cap: float = 3.0, qid: str = "q1") -> dict:
    return question("path_ahead", {"cap": cap}, qid, "number")


# -- prior records ------------------------------------------------------------------------------------------------
def at(x: float, y: float) -> dict:
    return {"type": "position", "xy": [x, y]}


def number(value: float) -> dict:
    return {"type": "number", "value": value}


def offset(ahead_m: float, left_m: float) -> dict:
    """An `offset` answer or gold value (metres ahead of the camera, then to its left)."""
    return {"type": "offset", "ahead_m": ahead_m, "left_m": left_m}


UNKNOWN = {"type": "unknown"}


def gold(value: dict, qid: str = "g") -> dict:
    """A gold row as the scorer reads one, on the distance ladder: `unknown` when the value is."""
    return {"qid": qid, "family": "object_where" if value.get("type") == "position" else "path_ahead", "gold": value,
            "slice": "unknown" if value["type"] == "unknown" else "answerable", "answer_class": None, "range_m": None,
            "credit": "ladder", "region": None, "tolerances": {"ladder": [0.25, 0.5, 1.0, 2.0]},
            "metric": "xy" if value["type"] == "position" else None,
            "scope": "memory", "reference": None, "critical": True, "subjects": [], "evidence_k": [],
            "entities": []}


def record(question: dict, value: dict, *, script: str, when: str = "end", phase: str = "cleaning",
           base: tuple = (0.0, 0.0), yaw: float = 0.0, rooms: tuple = (), nm: dict | None = None,
           siblings: tuple = (), clock: str = "", cameras: tuple | None = None) -> dict:
    """A prior record (`answer_keys.RECORD_KEYS`) of another script's asked `question`, with its gold `value`; `clock`
    is the clock of its ask's step, which only a time gold is read against; `cameras` are the camera poses it had faced
    up to the ask (default: the camera of a vacuum standing at `base` heading `yaw`, the ask's step alone)."""
    return {"script": script, "template": question["template"], "slots": question["slots"], "when": when,
            "clock": clock, "phase": phase, "base": base, "yaw": yaw, "rooms": rooms, "nm": nm,
            "siblings": list(siblings), "cameras": cameras or (step(0, base[0], base[1], yaw)["camera"],),
            "gold": gold(value)}
