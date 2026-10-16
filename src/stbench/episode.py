"""The writers of a script's own files: JSON and JSON-lines, the pose rows and `write_script`.

The layout, the index and the case records are in `case.py` (`ScriptPaths`, `CasePaths`). `write_script` lays out what a
timeline alone decides: the private `timeline.json`, and under `scripts/<sid>/` the robot's public `robot.json`,
`sensors/camera.json` and `sensors/pose.jsonl`. The renderer adds the camera images under `scripts/<sid>/sensors/rgb/`
and, private, the depth (`private/<sid>/depth/`) and the instance images; `perceive` adds the oracle channel.
`leak_scan` (leak.py) checks everything under `scripts/<sid>`, and flags any depth file there.

A pose row of a job that states a subgoal (`JobSpec.subgoals`) holds it as a top-level `subgoal`, beside `status`: what
the robot is doing at that keyframe, in words. It is worked out from the timeline alone and is not part of the ego, so
the ego does not change with it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from . import job_registry, map_files
from .case import ScriptPaths, claim_id, read_index, write_entry
from .geometry import quat_z
from .mapspec import load_map
from .robots import ROBOTS, camera_doc, robot_doc
from .timeline import Timeline, dumps, load, save


def subgoals_of(tl: Timeline) -> list[str] | None:
    """The subgoal of every keyframe of the timeline, from its job's `subgoals`; None when the job states none (or is no
    registered job: a hand-built timeline). ValueError when the job returns other than one non-empty text a keyframe,
    since a pose file with a gap would give a system a header with none."""
    try:
        job = job_registry.get(tl.episode.job)
    except KeyError:
        return None
    if job.subgoals is None:
        return None
    texts = job.subgoals(tl, load_map(tl.episode.map))
    if len(texts) != tl.n_keyframes or not all(isinstance(t, str) and t.strip() for t in texts):
        raise ValueError(f"{job.id} states a subgoal for {len(texts)} of the {tl.n_keyframes} keyframes, or an empty "
                         f"one: it must state one non-empty text for every keyframe")
    return texts


def pose_rows(tl: Timeline) -> list[dict]:
    """sensors/pose.jsonl rows (script numbering; a case rebases them with `case.case_rows`), fully determined by the
    timeline: the camera, the base (its position, `rot` as a quaternion about z, and its heading in degrees, which a
    camera pitched to nadir cannot give), the status with the task index and, for a job that states one, the `subgoal`
    (`subgoals_of`). No `t`: script time never reaches a system. Degrees are rounded to 3 places."""
    subgoals = subgoals_of(tl)
    rows = [
        {"k": kf.k, "clock": tl.clock(kf.t),
         "camera": {"pos": list(kf.pos), "rot": list(kf.rot), "pitch_deg": round(math.degrees(kf.cam_pitch), 3)},
         "pose": {"pos": list(kf.base), "rot": list(quat_z(kf.yaw)), "yaw_deg": round(math.degrees(kf.yaw), 3)},
         "status": {"phase": kf.phase, "task": kf.task, **kf.status}}
        for kf in map(tl.keyframe, range(tl.n_keyframes))
    ]
    if subgoals is not None:
        for row, text in zip(rows, subgoals, strict=True):
            row["subgoal"] = text
    return rows


def write_jsonl(path: Path, rows: list[dict], sort_keys: bool = False) -> None:
    """One JSON object per line. Agent-facing records pass `sort_keys=True`: key order carries no information."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(json.dumps(r, sort_keys=sort_keys) + "\n" for r in rows).encode("utf-8"))


def json_bytes(doc: dict) -> bytes:
    """The bytes `write_json` writes."""
    return (json.dumps(doc) + "\n").encode("utf-8")


def write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(doc))


def _timeline_differs(path: Path, tl: Timeline) -> bool:
    """Whether the `timeline.json` at `path`, if there is one, holds another timeline than `tl`. They are compared as
    the timeline the file parses to, written out again, because a file of an earlier build holds keys nothing reads (a
    schema string, digests, a generator, section hashes) that `dumps` no longer writes: the same content in other
    bytes is no change. A file that does not parse as a timeline differs."""
    if not path.is_file():
        return False
    try:
        return dumps(load(path)) != dumps(tl)
    except (ValueError, LookupError, TypeError):
        return True


def _camera_differs(path: Path, camera: dict) -> bool:
    """Whether the `camera.json` at `path`, if there is one, holds another document than `camera` (compared parsed)."""
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")) != json.loads(json_bytes(camera))
    except ValueError:
        return True


def script_split(map_name: str, split: str | None = None) -> str:
    """The split of a script on `map_name`: `heldout` for a map of `map_files.HELD_OUT` (read now, so a test can patch
    it), else `dev`. A split given must agree: `dev` or `test` on a held-out map, or `heldout` on another, is a
    ValueError naming both."""
    held_out = map_name in map_files.HELD_OUT
    if split is None:
        return "heldout" if held_out else "dev"
    if (split == "heldout") != held_out:
        raise ValueError(f"split {split!r} contradicts the map {map_name!r}, which is "
                         f"{'held out' if held_out else 'not held out'}: its scripts are "
                         f"{'heldout' if held_out else 'dev or test'}")
    return split


def write_script(dataset: str | Path, tl: Timeline, split: str | None = None) -> ScriptPaths:
    """Create the script's agent-facing and private folders and record its random id in the private index. The robot's
    files come from `robots` (the spec of the timeline's robot as it is now). Writing the same inputs again reuses their
    id and overwrites the files, except that a script that has a render log is rendered from its `timeline.json` and
    `sensors/camera.json`: a ValueError naming the file refuses to write either with other content than the ones on
    disk (the same content proceeds, in whatever keys and bytes an earlier build wrote it, so a rebuild changes
    nothing), before anything is written. The script's split follows its map (`script_split`: `heldout` for a held-out
    map, else `dev`; `split` may say `test` for a public map), and a dataset root holds one split: a script whose split
    differs from that of a script already in the root's index is a ValueError, before anything is written, so an export
    of a public root can never carry a held-out script."""
    ep = tl.episode
    robot = ROBOTS[ep.robot]
    split = script_split(ep.map, split)
    held = sorted({e["split"] for e in read_index(dataset).values() if e["kind"] == "script"} - {split})
    if held:
        raise ValueError(f"the dataset root {dataset} holds {held[0]} scripts and this one is {split}: a root holds "
                         f"one split")
    entry = {"kind": "script", "map": ep.map, "seed": ep.seed, "robot": ep.robot, "job": ep.job,
             "tasks": len(tl.ego.tasks), "split": split}
    paths = ScriptPaths(dataset, claim_id(dataset, entry))
    camera = camera_doc(robot)
    if paths.render_log.exists():
        for path, differs in ((paths.timeline, _timeline_differs(paths.timeline, tl)),
                              (paths.camera, _camera_differs(paths.camera, camera))):
            if differs:
                raise ValueError(f"{path} would be rewritten with other content, and script {paths.id} is rendered "
                                 f"from the file on disk; delete the script's folder to regenerate it")
    paths.mkdirs()
    save(tl, paths.timeline)
    write_json(paths.robot, robot_doc(robot))
    write_json(paths.camera, camera)
    write_jsonl(paths.pose, pose_rows(tl), sort_keys=True)
    write_entry(dataset, paths.id, entry)
    return paths
