"""The oracle perception channel (`perception/oracle.jsonl`, agent-facing) and its private key file.

Per keyframe, one detection for every entity whose visibility status is "seen" (carried, light and fixed alike; doors
add `"state": {"open": bool}`), in a seeded shuffle. "Seen" is by the entity's kind (`seen_table`: its own seen bound,
else 215 px at 640x480):

    {"k": 7, "clock": "2026-05-14T08:37:11",
     "detections": [{"d": "9f3a01bc", "kind": "mug", "attributes": {"color": "red"}, "bbox_px": [u0, v0, u1, v1],
                     "center_cam": [x, y, z], "center_map": [x, y, z], "extent_m": [sx, sy, sz],
                     "floor_hull": [[x, y], ...]}, ...]}

- `d` is a random 8-hex detection key, fresh for every detection: it names the detection within one frame and never
  persists, so telling objects apart over time stays the memory system's job. The generator of each keyframe is seeded
  from the script's map, seed, robot and job, a fixed seed key (`PERCEPTION_SEED_KEY`) and that keyframe's script `k`
  through `stbench.seeding` (no tier, no robot profile), so a rerun reproduces the file and the seed is never written
  out. The keys depend on where the keyframe sits in its script, so this file is never handed to a system whole: the
  stream protocol re-keys every detection at delivery from the case's id and the case's own `k`, and nothing delivered
  depends on the seed or on where the window starts.
- A row has no `t`: script time never reaches a system (the pose rows have none either).
- `attributes` are visual only (`VISUAL_ATTRIBUTES`): nothing that names an instance (map names, receptacle lists,
  which rooms a door joins).
- `center_cam` is the TRUE bounding-box centre, not a pixel centroid: the placement's base centre plus the placement
  rotation applied to (0, 0, height / 2), in the OpenCV camera frame of the keyframe. `extent_m` is the entity size.
  `center_map` is the same point in the map frame (so a system is not tested on a frame transform it can do with the
  pose). An open door is reported like any other entity: its centre is the middle of the doorway.
- `floor_hull` (0.7.0) is what the frame shows of the thing, on the floor: the convex hull of its own
  instance pixels' (x, y), unprojected through the frame's depth, in the map frame, to the millimetre, counter-clockwise
  (`nearest_part.floor_hull`, the function the `ego_now` and `ego_mem` gold uses). It is null when the thing is not
  recognisable in the frame (`recognise.recognisable`: too few pixels, too dark, a box at the image edge, no depth).
  The nearest point of the hull to a ground point is the answer to "the point of the thing nearest the camera", so a
  table that remembers the hulls of a thing's sightings answers `ego_mem` as the gold does; the pictures alone would
  give a table the thing's box but not its pixels.
- **A person's `floor_hull` (0.8.0)** is set exactly when the sighting is the person evidence rule's
  (`person_seen.hull`): the floor's conditions with the box allowed to touch the top and the sides of the image, and
  the contact tie at the job's clearance. `closest_person` and `person_mem` take their gold from the same function, so
  a table that keeps the person hulls reproduces both. Every other thing's is as above.

`private/<id>/perception_ids.jsonl` maps each key to its entity: {"k": 7, "ids": {"9f3a01bc": "E0042"}}. The leak scan's
key list for the oracle file is `leak.PERCEPTION_KEYS`: exactly the keys `build` writes.
"""

from __future__ import annotations

import numpy as np

from . import job_registry
from .camera import Camera, map_to_cam, rot_matrix
from .case import ScriptPaths
from .episode import write_jsonl
from .nearest_part import floor_hull
from .person_seen import PERSON_KIND, Frame, GapsOf
from .person_seen import hull as person_hull
from .pixels import PointsOf, ScriptPixels
from .recognise import recognisable
from .seeding import make_rng
from .timeline import Timeline, load
from .visibility import load_pose, read, status

# The fixed part of each keyframe's seed, after the script's map, seed, robot and job. It is a seed, never changed: it
# holds the strings the draw once took from the generator's release and the channel's tag, so a rerun of `perceive`
# gives the `oracle.jsonl` of the released data byte for byte. Changing it redraws every detection key and shuffle.
PERCEPTION_SEED_KEY = ("0.11.0", "oracle/4")
VISUAL_ATTRIBUTES = ("color", "size")


def visual_attributes(attributes: dict) -> dict:
    return {a: attributes[a] for a in VISUAL_ATTRIBUTES if a in attributes}


def bbox_center_map(tl: Timeline, entity_id: str, t: float) -> np.ndarray:
    """True bounding-box centre in the map frame: base centre plus the placement rotation applied to (0, 0, h/2)."""
    p = tl.placement_of(entity_id, t)
    h = tl.entity(entity_id).size[2]
    return np.asarray(p.pos, dtype=float) + rot_matrix(p.rot) @ np.array([0.0, 0.0, h / 2])


def _keys(rng: np.random.Generator, n: int) -> list[str]:
    """n distinct random 8-hex keys."""
    while True:
        keys = [f"{int(x):08x}" for x in rng.integers(0, 1 << 32, size=n)]
        if len(set(keys)) == n:
            return keys


def detection(tl: Timeline, rec: dict, key: str, pose: dict, cam: Camera | None = None,
              points_of: PointsOf | None = None, gaps_of: GapsOf | None = None, c_m: float | None = None) -> dict:
    e = tl.entity(rec["id"])
    t = tl.keyframe(pose["k"]).t
    center_map = bbox_center_map(tl, e.id, t)
    center = map_to_cam(center_map, pose["camera"]["pos"], pose["camera"]["rot"])
    det = {"d": key, "kind": e.kind, "attributes": visual_attributes(e.attributes), "bbox_px": list(rec["bbox"]),
           "center_cam": [round(float(x), 4) for x in center], "center_map": [round(float(x), 4) for x in center_map],
           "extent_m": [round(float(x), 4) for x in e.size], "floor_hull": None}
    if points_of is not None and cam is not None and e.kind == PERSON_KIND:
        if c_m is None:
            raise ValueError(f"{e.id} is a person: its floor hull needs the job's clearance (`c_m`)")
        frame = Frame(tl, cam, pose["k"], tuple(pose["camera"]["pos"]), rec, points_of, gaps_of)
        det["floor_hull"] = person_hull(frame, e.id, c_m)
    elif points_of is not None and cam is not None and recognisable(rec, cam, f"{e.id} at keyframe {pose['k']}"):
        det["floor_hull"] = floor_hull(points_of(pose["k"], e.id)) or None
    if e.kind == "door":
        det["state"] = {"open": bool(tl.state_of(e.id, t, "open"))}
    return det


def build(tl: Timeline, vis_rows: list[dict], poses: list[dict], cam: Camera, points_of: PointsOf | None = None,
          gaps_of: GapsOf | None = None, c_m: float | None = None) -> tuple[list[dict], list[dict]]:
    """(oracle rows, perception_ids rows) for every keyframe, each keyframe's keys and shuffle drawn from its own
    generator. `points_of(k, id)` gives a thing's pixels' points in a keyframe (`pixels.ScriptPixels.points`); with
    none, no detection has a `floor_hull`. A person's hull also reads `gaps_of(k, id)` (`ScriptPixels.gaps`; None: no
    pixel lacks a depth) and `c_m`, the job's clearance (`person_seen`); pictures and a person with no `c_m` raise
    ValueError."""
    ep = tl.episode
    by_k = {r["k"]: r for r in vis_rows}
    oracle, ids = [], []
    for pose in poses:
        rng = make_rng("perception", ep.map, ep.seed, ep.robot, ep.job, *PERCEPTION_SEED_KEY, pose["k"])
        seen = [r for r in by_k[pose["k"]]["entities"] if status(r["px"], cam, tl.entity(r["id"]).kind) == "seen"]
        keys = _keys(rng, len(seen))
        dets = [(detection(tl, r, key, pose, cam, points_of, gaps_of, c_m), r["id"])
                for r, key in zip(seen, keys, strict=True)]
        dets = [dets[i] for i in rng.permutation(len(dets))]
        oracle.append({"k": pose["k"], "clock": pose["clock"], "detections": [d for d, _ in dets]})
        ids.append({"k": pose["k"], "ids": {d["d"]: eid for d, eid in dets}})
    return oracle, ids


def write(paths: ScriptPaths) -> tuple[int, int]:
    """Write oracle.jsonl and perception_ids.jsonl from the visibility log; returns (distinct entities, detections)."""
    tl = load(paths.timeline)
    pixels = ScriptPixels(paths)
    people = any(e.kind == PERSON_KIND for e in tl.entities)   # only a person's hull reads the job's clearance
    oracle, ids = build(tl, read(paths.visibility), load_pose(paths.pose), Camera.load(paths.camera), pixels.points,
                        pixels.gaps, job_registry.get(tl.episode.job).clearance_m if people else None)
    write_jsonl(paths.oracle, oracle, sort_keys=True)
    write_jsonl(paths.perception_ids, ids)
    seen = {e for row in ids for e in row["ids"].values()}
    return len(seen), sum(len(r["detections"]) for r in oracle)
