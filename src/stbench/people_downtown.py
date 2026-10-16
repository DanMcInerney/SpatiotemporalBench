"""The downtown's people: eight pedestrians a script on the sidewalks and the square (M3's lane M3-people).

The overlay's third catalog module (see `people_warehouse`, `people_home`). A pedestrian is the same one standing body
as the warehouse's picker, the manifest's `worker`, a `mover` entity of kind `person` with no attributes. It walks on
the map's walk pieces (`WALK_ON`: the sidewalks and the plaza, 0.15 m up) with its whole footprint on them, at their
height, so it never overhangs a kerb into the roadway or the parked cars along it, and it keeps off the drone's drop
zones (`KEEP_OFF`: `drone/1` balances each zone's answers by a coin that a drop enforces before the flight, and a
pedestrian in a zone would turn a clear coin blocked). Stated simplifications: pedestrians never cross a roadway and
never enter the nest's yard (it is no walk piece), and no dog or moving traffic is drawn. A pedestrian walks under a
hovering drone: the drone flies over a person it is 7 m above (`robots.flies_over`). Every number is a modelling
choice, `SOURCES` says which and why; `BOX_M` is the one that must equal the asset's measured box.
"""

from __future__ import annotations

KIND = "person"  # the kind word, shared with the other catalogs: detections show it, no question names it
MODEL = "worker"  # the manifest's shod body (M3-people step 0)
BOX_M = (0.33176, 0.54019, 1.79046)
COUNT = 8  # pedestrians a script
P_HOME = 0.7  # in the district for a task with this probability
SPEED_MPS = (1.1, 1.5)  # free walking, drawn per leg
DWELL_S = (5.0, 45.0)  # waiting at a crossing, a shop front or a bench: a short stand between legs
SAME_ROOM = 0.5  # a next waypoint in the region the pedestrian stands in (the high street's sidewalk or the square)
CAMERA_M = 0.8  # its centre at least this far from the camera (validate's `mover_camera_clearance`)
ROBOT_KEEP_M = 0.5  # its footprint at least this far from the robot's body (where the drone flies over nothing)
START_ROBOT_M = 5.0  # it starts a task at least this far from the robot's first pose
WALK_ON = ("sidewalk", "plaza")  # fixed-piece kinds pedestrians walk on, all at one height
KEEP_OFF = ("drop_zone",)  # fixed-piece kinds a pedestrian never enters: the drone's drop zones
GRID_M = 0.1  # the walk grid's cell: 0.05 m over this district (100 x 96 m) would be 4 million cells

SOURCES: dict[str, str] = {
    "KIND": "modelling choice: the kind word every catalog's people share (one person kind, several bodies)",
    "MODEL": "M3-people step 0's `worker` entry in src/stbench/assets/manifest.json",
    "BOX_M": "the manifest's native_bbox_m of MODEL (a standing adult in boots, arms hanging), so the mover path draws "
             "it at scale 1",
    "COUNT": "modelling choice: a handful of people on a few hundred metres of pavement reads as a street without a "
             "question to ask of them; each costs a walk and a render",
    "P_HOME": "modelling choice: most pedestrians are there for a flight, so the street is seldom empty and a blind "
              "guess learns neither presence nor absence",
    "SPEED_MPS": "modelling choice: free walking outdoors is about 1.3-1.4 m/s (the figure `people_home` cites too), "
                 "so legs are drawn from 1.1 to 1.5 m/s",
    "DWELL_S": "modelling choice: a pedestrian stops for seconds at a crossing or shop front, rarely for minutes",
    "SAME_ROOM": "modelling choice: half the legs stay on the same sidewalk or square, half go to another region",
    "CAMERA_M": "modelling choice: the camera is never inside a body or against one: 0.8 m from the centre less the "
                "box's 0.33 m half diagonal leaves about 0.47 m (the same for every catalog's people)",
    "ROBOT_KEEP_M": "modelling choice: where the drone flies over nothing (landing, near the pad) a footprint keeps "
                    "half a metre beyond its body, or the body's own 3 m clearance when that is more (the robot "
                    "never reacts to a person)",
    "START_ROBOT_M": "modelling choice: the drone's first pose is its nest's pad; no pedestrian starts beside it",
    "WALK_ON": "modelling choice: people walk on sidewalks and the plaza, never in the roadway (a stated "
               "simplification: no crossing, so the roadways' cars and loading zones are never walked through)",
    "KEEP_OFF": "modelling choice: pedestrians keep off the drop zones (the drone's zone coins read them: M3-Y.md §4)",
    "GRID_M": "modelling choice: a 0.1 m grid keeps the walk grid to one million cells; a body 0.34 x 0.56 m fits "
              "through any sidewalk's 3.3 m of free width",
}
