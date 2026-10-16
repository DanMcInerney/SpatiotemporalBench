"""The home's people: one resident, home for about half of a script's tasks (M2's lane P1; docs/people.md).

`people` reads this module for the catalog `home` (`people_<catalog>.py` is the shape every catalog's people take: M3's
warehouse and downtown add theirs as files of their own). The resident is one standing model (lane P2's `resident`, an
ordinary mover asset drawn at scale 1: there is no pose and no gait), a `mover` entity of kind `person` with no
attributes: a person is not described, and no M2 question names one. Every number is a modelling choice, `SOURCES` says
which and why; `BOX_M` is the one that must equal the asset's measured box.
"""

from __future__ import annotations

KIND = "person"  # the kind word: detections show it; no question names it in M2
MODEL = "resident"  # lane P2's asset id
BOX_M = (0.33573, 0.56036, 1.76497)  # length along the facing, width (arms hanging), height: P2's native_bbox_m
COUNT = 1  # residents a script
P_HOME = 0.5  # home for a task with this probability
SPEED_MPS = (0.8, 1.2)  # indoor walking, drawn per leg
DWELL_S = (20.0, 120.0)  # standing at a counter, a shelf, a doorway: one standing body reads best when it mostly stands
SAME_ROOM = 0.7  # a next waypoint in the room the resident stands in
CAMERA_M = 0.8  # its centre at least this far from the camera (validate's `mover_camera_clearance`)
ROBOT_KEEP_M = 0.2  # its footprint at least this far from the robot's body, and from the body swept between keyframes
START_ROBOT_M = 1.5  # it starts a task at least this far from the robot's first pose

SOURCES: dict[str, str] = {
    "KIND": "modelling choice: the kind word of the resident's mover entity",
    "MODEL": "lane P2's `resident` entry in src/stbench/assets/manifest.json",
    "BOX_M": "the manifest's measured native_bbox_m of MODEL (one adult standing, arms hanging, shirt and jeans: about "
             "0.34 m deep, 0.56 m across and 1.76 m tall), so the mover path draws it at scale 1",
    "COUNT": "modelling choice: one resident a script, one standing body (VISION §4: simple before realistic)",
    "P_HOME": "modelling choice: home for about half the tasks, so a blind guess learns neither presence nor absence",
    "SPEED_MPS": "modelling choice: below the 1.3-1.4 m/s of free walking outdoors, since a flat's turns, doorways and "
                 "furniture slow a walk",
    "DWELL_S": "modelling choice: a person at home mostly stands or sits, so legs of walking are short between waits "
               "of 20 s to 2 min",
    "SAME_ROOM": "modelling choice: people mostly move about within a room",
    "CAMERA_M": "modelling choice: the camera is never inside a body or against one: 0.8 m from the centre less the "
                "box's 0.33 m half diagonal leaves about 0.47 m",
    "ROBOT_KEEP_M": "modelling choice: people step round a working vacuum, so a footprint keeps a hand's breadth "
                    "beyond the robot's body (the robot never reacts to a person, its path is planned first)",
    "START_ROBOT_M": "modelling choice: the resident does not stand at the dock when a task begins",
}
