"""The warehouse's people: three pickers a script, mostly on the floor (M3's lane M3-people; docs/people.md).

The overlay's second catalog module (`people_home` is the first; the downtown's is `people_downtown`). A picker is one
standing body, the manifest's `worker` (a MakeHuman adult in work clothes and boots, an ordinary mover asset drawn at
scale 1: no pose, no gait), a `mover` entity of kind `person` with no attributes: a person is not described, and no
question names one. A picker walks the floor of the dock and the aisles between the racks and keeps off the AMR's bays:
`amr/1` balances each bay's answers by a coin that a drop enforces before the task (an item into the bay or every item
out of it), and a picker standing in a bay would turn a clear coin blocked, making `no` the likelier answer. Every
number is a modelling choice, `SOURCES` says which and why; `BOX_M` is the one that must equal the asset's measured box.
"""

from __future__ import annotations

KIND = "person"  # the kind word, shared with the other catalogs: detections show it, no question names it
MODEL = "worker"  # the manifest's shod body (M3-people step 0)
BOX_M = (0.33176, 0.54019, 1.79046)
COUNT = 3  # pickers a script
P_HOME = 0.8  # on the floor for a task with this probability
SPEED_MPS = (1.0, 1.4)  # walking between pick faces, drawn per leg
DWELL_S = (10.0, 60.0)  # picking at a face: a standing pause between legs
SAME_ROOM = 0.6  # a next waypoint in the region the picker stands in
CAMERA_M = 0.8  # its centre at least this far from the camera (validate's `mover_camera_clearance`)
ROBOT_KEEP_M = 0.3  # its footprint at least this far from the robot's body, and from the body swept between keyframes
START_ROBOT_M = 2.0  # it starts a task at least this far from the robot's first pose
KEEP_OFF = ("bay",)  # fixed-piece kinds a picker never enters: the AMR's bays

SOURCES: dict[str, str] = {
    "KIND": "modelling choice: the kind word every catalog's people share (one person kind, several bodies)",
    "MODEL": "M3-people step 0's `worker` entry in src/stbench/assets/manifest.json",
    "BOX_M": "the manifest's native_bbox_m of MODEL (a standing adult in boots, arms hanging), so the mover path draws "
             "it at scale 1",
    "COUNT": "modelling choice: a shift's few pickers (VISION §4: simple before realistic); more would crowd the "
             "dock without a question to ask of them",
    "P_HOME": "modelling choice: mostly on the floor, so a blind guess learns neither presence nor absence, and a "
              "picker is usually there to keep off a bay",
    "SPEED_MPS": "modelling choice: walking between pick faces, a little below the 1.3-1.4 m/s of free walking, since "
                 "racks, pallets and the AMR slow a picker",
    "DWELL_S": "modelling choice: a pick takes seconds to a minute, so a picker mostly stands at a face between legs",
    "SAME_ROOM": "modelling choice: a picker works one region at a time (the dock, an aisle)",
    "CAMERA_M": "modelling choice: the camera is never inside a body or against one: 0.8 m from the centre less the "
                "box's 0.33 m half diagonal leaves about 0.47 m (the same for every catalog's people)",
    "ROBOT_KEEP_M": "modelling choice: a picker keeps a hand's breadth beyond the robot's body, more than a home's "
                    "resident does (a loaded AMR is a hazard; the robot never reacts to a person, its path is planned "
                    "first)",
    "START_ROBOT_M": "modelling choice: a picker does not stand at the AMR's first pose when a task begins",
    "KEEP_OFF": "modelling choice: pickers keep out of the AMR's bays (its zone coins read them: M3-W.md §4)",
}
