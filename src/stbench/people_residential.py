"""The suburb's people: a few pedestrians on its sidewalks (M4 suburb, part b; M3's overlay, `people.py`).

The overlay's fourth catalog module (see `people_downtown`, which it follows). A pedestrian is the same standing body as
the downtown's, the manifest's `worker`, a `mover` entity of kind `person` with no attributes: its kind, model, box and
camera clearance are the downtown's (`people_downtown`, imported, so there is one owner of the placeholder box and of
the shared camera clearance). What differs is the street: a residential street has fewer people than a high street, so
`COUNT` and `P_HOME` are lower and the walk is the same, over a 300 m street whose walk grid is coarser (`GRID_M`).
People walk on the tops of the `sidewalk` pieces (0.15 m, `WALK_ON`) with their
whole footprint on them, so they never overhang the kerb into the carriageway or the bays; garden paths and the passage
are not walked. They keep off the drone's drop zones (`KEEP_OFF`): `drone/1` balances each zone's answers by a coin that
a drop enforces before the flight, and a pedestrian in a zone would turn a clear coin blocked. Stated simplifications:
the two sidewalks are not joined (nobody crosses the carriageway), so a script's pedestrians walk the longer of the two
free stretches; no pet and no moving traffic is drawn; nobody enters a front garden or the car park. Every number is a
modelling choice, `SOURCES` says which and why.
"""

from __future__ import annotations

from .people_downtown import BOX_M, CAMERA_M, KIND, MODEL, ROBOT_KEEP_M, SPEED_MPS, START_ROBOT_M  # noqa: F401

COUNT = 4  # pedestrians a script
P_HOME = 0.5  # out on the street for a task with this probability: about two in view of a flight
DWELL_S = (5.0, 45.0)  # a stand between legs: a greeting at a gate, a phone, waiting for a child
SAME_ROOM = 0.5  # a next waypoint in the region the pedestrian stands in (all the walk is the street's)
WALK_ON = ("sidewalk", "plaza")  # fixed-piece kinds pedestrians walk on, all at one height
KEEP_OFF = ("drop_zone",)  # fixed-piece kinds a pedestrian never enters: the drone's drop zones
GRID_M = 0.2  # the walk grid's cell: 300 x 82 m is 0.6 million cells

SOURCES: dict[str, str] = {
    "KIND": "the downtown's `people_downtown.KIND`: the kind word every catalog's people share",
    "MODEL": "the downtown's `people_downtown.MODEL`: M3-people step 0's `worker` entry of the asset manifest",
    "BOX_M": "the downtown's `people_downtown.BOX_M`, the manifest's native_bbox_m of MODEL, so the mover path "
             "draws it at scale 1 (one owner: the placeholder is replaced there once)",
    "CAMERA_M": "the downtown's `people_downtown.CAMERA_M`, shared by every catalog's people (`test_one_person_kind`)",
    "SPEED_MPS": "the downtown's `people_downtown.SPEED_MPS`: free walking outdoors is about 1.3-1.4 m/s",
    "ROBOT_KEEP_M": "the downtown's `people_downtown.ROBOT_KEEP_M`: where the drone flies over nothing (landing, "
                    "near the pad) a footprint keeps half a metre beyond its body, or the body's own 3 m clearance "
                    "when that is more (the robot never reacts to a person)",
    "START_ROBOT_M": "the downtown's `people_downtown.START_ROBOT_M`: the drone's first pose is its nest's pad; no "
                     "pedestrian starts beside it",
    "COUNT": "modelling choice, sized by a ratio: Moudon et al.'s twelve Puget Sound centres (walkinginfo.org, "
             "\"Effects of Site Design on Pedestrian Travel in Mixed-Use, Medium-Density Environments\") counted 12.5 "
             "pedestrians an hour per 1,000 residents at the suburban sites against 37.7 at the urban ones, a third; "
             "the downtown's 8 pedestrians present with probability 0.7 are 5.6 in a flight, and 4 present with "
             "probability 0.5 are 2.0, a third of that (no source counts a residential street's people directly)",
    "P_HOME": "modelling choice: half the flights meet a given pedestrian, so the street is often empty of a given one "
              "and a blind guess learns neither presence nor absence",
    "DWELL_S": "modelling choice: a pedestrian stops for seconds on a residential sidewalk, rarely for minutes",
    "SAME_ROOM": "modelling choice: every walkable cell lies in the street region, so the share only breaks ties",
    "WALK_ON": "modelling choice: people walk on the sidewalks, never in the carriageway or a garden (a stated "
               "simplification: no crossing, so the bays' cars are never walked through)",
    "KEEP_OFF": "modelling choice: pedestrians keep off the drop zones (the drone's zone coins read them: M3-Y.md §4)",
    "GRID_M": "modelling choice: a 0.2 m grid keeps the walk grid to 0.6 million cells over the 300 m street; a body "
              "0.34 x 0.56 m fits through any sidewalk's 1.8 m (1.14 m of free width)",
}
