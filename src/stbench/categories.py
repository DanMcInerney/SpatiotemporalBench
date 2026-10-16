"""The question categories (docs/categories.md): the six decisions a question is authored from, and the category each
family instantiates.

A category is the authoring unit. A new job or place declares which categories it faces and fills each one's
`JobSpec` hooks; a family is the gold function that instantiates a category for a motion, an objective or a predicate.
The ability (the headline's cells) and the kind (a report split, `abilities.FAMILY_KIND`) are two other splits of the
same families and do not change with the category.

`FAMILY_CATEGORY` files every family the registry (`questions_text.FAMILIES`) knows, retired ones included, under
exactly one category (`tests/test_categories.py` holds it to the registry and to the document). A retired family files
under the category whose decision it came nearest to. C6 is gated on data: no family is asked for it today, and its
three retired families file here.

Neutral and standard-library only, as `abilities`.
"""

from __future__ import annotations

CATEGORIES = {"C1": "Room to move", "C2": "Separation", "C3": "Locate", "C4": "Which way", "C5": "What matters here",
              "C6": "Change"}

FAMILY_CATEGORY = {
    # C1: how far the body can go (`clear_path`; the older corridor families and the zone questions are retired)
    "clear_path": "C1", "path_ahead": "C1", "rear_clear": "C1", "zone_clear": "C1", "clearance_then": "C1",
    # C2: how close a person comes
    "closest_approach": "C2", "approach_mem": "C2", "closest_person": "C2",
    # C3: where a thing is (from the camera, as last seen, on the map); the questions about time, pairs of things and
    # the robot's own base ask what the same last sightings or the same log hold
    "ego_now": "C3", "ego_mem": "C3", "object_where": "C3", "person_mem": "C3", "person_where": "C3",
    "out_of_view": "C3", "when_seen": "C3", "order": "C3", "distance_between": "C3", "where_was_i": "C3",
    # C4: which heading (`adjacent` asked the layout's topology, the route between fixed places)
    "open_heading": "C4", "way_to": "C4", "adjacent": "C4",
    # C5: the set of things that meet the job's predicate
    "task_things": "C5", "tangle_ahead": "C5", "drop_zone_people": "C5", "uncovered": "C5", "slots_empty": "C5",
    # C6: change between visits, gated on data
    "object_was": "C6", "same_place": "C6", "changed": "C6",
}
