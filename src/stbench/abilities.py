"""The abilities (0.6.0, VISION §3; clearance in 0.8.0) and the kinds (0.9.0): which ability and which kind each
question family measures.

A result row's `family` maps to one ability, and the scorer reports by ability: the `now` and `memory` columns each
weigh their abilities equally (`score_stats`), so a family's weight depends on the ability it belongs to and not on its
count of questions. `DIAGNOSTIC` abilities (`prediction`) are reported but kept out of every mean. A family this
table does not list is an ability of its own (`ability_of`), so the rows of an older run, whose families this table
predates, stay scorable.

The kinds are the vision's four, a split the reports add beside the abilities (`score_stats`' `by.kind`) and never a
cell of the headline: **harm** (what would hurt the robot's task), **help** (what would most help it), **situation**
(what it sees and saw) and **reuse** (what is worth remembering next time). `FAMILY_KIND` files each family; `kind_of`.

Neutral and standard-library only: the scorer, the report and the viewer read it, and so may a baseline.
"""

from __future__ import annotations

FAMILY_ABILITY = {"ego_now": "distance", "ego_mem": "memory", "object_where": "memory", "object_was": "change",
                  "same_place": "change", "task_things": "relevance", "mover_next": "prediction",
                  "clear_path": "clearance", "closest_person": "clearance", "person_mem": "clearance",
                  # 0.11.0 (FD-0.11 §3.3, §3.4.2): a heading is a clearance decision (how much room the body has to go),
                  # and the way to a remembered thing is the memory of where it was and of the layout between
                  "open_heading": "clearance", "way_to": "memory",
                  # 0.10.0 (FD-questions-0.10 §2): the harm families of people and the robot's own hazards
                  "closest_approach": "clearance", "drop_zone_people": "clearance", "tangle_ahead": "clearance",
                  # 0.11.0 (FD-0.11-hard §3.2): `closest_approach` for the person last seen, out of view now; the harm
                  # row of the AMR's blind aisle end, so it pools into each place's clearance cell (scope `memory`)
                  "approach_mem": "clearance"}
KINDS = ("harm", "help", "situation", "reuse")
# The retired families file under their nearest kind: `object_was` and `same_place` ask what stayed and what moved
# (reuse, the vision's "what is worth remembering"), `mover_next` where a mover goes next (situation).
FAMILY_KIND = {"clear_path": "harm", "closest_person": "harm", "person_mem": "harm", "open_heading": "help",
               "way_to": "help",
               "ego_now": "situation", "ego_mem": "situation", "mover_next": "situation",
               "object_where": "reuse", "task_things": "reuse", "object_was": "reuse", "same_place": "reuse",
               "closest_approach": "harm", "drop_zone_people": "harm", "tangle_ahead": "harm",   # 0.10.0
               "approach_mem": "harm"}   # 0.11.0
# Reported beside the means, in none of them. `prediction`: where a mover goes next is a retired family. `heading` was
# one too in 0.9.0 and 0.10.0, while its credit was a cliff (a heading 2 to 10 degrees off the opening scored -0.4 to
# -1, as did one that left 1 to 2 m of room); since 0.11.0 it is room to move on the one need rule
# (`credit_kernel.KAPPA`), graded, and `open_heading` counts in clearance.
DIAGNOSTIC = frozenset({"prediction"})
MIN_ABILITY_N = 20   # a cell of fewer questions on a job in the dataset weighs n / 20 of a full one (FD-0.11 §3.7)


def ability_of(family: str) -> str:
    """The ability a family measures: its entry in `FAMILY_ABILITY`, else the family itself (an unlisted family is an
    ability of its own)."""
    return FAMILY_ABILITY.get(family, family)


def kind_of(family: str) -> str:
    """The kind a family belongs to (`KINDS`): its entry in `FAMILY_KIND`, else the family itself, as `ability_of`
    does, so the rows of an older run stay scorable."""
    return FAMILY_KIND.get(family, family)
