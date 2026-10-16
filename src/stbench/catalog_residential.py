"""The residential catalog's vocabulary: which kinds it asks about and how many things a world holds (M4-suburb §1).

`CATALOGS["residential"]` is the vocabulary of a street of detached houses and, held out, a block's courtyards: where a
delivery drone works. Its askable kinds are the downtown's own vehicles and containers, **the same `Kind` objects** (one
kind, one size, one set of colours and receptacles everywhere: `catalog.item_size`), because a car parked on a driveway
is the car of the street, and the drone's pictures and the manifest's models are shared. It has no box truck (no loading
zone fits a residential street), no portable toilet and no food cart (nobody sets one in a front garden), no movers (a
resident on foot is a person, `people_<catalog>.py`'s) and no clutter: a garden's movable things are these five kinds,
and everything else that stands in one is dressing (`render/dressing_layout_residential.py`).

This module holds the names and the numbers and imports nothing from `catalog`, as `catalog_clutter` does not: the
`Catalog` is built in `catalog.py` from the downtown's objects, in one entry,

    "residential": Catalog("residential", {k: DOWNTOWN_KINDS[k] for k in residential.KIND_NAMES}, residential.SET_KINDS,
                           residential.LEAVING_KINDS, {}, residential.N_MOVERS, residential.N_ITEMS,
                           residential.DELIVERY_KIND, DOWNTOWN_DELIVERY_COLORS),

after `from . import catalog_residential as residential`. The receptacles stay the downtown's (`parking`, `loading
zone`, `sidewalk`, `plaza`): a driveway and a kerbside bay carry a `parking` surface; a dumpster stands on a sidewalk,
or on a driveway by the 30% rule of `placement.py`, as a renovation's does.
"""

from __future__ import annotations

KIND_NAMES = ("car", "van", "pickup", "skip", "dumpster")  # the downtown's kinds, by name (the order the catalog lists)
SET_KINDS = ("dumpster",)  # the kinds that stand in identical sets
LEAVING_KINDS = ("car", "van", "pickup")  # what drives off between visits
DELIVERY_KIND = "van"  # what arrives between visits (in the downtown's colours: white first)
# The inclusive range of carried items a world draws. The brief asked for the downtown's (26, 34); but this catalog has
# 27 (kind, colour) pairs where the downtown has 43, and `test_catalog`'s never-seen-pool rule (a world of at most `hi`
# items holds at most `hi - 2` distinct descriptions; at least 5 of the catalog's must be left over) allows a `hi` of
# 24: 27 - 22 = 5. The range keeps the downtown's ratio (26 / 34), rounded.
N_ITEMS = (18, 24)
N_MOVERS = 0  # no movers: `mover_kinds` is empty
