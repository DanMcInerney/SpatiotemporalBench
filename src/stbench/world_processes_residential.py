"""The suburb's processes between flights: the downtown's, unchanged (docs/job-drone.md, "The downtown's processes").

`world_processes.gaps` loads this module for a map of the residential catalog and calls `start` before the first flight
and `gap` before each later one. The residential catalog's five kinds (car, van, pickup, skip, dumpster) are five of the
downtown's eight, and the downtown's half-lives are keyed by kind (`world_processes_downtown.HALF_LIFE_S`): cars, vans
and pickups 1 h (on-street and driveway parking turns over about hourly), skips and dumpsters 48 h. A suburb has no
doors, no cart and no portable toilet, so nothing else changes; no thing is added or removed. The functions and the
`SOURCES` of the rates are the downtown's own, so there is one owner of each number.
"""

from __future__ import annotations

from .world_processes_downtown import SOURCES, gap, movers, start  # noqa: F401  (the module's public functions)
