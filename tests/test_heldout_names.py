"""The names of every held-out map present (M4-integration call 8): no word a held-out layout carries, put into any text
a registered job can produce, leaks an id or names a map. This is `test_brief.test_every_brief_is_clean`'s scan, run on
the held-out maps' region names, kinds and furniture names (which a brief or an event may quote: a drop zone's name is
its address in a brief) instead of the bundled ones. Held-out maps come from `STBENCH_MAPS` as in
`test_heldout_maps.py`; no layout fact is stated."""

from __future__ import annotations

import pytest

from stbench import job_registry, map_files
from stbench.brief import DOMAINS, VARIANTS, compose
from stbench.mapspec import load_map
from stbench.robots import ROBOTS, robot_doc
from stbench.stream_protocol import CHANNELS
from stbench.timeline_ego import TaskPlan
from test_brief import leaks_in, template_fields
from test_heldout_maps import maps_present


@pytest.mark.parametrize("name", maps_present())
def test_every_held_out_map_names_nothing_it_should_not(name):
    """Contract (VISION rule 6, design §7; M4-integration call 8): every region name and kind and every furniture name
    of the map, put as every field of every registered job's assignment templates, for every domain, channel and
    variant the job allows, gives no text with a `leak.RULES` hit (an entity id, a region id, the word "truth") or the
    name of a map, bundled or held out. The scan is run on planted text first, so it is not passing vacuously.
    Regression caught: a held-out place name that leaks an id or a map's name through a brief."""
    m = load_map(name)
    words = sorted({*(w for r in m.regions for w in (r.name, r.kind)), *(f.name for f in m.furniture)})
    maps = [*map_files.bundled_maps(), *map_files.HELD_OUT]
    assert words and name in maps
    assert leaks_in(f"the {name} robot is near E0042 in R3", maps) == ["entity_id", "region_id", f"map name {name}"]
    scanned = 0
    for job in job_registry.all_jobs():
        variants = VARIANTS if job.swaps else ("correct", "neutral")
        fields = template_fields(job)
        for word in words:
            plans = [TaskPlan(kind, dict.fromkeys(fields, word)) for kind in job.task_kinds]
            for domain, channel, variant in ((d, c, v) for d in DOMAINS for c in CHANNELS for v in variants):
                window = [p for p in plans if variant != "swapped" or any(p.kind in pair for pair in job.swaps)]
                for text in compose(window, job=job, robot=robot_doc(ROBOTS[job.robot]), domain=domain, channel=channel,
                                    variant=variant):
                    assert leaks_in(text, maps) == [], (job.id, word, domain, channel, variant)
                    scanned += 1
    assert scanned >= len(words) * len(DOMAINS) * len(CHANNELS) * 2
