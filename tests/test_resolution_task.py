"""The resolution of every item of a `task_things` gold (0.9.0, `questions_task`, `resolution_at`): the row's tolerances
are the matching radius, one resolution an item in the gold's item order, and the false-item weight. Hand-worked on
`task_kit`'s world (the vacuum, camera 0.08 m up, mounted 0.13 m ahead of the base; keyframe 3 faces 135 degrees and
keyframe 5 225): one pixel at a floor point d from the camera is hypot((d^2 + 0.0064) / 30.5089, sqrt(d^2 + 0.0064) /
381.36115), at least 0.01."""

from __future__ import annotations

import pytest

import task_kit as tk
from stbench.questions_task import relax
from task_kit import PATH_POSES, cand, put, y

# the way the robot docked as `task_kit` has it, and the bases at keyframes 3 and 5 put by hand, so that the two
# sightings below are made from cameras far apart: (1.9081, 1.0919) at keyframe 3 and (5.9081, 0.9081) at keyframe 5
POSES = {**PATH_POSES, 3: (2.0, 1.0), 5: (6.0, 1.0)}


def test_each_item_is_resolved_at_its_own_last_sighting_in_the_golds_item_order():
    """Contract (brief 3.7, a list item: one pixel at the item's last recognisable sighting; `questions_task._build`):
    `resolutions_m` holds one resolution per gold item, in the order of the gold's items (sorted by place), each read
    from the camera of that item's own sighting. By hand: P1 at (5.0, 1.7), seen only at keyframe 5 from (5.9081,
    0.9081), is d = 1.2049 away: (1.4517 + 0.0064) / 30.5089 = 0.04779 radially, 0.00317 sideways, 0.0479 together; P2
    at (8.0, 1.7), seen only at keyframe 3 from (1.9081, 1.0919), d = 6.1222: 1.2288. The gold lists P1 then P2 (by x),
    though P2 was sighted first; read the other way round the list would be [0.1643, 0.3258]. `match_m` is the job's 0.5
    m, the false weight 0.5. Regression caught: resolutions in the order of sightings or of entities, the ask's or the
    last keyframe's camera, or a list that does not line up with the items."""
    s = tk.script(poses=POSES)
    put(s, "P2", (8.0, y(0.5)), at=(3,))
    put(s, "P1", (5.0, y(0.5)), at=(5,))
    c = cand(s)
    assert [i["xy"] for i in c.gold["items"]] == [[5.0, 1.7], [8.0, 1.7]]
    assert c.tolerances == {"match_m": 0.5, "resolutions_m": [pytest.approx(0.0479, abs=1e-4),
                                                              pytest.approx(1.2288, abs=1e-4)], "fp_weight": 0.5}


def test_a_relaxed_item_takes_its_resolution_with_it_and_an_empty_gold_has_none():
    """Contract (`questions_task.relax`, `_build`): when a look question names an item and it becomes a don't-care
    place, the resolution list is rebuilt for the items that remain, still in their order; a gold with no item holds an
    empty list and the job's matching radius. By hand (the module's scene): relaxing P1 leaves P2 alone, 1.2288; a task
    with no thing at all is `[]` and 0.5. Regression caught: a stale list longer than the items (a score that reads a
    resolution of an item that is no longer asked) or one that shifts onto the wrong item."""
    s = tk.script(poses=POSES)
    put(s, "P2", (8.0, y(0.5)), at=(3,))
    put(s, "P1", (5.0, y(0.5)), at=(5,))
    relaxed = relax(cand(s), ["P1"])
    assert [i["xy"] for i in relaxed.gold["items"]] == [[8.0, 1.7]]
    assert relaxed.tolerances == {"match_m": 0.5, "resolutions_m": [pytest.approx(1.2288, abs=1e-4)], "fp_weight": 0.5}
    empty = cand(tk.script(poses=POSES))
    assert empty.gold["items"] == [] and empty.tolerances == {"match_m": 0.5, "resolutions_m": [], "fp_weight": 0.5}
