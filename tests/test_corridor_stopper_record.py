"""The `Stopper` record of the corridor screens (`corridor_screens.stopper`, 0.9.0): the verdict of `stopper_tied` and,
beside it, the resolution the stop is read to. Fixtures are `test_corridor_screens`': `clear_kit`'s tote AMR, front
edge (1.0, 0), camera 0.5 m up at x 0.98 with fy 381.36 px, the corridor stopping at 2.0 m; a tie's one-pixel error is
(d^2 + H^2) / (H fy)."""

from __future__ import annotations

import math

import pytest

from clear_kit import LAST, Owned, drive_run, thing, wall, world
from space_kit import box
from stbench import corridor_screens as screens
from test_corridor_screens import GOLD, KS, START, seen_things, thing_run

FY, CAMERA_X, CAMERA_H = 381.36, 0.98, 0.5


def read(w, gold=GOLD) -> screens.Stopper:
    return screens.stopper(w, START, 0.0, gold, w.latest_cells(KS))


def table(leg_x: float, mobility: str = "carried", px: int = 1500):
    """A table top at 0.45 to 0.50 m from x 3.01 whose two legs stand at `leg_x`, seen at `px` pixels."""
    top = Owned(box(3.01, 4.01, -0.4, 0.4, 0.45, 0.5), "E0002")
    legs = [Owned(box(leg_x, leg_x + 0.05, y, y + 0.05, 0.0, 0.45), "E0002") for y in (-0.4, 0.35)]
    script = drive_run(boxes=[top.at, *(leg.at for leg in legs)])
    thing(script, "E0002", (1.0, 0.8, 0.5), (3.51, 0.0, 0.0), mobility)
    script.see(list(KS), "E0002", px)
    return world(script, [top, *legs])


def test_the_records_verdict_is_stopper_tieds_on_the_screen_fixtures():
    """Contract: `stopper_tied(...)` is the record's `tied and visible` on every scene of the screen tests: a floor box
    (tied), a table whose legs stand 0.4 m behind its edge (not), one whose legs are at the edge (tied), a recognisable
    and a glimpsed thing (the tie holds, the stopper is not visible), a corridor at the cap (no stop). Regression
    caught: the refactor moving a 0.8.0 verdict."""
    crate = Owned(box(3.01, 3.51, -0.25, 0.25, 0.0, 0.5), "E0001")
    scenes = {"box": world(seen_things(thing_run(crate), "E0001"), [crate]), "far legs": table(3.41),
              "near legs": table(3.01), "glimpsed": table(3.01, px=100), "fixed": table(3.01, "fixed", 100)}
    want = {"box": (True, True), "far legs": (False, False), "near legs": (True, True), "glimpsed": (True, False),
            "fixed": (True, True)}
    for name, w in scenes.items():
        found = read(w)
        assert (found.tied, found.visible) == want[name], name
        assert screens.stopper_tied(w, LAST, START, 0.0, GOLD, w.latest_cells(KS)) is (found.tied and found.visible)
    at_cap = read(scenes["box"], {"type": "number", "value": 5.0})
    assert at_cap == screens.NO_STOP and (at_cap.error_m, at_cap.gap_m) == (0.0, 0.0)
    wall_world = world(drive_run(boxes=[wall(3.01).at]), [wall(3.01)])
    assert read(wall_world).tied and read(wall_world).visible   # structure is always visible


def test_the_stoppers_error_is_the_contact_pixel_error_and_its_gap_how_far_the_contact_lies_beyond_the_stop():
    """Contract (brief §3.7): `error_m` is the least one-pixel error (d^2 + H^2) / (H fy) of the stopper's floor
    contact in the frame the stop rests on, and `gap_m` how far beyond the stop that contact lies. Hand scenes, camera
    at x 0.98, H 0.5: (a) a floor box whose face's foot is at x 3.01, d = 2.03: (2.03^2 + 0.25) / (0.5 x 381.36) =
    0.0229, the contact at the stop, gap 0; (b) a table top whose edge is at 3.01 and legs at 3.11, nearest leg point
    (3.11, +-0.35), d = hypot(2.13, 0.35): (4.5369 + 0.1225 + 0.25) / 190.68 = 0.0257, the contact 0.10 m beyond the
    stop (within the 0.02 sub-cell the stop is read to); (c) a far leg (0.4 m behind): no contact within C/2, infinite.
    Regression caught: an error read from another frame or point, or a gap that is not the distance beyond the stop."""
    crate = Owned(box(3.01, 3.51, -0.25, 0.25, 0.0, 0.5), "E0001")
    box_scene = read(world(seen_things(thing_run(crate), "E0001"), [crate]))
    assert box_scene.error_m == pytest.approx((2.03 ** 2 + CAMERA_H ** 2) / (CAMERA_H * FY), abs=5e-4)
    assert box_scene.gap_m == pytest.approx(0.0, abs=0.02)
    legs = read(table(3.11))
    assert legs.tied and legs.error_m == pytest.approx((math.hypot(2.13, 0.35) ** 2 + CAMERA_H ** 2)
                                                       / (CAMERA_H * FY), abs=1e-3)
    assert legs.gap_m == pytest.approx(0.10, abs=0.02)
    behind = read(table(3.41))
    assert not behind.tied and behind.error_m == math.inf and behind.gap_m == math.inf
