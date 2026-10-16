"""The world a realism probe measures (`realism_world`) and the command that runs it (`realism_cmd.register`): the
script's own world rebuilt by the generator's steps, the standing camera's script over it, and the look-dev worlds
that a probe can rebuild. Free: no Blender; the renders are `test_the_probe_names_each_dressing_piece_blender`'s and
the look-dev runs'."""

from __future__ import annotations

import argparse
import dataclasses
import math
from functools import cache

import numpy as np
import pytest

from people_kit import resident
from realism_kit import apartment, generated_world, script
from stbench import SEED_KEY, people, people_home, realism_cmd
from stbench.generate import generate_script, to_timeline
from stbench.people import without_people
from stbench.realism_world import STANDING_ROBOT, rebuild_world, standing_timeline
from stbench.robots import ROBOTS
from stbench.route import robot_free_space
from stbench.seeding import make_rng

SEED = 101


@cache
def generated():
    return generate_script("apartment_a", SEED, "vacuum", "vacuum/1", 1)


def test_the_world_of_a_script_is_rebuilt_by_generating_it_again_and_a_changed_one_is_refused(monkeypatch):
    """Contract (`rebuild_world`; the probe measures the script's own world, so a script that is not what its seed makes
    is refused): a script of seed 101 with a resident home (as the generator's overlay would add one) is rebuilt: the
    generated script and this one have one world hash once the people are taken away, and the world returned is the
    start world, with no resident, whose hash is the script's without its resident. A script with one clutter piece
    made larger is not that world: ValueError naming the map, the seed and the generator. Regression caught: a probe of
    a world other than the script's, whose pictures and counts would describe nothing a robot saw, and a rebuild that
    a resident in the script breaks."""
    tl, m = generated(), apartment()
    monkeypatch.setattr(people_home, "P_HOME", 1.0)
    homed = people.add(without_people(tl), m, make_rng(m.name, SEED, SEED_KEY, "people"))
    assert resident(homed) is not None and homed.world_hash != without_people(homed).world_hash
    w = rebuild_world(homed, m)
    assert to_timeline(m, w, tl.ego, tl.episode).world_hash == without_people(homed).world_hash
    piece = next(e for e in tl.entities if e.mobility == "clutter")
    bigger = dataclasses.replace(piece, size=tuple(s * 1.1 for s in piece.size))
    changed = dataclasses.replace(homed, entities=[bigger if e.id == piece.id else e for e in homed.entities])
    with pytest.raises(ValueError, match=f"seed {SEED}"):
        rebuild_world(changed, m)


def test_the_look_dev_world_is_the_one_the_generator_makes():
    """Contract (`realism_kit.generated_world`): the world a hand-posed look-dev script is built over is the world
    `generate_script` makes of the seed before its path, so a probe can rebuild it: a script of two hand poses over
    `generated_world(101)` has the generated script's world hash once the generated script's people are taken away (the
    overlay is the generator's last step, which the look-dev world never takes), and rebuilds without complaint.
    Regression caught: look-dev frames over a world the probe refuses, which is a probe of nothing."""
    tl, m = generated(), apartment()
    hand = script(SEED, "vacuum", [(2.1, 1.9, 0.0), (5.4, 3.0, 90.0)], world=generated_world(SEED))
    assert hand.world_hash == without_people(tl).world_hash
    assert rebuild_world(hand, m) is not None


def test_the_standing_camera_stands_on_the_helpers_free_floor_in_each_room_in_turn_at_1_30_m_and_minus_15_degrees():
    """Contract (`standing_timeline`; M1's look procedure: the helper's camera, 1.30 m up, 15 degrees down, 80
    degrees across, at `--poses` seeded poses a room on its free floor): over seed 101's world, 3 poses a room make
    15 frames in the home's five rooms, taking one room after another (the first five frames are in five different
    rooms, so the sheet's first few frames are not all one room), each base a free cell of the helper's free space
    (`route.robot_free_space`) in the room of that frame, the camera at base + 1.30 m, pitch -15 degrees, one second
    between frames, the headings various; the robot of the timeline is the helper; the same seed gives the same poses
    and another another. The timeline holds the script's own world. Regression caught: a camera in a wall or a table
    (a frame of nothing), all the poses in the big room, or a camera that is not the helper's."""
    tl, m = generated(), apartment()
    w = rebuild_world(tl, m)
    rng = lambda n: np.random.default_rng(n)  # noqa: E731
    st, rooms = standing_timeline(tl, m, w, 3, rng(5))
    assert st.n_keyframes == 15 and len(rooms) == 15 and st.episode.robot == STANDING_ROBOT
    names = [r.name for r in m.regions]
    assert rooms[:5] == names and sorted(rooms) == sorted(names * 3)
    free = robot_free_space(m, w, ROBOTS[STANDING_ROBOT])
    pitch = math.radians(ROBOTS[STANDING_ROBOT].camera.pitch_deg)
    yaws = set()
    for k, room in enumerate(rooms):
        kf = st.keyframe(k)
        assert free.is_free(kf.base[:2]) and m.region(m.region_of(kf.base[:2])).name == room
        assert kf.pos[2] == pytest.approx(1.30) and kf.cam_pitch == pytest.approx(pitch, abs=1e-5) and kf.t == k
        yaws.add(round(kf.yaw, 2))
    assert len(yaws) > 10
    assert st.world_hash == without_people(tl).world_hash
    again, other = standing_timeline(tl, m, w, 3, rng(5))[0], standing_timeline(tl, m, w, 3, rng(6))[0]
    assert [again.keyframe(k).base for k in range(15)] == [st.keyframe(k).base for k in range(15)]
    assert [other.keyframe(k).base for k in range(15)] != [st.keyframe(k).base for k in range(15)]


def test_the_realism_command_has_probe_report_and_sheet_with_the_defaults_the_look_procedure_names():
    """Contract (`realism_cmd.register`; the look procedure's commands): `stbench realism probe SCRIPT_DIR --out DIR`
    takes 30 poses a room and 4 release frames by default, `report DATASET --probe DIR --out DIR` the optional linear
    passes, references, crop answers and key and a floor, and `sheet DATASET --probe DIR --refs DIR --out DIR` a seed;
    each sets its handler. Regression caught: a flag the procedure writes that the command does not take."""
    parser = argparse.ArgumentParser()
    realism_cmd.register(parser.add_subparsers())
    probe = parser.parse_args(["realism", "probe", "scripts/x", "--out", "o"])
    assert (probe.poses, probe.rgb, probe.func) == (30, 4, realism_cmd._probe)
    report = parser.parse_args(["realism", "report", "d", "--probe", "p", "--linear", "l", "--refs", "r",
                                "--crop-answers", "a.json", "--recog-lum", "0.01", "--out", "o"])
    assert (report.crop_answers, report.crop_key, report.recog_lum, report.func) == ("a.json", None, 0.01,
                                                                                   realism_cmd._report)
    sheet = parser.parse_args(["realism", "sheet", "d", "--probe", "p", "--refs", "r", "--out", "o", "--seed", "3"])
    assert (sheet.seed, sheet.func) == (3, realism_cmd._sheet)
