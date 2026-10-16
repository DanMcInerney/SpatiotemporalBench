"""The mapper (`table.Mapper`, fed by a system's observes) against the free-space rule applied to the script's own files
(`freespace.script_states`), on ray-cast depth whose cells are worked out by hand."""

from __future__ import annotations

from baseline_kit import EVENTS, compiled_case, play
from space_kit import box, vacuum_run
from stbench import freespace
from stbench.baselines import MemoryTable

# Keyframes 19-21 stand with the front edge at x = 0 heading +x and have a render of a wall whose face is at x = 2.21;
# keyframe 20 also sees a box, its face at x = 1.03, 0.30 m wide and 0.267 m tall. Every other keyframe has no depth.
BOX = box(1.03, 1.43, -0.15, 0.15, 0.0, 0.267)


def test_the_mapper_equals_freespace(tmp_path):
    """Contract (plan §4.7): the cells a system's mapper keeps for every keyframe of a stream equal what
    `freespace.script_states` gives from the script's depth files, because both call `grid` on the same depth with the
    same camera, band and floor. Hand cells: at keyframes 19 and 21 the wall's face at x = 2.21 makes cell (22, 0)
    occupied (its points stand 0.02-0.095 m above the floor) and the floor under cell (10, 0) is seen free; at keyframe
    20 the box's face at x = 1.03 makes (10, 0) occupied and hides the wall behind it, so (22, 0) is not observed; a
    keyframe with no depth observed nothing. Regression caught: a gold the best system cannot reproduce, which would
    fail `ceiling` for the wrong reason."""
    case = compiled_case(tmp_path, script=vacuum_run(frame_boxes={20: [BOX]}))
    _, system = play(case, MemoryTable, events=EVENTS)
    want = freespace.script_states(case.script, range(28))
    assert system.store.cells == want
    assert [system.store.cells[k] for k in (19, 20, 21)] != [{}, {}, {}]
    assert all(system.store.cells[k] == {} for k in range(19))
    for k in (19, 21):
        assert system.store.cells[k][22, 0] == "occupied" and system.store.cells[k][10, 0] == "free"
    assert system.store.cells[20][10, 0] == "occupied" and (22, 0) not in system.store.cells[20]
