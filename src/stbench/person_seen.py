"""When a sighting of a person places them (0.8.0): the one owner of the person evidence rule.

The recognisability floor (`recognise.recognisable`) drops a thing whose box touches any edge of the image, because a
thing cut by the frame is a sliver of itself. For a person that rule threw away what `closest_person` needs: a person
close to the robot fills the frame and is cut at the top by the image's edge, and 41 of the AMR's 46 closest-approach
sightings were of that kind (FD-task-questions review, section 3.2). What places a person is their **feet**, and the
feet are at the bottom of the box. So the rule for a person is the floor's with one change, and a person's sighting is
**evidence** when all of these hold in one frame:
1. the pixel count is at least `RECOG_PX` and the mean luminance at least `RECOG_LUM`, as for any thing;
2. the frame has a depth at the person's every pixel (no pixel without one: `gaps` is 0), so the floor hull misses none;
3. the box lies at least `RECOG_EDGE_PX` above the bottom edge of the image. The top and the sides may touch the
   frame's edge; the bottom may not (the feet, and with them the nearest part, would be cut);
4. the contact tie holds at C (`nearest_part.surface_tie` over `ego_support.contact_bases`): the frame shows a pixel
   where the person meets the ground within C/2 of the point of their floor hull nearest that frame's own camera, and
   that pixel reads to within C/2 at one pixel. A camera-only reader can then place the person to C/2.

The visible floor hull of such a sighting lies a median 0.07 m (q90 0.10 m) beyond the truth's footprint on the AMR's
cut sightings, well inside C/2: the feet place the person.

The module is the rule and nothing else, called from two sides that must agree: the oracle channel's `floor_hull` of a
person (`perception.detection`: set exactly when the sighting is evidence) and the people golds (`questions_people`).
That agreement is what lets `memory_table` reproduce the golds from the channel (the ceiling gate). It reads one frame
at a time, never another and never a gold: a `Frame` is the keyframe's visibility row of the person, its delivered
camera position and the pictures (`pixels.PointsOf`), so a compile (`frame_of` a `World`) and `perceive` (a `Frame`
built from the files) take the same hull.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .camera import Camera
from .ego_support import contact_bases
from .nearest_part import contact_error, floor_hull, nearest_on_hull, surface_tie
from .pixels import PointsOf
from .recognise import RECOG_EDGE_PX, RECOG_LUM, RECOG_PX
from .timeline import Timeline

if TYPE_CHECKING:
    from .questions_world import World

PERSON_KIND = "person"   # the kind word every catalog's people share (`people_home.KIND`, and the others')
GapsOf = Callable[[int, str], int]   # (script keyframe, entity id) -> how many of its pixels have no depth


@dataclass(frozen=True)
class Frame:
    """One keyframe as the rule reads it: the timeline (the placements the contact tie needs), the camera, the delivered
    camera position `(x, y, z)` of the keyframe, the visibility row of the person in it (None: no pixel), and the
    pictures. `gaps_of` None means the pictures hold no pixel without a depth."""

    tl: Timeline
    cam: Camera
    k: int
    camera_pos: tuple[float, float, float]
    rec: dict | None
    points_of: PointsOf
    gaps_of: GapsOf | None = None


def frame_of(w: World, pid: str, k: int) -> Frame:
    """The frame at script keyframe `k` of a compile's window: the visibility row of `pid` there, the delivered camera
    (`World.camera_at`) and the compile's pictures. ValueError for inputs built without pictures."""
    if w.inp.points_of is None:
        raise ValueError("these inputs carry no pictures (Inputs.points_of), so no person's evidence can be read")
    return Frame(w.tl, w.cam, k, w.camera_at(k)[0], w.vis.get(k, {}).get(pid), w.inp.points_of, w.inp.gaps_of)


def _clear(rec: dict, cam: Camera, what: str) -> bool:
    """Whether the visibility row passes the rule's first and third condition: enough pixels and light, a depth for the
    person, a box `RECOG_EDGE_PX` above the bottom edge (`bbox` is `[u0, v0, u1, v1]`, `v1` exclusive). Raises
    ValueError, naming `what`, when the pixel count passes and the row lacks `lum`, `bbox` or `depth_med_m` (every
    visibility log has all three)."""
    if rec["px"] < RECOG_PX:
        return False
    missing = [key for key in ("lum", "bbox", "depth_med_m") if key not in rec]
    missing += [key for key in ("lum", "bbox") if rec.get(key, 0) is None]   # only the depth may be null
    if missing:
        raise ValueError(f"the visibility row of {what} has no {' or '.join(missing)}: the person evidence rule "
                         f"needs it")
    return rec["depth_med_m"] is not None and rec["lum"] >= RECOG_LUM and cam.height - rec["bbox"][3] >= RECOG_EDGE_PX


def hull(frame: Frame, pid: str, c_m: float) -> list[list[float]] | None:
    """The floor hull of the person `pid` in the frame (`nearest_part.floor_hull` of the pixels' points,
    counter-clockwise, to the millimetre) when the sighting is evidence at the clearance `c_m` (the module docstring),
    else None."""
    if frame.rec is None or not _clear(frame.rec, frame.cam, f"{pid} at keyframe {frame.k}"):
        return None
    if frame.gaps_of is not None and frame.gaps_of(frame.k, pid):
        return None
    points = frame.points_of(frame.k, pid)
    if not (floor := floor_hull(points)):
        return None
    xy = nearest_on_hull(floor, frame.camera_pos[:2])
    base = contact_bases(frame, pid, [frame.k], xy)[0]
    contact = contact_error([(frame.camera_pos, points, base)], xy, frame.cam.fy, c_m)
    return floor if surface_tie(contact, False, c_m) is not None else None


def evidence(w: World, pid: str, k: int, c_m: float) -> bool:
    """Whether the person's sighting in script keyframe `k` of the window is evidence (`hull(...)` is not None)."""
    return hull(frame_of(w, pid, k), pid, c_m) is not None
