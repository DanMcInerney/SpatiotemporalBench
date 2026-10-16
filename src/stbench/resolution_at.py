"""The resolution a gold rests on (0.9.0, FD-decision-questions section 3.7): `rho`, how far one pixel of the camera
that saw a point moves it, which the gold row carries as `tolerances.resolution_m`.

One owner for the golds of the existing families (`questions_ego`, `questions_people`, `questions_where`,
`questions_task`): they name the keyframe whose camera saw the point and the point, and this reads the camera from the
delivered pose row (`World.camera_at`), the focal lengths from the robot's camera and `rules.resolution_m` for the
arithmetic. The plane the point lies on is the base the thing rests on (`base_z`: its placement's, a support's top or
the floor); a camera level with it (`rules.resolution_m` is None) reads nothing there, so the floor stands in, and a
camera level with the floor too is an error: no question rests on a frame that cannot place a point.

A row that asks nothing (an `unknown` gold) carries `UNKNOWN_RHO_M`, the kernel's floor: no scorer reads it.
"""

from __future__ import annotations

from collections.abc import Sequence

from .credit_kernel import MIN_RHO_M
from .questions_world import World
from .rules import resolution_m

UNKNOWN_RHO_M = MIN_RHO_M   # the resolution an `unknown` row carries: a gold with no point has no pixel to resolve
FLOOR_Z = 0.0               # the plane that stands in for a base the camera is level with


def at_frame(w: World, k: int, xy: Sequence[float], base_z: float | None) -> float:
    """The resolution, metres, of the floor point `xy` on the plane z = `base_z` (None: the floor) from the delivered
    camera at keyframe `k`: `rules.resolution_m`, at least `MIN_RHO_M`. ValueError when the camera is level with both
    the base and the floor."""
    cam = w.camera_at(k)[0]
    for z in dict.fromkeys(z for z in (base_z, FLOOR_Z) if z is not None):
        if (rho := resolution_m(cam, xy, z, w.cam.fx, w.cam.fy)) is not None:
            return rho
    raise ValueError(f"the camera at keyframe {k} (z {cam[2]:.3f}) is level with the base ({base_z}) and the floor: it "
                     f"resolves no point of the floor from there")


def base_of(w: World, entity: str, k: int) -> float | None:
    """The z the entity rests on at keyframe `k`: its placement's base (the floor's 0 or a support's top); None when it
    has no placement then."""
    p = w.tl.placement_of(entity, w.tl.keyframe(k).t)
    return None if p is None else float(p.pos[2])
