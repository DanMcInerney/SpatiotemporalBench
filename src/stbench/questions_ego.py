"""`ego_now` and `ego_mem` (during, asked at look points): where the nearest part of a named thing is from the robot's
camera, in metres ahead of it and to its left, along the floor (0.7.0, docs/families/ego_now.md and ego_mem.md).

Both answer with an `offset` (`ego_frame`: the delivered camera's ground frame, the base's heading) and are credited
`graded` (0.9.0) on the job's clearance C (`JobSpec.clearance_m`): the gold row carries C (`need_m`), the
resolution `rho` of the evidence sighting that tied the point (`resolution_m`: one pixel of its camera at the point) and
the harmful side `radial` (a thing placed farther from the camera than it is), through `rules.graded_tolerances`; and
`range_m`, the slant from the camera to the point asked, which the report splits by. **The sighting that sets `rho`** is
the one whose contact gave the surface tie (the least contact error, `nearest_part.contact_error` per frame), the
earliest of equals; the last evidence frame when the edge tie held instead; the plane is the base the thing rests on in
it (`resolution_at`). **The target is the nearest part of the thing, which the pictures show**: the thing's own instance
pixels, through the frame's depth, dropped onto the floor; the point of their convex hull (the **floor hull**) nearest
the camera's floor point at the look point (`nearest_part`, `World.nearest_part`), which may lie on a hull edge. The
thing's middle sits behind the face a ground camera sees, so a perfect reader of the pictures could not name it
(FD-0.7.0 section 2.2); the nearest visible point has no hidden extent, so there is no widening. **A question is asked
only where a pixel-perfect reading could reach the full rung** (FD-0.7.0 review, change 5): a camera-only reader can
place the point only if it is tied to a surface whose height it can read (the surface tie). Some evidence frame must
show a **contact** (a visible pixel where the thing meets the floor or its support's top, within C/2 of the point and
reading to within C/2 at one pixel: `nearest_part.contact_error`), or the thing must stand at the **edge** of a support
whose top is at or above the camera, within C/2 of its near edge (`ego_support.edge_tie`, which uses the set-back
screen). A thing hanging on a support's face, or whose nearest point overhangs its contact by more than C/2, has neither
and is not asked. A thing any of whose evidence pixels has no depth is not asked either: its hull would miss them. A
family module (`questions_text.FAMILY_MODULES`):
`SPECS`, `GOLDS`, `LOOKS` (both are asked at the look points the questionnaire draws, `JobSpec.looks`, never at a
decision) and `SLICED` (`ego_mem`: its point offers an answerable and an unknown pool, drawn as an end moment's are). A
gold is `(w, k) -> Moment | None`: the point's pools, keyed `str(k)`; `k` is any frame of the window.

What is named. A **carried** thing by its kind and describable attributes (its colour and size), the colour being a word
its render bears out; **clutter** by kind alone. For a job whose robot works on the floor (`Relevance.floor`: the vacuum
and the tote robot) only a thing **on the floor** (`World.on_floor`) at the sighting its gold rests on is asked, and an
unseen thing only when it stands on the floor at the point (a mug on a table is no obstacle to such a robot, and a pool
that took the floor things in one slice alone would let a thing's kind predict the slice); the helper and the drone ask
of every thing. People are `mover_next`'s, never named here, and a thing the robot's
own log had told about by the point (`World.told(k)`: the carried thing of a task begun by then, the blocker of a
decision at or before it, and no later one) never is: the log, not the picture, would answer. **Light furniture is
never asked** (VISION rule 5: a fact that is the same in every case of a place is asked only of places the agent has
never seen): `generate.light_start` draws its place within `SHIFT_M` (1.0 m; a chair `CHAIR_PULL_M`, 0.6 m) of the
place the public map gives it, so an agent that learned the layout knows it to well inside the credit's outer rungs (a
roll cage at the end of an aisle, the apartment's stool: `path_log` scored +0.20 on the AMR's `ego_now`). It still
counts as a thing a description can be mistaken for. A description is
fair only if nothing else in the robot's view could be taken for the thing (`_hidden`): another named thing of a kind it
can be mistaken for (`descriptions.lookalikes`, its own kind included) whose render shows its colour or sits next to a
boundary of it (`colour_seen.could_be`: black, grey and white are one group), or one of the same description whatever
its render (a table tells tracks apart by description alone). A kind-only name is hidden by any such thing. A pair that
`descriptions.CONFUSABLE_UNSEEN` names (a bin and a drum, a sack and a pallet) hides only when the other thing is not
plainly made out (not recognisable in frame `k`; for `ego_mem`, at no frame by `k`) or the described thing is itself
unseen: a drum plainly in view is told from a bin by its shape. Every screen reads kinds, renders and which
things were sighted, never a place or a gold.

**`ego_now`** (scope `now`, reference `kind_range`). Asked of a thing recognisable in frame `k`, the latest frame
(`World.recognised`; the gold cites no other frame, so it is answerable from the picture just sent). The gold is the
point of the floor hull of the thing's pixels in frame `k` nearest the camera's ground position at `k`, put in the
delivered camera's frame at `k` (`World.camera_at`, the mount included: a drone's camera hangs below its base);
`range_m` is the slant from that camera to the point. A carried thing is named only if the colour frame `k`'s
`rgb_med` shows confidently is its label and the frames up to `k` bear the word out (`World.colour_holds`, cut at `k`,
so no frame after the question decides it); an older root's visibility log has no render, so the label stands. Hidden
by what has a visibility row, at any pixel count, in frame `k`.
There is no class: the point offers the whole pool as its `answerable` one and an empty `unknown`.

**`ego_mem`** (scope `memory`). Asked of a thing the robot saw before the recent view (`look_windows.recent`: the last
`LOOK_WINDOW` frames of the current task up to `k`, which slides with the point) and does not see in it:
- **answerable**: it has a recognisable sighting before the recent view (the last, `j`), no row at any pixel count in
  the recent view, and every sighting of it from `j` to `k` lies within `UNCHANGED_M` (1 cm) of that place
  (`World.steady`: the placement has not changed from the last sighting to the ask), and its sightings at that place
  lie in one placement of the timeline (a thing moved by a centimetre and back is not asked: a table that reads
  centres could not tell). The gold is the point nearest the camera's ground position at `k` of the union of the floor
  hulls of **every recognisable sighting before the recent view at that place**, put in the camera frame at `k`: "as
  you last saw it", whatever has happened since (truth rule 2), and a side no sighting showed does not count; no frame
  after the recent view is read for it. The name is fair among the things **sighted so far in the window** `[k0, k]`
  (`World.sighted`; a twin the robot never saw cannot confuse it, and a twin first seen after `k` does not exist yet),
  by the screens above, the colours read from the frames up to `k` alone.
  `range_m` is the slant from the camera at `k` to the point (at the height of the visible point nearest it);
- **unknown**: a real thing with no pixel anywhere in the window (so none is met later in the case) that exists at `k`
  (and, for a floor robot's job, stands on the floor then) and is not hidden by any look-alike sighted in the window
  (`CONFUSABLE_UNSEEN` counts: nothing seen says which is meant), or an absent description
  (`questions_where.never_seen`, which screens `CONFUSABLE` and `CONFUSABLE_UNSEEN` window-wide) of a kind already
  sighted by `k`; either way dropped when a look-alike of the asked colour was glimpsed and never recognised by `k`
  (the S4 screen, `questions_where.glimpse_hides`: a glimpse's render may show the colour the label does not). The text
  is the answerable one's.

Both are drawn from the same kinds, so a kind predicts no slice. `Cand.answer_class` is the pool's (`answerable`,
`unknown`); `ego_now` declares none.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping

from . import colour_seen, grid
from .descriptions import desc_attrs, lookalikes, phrase
from .ego_frame import slant_m, to_offset
from .ego_support import contact_bases, edge_tie
from .nearest_part import UNCHANGED_M, surface_tie
from .questions_select import Cand, Moment
from .questions_where import glimpse_hides, never_seen
from .questions_world import World
from .records import FamilySpec
from .resolution_at import UNKNOWN_RHO_M, at_frame
from .rules import graded_tolerances
from .timeline import Entity

# The mobilities a look question can mistake one thing for (a person is `mover_next`'s), and those it asks about: not
# "light", whose place is a fact of the layout (module docstring).
NAMED = ("carried", "light", "clutter")
ASKED = ("carried", "clutter")
TAIL = " Answer in metres ahead of the camera and to its left."


def _render_now(slots: dict) -> str:
    return (f"Where is the point of the {phrase(slots['kind'], slots['attributes'])} you can see now that is nearest "
            f"to your camera, measured along the floor?{TAIL}")


def _render_mem(slots: dict) -> str:
    return (f"As you last saw it, where is the point of the {phrase(slots['kind'], slots['attributes'])} you saw that "
            f"is nearest to your camera now, measured along the floor?{TAIL}")


# -- names ------------------------------------------------------------------------------------------------------
def _slots(e: Entity) -> dict:
    """The question's name for an entity: a carried thing's kind and describable attributes, any other's kind alone."""
    return {"kind": e.kind, "attributes": desc_attrs(e.attributes) if e.mobility == "carried" else {}}


def _colour(e: Entity) -> str | None:
    """The colour word a carried thing is named by; None for a thing named by kind alone."""
    return e.attributes.get("color") if e.mobility == "carried" else None


def _classes(w: World, e: Entity, upto: int | None) -> tuple[frozenset[str], ...]:
    """The words each recognisable frame of `e` (at or before keyframe `upto`; all the window's when None) shows its
    colour in (`colour_seen.classify` of the frame's `rgb_med`); empty when nothing was measured: an older root's
    visibility log, a thing never recognisable, or a label outside the palette (`World.rendered`, cut at `upto`)."""
    if e.attributes.get("color") not in colour_seen.WORDS:
        return ()
    return tuple(colour_seen.classify(rgb) for j in w.recognised(e.id) if upto is None or j <= upto
                 if (rgb := w.vis[j][e.id].get("rgb_med")) is not None)


def _could_be(w: World, o: Entity, colour: object, upto: int | None) -> bool:
    """Whether a viewer of `o`'s frames up to `upto` could take it for a colour word (`World.may_be_taken_for`, cut)."""
    classes = _classes(w, o, upto)
    words = colour_seen.taken_for(classes) if classes else colour_seen.label_reads(o.attributes.get("color"))
    return colour_seen.could_be(words, colour)


def _held(w: World, e: Entity, upto: int) -> bool:
    """The frames up to `upto` bear out the entity's colour word (`World.colour_holds`, cut): the colour most of them
    confidently show it in is its label; true when nothing was measured."""
    classes = _classes(w, e, upto)
    return not classes or colour_seen.held(classes) == e.attributes.get("color")


def _hidden(w: World, e: Entity, others: Iterable[Entity], upto: int | None,
            plain: Callable[[Entity], bool] | None = None) -> bool:
    """Whether some other named thing among `others` could be taken for `e`'s description (the module docstring); their
    renders are read up to keyframe `upto` (None: the whole window). `plain(o)` says whether `o` is made out plainly
    enough that its shape tells it from `e` where `descriptions.CONFUSABLE_UNSEEN` names the pair (a seen drum is not a
    bin); for a `o` it is not true of, and for every `o` when `plain` is None (nothing seen answers the description, an
    unseen thing), those pairs count too."""
    seen, unseen, slots, colour = lookalikes(e.kind), lookalikes(e.kind, True), _slots(e), _colour(e)
    return any(o.id != e.id and o.kind in (seen if plain is not None and plain(o) else unseen)
               and (colour is None or _slots(o) == slots or _could_be(w, o, colour, upto)) for o in others)


def _sighted(w: World, upto: int | None = None) -> list[Entity]:
    """The named things with a sighting in the window (`World.sighted`), at or before keyframe `upto` when given: what a
    description must be unique among."""
    return [e for e in w.tl.entities if e.mobility in NAMED
            and any(upto is None or j <= upto for j in w.sighted(e.id))]


def _tolerances(w: World, rho: float) -> dict:
    """The `tolerances` of a row: the sighting's resolution `rho`, the job's clearance C as the need and the harmful
    side radial (`rules.graded_tolerances`), whatever the range or the kind."""
    return graded_tolerances(rho, w.job.clearance_m, "radial")


def _floor_ok(w: World, e: Entity, k: int, seen: int) -> bool:
    """Whether the thing may be asked at the look point `k`, its sighting at `seen` being what the gold rests on:
    always, except for a job whose robot works on the floor (`Relevance.floor` of the task at `k`), where the thing must
    stand on the floor it drives on (`World.on_floor`) at that keyframe."""
    return not w.relevance_of(w.task_index(k)).floor or w.on_floor(e.id, seen)


def _cand(family: str, e: Entity, w: World, k: int, evidence: list[int], answer_class: str | None) -> Cand | None:
    """The answerable candidate of the look point `k` about `e`, whose gold place is the point of the floor hulls of its
    pixels in the keyframes `evidence` (`World.nearest_part`) nearest the camera's ground position at `k`: the offset in
    the camera frame at `k`, the clearance C as the need, `rho` the resolution at the point from the camera that tied it
    (the module docstring), `range_m` the slant to that point. None when those frames hold no point of it, when any of
    its pixels there has no depth (the hull would miss them), or when neither surface tie holds (`nearest_part.
    surface_tie`: the contact tie, read from the camera that saw it, or the edge tie) unless the job keeps the family
    (`JobSpec.screen_fallback`) and some frame shows a contact: that contact's sighting then sets `rho`, which the need
    widens to (`credit_kernel`)."""
    cam, yaw = w.camera_at(k)
    if (part := w.nearest_part(e.id, evidence, cam[:2])) is None or any(w.gaps(e.id, j) for j in evidence):
        return None
    bases = contact_bases(w, e.id, evidence, part.xy)
    errors = [w.contact_error(e.id, [j], part.xy, [base]) for j, base in zip(evidence, bases, strict=True)]
    contact = min((x for x in errors if x is not None), default=None)   # the least error over the frames' pixels
    tie = surface_tie(contact, edge_tie(w, e.id, evidence, part.xy), w.job.clearance_m)
    if tie is None and (contact is None or family not in w.job.screen_fallback):
        return None
    i = len(evidence) - 1 if tie == "edge" else errors.index(contact)   # the sighting that tied the point
    rho = at_frame(w, evidence[i], part.xy, bases[i])
    ahead, left = to_offset(part.xy, cam[:2], yaw)
    gold = {"type": "offset", "ahead_m": round(ahead, 4), "left_m": round(left, 4)}
    return Cand(family, _slots(e), gold, "answerable", tuple(evidence), (e.id,) * len(evidence), _tolerances(w, rho),
                decision_k=k, subjects=(f"e:{e.id}",), answer_class=answer_class,
                range_m=round(slant_m(cam, (*part.xy, part.z)), 4))


# -- ego_now ----------------------------------------------------------------------------------------------------
def _frame_names(w: World, e: Entity, k: int) -> bool:
    """Frame `k` shows `e` in a colour that names its label confidently (`colour_seen.classify` of its `rgb_med` is
    exactly the label); a thing with no colour word, a label outside the palette or no render measured passes."""
    colour, rgb = _colour(e), w.vis[k][e.id].get("rgb_med")
    return colour not in colour_seen.WORDS or rgb is None or colour_seen.classify(rgb) == frozenset({colour})


def ego_now(w: World, k: int) -> Moment | None:
    """The look point `k`: every named thing recognisable in frame `k` that its description picks out among the things
    with a row in that frame (and, for a job whose robot works on the floor, that stands on the floor), in timeline
    order. None when there is none."""
    told, rows = w.told(k), w.vis.get(k, {})
    in_frame = [e for e in w.tl.entities if e.mobility in NAMED and e.id in rows]
    found = []
    for e in in_frame:
        if (e.mobility not in ASKED or e.id in told or k not in w.recognised(e.id) or not _floor_ok(w, e, k, k)
                or not _frame_names(w, e, k) or (_colour(e) is not None and not _held(w, e, k))
                or _hidden(w, e, in_frame, k, lambda o: k in w.recognised(o.id))):
            continue
        if (cand := _cand("ego_now", e, w, k, [k], None)) is not None:
            found.append(cand)
    return Moment(str(k), found, []) if found else None


# -- ego_mem ----------------------------------------------------------------------------------------------------
def _remembered(w: World, e: Entity, k: int, window: range, sighted: list[Entity]) -> Cand | None:
    """The answerable candidate about `e` at the look point `k`, or None when it is not one (the module docstring);
    `window` is the recent view at `k`, `sighted` the named things sighted by `k`."""
    seen = [j for j in w.recognised(e.id) if j < window.start]
    if not seen or w.in_ks(e.id, window):
        return None
    last = seen[-1]
    if not _floor_ok(w, e, k, last):
        return None
    xy = w.xy_at(e.id, last)
    if xy is None or not w.steady(e, last, xy, k, UNCHANGED_M):
        return None
    now = w.xy_at(e.id, k)   # a move nobody saw after the last sighting passes `steady`; the truth at `k` does not
    if now is None or math.dist(now, xy) >= UNCHANGED_M - grid.EPS:
        return None
    if _colour(e) is not None and not _held(w, e, k):
        return None
    if _hidden(w, e, sighted, k, lambda o: any(j <= k for j in w.recognised(o.id))):
        return None
    here = [j for j in seen if math.dist(w.xy_at(e.id, j), xy) < UNCHANGED_M - grid.EPS]   # at its current place
    if len({(p.t0, p.t1) for j in here if (p := w.tl.placement_of(e.id, w.tl.keyframe(j).t)) is not None}) > 1:
        return None   # more than one placement: the table, which reads centres, and the gold would not agree
    return _cand("ego_mem", e, w, k, here, "answerable")


def _unknown(w: World, k: int, slots: Mapping, subject: str) -> Cand:
    """An unknown candidate at the look point `k`: no place, no evidence. Its tolerances are the job's with the floor
    resolution (`UNKNOWN_RHO_M`), which no scorer reads on an `unknown` gold."""
    return Cand("ego_mem", dict(slots), {"type": "unknown"}, "unknown", (), (), _tolerances(w, UNKNOWN_RHO_M),
                decision_k=k, subjects=(subject,), answer_class="unknown")


def _unknown_pool(w: World, k: int, sighted_ever: list[Entity], sighted_now: list[Entity]) -> list[Cand]:
    """The unknown candidates of the look point `k`: the named real things no frame of the window shows that exist at
    `k` (and, for a job whose robot works on the floor, stand on the floor then, as the answerable pool's do) and no
    sighted look-alike hides, then the absent descriptions of a kind sighted by `k`, one per description; a candidate
    whose colour a look-alike glimpse showed by `k` is left out (`questions_where.glimpse_hides`)."""
    t, told = w.tl.keyframe(k).t, w.told(k)
    out = []
    for e in w.tl.entities:
        if (e.mobility in ASKED and e.id not in told and not w.in_window(e.id)
                and w.tl.placement_of(e.id, t) is not None and _floor_ok(w, e, k, k)
                and not _hidden(w, e, sighted_ever, None)
                and not glimpse_hides(w, e.kind, _slots(e)["attributes"], e.id, k)):
            out.append(_unknown(w, k, _slots(e), f"e:{e.id}"))
    kinds = {e.kind for e in sighted_now}
    for kind, attrs in dict.fromkeys(key for i, _, _ in w.tasks for key in never_seen(w, w.relevance_of(i))):
        if kind in kinds and not glimpse_hides(w, kind, dict(attrs), None, k):
            name = f"d:{kind}/{json.dumps(dict(attrs), sort_keys=True, separators=(',', ':'))}"
            out.append(_unknown(w, k, {"kind": kind, "attributes": dict(attrs)}, name))
    return out


def ego_mem(w: World, k: int) -> Moment | None:
    """The look point `k`: the named things the robot saw before the recent view and does not see in it (answerable),
    and the things it cannot have seen (unknown), both pools in timeline order. None when nothing is answerable: the
    draw needs both pools."""
    window = w.recent(k)
    sighted_now, told = _sighted(w, k), w.told(k)
    answerable = [c for e in w.tl.entities if e.mobility in ASKED and e.id not in told
                  and (c := _remembered(w, e, k, window, sighted_now)) is not None]
    if not answerable:
        return None
    return Moment(str(k), answerable, _unknown_pool(w, k, _sighted(w), sighted_now))


SPECS = (FamilySpec("ego_now", "during", "offset", "graded", ("see", "pose"), "now", "kind_range", False,
                    ("kind", "attributes"), _render_now),
         FamilySpec("ego_mem", "during", "offset", "graded", ("see", "pose"), "memory", None, False,
                    ("kind", "attributes"), _render_mem))
GOLDS = {"ego_now": ego_now, "ego_mem": ego_mem}
LOOKS = frozenset({"ego_now", "ego_mem"})
SLICED = frozenset({"ego_mem"})
