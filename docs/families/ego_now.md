# `ego_now`

"Where is the point of the {thing} you can see now that is nearest to your camera, measured along the floor? Answer in
metres ahead of the camera and to its left." (0.7.0, ability *distance*): the offset, in metres ahead of the camera and
to its left along the floor, of the nearest visible part of a named thing the robot has just seen. During, `now`. Code:
`stbench.questions_ego` (shared with [`ego_mem`](ego_mem.md)), `stbench.nearest_part` (the geometry, the one owner),
`stbench.pixels` (the reading of the pictures, shared with the oracle channel), `stbench.ego_frame` (the frame), `stbench.rules`
(`resolution_m`, `graded_tolerances`). The record and questionnaire rules are in [questions.md](../questions.md). 0.6.0 asked for the
thing's middle with relative rungs (`relative_ladder`, `widening`); both are gone from the family.

**Who asks it (0.11.0).** A look point draws its family from a table each job declares
(`JobSpec.look_shares`, [questions.md](../questions.md), "Look points"). `ego_now` is the AMR's third slot at 50%
beside [`approach_mem`](approach_mem.md) at 50% (the whole slot, 100%, in 0.10.0), and the helper's third slot at 20%
beside [`open_heading`](open_heading.md) at 15% and `approach_mem` at 30%. Through 0.9.0 it was the AMR's first slot at
70% and the helper's at 30%. It left the **drone** (nothing it names can be hit from 15 m; the corridor
takes its slot) and the **vacuum** (16 rows in 0.7.0, under the 20-question line in any table that holds the budget). The
family's rules below are unchanged, except that a floor robot is asked only about things on the floor (below) and that a
look point is any frame.

| | |
| --- | --- |
| Category | C3 Locate |
| Asked | during, at a look point: any frame at least 4 steps into its task but not its last (`JobSpec.looks` and `look_shares`, drawn by the questionnaire; never at a decision point), by the AMR and the helper only. At most one candidate a point; none is a dead band. Only where the gold point is tied to a surface (a contact pixel within C/2 of it that one pixel reads to within C/2, or the near edge of a high support; below) |
| Text (slots) | "Where is the point of the {thing} you can see now that is nearest to your camera, measured along the floor? Answer in metres ahead of the camera and to its left." (`kind`, `attributes`) |
| Answer, credit | `offset` (`{"type": "offset", "ahead_m", "left_m"}`), `graded` on the 2-D error `hypot(d_ahead, d_left)` against the job's clearance C (`JobSpec.clearance_m`), or a tenth of the gold's distance where that is more (0.11.0): the four quarters, with the resolution `rho` one pixel at the evidence sighting's camera to the gold point, the harmful side a thing placed farther from the camera than it is (`harm` `radial`: the need is halved) and the direction quarter a bearing within 45 degrees of the gold's or `e <= C` |
| Scope, `needs` | `now`; see, pose. The camera alone answers it; the gold reads the delivered pose row, the log of the window, and the scorer's depth and instance masks (`private/<sid>/`), which no system under test is sent |
| Reference | `kind_range` (A6): the true bearing to the nearest part at the median floor range of any other script's gold of the kind, `unknown` for a kind none has. Declared, not a gated shortcut; distance is reported above the larger of it and `all_unknown` per job |
| Gold function | `questions_ego.ego_now(w, k) -> Moment \| None`: the point's whole pool as `answerable`, an empty `unknown`, key `str(k)` |
| Class | none (`answer_class` null). Range bands (`ego_frame.band`: near under 6 units, far from 16) are a report, on `range_m` |

## The target: the nearest part of what the frame shows

The question asks for the point of the thing nearest the camera along the floor, because that is what avoidance acts on
and what a ground camera sees: the middle of a drum sits behind the face the camera shows, so under a job's tight
clearance a perfect reader of the pictures could not name it.
1. The thing's **visible points** in frame `k`: the pixels of its own instance in the frame's instance image,
   unprojected through that frame's depth at the pixel centres and moved to the map by the pose row's camera
   (`nearest_part.visible_points`, as `grid.points` does for the free-space rule; a pixel of depth 0 adds no point).
2. Their **floor hull**: the points dropped onto the floor as (x, y) and their convex hull, counter-clockwise, every
   vertex to the millimetre (`nearest_part.floor_hull`).
3. The gold is the **point of the filled hull nearest the camera's floor point** at `k` (`nearest_part.nearest_on_hull`),
   which lies on a hull edge as often as at a vertex, and is the floor point itself, an offset of (0, 0), when the camera
   is over the thing (a drone lowering). It is put in the delivered camera's frame (below).

Along the floor: on the drone the 3-D-nearest and floor-nearest points differ by more than 0.3 m in half the golds.
`range_m` is the 3-D slant from the camera to the point, at the height of the visible point nearest it, for the report.
A pixel-perfect reader of the instance image and depth reproduces the gold exactly (`tests/test_nearest_part.py`; on
release-c 412 of 412 offset rows). No hidden extent is asked, so there is no widening.

## Resolvability: the surface tie

A question is asked only where a camera-only reader, with **no depth**, could place the point (methodology rule 5; the dev
pilot's abstention review). Such a reader places a point by where its pixel ray meets a
surface whose height it can read, so the gold point must be **tied to a surface**. At least one evidence frame the gold
uses must give one of two ties.

1. **The contact tie.** The frame shows a visible pixel where the thing meets the surface it rests on, within C/2 along
   the floor of the gold point.
   - The surface is the floor (z = 0) or the support's top: the placement's base z, whether a slab, a shelf or a table.
   - A **contact pixel** is a visible point of the thing whose z is within `max(2 cm, slant / fy)` of the base z, where
     slant / fy is one pixel's height at that point's range (`nearest_part.CONTACT_M`, `nearest_part.contact_error`).
   - It must also read there to within C/2 at one pixel: `e = (d^2 + H^2) / (H * fy)` (`contact_pixel_error`), d the
     floor distance from that frame's camera to the contact point, H = |camera z - base z|, fy in pixels. A camera
     within 2 cm of the base plane reads nothing.
   - The error of the gold point is the smallest over the contact pixels and the sightings, each from its own camera.
2. **The edge tie** (`ego_support.edge_tie`). The thing rests on top of a support (`Placement.on` names it and the gold
   point lies inside its footprint, within 1 cm) whose top is at or above that sighting's camera, so the contact is
   hidden from below, and the set-back screen (below) says it is not set back: it stands within C/2 of the support's near
   edge along the sight line.

`nearest_part.surface_tie` combines them: contact first, else edge, else the question is dropped.

**Never tied.**
- A thing hanging on a support's face, `on` set but the gold point outside the footprint (a coat on a wardrobe's side):
  it has no base (`ego_support.contact_bases`) and no edge tie.
- A thing whose nearest point overhangs its contact by more than C/2 with no edge tie (a floor lamp's shade, its base
  hidden or far).
- A contact too far for one pixel to read: for the 8 cm vacuum camera on the floor, one pixel reaches C/2 out to about
  1.2 m and C out to about 1.7 m.

The rule is the same for every job and reads only the cameras, the poses, the points, the placements' bases and
supports, and C. A thing any of whose evidence pixels has no depth is not asked (the hull is of the pixels that have
one).

Earlier rules in the same file: the first read each point at its true height above or below the camera and took the
smallest error over the points tied for nearest, which a reader with no depth cannot do (floor lamps' shades and coats
on a wardrobe passed); the second asked only a floor pixel (z <= 2 cm), which dropped every thing on a rack, a shelf, a
table or a 0.15 m slab, and read a drone pixel (7 to 21 cm tall at 25 to 80 m) against a fixed 2 cm.

**The fallback**, declared with the screen: a job names in `JobSpec.screen_fallback` the families whose supply fell
under `abilities.MIN_ABILITY_N` (20 questions) on a dataset; for those the dropped questions that have a contact pixel
(too far to read) are kept with the rungs `max(C m, 2 m e)` for m in 0.5, 1 and 2, e the contact tie's one-pixel error
above (`nearest_part.fallback_ladder`), so the full rung is what one pixel could miss. A thing with no contact pixel has
no error and stays dropped; one the edge tie asks needs no fallback. The field is empty on every job until a dataset's
supply asks for it.

## Raised things: the set-back screen and the edge tie

A thing resting on a surface (a table, a desk, a rack shelf) whose top is at or above the camera is seen from below: the
camera sees its nearest face and the surface's front edge, and the top is edge-on or hidden, so how far back from the
edge the thing stands is not in the frame, and the contact with the surface is hidden too (the dev pilot's log review: AMR
case 5, a box 0.23 m back on a 0.9 m packing table seen from the 0.5 m camera; the vacuum's picture frame on a desk).
- A thing whose placement names a supporting entity (`Placement.on`) with its base at or above the camera is **set back**
  when its gold point lies more than C/2 behind the support's near edge (`ego_support.set_back`): measured along the
  sight line from the camera's floor point to the gold point into the support's true footprint (`Timeline.footprint`),
  so the whole near edge counts and not only its point nearest the camera. A gold point outside the support's footprint
  (a thing mounted on a face) is not behind it.
- The **edge tie** (`ego_support.edge_tie`) asks such a thing when it is on top of the support (inside the footprint),
  the support is at or above the camera of a sighting the gold uses, and it is not set back from that sighting's camera:
  at the edge, the edge itself is the tie. A set-back one is not asked unless a contact pixel at the support's top ties
  it (it rarely shows one from below).
- A thing on a surface below the camera, which looks down on it, is tied by the contact tie at the surface top.

The screens read the timeline and the cameras, never a gold or a score.

## The frame

An offset is where a point of the map is from the camera, along the floor: **ahead** along the base's heading and then
**left**, both in metres, whatever the height of the camera or the point (`ego_frame.to_offset`; left is positive, a
thing behind the camera has a negative `ahead_m`). The camera is the delivered one (`World.camera_at`: the pose row's
`camera.pos`, the robot's mount turned by the heading and added to the base, so the vacuum's is 0.13 m ahead of its base
and a drone's hangs below it); the heading is the base's `yaw_deg`, which the camera shares.

## The credit

It is the job's, not the range's: a margin a controller keeps is its own where harm happens. The need is C, the margin
the job's avoidance and approach keep (`JobSpec.clearance_m`), and since 0.11.0 a tenth of the gold's distance from the
camera where that is more (`credit_kernel.KAPPA`), so it widens only beyond ten times C. The row's tolerances
(`rules.graded_tolerances(rho, C, "radial")`) hold `resolution_m`, `need_m` and `harm`; `rho` is `rules.resolution_m` at
the camera of the evidence sighting that set the surface tie (the last evidence frame when no contact did), to the gold
point: one pixel radially, `(d^2 + h^2) / (h fy)`, and laterally, `sqrt(d^2 + h^2) / fx`, combined as their hypotenuse,
at least 1 cm. Within `rho` the answer earns the whole credit, at the need three quarters, at twice it a half, and at
four times it a quarter if the bearing is right ([questions.md](../questions.md), "Credit"). The slant (`range_m`) stays
on the row for the reports and sets no credit. On the 0.11.0 release the median `rho / N` of the helper's `ego_now` is
0.07 (q90 0.15) and the AMR's 0.20 (q90 0.37), so the whole credit is reachable at the camera's own resolution.

## What may be named

The certificate is **recognisable in frame `k`** (`World.recognised`: the pixel floor `recognise.RECOG_PX`, light, a box
clear of the image's edge, a depth); no other frame stands in for it, so the question is answerable from the picture
just sent.
- A **carried** thing is named by its kind and describable attributes (colour, size). Its colour word must be borne out:
  the colour frame `k`'s `rgb_med` confidently shows (`colour_seen.classify` names exactly one word) is its label, and
  so is the colour most of its recognisable frames up to `k` show (`World.colour_holds`, cut at `k`: no frame after the
  look decides a question). An older root's visibility log has no render, so the label stands.
- **Clutter** is named by its kind alone ("the floor fan").
- **A floor robot is asked only about things on the floor** (0.10.0; `questions_ego._floor_ok`). For a job whose robot works
  the floor (`Relevance.floor`: the tote robot here, the vacuum for `ego_mem`) only a thing that stands on the floor it
  drives on (`World.on_floor`) at the sighting its gold rests on is asked. A pallet's stock on a rack or a mug on a table is
  no obstacle to it. This replaced a filter by `JobSpec.relevance`, which emptied the cells (review S3). The filter cut the AMR's
  `ego_now` by 26%. The helper and the drone ask of every thing.
- **Light furniture is never asked** (methodology rule 5, lane A3b): `generate.light_start` draws its place within 1.0 m (a
  chair 0.6 m) of the place the public map gives it, so it is a fact of the layout that an agent which learned the layout
  knows. It still counts as a thing a description can be mistaken for.
- People are [`mover_next`](../questions.md)'s, never named here; **told things** (`World.told(upto=k)`: the carried thing
  of a task begun by the look point, the blocker of a decision at or before it) never are.
- **Hidden** when another named thing with a visibility row in frame `k` (at any pixel count) could be taken for it: a
  kind it can be mistaken for (`descriptions.CONFUSABLE`, its own kind included) whose render shows the colour or sits
  next to a boundary of it (black, grey and white are one group), or one of the same description whatever its render
  (a table tells tracks apart by description alone). A kind-only name is hidden by any such thing. A bin and a drum, a
  sack and a pallet (`CONFUSABLE_UNSEEN`) hide each other only when the other is not plainly made out (not recognisable in
  frame `k`): a drum plainly in view is told from a bin by its shape.

The screens read kinds, renders and which things have a row in the frame, never a place or a gold.

## Answerers

- **Tables** (`table_ego`): the described thing's unique sighting at the ask step, its `floor_hull` (the oracle channel
  sends it for a recognisable detection, [format.md](../format.md)) and `nearest_part.nearest_on_hull` from
  that step's camera, in that frame. None, or two (a twin detected beside it), or a sighting with no hull, is `unknown`.
  The oracle and `memory_table` read the same hulls the gold takes and reproduce it; `current_view` is the same here
  (the thing is in the latest frame).
- **`kind_range`** (`baselines_kind_range`): the bearing `current_view` gives to the nearest part, at the median floor
  range of the kind over other scripts' golds, which are nearest-part ranges. Its score is a reference the family is
  reported above, not a floor that holds everywhere.
- **`path_log`** (`baselines_path_ego`): the prior's `ego_now` golds of the kind are put back on the map (each record's
  base, heading and the robot's camera mount); a dense place among them (the golds within their last rung, 2C of the
  job's clearance) that is in view at the ask step alone is a candidate. A place is judged on the records whose own
  camera had it in view at their ask (the record's `cameras`), by the point of its cluster nearest each record's camera
  with that record's script left out, and kept only where the held-out gain over abstaining has a bootstrap interval
  over the scripts above 0; so a kind that stands at one spot in some scripts and another in the rest keeps both. The
  answer is the point of that cluster nearest the ask's camera (the nearest point of the footprint the golds have
  shown), in the camera frame of the ask. Else the pool's learned answer, kept only where its held-out gain over
  abstaining has a bootstrap interval over the scripts above 0, else `unknown`.

## Gates and risks

`ceiling`, `knowable`, `shortcuts` (`blind`, `path_log`, `other_case`, `questionnaire_blind`, `form` within 0.10 of the
floor; `kind_range` is excluded) and `separation` from the best of the shortcuts and `kind_range`.
No balance is needed: the best constant offset scores far under 0. The risk is a learned typical range per kind, which
`kind_range` measures; the bearing is nearly free once the thing is found in the frame, so what the family adds is
range.

## Supply

On release-c (24 short and medium cases a job; C = 0.10 vacuum, 0.15 helper, 0.45 AMR, 2.4 drone) 0.6.0 asked 31, 66, 40
and 63 offset rows; with the nearest-part gold the screen leaves 18, 54, 35 and 63 (it drops 42%, 18%, 12% and 0%), and
nothing else changes the count. The vacuum was under the 20-question line (`abilities.MIN_ABILITY_N`) and the fallback
was what lifted it; it no longer asks the family.

On the dev read (144 cases, 0.7.0) the surface tie takes the answerable `ego_now` rows from 37 to 48 on the AMR (the
questionnaire redraws among what is left, and a contact read at a rack's top reaches rows the older error dropped), 78
to 74 on the helper, 33 to 20 on the vacuum and 52 to 42 on the drone. Of the AMR's 48 asked rows the contact tie alone
admits 29, the edge tie alone 14 and either 5; the helper's, the drone's and the vacuum's are all contact. What the rule
drops is a floor lamp or a coat on the vacuum, a drone thing whose contact is not in view, and a few helper things whose
contact is hidden.

In 0.10.0 (the dev root) the AMR's third slot asked 105 questions over all lengths (86 in the short and medium cases, a
fill of 0.101 a drawn slot) and the helper's 55 (46; a fill of 0.342). In the 0.11.0 release the family has 65 questions
on the AMR (53 in the short and medium cases), where `approach_mem` now takes half of the slot, and 49 on the helper
(41), all of them `now`.
