# Map spec

An engine-neutral description of one map, indoor or outdoor. The generator, the planner and the Blender builder all
read this file; every episode names it (`episode.map`). Python
reader: `stbench.mapspec.load_map`. Self-check: `stbench.mapspec.check_map`.

**Frame.** Metres. x east, y north, z up. Yaw is counter-clockwise from +x, in radians. The floor is z = 0.

**Top level.** `name`, `bounds [xmin, ymin, xmax, ymax]`, `wall_thickness` (0.1; the warehouse 0.2),
`wall_height` (2.5; the warehouse 6.0; the city's 45.0), `regions`, `walls`, `openings`, `windows`, `furniture`,
`ceiling_lights`, and three optional keys:
- `catalog`: the name of the vocabulary the map's items and movers are drawn from (`stbench.catalog.CATALOGS`: `home`,
  `warehouse` or `downtown`). A map without it is `home`, so `apartment_a.json` has no such key.
  `mapspec.check_map` reports a name that is not a catalog.
- `depth_unit`: kept for the older releases' maps, whose depth PNGs it scaled; since 0.4.0 the robot's `DepthSpec`
  owns a script's depth unit. In those releases it was the unit of the episode's depth PNGs, `"mm"` or `"cm"` (`mapspec.MAP_DEPTH_UNITS`), written to
  `sensors/camera.json`, which every reader scales depth by (a camera file may also say `"m"`, which a map may not: a
  step of a metre rounds a depth by up to 0.5 m, beyond the 0.1 m tolerance of the in-view test). A map without it is
  `"mm"`, as the apartment and the warehouse are. A depth image is 16 bits, so it holds 65,535 steps of its unit (mm:
  65.535 m, cm: 655.35 m), and `check_map` reports a map whose 3D diagonal (its `bounds` and `wall_height`) is longer
  than that, or a unit that is neither.
- `outdoor`: `{sun_elevation_deg, sun_rotation_deg, sun_strength, sky_strength}`, the sun (its elevation above the
  horizon and its rotation about the vertical, in degrees, and its strength in W/m2) and the strength of the sky. A map
  without it is indoors: it has ceilings at the regions' `z1` and its lights are its `ceiling_lights`. An outdoor map
  has no ceilings, so `check_map` reports ceiling lights in one.

## regions

Rooms. `{id, name, polygon, anchor, level, z0, z1, kind}`. `polygon` is the floor outline on the inner faces of the
walls (counter-clockwise, convex here). `anchor` is a standing point in the room (the route's first viewpoint of a tour
is the free cell nearest it, and region 0's anchor starts the reachability search, so it must be a free cell). `level`
is the storey (0), `z0`/`z1` the floor and ceiling heights. Names are unique per map. Ids (`R1`...) are truth and never
reach the agent. Regions may share an edge with no wall between them (an open-plan hall, the warehouse): the planner
joins them. `kind` is optional: the room's kind word (`corridor`, `bedroom`), which text uses to name a room only where
the kind is unique in the layout. A region without it has its `name` as its kind (`mapspec.Region.kind`), so
`apartment_a.json` has none.

## walls

`{id, a, b, thickness, height, exterior, facade}`. A wall is a box centred on the segment a-b (length |ab|, width
`thickness`, height `height`), standing on the floor. Extend each wall by `thickness / 2` at both ends so corners close.
Openings are cut out of walls (below). `facade` is optional: the name of the style the renderer draws the wall in (a
building's face: brick, stone, glass, concrete, stucco); a wall without it is the plain wall.

## openings

Doors and arches cut through a wall. `{id, type, wall, at, width, height, ...}`:
- `wall` is a wall id; `at` is the distance from the wall's `a` end to the centre of the opening along the wall; the
  opening spans `at ± width/2` for the full wall thickness and for `height` from the floor (a gap in the wall box).
- `type` is `door` (has a leaf and an `open` state) or `arch` (a doorless gap; no entity).
- `between` names the two regions joined (`outside` for the entrance).
- `interior` is false only for the entrance, which stays closed. In M1 every interior door is open throughout.

Door leaf (doors only). Let u = (b - a) / |b - a| be the wall direction and n = (-u_y, u_x) its left normal.
- `hinge`: `a` puts the hinge at the jamb nearer the wall's `a` end, `b` at the other jamb.
- `swing`: `left` swings the open leaf toward +n, `right` toward -n.
- Closed: the leaf lies along the wall line, from the hinge jamb to the other jamb (thickness about 0.04 m).
- Open: the leaf stands perpendicular to the wall, from the hinge jamb out to `width` along the swing side (90 degrees).
  The renderer draws a quarter-turn about the vertical hinge axis between the two.
- The region on the swing side is the room the door swings into (`MapSpec.openings[i].swings_into`). Furniture keeps
  `width/2 + 0.3` m clear of every interior opening centre, so the swing and the passage stay free.

## windows

On exterior walls only. `{id, wall, at, width, sill, height}`: same `at` convention; the glazed gap spans `sill` to
`sill + height` above the floor. Windows never overlap openings.

## furniture

Fixed (`mobility: "fixed"`) and light (`"light"`: chairs, stools, the laundry basket; the warehouse's pallet jacks and
roll cage, whose boxes are their procedural builders' `DESIGN_BOX`) pieces.
`{id, name, kind, mobility, center, size, yaw, color, blocks, surfaces, under_m}`:
- `size` is `[depth, width, height]`: depth along the facing direction, width across it.
- `yaw` is the direction the **front** faces; the footprint is a rectangle centred on `center`, `depth` along that
  direction and `width` across, standing on the floor. The back is at local -x, the left at local +y.
- `blocks` is false for flat floor coverings (rugs, mats): they obstruct neither the ego nor furniture.
- `under_m` is optional: the height of the piece's open underside above the floor (a table's apron, a sofa's skirt),
  0 for a piece solid to the floor; a map without it has 0, and `check_map` requires `0 <= under_m < size[2]`. A piece
  with `under_m > 0` stands on four corner squares of side `geometry.LEG_SQUARE_M` (0.15 m), which obstruct at every
  height; the rest of its footprint obstructs only from `under_m` up, so a robot that fits under it
  (`robots.passes_under`) may drive beneath (`geometry.under_obstacles`).
- `surfaces` are the receptacles: `{id, kind, rect: [cx, cy, sx, sy], z}`. `rect` is in the furniture's local frame
  (origin at its centre, x along the facing direction), `z` is the absolute height of the surface. `kind` is the
  receptacle kind name (`kitchen counter`, `dining table`, `desk`, `nightstand`, `shelf`, `coffee table`,
  `sink edge`, `sofa`, `bed`, `tv console`, `hall console`, `shoe rack`, `floor mat`, `rug`; the warehouse adds
  `rack`, `staging lane` and `work table`). A shelf has one surface per level. Carried items rest on a surface with
  their base centre at `(x, y, z)`.
- Only fixed furniture has surfaces. Light furniture is movable; its `center` and `yaw` are the initial pose.
- A light piece's `size` follows its asset (M2's map revision; the stool's colour changed later):
  the renderer fits the piece's mesh to this box, and the footprint, the route's clearances and the in-view cover all
  read it. The chairs and the stool are fitted by height (`uniform`), so `size` is the asset's own proportions at the
  piece's height: chair 0.533 x 0.401 x 0.9 (a 0.9 m dining chair, 0.533 deep, tucked 2 cm from the table end) and stool
  0.385 x 0.406 x 0.579 (the asset at its scanned size). The laundry basket is built to its box, 0.4 x 0.4 x 0.5. A
  chair's `color` is its upholstery (beige or black): its leather is tinted. The stool's `color` is "brown" and the
  laundry basket's "beige": each asset is tinted to its colour (`tint.mode` `luminance` in the
  manifest), so the colour attribute is what the pixels show; "oak" is a material word and names the wood texture of
  fixed furniture only. `tests/test_mapspec.py` holds the boxes to the assets' measured boxes (`native_bbox_m` in the
  asset manifest) to 1 mm.

## ceiling_lights

`{region, pos, watts}`: a point light fixture at `pos` (z about 2.45; the warehouse's high bays 5.8) in the region.
`watts` is optional and defaults to 45, the apartment's value, so `apartment_a.json` has none; the warehouse gives 270,
about six times the power for a fixture 2.4 times as high (the same illuminance on the floor). Changing it changes every
render of the map, so it is settled before any episode of the map is generated: 270 W was kept after a sweep of 135, 270
and 540 W on the render's dev frames (Release 0.2, lane B). An outdoor map (`outdoor`
above) has no ceiling lights: its light is the sun and the sky.

## Planar facts the planner relies on

- The ego stands on cells at least 0.25 m from every wall box and every blocking furniture footprint.
- Openings are passable regardless of door state.

## Private maps

The held-out layouts (`stbench.map_files.HELD_OUT`: `home_large`, `cross_dock`, `courtyards`) are never in this folder,
the wheel or the repository. A held-out map lives in the held-out dataset root (its own root, `split: heldout` in its
index, never published), at
`<HELDOUT>/private/maps/<name>.json`, and every command finds it through one variable, `STBENCH_MAPS`: the folders of
private maps, separated by `;` on Windows and `:` elsewhere. `mapspec.load_map` and the renderer look there before the
bundled maps, and a subprocess or Blender inherits the variable, so every step that loads a map by the name its
timeline records (render, perceive, compile, validate, realism, check, eval) finds it with no flag:

```sh
STBENCH_MAPS=<HELDOUT>/private/maps stbench build --map home_large ... --out <HELDOUT>
STBENCH_MAPS=<HELDOUT>/private/maps stbench check --data <HELDOUT> ...   # and every later command on that root
```

Set it per command, as above, never exported in a shell and never for `stbench run --local`: a `--local` agent inherits
the shell's environment (`docs/agent.md`, "Not isolated"), so the variable would hand it the held-out maps' folder.

**One name, one map.** A name found in two places (a private folder and this one, or two private folders) is refused,
and so is a `STBENCH_MAPS` entry that is not a folder (a typo must not fall back to the bundled maps). A map loaded by
name must carry that name in its own `name` key. `--map` also takes a path, but nothing after `generate` could load a
map by path again, so a private map is always loaded by name. A map is named once, before its look-dev: the home decor
is seeded by the name. A held-out map is never committed, never copied into `src/stbench/maps`, and never used as a
test fixture or an example (`tests/test_map_files.py` fails if the package or a tracked file holds one).

## apartment_a

`catalog` home (the file has no such key). A 9.0 x 6.2 m one-bedroom flat (about 56 m2): a hall (R1, x 4.65-6.15, the
full depth) with the entrance D0 on its south wall, and off it the kitchen (R3, door D1), the living room (R2, door D2),
the bathroom (R5, D3) and the bedroom (R4, D4); the kitchen opens into the living room by the arch O1. Fixed furniture
is `F` and light furniture (four chairs, a stool, the laundry basket) `L`.

**The file is frozen for M1 to M3** (M0.2d made the one edit the redesign needs): a second edit would stale every
rendered development script. That edit:
- **A robot-vacuum dock**, `F23` (`dock`, 0.12 x 0.28 x 0.10 m, black, solid from the floor): against the hall's west
  wall (its face at x 4.65) at (4.71, 5.05), facing east into the hall. Its apron, the floor the vacuum needs to
  undock and to find the dock again (`mapspec.DOCK_APRON_FRONT_M` 1.2 m in front, `DOCK_APRON_SIDE_M` 0.5 m each side),
  is x 4.65-5.97 and y 4.41-5.69: inside the hall, clear of every blocking piece and of the swing and the band of every
  door (it stays 0.06 m below the band of D2, which ends at y 4.35).
- **The hall console `F20`** moved from (4.8, 5.5) to the north wall, (5.4, 6.0), facing south, which clears the apron.
- **A stacked washer-dryer**, `F08` (`washer_dryer`, 0.6 x 0.6 x 1.7 m, white, solid): in the kitchen's north-west
  corner at (0.35, 2.4), at the end of counter `F06`'s run, facing east. The **stool `L05`** moved from (0.5, 2.5) to
  (1.6, 2.52), out of the washer-dryer's door.
- **`under_m`** on the eleven pieces that stand on legs, as `render/furniture.py` draws them (a ray-cast test,
  `tests/test_furniture_under_blender.py`, holds the builders to it): the dining table `F07` 0.64 (the apron), the
  coffee table `F11` 0.10 (its shelf), the sofa `F10` 0.08 (its frame), the bed `F30` 0.05 (its base), the hall console
  `F20` 0.64 (its drawer), the desk `F33` 0.60 (its drawer apron, on four legs) and the chairs `L01` to `L04` 0.42 and
  the stool `L05` 0.529 (their seats). Every other piece is solid to the floor, `under_m` 0. The vacuum
  (`robots.passes_under`: an underside of 0.10 m or more) drives under exactly `F07`, `F11`, `F20`, `F33` and `L01` to
  `L05`; the sofa and the bed it cannot enter.

## warehouse_a

`catalog: warehouse`; the design is the warehouse plan (§1). A 17.4 x 19.8 m hall (345 m2), x east, y north, walls
0.2 m thick and 6.0 m high, no doors and no windows. Four regions share their edges, with no wall between them:
- `dock` (R1, x 0.1-17.5, y 0.1-9.1): four dock doors (closed roller-shutter panels, `dock_door`, fixed furniture 0.1 m
  deep against the south wall, no surface), four staging lanes (2.8 x 1.4 m floor markings 5 mm high, `blocks: true`, one
  `staging lane` surface each), two packing tables (`work table`), two shelving units on the west wall (three `shelf`
  surfaces each), the AMR's charger and a bay in front of each of the four stations (below), and, south of the cross
  band (y < 3.4), two pallet jacks and a roll cage (light furniture). The cross band (y 3.4-9.1) is free of furniture
  for the forklift. The lanes of doors 1 and 4 (`F05`, `F08`) stand off their doors' axes (x 4.5 and 12.9, where the
  doors are at 3.7 and 13.9), and the roll cage between `F07` and `F08` (11.7, 1.3): see "The AMR's pieces".
- `west aisle`, `middle aisle`, `east aisle` (R2-R4, x 0.1-5.9, 5.9-11.7, 11.7-17.5, y 9.1-19.9), each of **`kind`
  `aisle`** (the dock's kind is its name): each a 3.6 m lane between two single-deep pallet-rack rows (`pallet_rack`,
  1.1 x 10.8 x 3.9 m, front towards the lane, four 2.7 m bays, three levels at z 0.0, 1.3 and 2.6) and a dead end at the
  north wall. Back-to-back rows of neighbouring aisles touch. A job and the render read the aisles by this kind, never
  by name, so another warehouse's aisles need no row of their own (`render/look_warehouse.py` and the manifest's
  `floors` are keyed `dock` and `aisle`); the three share a kind, so the `changed` family, which asks only about a
  kind that is unique in the layout, asks about the dock alone.

**The AMR's pieces** (M3; a job finds each by kind and geometry):
- *The charger* `F19` (`charger`, 0.12 x 0.60 x 0.30 m, black, solid from the floor, `blocks: true`, no surface): against
  the east wall's face at (17.44, 1.40), facing west into the dock. Its apron, 1.6 m in front and 0.5 m to either side
  (x 15.78-17.5, y 0.6-2.2), holds nothing: packing table `F10` starts at y 3.1 and rack `F18` at y 9.1, 1.4 m and 7.4
  m from the charger.
- *The bays* `F20`-`F23` (`bay`, 1.6 x 2.2 x 0.005 m, yellow, `blocks: false`, no surface): one on each station's front
  face, 1.6 m deep and 2.0 + 0.1 + 0.1 m wide, the station's yaw: `F20` (15.9, 6.5) and `F21` (15.9, 4.1) in front of
  packing tables `F09` and `F10`, `F22` (1.5, 6.4) and `F23` (1.5, 4.2) in front of shelving `F11` and `F12`. A bay
  touches its station (the zone rule insets each zone by 0.05 m, so a station never blocks its own bay).
- *Markings are at most 0.005 m high.* The staging lanes and the bays are paint: the AMR's `climb_m` is 0.01 m and
  `grid.observe` calls a point floor only at or under it, so a 0.01 m marking, lifted by the depth's millimetre
  rounding, would read as an obstacle in every frame. (The apartment's rugs and mat stay at 0.01 m, its helper's
  `out_of_view` residual.)
- *The lanes stand clear of the stations' arrivals.* `amr/1` turns to face a station at T (its base 1.8-2.4 m out from
  the face, centred on the bay) and arrives along the face; a lane within 0.65 m of T or of that leg would put a
  pallet's footprint on the visit, and the AMR plans with every lane as a keep-out. `F05` moved from x 3.7 to 4.5 and
  `F08` from 13.9 to 12.9, 0.7 m or more from every T and leg; the roll cage `L03` from (15.6, 1.4), which was inside
  bay `F20` and now would stand on the charger's apron, to (11.7, 1.3), 0.1 m from `F07` and from `F08`. A light piece
  starts at most 1 m from its map place and never on a lane, so none ever starts in a bay.
- *The stations stand off the lanes' band* (y 0.6-3.4). The AMR reverses out of a station along its dock line and a
  reverse corridor reaches 5 m behind the docked rear edge, so a station in the band put its corridor across the lanes,
  where a rear drop found no room (8 of 94 visits at the first packing table). The packing table `F09` and its bay
  `F20` stand at y 6.5 (they were at 1.7), the shelving `F11` and its bay `F22` at y 6.4 (1.6), and the charger `F19` at
  y 1.4 (7.1), so every station's corridor runs along y 3.8-6.8 and is 0.4 m or more from every lane (`docs/job-amr.md`).

A rack row has a `rack` surface (1.08 x 2.6 m) only on its **free** bay-levels, a checkerboard that is the same in every
aisle: with `row` 0 for the west row and 1 for the east one, `bay` counted from the dock end (0-3) and `level` 0-2, the
bay-level is free when `row + bay + level` is even. The other bay-levels hold static stock, which the rack builder draws
as part of the rack: never an entity. That makes 36 free rack surfaces of 72 bay-levels, and 48 surfaces in all (with 4
lanes, 2 tables, 6 shelves).

**What the warehouse adds around the map file** (none of it is in the file):
- *The vocabulary* is the `warehouse` catalog (`catalog.WAREHOUSE_KINDS`): six carried kinds (pallet, tote,
  drum, box, sack, bin) with 31 (kind, colour) pairs, 22-28 items a world, a pallet as the delivery, and a forklift and
  a tote robot as the T3 movers. The receptacle kinds the kinds name are the surface kinds above (`rack`, `staging
  lane`, `shelf`, `work table`).
- *The drawing.* `render/furniture_warehouse.py` draws the racks (blue uprights, orange beams, a solid back panel, so
  two back-to-back rows hide each other's aisles, and a thin deck between the beams of each free bay-level above the
  floor, so an item on level 1 or 2 rests on something), the lanes and the bays (painted outlines), the charger (a dark
  plate with two steel contact strips), the tables, the shelving and the dock doors (closed roller shutters) as
  compositions of primitives. A rack draws carton stock in exactly the bay-levels the
  file gives no surface (`render/rack_layout.py` holds the rule), as part of the rack's mesh and its `pass_index`: never
  an entity. The floor is `shader:concrete`, the manifest's floor for both region kinds (a procedural shader, no
  download). The light equipment's boxes equal their procedural builders' `DESIGN_BOX` (`tests/test_warehouse_map.py`).
- *The route.* The anchors of the aisles are at y 12.0, 3 m in from the dock: with them the unchanged route planner
  gives 168.3 keyframes an episode (142-180) in 5.8 attempts on average; a different anchor is a new map
  and changes that number.
- *The checks.* `mapspec.check_map(warehouse_a)` reports nothing; `tests/test_warehouse_map.py` holds the arithmetic
  above (surfaces, the checkerboard, the forklift's cross band and lane).

## downtown_a

`catalog: downtown`; the design is the city plan (§1). A 100 x 96 m district of streets, x east, y north, seen by a
drone (`profile drone`: a camera 12 m up, pitched down 35 degrees); `depth_unit: "cm"` and an `outdoor` sun and sky,
with no ceilings, no ceiling lights, no doors and no windows. Four regions, each a 45 m tall rectangle, share their
edges with no wall between them:
- `high street` (R1, x 0-100, y 0-27): the hub, running east-west, open at both ends.
- `west street` (R2, x 15-35, y 27-72) and `east street` (R3, x 65-85, y 27-72): 20 m side streets running 45 m north
  from the high street between blocks, open at their north ends. Their anchors are 12 m in from the high street.
- `square` (R4, x 35-65, y -24-0): a 30 x 24 m paved square south of the high street, closed behind by a facade, with
  a bare yard in its south-west corner (below).

Every other street edge is a **building facade**: twelve exterior walls 0.6 m thick, their centrelines 0.3 m behind the
region edge, each with a `facade` style. A block is a hollow box of walls whose heights are 4.5 m plus 3.6 m storeys,
18.9 to 44.1 m, all above the 12 m the drone flies, so no roof ever shows; the walls of one building share their height
and style (`brick_red` 22.5 m, `brick_tan` 33.3, `stone_grey` 18.9, `glass_blue` 44.1, `concrete_white` 29.7,
`stucco_cream` 26.1). The street ends at the map edge have no wall: the renderer continues the city beyond them as
static scenery, which no region or route enters. The ego and the items never leave the map and a mover's centre stays
on it, but a bus or taxi body may reach a few metres onto the scenery road past an open end; the renderer keeps that
carriageway clear of scenery objects.

Furniture is fixed only (28 pieces, 37 surfaces):
- **8 sidewalks** (`sidewalk`, 4 m deep, 0.15 m high, grey, `blocks: true`): along the facades of the high street and
  the side streets. Each has one surface (a `sidewalk`, 1.6 m deep, 1 m short of the kerb and 0.5 m short of the strip's
  ends) at the top of the strip. A blocking sidewalk keeps every ground mover on the roadway with no new code; the
  drone flies over it. The metre between a surface and the kerb is where the scenery (lamp posts, signal poles,
  hydrants) stands.
- **10 parking lanes** (`parking_lane`, 2.6 m wide, 1 cm high, white, `blocks: false`) with **22 bays** (`parking`
  surfaces, 2.5 x 6.7 m, 7 m apart), and **2 loading zones** (`loading_zone`, 2.6 x 15 m, yellow, one `loading zone`
  surface of 2.5 x 14.8 m, one on each side street). A lane is a floor marking, so a mover may pull into an empty bay.
  No bay is within 6 m of a corner.
- **The square's paving** (`plaza` named `square paving`, 0.15 m, beige, `blocks: true`) in two pieces that tile the
  square less a bare yard in its south-west corner (x 35.005-45.005, y -23.995 to -12.995): `F21`, 29.99 x 12.99 m
  centred at (50.0, -6.5), and `F24`, 19.99 x 10.99 m centred at (55.0, -18.5), with **5 pads** between them (`plaza`
  surfaces, 3.5 x 3.5 m, at (40, -6), (50, -6), (60, -6), (50, -18) and (60, -18)), which hold carts, toilets,
  dumpsters and skips; a car or a van does not fit one. The pad at (40, -18) went with the yard.
- **The drone's nest** `F25` (`nest`, 1.6 x 1.6 x 0.02 m, grey, `blocks: false`, no surface) at (40.0, -18.5), in the
  middle of the yard on bare ground: 5.0 m from the facade `W09`'s face and from `F24`, 5.5 m from `W12`'s face and
  from `F21`, where the drone (radius 0.65 m, clearance 3.0 m) needs 3.65 m. Nothing stands on the yard.
- **Five drop zones** `F26`-`F30` (`drop_zone`, `blocks: false`, white, no surface), each a marking 0.005 m thick on the
  ground under it, so its height is that ground's plus 0.005 m, with that piece's yaw; `name` is the address a brief
  names: "Square West" (40, -6), "Square East" (60, -6) and "Square South" (60, -18), 3.0 x 3.0 x 0.155 m on three of
  the paving's pads, and "West Street" (20.3, 44.1) and "East Street" (79.7, 44.1), 2.6 x 3.0 x 0.015 m at the south end
  of the two loading zones. Every centre is at least 5.0 m from every wall's face (the drone hovers 7 m over a zone and
  keeps 3.65 m from walls). The pads at (50, -6) and (50, -18) stay plain, street furniture's places. A zone has no
  surface, so the world never places an item onto one; its state comes from depth alone.

**What the downtown adds around the map file** (none of it is in the file):
- *The vocabulary* is the `downtown` catalog (`catalog.DOWNTOWN_KINDS`): eight carried kinds (car, van, pickup, box
  truck, skip, dumpster, portable toilet, food cart) with 43 (kind, colour) pairs, 26-34 items a world, a van as the
  delivery, and a taxi and a 9 m bus as the T3 movers. The receptacle kinds the kinds name are the surface kinds above
  (`parking`, `loading zone`, `sidewalk`, `plaza`).
- *The drawing.* Asphalt streets (`shader:asphalt`, the manifest's floor for the three streets) and paving
  (`shader:paving`, the square's), procedural, no download; the nest and the drop zones are placeholders
  (`render/furniture_city.py`: a dark mat with a painted ring; an outline and a cross of paint). The facades, the street furniture and the city beyond the
  edge are the renderer's (`render/`).
- *The route.* The drone's profile gives its own floor-plan path: it flies over everything lower than 12 - 3 = 9 m and
  keeps 3 m from the facades, at 3 m/s (a keyframe every 4.5 m). A different anchor, street width or block is a new map
  and changes the route lengths measured in the plan.
- *The checks.* `mapspec.check_map(downtown_a)` reports nothing; `tests/test_downtown_map.py` holds the arithmetic above
  (regions, surfaces, the kerb strip, the facade heights).

## house_floor

Catalog `home` (the file has no such key). The ground floor of a large detached two-storey house, 11.4 x 8.8 m, x east,
y north, walls 0.1 m thick and 2.5 m high: 94.0 m² of floor in ten rooms, the front door in the south wall. Written by
`tools/maps/house_floor.py`, whose `spec()` builds it from the plan and checks it with `mapspec.check_map` before it
writes (`tests/test_house_floor_map.py` holds the file to the maker). Each region is named by its kind, each of the ten
home kinds once (`vacuum/2` names a room by a unique kind; the household's clutter matches room names).
- **Rooms** (m²): hall R1 14.2, corridor R2 6.9, living room R3 14.7, dining room R4 11.5, kitchen R5 12.6, utility room
  R6 3.9, cloakroom R7 3.2, study R8 11.3, bedroom R9 11.1, bathroom R10 4.6. No room is larger than 15.2 m²: a
  `vacuum/2` task takes about 38 s a square metre of room floor, and `TASK_MAX_S` is 1,200 s.
- **Openings:** D0 the front door (shut); doors D1-D9 between the hall, living room, study, corridor, kitchen,
  cloakroom, utility room, bedroom, bathroom and dining room (0.8-0.9 m, each swinging into a room, never into the
  corridor); arches O1 hall-corridor and O2 living room-dining room (1.2 m). Two independent cycles.
- **The stairs** `F01` (`stairs`, 3.4 x 0.9 x 2.5 m, blocking, no surface) against the hall's east wall, rising north
  into a bulkhead; the floor above is not modelled. `render/furniture_house.py` draws it.
- **The dock** `F02` on the hall's west wall, its apron clear of every blocking piece and every door's swing.
- **Windows:** a window 0.9 m or wider stands over furniture (the dressing hangs curtains there, out of the helper's
  swept body); the rest are narrower. Toilets are 0.46 m wide (a wall cabinet over a narrower one would overhang into
  the helper's body).

## suburb

Catalog `residential`. A street of detached houses, 300 x 82 m, x east, y north, with no walls: the far scenery starts at
the bounds. Written by `tools/maps/suburb.py`, whose `spec()` builds it and checks it with `mapspec.check_map` before
it writes (`tests/test_suburb_map.py` holds the file to the maker).

```
y 82 +---------------------------+------+---------------------------+
     | north plots 1-17 (odd)    | car  | north plots 19-35 (odd)   |
     | nine, facing south        | park | nine, facing south        |
     |                           | (N)  |                           |
  47 +---------------------------+------+---------------------------+
     | street: verge, sidewalk, carriageway, sidewalk, verge (12 m) |
  35 +---------------------------+-+----+---------------------------+
     | south plots 2-18 (even)   |p|G   | south plots 20-36 (even)  |
     | nine, facing north        | |    | nine, facing north        |
   0 +---------------------------+-+----+---------------------------+
     x 0                        135 165                            300
```

N is the drone's nest (150, 64.5), on bare asphalt 14.7 m from the nearest piece; p the passage (x 135-138); G the green
with its playground (x 138-165).
- **Regions:** 36 of kind `garden`, each named by its address (north odd, south even, "<n> Elm Street"), and one each
  of `street`, `passage`, `green` and `car park`: 40 regions.
- **The street** (12 m): a 1.2 m verge against the plots, a 1.8 m sidewalk with the kerb on the carriageway side (its
  `sidewalk` surface 1.5 m wide, so a 1.3 m dumpster fits), a 6 m carriageway; 20 kerbside bays in lanes of two, never
  opposite each other, and 8 bays in the car park.
- **A plot** is 15 x 35 m: a house 6.5-8 m back, a 3.4 m driveway on the garage side, a path, lawns, a hedge or a fence
  on the west side, a rear fence and two garden beds.
- **Drop zones:** 12 of the 36 plots (1, 4, 7, ... 34 Elm Street), 3 m square, centred 1.9 m back from the street on
  the driveway's `parking` surface, at least 4.6 m from the house; no blocking piece stands on one.
- **People** (`people_residential`): the downtown's worker body, 4 a script with `P_HOME` 0.5 (about 2 present in a
  flight), walking on `sidewalk` tops (the longer free stretch of one side: the two sidewalks are not joined), keeping
  off drop zones. **Processes** (`world_processes_residential`): the downtown's laws, half-lives keyed by kind.
