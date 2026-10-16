"""Carried-item kinds: sizes, colours, receptacle priors and a geometry hint for the renderer.

`size` is the kind's bounding box (x, y, z) in metres. Footprints use x and y (rotated by the placement's yaw); z is the
height. An item has one attribute, its colour: the size attribute ("small" or "large") was dropped in generator 0.4.0,
because the probe measured the models' size judgement at .39 (n 36) against a bar of .80 (DECISIONS, "Probe run 1").
`receptacles` lists the receptacle kinds (map spec surface kinds) where the item is typically found; the generator
puts 70% of items on one of them and 30% on any other surface. `geometry` is the procedural placeholder the Blender
builder draws in M1; the builder may use `size` to scale the parts.
Sizes follow the photoreal assets: the renderer fits each kind's mesh to its `size`, so a kind's box is its asset's
measured box in the manifest (`native_bbox_m`), uniformly scaled where a scan is not a realistic size (the bag, the box,
the toy), and a procedural kind's box is the one its builder in `render/procedural.py` is built to.
`tests/test_catalog.py` holds the two together.
The laptop is open (a closed one is a slab that a person cannot tell from a book): its box is the open one, lid
included, 0.306 deep (x, with the lid's lean), 0.33 wide and 0.222 high.
A colour is what the pixels show (VISION rule 10): the plant is green only (its leaves dominate its pot). A kind with
one colour has no lookalike, since a lookalike differs in the one attribute that exists.
Glass and bottle have no "clear" colour: a clear item would show what lies behind it in the RGB image while the ID pass
shows the item, so the two would disagree (M2: tinted glass only).
Keys, phone and glasses were dropped in generator 0.4.0 (DECISIONS, "Probe run 1" and "v0 scope"): the probe did not
recognise keys and phone (a tag colour and a face-up screen), and models do not reliably recognise glasses even at
256-512 px, nor see them in half of their worlds, so their questions would test perception, not memory.
Mover kinds (`MOVER_KINDS`, tier T3 from generator 0.5.0) are things that move by themselves: a robot vacuum, a home
robot and a ball robot, all procedural. Each has one attribute, its colour, and the motion numbers of the
movers' motion model (`mover_walk.py`, retired in M0 until M2's movers lane ports it from `redesign-base`).
A `Catalog` is one map's vocabulary: its carried kinds, the kinds of its identical sets and of what leaves between
visits, the kind and colours of what is delivered, its mover kinds, and how many movers and items a world holds.
`CATALOGS["home"]` is the constants above, unchanged in value and order (the apartment's
worlds are drawn from them), `CATALOGS["warehouse"]` is `warehouse_a`'s, and `for_map` picks the one a map names
(`catalog` in the map file, absent for "home"). A kind that two catalogs share (`box`) has one size; `item_size` finds a
kind in any catalog. A kind with `square` set sits turned a quarter turn at a time to its surface (the warehouse's:
stock on racks is squared up, not scattered at any angle; the downtown's: vehicles park along the kerb); the household's
kinds sit at any angle. `CATALOGS["downtown"]` is `downtown_a`'s: eight kinds that stand in the streets (parked
vehicles, skips, dumpsters, portable toilets, food carts), all procedural, with a taxi and a bus as the T3 movers; its
sizes are real ones (a 4.6 m sedan, a 7.3 m box truck).
A `ClutterKind` is world content that no question names (`catalog_clutter.py` has the type and the household's
table): it is a catalog's second vocabulary, `Catalog.clutter`, never part of `kinds`. `Catalog.floor_kinds` names the
tracked kinds that real floors hold, and per kind the furniture kinds it is dropped beside.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import catalog_residential as residential
from .catalog_clutter import (  # noqa: F401  (the clutter vocabulary lives in `catalog_clutter`; the names stay here)
    FLOOR,
    HOME_CLUTTER,
    HOME_FLOOR_KINDS,
    ROOM_SEP,
    ClutterKind,
    place_of,
)


@dataclass(frozen=True)
class Kind:
    name: str
    size: tuple[float, float, float]
    colors: tuple[str, ...]
    receptacles: tuple[str, ...]
    geometry: str
    square: bool = False  # sits at a multiple of 90 degrees to its surface (`placement.SQUARE_YAW` of slack)


def _k(name, size, colors, receptacles, geometry, square=False):
    return Kind(name, size, tuple(colors), tuple(receptacles), geometry, square)


KINDS: dict[str, Kind] = {
    k.name: k
    for k in (
        _k("mug", (0.131, 0.097, 0.115), ("red", "blue", "green", "yellow", "white", "black", "orange"),
           ("kitchen counter", "dining table", "desk", "nightstand", "coffee table"), "cylinder+handle"),
        _k("glass", (0.07, 0.07, 0.12), ("blue", "green"),
           ("kitchen counter", "dining table", "sink edge", "nightstand", "coffee table"), "open cylinder"),
        _k("bowl", (0.16, 0.16, 0.07), ("white", "blue", "red", "green", "yellow", "black"),
           ("kitchen counter", "dining table", "sink edge"), "hemisphere"),
        _k("plate", (0.222, 0.222, 0.028), ("white", "blue", "black", "green", "yellow"),
           ("kitchen counter", "dining table", "sink edge"), "disc"),
        _k("book", (0.22, 0.15, 0.03), ("red", "blue", "green", "yellow", "black", "orange", "purple"),
           ("shelf", "coffee table", "nightstand", "desk", "sofa"), "box"),
        _k("remote", (0.18, 0.05, 0.02), ("black", "white", "grey"),
           ("coffee table", "sofa", "tv console"), "slab+buttons"),
        _k("bag", (0.374, 0.304, 0.314), ("black", "red", "blue", "green", "brown", "grey", "orange"),
           ("floor mat", "sofa", "bed", "hall console", "desk"), "box+handles"),
        _k("shoes", (0.244, 0.191, 0.080), ("black", "white", "red", "blue", "brown"),
           ("shoe rack", "floor mat"), "pair of wedges"),
        _k("laptop", (0.306, 0.33, 0.222), ("grey", "black", "white"),
           ("desk", "coffee table", "dining table", "sofa", "bed"), "two hinged slabs, open at 110 degrees"),
        _k("plant", (0.16, 0.16, 0.30), ("green",),
           ("shelf", "coffee table", "dining table", "tv console", "kitchen counter"), "pot cylinder+foliage blob"),
        _k("bottle", (0.07, 0.07, 0.25), ("green", "blue", "brown", "red"),
           ("kitchen counter", "dining table", "coffee table", "desk", "shelf"), "cylinder+neck"),
        _k("box", (0.40, 0.30, 0.267), ("brown", "white", "blue", "red", "green"),
           ("dining table", "hall console", "floor mat", "rug", "bed"), "box"),
        _k("towel", (0.356, 0.263, 0.088), ("white", "blue", "red", "green", "yellow", "grey"),
           ("shelf", "bed", "sink edge"), "folded slab"),
        _k("toy", (0.14, 0.10, 0.134), ("red", "blue", "yellow", "green", "orange", "pink"),
           ("rug", "sofa", "floor mat", "coffee table", "bed"), "stacked blocks"),
    )
}

# Kinds that people own and keep in matching sets: the generator draws its identical sets from these only.
SET_KINDS = ("mug", "glass", "bowl", "plate", "towel")

# Kinds the generator removes between visits when it removes something (a bag leaves; so can a delivery box).
LEAVING_KINDS = ("bag", "box")

# What arrives between visits when something does: a cardboard delivery box, in a colour no item of the world has had
# (brown first).
DELIVERY_KIND = "box"
DELIVERY_COLORS = ("brown", "white", "blue", "red", "green")


# How a mover keeps clear (metres), shared by the motion model (`mover_walk.py`) and `validate`: from the camera (xy
# distance at every keyframe; `MoverKind.camera_m`, this value unless the kind says otherwise), from another mover (on
# top of the two footprints' circumradii), and the height under which a carried item lies on the floor and so
# obstructs a mover.
MOVER_CAMERA_M = 0.6
MOVER_PAD_M = 0.1
FLOOR_ITEM_Z = 0.1


@dataclass(frozen=True)
class MoverKind:
    """A thing that moves by itself (tier T3, generator 0.5.0): a rigid body with a box, colours, and the numbers of
    its motion model (`mover_walk.py`). It has one attribute, its colour, like a carried item; its description is colour
    then kind ("white robot vacuum")."""

    name: str
    size: tuple[float, float, float]  # box (x, y, z) m; the procedural builder's DESIGN_BOX
    colors: tuple[str, ...]  # palette keys that already exist in render/shapes.PALETTE
    speed: tuple[float, float]  # m/s, drawn per leg
    dwell: tuple[float, float]  # s, a pause before each leg
    same_room: float  # chance the next waypoint is in the current room
    camera_m: float = MOVER_CAMERA_M  # its centre stays this far (xy) from the camera: outside its own body


# Sizes and speeds trace to real robots (PRIOR-ART §7, "Movers"). robot_vacuum: a Roomba is a 338-353 mm disc, 86-94 mm
# high, and iRobot's patent gives its default straight-line speed as 0.306 m/s. home_robot: Amazon Astro is 424 x 250 x
# 440 mm with a top speed of 3.3 ft/s (1.0 m/s); the range sits under it. ball_robot: Sphero BOLT tops out at 2 m/s and
# the range is half of that or less. Dwell is a modelling choice with no source: a mover waits a few seconds to
# half a minute between legs, capped at 20 s so it stays under the 30 s window of `where_at`'s unknown rule.
MOVER_KINDS: dict[str, MoverKind] = {
    "robot_vacuum": MoverKind("robot_vacuum", (0.34, 0.34, 0.09), ("black", "white", "grey"), (0.25, 0.35), (2, 10),
                              0.75),
    "home_robot": MoverKind("home_robot", (0.35, 0.30, 0.45), ("white", "grey", "blue"), (0.40, 0.80), (3, 20), 0.5),
    "ball_robot": MoverKind("ball_robot", (0.22, 0.22, 0.22), ("white", "yellow", "orange"), (0.50, 1.00), (3, 20),
                            0.4),
}

# Movers per T3 episode: two different kinds and a colour each, so every mover's description is unique.
N_MOVERS = 2


# -- the warehouse -----------------------------------------------------------------------------------------------
# Sizes follow real stock and equipment (PRIOR-ART §7): a 1200 x 1000 mm ISO pallet with a load to 1.0 m, a 600 x
# 400 mm Euro tote, a 200 l drum (585 mm), a woven sack, a small-parts bin. The box is the household's cardboard box:
# one kind at one size in both catalogs, but not in brown: the racks' stock is brown cartons that are no entity, so
# "the brown box" would not be unique in the pictures. Receptacles name map surface kinds (`rack`, `staging lane`,
# `shelf`, `work table`), the most typical first.
WAREHOUSE_KINDS: dict[str, Kind] = {
    k.name: k
    for k in (
        _k("pallet", (1.2, 1.0, 1.0), ("blue", "red", "green", "yellow", "black", "white"),
           ("rack", "staging lane"), "wooden pallet + stretch-wrapped load", square=True),
        _k("tote", (0.6, 0.4, 0.32), ("blue", "red", "green", "yellow", "grey", "black"),
           ("rack", "shelf", "work table"), "open tub + handle cut-outs", square=True),
        _k("drum", (0.58, 0.58, 0.88), ("blue", "red", "green", "black", "yellow"),
           ("rack", "staging lane"), "cylinder + two rolled rings", square=True),
        _k("box", KINDS["box"].size, tuple(c for c in KINDS["box"].colors if c != "brown"),
           ("rack", "shelf", "work table"), "box", square=True),
        _k("sack", (0.9, 0.5, 0.25), ("white", "brown", "green", "blue", "yellow"),
           ("rack", "staging lane"), "pillow-shaped woven sack", square=True),
        _k("bin", (0.40, 0.25, 0.20), ("blue", "red", "yellow", "green", "grey"),
           ("shelf", "work table", "rack"), "open picking bin", square=True),
    )
}
WAREHOUSE_SET_KINDS = ("tote", "drum", "box", "bin")
WAREHOUSE_LEAVING_KINDS = ("pallet", "box", "tote")
# A pallet arrives, in a wrap colour no item of the world has had (white first, then the pallet's own order).
WAREHOUSE_DELIVERY_KIND = "pallet"
WAREHOUSE_DELIVERY_COLORS = ("white", "blue", "red", "green", "yellow", "black")
# A forklift (counterbalance, forks included) and a tote robot (an AMR carrying totes). The speeds and dwell are
# placeholders until PRIOR-ART §7 cites site speed limits and an AMR maker's specification. `camera_m` is the
# circumradius plus 0.35 m to the centimetre (1.41 + 0.35, 0.50 + 0.35): the camera is never inside the body.
WAREHOUSE_MOVER_KINDS: dict[str, MoverKind] = {
    "forklift": MoverKind("forklift", (2.6, 1.1, 2.1), ("yellow", "orange", "red"), (0.8, 1.5), (3, 20), 0.5, 1.76),
    "tote_robot": MoverKind("tote_robot", (0.8, 0.6, 1.3), ("blue", "grey", "orange"), (0.6, 1.0), (3, 20), 0.5, 0.85),
}


# -- downtown -----------------------------------------------------------------------------------------------------
# Sizes follow real vehicles and street furniture (PRIOR-ART §7): a mid-size sedan, a compact cargo van, a crew-cab
# pickup, a 16-ft box truck (2.4 m body), a 6-yard builders' skip, a 4-yard front-load container, a portable toilet
# and a street vendor's cart with its umbrella. Every kind is `square`: it stands turned a quarter turn to its
# surface, along the kerb. Colours are the ones that read from 12-35 m in the look-dev's frames: black and grey are
# lost in shade and so are no colour of any kind here. Receptacles name map surface kinds (`parking`, `loading zone`,
# `sidewalk`, `plaza`), the most typical first.
DOWNTOWN_KINDS: dict[str, Kind] = {
    k.name: k
    for k in (
        _k("car", (4.6, 1.8, 1.45), ("white", "red", "blue", "green", "beige"), ("parking",),
           "sedan: lofted body, glasshouse, four wheels", square=True),
        _k("van", (5.2, 2.0, 2.1), ("white", "blue", "red", "green", "yellow", "orange"),
           ("parking", "loading zone"), "compact cargo van: box body, cab, four wheels", square=True),
        _k("pickup", (5.4, 2.0, 1.9), ("red", "blue", "white", "green", "orange", "beige"), ("parking",),
           "crew-cab pickup: cab and open bed, four wheels", square=True),
        _k("box_truck", (7.3, 2.4, 3.3), ("white", "yellow", "red", "blue", "green", "orange"), ("loading zone",),
           "16-ft box truck: cab and box body, four wheels", square=True),
        _k("skip", (3.0, 1.7, 1.15), ("yellow", "orange", "green", "blue", "red"), ("parking", "loading zone"),
           "builders' skip: tapered open tub with rubble", square=True),
        _k("dumpster", (1.8, 1.3, 1.4), ("green", "blue", "red", "orange", "yellow"), ("sidewalk", "loading zone"),
           "front-load container: lidded bin on casters", square=True),
        _k("portable_toilet", (1.1, 1.1, 2.3), ("blue", "green", "orange", "red"), ("plaza", "sidewalk"),
           "standing cabin with a door and a roof vent", square=True),
        _k("food_cart", (2.0, 1.0, 2.2), ("red", "blue", "yellow", "green", "white", "orange"), ("plaza", "sidewalk"),
           "street cart under an umbrella (the colour)", square=True),
    )
}
DOWNTOWN_SET_KINDS = ("portable_toilet", "dumpster")
DOWNTOWN_LEAVING_KINDS = ("car", "van", "pickup")
# A van arrives, in a colour no item of the world has had (white first, then the van's own order).
DOWNTOWN_DELIVERY_KIND = "van"
DOWNTOWN_DELIVERY_COLORS = ("white", "blue", "red", "green", "yellow", "orange")
# A taxi (a sedan with a roof sign: yellow or orange, no carried car's colour) and a 9 m midibus. Speeds sit under the
# 25 mph (11.2 m/s) downtown limit; a city bus averages 3-4 m/s with its stops and moves faster between them (PRIOR-ART
# §7, "Movers"). Dwell is capped at 20 s as indoors. `camera_m` is the circumradius plus 0.35 m to the centimetre (2.47
# + 0.35, 4.67 + 0.35) for a ground profile; a drone far above them never reads it.
DOWNTOWN_MOVER_KINDS: dict[str, MoverKind] = {
    "taxi": MoverKind("taxi", (4.6, 1.8, 1.55), ("yellow", "orange"), (4.0, 8.0), (3, 20), 0.5, 2.82),
    "bus": MoverKind("bus", (9.0, 2.5, 3.1), ("blue", "red", "white", "green"), (3.0, 7.0), (3, 20), 0.5, 5.02),
}


@dataclass(frozen=True, eq=False)
class Catalog:
    """One map's vocabulary (see the module docstring). `n_items` is the inclusive range of carried items a world
    draws; `n_movers` is how many movers a T3 world holds, each of a different kind. `delivery_kind` arrives between
    visits in the first of `delivery_colors` that no item of the world has had. `clutter` and `floor_kinds` come last
    and default to none: the world content no question names, and the tracked kinds left on the floor."""

    name: str
    kinds: dict[str, Kind]
    set_kinds: tuple[str, ...]
    leaving_kinds: tuple[str, ...]
    mover_kinds: dict[str, MoverKind]
    n_movers: int
    n_items: tuple[int, int]
    delivery_kind: str
    delivery_colors: tuple[str, ...]
    clutter: tuple[ClutterKind, ...] = ()
    floor_kinds: tuple[tuple[str, tuple[str, ...]], ...] = ()  # tracked kind -> the furniture kinds it is dropped by

    def clutter_kind(self, name: str) -> ClutterKind:
        found = [k for k in self.clutter if k.name == name]
        if len(found) != 1:
            raise KeyError(f"{len(found)} clutter kinds named {name!r} in the {self.name} catalog")
        return found[0]


CATALOGS: dict[str, Catalog] = {
    "home": Catalog("home", KINDS, SET_KINDS, LEAVING_KINDS, MOVER_KINDS, N_MOVERS, (30, 40), DELIVERY_KIND,
                    DELIVERY_COLORS, clutter=HOME_CLUTTER, floor_kinds=HOME_FLOOR_KINDS),
    "warehouse": Catalog("warehouse", WAREHOUSE_KINDS, WAREHOUSE_SET_KINDS, WAREHOUSE_LEAVING_KINDS,
                         WAREHOUSE_MOVER_KINDS, 2, (22, 28), WAREHOUSE_DELIVERY_KIND, WAREHOUSE_DELIVERY_COLORS),
    "downtown": Catalog("downtown", DOWNTOWN_KINDS, DOWNTOWN_SET_KINDS, DOWNTOWN_LEAVING_KINDS, DOWNTOWN_MOVER_KINDS, 2,
                        (26, 34), DOWNTOWN_DELIVERY_KIND, DOWNTOWN_DELIVERY_COLORS),
    "residential": Catalog("residential", {k: DOWNTOWN_KINDS[k] for k in residential.KIND_NAMES}, residential.SET_KINDS,
                           residential.LEAVING_KINDS, {}, residential.N_MOVERS, residential.N_ITEMS,
                           residential.DELIVERY_KIND, DOWNTOWN_DELIVERY_COLORS),
}
HOME = CATALOGS["home"]


def for_map(m) -> Catalog:
    """The catalog a map (`MapSpec.catalog`) draws its items and movers from."""
    if m.catalog not in CATALOGS:
        raise ValueError(f"map {m.name!r} names catalog {m.catalog!r}; known: {', '.join(CATALOGS)}")
    return CATALOGS[m.catalog]


def mover_kind(kind: str) -> MoverKind:
    """The mover kind of that name, from whichever catalog has it (the names of the catalogs' mover kinds are
    disjoint); KeyError when none has it. People's kinds are `people.person_kind`'s (the truth side reads both)."""
    for c in CATALOGS.values():
        if kind in c.mover_kinds:
            return c.mover_kinds[kind]
    raise KeyError(kind)


def item_size(kind: str) -> tuple[float, float, float]:
    """Bounding box (x, y, z) of an item of the kind (a carried kind or a mover kind of any catalog): every item of a
    kind has the same box."""
    for c in CATALOGS.values():
        if kind in c.kinds:
            return c.kinds[kind].size
        if kind in c.mover_kinds:
            return c.mover_kinds[kind].size
    raise KeyError(kind)


def describe(kind: str, attributes: dict[str, str]) -> str:
    """Words for a description: "red bowl". Colour, then kind."""
    colour = [attributes["color"]] if "color" in attributes else []
    return " ".join([*colour, kind])
