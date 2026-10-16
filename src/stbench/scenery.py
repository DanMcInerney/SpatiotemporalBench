"""Fixed scenery that looks like a tracked kind: data per catalog.

A map's furniture can carry objects that are no entity: the warehouse's racks are drawn full of beige cartons stacked
on bare wooden pallets in every bay-level that holds no free surface (`render/furniture_warehouse.py`,
`rack_layout.stock`). They never move and no gold counts them, yet "all the pallets", "all the boxes" and a
`where_went` referent named by its kind alone all match them on the pictures. A `Scenery` entry declares what the
scenery looks like: per tracked kind it resembles (a box, a pallet), the colours the pictures show it in.

A declaration is safe to act on only if no tracked thing can look like the scenery: a (kind, colour) that the catalog's
own things can have would be a thing the robot is told to ignore. A tracked kind's colours are its `Kind.colors` and,
for the catalog's delivery kind, its `delivery_colors` too (a delivery arrives in a colour of that list).
`tracked_colours` reads them, `lookalike_problems` finds a declared colour among them, and `tests/test_scenery.py`
requires none and ties the declaration to the render's carton colour. So a catalog that later gives its box a brown or
beige variant fails the test instead of turning stock into scenery. The warehouse's declaration is the `resembles` of
its `rack_stock` class (`WAREHOUSE_STOCK_LOOKALIKES`); the household's catalog needs none: its bookshelf and shoe rack
are drawn empty. No code names a map.

The dressing registry (`DRESSING`) is the one list of every non-entity object class a render
draws: the warehouse's rack stock above and the render-only "dressing" a map's layout module places
(`render/dressing_layout_<catalog>.py`: trim, fittings, fixtures, building structure). A `DressingClass` says which
catalogs it belongs to, which tracked kinds it can be mistaken for (`resembles`, `()` when it is disjoint from all of
them: the usual case) and in which phase it is drawn. A class that resembles a kind is held to the rule above (its
colours must be ones a tracked thing of that kind never has, `class_problems`). `names_a_kind` is the free guard on the
name alone: a class called what a tracked kind is called, or one of its synonyms (`SYNONYMS`), must declare the
resemblance. `tests/test_dressing.py` enforces all of it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .catalog import CATALOGS, Catalog, place_of


@dataclass(frozen=True)
class Scenery:
    """One catalog's fixed look-alikes: `lookalikes` are, per tracked kind of the catalog that the scenery resembles,
    the colours it looks like on the pictures (none of which a tracked thing of that kind has)."""

    catalog: str
    lookalikes: tuple[tuple[str, tuple[str, ...]], ...]


# The warehouse's rack stock: cartons that read beige (the render's `CARTON`, "beige*0.8", and its darker shade) and,
# in a dim view, brown, stacked on a bare wooden pallet. They look like a box and, as a load on its pallet, like a
# pallet; no tracked box is beige or brown (Release 0.2 dropped the household's brown) and a tracked pallet's colour
# is its stretch wrap.
WAREHOUSE_STOCK_LOOKALIKES = (("box", ("beige", "brown")), ("pallet", ("beige", "brown")))


def tracked_colours(catalog: str, kind: str) -> set[str]:
    """The colours a tracked thing of the kind can have in the catalog: the kind's own and, for the catalog's delivery
    kind, the colours of what is delivered. Raises KeyError for a kind that is not one of the catalog's."""
    c = CATALOGS[catalog]
    return {*c.kinds[kind].colors, *(c.delivery_colors if kind == c.delivery_kind else ())}


def lookalike_problems(entry: Scenery) -> list[str]:
    """What is wrong with an entry: a look-alike kind that is not one of the catalog's, a kind with no colour, or a
    colour that a tracked thing of the kind can have (it would be a thing the robot is told to ignore)."""
    problems = []
    for kind, colours in entry.lookalikes:
        if kind not in CATALOGS[entry.catalog].kinds:
            problems.append(f"{kind!r} is not a kind of the {entry.catalog} catalog")
            continue
        if not colours:
            problems.append(f"{kind!r} has no look-alike colour")
        problems += [f"({kind}, {c}) is a tracked thing of the {entry.catalog} catalog, not scenery"
                     for c in colours if c in tracked_colours(entry.catalog, kind)]
    return problems


# -- the dressing registry -----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DressingClass:
    """One class of object a render draws that is no entity. `catalogs` are the catalogs whose maps use it (a layout
    may emit it only for those, and a catalog's layout must use every phase 1 class listed for it); `resembles` is `()`
    for a class that cannot be mistaken for any tracked kind and else, per tracked kind it looks like, the colours the
    pictures show it in (as `Scenery.lookalikes`); `phase` is 0 for what the furniture builders already draw (the rack
    stock), 1 for render-only dressing (the timelines and golds stay as they are) and 2 for world clutter (new maps and
    a new generator); `note` says what it is and, when it is not obvious, why it is not a tracked kind."""

    name: str
    catalogs: tuple[str, ...]
    resembles: tuple[tuple[str, tuple[str, ...]], ...]
    phase: int
    note: str


PHASES = (0, 1, 2)
# What a class of the household's dressing is, in the order the plan lists them: trim and
# fittings on walls and ceilings, the fixtures of the kitchen and bathroom, hung and ceiling pieces.
_HOME_DRESSING = (
    ("baseboard", "8 cm trim along the foot of a wall face, a centimetre or two proud"),
    ("door_casing", "trim around a door or arch opening (the door itself is an entity and is never drawn here)"),
    ("window_sill", "sill and sash trim around a window: under and beside the glass, never across it"),
    ("outlet", "a wall socket plate"),
    ("light_switch", "a wall switch plate"),
    ("thermostat", "a small wall unit"),
    ("smoke_detector", "a ceiling disc"),
    ("radiator", "a panel radiator under a window, a hand's width proud of the wall"),
    ("curtain", "drapes beside a window, outside the glass; a hung textile, not a folded towel"),
    ("roller_blind", "a rolled blind in the head of a window opening, drawn open"),
    ("wall_tv", "a flat screen on the wall above the TV console"),
    ("upper_cabinet", "kitchen wall cabinets from 1.45 m up, inside the footprint of the counter run below"),
    ("range_hood", "a hood over the stove"),
    ("backsplash", "a tiled strip between the counters and the upper cabinets"),
    ("bathroom_cabinet", "a closed wall cabinet (no mirror: the ID pass could not show it)"),
    ("toilet_roll_holder", "a wall holder with a roll"),
    ("shower_curtain", "a curtain and rail over the bathtub's footprint"),
    ("picture_frame", "framed art flat on a wall"),
    ("wall_clock", "a round clock on a wall, hung vertically (a plate lies flat)"),
    ("ceiling_lamp", "a fitting under a ceiling light"),
    ("ceiling_fan", "a fan under the bedroom ceiling"),
    ("dust_line", "a patchy film of dust along the foot of a baseboard, a few millimetres thick, flush"),
    ("hair", "a few strands lying on the floor, a millimetre thick, flush"),
    ("crumbs", "specks of food on the floor round the worktops and tables, a few millimetres, flush"),
    ("scuff", "a faint dark mark of soles on the floor in front of a door, flush"),
)
_WAREHOUSE_DRESSING = (
    ("column", "a steel column set into a wall, no more than a hand's width proud"),
    ("roof_truss", "roof structure above 5 m"),
    ("purlin", "roof structure above 5 m"),
    ("roof_deck", "the underside of the roof, above 5 m"),
    ("skylight", "a translucent roof panel"),
    ("sprinkler_main", "a red ESFR main pipe overhead, above 5 m"),
    ("sprinkler_head", "a sprinkler head under a main"),
    ("cable_tray", "a cable tray overhead"),
    ("high_bay_light", "a round LED fixture under the roof (the map's nine ceiling lights stay where they are)"),
    ("wall_cladding", "cladding over a concrete dado on the wall faces"),
    ("safety_band", "a yellow band along the foot of a wall"),
    ("dock_door_seal", "a foam seal framing a dock door"),
    ("dock_bumper", "a black rubber block beside a dock door at floor level"),
    ("dock_leveler", "a flush steel plate (no more than 2 cm) in front of a dock door"),
    ("dock_light", "a small lamp over a dock door"),
    ("door_number", "a painted number over a dock door"),
    ("rack_upright_protector", "a yellow guard on the foot of an upright, clear of the bay-levels things rest in"),
    ("end_of_aisle_guard", "a yellow guard at the end of a rack row"),
    ("beam_label", "a readable label on a beam's front face, below the level a thing rests on"),
    ("aisle_sign", "a readable sign at the end of an aisle, above the racks' stock"),
    ("fire_extinguisher", "a red wall-hung cylinder half a metre tall; a drum is a 0.88 m floor-standing one (review "
                          "the contact sheet)"),
    ("fire_alarm", "a small red wall unit; a box is a 0.4 m carton on a surface (review the contact sheet)"),
    ("security_camera", "a wall or ceiling camera"),
    ("power_panel", "a grey electrical panel on a wall (the plan's power_box, not called a box: a tracked kind)"),
    ("utility_enclosure", "a grey utility cabinet on a wall (the plan's utility_box, renamed for the same reason)"),
    ("air_duct", "a round duct overhead, above 5 m"),
    ("cable_run", "cables along a wall or overhead"),
)
# The downtown's (render/dressing_layout_downtown.py): paint and iron flush with the road, and what stands in the kerb
# strip. None may look like a tracked kind (cars, vans, pickups, box trucks, skips, dumpsters, portable toilets, food
# carts; the taxi and the bus): so no vehicle, container, kiosk or phone box (a portable toilet), and no parasol or cart
# (a food cart). A tree is no tracked thing: its crown starts above 3.4 m, over the sidewalk, and its colour is a dull
# natural green, not the palette's.
_DOWNTOWN_DRESSING = (
    ("lane_marking", "yellow double centre lines and white lane dashes painted flush on the carriageway"),
    ("crosswalk", "white zebra stripes across a side street's mouth or the street beside the square, flush"),
    ("stop_bar", "a white bar across a lane before a crosswalk, flush"),
    ("manhole_cover", "a round cast-iron cover in a travel lane, under 2 cm"),
    ("storm_drain", "a gutter inlet grate beside a kerb, under 2 cm"),
    ("street_tree", "a tree pit and stem in the kerb strip, up to 3.4 m (the crown is its own class)"),
    ("tree_crown", "a street tree's leaf mass from 3.4 m up, over the sidewalk, never past the kerb"),
    ("bench", "a slatted park bench in the kerb strip, facing the building line"),
    ("planter", "a low concrete trough with shrubs in the kerb strip (the household's plant is no downtown kind)"),
    ("bike_rack", "a row of steel inverted-U bike racks in the kerb strip"),
)
# The residential catalog's (render/dressing_layout_residential.py): what a drone sees from 7 to 15 m over a street of
# detached houses. None may look like a tracked kind (cars, vans, pickups, skips, dumpsters): no bin, no mailbox, no
# vehicle. A tree is no tracked thing: its stem is dull bark and its canopy, from 5 m up, a dull natural green.
_RESIDENTIAL_DRESSING = (
    ("tree_stem", "the trunk and first limbs of a street or garden tree, up to 5 m, on a verge or a garden bed"),
    ("tree_canopy", "a tree's leaf mass from 5 m up, over its stem; never over a house, a block or a drop zone"),
    ("lamp_post", "a street lamp on a verge, 5 m, its arm towards the carriageway"),
    ("shrub", "a low clump of leaf (0.7 to 1.3 m) on a garden bed"),
    ("playground", "swings, a slide or a climbing frame on a bed in a green, up to 2.4 m"),
)
_BOOK_COLOURS = ("white", "grey", "beige", "brown")  # no tracked book has one: red, blue, green, yellow, black, ...


def _classes(catalog: str, rows: tuple[tuple[str, str], ...]) -> list[DressingClass]:
    return [DressingClass(name, (catalog,), (), 1, note) for name, note in rows]


def _registry(classes: Iterable[DressingClass]) -> dict[str, DressingClass]:
    out: dict[str, DressingClass] = {}
    for c in classes:
        if c.name in out:
            raise ValueError(f"dressing class {c.name!r} is listed twice")
        out[c.name] = c
    return out


DRESSING: dict[str, DressingClass] = _registry([
    *_classes("home", _HOME_DRESSING),
    *_classes("warehouse", _WAREHOUSE_DRESSING),
    *_classes("downtown", _DOWNTOWN_DRESSING),
    *_classes("residential", _RESIDENTIAL_DRESSING),
    DressingClass("rack_stock", ("warehouse",), WAREHOUSE_STOCK_LOOKALIKES, 0,
                  "the cartons on bare pallets that fill the racks' bay-levels without a surface "
                  "(`render/furniture_warehouse.py`); the warehouse's scenery sentence declares them"),
    DressingClass("shelved_book", ("home",), (("book", _BOOK_COLOURS),), 2,
                  "books standing on the bookshelf in colours no tracked book has; the shelves are receptacles, so "
                  "this edits the map; declare it in the apartment's scenery sentence when it lands"),
])


# What else a thing is called: per kind, the phrases that name the same sort of object (lower case, words apart). A
# class named by one of them is, for the pictures, that kind: a crate is a box, a barrel a
# drum, a trash can a bin, a suitcase a bag, a potted plant a plant, a vase a bottle, a rubber duck (the toy's model) a
# toy. A cushion or a pillow is a towel:
# a throw pillow is no towel, but both are soft, saturated, rectangular textile and the towel may lie on a sofa or a bed
# (any surface takes any kind, `placement.find_spot`), so on the pictures they cannot be told apart. Kinds of every
# catalog share the table; `names_a_kind` looks only at the kinds it is given.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "mug": ("cup", "teacup", "coffee cup"),
    "glass": ("tumbler", "drinking glass", "wine glass"),
    "bowl": ("dish", "salad bowl", "soup bowl", "basin"),
    "plate": ("dish", "saucer", "platter", "dinner plate"),
    "book": ("binder", "notebook", "novel", "magazine", "paperback", "textbook"),
    "remote": ("remote control", "controller", "clicker"),
    "bag": ("handbag", "backpack", "suitcase", "satchel", "rucksack", "duffel", "purse", "luggage"),
    "shoes": ("sneakers", "boots", "slippers", "sandals", "socks"),
    "laptop": ("notebook computer", "netbook", "tablet"),
    "plant": ("potted plant", "houseplant", "planter", "flower pot", "succulent", "fern"),
    "bottle": ("vase", "flask", "jug", "carafe"),
    "box": ("crate", "carton", "cardboard box", "parcel", "package", "case"),
    "towel": ("cushion", "pillow", "throw pillow", "washcloth", "blanket", "linen"),
    "toy": ("block", "doll", "teddy", "plush", "ball", "duck", "rubber duck", "rubber ducky", "stuffed animal"),
    "chair": ("seat", "armchair"),
    "stool": ("footstool", "ottoman", "pouf"),
    "laundry_basket": ("basket", "hamper", "waste basket", "wastebasket", "trash can", "waste bin", "trash bin",
                       "rubbish bin"),
    "robot_vacuum": ("vacuum", "roomba", "vacuum cleaner"),
    "home_robot": ("robot",),
    "door": ("door leaf",),
    "pallet": ("skid", "wooden pallet"),
    "tote": ("crate", "plastic crate", "tub", "container"),
    "drum": ("barrel", "keg", "cask", "oil drum", "propane tank"),
    "sack": ("bag", "burlap sack", "grain sack"),
    "bin": ("trash can", "trash bin", "waste basket", "wastebasket", "garbage can", "storage bin"),
    "forklift": ("fork lift", "lift truck"),
    "tote_robot": ("amr", "agv", "mobile robot"),
    "pallet_jack": ("pallet truck", "hand truck", "pump truck"),
    "roll_cage": ("roll container", "cage", "trolley", "cart", "dolly", "ladder"),  # the mission's movable equipment
}


def _stem(word: str) -> str:
    """A crude singular, enough for the names of things: boxes to box, shoes to shoe, glasses to glass."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _words(text: str) -> tuple[str, ...]:
    return tuple(_stem(w) for w in text.lower().replace("_", " ").replace("-", " ").split())


def names_a_kind(name: str, kinds: Iterable[str]) -> list[str]:
    """The kinds among `kinds` that a class name is called: the name ends in the kind's own name or in one of its
    `SYNONYMS` (the last word of a compound is its head noun, so "shelved_book" is a book and "door_casing" is a
    casing; "toilet_roll_holder" is no roll cage). Sorted."""
    words = _words(name)
    out = []
    for kind in kinds:
        if any(words[-len(p):] == p for p in map(_words, (kind, *SYNONYMS.get(kind, ())))):
            out.append(kind)
    return sorted(out)


def class_problems(c: DressingClass) -> list[str]:
    """What is wrong with a class on its own: a phase or catalog that does not exist, no note, or a resemblance that
    `lookalike_problems` rejects in any of its catalogs (a kind the catalog lacks, no colour, or a colour a tracked
    thing has). Empty for a sound class."""
    problems = []
    if c.phase not in PHASES:
        problems.append(f"{c.name}: phase {c.phase} is not one of {PHASES}")
    if not c.catalogs or not c.note:
        problems.append(f"{c.name}: it needs a catalog and a note")
    for catalog in c.catalogs:
        if catalog not in CATALOGS:
            problems.append(f"{c.name}: unknown catalog {catalog!r}")
        elif c.resembles:
            problems += [f"{c.name}: {p}" for p in lookalike_problems(Scenery(catalog, c.resembles))]
    return problems


def dressing_of(catalog: str, phase: int | None = None) -> list[DressingClass]:
    """The registered classes of a catalog (those of one phase when given), in registry order."""
    return [c for c in DRESSING.values() if catalog in c.catalogs and phase in (None, c.phase)]


def layout_problems(catalog: str, emitted: Iterable[tuple[str, int]], max_phase: int = 1) -> list[str]:
    """What is wrong with what a catalog's layout emits, as (class name, phase) pairs: a class that is not registered
    for the catalog, one emitted in another phase than the registry gives it, one above `max_phase` (phase 2 edits the
    world, so a render-only build never draws it), and every registered phase 1 class of the catalog that is never
    emitted (a class no layout uses is a name nobody can see on a contact sheet). Sorted."""
    problems, seen = [], set()
    for name, phase in emitted:
        seen.add(name)
        c = DRESSING.get(name)
        if c is None or catalog not in c.catalogs:
            problems.append(f"{name} is not a registered dressing class of the {catalog} catalog")
        elif phase != c.phase:
            problems.append(f"{name} is emitted in phase {phase} but registered in phase {c.phase}")
        elif phase > max_phase:
            problems.append(f"{name} is phase {phase}: a phase {max_phase} build never draws it")
    problems += [f"{c.name} is a phase 1 class of the {catalog} catalog that the layout never emits"
                 for c in dressing_of(catalog, 1) if c.name not in seen]
    return sorted(set(problems))


# -- clutter -------------------------------------------------------------------------------------------------------
def clutter_problems(catalog: Catalog, askable: Iterable[str]) -> list[str]:
    """What is wrong with a catalog's clutter (`Catalog.clutter`, world content no question names): a kind whose name is
    an askable kind's or one of its synonyms (`names_a_kind`: a cushion is a towel, a sock a pair of shoes, a waste bin
    a laundry basket) unless it declares that resemblance; a declared resemblance that is not a tracked kind of the
    catalog, has no colour, or is in a colour a tracked thing of that kind has (`lookalike_problems`: it would be a
    thing the robot is told to ignore); and a kind with no model, an empty or inverted count, no place, a place that is
    the floor and a surface at once, or a name another kind has. `askable` lists every kind a question can name: the
    catalog's, its movers', the map's light furniture and the door. Empty for sound clutter."""
    problems, seen = [], set()
    for k in catalog.clutter:
        declared = {kind for kind, _ in k.resembles}
        problems += [f"{k.name}: is called what the askable {a} is called (declare it in `resembles`, in colours no "
                     f"tracked {a} has)" for a in names_a_kind(k.name, askable) if a not in declared]
        problems += [f"{k.name}: {p}" for p in lookalike_problems(Scenery(catalog.name, k.resembles))]
        if k.name in seen:
            problems.append(f"{k.name}: two clutter kinds of that name")
        seen.add(k.name)
        if not k.models or not k.colors or not k.places:
            problems.append(f"{k.name}: it needs models, colours and places")
        if not 0 <= k.count[0] <= k.count[1]:
            problems.append(f"{k.name}: its count {k.count} is not a range from 0")
        floor = {place_of(p)[0] == "floor" for p in k.places}
        if len(floor) > 1:
            problems.append(f"{k.name}: it stands on the floor and on a surface (a kind is one or the other)")
    return problems
