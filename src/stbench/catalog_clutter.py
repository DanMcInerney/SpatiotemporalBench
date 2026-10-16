"""The household's clutter: world content that no question names, as data.

A `ClutterKind` is the toaster, the router, the floor fan, the books on a shelf. It is a catalog's second vocabulary
(`catalog.Catalog.clutter`), never part of `kinds`, so nothing that samples, describes or asks treats it as an item.
`places` says where it stands: "floor" (or "floor:<piece kind>", beside a piece: the toilet's mat), a surface kind of
the map, "on:<piece kind>" (the top of a fixed piece the map gives no surface, the stove's) or "front:<piece kind>"
(the face of a piece, the fridge's, where paper is held by magnets), each optionally followed by `@` and a room's name
("sink edge@bathroom"). `count` is how many a script holds;
`models` are the manifest's `clutter` assets, each with its own measured box (an entity's size is its model's, `size` is
the box that holds them all) and, when untinted, its own colour. `resembles` declares a look-alike of a tracked kind in
colours no tracked thing has (`scenery.clutter_problems`). `HOME_FLOOR_KINDS` names the tracked kinds that real floors
hold, and per kind the furniture kinds it is dropped beside (`clutter.place` moves about half of them).

What a lived-in apartment holds that no question names: appliances, containers and papers on the counters,
the fridge's door, the dining table and the hall console; lamps and clocks on the tables; books on the bookshelf; cables
and fans on the floor; a mat, a bin and a brush in the bathroom. `size` holds every model's measured box. Out, because a
picture reads them as a tracked kind: cushions and pillows (the towel's synonyms), socks (shoes), the waste bin (the
laundry basket's), and any tableware, flower arrangement, monitor or tissue box (a mug, plant, laptop or box). Floor
kinds are the ones people keep on the floor (a heater, fan and purifier are grey or black because the box is brown,
white, blue, red or green; dumbbells are black or grey because the toy is red, blue, yellow, green, orange or pink; a
side table is white or grey, as from 8 cm a stool is four legs and an underside). Cloth and mats are beige, black,
brown, purple or pink, the colours no tracked towel has; bottles of soap and shampoo are white, yellow, orange, pink
or purple, the colours no tracked bottle has. The counts are a real home's: one toaster, one router, and books, fruit,
papers and frames that repeat; their sums reach the floor and surface quotas of `clutter.place`. `shelved_book`
declares what it resembles: a book, in colours no tracked book has.
"""

from __future__ import annotations

from dataclasses import dataclass

FLOOR = "floor"  # the place of a clutter kind that stands on the floor (in a room: "floor@bathroom")
ROOM_SEP = "@"  # a place token's optional room: "sink edge@bathroom"
ON = "on:"  # a place token's prefix for the top of a fixed piece of a kind: "on:stove"
FRONT = "front:"  # and for its front face: "front:fridge"


def place_of(token: str) -> tuple[str, str | None]:
    """A place token as (a surface kind, "floor", "floor:<kind>", "on:<kind>" or "front:<kind>", and a room name or
    None)."""
    where, _, room = token.partition(ROOM_SEP)
    return where, room or None


def floor_beside(token: str) -> str | None:
    """The piece kind a floor token stands beside ("floor:toilet@bathroom" is "toilet"), None for the open floor."""
    where = place_of(token)[0]
    return where.partition(":")[2] if where.partition(":")[0] == FLOOR and ":" in where else None


@dataclass(frozen=True)
class ClutterKind:
    name: str  # never an askable kind or one's synonym (`scenery.names_a_kind`) unless it declares `resembles`
    size: tuple[float, float, float]  # the box that holds every model's box (an entity's size is its model's)
    colors: tuple[str, ...]  # palette keys: an untinted model's own colour, or the tints a procedural one takes
    places: tuple[str, ...]  # "floor", surface kinds, "on:<piece kind>" or "front:<piece kind>", each maybe "@<room>"
    count: tuple[int, int]  # per script, inclusive
    models: tuple[str, ...]  # manifest assets of role "clutter"
    resembles: tuple[tuple[str, tuple[str, ...]], ...] = ()  # declared: ("book", (white, grey, beige, brown))

    @property
    def on_floor(self) -> bool:
        return all(place_of(p)[0].partition(":")[0] == FLOOR for p in self.places)

    @property
    def on_front(self) -> bool:
        return all(place_of(p)[0].startswith(FRONT) for p in self.places)


def _c(name, size, colors, places, count, models, resembles=()):
    return ClutterKind(name, size, tuple(colors), tuple(places), count, tuple(models),
                       tuple((k, tuple(c)) for k, c in resembles))


# What real floors hold, and beside which furniture kinds (names of the apartment's pieces): shoes by the door (the mat,
# the rack, the hall console) and by the wardrobe and the sofa, a bag by the desk, the hall console and the sofa, a toy
# near the sofa, the coffee table and the rug, a box by the console and the dining table.
HOME_FLOOR_KINDS = (
    ("shoes", ("floor_mat", "shoe_rack", "hall_console", "wardrobe", "sofa")),
    ("bag", ("desk", "hall_console", "sofa")),
    ("toy", ("sofa", "coffee_table", "rug")),
    ("box", ("hall_console", "dining_table")),
)

PAPER = ("white", "beige", "yellow", "pink")
CLOTH = ("beige", "black", "brown", "purple", "pink")
SOAP = ("white", "yellow", "orange", "pink", "purple")
DRY_FLOOR = ("floor@hall", "floor@living room", "floor@kitchen", "floor@bedroom")  # where cables lie: no bathroom floor

HOME_CLUTTER = (
    # on the floor
    _c("floor_fan", (0.22, 0.34, 0.47), ("black", "grey"), ("floor@bedroom", "floor@living room"), (0, 1),
       ("proc_floor_fan",)),
    _c("air_purifier", (0.25, 0.25, 0.55), ("grey", "black"), ("floor@bedroom", "floor@living room"), (0, 1),
       ("proc_air_purifier",)),
    _c("space_heater", (0.16, 0.22, 0.55), ("grey", "black"), ("floor@bedroom", "floor@living room"), (0, 1),
       ("proc_space_heater",)),
    _c("bathroom_scale", (0.3, 0.3, 0.03), ("black", "grey"), ("floor@bathroom",), (1, 1), ("proc_bathroom_scale",)),
    _c("plunger", (0.175, 0.175, 0.425), ("red",), ("floor@bathroom",), (1, 1), ("ph_plunger",)),
    _c("toilet_brush", (0.11, 0.11, 0.38), ("white",), ("floor@bathroom",), (1, 1), ("proc_toilet_brush",)),
    _c("bath_mat", (0.4, 0.6, 0.015), CLOTH, ("floor:bathtub@bathroom",), (1, 1), ("proc_bath_mat",)),
    _c("pedestal_mat", (0.4, 0.5, 0.012), CLOTH, ("floor:toilet@bathroom",), (1, 1), ("proc_pedestal_mat",)),
    _c("pedal_bin", (0.266, 0.266, 0.42), ("grey", "white", "black"), ("floor@kitchen",), (1, 1), ("proc_pedal_bin",)),
    _c("bath_bin", (0.2, 0.2, 0.3), ("grey", "white", "black"), ("floor@bathroom",), (1, 1), ("proc_bath_bin",)),
    _c("boot_tray", (0.3, 0.4, 0.035), ("grey", "black", "brown"), ("floor:shoe_rack@hall",), (1, 1),
       ("proc_boot_tray",)),
    _c("drying_rack", (0.55, 0.9, 1.005), CLOTH, ("floor@living room", "floor@bedroom"), (0, 1),
       ("proc_drying_rack",)),
    _c("clothes_pile", (0.3, 0.4, 0.137), CLOTH, ("floor@bedroom", "floor@living room"), (1, 3),
       ("proc_clothes_pile",)),
    _c("umbrella", (0.85, 0.075, 0.075), ("black", "blue", "red", "grey"), ("floor@hall", "floor@living room"), (0, 2),
       ("proc_umbrella",)),
    _c("yoga_mat", (0.62, 0.15, 0.15), ("purple", "grey", "black"), ("floor@bedroom", "floor@living room"), (0, 2),
       ("proc_yoga_mat",)),
    _c("dumbbells", (0.28, 0.28, 0.11), ("black", "grey"), ("floor@bedroom", "floor@living room"), (0, 3),
       ("proc_dumbbells",)),
    _c("extension_cord", (0.5, 0.18, 0.03), ("white", "black", "orange"), DRY_FLOOR, (2, 5), ("proc_extension_cord",)),
    _c("power_strip", (0.5, 0.17, 0.04), ("white", "black"), DRY_FLOOR, (2, 4), ("proc_power_strip",)),
    _c("cable_bundle", (0.18, 0.16, 0.035), ("white", "black"), DRY_FLOOR, (3, 8), ("proc_cable_bundle",)),
    _c("door_stop", (0.1, 0.045, 0.035), ("grey", "black", "beige"), ("floor",), (2, 4), ("proc_door_stop",)),
    _c("side_table", (0.51, 0.55, 0.62), ("white", "grey"), ("floor@bedroom", "floor@living room"), (0, 2),
       ("ph_side_table_01", "ph_painted_wooden_nightstand")),
    _c("floor_lamp", (0.34, 0.34, 1.5), ("black", "grey", "white"), ("floor@bedroom", "floor@living room"), (1, 2),
       ("proc_floor_lamp",)),
    # on surfaces: the kitchen's counters, stove and sink, then the tables, shelves, desk and bed
    _c("microwave", (0.36, 0.46, 0.27), ("white", "black", "grey"), ("kitchen counter",), (1, 1),
       ("proc_microwave",)),
    _c("stand_mixer", (0.22, 0.36, 0.36), ("red", "white", "black"), ("kitchen counter",), (0, 1),
       ("proc_stand_mixer",)),
    _c("blender", (0.17, 0.17, 0.4), ("black", "white", "grey"), ("kitchen counter",), (0, 1), ("proc_blender",)),
    _c("canister", (0.12, 0.12, 0.22), ("white", "grey", "black", "beige"), ("kitchen counter",), (0, 4),
       ("proc_canister_tall", "proc_canister_short")),
    _c("kitchen_roll", (0.15, 0.15, 0.3), ("white", "beige"), ("kitchen counter", "sink edge@kitchen"), (1, 1),
       ("proc_paper_towel",)),
    _c("soap_pump", (0.0625, 0.06, 0.2), SOAP[:4], ("sink edge",), (1, 2), ("proc_soap_pump",)),
    _c("dish_rack", (0.33, 0.42, 0.165), ("grey", "white", "black"), ("kitchen counter",), (1, 1),
       ("proc_dish_rack",)),
    _c("stock_pot", (0.271, 0.327, 0.19), ("grey", "black"), ("on:stove",), (0, 2), ("proc_stock_pot",)),
    _c("frying_pan", (0.45, 0.28, 0.069), ("black", "grey"), ("on:stove",), (0, 1), ("proc_frying_pan",)),
    _c("fridge_paper", (0.004, 0.25, 0.32), PAPER, ("front:fridge",), (4, 9),
       ("proc_fridge_sheet", "proc_fridge_calendar", "proc_fridge_photo")),
    _c("mail_pile", (0.162, 0.207, 0.03), ("white", "beige"),
       ("dining table", "hall console", "desk", "coffee table"), (1, 4), ("proc_mail_pile",)),
    _c("placemat", (0.32, 0.44, 0.006), ("beige", "grey", "brown", "black"), ("dining table",), (0, 4),
       ("proc_placemat",)),
    _c("book_stack", (0.186, 0.255, 0.075), ("white", "grey", "beige", "brown"),
       ("coffee table", "nightstand", "desk", "sofa", "bed", "dining table", "tv console"), (1, 6),
       ("proc_book_stack",), resembles=(("book", ("white", "grey", "beige", "brown")),)),
    _c("jacket", (0.36, 0.493, 0.132), ("black", "brown", "beige"), ("hall console", "sofa", "bed", "desk"), (0, 3),
       ("proc_jacket",)),
    _c("folded_clothes", (0.248, 0.289, 0.099), CLOTH, ("bed", "sofa", "shelf@bathroom", "nightstand"), (1, 5),
       ("proc_folded_clothes",)),
    _c("shampoo", (0.045, 0.065, 0.2), SOAP, ("shelf@bathroom", "sink edge@bathroom"), (2, 5), ("proc_shampoo",)),
    _c("notepad", (0.21, 0.3, 0.02), ("white", "beige", "yellow"),
       ("desk", "nightstand", "coffee table"), (0, 3), ("proc_notepad",)),
    _c("speaker", (0.16, 0.16, 0.3), ("black", "grey"), ("tv console",), (0, 2), ("proc_speaker",)),
    _c("hanging_coat", (0.081, 0.459, 0.795), ("black", "brown", "beige"), ("front:wardrobe",), (0, 2),
       ("proc_hanging_coat",)),
    _c("shelved_book", (0.095, 0.038, 0.19), ("white", "grey", "beige", "brown"),
       ("shelf@living room", "desk", "tv console", "nightstand"), (8, 30),
       ("proc_shelved_book_s", "proc_shelved_book_m", "proc_shelved_book_l"),
       resembles=(("book", ("white", "grey", "beige", "brown")),)),
    _c("produce", (0.17, 0.13, 0.1), ("red", "yellow", "green", "brown"), ("kitchen counter", "dining table"), (2, 12),
       ("ph_food_apple_01", "ph_food_pears_asian_01", "ph_lemon", "ph_food_lime_01", "ph_yellow_onion")),
    _c("picture_frame", (0.09, 0.185, 0.23), ("black", "white"),
       ("shelf@living room", "tv console", "nightstand", "hall console", "desk", "coffee table"), (3, 12),
       ("ph_standing_picture_frame_01", "ph_standing_picture_frame_02")),
    _c("ornament", (0.175, 0.11, 0.2), ("brown", "grey"),
       ("shelf@living room", "tv console", "hall console", "nightstand", "coffee table"), (0, 10),
       ("ph_carved_wooden_elephant", "ph_concrete_cat_statue")),
    _c("utensil", (0.09, 0.28, 0.03), ("brown",), ("kitchen counter",), (0, 4), ("ph_wooden_spoon",)),
    _c("bread", (0.08, 0.2, 0.065), ("brown",), ("kitchen counter",), (0, 3), ("ph_croissant",)),
    _c("toaster", (0.17, 0.3, 0.195), ("grey",), ("kitchen counter",), (1, 1), ("gso_toaster_bd_tr3500sd",)),
    _c("kettle", (0.2, 0.26, 0.245), ("grey",), ("kitchen counter",), (1, 1), ("ph_vintage_electric_kettle",)),
    _c("coffee_maker", (0.16, 0.23, 0.24), ("black",), ("kitchen counter",), (1, 1),
       ("gso_coffee_maker_bd_cm2035b",)),
    _c("cutting_board", (0.21, 0.385, 0.04), ("brown",), ("kitchen counter",), (0, 2), ("ph_wooden_cutting_board",)),
    _c("desk_lamp", (0.105, 0.31, 0.45), ("orange",), ("desk", "nightstand"), (1, 2), ("ph_desk_lamp_arm_01",)),
    _c("alarm_clock", (0.055, 0.11, 0.165), ("blue", "grey"), ("nightstand",), (1, 2),
       ("ph_alarm_clock_01", "gso_alarm_clock_crosley")),
    _c("router", (0.14, 0.24, 0.19), ("black", "white"), ("tv console", "desk"), (1, 1), ("proc_router",)),
    _c("keyboard", (0.2, 0.445, 0.075), ("white", "black"), ("desk",), (1, 1),
       ("gso_keyboard_kanex", "gso_keyboard_razer_bw_stealth")),
    _c("hair_dryer", (0.18, 0.235, 0.065), ("red",), ("sink edge@bathroom",), (1, 1), ("gso_hair_dryer_remington",)),
    _c("radio", (0.05, 0.13, 0.24), ("grey",), ("nightstand", "shelf@living room", "kitchen counter"), (0, 1),
       ("ph_cassette_player",)),
    _c("stapler", (0.05, 0.21, 0.1), ("blue",), ("desk",), (0, 1), ("ph_vintage_stapler",)),
)
