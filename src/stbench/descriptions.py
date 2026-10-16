"""Descriptions: how a thing is named in a question, and how a name is matched to a thing.

A description is a kind plus the attributes a person would say (`size`, `color`). Detections and entities may carry
more keys (a furniture name, receptacles); matching reads only DESC_KEYS, so the vocabulary has one owner. A question
slot may give a subset of the attributes ("mug", "red mug"); a thing matches when its kind and every given attribute
are equal. Both the question compiler and the baselines import this module, so it depends on nothing else.
"""

from __future__ import annotations

from collections.abc import Mapping

DESC_KEYS = ("size", "color")
Key = tuple[str, tuple[tuple[str, str], ...]]

# Kinds a person could mistake for one another at the camera's resolution (from the catalogs'
# kinds). A description of one with the colour of a thing of another in its group is no clear absence: the question
# compiler offers an absent description only when no sighted thing of its own kind or such a kind could be taken for its
# colour, and asks a real thing only when no other sighted thing of such a kind could be (`questions_where`; "taken
# for" is `World.may_be_taken_for`: the colour its render shows, else its label; black, grey and white are one group
# there). The groups overlap (a bowl looks like a mug, a glass and a plate; a bag like a box or a tote and a laundry
# basket), so `lookalikes` unions the groups of one kind. The 0.5.0 log review added what caught its agent: the toy's
# stacked blocks for a box, a clothes pile for a toy, and a book stack or a shelved book for a book. The clothes pile
# and the books are clutter kinds of the home catalog (never asked, but sighted like a tracked kind:
# `World.sighted_looks`). The groups screen a seen thing and a never-seen one alike. The 0.6.0 look-dev review added
# two pairs of home clutter that render alike, a floor tower or a pedal-bin cylinder in the same colours (an air
# purifier and a space heater, a pedal bin and a bath bin; unlike the bin and the drum they differ in no shape), which
# `ego_now` and `ego_mem` name by kind alone. The 0.7.0 given-name check found the home's soft goods rendered alike
# side by side, folded clothes beside a towel, a book and a jacket, none of them in a group: a stack of folded clothes
# reads as a towel or a book, and a jacket, a hanging coat, a clothes pile and folded clothes are all cloth heaps. A
# person could take one for another in a single frame, so a question must not name one while another of its colour is
# sighted: three more groups, which overlap the book group and each other, so `lookalikes` unions them. The 0.7.0 dev
# pilot's log review found two pairs the groups missed: a "red book" never seen was answered at a folded red towel (the
# towel and the book each shared a group with folded clothes, none with each other), and "the cable bundle" at an
# extension cord. The two folded-clothes groups became one, {folded clothes, towel, book, book stack}, and a cable
# bundle and an extension cord are a group.
CONFUSABLE = (frozenset({"tote", "bin", "box", "bag"}), frozenset({"van", "box_truck", "pickup", "car"}),
              frozenset({"bag", "laundry_basket"}), frozenset({"mug", "glass", "bowl"}), frozenset({"bowl", "plate"}),
              frozenset({"dumpster", "skip"}), frozenset({"chair", "stool"}), frozenset({"toy", "box"}),
              frozenset({"toy", "clothes_pile"}), frozenset({"book", "book_stack", "shelved_book"}),
              frozenset({"air_purifier", "space_heater"}), frozenset({"pedal_bin", "bath_bin"}),
              frozenset({"folded_clothes", "towel", "book", "book_stack"}),
              frozenset({"cable_bundle", "extension_cord"}),
              frozenset({"jacket", "hanging_coat", "clothes_pile", "folded_clothes"}))
# Kinds a name is stretched to when the thing it names was never seen (the AMR's log review, defect 2: "the yellow bin"
# answered with the yellow drum in the bay, "the blue sack" with the blue pallet's wrapped load). A bin, a drum, a sack
# and a wrapped pallet are different shapes at the camera's range, so a seen drum or pallet is a fair identification
# test and these groups never hide a question about a seen thing. They count only where nothing was seen to say which
# one is meant: the questions about a thing no frame of the window shows, and the absent descriptions.
CONFUSABLE_UNSEEN = (frozenset({"bin", "drum"}), frozenset({"sack", "pallet"}))
CONFUSED_KINDS = frozenset().union(*CONFUSABLE)  # every kind some group holds

_DISPLAY = {"glass": "drinking glass"}  # a bare "glass" reads as the material
# kinds whose name is a plural noun come as a unit: (singular prefix, plural prefix, noun) -> "pair of red shoes",
# "pairs of shoes". A question says "Where is the pair of dumbbells", never "Where is the dumbbells".
_UNITS = {"shoes": ("pair of", "pairs of", "shoes"), "dumbbells": ("pair of", "pairs of", "dumbbells"),
          "folded_clothes": ("stack of", "stacks of", "folded clothes")}


def desc_attrs(attributes: Mapping[str, object]) -> dict[str, str]:
    """The describable attributes of a thing, in DESC_KEYS order."""
    return {k: str(attributes[k]) for k in DESC_KEYS if k in attributes}


def desc_key(kind: str, attributes: Mapping[str, object]) -> Key:
    """A hashable full description: (kind, sorted describable attributes)."""
    return (kind, tuple(sorted(desc_attrs(attributes).items())))


def lookalikes(kind: str, unseen: bool = False) -> frozenset[str]:
    """The kind and every kind it can be mistaken for (`CONFUSABLE`): the union of the groups that hold it. For a
    thing nothing has shown (`unseen`: an unseen real thing, an absent description) `CONFUSABLE_UNSEEN` counts too."""
    groups = CONFUSABLE + CONFUSABLE_UNSEEN if unseen else CONFUSABLE
    return frozenset({kind}).union(*(group for group in groups if kind in group))


def key_attrs(key: Key) -> dict[str, str]:
    return dict(key[1])


def matches(kind: str, attributes: Mapping[str, object], want_kind: str, want_attrs: Mapping[str, object]) -> bool:
    """True when the thing is of `want_kind` and carries every attribute in `want_attrs` with the same value."""
    have = desc_attrs(attributes)
    return kind == want_kind and all(have.get(k) == str(v) for k, v in want_attrs.items())


def kind_word(kind: str, plural: bool = False) -> str:
    """The kind as words, without attributes: "mug", "mugs", "pair of shoes", "pairs of shoes"."""
    if kind in _UNITS:
        one, many, noun = _UNITS[kind]
        return f"{many if plural else one} {noun}"
    word = _DISPLAY.get(kind, kind).replace("_", " ")
    if not plural:
        return word
    return word + ("es" if word.endswith(("s", "x", "ch", "sh")) else "s")


def phrase(kind: str, attributes: Mapping[str, object], plural: bool = False) -> str:
    """Words for a description: "large red bowl", "pair of red shoes". Size first, then colour, then the kind."""
    a = desc_attrs(attributes)
    adjectives = [a[k] for k in DESC_KEYS if k in a]
    if kind in _UNITS:
        one, many, noun = _UNITS[kind]
        return " ".join([many if plural else one, *adjectives, noun])
    return " ".join([*adjectives, kind_word(kind, plural)])
