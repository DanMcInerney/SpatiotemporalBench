"""Descriptions: the wording of a question and the matching of a detection to a description."""

from __future__ import annotations

import pytest

from stbench.catalog import KINDS
from stbench.descriptions import DESC_KEYS, desc_attrs, desc_key, kind_word, lookalikes, matches, phrase


def test_matching_reads_only_the_describable_attributes():
    """Contract: a description matches on kind and the given attributes; extra keys on the thing (a furniture name,
    a receptacle list) never take part. Regression caught: comparing whole attribute dicts, so a detection carrying
    an extra key stops matching its question."""
    thing = {"color": "red", "name": "mug 3", "receptacles": ["desk"]}
    assert matches("mug", thing, "mug", {"color": "red"})
    assert matches("mug", thing, "mug", {})
    assert not matches("mug", thing, "mug", {"color": "blue"})
    assert not matches("mug", thing, "mug", {"size": "large"})
    assert not matches("bowl", thing, "mug", {})
    assert desc_attrs(thing) == {"color": "red"}
    assert desc_key("mug", thing) == ("mug", (("color", "red"),))


def test_words_put_size_before_colour_before_kind():
    """Contract: "large red bowl", plural "large red bowls"; the kind "glass" is a drinking glass, never the bare word;
    kinds that come as a unit read naturally ("pair of red shoes", "pairs of red shoes", "pair of shoes") with the
    adjectives inside the phrase.
    Regression caught: a question that says "glass" for a drinking glass, or "red shoes" read as one shoe."""
    assert phrase("bowl", {"color": "red", "size": "large"}) == "large red bowl"
    assert phrase("bowl", {"color": "red", "size": "large"}, plural=True) == "large red bowls"
    assert phrase("glass", {"color": "blue"}) == "blue drinking glass"
    assert phrase("glass", {}, plural=True) == "drinking glasses"
    assert phrase("shoes", {"color": "red"}) == "pair of red shoes"
    assert phrase("shoes", {"color": "red"}, plural=True) == "pairs of red shoes"
    assert phrase("shoes", {}) == "pair of shoes" and kind_word("shoes", plural=True) == "pairs of shoes"
    assert kind_word("box", plural=True) == "boxes"
    assert kind_word("laundry_basket") == "laundry basket"


def test_lookalikes_union_the_groups_a_kind_belongs_to():
    """Contract: `lookalikes(kind)` is the kind and every kind it can be mistaken for at the camera's resolution, the
    union of the `CONFUSABLE` groups that hold it, so a kind in two groups takes both and a kind in none is alone.
    Hand values: a box also looks like a tote, a bin and a bag (0.5.0's fix round: a bag beside a same-coloured box was
    asked) and, since 0.5.0-control, a toy (its stacked blocks: a green toy for a "green box"), while a tote, a bin and
    a bag do not look like a toy (the groups are not transitive); a toy like a clothes pile (clutter), a clothes pile
    also like a jacket, a hanging coat and folded clothes (0.7.0), a book like a book stack, a shelved book (clutter),
    a towel and folded clothes (0.7.0: a towel is a book's look-alike too); a cable bundle like an extension cord and
    back (the dev pilot's log review); a van like a box truck, a pickup and a car; a bag like a laundry basket as well
    as a box, a tote and a bin (it is in two groups; the basket and the box are not each other's); a mug like a glass
    and a bowl, a plate only like a bowl, and a bowl like a mug, a glass and a plate (it is in two groups; mug and
    plate are not each other's); a skip like a dumpster, a chair like a stool; a sofa is only itself. Since the
    AMR's log review `lookalikes(kind, unseen=True)`, for a thing nothing has shown, adds the `CONFUSABLE_UNSEEN`
    groups: a bin then takes the box group and the drum, a drum the bin, a sack the pallet and a pallet the sack, a tote
    still only the box group, and every other kind is as without. Regression caught: a kind with two groups taking only
    one, a bag left out of the tote, bin and box group, a review's look-alike (toy blocks, clothes pile, book stack)
    missing, a group merged into its neighbour's (a bag taking a toy, a tote taking a drum), the unseen groups leaking
    into the seen kinds' look-alikes (a drum taking a bin when seen), or a kind left out of its own lookalikes (a
    description of the kind in a sighted colour is a near miss too)."""
    box_group = {"box", "tote", "bin", "bag"}
    assert lookalikes("box") == {*box_group, "toy"}
    assert lookalikes("tote") == box_group and lookalikes("bin") == box_group
    assert lookalikes("toy") == {"toy", "box", "clothes_pile"}
    assert lookalikes("clothes_pile") == {"clothes_pile", "toy", "jacket", "hanging_coat", "folded_clothes"}
    books = {"book", "book_stack", "shelved_book"}
    assert lookalikes("book") == {*books, "folded_clothes", "towel"} and lookalikes("shelved_book") == books
    assert lookalikes("towel") == {"towel", "folded_clothes", "book", "book_stack"}   # 0.7.0: a towel is a book's too
    assert lookalikes("cable_bundle") == {"cable_bundle", "extension_cord"}
    assert lookalikes("extension_cord") == {"extension_cord", "cable_bundle"}
    assert lookalikes("van") == {"van", "box_truck", "pickup", "car"}
    assert lookalikes("bag") == {*box_group, "laundry_basket"}
    assert lookalikes("laundry_basket") == {"laundry_basket", "bag"}
    assert lookalikes("mug") == {"mug", "glass", "bowl"} and lookalikes("plate") == {"plate", "bowl"}
    assert lookalikes("bowl") == {"bowl", "mug", "glass", "plate"}
    assert lookalikes("skip") == {"skip", "dumpster"} and lookalikes("chair") == {"chair", "stool"}
    assert lookalikes("drum") == {"drum"} and lookalikes("sack") == {"sack"} and lookalikes("pallet") == {"pallet"}
    assert lookalikes("bin", unseen=True) == {*box_group, "drum"} and lookalikes("tote", unseen=True) == box_group
    assert lookalikes("drum", unseen=True) == {"drum", "bin"}
    assert lookalikes("sack", unseen=True) == {"sack", "pallet"} == lookalikes("pallet", unseen=True)
    assert lookalikes("box", unseen=True) == {*box_group, "toy"} and lookalikes("sofa", unseen=True) == {"sofa"}
    assert lookalikes("sofa") == {"sofa"}


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_every_catalog_kind_has_a_distinct_plural_word_form(kind):
    """Contract: the question wording for every catalog kind is non-empty and its plural is a readable word."""
    assert kind_word(kind) and kind_word(kind, plural=True)


def test_describable_keys_are_what_perception_emits():
    """Contract: DESC_KEYS is the visual attribute vocabulary of the oracle channel (`perception.VISUAL_ATTRIBUTES`).
    Regression caught: a new visual attribute added to perception but not to the question vocabulary."""
    from stbench.perception import VISUAL_ATTRIBUTES

    assert set(DESC_KEYS) == set(VISUAL_ATTRIBUTES)
