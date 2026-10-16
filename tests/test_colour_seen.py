"""The colour a render shows (`colour_seen`, 0.5.0-control, review defect 1): an sRGB median is named with one of the
palette's twelve words, or with the words of the boundary it is near. Hand-made colours, chosen well inside a band, or
(for the margin) at a boundary worked out on paper; no expectation is read from the code under test."""

from __future__ import annotations

import pytest

from stbench.colour_seen import WORDS, classify, could_be, held, label_reads, taken_for
from synth import RENDER_RGB


@pytest.mark.parametrize("rgb, word", [
    ((0, 0, 0), "black"), ((15, 15, 15), "black"), ((45, 45, 45), "black"),
    ((85, 85, 85), "grey"), ((100, 100, 100), "grey"),
    ((160, 160, 160), "white"), ((225, 225, 225), "white"), ((255, 255, 255), "white"),
    ((85, 85, 90), "grey"),  # a neutral with the faint blue cast of a white-balanced picture is still grey
])
def test_a_neutral_is_black_grey_or_white_by_its_lightness(rgb, word):
    """Contract: a colour with no hue (chroma under 8 in CIELAB) is black under L 24, white from L 50 and grey between,
    and a cast of a few units of chroma does not make it a colour. Hand values in sRGB (equal channels, L worked out
    from the sRGB transfer): 0, 15 and 45 are L 0, 4 and 18 (black); 85 and 100 are L 36 and 42 (grey); 160, 225 and
    255 are L 66, 90 and 100 (white); (85, 85, 90) has chroma 3. Regression caught: a band moved, a neutral with a
    camera's cast read as blue."""
    assert classify(rgb) == {word}


@pytest.mark.parametrize("rgb, words", [
    ((57, 57, 57), {"black", "grey"}),  # L 24.0: the black / grey edge
    ((119, 119, 119), {"grey", "white"}),  # L 50.0: the grey / white edge, the camera's metered mean
    ((210, 60, 30), {"red", "orange"}),  # hue 41 degrees: the red / orange edge is at 40
])
def test_a_colour_at_a_boundary_has_no_confident_class_and_names_both_sides(rgb, words):
    """Contract: a colour within 3 CIELAB units of a boundary is not named by one word: the answer is the words on its
    sides, which is what the screens read ("could be taken for"). Hand values: a grey of sRGB 57 is L 24.0, the black
    edge, and 119 is L 50.0, the white edge; (210, 60, 30) is at Lab hue 41 degrees, one from the red/orange edge at
    40. Regression caught: a boundary colour forced into one band (a thing near the edge is the look-alike of both)."""
    assert classify(rgb) == words


@pytest.mark.parametrize("word, rgb", [
    ("red", (200, 30, 30)), ("orange", (240, 130, 20)), ("yellow", (240, 210, 30)), ("green", (30, 150, 50)),
    ("blue", (30, 60, 200)), ("purple", (140, 40, 160)), ("pink", (240, 140, 180)),
])
def test_a_vivid_colour_is_its_hue_family(word, rgb):
    """Contract: a colour with chroma, 22 or more when warm, takes its hue family's word by its CIELAB hue angle (red
    about 12 to 40 degrees, orange 40 to 72, yellow 72 to 118, green 118 to 220, blue 220 to 305, purple 305 to 332,
    pink 332 to 12). Hand values are the textbook primaries and secondaries: red (200, 30, 30), orange (240, 130, 20),
    yellow (240, 210, 30), green (30, 150, 50), blue (30, 60, 200), purple (140, 40, 160), pink (240, 140, 180).
    Regression caught: a family edge moved past a palette colour, pink and red or blue and purple swapped."""
    assert classify(rgb) == {word}


def test_a_muted_warm_colour_is_brown_when_dark_and_beige_when_light_and_a_vivid_one_stays_orange():
    """Contract: in the warm families a colour under chroma 22 is muted: brown under L 38, beige from it; a vivid
    colour keeps its family's word however dark (a dim orange bag is still orange, a dark chroma-less one is black).
    Hand values: (75, 55, 40) is L 25, chroma 14 (a dark muted orange: brown); (190, 175, 150) is L 72, chroma 15 (the
    same hue, light: beige); (120, 70, 30) is L 35, chroma 37, hue 61 (dark but vivid: orange); (25, 18, 14) is L 7,
    chroma 4 (a very dark brown has no colour left: black). Regression caught: brown and beige merged or swapped, a
    dark vivid orange called brown (the 0.5.0 renders' dim oranges), or a nearly black brown named for a hue it cannot
    show."""
    assert classify((75, 55, 40)) == {"brown"} and classify((190, 175, 150)) == {"beige"}
    assert classify((120, 70, 30)) == {"orange"} and classify((25, 18, 14)) == {"black"}


def test_the_kits_faithful_colours_are_each_named_by_their_own_word():
    """Contract: `synth.RENDER_RGB`, the sRGB a faithful synthetic picture gives each palette colour, is a colour that
    `classify` names with that word alone, so a test world that says nothing of a render keeps its labels true.
    Regression caught: a kit colour edited to a boundary (every kit test's things would drift into 'no confident
    class'), or a palette word the kit lacks."""
    assert set(RENDER_RGB) == WORDS
    assert {word: classify(rgb) for word, rgb in RENDER_RGB.items()} == {word: {word} for word in WORDS}


def test_a_colour_needs_three_channels():
    """Contract: `classify` takes an sRGB triple and refuses anything else, so a visibility row with a short `rgb_med`
    stops the compile and is never read as a colour. Regression caught: a two-channel median classed as a neutral."""
    with pytest.raises(ValueError, match="three channels"):
        classify([10, 20])


def test_held_is_the_one_word_that_the_most_sightings_confidently_show():
    """Contract: `held(classes)` is the confident class (a single word) that the most sightings show, alone: a tie for
    the most is no word, a sighting near a boundary (several words) votes for none, and none is confident for no
    sightings. Hand values: three black and two grey hold black; two black, two grey and one white tie (nothing); two
    black, one grey and one white hold black (the most, alone); three black and one {black, grey} hold black; two black
    and two {black, grey} hold black too (the boundary ones do not vote, so they cannot veto); two {black, grey} and no
    confident one, one each of black, grey and white, and no sighting at all hold nothing. Regression caught: a tie
    broken by the word's order, a boundary set counted as its word or as a vote against, a thing of a few confident
    frames and many boundary ones vetoed by the boundary ones."""
    b, g, w, bg = frozenset({"black"}), frozenset({"grey"}), frozenset({"white"}), frozenset({"black", "grey"})
    assert held([b, b, b, g, g]) == "black" and held([b, b, g, g, w]) is None and held([b, b, g, g]) is None
    assert held([b, b, g, w]) == "black" and held([b, b, b, bg]) == "black" and held([b, b, bg, bg]) == "black"
    assert held([bg, bg]) is None and held([b, g, w]) is None and held([]) is None


def test_taken_for_unions_every_sighting_and_a_neutral_label_is_taken_for_all_three_neutrals():
    """Contract: `taken_for` is every word any sighting allows (a thing may be taken for what any frame of it shows);
    `label_reads(label)` is what a thing labelled so is taken for when no render says: for a neutral label all three
    neutrals (black, grey and white are one group), for any other colour, and a word outside the palette, only itself.
    Regression caught: black and white kept apart (the black remote and the white one a viewer cannot tell from the
    grey), a neutral taken for itself alone, a chromatic colour given neighbours."""
    assert taken_for([frozenset({"grey"}), frozenset({"black", "white"})]) == {"black", "grey", "white"}
    assert label_reads("black") == label_reads("grey") == label_reads("white") == {"black", "grey", "white"}
    assert label_reads("red") == {"red"} and label_reads("oak") == {"oak"} and label_reads(None) == {None}


def test_could_be_is_a_word_in_the_set_or_both_neutral_and_a_chromatic_colour_needs_its_own_word():
    """Contract: a thing that may be taken for `words` could be taken for the colour `c` when `c` is one of them, or `c`
    is a neutral and any word is: so a set with only 'grey' hides 'white' and 'black', a set with only 'blue' hides
    no neutral, and a chromatic colour is hidden by its own word or a boundary next to it, never by another colour or a
    neutral. Hand values: {grey} hides white, black and grey, not red; {blue, white} hides black (white is neutral) and
    blue, not red; {blue} does not hide white; {red, orange} hides red and orange, not yellow or white. Regression
    caught: the neutral group spread to chromatic colours (a red box hidden by a grey bag), or a boundary set of neutral
    and chromatic words ignored."""
    grey, blue, edge = frozenset({"grey"}), frozenset({"blue", "white"}), frozenset({"red", "orange"})
    assert all(could_be(grey, c) for c in ("white", "black", "grey")) and not could_be(grey, "red")
    assert could_be(blue, "black") and could_be(blue, "blue") and not could_be(blue, "red")
    assert not could_be(frozenset({"blue"}), "white")
    assert could_be(edge, "red") and could_be(edge, "orange")
    assert not could_be(edge, "yellow") and not could_be(edge, "white")
