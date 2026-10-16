"""The kinds the vacuum's two lists of floor things name in their questions (final review S4): `tangle_ahead` and
`task_things` both ask for what the brush could catch, and a thing of a kind neither asks for is charged as a false
place, which is fair only when the words say which kinds. Expected phrases are written by hand."""

from __future__ import annotations

import re

from stbench import job_registry
from stbench.descriptions import kind_word
from stbench.hazard_rule import CATCH_KINDS, SPOT_KINDS, TANGLE_KINDS, catch_words
from stbench.questions_text import FAMILIES

PARENTHESIS = ("(a cable bundle, an extension cord, a power strip or a towel; a bath mat, a clothes pile or a pedestal "
               "mat counts neither way)")


def named(text: str) -> tuple[set[str], set[str]]:
    """The kinds a question's parenthesis names, as (asked for, counted neither way), each a set of kind words."""
    inside = re.search(r"\(([^()]*counts neither way)\)", text)[1]
    asked, neither = inside.split("; ")

    def words(part: str) -> set[str]:
        return {re.sub(r"^an? ", "", w) for w in re.split(r", | or ", part.removesuffix(" counts neither way"))}

    return words(asked), words(neither)


def test_the_phrase_names_what_is_asked_for_and_what_counts_neither_way():
    """Contract (final review S4): `catch_words` says the items (`TANGLE_KINDS`) and then the kinds that count neither
    way (`SPOT_KINDS`), from `CATCH_KINDS` with their articles. By hand: the catch kinds are bath_mat, cable_bundle,
    clothes_pile, extension_cord, pedestal_mat, power_strip and towel; the three that recur at one spot go after the
    semicolon, the four others before it, "an" before a vowel. Regression caught: a hand-typed list that drifts from
    the kinds the gold uses, a missing article, or a kind named on both sides."""
    assert catch_words() == PARENTHESIS
    asked, neither = named(catch_words())
    assert asked == {kind_word(k) for k in TANGLE_KINDS} and neither == {kind_word(k) for k in SPOT_KINDS}
    assert asked | neither == {kind_word(k) for k in CATCH_KINDS} and not asked & neither
    assert len(CATCH_KINDS) == 7 and len(SPOT_KINDS) == 3


def test_both_vacuum_lists_say_the_kinds_the_gold_uses():
    """Contract (final review S4): the rendered question of `tangle_ahead` and that of the vacuum's `task_things` name
    exactly the kinds of the gold: the words asked for are `hazard_rule.TANGLE_KINDS`', those counted neither way
    `SPOT_KINDS`', and together they are the job's `item_kinds` (`CATCH_KINDS`) with `dont_care_kinds` (`SPOT_KINDS`).
    By hand, the phrase is `PARENTHESIS`, after 'that your brush could catch' in both; the old 'such as a cable or a
    cloth' is gone. Regression caught: a list whose words leave a kind open (a model listing a shoe charged for a
    defensible reading), or words that name a kind the gold does not use."""
    predicate = job_registry.get("vacuum/2").task_things
    tangle = FAMILIES["tangle_ahead"].render({})
    task = FAMILIES["task_things"].render({"phrase": predicate.phrase, "task": 0})
    for text in (tangle, task):
        assert f"that your brush could catch {PARENTHESIS}" in text
        asked, neither = named(text)
        assert asked == {kind_word(k) for k in TANGLE_KINDS} and neither == {kind_word(k) for k in SPOT_KINDS}
    assert "such as a cable or a cloth" not in tangle
    assert predicate.item_kinds == frozenset(CATCH_KINDS) and predicate.dont_care_kinds == frozenset(SPOT_KINDS)
    assert {kind_word(k) for k in predicate.item_kinds} == named(task)[0] | named(task)[1]
