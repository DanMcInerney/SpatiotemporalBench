"""The family registry (`questions_text`): modules found through a pre-seeded list, the families they hold, and M1's two
unmoved by the move into modules."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from family_kit import fake_spec
from stbench import questionnaire, questions, questions_space, questions_text, questions_where
from stbench.questions_text import FAMILIES, FAMILY_MODULES, _load


def fake_module(specs, golds, sliced=frozenset()):
    return SimpleNamespace(SPECS=tuple(specs), GOLDS=golds, SLICED=sliced)  # no COINS: a module may have none


def test_a_missing_family_module_is_skipped(monkeypatch):
    """Contract: a `FAMILY_MODULES` name that does not exist adds nothing (so M3's and M4's names cost nothing until
    their modules are written), while a module that exists and fails to import for another reason raises, and a module
    adds its specs and golds; a family named twice (by two modules or one), a gold function with no spec, a spec with
    no gold and a `SLICED` name that is no end family of its module each raise, naming the family. Hand-worked on the
    real `questions_space` and fakes. Regression caught: a family lane's module left unregistered or silently skipped
    over a typo in its imports, or two lanes claiming one family."""
    assert FAMILY_MODULES[:2] == ("questions_space", "questions_where")
    found = _load(("questions_space", "questions_no_such_module"))
    assert (list(found.families), list(found.golds), found.sliced, found.coins) == (
        ["path_ahead", "rear_clear"], ["path_ahead", "rear_clear"], frozenset(), frozenset())
    assert (found.looks, found.capped, found.relax) == (frozenset(), frozenset(), {})   # a module may have none

    def broken(name):
        raise ModuleNotFoundError("No module named 'numpyy'", name="numpyy")

    with monkeypatch.context() as patched:
        patched.setattr(importlib, "import_module", broken)
        with pytest.raises(ModuleNotFoundError, match="numpyy"):
            questions_text._import("questions_typo")

    ghost, other = fake_spec("ghost"), fake_spec("other", when="during", scope="now")
    mods = {"m0": questions_space}
    monkeypatch.setattr(questions_text, "_import", mods.get)

    def load():
        return _load(("m0", "m1", "m2"))  # m2 does not exist: skipped

    mods["m1"] = fake_module([ghost, other], {"ghost": print, "other": print}, frozenset({"ghost"}))
    found = load()
    fams, golds, sliced = found.families, found.golds, found.sliced
    assert (list(fams), set(golds), sliced) == (["path_ahead", "rear_clear", "ghost", "other"],
                                                {"path_ahead", "rear_clear", "ghost", "other"}, frozenset({"ghost"}))
    for name, module in {
            "path_ahead": fake_module([fake_spec("path_ahead")], {"path_ahead": print}),  # named by two modules
            "ghost": fake_module([ghost, ghost], {"ghost": print}),  # named twice by one
            "stray": fake_module([ghost], {"ghost": print, "stray": print}),  # a gold with no spec
            "other": fake_module([ghost, other], {"ghost": print}),  # a spec with no gold
            "nothere": fake_module([ghost], {"ghost": print}, frozenset({"nothere"})),  # a stray SLICED name
            "during": fake_module([other], {"other": print}, frozenset({"other"})),  # SLICED on a during family
    }.items():
        mods["m1"] = module
        with pytest.raises(ValueError, match="other" if name == "during" else name):
            load()


def test_registry_holds_m1s_families_unchanged():
    """Contract: the registry holds `path_ahead`, then `rear_clear` (M3's, the other corridor family of
    `questions_space`), then `object_where`: M1's two in that order, `path_ahead` rendering M1's text and `object_where`
    its 0.5.0-control form ("last seen": the slots are the thing's kind and attributes alone, no moment and no task),
    `object_where` is `SLICED` and `path_ahead` is not (later lanes' families join the registry after them), and the
    gold functions of `questions` and `questionnaire` are the registry's one dict (the completeness test reads
    `questions.GOLDS`). Hand texts: 'How far could you move straight ahead now before reaching anything? Answer 3 if
    nothing is within 3 m.' and 'Where did you last see the pair of black shoes? Answer in map coordinates.'.
    Regression caught: the move into
    modules changing a text or a key order, a moment or a task back in `object_where`'s slots, or the completeness test
    reading another dict than the compiler."""
    assert list(FAMILIES)[:3] == ["path_ahead", "rear_clear", "object_where"]
    assert FAMILIES["path_ahead"].render({"cap": 3.0}) == (
        "How far could you move straight ahead now before reaching anything? Answer 3 if nothing is within 3 m.")
    assert FAMILIES["object_where"].render({"kind": "shoes", "attributes": {"color": "black"}}) == (
        "Where did you last see the pair of black shoes? Answer in map coordinates.")
    assert FAMILIES["object_where"].slots == ("kind", "attributes")
    assert "object_where" in questions_text.SLICED and "path_ahead" not in questions_text.SLICED
    assert questions.GOLDS is questionnaire.GOLDS is questions_text.GOLDS
    assert questions.GOLDS["path_ahead"] is questions_space.path_ahead
    assert questions.GOLDS["object_where"] is questions_where.moments
