"""The registry's 0.6.0 hooks (`questions_text`): `LOOKS`, `CAPPED` and `RELAX`, and `SLICED` on a look family."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from family_kit import fake_spec
from stbench import baselines_path, questions_text, table_answers
from stbench.questions_text import FAMILY_MODULES, Registry, _load

LOOK = fake_spec("look", when="during", scope="evidence", answer_type="offset")
END = fake_spec("last")
OTHER = fake_spec("other", when="during", scope="now")


def module(specs, golds=None, **hooks):
    return SimpleNamespace(SPECS=tuple(specs), GOLDS=golds or {s.name: print for s in specs}, **hooks)


@pytest.fixture
def mods(monkeypatch):
    found: dict = {}
    monkeypatch.setattr(questions_text, "_import", found.get)
    return found


def test_the_hooks_are_collected_per_module_and_each_is_optional(mods):
    """Contract (plan §3): `_load` gathers `LOOKS` (during families), `SLICED` (an end family or a look family),
    `COINS`, `CAPPED` (end families) and `RELAX` (end family -> function) from each module that has them; a module with
    none contributes none. Hand-worked on two fakes. Regression caught: a hook defined in a family module and dropped
    by the loader, so a look family would be drawn like an end family or never at all."""
    thin = print
    mods["m1"] = module([LOOK, END], LOOKS=frozenset({"look"}), SLICED=frozenset({"look"}), CAPPED=frozenset({"last"}),
                        RELAX={"last": thin})
    mods["m2"] = module([OTHER])   # no hooks at all
    found = _load(("m1", "m2"))
    assert isinstance(found, Registry) and list(found.families) == ["look", "last", "other"]
    assert (found.looks, found.sliced, found.capped) == (frozenset({"look"}),) * 2 + (frozenset({"last"}),)
    assert found.coins == frozenset() and found.relax == {"last": thin}


@pytest.mark.parametrize("hooks, why", [
    ({"LOOKS": frozenset({"last"})}, r"last.*LOOKS"),            # an end family asked at look points
    ({"LOOKS": frozenset({"nothere"})}, r"nothere.*LOOKS"),      # a stray name
    ({"CAPPED": frozenset({"look"})}, r"look.*CAPPED"),          # a during family capped
    ({"CAPPED": frozenset({"nothere"})}, r"nothere.*CAPPED"),
    ({"RELAX": {"look": print}}, r"look.*RELAX"),                # a during family relaxed
    ({"RELAX": {"nothere": print}}, r"nothere.*RELAX"),
    ({"SLICED": frozenset({"look"})}, r"look.*SLICED"),          # a look family sliced without being in LOOKS
])
def test_a_hook_naming_the_wrong_kind_of_family_raises_naming_it(mods, hooks, why):
    """Contract: a `LOOKS` name that is no during family of its module, a `CAPPED` or `RELAX` name that is no end family
    of it, and a `SLICED` name that is neither an end family nor a look family each raise, naming the family and the
    hook. Regression caught: a look family drawn at the end (its gold takes a point and gives a moment, so the draw
    would crash on a case, or ask nothing), or a typo in a name that thins nothing."""
    mods["m1"] = module([LOOK, END], **hooks)
    with pytest.raises(ValueError, match=why):
        _load(("m1",))


def test_the_live_registry_is_unchanged_by_the_new_modules_and_names_them():
    """Contract: the module lists name `questions_ego`, `questions_task`, `questions_clear`, `questions_people`,
    `questions_heading`, (0.10.0) `questions_approach` and `questions_hazard` and (0.11) `questions_way`, and their
    answerers and `path_log` answerers in step, a module that does not exist yet is
    skipped, and the live registry's look, capped and relaxed families are during, end and end families. Regression
    caught: a list updated on one side only (a family answered by the tables and not `path_log`), or a hook naming the
    wrong kind of family."""
    assert FAMILY_MODULES[-8:] == ("questions_ego", "questions_task", "questions_clear", "questions_people",
                                   "questions_heading", "questions_approach", "questions_hazard", "questions_way")
    assert table_answers.ANSWER_MODULES[-8:] == ("table_ego", "table_task", "table_clear", "table_people",
                                                 "table_heading", "table_approach", "table_hazard", "table_way")
    assert baselines_path.PATH_MODULES[-8:] == ("baselines_path_ego", "baselines_path_task", "baselines_path_clear",
                                                "baselines_path_people", "baselines_path_heading",
                                                "baselines_path_approach", "baselines_path_hazard",
                                                "baselines_path_way")
    assert isinstance(questions_text.LOOKS, frozenset) and isinstance(questions_text.CAPPED, frozenset)
    assert isinstance(questions_text.RELAX, dict)
    assert set(questions_text.FAMILIES) >= questions_text.LOOKS
    assert all(questions_text.FAMILIES[f].when == "during" for f in questions_text.LOOKS)
    assert all(questions_text.FAMILIES[f].when == "end" for f in (*questions_text.CAPPED, *questions_text.RELAX))
