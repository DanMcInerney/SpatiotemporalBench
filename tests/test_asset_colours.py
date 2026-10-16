"""A kind with several models gives each its colours (`Asset.for_colours`, `Manifest.model_for(kind, colour)`): what
`validate` holds that to on synthetic manifests from `asset_synth`, and which model the packaged manifest gives each
description of the kinds a question can ask about (the look round's bag, box, toy, shoes and stool). Nothing touches
the network."""

from __future__ import annotations

import copy

import pytest

from asset_synth import asset, manifest
from stbench import asset_manifest as A
from stbench.catalog import CATALOGS

COLOURS = {"mug": ("red", "blue", "green"), "glass": ("blue",)}
KINDS = {"catalog_kinds": ("mug", "glass"), "light_kinds": (), "colours": COLOURS}


def second_mug(colours: tuple[str, ...], **extra):
    """mug_b turned into a second primary model of the mug, drawn in `colours`."""
    def edit(raw):
        raw["assets"] = [a for a in raw["assets"] if a["id"] != "mug_b"]
        raw["assets"].append({**copy.deepcopy(asset(raw, "mug_a")), "id": "mug_b", "title": "Mug B",
                              "for_colours": list(colours), **extra})
        asset(raw, "mug_a")["for_colours"] = ["red"]
    return edit


def problems(edit=None) -> list[str]:
    return A.validate(manifest(edit=edit), **KINDS)


def test_a_model_follows_the_colour_so_a_description_always_has_the_same_one():
    """Contract: with two models of a kind, `model_for(kind, colour)` is the one that lists the colour, and the same
    every call (identical items are identical in every pixel); a kind with one model has it in every colour, and one
    with several and no colour asked for is refused. Regression caught: a model picked by position or by entity (two
    red mugs that differ), or a colour that silently gets whichever model is first."""
    m = manifest(edit=second_mug(("blue", "green")))
    assert [m.model_for("mug", c).id for c in ("red", "blue", "green")] == ["mug_a", "mug_b", "mug_b"]
    assert m.model_for("glass").id == m.model_for("glass", "blue").id == "proc_glass"
    with pytest.raises(A.AssetError, match="0 models for kind 'mug'"):
        m.model_for("mug")
    with pytest.raises(A.AssetError, match="0 models for kind 'mug' in 'purple'"):
        m.model_for("mug", "purple")


def test_a_model_that_lists_no_colours_serves_the_colours_no_other_model_lists():
    """Contract: a model with no `for_colours` is the kind's default, used for every colour that no sibling lists, so a
    kind may keep one general model and give a few colours their own. Regression caught: a default that wins over a
    specific model, or one that is never used."""
    def edit(raw):
        second_mug(("blue",))(raw)
        del asset(raw, "mug_a")["for_colours"]
    m = manifest(edit=edit)
    assert [m.model_for("mug", c).id for c in ("red", "blue", "green")] == ["mug_a", "mug_b", "mug_a"]


@pytest.mark.parametrize("edit, expected", [
    pytest.param(second_mug(("blue", "green")), None, id="every colour has one model"),
    pytest.param(second_mug(("blue",)), "2 models for kind 'mug' in 'green'|0 models", id="a colour with none"),
    pytest.param(second_mug(("blue", "green", "red")), "2 models for kind 'mug' in 'red'", id="a colour with two"),
    pytest.param(second_mug(("blue", "green", "purple")), "mug_b: it is for 'purple', a colour no mug has",
                 id="a colour the kind never has"),
    pytest.param(second_mug(("blue", "blue")), "for_colours must be distinct", id="a colour listed twice"),
    pytest.param(second_mug(("blue", "green"), alternate_for="mug_a"), "an alternate serves no colour",
                 id="an alternate with colours"),
    pytest.param(lambda r: asset(r, "mat_a").update(for_colours=["red"]), "for_colours is for a model",
                 id="a material with colours"),
])
def test_validate_holds_each_colour_to_exactly_one_model(edit, expected):
    """Contract (`validate(..., colours=)`): every colour of a kind with several models resolves to exactly one of
    them, a model's colours are distinct and ones the kind has, an alternate has none of its own and only a model has
    any. Regression caught: a colour left on a placeholder or drawn two ways, a typo in a colour that makes a model
    dead, an alternate that would be used."""
    found = problems(edit)
    if expected is None:
        assert found == []
    else:
        assert any(any(e in p for e in expected.split("|")) for p in found), found


def test_the_kinds_a_question_can_ask_about_have_the_models_the_look_round_chose():
    """Contract (M1's pre-screen): the bag is a canvas tote in red, green, orange and brown
    and a zipped holdall in black, blue and grey; the box a taped carton in brown and a lidded storage box in white,
    blue, red and green; the toy a rubber duck in yellow, a plush bear in pink and orange and wooden blocks in red, blue
    and green; shoes and the stool each one model in every colour. Hand-listed from the review's fixes. Regression
    caught: a colour that falls back to the old scan (a blue box that is a flat tinted cuboid, a toy that is only a
    duck), or a kind whose models stop covering a colour its catalog draws."""
    m = A.packaged()
    wanted = {"bag": {"proc_tote_bag": ["red", "green", "orange", "brown"],
                      "proc_holdall_bag": ["black", "blue", "grey"]},
              "box": {"proc_carton": ["brown"], "proc_storage_box": ["white", "blue", "red", "green"]},
              "toy": {"ph_rubber_duck_toy": ["yellow"], "proc_teddy_bear": ["pink", "orange"],
                      "proc_toy_blocks": ["red", "blue", "green"]}}
    for kind, models in wanted.items():
        assert {a.id: list(a.for_colours) for a in m.models_of(kind)} == models, kind
        every = {c for cs in models.values() for c in cs}
        assert every == set(A.queried_colours()[kind]), kind
    for kind in ("shoes", "stool"):
        (model,) = m.models_of(kind)
        assert model.source == A.PROCEDURAL and not model.for_colours, kind
    assert not any(a.id.startswith("gso_bag") or a.id.startswith("gso_shoes") for a in m.assets)


def test_every_colour_a_catalog_draws_of_these_kinds_has_a_model_inside_the_catalogs_box():
    """Contract (V0 section 5: a catalog size is what renders): each model of a kind with several is built to the
    kind's one catalog box (fit `exact`, box equal to it to a millimetre), whichever colour it serves, so the footprint
    the planner and the visibility log use is the model's. Regression caught: a model measured to another box than its
    siblings (a bag whose duffel is bigger than its tote), which the exact fit would squash."""
    m = A.packaged()
    for kind in ("bag", "box", "toy", "shoes"):
        box = CATALOGS["home"].kinds[kind].size
        for a in m.models_of(kind):
            if a.source == A.PROCEDURAL:
                assert tuple(a.native_bbox_m) == pytest.approx(box, abs=1e-3), a.id
                assert a.fit["mode"] == "exact", a.id
