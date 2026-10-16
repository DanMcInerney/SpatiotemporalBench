"""The warehouse's look-alikes for unseen things (the AMR's log review, defect 2;
`descriptions.CONFUSABLE_UNSEEN`'s `{bin, drum}` and `{sack, pallet}`, read by `questions_where.lookalike_sighted` and
`never_seen`): a thing that no frame of the window shows is not named beside a sighted look-alike of its colour in the
other kind of its pair, and neither is an absent description; a seen thing is never hidden by them. Hand-built worlds of
the kits on the vacuum's rooms, with the warehouse's kinds made relevant; every expected set is written out by hand."""

from __future__ import annotations

from dataclasses import replace

import family_kit as fk
from space_kit import JOB, RELEVANT_KINDS, vacuum_run
from stbench import questions_where
from stbench.job_registry import Relevance
from stbench.questions_where import lookalike_sighted, never_seen, pools

WAREHOUSE_JOB = replace(JOB, relevance=lambda plan: Relevance(
    RELEVANT_KINDS | {"bin", "tote", "drum", "sack", "pallet"}, frozenset({"kitchen", "living room"}),
    frozenset({"cleaning", "transit", "docking"})))


def put(s, eid, kind, colour, xy, seen=True):
    """A carried thing labelled `colour` at (x, y), sighted (1,500 px, recognisable, in its own colour) at keyframe 1,
    or never sighted."""
    s.item(eid, kind, {"color": colour}, [(0, None, (*xy, 0.0))])
    if seen:
        s.see([1], eid, 1500)


def entities(cands) -> set[str]:
    """The ids of the world's things the candidates name (a description the world lacks is not an entity)."""
    return {x[2:] for c in cands for x in c.subjects if x.startswith("e:")}


def test_an_unseen_thing_is_not_named_beside_a_sighted_lookalike_of_the_other_warehouse_kind():
    """Contract (`CONFUSABLE_UNSEEN`): a real thing that no frame of the window shows is in neither pool when a sighted
    thing of the other kind of its pair (bin and drum; sack and pallet) has its colour, either way round, and is the
    unknown pool's when that kind is in another colour or not sighted. Hand-worked on one task: the unseen yellow bin
    (a sighted yellow drum), blue sack (a sighted blue pallet), blue drum (a sighted blue bin) and pink pallet (a
    sighted pink sack) are hidden; the unseen green bin (a sighted red drum) and white sack (a blue pallet and a pink
    sack) are the unknown pool's two things; the five sighted things (the yellow and red drums, the blue pallet and bin,
    the pink sack) are answerable, for a sighted thing is
    not hidden by them. Regression caught: the review's 'where did you last see the yellow bin?' beside
    a yellow drum (both reference models pointed at the drum), the blue sack beside a blue pallet, a screen that hides
    only one way round, or one widened to every colour."""
    s = vacuum_run()
    put(s, "E0001", "bin", "yellow", (1.0, 1.0), seen=False)
    put(s, "E0002", "drum", "yellow", (3.0, 1.0))
    put(s, "E0003", "bin", "green", (1.0, 2.0), seen=False)
    put(s, "E0004", "drum", "red", (3.0, 2.0))
    put(s, "E0005", "sack", "blue", (5.0, 1.0), seen=False)
    put(s, "E0006", "pallet", "blue", (7.0, 1.0))
    put(s, "E0007", "sack", "white", (5.0, 2.0), seen=False)
    put(s, "E0008", "drum", "blue", (3.0, 0.5), seen=False)
    put(s, "E0009", "bin", "blue", (1.0, 0.5))
    put(s, "E0010", "pallet", "pink", (7.0, 0.5), seen=False)
    put(s, "E0011", "sack", "pink", (5.0, 0.5))
    answerable, unknown = pools(fk.world(s, job=WAREHOUSE_JOB))
    assert entities(unknown) == {"E0003", "E0007"}
    assert entities(answerable) == {"E0002", "E0004", "E0006", "E0009", "E0011"}


def test_a_seen_drum_pallet_bin_or_sack_is_asked_beside_a_seen_one_of_the_other_kind_of_its_colour():
    """Contract (`CONFUSABLE_UNSEEN` hides no seen thing): the look-alike groups of unseen things never hide an
    answerable question, so a seen drum beside a seen bin of its colour, or a seen pallet beside a seen sack, are both
    asked, whatever they stand beside; a different pair is the same. Hand-worked on one task: a yellow bin and drum, a
    blue sack and pallet, all four sighted, are the four answerable things. Regression caught: the groups made
    symmetric for seen things, which cost the AMR's release most of its answerable `object_where` questions (a fair
    identification test lost)."""
    s = vacuum_run()
    put(s, "E0001", "bin", "yellow", (1.0, 1.0))
    put(s, "E0002", "drum", "yellow", (3.0, 1.0))
    put(s, "E0003", "sack", "blue", (5.0, 1.0))
    put(s, "E0004", "pallet", "blue", (7.0, 1.0))
    answerable, unknown = pools(fk.world(s, job=WAREHOUSE_JOB))
    assert entities(answerable) == {"E0001", "E0002", "E0003", "E0004"}
    assert entities(unknown) == set()


def test_a_decoy_is_not_offered_beside_a_sighted_lookalike_of_the_other_warehouse_kind(monkeypatch):
    """Contract (`CONFUSABLE_UNSEEN`, `questions_where.never_seen`): an absent description of a bin, drum, sack or
    pallet is dropped when a sighted thing of the other kind of its pair has its colour, and offered otherwise.
    Hand-worked on a world with a sighted red bin, yellow drum, brown sack and blue pallet and the other seeds' worlds
    holding a yellow bin, a blue bin, a blue sack, a yellow sack, a red drum, a blue drum, a brown pallet and a red
    pallet: the yellow bin (the yellow drum), the blue sack (the blue pallet), the red drum (the red bin) and the
    brown pallet (the brown sack) are dropped; the blue bin, the yellow sack, the blue drum and the red pallet are
    offered. Regression caught: a decoy 'yellow bin' offered beside a yellow drum, or a decoy 'blue sack' beside a blue
    pallet (an absent description that a look-alike of the world answers is no 'unknown')."""
    combos = (("bin", "yellow"), ("bin", "blue"), ("sack", "blue"), ("sack", "yellow"), ("drum", "red"),
              ("drum", "blue"), ("pallet", "brown"), ("pallet", "red"))
    other_seeds = tuple(sorted((kind, (("color", colour),)) for kind, colour in combos))
    monkeypatch.setattr(questions_where, "_other_combos", lambda map_name, own_seed: other_seeds)
    s = vacuum_run()
    put(s, "E0001", "bin", "red", (1.0, 1.0))
    put(s, "E0002", "drum", "yellow", (3.0, 1.0))
    put(s, "E0003", "sack", "brown", (5.0, 1.0))
    put(s, "E0004", "pallet", "blue", (7.0, 1.0))
    offered = {(kind, dict(attrs)["color"]) for kind, attrs in never_seen(fk.world(s, job=WAREHOUSE_JOB))}
    assert offered == {("bin", "blue"), ("sack", "yellow"), ("drum", "blue"), ("pallet", "red")}


def test_the_warehouse_groups_do_not_reach_the_other_kinds():
    """Contract (`CONFUSABLE_UNSEEN` keeps its groups pairs of kinds, not a merged class): a drum does not hide an
    unseen tote, box or sack of its colour, nor a pallet a tote or a box. Hand-worked: a sighted yellow drum and a
    sighted blue pallet are answerable, and the unseen yellow tote, yellow box, yellow sack and blue tote are all in the
    unknown pool. Regression caught: the groups merged into the box group (a tote hidden by a drum) or into each other
    (a sack hidden by a drum)."""
    s = vacuum_run()
    put(s, "E0001", "drum", "yellow", (3.0, 1.0))
    put(s, "E0002", "pallet", "blue", (7.0, 1.0))
    put(s, "E0003", "tote", "yellow", (1.0, 1.0), seen=False)
    put(s, "E0004", "box", "yellow", (1.0, 2.0), seen=False)
    put(s, "E0005", "sack", "yellow", (5.0, 2.0), seen=False)
    put(s, "E0006", "tote", "blue", (5.0, 1.0), seen=False)
    answerable, unknown = pools(fk.world(s, job=WAREHOUSE_JOB))
    assert entities(answerable) == {"E0001", "E0002"}
    assert entities(unknown) == {"E0003", "E0004", "E0005", "E0006"}


def test_a_description_the_world_lacks_takes_the_unseen_groups_in_lookalike_sighted():
    """Contract (`questions_where.lookalike_sighted` is the one owner of the screen, decoys included): called with no
    entity (a description the world lacks) it applies `CONFUSABLE_UNSEEN`, as it does for a thing no frame shows, and a
    seen entity does not. Hand-worked: beside a sighted yellow drum alone a yellow bin description is hidden and a red
    one is not; beside a sighted yellow drum and yellow bin the seen yellow drum is not hidden. Regression caught: a
    caller passing None and silently missing the groups (a decoy bin offered beside a yellow drum)."""
    s = vacuum_run()
    put(s, "E0001", "drum", "yellow", (3.0, 1.0))
    w = fk.world(s, job=WAREHOUSE_JOB)
    assert lookalike_sighted(w, "bin", {"color": "yellow"}, None)
    assert not lookalike_sighted(w, "bin", {"color": "red"}, None)
    put(s, "E0002", "bin", "yellow", (1.0, 1.0))
    assert not lookalike_sighted(fk.world(s, job=WAREHOUSE_JOB), "drum", {"color": "yellow"}, "E0001")
