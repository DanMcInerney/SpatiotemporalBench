"""What the manifest must cover (`asset_manifest.validate` calls it): every queried kind has models of the right
mobility and each colour a description of it can have resolves to exactly one of them, and every clutter kind of every
catalog lists exactly the clutter models the manifest holds of it. `queried_kinds`, `queried_colours` and
`queried_clutter` read those lists from the catalogs and the bundled maps; `asset_manifest` re-exports them. Pure: no
network, no Blender.
"""

from __future__ import annotations

from .asset_types import AssetError, Manifest
from .catalog import CATALOGS, ClutterKind
from .mapspec import bundled_maps, load_map

BOX_SLACK_M = 0.001  # a clutter kind's box holds each model's box to a millimetre
PEOPLE_KINDS = ("person",)  # the kinds of people (`people.py`): no catalog lists them, each may have several bodies


def coverage_problems(m: Manifest, catalog_kinds, light_kinds, mover_kinds, colours=None) -> list[str]:
    """Every queried kind has models of the right mobility, and each of its colours (`colours`: kind -> the colours a
    description of it can have) resolves to exactly one of them; a kind with no colours given has exactly one model."""
    out = []
    wanted = ([(k, "carried") for k in catalog_kinds] + [(k, "light") for k in light_kinds]
              + [(k, "mover") for k in mover_kinds])
    for kind, mobility in wanted:
        found = m.models_of(kind)
        if colours is None or kind not in colours:
            if len(found) != 1 or found[0].mobility != mobility:
                out.append(f"kind {kind}: needs exactly one {mobility} model, found {[a.id for a in found]}")
        elif not found or any(a.mobility != mobility for a in found):
            out.append(f"kind {kind}: needs {mobility} models, found {[(a.id, a.mobility) for a in found]}")
        else:
            for colour in colours[kind]:
                try:
                    m.model_for(kind, colour)
                except AssetError as e:
                    out.append(f"{e}: {[a.id for a in found]}")
            out += [f"{a.id}: it is for {c!r}, a colour no {kind} has" for a in found for c in a.for_colours
                    if c not in colours[kind]]
    for kind in PEOPLE_KINDS:  # a person's models are optional here (a synthetic manifest has none), never a prop or
        # a colour's: a people module names its body by id (`people_<catalog>.MODEL`), so a kind may have several
        wrong = [(a.id, a.mobility, a.for_colours) for a in m.models_of(kind) if a.mobility != "mover" or a.for_colours]
        if wrong:
            out.append(f"kind {kind}: needs mover models, each for no colour, found {wrong}")
    known = {*catalog_kinds, *light_kinds, *mover_kinds, *PEOPLE_KINDS}
    return out + [f"{a.id}: kind {a.kind!r} is not a queried kind" for a in m.assets
                  if a.role == "model" and a.kind not in known]


def queried_kinds() -> tuple[list[str], list[str], list[str]]:
    """The (carried, light, mover) kinds the manifest must hold exactly one model of each for: every kind of every
    catalog (a kind two catalogs share, the box, once) and every light-furniture kind of every bundled map. Pass them
    to `validate`."""
    carried = sorted({k for c in CATALOGS.values() for k in c.kinds})
    movers = sorted({k for c in CATALOGS.values() for k in c.mover_kinds})
    light = sorted({f.kind for name in bundled_maps() for f in load_map(name).furniture if f.mobility == "light"})
    return carried, light, movers


def queried_colours() -> dict[str, tuple[str, ...]]:
    """Per queried kind, every colour a description of it can have: a catalog kind's colours (the same kind of two
    catalogs, the box, in both), what is delivered, a mover's colours and the colour of each light piece of a bundled
    map. Pass it to `validate` beside `queried_kinds`."""
    out: dict[str, list[str]] = {}

    def add(kind: str, colours) -> None:
        have = out.setdefault(kind, [])
        have += [c for c in colours if c not in have]

    for c in CATALOGS.values():
        for k in c.kinds.values():
            add(k.name, (*k.colors, *(c.delivery_colors if k.name == c.delivery_kind else ())))
        for mover in c.mover_kinds.values():
            add(mover.name, mover.colors)
    for name in bundled_maps():
        for f in load_map(name).furniture:
            if f.mobility == "light":
                add(f.kind, (f.color,))
    return {k: tuple(v) for k, v in out.items()}


def queried_clutter() -> list[ClutterKind]:
    """Every clutter kind of every catalog: the manifest must hold the models each lists."""
    return [k for c in CATALOGS.values() for k in c.clutter]


def clutter_coverage_problems(m: Manifest, kinds: list[ClutterKind]) -> list[str]:
    """Each clutter kind's `models` are exactly the manifest's clutter models of that kind, every model's colour is one
    the kind lists (an untinted model's own), and the kind's box holds every measured model box."""
    out, by_id = [], {a.id: a for a in m.assets}
    for k in kinds:
        have = sorted(a.id for a in m.clutter_models(k.name))
        if not k.models or sorted(k.models) != have:
            out.append(f"clutter kind {k.name}: lists models {sorted(k.models)}, the manifest holds {have}")
        for model in k.models:
            a = by_id.get(model)
            if a is None or a.role != "clutter":
                continue
            if a.colour is not None and a.colour not in k.colors:
                out.append(f"{a.id}: its colour {a.colour!r} is not one of {k.name}'s {list(k.colors)}")
            if a.native_bbox_m and any(b > s + BOX_SLACK_M for b, s in zip(a.native_bbox_m, k.size, strict=True)):
                out.append(f"{a.id}: its box {a.native_bbox_m} is not inside {k.name}'s {list(k.size)}")
    known = {k.name for k in kinds}
    return out + [f"{a.id}: kind {a.kind!r} is not a clutter kind of any catalog" for a in m.assets
                  if a.role == "clutter" and a.kind not in known]
