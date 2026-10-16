"""What a generated source names and what a generated model is made of (`asset_manifest` checks both).

A source is `hosted` (its files are downloaded from its hosts) or `generated`: a recipe in this repository makes the
file once (the resident's body, `render/people_make.py`), the source has no host, and its `generator` names what made
it: the recipe's path and every input (a name, a `kind`, a licence, an https url to find it by). A `tool` input is code
the recipe runs (GPL allowed, nothing of it ships); a `data` input is read, and carries a licence of the allowlist. A
model's `parts` list what a generated mesh is made of, each with its own licence (a CC BY part also its title, author
and what we changed). Pure: no network, no Blender.
"""

from __future__ import annotations

from .asset_types import Asset, Source

GENERATED = "generated"
SOURCE_KINDS = ("hosted", GENERATED)
INPUT_KINDS = ("tool", "data")


def _text(value) -> bool:
    return isinstance(value, str) and value != ""


def _relative(value) -> bool:
    return _text(value) and not value.startswith("/") and ":" not in value and ".." not in value.split("/")


def _input_problems(where: str, item, licences: frozenset[str], code_licences: frozenset[str]) -> list[str]:
    if not isinstance(item, dict):
        return [f"{where} must be an object"]
    checks = (("name", "text", _text(item.get("name"))),
              ("kind", f"one of {INPUT_KINDS}", item.get("kind") in INPUT_KINDS),
              ("url", "an https url", _text(item.get("url")) and item["url"].startswith("https://")))
    out = [f"{where}.{name} must be {what}" for name, what, ok in checks if not ok]
    allowed = licences | code_licences if item.get("kind") == "tool" else licences
    if item.get("licence") not in allowed:
        out.append(f"{where}.licence {item.get('licence')!r} is not on the allowlist {sorted(allowed)}")
    return out


def source_problems(key: str, s: Source, licences: frozenset[str], code_licences: frozenset[str]) -> list[str]:
    """The source's kind, and for a generated one its hosts (none) and its generator."""
    out = []
    if s.kind not in SOURCE_KINDS:
        out.append(f"source {key}: kind {s.kind!r} must be one of {SOURCE_KINDS}")
    if (s.kind == GENERATED) != (s.generator is not None):
        out.append(f"source {key}: a generator belongs to a generated source, and a generated source needs one")
    if s.kind != GENERATED or not isinstance(s.generator, dict):
        return out
    g, inputs = s.generator, s.generator.get("inputs")
    checks = (("recipe", "a relative path", _relative(g.get("recipe"))),
              ("inputs", "a non-empty list", isinstance(inputs, list) and inputs != []))
    out += [f"source {key}: generator.{name} must be {what}" for name, what, ok in checks if not ok]
    if s.hosts:
        out.append(f"source {key}: a generated source has no hosts (nothing is downloaded)")
    for i, item in enumerate(inputs if isinstance(inputs, list) else ()):
        out += _input_problems(f"source {key}: generator.inputs[{i}]", item, licences, code_licences)
    return out


def parts_problems(a: Asset, licences: frozenset[str]) -> list[str]:
    """Each of a model's `parts` names itself, its kind, its source and its own licence."""
    out = []
    for i, part in enumerate(a.parts):
        where = f"{a.id}: part {i}"
        if not (isinstance(part, dict) and all(_text(part.get(k)) for k in ("name", "kind", "source"))):
            out.append(f"{where} needs a name, a kind and a source (text)")
            continue
        if part.get("licence") not in licences:
            out.append(f"{where} ({part['name']}): licence {part.get('licence')!r} is not on the allowlist")
        credited = all(part.get(k) for k in ("title", "author", "changes"))
        if str(part.get("licence")).startswith("CC-BY") and not credited:
            out.append(f"{where} ({part['name']}): a CC BY part needs a title, an author and a note of what we changed")
    return out
