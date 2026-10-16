"""The crop check of `stbench realism sheet | report` (M1's R.2): whether the things a question can
name are recognisable, in the picture the agent gets, at the size the vacuum sees them.

Each crop (`contact_sheet.crop_box`: the object's box and a quarter of its size of margin all round, in native pixels)
goes to two fresh reviewers (Claude Sonnet, one crop and one question each, never Codex), who see nothing else:
- the **forced choice** (`FORCED_PROMPT`), first: the object is one of the kinds a question can ask about, or none of
  them. It measures directly the confusions that matter, a look-alike read as a kind a question names;
- the **free naming** (`FREE_PROMPT`), second and stricter: what is it called, with nothing suggested.
The prompts are fixed. An answer is read as `<colour>
<object>` (`read_answer`), the colour through `COLOUR_MAP`. A crop is *confused* when the object names an askable kind
(`scenery.names_a_kind`: the name or a synonym) in one of that kind's tracked colours and the crop is not that kind's
own; a crop of an askable kind is *right* when it names its own kind in its colour. Pure: answers, the key and the
askable kinds in, scores out."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from .scenery import names_a_kind

FREE_PROMPT = ("This is a small crop of a photo taken in a home. Name the main object in it and its colour, in a few "
               "words, as '<colour> <object>'.")
FORCED_PROMPT = ("This is a small crop of a photo taken in a home. The main object in it is one of these kinds, or "
                 "none of them: {kinds}. Reply with its colour and its kind from the list, as '<colour> <kind>', or "
                 "with 'none' if it is none of them.")
COLOUR_MAP = {"navy": "blue", "teal": "blue", "turquoise": "blue", "cyan": "blue", "tan": "beige", "cream": "beige",
              "silver": "grey", "gray": "grey", "charcoal": "black", "maroon": "red", "burgundy": "red",
              "crimson": "red", "gold": "yellow", "violet": "purple", "lilac": "purple", "lavender": "purple",
              "olive": "green", "ivory": "white", "off-white": "white", "tangerine": "orange"}
PROTOCOLS = ("forced", "free")  # the forced choice is scored first


def prompts(kinds: Sequence[str]) -> dict:
    """The two fixed prompts, the forced choice's with the askable kinds listed (underscores as spaces, in order)."""
    return {"forced": FORCED_PROMPT.format(kinds=", ".join(k.replace("_", " ") for k in sorted(kinds))),
            "free": FREE_PROMPT, "kinds": sorted(kinds)}


def read_answer(text: str) -> tuple[str, str]:
    """A reviewer's answer `<colour> <object>` as (colour, object), the colour word mapped by `COLOUR_MAP`; an answer
    of one word is all object."""
    parts = text.lower().replace(",", " ").split()
    colour = COLOUR_MAP.get(parts[0], parts[0]) if len(parts) > 1 else ""
    return colour, " ".join(parts[1:] if len(parts) > 1 else parts)


def split_answers(answers: dict) -> dict[str, dict[str, str]]:
    """The reviewers' file `{crop: {"forced": "...", "free": "..."}}` as one answer set a protocol; a plain string is
    the free naming alone (the first protocol's file)."""
    out: dict[str, dict[str, str]] = {p: {} for p in PROTOCOLS}
    for crop, a in answers.items():
        got = a if isinstance(a, dict) else {"free": a}
        for p in PROTOCOLS:
            if isinstance(got.get(p), str) and got[p].strip():
                out[p][crop] = got[p]
    return out


def score(answers: dict[str, str], key: Sequence[dict], askable: dict[str, Sequence[str]]) -> dict:
    """Score one protocol's answers against the key (each `{crop, class, kind, colour}`: `kind` is an askable kind or
    None for a clutter or dressing class). `askable` maps each askable kind to its tracked colours. Returns the crops
    answered, the confused crops (`{crop, class, named, colour}`) and, per askable kind, crops, right and rate."""
    confused: list[dict] = []
    rights: dict[str, list[bool]] = defaultdict(list)
    for row in key:
        if row["crop"] not in answers:
            continue
        colour, obj = read_answer(answers[row["crop"]])
        named = names_a_kind(obj.replace(" ", "_"), askable) or [k for k in askable if k.replace("_", " ") in obj]
        if row["kind"] is not None:
            rights[row["kind"]].append(row["kind"] in named and colour == row["colour"])
        for kind in (k for k in named if k != row["kind"] and colour in askable[k]):
            confused.append({"crop": row["crop"], "class": row["class"], "named": kind, "colour": colour})
    per_kind = {k: {"crops": len(v), "right": sum(v), "rate": sum(v) / len(v)} for k, v in sorted(rights.items())}
    return {"answered": sum(1 for r in key if r["crop"] in answers), "confused": confused, "per_kind": per_kind}


def crop_scores(answers: dict, key: Sequence[dict], askable: dict[str, Sequence[str]]) -> dict:
    """`score` of each protocol of `PROTOCOLS` that has answers, `{"forced": ..., "free": ...}` (None for none)."""
    split = split_answers(answers)
    return {p: score(split[p], key, askable) if split[p] else None for p in PROTOCOLS}
