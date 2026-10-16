"""Helpers for the tests of the balance rule (`suite_balance`): hand-made asked questions with their golds.

`yesno` makes a family of yes/no questions script by script from how many of each script's are yes, the classes those of
`same_place` (yes against moved); `numbers` makes a number family whose gold is one value a script. Both give the
`form_check.Asked` list `suite_balance.read` takes: one (layout, job) group, one question a case, every case its
script's own.
"""

from __future__ import annotations

from collections.abc import Sequence

from stbench.form_check import Asked

LAYOUT, JOB = "apartment_a", "vacuum/1"


def gold(qid: str, family: str, value: dict, *, answer_class: str | None, ladder: list[float] | None = None,
         reference: str | None = None) -> dict:
    """A gold row (`records.GOLD_KEYS`) of a `yesno` (exact credit) or a `number` (a ladder) answer, answerable."""
    return {"qid": qid, "family": family, "gold": value, "slice": "answerable", "answer_class": answer_class,
            "range_m": None, "credit": "ladder" if ladder else "exact", "region": None,
            "tolerances": {"ladder": ladder} if ladder else {}, "metric": None, "scope": "memory",
            "reference": reference, "critical": False, "subjects": ["w:0"], "evidence_k": [0], "entities": []}


def asked(script: str, n: int, family: str, g: dict, *, layout: str = LAYOUT) -> Asked:
    """The `n`th question of `script`'s one case, as delivered (an end question), with the gold `g`."""
    info = {"case": f"{script}-case", "script": script, "layout": layout, "robot": "vacuum", "job": JOB,
            "length": "short"}
    question = {"qid": g["qid"], "when": "end", "text": "", "slots": {}, "answer_type": g["gold"]["type"]}
    return Asked(info["case"], script, question, g, info, {}, None)


def yesno(yes_per_script: Sequence[int], per_script: int, family: str = "same_place", *, layout: str = LAYOUT,
          reference: str | None = None) -> list[Asked]:
    """A yes/no family of `len(yes_per_script)` scripts of `per_script` questions, script `i` with `yes_per_script[i]`
    yes golds (class `yes`) and the rest no (class `moved`)."""
    out = []
    for i, yes in enumerate(yes_per_script):
        for n in range(per_script):
            g = gold(f"s{i}q{n:02d}".ljust(16, "0"), family, {"type": "yesno", "value": n < yes},
                     answer_class="yes" if n < yes else "moved", reference=reference)
            out.append(asked(f"s{i}", n, family, g, layout=layout))
    return out


def numbers(values: Sequence[float], per_script: int | Sequence[int], family: str = "ego_now") -> list[Asked]:
    """A number family of `len(values)` scripts, every gold of script `i` `values[i]`, `per_script` questions each (or
    `per_script[i]`), no class; the ladder is 0.25, 0.5, 1 and 2."""
    counts = [per_script] * len(values) if isinstance(per_script, int) else list(per_script)
    return [asked(f"s{i}", n, family, gold(f"s{i}q{n:02d}".ljust(16, "0"), family, {"type": "number", "value": v},
                                           answer_class=None, ladder=[0.25, 0.5, 1, 2]))
            for i, (v, count) in enumerate(zip(values, counts, strict=True)) for n in range(count)]
