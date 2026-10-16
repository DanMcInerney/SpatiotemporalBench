"""The credit sensitivity of a run (0.7.0, FD-0.7.0 §2.3; the need of the graded credit): how far the headline moves
if the job's numbers were looser or tighter, by re-scoring the same answers with the gold rows' need multiplied. No
recompile is needed.

A job's clearance C is the need of an offset, a number or a heading (`tolerances.need_m` of a `graded` or `regret` row;
before the graded credit the middle rung of an offset's ladder, `[C/2, C, 2C]`) and its re-find radius R the need of an
`object_where` place (`need_m` of a `graded` row; before it the middle rung of `[R/2, R, 2R]`, each rung plus the
widening w of a thing whose top was hidden, `rules.hidden_widening`). So there are two sensitivities, each reported on
its own:
- **C**: every offset, number and heading gold has its need multiplied by the factor, which is scaling C by it; a
  number's or a heading's threshold `threshold_m` (the body's length plus C) moves with it, by the change in C; a
  ladder offset gold has its rungs multiplied;
- **R**: every `object_where` gold of the `graded` credit has its `need_m` multiplied (its resolution, which holds the
  widening, is kept); a ladder place of the form `[R/2 + w, R + w, 2R + w]` for its job's `reacquire_m` has its R part
  multiplied and its w kept: `w = ladder[1] - reacquire_m`, the new rungs `(rung - w) * factor + w`. A gold of another
  form (a compile before 0.7.0 has relative ladders) is left as it is, and the report counts the golds scaled.
Nothing else is scaled: not a list's match radius, `path_ahead`, `distance_between`, the change or self families, a time
ladder, nor any other credit.

Each row is scored by `score.row`, the scorer's own credit, on the scaled gold, and the layout is aggregated as
`eval_results.build` does (the dataset's thin pairs), so at a factor of 1 both headlines are the run's published one.
Reads the run folder (`results.json`, `cases/<cid>/answers.jsonl`) and the dataset's golds; adds nothing to either.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from . import ability_counts, eval_results, job_registry, score
from .case import PRIVATE
from .eval_distance import check_gold, raw_answers, run_folder
from .eval_report import interval, signed
from .score_stats import INFO_KEYS, aggregate
from .stream_protocol import failed
from .suite import read_jsonl

SENSITIVITY_FACTORS = (0.5, 1.0, 2.0)   # the rungs' scales the credit is re-scored at
SHAPE_TOL_M = 1e-6                      # how far a ladder may be from `[R/2 + w, R + w, 2R + w]` and still be one
LINES = (("C", "offset, number and heading answers, the clearance C scaled"),
         ("R", "`object_where` places, the re-find radius R scaled"))
C_TYPES = ("offset", "number", "heading")   # the answers whose need is the clearance C


def _ladder(gold: dict) -> list | None:
    """The gold row's `tolerances.ladder` when its credit is the ladder, else None."""
    return (gold.get("tolerances") or {}).get("ladder") if gold.get("credit") in (None, "ladder") else None


def is_offset_ladder(gold: dict) -> bool:
    """Whether the gold row is an offset on a ladder it carries: what a ladder C sensitivity scales."""
    return gold["gold"].get("type") == "offset" and bool(_ladder(gold))


def is_graded(gold: dict, *types: str) -> bool:
    """Whether the gold row is of the `graded` credit (`regret` for a heading) on a gold of one of `types`."""
    return gold.get("credit") in ("graded", "regret") and gold["gold"].get("type") in types


def is_c_row(gold: dict) -> bool:
    """Whether the C sensitivity scales the gold row: an offset on a ladder, or a graded offset, number or heading."""
    return is_offset_ladder(gold) or is_graded(gold, *C_TYPES)


def scaled_need(gold: dict, factor: float) -> dict:
    """The graded gold row with its `need_m` multiplied by `factor` and its `threshold_m` (the body's length plus the
    need) moved by the change in the need; its resolution is kept. A catastrophe rule of an `over` or `radial` row has a
    margin that is `max(need, 2 rho)` (`rules.with_catastrophe`), so it is made so again from the new need, and its
    own `threshold_m` moves with the row's; a `miss` rule's points and the match radius stay."""
    tolerances = gold["tolerances"]
    change = tolerances["need_m"] * (factor - 1)
    scaled = {**tolerances, "need_m": tolerances["need_m"] + change}
    if "threshold_m" in tolerances:
        scaled["threshold_m"] = tolerances["threshold_m"] + change
    if (rule := tolerances.get("catastrophe")) and rule["rule"] != "miss":
        margin = max(scaled["need_m"], 2 * tolerances["resolution_m"])
        moved = {"threshold_m": round(rule["threshold_m"] + change, 6)} if "threshold_m" in rule else {}
        scaled["catastrophe"] = {**rule, "margin_m": round(margin, 6), **moved}
    return {**gold, "tolerances": scaled}


def place_widening(gold: dict, reacquire_m: float) -> float | None:
    """The widening w of an `object_where` gold row whose ladder is `[R/2 + w, R + w, 2R + w]` for `reacquire_m` = R
    (`w = ladder[1] - R`, at least 0), else None: the row is not a place on R's rungs (another family, another type, a
    relative ladder of a compile before 0.7.0)."""
    ladder = _ladder(gold)
    if gold.get("family") != "object_where" or gold["gold"].get("type") != "position" or not ladder or len(ladder) != 3:
        return None
    w = ladder[1] - reacquire_m
    shaped = (abs(ladder[0] - w - reacquire_m / 2) <= SHAPE_TOL_M
              and abs(ladder[2] - w - 2 * reacquire_m) <= SHAPE_TOL_M)
    return max(0.0, w) if shaped and w >= -SHAPE_TOL_M else None


def _factor(factor: float) -> None:
    if isinstance(factor, bool) or not isinstance(factor, int | float) or not factor > 0:
        raise ValueError(f"factor must be a number above 0, got {factor!r}")


def scaled_offset(gold: dict, factor: float) -> dict:
    """The gold row with C scaled by `factor`: a graded offset, number or heading has its need scaled (`scaled_need`),
    an offset on a ladder every rung; any other row unchanged. ValueError for a factor that is not a number above 0."""
    _factor(factor)
    if is_graded(gold, *C_TYPES):
        return scaled_need(gold, factor)
    if not is_offset_ladder(gold):
        return gold
    return {**gold, "tolerances": {**gold["tolerances"], "ladder": [x * factor for x in gold["tolerances"]["ladder"]]}}


def scaled_place(gold: dict, factor: float, reacquire_m: float) -> dict:
    """The gold row of a graded `object_where` place with its need R multiplied by `factor`, or of one on R's rungs
    (`place_widening`) with R multiplied and its widening kept; any other row unchanged. ValueError for a factor that is
    not a number above 0."""
    _factor(factor)
    if is_graded(gold, "position"):
        return scaled_need(gold, factor)
    w = place_widening(gold, reacquire_m)
    if w is None:
        return gold
    return {**gold, "tolerances": {**gold["tolerances"],
                                   "ladder": [(x - w) * factor + w for x in gold["tolerances"]["ladder"]]}}


def _reacquire(row: dict) -> float:
    try:
        return job_registry.get(row["job"]).reacquire_m
    except KeyError as error:
        raise ValueError(f"{error.args[0]}: no re-find radius to scale") from None


def _layout_sensitivity(mine: list[dict], golds: Mapping[tuple[str, str], dict],
                        answers: Mapping[tuple[str, str], object], thin: frozenset, thin_ignored: bool,
                        factors: tuple[float, ...]) -> dict:
    """`credit_sensitivity`'s entry for one layout's result rows `mine`."""
    keys = [(r["case"], r["qid"]) for r in mine]
    graded_places = {key for key in keys
                     if golds[key].get("family") == "object_where" and is_graded(golds[key], "position")}
    reach = {key: _reacquire(r) for r, key in zip(mine, keys, strict=True)   # the ladder places on R's rungs: their R
             if key not in graded_places and golds[key].get("family") == "object_where"
             and place_widening(golds[key], _reacquire(r)) is not None}

    def headline(pick: Callable[[tuple[str, str], float], dict], factor: float) -> dict:
        again = [score.row(answers.get(key, failed("unanswered")), pick(key, factor), when=r["when"],
                           info={name: r[name] for name in INFO_KEYS}) for r, key in zip(mine, keys, strict=True)]
        return aggregate(again, thin=thin, thin_ignored=thin_ignored)["headline"]

    def offsets(key: tuple[str, str], factor: float) -> dict:
        return scaled_offset(golds[key], factor)

    def places(key: tuple[str, str], factor: float) -> dict:
        if key in reach or key in graded_places:
            return scaled_place(golds[key], factor, reach.get(key, 0.0))
        return golds[key]

    scaled = {"C": sum(is_c_row(golds[key]) for key in keys), "R": len(reach) + len(graded_places)}
    return {"questions": len(mine), "scaled": scaled,
            "headline": {"C": {f: headline(offsets, f) for f in factors},
                         "R": {f: headline(places, f) for f in factors}}}


def credit_sensitivity(rows: list[dict], golds: Mapping[tuple[str, str], dict],
                       answers: Mapping[tuple[str, str], object], *, thin: frozenset = frozenset(),
                       thin_ignored: bool = False, factors: tuple[float, ...] = SENSITIVITY_FACTORS) -> dict[str, dict]:
    """The headline of each layout of the result `rows` re-scored with the offsets' rungs scaled (`C`) and, apart, the
    places' R scaled (`R`) by each of `factors`: `{layout: {"questions": n, "scaled": {"C": n, "R": n}, "headline":
    {"C": {factor: {mean, ci}}, "R": {...}}}}`, `scaled` the rows whose ladder is scaled. `golds` holds the gold rows
    and `answers` the raw answers by (case, qid); a question with no answer scores as the run scored it, as
    `failed("unanswered")`. `thin`, `thin_ignored` are the aggregate's, as `eval_results.build` passes them. ValueError
    for a row of a job that is not registered, when its gold is an `object_where` one."""
    return {layout: _layout_sensitivity([r for r in rows if r["layout"] == layout], golds, answers, thin, thin_ignored,
                                        factors)
            for layout in sorted({r["layout"] for r in rows})}


def _same_stat(a: dict | None, b: dict | None) -> bool:
    """Whether two `{mean, ci}` statistics are the same number, up to the rounding of a JSON round trip."""
    if a is None or b is None or (a["mean"] is None) != (b["mean"] is None) or (a["ci"] is None) != (b["ci"] is None):
        return False
    pairs = [] if a["mean"] is None else [(a["mean"], b["mean"])]
    pairs += [] if a["ci"] is None else list(zip(a["ci"], b["ci"], strict=True))
    return all(abs(x - y) <= 1e-9 for x, y in pairs)


def sensitivity_lines(found: Mapping[str, dict], published: Mapping[str, dict]) -> list[str]:
    """The markdown of `credit_sensitivity`'s `found`: per layout a table for C and one for R, a line for each factor
    with the headline and its 95% interval, and a line saying whether the factor 1 headline is the `published` one
    (`{layout: {mean, ci}}`, the run's own `layouts[layout]["headline"]`), so a mismatch shows."""
    lines = ["# SpatiotemporalBench: credit sensitivity", "",
             "The headline re-scored with the need of one kind of answer multiplied by a factor (the rungs of a "
             "ladder), which is scaling the job's clearance C or its re-find radius R by it; the same "
             "answers, the scorer's own credit. The resolution, which holds the widening of a thing whose top was "
             "hidden, is kept when R is scaled. Nothing else is scaled (a list's matching radius, `path_ahead`, "
             "`distance_between`, the change and self families, times).", ""]
    for layout, entry in found.items():
        lines += [f"## {layout}", ""]
        for line, title in LINES:
            lines += [f"### {line}: {entry['scaled'][line]} of {entry['questions']} questions: {title}", "",
                      "| rungs scaled by | headline | 95% CI |", "| --- | --- | --- |",
                      *(f"| x{factor:g} | {signed(stat['mean'])} | {interval(stat['ci'])} |"
                        for factor, stat in entry["headline"][line].items()), ""]
        base = [entry["headline"][line].get(1.0) for line, _ in LINES]
        same = all(_same_stat(stat, published.get(layout)) for stat in base)
        lines += ["- the x1 headline is the published headline" if same else
                  f"- the x1 headline is NOT the published one ({signed((published.get(layout) or {}).get('mean'))}): "
                  "the answers or golds are not the ones the run was scored on", ""]
    return lines


def sensitivity(run: Path, data: Path, factors: tuple[float, ...] = SENSITIVITY_FACTORS, *,
                cells_from: Path | None = None) -> str:
    """The credit sensitivity report of the run at `run` (a results folder or file) over `data`'s golds, beside the
    published headline, which the factor 1 must equal. The answers are the case folders' `answers.jsonl`, the thin cells
    those of the dataset or of `cells_from` (`ability_counts.thin_rule`), as `eval_results.build` took them. ValueError
    for results scored on another compile than the golds (`check_gold`) or results that cannot be read."""
    doc = eval_results.load(run)
    check_gold(doc, data, str(run))
    folder, golds, answers = run_folder(run), {}, {}
    for case in sorted({row["case"] for row in doc["rows"]}):
        golds |= {(case, g["qid"]): g for g in read_jsonl(data / PRIVATE / case / "gold.jsonl")}
        answers |= {(case, qid): answer for qid, answer in raw_answers(folder, case).items()}
    if missing := [(row["case"], row["qid"]) for row in doc["rows"] if (row["case"], row["qid"]) not in golds]:
        raise ValueError(f"{run}: {len(missing)} scored question(s) have no gold in {data}, e.g. {missing[0]}")
    if cells_from is not None:
        ability_counts.require_cases(cells_from)
    thin, relaxed = ability_counts.thin_rule(cells_from or data)
    found = credit_sensitivity(doc["rows"], golds, answers, thin=thin, thin_ignored=relaxed, factors=factors)
    published = {layout: agg["headline"] for layout, agg in doc["layouts"].items() if agg}
    return "\n".join(sensitivity_lines(found, published)).rstrip("\n") + "\n"
