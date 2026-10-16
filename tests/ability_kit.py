"""Result rows and results documents for the tests of scoring by ability (lane A1): every number a test expects is
worked by hand from the scores written here, never from the code under test."""

from __future__ import annotations

from score_kit import result_row
from stbench.score_stats import aggregate

NOW_FAMILIES = ("ego_now", "mover_next", "zone_clear")   # families asked of what is in view (the `now` column)


def r(case: str, qid: str, score: float, family: str, *, job: str = "helper/1", script: str | None = None,
      layout: str = "apartment_a", **kw) -> dict:
    """A result row of `family` scoring `score` (1 correct, 0 abstained, below 0 wrong, else partial); its scope is
    `now` for the families of `NOW_FAMILIES`, else `memory`; its script defaults to the case."""
    outcome = "correct" if score >= 1 else "wrong" if score < 0 else "abstained" if score == 0 else "partial"
    scope = kw.pop("scope", "now" if family in NOW_FAMILIES else "memory")
    return result_row(case, qid, score, outcome, script=script, family=family, scope=scope, job=job, layout=layout,
                      robot=job.split("/")[0], **kw)


def scored(family: str, scores: list[float], *, job: str = "helper/1", case: str = "c1", script: str | None = None,
           **kw) -> list[dict]:
    """One row a score, of `family` on `job`, in one case (qids `<family>-<i>`)."""
    return [r(case, f"{family}-{job}-{i}", s, family, job=job, script=script, **kw) for i, s in enumerate(scores)]


def results_doc(rows: list[dict], *, thin: frozenset = frozenset(), thin_ignored: bool = False, system: str = "sys",
                layout: str = "apartment_a", n_boot: int = 50, **over) -> dict:
    """A minimal results document for `eval compare` and `eval_report.render`: the rows of one layout, their
    aggregate (with `thin`, `thin_ignored` as given), one finished case a case id, no cost."""
    ids = sorted({r_["case"] for r_ in rows})
    held = dict.fromkeys(("tokens_in", "tokens_cached_in", "tokens_cache_write_in", "tokens_out"), 0)
    cases = [{"id": cid, "script": next(r_["script"] for r_ in rows if r_["case"] == cid), "layout": layout,
              "robot": "robot", "job": "helper/1", "length": "short", "status": "complete", "failures": {},
              "wall_s": {}, "state_bytes": {}, "usage": None, "usage_problems": [], "tokens": held,
              "remote_memory": False, "cost_usd": None, "cost_note": None, "error": None}
             for cid in ids]
    doc = {"system": system, "channel": "rgb_pose", "brief": "correct", "during": True, "sandbox": "in-process",
           "reasoning": None, "text_only": False, "verified": True, "partial": False, "max_usd": None,
           "cost_usd": None, "cases": cases, "rows": rows,
           "layouts": {layout: aggregate(rows, n_boot=n_boot, thin=thin, thin_ignored=thin_ignored)}, "notes": []}
    return {**doc, **over}
