"""What a token costs, by model, and the cost of a case from the tokens a system reported.

The table is `prices.json` (package data): USD per million tokens for each model id an agent of the benchmark runs, from
the provider's own published price page, with the page and the day it was read beside each entry. The base tier only
(GPT-6's long-prompt premium, Anthropic's one-hour cache writes, batch and fast tiers are not modelled; each entry's
`note` says which). A model id is a key of the table and nothing else: nothing here branches on a provider or a model.

A cost is always the case's tokens times the table (`cost_usd`), whatever the run paid (a plan, a discount) and never a
figure of the agent's own. It is None, with its reason in `cost_note`, when the report cannot be priced: missing, not a
well-formed usage report, no token counts, a model the table lacks, or a count the system did not know. Both
answers come from one function, so the number and its reason cannot disagree.

It imports `stream_protocol` and the standard library only, so the harness, the eval layer and a report may all import
it.
"""

from __future__ import annotations

import functools
import json
import math
import re
from datetime import date
from importlib import resources

from .stream_protocol import usage_problems

PER_TOKENS = 1_000_000
PRICE_KEYS = ("in", "cached_in", "cache_write_in", "out")   # a fresh input token, a cache read, a write, an output


def _iso_date(value: object) -> bool:
    """True for a YYYY-MM-DD string that is a real date."""
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def parse(raw: dict) -> dict[str, dict[str, float]]:
    """The table of a `prices.json` object: {model id: {price key: USD per PER_TOKENS}}. ValueError for a wrong unit,
    no models, or an entry without the four positive finite prices, an https `source` and an ISO `retrieved` date."""
    unit = f"USD per {PER_TOKENS:,} tokens"
    if raw.get("unit") != unit:
        raise ValueError(f"prices.json: the unit must be {unit!r}, not {raw.get('unit')!r}")
    models = raw.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("prices.json: `models` must be a non-empty object by model id")
    out: dict[str, dict[str, float]] = {}
    for model, entry in models.items():
        if not isinstance(entry, dict):
            raise ValueError(f"prices.json: the entry of {model!r} is not an object")
        for key in PRICE_KEYS:
            price = entry.get(key)
            if isinstance(price, bool) or not isinstance(price, int | float) or not math.isfinite(price) or price <= 0:
                raise ValueError(f"prices.json: {model!r} needs a positive finite {key!r} price, not {price!r}")
        source, retrieved = entry.get("source"), entry.get("retrieved")
        if not isinstance(source, str) or not re.fullmatch(r"https://\S+", source):
            raise ValueError(f"prices.json: {model!r} needs an https `source`, not {source!r}")
        if not _iso_date(retrieved):
            raise ValueError(f"prices.json: {model!r} needs an ISO `retrieved` date, not {retrieved!r}")
        out[model] = {key: float(entry[key]) for key in PRICE_KEYS}
    return out


@functools.cache
def _shipped() -> dict[str, dict[str, float]]:
    return parse(json.loads(resources.files("stbench").joinpath("prices.json").read_text(encoding="utf-8")))


def table() -> dict[str, dict[str, float]]:
    """prices.json's models as {model id: {price key: USD per PER_TOKENS}}, read once (importlib.resources). ValueError
    for an entry without the four positive finite prices, an https `source` and an ISO `retrieved` date."""
    return {model: dict(prices) for model, prices in _shipped().items()}


def _cost(usage: dict | None) -> tuple[float | None, str | None]:
    """(cost, None) or (None, why): the one place a report is priced."""
    if usage is None:
        return None, "no usage reported"
    problems = usage_problems(usage)
    if problems:
        return None, f"usage: {problems[0]}"
    models = usage.get("models")
    if models is None:
        return None, "no token counts reported"
    prices = table()
    for model in models:
        if model not in prices:
            return None, f"no price for {model}"
    total = 0.0
    for model, tokens in models.items():
        if None in tokens.values():
            return None, f"a token count of {model} is unknown"
        price = prices[model]
        fresh = tokens["tokens_in"] - tokens["tokens_cached_in"] - tokens["tokens_cache_write_in"]
        total += (fresh * price["in"] + tokens["tokens_cached_in"] * price["cached_in"]
                  + tokens["tokens_cache_write_in"] * price["cache_write_in"] + tokens["tokens_out"] * price["out"])
    return round(total / PER_TOKENS, 6), None


def cost_usd(usage: dict | None) -> float | None:
    """The case's cost in USD, or None when it cannot be priced (`cost_note` says why). For each model of a usage
    report: (tokens_in - tokens_cached_in - tokens_cache_write_in) × in + tokens_cached_in × cached_in +
    tokens_cache_write_in × cache_write_in + tokens_out × out, over PER_TOKENS, summed and rounded to 1e-6. `models`
    {} costs 0.0. Nothing but the token counts and the table is read."""
    return _cost(usage)[0]


def cost_note(usage: dict | None) -> str | None:
    """Why `cost_usd(usage)` is None, else None: "no usage reported" (None), "usage: <the first of usage_problems>",
    "no token counts reported" (models null), "no price for <id>", "a token count of <id> is unknown"."""
    return _cost(usage)[1]
