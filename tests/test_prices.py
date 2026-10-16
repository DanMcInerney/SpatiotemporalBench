"""The price table (`stbench/prices.py`, `stbench/prices.json`): a case's cost is its reported tokens times the table,
and is None, with its reason, when it cannot be priced. The hand table below is the one the cost tests price against;
the shipped table is checked for what it must hold, not for its numbers (no price is recorded in a test)."""

from __future__ import annotations

import json
import re
from datetime import date
from importlib import resources

import pytest

from stbench import prices

HAND = {"m1": {"in": 2.0, "cached_in": 0.2, "cache_write_in": 2.5, "out": 10.0},
        "m2": {"in": 1.0, "cached_in": 0.1, "cache_write_in": 1.25, "out": 5.0}}


def entry(tokens_in, cached, written, out) -> dict:
    return {"tokens_in": tokens_in, "tokens_cached_in": cached, "tokens_cache_write_in": written, "tokens_out": out}


@pytest.fixture
def hand_table(monkeypatch):
    monkeypatch.setattr(prices, "table", lambda: HAND)


def test_cost_is_tokens_times_the_table_by_hand(hand_table):
    """Contract: a model's cost is (tokens_in - cached - written) x in + cached x cached_in + written x cache_write_in
    + out x out, over a million, summed over the models. m1 = 1,000,000 in with 400,000 read from the cache and 100,000
    written, and 20,000 out: 500,000 x $2 + 400,000 x $0.20 + 100,000 x $2.50 + 20,000 x $10 = $1.53 per million, and
    m2 = 10,000 in and 2,000 out = $0.02, so the report costs exactly $1.55. Regression caught: cache reads or writes
    priced as fresh input, a model left out of the sum, or a wrong unit."""
    usage = {"self_reported": True, "models": {"m1": entry(1_000_000, 400_000, 100_000, 20_000),
                                               "m2": entry(10_000, 0, 0, 2_000)},
             "peak_context": None, "remote_store_bytes": None}
    assert prices.cost_usd(usage) == 1.55 and prices.cost_note(usage) is None
    assert prices.cost_usd({"models": {"m2": entry(10_000, 0, 0, 2_000)}}) == 0.02


def test_no_cost_without_a_price_or_a_count(hand_table):
    """Contract: `cost_usd` is None, and `cost_note` says why, for no report, a malformed one, `models` null, a model
    the table lacks (even beside a priced one) and a count the system did not report; a system that called no model
    costs 0.0 with no note, and so does a priced report none. Regression caught: a guessed cost for a model the table
    lacks, or a cost read from an incomplete report (a null count priced as 0)."""
    priced = entry(100, 10, 0, 5)
    cases = {
        "no report": (None, "no usage reported"),
        "a list of ids": ({"models": ["m1"]},
                                   "usage: models is not null or an object by model id: ['m1']"),
        "a negative count": ({"models": {"m1": entry(100, 10, 0, -5)}},
                             "usage: tokens_out of model 'm1' is not null or a non-negative integer: -5"),
        "models null": ({"self_reported": True, "models": None, "peak_context": None,
                         "remote_store_bytes": None}, "no token counts reported"),
        "no report body": ({}, "no token counts reported"),
        "an unpriced model": ({"models": {"m3": priced}}, "no price for m3"),
        "an unpriced model beside a priced one": ({"models": {"m1": priced, "m3": priced}}, "no price for m3"),
        "an unknown count": ({"models": {"m1": entry(100, None, 0, 5)}}, "a token count of m1 is unknown"),
        "an unknown count beside a priced model": ({"models": {"m2": priced, "m1": entry(None, 0, 0, 5)}},
                                                   "a token count of m1 is unknown"),
    }
    for name, (usage, note) in cases.items():
        assert (prices.cost_usd(usage), prices.cost_note(usage)) == (None, note), name
    assert (prices.cost_usd({"models": {}}), prices.cost_note({"models": {}})) == (0.0, None)
    assert prices.cost_note({"models": {"m1": priced}}) is None and prices.cost_usd({"models": {"m1": priced}}) > 0


# the agents the benchmark runs: Codex's two test models, and through Claude Code (0.11.0) haiku, sonnet and opus plus
# the model Claude Code uses for its background calls
SHIPPED_IDS = {"gpt-6-luna", "gpt-6.1-sol", "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5",
               "claude-haiku-4-5"}


def test_the_shipped_table_prices_the_benchmarks_agents():
    """Contract: the shipped table holds exactly the model ids of the benchmark's agents (the Codex reference agent's
    two test models, the Claude Code agent's haiku, sonnet and opus, and the model Claude Code uses for its background
    calls), each with four
    positive prices and a cache read cheaper than fresh input; every entry of `prices.json` has an https `source` and
    an ISO `retrieved` date. Regression caught: an agent row with no cost, or an unsourced price."""
    shipped = prices.table()
    assert set(shipped) == SHIPPED_IDS
    for model, row in shipped.items():
        assert tuple(row) == prices.PRICE_KEYS and all(price > 0 for price in row.values()), model
        assert row["cached_in"] < row["in"], model
    raw = json.loads(resources.files("stbench").joinpath("prices.json").read_text(encoding="utf-8"))
    assert set(raw["models"]) == SHIPPED_IDS
    for model, row in raw["models"].items():
        assert row["source"].startswith("https://") and len(row["source"]) > 12, model
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["retrieved"]) and date.fromisoformat(row["retrieved"]), model


GOOD = {"in": 1.0, "cached_in": 0.1, "cache_write_in": 1.25, "out": 5.0, "source": "https://example.com/p",
        "retrieved": "2026-10-07"}


@pytest.mark.parametrize(("raw", "message"), [
    ({"unit": "USD per 1,000 tokens", "models": {"m": GOOD}}, "unit"),
    ({"unit": "USD per 1,000,000 tokens", "models": {}}, "models"),
    ({"unit": "USD per 1,000,000 tokens", "models": {"m": {**GOOD, "out": None}}}, "'out'"),
    ({"unit": "USD per 1,000,000 tokens", "models": {"m": {**GOOD, "in": 0}}}, "'in'"),
    ({"unit": "USD per 1,000,000 tokens", "models": {"m": {**GOOD, "cached_in": -0.1}}}, "'cached_in'"),
    ({"unit": "USD per 1,000,000 tokens", "models": {"m": {**GOOD, "cache_write_in": float("inf")}}},
     "'cache_write_in'"),
    ({"unit": "USD per 1,000,000 tokens", "models": {"m": {**GOOD, "in": True}}}, "'in'"),
    ({"unit": "USD per 1,000,000 tokens", "models": {"m": {**GOOD, "source": "http://example.com"}}}, "source"),
    ({"unit": "USD per 1,000,000 tokens", "models": {"m": {k: v for k, v in GOOD.items() if k != "source"}}},
     "source"),
    ({"unit": "USD per 1,000,000 tokens", "models": {"m": {**GOOD, "retrieved": "2026-13-01"}}}, "retrieved"),
    ({"unit": "USD per 1,000,000 tokens", "models": {"m": {**GOOD, "retrieved": "7 Oct 2026"}}}, "retrieved"),
])
def test_a_malformed_table_is_refused(raw, message):
    """Contract: `parse` accepts an entry with four positive finite prices, an https `source` and an ISO `retrieved`
    date, and raises ValueError, naming what is wrong, for another unit, no models, a missing, zero, negative, infinite
    or boolean price, a source that is not https or is missing, and a date that is not a real ISO one. Regression
    caught: a price table that loads with a wrong unit, a free model or an unsourced number, which would understate or
    invent a cost."""
    assert prices.parse({"unit": "USD per 1,000,000 tokens", "models": {"m": GOOD}}) == {
        "m": {"in": 1.0, "cached_in": 0.1, "cache_write_in": 1.25, "out": 5.0}}
    with pytest.raises(ValueError, match=message):
        prices.parse(raw)
