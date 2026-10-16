"""The stream protocol's usage report (`stbench/stream_protocol.py`): which reports are well formed, how a report is
summed over its models and when it names a remote store. Every report below is written out by hand."""

from __future__ import annotations

import pytest

from stbench.stream_protocol import MODEL_TOKEN_KEYS, remote_memory, usage_problems, usage_totals


def tokens(tokens_in=100, cached=40, written=0, out=10) -> dict:
    """One model's entry of a report, the four counts in the order of `MODEL_TOKEN_KEYS`."""
    return {"tokens_in": tokens_in, "tokens_cached_in": cached, "tokens_cache_write_in": written, "tokens_out": out}


CODEX = {"models": {"gpt-6-luna": tokens(600, 240, 0, 60)}, "peak_context": None, "remote_store_bytes": None}
TWO = {"models": {"m1": tokens(1000, 400, 100, 20), "m2": tokens(50, 0, 0, 5)}, "peak_context": 900,
       "remote_store_bytes": 0}
SCRIPTED = {"self_reported": True, "models": None, "peak_context": None, "remote_store_bytes": None}


def test_a_report_is_read_by_hand():
    """Contract: `usage_problems` is [] for no report, a Codex report (one model, no cache writes), a two-model report
    and the all-null report a scripted system gets (with the harness's `self_reported`); and it names the problem for
    each of an unknown key, `models` as a list (a list of model ids), an empty model id, an entry missing a key or
    holding a fifth, a negative, a float, a bool and a string count, cached plus written above `tokens_in`, a
    negative `peak_context` and a string `remote_store_bytes`. Regression caught: a malformed report priced as if it
    were good, or a list of ids read as per-model counts."""
    assert usage_problems(None) == []
    assert usage_problems(CODEX) == [] and usage_problems(TWO) == [] and usage_problems(SCRIPTED) == []
    assert usage_problems({"models": {}}) == []  # a system that called no model
    bad = {
        "an unknown key": ({**CODEX, "tokens_in": 5}, ["unknown key 'tokens_in'"]),
        "models a list": ({**CODEX, "models": ["gpt-6-luna"]}, ["models is not null or an object by model id"]),
        "an empty model id": ({**CODEX, "models": {"": tokens()}}, ["a model id is not a non-empty string"]),
        "an entry that is not an object": ({**CODEX, "models": {"m1": 5}},
                                           ["the tokens of model 'm1' are not an object"]),
        "a missing key": ({**CODEX, "models": {"m1": {"tokens_in": 1, "tokens_cached_in": 0, "tokens_out": 0}}},
                          ["model 'm1' lacks tokens_cache_write_in"]),
        "a fifth key": ({**CODEX, "models": {"m1": {**tokens(), "reasoning": 3}}},
                        ["model 'm1' has an unknown key 'reasoning'"]),
        "a negative count": ({**CODEX, "models": {"m1": tokens(out=-1)}},
                             ["tokens_out of model 'm1' is not null or a non-negative integer: -1"]),
        "a float count": ({**CODEX, "models": {"m1": tokens(tokens_in=1.5)}},
                          ["tokens_in of model 'm1' is not null or a non-negative integer: 1.5"]),
        "a bool count": ({**CODEX, "models": {"m1": tokens(cached=True)}},
                         ["tokens_cached_in of model 'm1' is not null or a non-negative integer: True"]),
        "a string count": ({**CODEX, "models": {"m1": tokens(written="3")}},
                           ["tokens_cache_write_in of model 'm1' is not null or a non-negative integer: '3'"]),
        "cached and written above tokens_in": ({**CODEX, "models": {"m1": tokens(100, 60, 50)}},
                                               ["tokens_in of model 'm1' is below tokens_cached_in"]),
        "a negative peak": ({**CODEX, "peak_context": -5}, ["peak_context is not null or a non-negative integer"]),
        "a string store": ({**CODEX, "remote_store_bytes": "12"},
                           ["remote_store_bytes is not null or a non-negative integer"]),
    }
    for name, (usage, expected) in bad.items():
        found = usage_problems(usage)
        assert len(found) == len(expected), (name, found)
        assert all(want in problem for want, problem in zip(expected, found, strict=True)), (name, found)
    assert usage_problems({"models": {"m1": tokens(100, 60, 40)}}) == []  # cached + written equal to tokens_in is fine
    assert usage_problems({"models": {"m1": tokens(None, 60, 50)}}) == []  # the sum is checked only when all are known
    assert usage_problems([1, 2]) == ["usage is not an object: [1, 2]"]


def test_totals_sum_the_models_and_keep_what_is_unknown():
    """Contract: `usage_totals` sums each of the four counts over the models; a count that is null for any model is
    None in the total, whatever the other models hold; `models` null or no report gives None for all four; and
    `models` `{}` gives zeros; and a report `usage_problems` finds a problem with (a list of model ids, an
    entry missing a count, a bool count) totals to None for all four instead of raising, as `cost_usd` refuses it.
    Regression caught: a missing count silently read as 0, so a cost is understated, or one buggy agent's `finish`
    crashing the eval layer that sums every case."""
    assert usage_totals(TWO) == {"tokens_in": 1050, "tokens_cached_in": 400, "tokens_cache_write_in": 100,
                                 "tokens_out": 25}
    assert usage_totals(CODEX) == {"tokens_in": 600, "tokens_cached_in": 240, "tokens_cache_write_in": 0,
                                   "tokens_out": 60}
    partly = {"models": {"m1": tokens(10, 4, 0, 1), "m2": tokens(20, None, 0, 2)}}
    assert usage_totals(partly) == {"tokens_in": 30, "tokens_cached_in": None, "tokens_cache_write_in": 0,
                                    "tokens_out": 3}
    nothing = dict.fromkeys(MODEL_TOKEN_KEYS)
    assert usage_totals(SCRIPTED) == nothing and usage_totals(None) == nothing
    assert usage_totals({"models": {}}) == dict.fromkeys(MODEL_TOKEN_KEYS, 0)
    for malformed in ({"models": ["m1"]}, {"models": {"m1": {"tokens_in": 5}}}, {"models": {"m1": tokens(cached=True)}},
                      {**CODEX, "tokens_in": 5}, [1, 2]):
        assert usage_problems(malformed), malformed
        assert usage_totals(malformed) == nothing, malformed


@pytest.mark.parametrize(("usage", "remote"), [
    ({"remote_store_bytes": 0}, True), ({"remote_store_bytes": 12}, True), ({"remote_store_bytes": None}, False),
    ({"models": {}}, False), ({}, False), (None, False), (SCRIPTED, False), (TWO, True)])
def test_a_reported_store_is_remote_memory(usage, remote):
    """Contract (D4): `remote_memory` is True exactly when `remote_store_bytes` is a number, 0 included, and False when
    it is null, missing, or there is no report. Regression caught: a run with a remote store scored as verified, or a
    store of 0 bytes read as none."""
    assert remote_memory(usage) is remote
