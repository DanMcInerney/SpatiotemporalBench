"""What a case of `stbench eval` costs (`eval_run.py`'s budget, `eval_results.py`'s resource fields, plan §5.1's B3 and
M5 §3.1): the spend cap and the tokens, remote-store stamp and dollars of each case, which come from M5's functions and
never from an agent. The fake agent is `eval_kit.fake_agent` (`usage`: two models and a remote store, priced by the hand
table in `eval_kit`, $0.00339 a case)."""

from __future__ import annotations

import json

import pytest

from eval_kit import COST, PRICES, TOKENS, cli, compiled, hand_dataset, hand_script, register_agent, results
from stbench import eval_results, prices


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("eval_spend")
    hand_dataset(root)
    return root


def test_the_budget_stops_new_cases(data, tmp_path, monkeypatch, capsys):
    """Contract (plan §5.1, design §6.3): with `--max-usd TOTAL --case-usd X` a case starts only while what the folders
    spent plus what the cases in flight reserve plus X stays within TOTAL. With every case priced at $0.40 and X $0.50
    a $1.00 cap admits two of four (0.4 + 0.5 and 0.8 + 0.5 are within it, 1.2 + 0.5 is not), the result is partial
    and says the cap stopped it, the exit is 1; a case that cannot be priced counts as X, so X $0.40 admits two of
    the four too (0.8 spent), and the same cap on a rerun admits none (what the folders spent is counted). Regression
    caught: spending past the cap, a case with no price spent for free, or a resume that forgets what was spent."""
    register_agent(monkeypatch)
    four = tmp_path / "four"
    for seed in range(1, 5):
        compiled(four, hand_script(seed))
    monkeypatch.setattr(prices, "cost_usd", lambda usage: 0.4)
    out = tmp_path / "priced"
    assert cli("fake:ok", "--data", four, "--out", out, "--max-usd", "1", "--case-usd", "0.5") == 1
    doc = results(out)
    assert sorted(c["status"] for c in doc["cases"]) == ["complete", "complete", "pending", "pending"]
    assert doc["partial"] is True and doc["max_usd"] == 1.0 and doc["cost_usd"] == pytest.approx(0.8)
    assert any("the --max-usd cap of $1.00 stopped new cases" in note for note in doc["notes"])
    assert len(list((out / "cases").iterdir())) == 2
    monkeypatch.setattr(prices, "cost_usd", lambda usage: None)
    unpriced = tmp_path / "unpriced"
    assert cli("fake:ok", "--data", four, "--out", unpriced, "--max-usd", "1", "--case-usd", "0.4") == 1
    assert sorted(c["status"] for c in results(unpriced)["cases"]).count("complete") == 2
    ran = len(list((unpriced / "cases").iterdir()))
    assert cli("fake:ok", "--data", four, "--out", unpriced, "--max-usd", "1", "--case-usd", "0.4") == 1
    assert len(list((unpriced / "cases").iterdir())) == ran == 2
    capsys.readouterr()


def test_the_cap_counts_every_arm(tmp_path, monkeypatch):
    """Contract (plan §5.1): the spend cap is over the run folders of every arm, and the arms of one case run one after
    another under it. With every run priced at $0.40 and $0.50 reserved, a $1.00 cap admits two runs (0.4 + 0.5, then
    0.8 + 0.5 is over): the two arms of the first case, and none of the second. Regression caught: a cap per folder,
    which the arms would spend twice over, or an arm that jumps the queue of another case."""
    register_agent(monkeypatch)
    for seed in (1, 2):
        compiled(tmp_path, hand_script(seed))
    monkeypatch.setattr(prices, "cost_usd", lambda usage: 0.4)
    out = tmp_path / "run"
    assert cli("fake:ok", "--data", tmp_path, "--out", out, "--arms", "during=on,off", "--max-usd", "1",
               "--case-usd", "0.5") == 1
    docs = [results(out / arm) for arm in ("during-on", "during-off")]
    assert [sorted(c["status"] for c in d["cases"]) for d in docs] == [["complete", "pending"]] * 2
    assert docs[0]["cases"][0]["status"] == docs[1]["cases"][0]["status"]   # the same case ran in both arms
    assert all(d["partial"] and d["max_usd"] == 1.0 for d in docs)


def test_resources_come_from_m5s_functions(data, tmp_path, monkeypatch):
    """Contract (M5 §3.1): a case's `tokens`, `remote_memory`, `usage_problems`, `cost_usd` and `cost_note` are
    `stream_protocol`'s and `prices`' functions of the usage it reported, never a figure of the agent's own. An agent
    reporting two models (1,000 + 500 tokens in, 200 of them cached, 100 + 50 out) and a remote store has the totals
    1,500, 200, 0 and 150, the cost $0.00339 by the hand table (800 x $2 + 200 x $0.2 + 100 x $10 and 500 x $1 + 50 x
    $5, over a million), and the result is not `verified` and says so; an agent reporting a malformed usage has no
    cost, its note and the result's say why, and the result is still verified. Regression caught: a cost read from an
    agent, or a remote store scored as a verified one."""
    register_agent(monkeypatch)
    monkeypatch.setattr(prices, "table", lambda: PRICES)
    out = tmp_path / "usage"
    assert cli("fake:usage", "--data", data, "--out", out) == 0
    doc = results(out)
    case = doc["cases"][0]
    assert case["tokens"] == {"tokens_in": 1500, "tokens_cached_in": 200, "tokens_cache_write_in": 0,
                              "tokens_out": 150}
    assert (case["remote_memory"], case["usage_problems"], case["cost_usd"], case["cost_note"]) == (
        True, [], pytest.approx(COST), None)
    assert case["usage"] == {"self_reported": True, "models": TOKENS, "peak_context": 777, "remote_store_bytes": 4096}
    assert doc["verified"] is False and doc["cost_usd"] == pytest.approx(2 * COST)
    assert any("remote store" in note and "unverified" in note for note in doc["notes"])
    assert eval_results.problems(doc) == []
    summary = json.loads((out / "cases" / case["id"] / "summary.json").read_text(encoding="utf-8"))
    assert (summary["cost_usd"], summary["remote_memory"]) == (pytest.approx(COST), True)
    monkeypatch.setattr("eval_kit.TOKENS", {"m1": "lots"})
    bad = tmp_path / "bad"
    assert cli("fake:usage", "--data", data, "--out", bad) == 0
    doc = results(bad)
    assert doc["cases"][0]["usage_problems"] and doc["cases"][0]["cost_usd"] is None
    assert doc["cases"][0]["cost_note"].startswith("usage: ") and doc["cost_usd"] is None
    assert doc["cases"][0]["tokens"] == dict.fromkeys(case["tokens"])
    assert any("malformed usage" in note for note in doc["notes"])
