"""`results.md` (`eval_report.py`, plan §5.1's B3): the notes first, the facts of the run, a block per layout with its
resources, and the cases that did not finish. The dataset is `eval_kit.hand_dataset`; the agent is `eval_kit.fake_agent`
(`usage`: two models and a remote store, priced by the hand table in `eval_kit`, $0.00339 a case)."""

from __future__ import annotations

import re

from eval_kit import COST, PRICES, cli, hand_dataset, register_agent, results
from score_kit import result_row
from stbench import prices
from stbench.eval_report import render
from stbench.eval_results import RELAXED_NOTE
from stbench.score_stats import aggregate


def test_the_report_leads_with_the_notes_and_ends_with_the_resources(tmp_path, monkeypatch):
    """Contract (plan §5.1): `results.md` opens with the notes that say a result is not a benchmark score (partial, a
    remote store, the budget's stop), then the facts of the run (system, options, sandbox, whether it is verified, the
    versions, the prices' version, the cases finished and what they cost under the cap), then the cases that did not
    finish, then per layout the headline, the splits, the outcomes and the resources of the finished cases (tokens
    summed over the models: 1,500 in, 200 of them cached, 150 out; the cost the price table gives, $0.0034), and the
    notes again at the end. A run that reports no usage reads `n/a` where it cannot say. Regression caught: a result
    whose partial or unverified stamp is buried, or resources that sum an unknown as zero."""
    register_agent(monkeypatch)
    monkeypatch.setattr(prices, "table", lambda: PRICES)
    hand_dataset(tmp_path)
    out = tmp_path / "run"
    assert cli("fake:usage", "--data", tmp_path, "--out", out, "--max-usd", "0.01", "--case-usd", "0.008") == 1
    md = (out / "results.md").read_text(encoding="utf-8")
    doc = results(out)
    (done,) = [c["id"] for c in doc["cases"] if c["status"] == "complete"]
    (unfinished,) = [c["id"] for c in doc["cases"] if c["status"] != "complete"]
    lines = md.splitlines()
    assert lines[:4] == ["*** partial: 1 of 2 cases finished; not a benchmark score ***",
                         f"*** 1 case(s) used a remote store (remote memory, e.g. {done}): unverified ***",
                         "*** the --max-usd cap of $0.01 stopped new cases ***", f"*** {RELAXED_NOTE} ***"]
    assert lines[4] == "# SpatiotemporalBench results: fake:usage"
    for line in ("- rgb_pose channel, correct brief, during on", "- sandbox: in-process (the benchmark's own code)",
                 "- verified: NO", "- cases: 1 of 2 finished",
                 f"- cost of the cases that ran: ${COST:.4f} (cap $0.01)", "## Cases not finished",
                 f"- apartment_a {unfinished}: pending", "## Layout apartment_a: 1 of 2 cases finished",
                 "- tokens: in 1,500, of them read from a cache 200, written to one 0, out 150",
                 f"- cost: ${COST:.4f} (${COST:.4f} a case)"):
        assert line in lines, line
    assert re.search(r"^- wall time: \d+\.\d s in all, \d+\.\d s a case$", md, re.M)
    assert lines[-4:] == lines[:4]   # the notes again at the end
    plain = tmp_path / "plain"
    assert cli("all_unknown", "--data", tmp_path, "--out", plain) == 0
    unknown = (plain / "results.md").read_text(encoding="utf-8")
    assert "- tokens: in n/a, of them read from a cache n/a, written to one n/a, out n/a" in unknown
    assert "- cost: n/a" in unknown and "- cost of the cases that ran: n/a" in unknown
    assert unknown.count("*** ") == 2   # only the dataset's note, at the top and at the end


def test_the_report_gives_the_floor_the_lengths_short_to_long_and_the_relevance_classes(tmp_path):
    """Contract (FD-0.5.0-impl §2, "Reporting"): a layout's block shows the score above the floor (the headline less
    the unknown share, which it names) beside the headline, the length table from short to long, and the answerable
    questions by relevance beside the unknown ones. By hand, six questions: a situational answerable -1 and an unknown
    +1 (short), a critical answerable +1 and a situational 0 (medium), a critical +1 and an unknown -1 (long): headline
    1/6, unknown share 2/6, above the floor -1/6; lengths short 0, medium 0.5, long 0; answerable 1/4, critical 1,
    situational -1/2, unknown 0. Regression caught: a headline shown without the floor under it, lengths in
    alphabetical order (long, medium, short), or the critical rows lost among the others."""
    hand_dataset(tmp_path)
    out = tmp_path / "run"
    assert cli("all_unknown", "--data", tmp_path, "--out", out) == 0
    doc = results(out)
    spec = [("short", -1.0, {}), ("short", 1.0, {"slice_": "unknown"}), ("medium", 1.0, {"critical": True}),
            ("medium", 0.0, {}), ("long", 1.0, {"critical": True}), ("long", -1.0, {"slice_": "unknown"})]
    rows = [result_row(f"c{i}", f"q{i}", score, "correct" if score > 0 else "wrong", length=length, **kw)
            for i, (length, score, kw) in enumerate(spec)]
    doc["layouts"]["apartment_a"] = aggregate(rows, n_boot=50)
    doc["rows"] = rows   # the rows the aggregate was made from: the report's shares and bound read them
    lines = render(doc).splitlines()

    def at(prefix: str) -> int:
        (found,) = [i for i, line in enumerate(lines) if line.startswith(prefix)]
        return found
    at("| headline | 6 | +0.167 | [")
    at("| above the floor (the headline less the unknown share, 0.333) | 6 | -0.167 | [")
    assert at("### Per length") < at("| short | 2 | +0.000 |") < at("| medium | 2 | +0.500 |") < at(
        "| long | 2 | +0.000 |") < at("### Per relevance")
    order = ["### Per relevance", "| answerable | 4 | +0.250 |", "| answerable, critical | 2 | +1.000 |",
             "| answerable, situational | 2 | -0.500 |", "| unknown | 2 | +0.000 |"]
    assert [at(prefix) for prefix in order] == sorted(at(prefix) for prefix in order)
