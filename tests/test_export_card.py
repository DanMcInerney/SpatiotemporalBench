"""The dataset card (A6): the package's template with one block generated from `release.json`, and what the template
says. Numbers are written out by hand."""

from __future__ import annotations

import re
from itertools import takewhile
from pathlib import Path

import pytest

from stbench import export_card

TEMPLATE = (Path(export_card.__file__).parent / "release" / "dataset-card.md").read_text(encoding="utf-8")
SIZE_CATEGORIES = {"n<1K", "1K<n<10K", "10K<n<100K", "100K<n<1M", "1M<n<10M"}   # Hugging Face's
SECTIONS = ["Summary", "The jobs and the places", "Layout", "The protocol and the run commands",
            "Scoring and the gates", "One public split", "Reference results", "Licence and credits", "Citation",
            "Limitations"]
RELEASE = {
    "package": "0.4.0",
    "groups": [
        {"layout": "apartment_a", "job": "vacuum/2", "scripts": 6, "cases": {"short": 6, "medium": 6, "long": 6},
         "asked_nothing": 1, "questions": 12345, "frames": 4821},
        {"layout": "warehouse_a", "job": "amr/1", "scripts": 4, "cases": {"short": 4, "medium": 3, "long": 4},
         "asked_nothing": 0, "questions": 2000, "frames": 9000}],
    "cases": [], "results": []}
COUNTS = """\
| Layout | Job | Scripts | Cases (short / medium / long) | Cases asking nothing | Questions | Frames |
| --- | --- | --- | --- | --- | --- | --- |
| `apartment_a` | `vacuum/2` | 6 | 6 / 6 / 6 | 1 | 12,345 | 4,821 |
| `warehouse_a` | `amr/1` | 4 | 4 / 3 / 4 | 0 | 2,000 | 9,000 |
| **Total** |  | 10 | 10 / 9 / 10 | 1 | 14,345 | 13,821 |

Exported by stbench 0.4.0.
"""


def card(release: dict = RELEASE) -> str:
    return export_card.card_bytes(release).decode("utf-8")


def test_only_the_counts_block_of_the_card_is_generated():
    """Contract (A6): the card is the template with the lines between `<!-- counts -->` and `<!-- /counts -->` replaced
    and nothing else changed, the dataset id included (the template names it; the reference rows are filled into the
    template itself at the release, by `tools/fill_report.py`). Regression caught: a template filled by search and
    replace, which would change a sentence the template wrote."""
    assert TEMPLATE.count("<!-- counts -->") == 1 and TEMPLATE.count("<!-- /counts -->") == 1
    head, rest = TEMPLATE.split("<!-- counts -->\n")
    tail = rest.split("<!-- /counts -->", 1)[1]
    made = card()
    assert made.startswith(head + "<!-- counts -->\n") and made.endswith("<!-- /counts -->" + tail)
    assert "DanMcInerney/spatiotemporalbench" in TEMPLATE
    assert made.count("DanMcInerney/spatiotemporalbench") == TEMPLATE.count("DanMcInerney/spatiotemporalbench")


def test_the_counts_block_is_the_release_documents_numbers():
    """Contract (A6): the block is a table with a row per (layout, job) and a total, thousands separated, and a last
    line naming the package version (just `stbench` where the document holds none: an uninstalled copy), all read from
    `release.json`: 6 + 4 scripts, cases 10 / 9 / 10 by length, 12,345 + 2,000 questions, 4,821 + 9,000 keyframes.
    Regression caught: a total that double-counts, a column in the wrong order, or a commit or definitions line the
    document does not hold."""
    assert export_card.counts_block(RELEASE) == COUNTS
    assert "<!-- counts -->\n" + COUNTS + "<!-- /counts -->" in card()
    assert export_card.counts_block({**RELEASE, "package": None}).endswith("\n\nExported by stbench.\n")


def test_the_counts_block_counts_questions_by_ability():
    """Contract: when the release document counts questions by ability, the block adds a table of them, a column per
    ability in name order (a job that asks none has 0) and a row per (layout, job) with a total, thousands separated,
    by hand: distance 1,234 + 0, memory 30 + 5, relevance 0 + 8 give totals 1,234, 35 and 8. A document without them
    (an older one) gets no such table. Regression caught: an ability dropped from the columns when
    only one job asks it, a total that reads one job, or a table made for a document that counts nothing."""
    counted = {**RELEASE, "groups": [
        {**RELEASE["groups"][0], "abilities": {"distance": 1234, "memory": 30}},
        {**RELEASE["groups"][1], "abilities": {"memory": 5, "relevance": 8}}]}
    block = export_card.counts_block(counted)
    table = ["| Layout | Job | distance | memory | relevance |", "| --- | --- | --- | --- | --- |",
             "| `apartment_a` | `vacuum/2` | 1,234 | 30 | 0 |", "| `warehouse_a` | `amr/1` | 0 | 5 | 8 |",
             "| **Total** |  | 1,234 | 35 | 8 |"]
    counts, made = COUNTS.split("\n\nExported by")
    assert block == counts + "\n\nQuestions by ability:\n\n" + "\n".join(table) + "\n\nExported by" + made
    assert "Questions by ability" not in export_card.counts_block(RELEASE)


def test_the_card_has_the_first_drafts_sections_in_order():
    """Contract (A6): the card's sections are the summary, the jobs and the places, the layout, the protocol and run
    commands, scoring and the gates, the one public split, the reference rows, licence and credits, the citation and
    the limitations, in that order. Regression caught: a section dropped or a leftover of Release 0.3's card (its two
    tracks, its Perceive set, its three tiers) kept."""
    assert re.findall(r"^## (.+)$", card(), re.M) == SECTIONS
    assert not re.findall(r"\{\{\w+\}\}|Perceive|tier|episodes/", card())


def test_the_front_matter_names_the_licence_and_hides_the_viewer():
    """Contract (A6): the card opens with YAML front matter that says `license: cc-by-4.0`, the pretty name, a list of
    tags, one valid size category and `viewer: false` (the folder is not a table the Hub could show). Regression
    caught: a licence line that disagrees with the data's CC BY 4.0, a size category the Hub does not know, or a viewer
    left on over a folder of images and JSON lines."""
    assert card().startswith("---\n")
    front = card().split("---\n")[1].splitlines()
    assert {"license: cc-by-4.0", "pretty_name: SpatiotemporalBench", "viewer: false"} <= set(front)

    def items(key: str) -> list[str]:
        after = front[front.index(f"{key}:") + 1:]
        return [line[4:].strip('"') for line in takewhile(lambda line: line.startswith("  - "), after)]

    assert items("tags") and set(items("size_categories")) <= SIZE_CATEGORIES and len(items("size_categories")) == 1


def test_the_card_states_the_terms_of_the_release():
    """Contract (A6): the card says that the code is MIT and the data CC BY 4.0, that the pictures and the harness run
    without Blender, and that there is one public split with public gold and no held-out set. Regression caught: a card
    that leaves the old claim of a held-out set or of a secret test split, or of a licence the files do not carry."""
    text = " ".join(TEMPLATE.split())
    for phrase in ("released under [CC BY 4.0](LICENSE)", "code is MIT", "runs without Blender",
                   "There is one split, and all of it is public, gold included: there is no held-out set"):
        assert phrase in text, phrase


def test_a_template_without_exactly_one_counts_block_is_refused_and_line_ends_are_normalised():
    """Contract (A6): a template with no counts block, or two, is a ValueError (the card would not say what the release
    holds, or would say it twice), and a template with Windows line ends gives a card with LF only. Regression caught:
    a template edit that drops a marker and ships a card with the placeholder text in it, or a card whose bytes depend
    on the checkout's `autocrlf`."""
    block = "<!-- counts -->\nx\n<!-- /counts -->"
    for bad in ("no block", f"{block}\n{block}"):
        with pytest.raises(ValueError, match="exactly one"):
            export_card.card_bytes(RELEASE, bad)
    crlf = export_card.card_bytes(RELEASE, "a\r\n" + block.replace("\n", "\r\n") + "\r\nb\r\n")
    assert b"\r" not in crlf and crlf.decode("utf-8").startswith("a\n<!-- counts -->\n| Layout")


def test_the_card_says_the_agent_has_a_camera_and_depth_is_private():
    """Contract (0.6.0, VISION rule 9): the card's layout puts the depth frames under `private/<sid>/`, none under
    `scripts/<sid>/`, says the oracle file (3-D centres) stays agent-facing by design and feeds the `detections`
    channel only, lists the two channels without `rgbd_pose`, and names the five abilities. Regression caught: a card
    that still shows `sensors/depth/` as what a robot receives, or offers a channel the harness refuses."""
    layout = TEMPLATE.split("## Layout")[1].split("```text\n")[1].split("```")[0]
    agent, private = layout.split("private/", 1)
    assert "depth" not in agent and "depth/000000.png" in private and "perception/oracle.jsonl" in agent
    flat = " ".join(TEMPLATE.split())   # the template wraps its lines
    assert "stays in `scripts/<sid>/` by design" in flat and "no agent channel is sent it" in flat
    assert "describe the scorer's sensor, and no agent channel delivers its images" in flat
    assert "| `rgbd_pose` |" not in TEMPLATE and "| `detections` |" in TEMPLATE
    for ability in ("distance", "memory", "relevance", "change", "prediction"):
        assert f"| {ability} |" in TEMPLATE
