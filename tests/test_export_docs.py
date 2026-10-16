"""The README, the dataset card and `docs/format.md` against the code they describe."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

import export_kit as kit
from stbench import cli, export_files

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
CARD = (ROOT / "src" / "stbench" / "release" / "dataset-card.md").read_text(encoding="utf-8")
FORMAT = (ROOT / "docs" / "format.md").read_text(encoding="utf-8")
RELEASE_HEADING = "## A release (`stbench export`)"


def stbench_lines(text: str) -> list[list[str]]:
    """The words of each `stbench ...` command line in the fenced `sh` blocks of a document (comments cut off)."""
    out = []
    for block in re.findall(r"```sh\n(.*?)```", text, re.S):
        for line in block.splitlines():
            words = line.split("#")[0].split()
            if words[:1] == ["stbench"] and len(words) > 1:
                out.append(words)
    return out


def test_the_readme_names_the_headlines_columns():
    """Contract (VISION §3: perception and memory are both scored and reported apart): the README names the headline's
    `now`, `memory` and own-path columns. Regression caught: a rewrite that reports one pooled number, so a score that
    comes from perception alone could be read as memory."""
    assert "**`now`**" in README and "**`memory`**" in README and "own-path" in README


def test_the_commands_in_the_readme_exist_with_their_flags(capsys):
    """Contract: each `stbench` command in a README code block is a real subcommand and every `--flag` it uses is in
    that command's help. Regression caught: a flag renamed or a command removed while the quick start still shows it."""
    commands = stbench_lines(README)
    assert commands
    for words in commands:
        with pytest.raises(SystemExit) as stop:
            cli.main([words[1], "--help"])
        assert stop.value.code == 0, words
        help_text = capsys.readouterr().out
        assert [f for f in words if f.startswith("--") and f not in help_text] == [], words


def test_the_commands_in_the_card_and_format_md_exist_with_their_flags(capsys):
    """Contract: each `stbench` command in a code block of the dataset card and of `docs/format.md` is a real subcommand
    and every `--flag` it uses, an optional one in brackets included, is in that command's help. Regression caught: a
    flag renamed (the card tells a user to run `stbench eval ... --reasoning`) or a command removed while a document
    still shows it."""
    for name, text in (("the card", CARD), ("format.md", FORMAT)):
        commands = stbench_lines(text)
        assert commands, name
        for words in commands:
            with pytest.raises(SystemExit) as stop:
                cli.main([words[1], "--help"])
            assert stop.value.code == 0, (name, words)
            help_text = capsys.readouterr().out
            used = [w.strip("[]") for w in words if w.lstrip("[").startswith("--")]
            assert [f for f in used if f not in help_text] == [], (name, words)
    assert any(words[1] == "export" for words in stbench_lines(FORMAT))


def test_format_md_names_every_file_pattern_of_an_export():
    """Contract: the release section of `docs/format.md` names each file pattern of an export (the five top-level files,
    `results/<name>.json`, `scripts/<sid>/`, the index entries, the four private files of a script and the three of a
    case) and what an export leaves out. Regression caught: a file added to the released set that the format page
    never mentions, or a page that still says an export is to come."""
    assert RELEASE_HEADING in FORMAT
    section = FORMAT.split(RELEASE_HEADING, 1)[1].split("\n## ", 1)[0]
    assert [p for p in (*kit.PATTERNS, *kit.LEFT_OUT) if p not in section] == []
    assert "M7 brings" not in FORMAT


def test_the_card_names_this_datasets_id_and_this_versions_tag():
    """Contract: the dataset card names the dataset by the id of `pyproject.toml`'s Dataset URL, leaves no placeholder
    for it, and links the code repository's files at the tag of the package's own version. Regression caught: a card
    that still says `<DATASET_ID>`, or links the documents of the release before (`blob/v0.4.1/`) so a reader of this
    release is sent to the wrong page."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    dataset_id = project["urls"]["Dataset"].split("/datasets/", 1)[1]
    assert dataset_id in CARD and "<DATASET_ID>" not in CARD
    tags = set(re.findall(r"SpatiotemporalBench/blob/(v[^/]+)/", CARD))
    assert tags == {f"v{project['version']}"}


def test_neither_the_card_nor_format_md_holds_a_machine_path():
    """Contract (the export refuses a released md file with a machine path, and the card ships): the card and
    `docs/format.md` hold no drive path, system folder or `~/` home, by the pattern the export scans with. Regression
    caught: a command example pasted from a terminal with the author's own path in it."""
    for name, text in (("the card", CARD), ("format.md", FORMAT)):
        found = [m.group(0) for m in re.finditer(export_files.ABSOLUTE_PATH, text, re.IGNORECASE)]
        assert found == [], (name, found)
