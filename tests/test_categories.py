"""The question categories (`stbench.categories`, docs/categories.md): the code, the authoring document and the family
pages say the same thing about which category each family instantiates."""

from __future__ import annotations

import re
from pathlib import Path

from stbench.categories import CATEGORIES, FAMILY_CATEGORY
from stbench.questions_text import FAMILIES

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "categories.md"
FAMILY_PAGES = ROOT / "docs" / "families"
SECTION = re.compile(r"^## (C\d) (.+?)\s*$", re.MULTILINE)
FAMILIES_LINE = re.compile(r"^- \*\*Families\.\*\*(.*?)(?=^- |^\s*$|^#)", re.MULTILINE | re.DOTALL)
CATEGORY_ROW = re.compile(r"^\| Category \| (C\d) (.+?) \|\s*$", re.MULTILINE)


def sections(text: str) -> dict[str, str]:
    """The body of each `## C<n> <name>` section of the document, by id (a section ends at the next `## `)."""
    marks = [(m.group(1), m.end()) for m in SECTION.finditer(text)]
    ends = [m.start() for m in re.finditer(r"^## ", text, re.MULTILINE)]
    return {cid: text[start:min((e for e in ends if e > start), default=len(text))] for cid, start in marks}


def named_families(body: str) -> list[str]:
    """The backticked names on a section's `- **Families.**` bullet (it runs to the next bullet or a blank line)."""
    found = FAMILIES_LINE.search(body)
    return re.findall(r"`([a-z_]+)`", found.group(1)) if found else []


def test_every_registered_family_has_exactly_one_category_and_nothing_else_does():
    """Contract: `FAMILY_CATEGORY` has a key for every family the registry knows (retired ones with a spec included),
    no key for a name the registry does not know, and each value is a category of `CATEGORIES`. Regression caught: a new
    family added with no category, which a new job could not be authored from (or filed under a typo such as `C7`), and
    a family renamed or removed in the registry while the map still files the old name."""
    registered, filed = set(FAMILIES), set(FAMILY_CATEGORY)
    assert registered - filed == set(), f"families with no category: {sorted(registered - filed)}"
    assert filed - registered == set(), f"categories filed for no registered family: {sorted(filed - registered)}"
    assert set(FAMILY_CATEGORY.values()) <= set(CATEGORIES), set(FAMILY_CATEGORY.values()) - set(CATEGORIES)
    assert list(CATEGORIES) == ["C1", "C2", "C3", "C4", "C5", "C6"]


def test_the_document_names_every_category_and_each_family_it_names_is_registered_under_that_category():
    """Contract: `docs/categories.md` has a `## <id> <name>` section for every category of `CATEGORIES`; the families
    named on a section's `Families` bullet are registered, sit under that section's category in `FAMILY_CATEGORY`, and
    together are every registered family. Regression caught: the document and the code drifting apart, whether a
    category renamed in one place, a family named in the document that no longer exists, a family listed under the
    wrong category, or a registered family the document never mentions (a new job authored from the document would
    miss it)."""
    text = DOC.read_text(encoding="utf-8")
    for cid, name in CATEGORIES.items():
        assert f"## {cid} {name}\n" in text, f"{DOC.name} has no section '## {cid} {name}'"
    body = sections(text)
    assert list(body) == list(CATEGORIES)
    named: set[str] = set()
    for cid, part in body.items():
        families = named_families(part)
        assert families, f"section {cid} has no `- **Families.**` bullet naming a family"
        for family in families:
            assert family in FAMILIES, f"section {cid} names {family!r}, which is not a registered family"
            assert FAMILY_CATEGORY[family] == cid, f"section {cid} names {family!r}, which is filed under " \
                                                   f"{FAMILY_CATEGORY[family]}"
        named |= set(families)
    unnamed = set(FAMILIES) - named
    assert unnamed == set(), f"registered families the document never names: {sorted(unnamed)}"


def test_each_family_page_states_the_category_the_code_files_it_under():
    """Contract: the `Category` row of `docs/families/<family>.md` reads the id and name `FAMILY_CATEGORY` and
    `CATEGORIES` give. Regression caught: a family moved between categories in the code while its page still sends an
    author to the old one, or a new family page written without the row."""
    for family, cid in FAMILY_CATEGORY.items():
        page = FAMILY_PAGES / f"{family}.md"
        assert page.is_file(), f"{page.name} is missing"
        rows = CATEGORY_ROW.findall(page.read_text(encoding="utf-8"))
        assert rows == [(cid, CATEGORIES[cid])], f"{page.name} says {rows}, the code files {family} under {cid}"
