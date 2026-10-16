"""`stbench view`'s page: the reader's dict inside the one HTML file the package ships (`viewer/`).

The page is `viewer/page.html` with its style (`page.css`) and scripts (`util.js`, `log.js`, `frames.js`, `page.js`)
written into it, and the data in a `<script type="application/json">` block, so the file needs no server and no network;
only the pictures it links, the dataset's own and the depth ones drawn beside it, are other files. The data is
untrusted text (a solver's answers, a tool's results, the dataset's own words): it is written with every `<`, `>` and
`&` as a `\\u` escape, which no HTML parser reads as markup (a `</script>` or `<!--` in it cannot end the block), and
the scripts put it on the page with `textContent` only, never as HTML (`tests/test_view_page.py` guards both). Same
data, same bytes: nothing in the page is a time or a random value.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

PACKAGE_DIR = "viewer"
SCRIPTS = ("util.js", "log.js", "frames.js", "page.js")   # in the order they run; each defines what the next one uses


def _resource(name: str) -> str:
    return resources.files("stbench").joinpath(PACKAGE_DIR, name).read_text(encoding="utf-8")


def embed(doc: dict) -> str:
    """DOC as JSON for a `<script type="application/json">` block: ASCII, no `<`, `>` or `&`."""
    text = json.dumps(doc, ensure_ascii=True, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def render(doc: dict) -> str:
    """The page's HTML. The data goes in last, so nothing in it is ever read as one of the template's own marks."""
    page = _resource("page.html")
    page = page.replace("/*__CSS__*/", _resource("page.css"))
    page = page.replace("/*__JS__*/", "\n".join(_resource(name) for name in SCRIPTS))
    return page.replace("__DATA__", embed(doc))


def write(doc: dict, out: Path) -> Path:
    """The page at OUT (its folder made), replaced whole."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(doc), encoding="utf-8", newline="\n")
    return out
