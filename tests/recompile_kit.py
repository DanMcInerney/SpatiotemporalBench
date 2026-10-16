"""A recompile of a dataset, as the compiler makes one: every question id drawn anew (`questions.py`'s
`secrets.token_hex(8)`), consistently in the cases' `questions.jsonl` and `gold.jsonl`. No answer, number or other field
changes, and no gold value is edited: the tests of the qid pairing (resume, `eval compare`, `eval distance`, the export,
`fill_report`) use it to make results that were scored on another compile than the dataset they meet."""

from __future__ import annotations

import json
import secrets
from pathlib import Path

PRIVATE = "private"


def recompile(data: Path, *cases: str) -> dict[tuple[str, str], str]:
    """Replace every qid of the questions and gold of CASES (default: every case folder under `data/private` that holds
    a gold) with a fresh 16-hex id; the same old qid of one case gets one new id in both files. Returns
    {(case, old qid): new qid}."""
    root = data / PRIVATE
    chosen = cases or tuple(sorted(p.name for p in root.iterdir() if (p / "gold.jsonl").is_file()))
    renamed: dict[tuple[str, str], str] = {}
    for cid in chosen:
        for name in ("questions.jsonl", "gold.jsonl"):
            path = root / cid / name
            if not path.is_file():   # a fixture that writes the gold alone
                continue
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for row in rows:
                row["qid"] = renamed.setdefault((cid, row["qid"]), secrets.token_hex(8))
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return renamed
