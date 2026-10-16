"""The leak scan of a Codex run's log: what the harness put in front of the model, checked after the run (V0 §9).

The scan reads `codex_log.transcripts` of the journal: the conversation the model held at each ask (the first prompt,
then every item served to it as a tool message, in order, and the replies to its other calls) and the whole case's, in
which a message repeated by a later transcript is scanned once. Two kinds of check, by what a hit means:

- the leak scan's markers (`leak.RULES`: an entity id, a region id, the word truth) and the map's name, in every message
  the harness made (system, user and tool: an item it served and the reply to a call are the harness's words). They are
  low-precision words, so the model's own messages are left out: the model may say anything;
- identifiers and the future, in the same messages (the model's own words can only repeat what these showed it).
  Nothing but a leak puts a question id, the case id or the dataset's path in front of a model. A question's text may
  appear only from its ask on: a message that holds the text of a question no ask has yet put (the questions asked so
  far may appear later in the thread). And an ask's transcript may hold no frame whose header number `k` is above the
  ask's `now_k`. The ask's own questions count as asked only in the transcript's last message, the ask's item (an ask
  alone, or a frame item that carries it: the frame's text, its image, then the questions, one message,
  `codex_log.transcripts`); every message before it is read against the asks before it, so a question's text shown in
  an earlier frame is a hit even when its ask comes next. A header in the ask's item above `now_k` (the frame's own `k`
  is `now_k`) is a hit like any other. What was served after the last ask, up to the end of the case, is in the case's
  transcript (the last one) and is read against all the asks: it may show a question that was asked, never one that
  was not.

A message of a pictures run is a list of blocks; its text blocks are scanned, joined by newlines so that each frame's
header line starts a line, and its images (which the log holds as `<jpeg N bytes sha256 HEX16>`, or `<png ...>`) are
not: a picture is not text, and what it shows is the measurement.

`scan_log` returns one line per hit; an empty list is a clean log. The questions are the harness's own: `qid` and `text`
are read, nothing else. A question the run asked is known by the `ask` field of an ask's transcript (`labels`, the qid
of each label, and `now_k`).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

from . import leak
from .codex_log import read_log, transcripts

TOKEN_BEFORE, TOKEN_AFTER = r"(?<![A-Za-z0-9])", r"(?![A-Za-z0-9])"
HEADER = re.compile(r"^\[k(\d+) ", re.MULTILINE)  # a frame's header line (`llm_render.header`)
SHOWN = ("system", "user", "tool")  # the roles the harness writes in: what can show a model something it must not see


def _text(content) -> str:
    """A message's text: a string, or the text blocks of a list joined by newlines (so that the header line of a frame
    starts a line)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str)]
        if parts:
            return "\n".join(parts)
    return json.dumps(content)


def _paths(dataset: Path | str | None) -> list[str]:
    """The ways the dataset's absolute folder is written, lower-cased: with the system's separators and with slashes.
    A relative path is not scanned for as it was given (`data` would match prose)."""
    if dataset is None:
        return []
    folder = Path(dataset).resolve()
    return sorted({str(folder).lower(), folder.as_posix().lower()})


def _unasked(questions: Sequence[dict], asked: Sequence[str]) -> tuple[list[dict], list[str]]:
    """The questions with a text whose ask is not among `asked`, and the texts of those that are, longest first."""
    return ([q for q in questions if q["qid"] not in asked and q["text"]],
            sorted((q["text"] for q in questions if q["qid"] in asked and q["text"]), key=len, reverse=True))


def scan_log(log: Path, map_names: Sequence[str] = (), *, questions: Sequence[dict] = (), episode: str = "",
             dataset: Path | str | None = None) -> list[str]:
    """One line per hit, `call <i>, message <n> (<role>): <what>`, over the transcripts of a journal. `map_names`,
    `questions` (dicts with `qid` and `text`), `episode` (the case's id) and `dataset` (its folder) switch on their
    checks; the markers of `leak.RULES` are always scanned."""
    return scan_records(transcripts(read_log(log)), map_names, questions=questions, episode=episode, dataset=dataset)


def scan_records(records: Sequence[dict], map_names: Sequence[str] = (), *, questions: Sequence[dict] = (),
                 episode: str = "", dataset: Path | str | None = None) -> list[str]:
    """`scan_log` over transcripts already built: dicts with `i`, `purpose`, `ask` (an ask's transcript: `labels`,
    `now_k`) and `request.messages`, in the order they were made."""
    named = {q["qid"]: "qid" for q in questions} | ({episode: "episode_id"} if episode else {})
    named_rx = re.compile(TOKEN_BEFORE + "(" + "|".join(map(re.escape, named)) + ")" + TOKEN_AFTER) if named else None
    folders = _paths(dataset)
    seen_marker: set[tuple[str, str]] = set()  # (role, text) already scanned for the markers and the map's name
    seen_shown: set[tuple[str, str]] = set()  # ... for the identifiers and the dataset's path
    seen_future: set[tuple[str, str]] = set()  # ... for the text of a question not asked yet
    frames: dict[str, list[int]] = {}  # message text -> its frame headers' numbers
    asked: list[str] = []  # the qids of the asks before this transcript's
    hits = []
    for record in records:
        ask = record.get("ask")
        now = [qid for qid in ask["labels"].values() if qid not in asked] if ask else []
        messages = record["request"]["messages"]
        before = _unasked(questions, asked)  # a message before the ask's item is read against the asks before it
        item = _unasked(questions, asked + now)  # the ask's item, the last message, may show this ask's questions
        for at, message in enumerate(messages):
            role, text = message.get("role"), _text(message.get("content"))
            where = f"call {record['i']}, message {at} ({role})"
            if role in SHOWN and (role, text) not in seen_marker:
                seen_marker.add((role, text))
                hits += [f"{where}: {rule} {m.group(0)!r}" for rule, rx in leak.RULES.items()
                         for m in rx.finditer(text)]
                hits += [f"{where}: map_name {name!r}" for name in map_names if name and name.lower() in text.lower()]
            if role not in SHOWN:
                continue
            if (role, text) not in seen_shown:
                seen_shown.add((role, text))
                if named_rx:
                    hits += [f"{where}: {named[m.group(0)]} {m.group(0)!r}" for m in named_rx.finditer(text)]
                hits += [f"{where}: dataset path {folder!r}" for folder in folders if folder in text.lower()]
            unasked, known = item if at == len(messages) - 1 else before
            if unasked and (role, text) not in seen_future:  # first shown here: no question not yet asked is in it
                seen_future.add((role, text))
                rest = text
                for known_text in known:  # a question asked already may be shown, and its text may hold another's
                    rest = rest.replace(known_text, "")
                hits += [f"{where}: the text of question {q['qid']} before its ask" for q in unasked
                         if q["text"] in rest]
            if ask:  # a frame after this ask's step
                if text not in frames:
                    frames[text] = [int(k) for k in HEADER.findall(text)]
                hits += [f"{where}: frame k{k} after the ask's step k{ask['now_k']}"
                         for k in sorted({k for k in frames[text] if k > ask["now_k"]})]
        asked += now
    return hits
