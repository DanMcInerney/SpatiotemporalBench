"""The memory tool of the reference LLM system: five function tools over its state directory, and nothing outside it.

A model reaches a folder only through `Memory.call`. Whatever it sends is data: a path is checked against one grammar
before it touches the disk (relative, `/`-separated, at most `MAX_SEGMENTS` plain names, no `..`, drive, backslash or
Windows device name), every existing component under the root is read with `lstat` and a link, junction, hard-linked
file or special file is refused, and the final real path must sit under the root. Only folders and regular files are
ever written, which is what `safecopy` handles. The check-then-open race has no writer: the folder is written by
this tool alone (the harness only measures it). Sizes are capped
(`MAX_FILE`, `MAX_TOTAL`, `MAX_FILES`), a refused write changes nothing, and a result that names a file uses the
relative path, never a host path (the model's messages leave the machine). Every refusal is a string `"error: ..."`
the model can recover from; an exception from here would be a bug.

Unused: the reference agent remembers with its own context and is offered no memory tool (DECISIONS,
2026-10-10, "The reference agent remembers with its own context"). The module and its tests stay, for an agent that
is given a memory folder; nothing in the package imports it.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .safecopy import _kind, long_path

MAX_FILE = 2**20  # bytes in one file
MAX_TOTAL = 16 * 2**20  # bytes in all files
MAX_FILES = 200  # files in the folder, any depth
MAX_SEGMENTS = 4  # names in one path
MAX_READ = 40_000  # characters one memory_read returns
DEFAULT_LINES = 400  # lines one memory_read returns unless asked for fewer
MAX_HITS = 100  # matches one memory_search returns
MAX_HIT_CHARS = 400  # characters of a matching line that a hit shows

_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")  # fullmatch only; `.`, `..` and `.x` cannot match
_DEVICES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)), *(f"LPT{i}" for i in range(10))}


def _tool(name: str, description: str, properties: dict, required: tuple[str, ...] = ()) -> dict:
    parameters = {"type": "object", "properties": properties, "required": list(required)}
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


_PATH = {"type": "string", "description": (
    "Relative path in the memory folder, such as notes.md or places/kitchen.md: up to 4 names separated by '/', each "
    "of letters, digits, '_', '-' and '.', starting with a letter or digit, at most 64 characters.")}
_TEXT = {"type": "string", "description": "The text, UTF-8."}
TOOLS: list[dict] = [
    _tool("memory_list", "List every file in the memory folder with its size in bytes.", {}),
    _tool("memory_read", "Read a text file from the memory folder. A long file is cut and the reply says where to "
          "continue.", {"path": _PATH,
                        "start_line": {"type": "integer", "minimum": 1, "description": "First line, from 1."},
                        "max_lines": {"type": "integer", "minimum": 1,
                                      "description": f"Most lines to read (default {DEFAULT_LINES})."}}, ("path",)),
    _tool("memory_write", f"Create a file or replace all of its content. A file holds at most {MAX_FILE} bytes, the "
          f"folder at most {MAX_FILES} files and {MAX_TOTAL} bytes; a refused write changes nothing.",
          {"path": _PATH, "text": _TEXT}, ("path", "text")),
    _tool("memory_append", "Add text to the end of a file, creating it if needed. A newline is added first when "
          "the file does not end with one.", {"path": _PATH, "text": _TEXT}, ("path", "text")),
    _tool("memory_search", f"Find the lines that contain a text, ignoring case, in every file. Each reply line is "
          f"path:line: text, and at most {MAX_HITS} are shown.",
          {"query": {"type": "string", "description": "The text to find."}}, ("query",)),
]
_SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}


class _Refused(Exception):
    """A request the tool will not carry out; the message goes back to the model as `error: <message>`."""


def _names(path: str) -> list[str]:
    names = path.split("/")
    if len(names) > MAX_SEGMENTS or not all(
            _SEGMENT.fullmatch(n) and not n.endswith(".") and n.split(".")[0].upper() not in _DEVICES for n in names):
        shown = path if len(path) <= 60 else path[:60] + "..."
        raise _Refused(f"invalid path {shown!r}: a path is relative, with up to {MAX_SEGMENTS} names separated by '/', "
                       "each of letters, digits, '_', '-' and '.' that starts with a letter or digit, is at most 64 "
                       "characters, does not end in '.' and is not a reserved name such as con or nul")
    return names


def _plain(st: os.stat_result) -> str | None:
    """"dir" or "file" for a plain folder or a regular file with one name; None for a link, a reparse point, a special
    file, or a file with a second hard link (a name inside the folder for a file that may live outside it)."""
    kind = _kind(st)
    return None if kind == "file" and st.st_nlink > 1 else kind


def _lines(text: str) -> list[str]:
    """The lines of `text` split on "\\n" only, each with its newline, so `"".join` restores the text."""
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def _text(full: str) -> str:
    with open(full, "rb") as f:
        data = f.read(MAX_FILE + 1)
    if len(data) > MAX_FILE:
        raise _Refused(f"larger than {MAX_FILE} bytes")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise _Refused("not UTF-8 text") from None


def _arguments(arguments: dict | str | None) -> dict:
    if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
        return {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (ValueError, RecursionError) as exc:
            raise _Refused(f"the arguments are not valid JSON ({exc})") from None
    if not isinstance(arguments, dict):
        raise _Refused("the arguments must be a JSON object")
    return arguments


def _check(name: str, args: dict) -> None:
    """Refuse an argument the tool's schema does not allow: an unknown one, a missing one, or one of the wrong type."""
    schema = _SCHEMAS[name]
    props = schema["properties"]
    for key in args:
        if key not in props:
            raise _Refused(f"unexpected argument: {key} ({name} takes {', '.join(props) or 'no arguments'})")
    for key in schema["required"]:
        if key not in args:
            raise _Refused(f"missing required argument: {key}")
    for key, value in args.items():
        if props[key]["type"] == "string" and not isinstance(value, str):
            raise _Refused(f"{key} must be a string")
        if props[key]["type"] == "integer" and not (
                isinstance(value, int) and not isinstance(value, bool) and value >= props[key]["minimum"]):
            raise _Refused(f"{key} must be an integer of at least {props[key]['minimum']}")


class Memory:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._base = long_path(self.root)
        self._handlers = {"memory_list": self._list, "memory_read": self._read, "memory_write": self._write,
                          "memory_append": self._append, "memory_search": self._search}

    def call(self, name: str, arguments: dict | str) -> str:
        """Run one tool call and return its reply: the result, or `error: <what is wrong>`. Never raises for a request
        (bad JSON, bad arguments, a refused path or limit) or for an operating-system failure on the folder."""
        handler = self._handlers.get(name)
        if handler is None:
            return f"error: unknown tool {str(name)[:60]!r}; the tools are {', '.join(self._handlers)}"
        try:
            args = _arguments(arguments)
            _check(name, args)
            self._check_root()
            return handler(**args)
        except _Refused as exc:
            return f"error: {exc}"
        except OSError as exc:  # the message of an OSError names the host path; its reason does not
            return f"error: {name} failed: {exc.strerror or type(exc).__name__}"

    def listing(self) -> str:
        """What `memory_list` returns: `(empty)`, or one `<path> <n> bytes` line per file, sorted by path."""
        return self.call("memory_list", {})

    # -- the folder ---------------------------------------------------------------------------------------------
    def _check_root(self) -> None:
        try:
            st = os.lstat(self._base)
        except FileNotFoundError:
            return  # not made yet: empty, and the first write makes it
        if _kind(st) != "dir":
            raise _Refused("the memory folder is not a plain folder")

    def _full(self, rel: str) -> str:
        return os.path.join(self._base, *rel.split("/"))

    def _find(self, path: str) -> tuple[str, str, os.stat_result | None]:
        """(path, host path, lstat of the entry or None if it is absent) for a path the model gave. Refuses a bad path
        and any link or special file on the way; the host path is for the tool's own use and never shown."""
        names, rel = _names(path), path
        full, st = self._base, None
        for i, name in enumerate(names):
            full = os.path.join(full, name)
            try:
                st = os.lstat(full)
            except FileNotFoundError:
                return rel, self._full(rel), None  # everything below is absent too
            kind = _plain(st)
            if kind is None:
                raise _Refused(f"{'/'.join(names[:i + 1])} is a link or a special file, which the memory tool "
                               "does not follow")
            if kind == "file" and i < len(names) - 1:
                raise _Refused(f"{'/'.join(names[:i + 1])} is a file, not a folder")
        if not Path(os.path.realpath(full)).is_relative_to(os.path.realpath(self._base)):
            raise _Refused(f"{rel} is outside the memory folder")
        return rel, full, st

    def _scan(self) -> list[tuple[str, str, int]]:
        """(path, "file" or "refused", bytes) for every entry under the folder, sorted by path. A link or special
        file is named and never entered, so a cycle or a link out of the folder is not walked."""
        found: list[tuple[str, str, int]] = []
        pending = [(self._base, "")]
        while pending:
            here, rel = pending.pop()
            try:
                with os.scandir(here) as entries:
                    items = [(e.name, e.path) for e in entries]
            except FileNotFoundError:
                continue  # the folder itself is not made yet
            for name, path in items:
                st = os.lstat(path)  # not DirEntry.stat(): on Windows that has no link count
                kind = _plain(st)
                if kind == "dir":
                    pending.append((path, f"{rel}{name}/"))
                else:
                    found.append((f"{rel}{name}", "file" if kind else "refused", st.st_size))
        return sorted(found)

    # -- the tools ----------------------------------------------------------------------------------------------
    def _list(self) -> str:
        rows = [f"{rel} {size} bytes" if kind == "file" else f"{rel} (refused: not a regular file)"
                for rel, kind, size in self._scan()]
        return "\n".join(rows) or "(empty)"

    def _read(self, path: str, start_line: int = 1, max_lines: int = DEFAULT_LINES) -> str:
        rel, full, st = self._find(path)
        if st is None:
            raise _Refused(f"{rel} does not exist")
        if _plain(st) == "dir":
            raise _Refused(f"{rel} is a folder; memory_list shows the files")
        try:
            lines = _lines(_text(full))
        except _Refused as exc:
            raise _Refused(f"{rel} is {exc}") from None
        if not lines:
            return "(empty file)"
        total = len(lines)
        if start_line > total:
            raise _Refused(f"start_line {start_line} is past the end: {rel} has {total} "
                           f"line{'' if total == 1 else 's'}")
        shown, size, cut = [], 0, False
        for line in lines[start_line - 1:start_line - 1 + max_lines]:
            if size + len(line) > MAX_READ:
                if not shown:  # one line over the limit: its first MAX_READ characters
                    shown, cut = [line[:MAX_READ]], True
                break
            shown.append(line)
            size += len(line)
        last = start_line - 1 + len(shown)
        out = "".join(shown)
        if not cut and last == total:
            return out
        note = f"[showing lines {start_line}-{last} of {total}"
        note += f"; line {last} was cut at {MAX_READ} characters" if cut else ""
        note += f"; call memory_read with start_line={last + 1} to continue]" if last < total else "]"
        return out + ("" if out.endswith("\n") else "\n") + note

    def _write(self, path: str, text: str) -> str:
        return self._put(path, text, append=False)

    def _append(self, path: str, text: str) -> str:
        return self._put(path, text, append=True)

    def _put(self, path: str, text: str, *, append: bool) -> str:
        rel, full, st = self._find(path)
        if st is not None and _plain(st) == "dir":
            raise _Refused(f"{rel} is a folder, not a file")
        try:
            data = text.encode("utf-8")
        except UnicodeEncodeError:
            raise _Refused("text is not valid Unicode") from None
        old = st.st_size if st is not None else 0
        newline = b""
        if append and old and data:
            with open(full, "rb") as f:
                f.seek(-1, os.SEEK_END)
                newline = b"" if f.read(1) == b"\n" else b"\n"
        added = newline + data
        new_size = old + len(added) if append else len(added)
        files = [size for _, kind, size in self._scan() if kind == "file"]
        if new_size > MAX_FILE:
            raise _Refused(f"{rel} would be {new_size} bytes; a file may hold at most {MAX_FILE} bytes")
        if st is None and len(files) >= MAX_FILES:
            raise _Refused(f"the memory folder already holds {MAX_FILES} files, the most allowed; write into an "
                           "existing file instead")
        if sum(files) - old + new_size > MAX_TOTAL:
            raise _Refused(f"the memory folder would hold {sum(files) - old + new_size} bytes; it may hold at most "
                           f"{MAX_TOTAL}")
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "ab" if append else "wb") as f:
            f.write(added)
        if not append:
            return f"wrote {len(data)} bytes to {rel}"
        return f"appended {len(data)} bytes to {rel}" + (" (a newline was added first)" if newline else "")

    def _search(self, query: str) -> str:
        if not query:
            raise _Refused("query must not be empty")
        needle, rows, hits = query.casefold(), [], 0
        for rel, kind, _size in self._scan():
            if kind != "file":
                rows.append(f"{rel} (refused: not a regular file)")
                continue
            try:
                lines = _lines(_text(self._full(rel)))
            except _Refused as exc:
                rows.append(f"{rel} (skipped: {exc})")
                continue
            for number, line in enumerate(lines, 1):
                if needle not in line.casefold():
                    continue
                if hits == MAX_HITS:
                    rows.append(f"[more than {MAX_HITS} matches; showing the first {MAX_HITS}; use a more specific "
                                "query]")
                    return "\n".join(rows)
                line = line.rstrip("\r\n")
                shown = line if len(line) <= MAX_HIT_CHARS else line[:MAX_HIT_CHARS] + "..."
                rows.append(f"{rel}:{number}: {shown}")
                hits += 1
        if not hits:
            rows.append(f"no matches for {query!r}")
        return "\n".join(rows)
