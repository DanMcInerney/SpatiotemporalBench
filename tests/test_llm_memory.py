"""The agent's memory tool: what it does for a model, and what it never lets the model reach (VISION rule 6)."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from links import dir_link, junction
from stbench.llm_memory import TOOLS, Memory

LINKS = [
    dir_link,
    pytest.param(junction, marks=pytest.mark.skipif(os.name != "nt", reason="junctions exist on Windows only")),
]
LINK_IDS = ["link", "junction"]
REFUSED = " (refused: not a regular file)"
LINK_ERROR = "error: {} is a link or a special file, which the memory tool does not follow"


def _memory(tmp_path: Path) -> Memory:
    (tmp_path / "state").mkdir()
    return Memory(tmp_path / "state")


def _write(mem: Memory, path: str, text: str) -> str:
    return mem.call("memory_write", {"path": path, "text": text})


def _read(mem: Memory, path: str, **window: int) -> str:
    return mem.call("memory_read", {"path": path, **window})


def _secret(tmp_path: Path) -> Path:
    """A file outside the memory folder that the tool must never read, change or list."""
    (tmp_path / "outside").mkdir()
    secret = tmp_path / "outside" / "secret.txt"
    secret.write_bytes(b"TOPSECRET\n")
    return secret


def test_the_tools_do_what_the_model_is_told(tmp_path):
    """Contract (V0 §9, plan 3.5): write creates or replaces a file (folders on demand), append adds a newline first
    only when the file does not end with one, read gives the exact text or a numbered window, list gives one line per
    file sorted by path, search is case-insensitive and names each hit `path:line: text`, and the bytes on disk have
    no newline translation. Regression caught: a tool the prompt advertises that does something else (an append that
    glues two notes into one line, a read that shifts line numbers), or Windows turning `\\n` into `\\r\\n`."""
    mem = _memory(tmp_path)
    state = tmp_path / "state"
    assert mem.listing() == "(empty)" == mem.call("memory_list", {})

    assert _write(mem, "notes.md", "alpha\nBeta\n") == "wrote 11 bytes to notes.md"
    assert (state / "notes.md").read_bytes() == b"alpha\nBeta\n"
    # the file ends with a newline, so nothing is added; then it ends with "gamma", so a newline goes first
    assert mem.call("memory_append", {"path": "notes.md", "text": "gamma"}) == "appended 5 bytes to notes.md"
    assert mem.call("memory_append", {"path": "notes.md", "text": "delta\n"}) == (
        "appended 6 bytes to notes.md (a newline was added first)")
    assert _read(mem, "notes.md") == "alpha\nBeta\ngamma\ndelta\n"
    assert (state / "notes.md").read_bytes() == b"alpha\nBeta\ngamma\ndelta\n"  # 23 bytes
    assert _read(mem, "notes.md", start_line=2, max_lines=2) == (
        "Beta\ngamma\n[showing lines 2-3 of 4; call memory_read with start_line=4 to continue]")
    assert _read(mem, "notes.md", start_line=4) == "delta\n"

    assert _write(mem, "places/kitchen.md", "The Alpha cup\nnothing\nalpha again\n") == (
        "wrote 34 bytes to places/kitchen.md")
    assert mem.listing() == "notes.md 23 bytes\nplaces/kitchen.md 34 bytes"
    assert mem.call("memory_search", {"query": "ALPHA"}) == (
        "notes.md:1: alpha\nplaces/kitchen.md:1: The Alpha cup\nplaces/kitchen.md:3: alpha again")
    assert mem.call("memory_search", {"query": "zzz"}) == "no matches for 'zzz'"

    assert _write(mem, "notes.md", "z") == "wrote 1 bytes to notes.md"  # replaces the whole file
    assert _read(mem, "notes.md") == "z"
    assert mem.call("memory_append", {"path": "fresh.md", "text": "one"}) == "appended 3 bytes to fresh.md"


def test_a_read_is_cut_where_it_says_and_the_reply_says_how_to_go_on(tmp_path):
    """Contract (plan 3.5): a read returns at most 400 lines unless asked for fewer and at most 40,000 characters,
    cuts at a whole line, and ends with a note naming the line to continue from; a first line over the limit is cut
    and says so; an empty file, a missing file, a folder and a start past the end each say what is wrong. Regression
    caught: a long note file flooding the context, or a cut that does not say where it cut, so the model believes it
    read everything."""
    mem = _memory(tmp_path)
    _write(mem, "many.md", "n\n" * 450)
    assert _read(mem, "many.md") == "n\n" * 400 + (
        "[showing lines 1-400 of 450; call memory_read with start_line=401 to continue]")
    assert _read(mem, "many.md", start_line=401) == "n\n" * 50

    _write(mem, "wide.md", ("y" * 29_999 + "\n") * 5)  # 30,000 characters a line: one fits, two do not
    assert _read(mem, "wide.md", max_lines=5) == ("y" * 29_999 + "\n") + (
        "[showing lines 1-1 of 5; call memory_read with start_line=2 to continue]")

    _write(mem, "long.md", "z" * 50_000)
    assert _read(mem, "long.md") == "z" * 40_000 + "\n[showing lines 1-1 of 1; line 1 was cut at 40000 characters]"

    _write(mem, "bare.md", "a\nb")  # no newline at the end: the note starts on a line of its own
    assert _read(mem, "bare.md", max_lines=1) == (
        "a\n[showing lines 1-1 of 2; call memory_read with start_line=2 to continue]")
    assert _read(mem, "bare.md", start_line=2) == "b"

    _write(mem, "empty.md", "")
    assert _read(mem, "empty.md") == "(empty file)"
    assert _read(mem, "bare.md", start_line=3) == "error: start_line 3 is past the end: bare.md has 2 lines"
    _write(mem, "one.md", "x")
    assert _read(mem, "one.md", start_line=2) == "error: start_line 2 is past the end: one.md has 1 line"
    assert _read(mem, "nope.md") == "error: nope.md does not exist"
    _write(mem, "places/a.md", "x")
    assert _read(mem, "places") == "error: places is a folder; memory_list shows the files"
    assert mem.call("memory_write", {"path": "places", "text": "x"}) == "error: places is a folder, not a file"


def test_a_search_stops_at_100_hits_and_shows_a_line_in_part(tmp_path):
    """Contract (plan 3.5): a search shows at most 100 hits and says when there are more, and a hit shows at most 400
    characters of its line. Regression caught: one search over a big notes file flooding the context with thousands
    of lines, or one very long line doing it."""
    mem = _memory(tmp_path)
    _write(mem, "f.md", "".join(f"hit {i}\n" for i in range(1, 151)))
    expected = [f"f.md:{i}: hit {i}" for i in range(1, 101)]
    expected.append("[more than 100 matches; showing the first 100; use a more specific query]")
    assert mem.call("memory_search", {"query": "HIT"}) == "\n".join(expected)

    _write(mem, "g.md", "needle" + "x" * 1000 + "\n")
    assert mem.call("memory_search", {"query": "needle"}) == f"g.md:1: needle{'x' * 394}..."


def test_files_the_tool_cannot_read_are_named_not_skipped(tmp_path):
    """Contract (plan 3.5): listing and searching skip nothing silently, and reading says why it cannot. A file
    that is not UTF-8 text or is over the size limit (the tool cannot make one; a damaged state could hold one) is
    reported by its path and the reason. Regression caught: a search that quietly passes over a file, so the model
    concludes a note it cannot read does not exist."""
    mem = _memory(tmp_path)
    state = tmp_path / "state"
    (state / "bad.md").write_bytes(b"needle \xff\xfe\n")
    (state / "big.md").write_bytes(b"a" * (2**20 + 1))
    _write(mem, "ok.md", "a needle\n")
    assert mem.call("memory_search", {"query": "needle"}) == (
        "bad.md (skipped: not UTF-8 text)\nbig.md (skipped: larger than 1048576 bytes)\nok.md:1: a needle")
    assert _read(mem, "bad.md") == "error: bad.md is not UTF-8 text"
    assert _read(mem, "big.md") == "error: big.md is larger than 1048576 bytes"
    assert mem.listing() == "bad.md 10 bytes\nbig.md 1048577 bytes\nok.md 9 bytes"


# Paths a model may send that the tool must refuse before touching the disk (the escapes first).
def _bad_paths(tmp_path: Path) -> list[str]:
    secret = tmp_path / "outside" / "secret.txt"
    return ["../x", "../outside/secret.txt", "a/../../outside/secret.txt", "..", ".", "/etc/x", "/", "C:/x", "C:x",
            str(secret), secret.as_posix(), "\\\\?\\" + str(secret), "\\\\server\\share\\x", "\\\\?\\C:\\x",
            "a\\b", "a//b", "a/", " a", "a b", "a:b", "~1", "é.md", "a\x00b", "a\nb", "", ".hidden", "x.", "a./b",
            "a" * 65, "a/b/c/d/e", "con.txt", "CON", "Nul.tar.gz", "aux", "lpt9.x", "Com1", "com0"]


def test_an_escaping_or_malformed_path_is_refused_before_it_touches_the_disk(tmp_path):
    """Contract (VISION rule 6, plan 3.5): the model reads and writes only inside the memory folder. `..`, absolute
    paths (drive, UNC, `\\\\?\\`, POSIX), backslashes, empty or hidden or dotted-end names, Windows device names, a
    name over 64 characters and a fifth segment each give `error: invalid path`, from every tool that takes a path,
    and nothing under the test folder (the memory folder and a sibling holding a secret) changes. Regression caught:
    a model writing or reading outside its state directory, which on the harness host could reach the dataset and
    the gold."""
    mem = _memory(tmp_path)
    secret = _secret(tmp_path)
    _write(mem, "keep.md", "kept\n")

    def tree():
        return sorted((p.relative_to(tmp_path).as_posix(), p.stat().st_size if p.is_file() else -1)
                      for p in tmp_path.rglob("*"))

    before = tree()
    for path in _bad_paths(tmp_path):
        for tool in ("memory_read", "memory_write", "memory_append"):
            reply = mem.call(tool, {"path": path, "text": "x"} if tool != "memory_read" else {"path": path})
            assert reply.startswith("error: invalid path "), (tool, path, reply)
            assert "TOPSECRET" not in reply, (tool, path)
    assert tree() == before
    assert secret.read_bytes() == b"TOPSECRET\n"
    assert mem.listing() == "keep.md 5 bytes"


def test_well_formed_names_up_to_the_limits_are_accepted(tmp_path):
    """Contract (plan 3.5): the path grammar refuses only what it names, so a model can use ordinary note names:
    four folders deep, 64-character names, dots and dashes inside a name, names that merely start like a device
    (`com10`, `console`). Regression caught: a grammar so tight that notes cannot be written, which would make the
    reference system forget everything."""
    mem = _memory(tmp_path)
    for path in ["a1", "a.md", "9lives", "x..y", "a.b-c_d.md", "a" * 64, "d1/b/c/d", "com10.md", "console.md",
                 "nulx", "com", "Z"]:
        assert _write(mem, path, "x") == f"wrote 1 bytes to {path}", path
    assert (tmp_path / "state" / "d1" / "b" / "c" / "d").read_text(encoding="utf-8") == "x"
    assert _write(mem, "a1/b", "x") == "error: a1 is a file, not a folder"


@pytest.mark.parametrize("make_link", LINKS, ids=LINK_IDS)
def test_a_link_inside_the_folder_is_never_followed(tmp_path, make_link):
    """Contract (VISION rule 6, plan 3.5): a symlink or junction inside the memory folder (a system's own, or one
    that a container made) is listed as refused and never read, written, searched or entered, including a link to
    the folder itself (no endless walk). The folder it points at keeps its listing and contents. Regression caught:
    a write through `state/out -> elsewhere` landing outside the state directory, or a read of the gold through it."""
    secret = _secret(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (state / "real.md").write_bytes(b"fine\n")
    make_link(state / "out", secret.parent)
    make_link(state / "loop", state)
    mem = Memory(state)

    assert mem.listing() == f"loop{REFUSED}\nout{REFUSED}\nreal.md 5 bytes"
    assert mem.call("memory_search", {"query": "fine"}) == f"loop{REFUSED}\nout{REFUSED}\nreal.md:1: fine"
    assert mem.call("memory_search", {"query": "topsecret"}) == (
        f"loop{REFUSED}\nout{REFUSED}\nno matches for 'topsecret'")
    assert _read(mem, "out/secret.txt") == LINK_ERROR.format("out")
    assert _read(mem, "loop/real.md") == LINK_ERROR.format("loop")
    assert _write(mem, "out/new.md", "x") == LINK_ERROR.format("out")
    assert _write(mem, "out", "x") == LINK_ERROR.format("out")
    assert mem.call("memory_append", {"path": "out/secret.txt", "text": "x"}) == LINK_ERROR.format("out")
    assert mem.call("memory_append", {"path": "loop/new.md", "text": "x"}) == LINK_ERROR.format("loop")

    assert os.listdir(secret.parent) == ["secret.txt"]
    assert secret.read_bytes() == b"TOPSECRET\n"
    assert sorted(os.listdir(state)) == ["loop", "out", "real.md"]


@pytest.mark.parametrize("kind", ["symlink", "hard link"])
def test_a_linked_file_is_never_read_or_written(tmp_path, kind):
    """Contract (VISION rule 6, plan 3.5): a file in the memory folder that is a symlink, or a second name for a file
    that may live elsewhere (a hard link), is refused by every tool and listed as refused. The file it names is left
    alone. Regression caught: `memory_write` truncating a file outside the state directory through a link inside it."""
    secret = _secret(tmp_path)
    mem = _memory(tmp_path)
    try:
        if kind == "symlink":
            os.symlink(secret, tmp_path / "state" / "f.md")
        else:
            os.link(secret, tmp_path / "state" / "f.md")
    except OSError:
        pytest.skip(f"this user or filesystem cannot make a {kind}")

    assert mem.listing() == f"f.md{REFUSED}"
    assert _read(mem, "f.md") == LINK_ERROR.format("f.md")
    assert _write(mem, "f.md", "x") == LINK_ERROR.format("f.md")
    assert mem.call("memory_append", {"path": "f.md", "text": "x"}) == LINK_ERROR.format("f.md")
    assert mem.call("memory_search", {"query": "topsecret"}) == f"f.md{REFUSED}\nno matches for 'topsecret'"
    assert secret.read_bytes() == b"TOPSECRET\n"


@pytest.mark.parametrize("make_link", LINKS, ids=LINK_IDS)
def test_a_memory_folder_that_is_itself_a_link_is_refused(tmp_path, make_link):
    """Contract (plan 3.5, `safecopy.copy_tree`): the root, like everything under it, must be a plain folder.
    Regression caught: a state directory replaced by a link, so every path under it resolves somewhere else."""
    secret = _secret(tmp_path)
    make_link(tmp_path / "state", secret.parent)
    mem = Memory(tmp_path / "state")
    for reply in (mem.listing(), _read(mem, "secret.txt"), _write(mem, "n.md", "x"),
                  mem.call("memory_search", {"query": "x"})):
        assert reply == "error: the memory folder is not a plain folder"
    assert os.listdir(secret.parent) == ["secret.txt"]


def test_a_folder_that_is_not_made_yet_is_empty_and_the_first_write_makes_it(tmp_path):
    """Contract (plan 3.3): a fresh state directory may not exist yet; reading it finds nothing and the first write
    creates it. Regression caught: a first run that fails on a missing folder instead of starting empty."""
    mem = Memory(tmp_path / "state")
    assert mem.listing() == "(empty)"
    assert _read(mem, "x.md") == "error: x.md does not exist"
    assert mem.call("memory_search", {"query": "x"}) == "no matches for 'x'"
    assert _write(mem, "x.md", "hi") == "wrote 2 bytes to x.md"
    assert (tmp_path / "state" / "x.md").read_text(encoding="utf-8") == "hi"


def test_size_limits_refuse_a_write_and_change_nothing(tmp_path):
    """Contract (plan 3.5): a file holds at most 1,048,576 bytes, the folder at most 200 files and 16,777,216 bytes,
    and a refused write or append leaves every file and folder as it was, while a write that stays within the
    limits (even into a full folder) still works. Regression caught: unbounded state growth (the folder's size is a
    reported resource of every case), or a refused write that still truncated the old file."""
    mem = _memory(tmp_path)
    state = tmp_path / "state"
    too_big = "error: big.md would be 1048577 bytes; a file may hold at most 1048576 bytes"
    assert _write(mem, "big.md", "a" * 2**20) == "wrote 1048576 bytes to big.md"
    assert _write(mem, "big.md", "b" * (2**20 + 1)) == too_big
    assert mem.call("memory_append", {"path": "big.md", "text": "c"}) == (  # a newline goes first: 2**20 + 2
        "error: big.md would be 1048578 bytes; a file may hold at most 1048576 bytes")
    assert (state / "big.md").read_bytes() == b"a" * 2**20
    assert _write(mem, "deep/er/new.md", "d" * (2**20 + 1)) == (
        "error: deep/er/new.md would be 1048577 bytes; a file may hold at most 1048576 bytes")
    assert sorted(os.listdir(state)) == ["big.md"]  # not even the folders were made
    assert _write(mem, "big.md", "e") == "wrote 1 bytes to big.md"

    full = tmp_path / "full" / "state"  # 200 files across 4 folders, made directly
    for i in range(200):
        (full / f"d{i % 4}").mkdir(parents=True, exist_ok=True)
        (full / f"d{i % 4}" / f"f{i}.md").write_bytes(b"")
    crowded = Memory(full)
    assert _write(crowded, "x.md", "x") == (
        "error: the memory folder already holds 200 files, the most allowed; write into an existing file instead")
    assert _write(crowded, "n/x.md", "x").startswith("error: the memory folder already holds 200 files")
    assert sorted(os.listdir(full)) == ["d0", "d1", "d2", "d3"]
    assert _write(crowded, "d0/f0.md", "x") == "wrote 1 bytes to d0/f0.md"
    assert crowded.call("memory_append", {"path": "d1/f1.md", "text": "y"}) == "appended 1 bytes to d1/f1.md"

    heavy = tmp_path / "heavy" / "state"  # 15 MiB in 15 files, made directly
    heavy.mkdir(parents=True)
    for i in range(15):
        (heavy / f"p{i}.bin").write_bytes(bytes(2**20))
    loaded = Memory(heavy)
    assert _write(loaded, "q.md", "q" * 2**20) == "wrote 1048576 bytes to q.md"  # 16 MiB exactly: allowed
    assert _write(loaded, "r.md", "r") == (
        "error: the memory folder would hold 16777217 bytes; it may hold at most 16777216")
    assert not (heavy / "r.md").exists()
    assert _write(loaded, "q.md", "s") == "wrote 1 bytes to q.md"  # replacing frees the old bytes
    assert _write(loaded, "r.md", "r") == "wrote 1 bytes to r.md"


def test_bad_arguments_are_replies_the_model_can_fix(tmp_path):
    """Contract (plan 3.5): arguments arrive as an object or as a JSON string (what a model sends); an empty string
    means none; malformed JSON, a non-object, an unknown, missing or mistyped argument, an unknown tool and text that
    cannot be stored each give `error: ...` naming the problem, never an exception, and change nothing. Regression
    caught: a model's typo (`start` for `start_line`) silently ignored, or a bad call stopping a paid run."""
    mem = _memory(tmp_path)
    assert mem.call("memory_write", '{"path": "a.md", "text": "hi"}') == "wrote 2 bytes to a.md"
    assert mem.call("memory_list", "") == mem.call("memory_list", "{}") == "a.md 2 bytes"
    assert mem.call("memory_read", '{"path": "a.md"}') == "hi"

    assert mem.call("memory_write", "{nope").startswith("error: the arguments are not valid JSON")
    assert mem.call("memory_write", "[1]") == "error: the arguments must be a JSON object"
    assert mem.call("memory_write", {}) == "error: missing required argument: path"
    assert mem.call("memory_write", {"path": "a.md"}) == "error: missing required argument: text"
    assert mem.call("memory_read", {"path": "a.md", "start": 2}) == (
        "error: unexpected argument: start (memory_read takes path, start_line, max_lines)")
    assert mem.call("memory_list", {"path": "a.md"}) == (
        "error: unexpected argument: path (memory_list takes no arguments)")
    assert mem.call("memory_read", {"path": 5}) == "error: path must be a string"
    assert mem.call("memory_write", {"path": "a.md", "text": 5}) == "error: text must be a string"
    for bad in (0, -1, True, "2", 1.0, None):
        assert mem.call("memory_read", {"path": "a.md", "start_line": bad}) == (
            "error: start_line must be an integer of at least 1"), bad
    assert mem.call("memory_search", {"query": ""}) == "error: query must not be empty"
    assert mem.call("memory_delete", {"path": "a.md"}) == (
        "error: unknown tool 'memory_delete'; the tools are memory_list, memory_read, memory_write, memory_append, "
        "memory_search")

    assert _write(mem, "b.md", "\ud800") == "error: text is not valid Unicode"  # a lone surrogate has no UTF-8
    assert mem.listing() == "a.md 2 bytes"


def test_an_operating_system_error_never_shows_the_host_path(tmp_path):
    """Contract (plan 3.5): a failure of the file system (here a read-only file) comes back as `error: <tool> failed:
    <reason>` without the host path, because every reply goes to the model's provider and the path holds the user's
    name and the dataset layout. Regression caught: `str(OSError)` in a reply, which includes the full path."""
    mem = _memory(tmp_path)
    _write(mem, "ro.md", "x")
    target = tmp_path / "state" / "ro.md"
    os.chmod(target, stat.S_IREAD)
    try:
        reply = _write(mem, "ro.md", "y")
        if target.read_bytes() == b"y":
            pytest.skip("this user can write a read-only file")
    finally:
        os.chmod(target, stat.S_IREAD | stat.S_IWRITE)
    assert reply.startswith("error: memory_write failed: ")
    assert str(tmp_path) not in reply and "state" not in reply and "ro.md" not in reply


def test_the_schemas_and_the_executor_agree(tmp_path):
    """Contract (plan 3.5): every tool in `TOOLS` is exactly the five the prompt describes, is plain JSON, and is
    handled by `Memory.call` with every argument its schema lists (no 'unexpected argument' or 'unknown tool' for a
    tool the model was offered); a tool that is not offered gives `error:`. Regression caught: an advertised tool
    or argument the executor does not know, so the model is invited to do what always fails."""
    names = [t["function"]["name"] for t in TOOLS]
    assert names == ["memory_list", "memory_read", "memory_write", "memory_append", "memory_search"]
    assert [t["function"]["parameters"]["required"] for t in TOOLS] == [[], ["path"], ["path", "text"],
                                                                         ["path", "text"], ["query"]]
    assert json.loads(json.dumps(TOOLS)) == TOOLS and all(t["type"] == "function" for t in TOOLS)

    sample = {"path": "a.md", "text": "t", "query": "t", "start_line": 1, "max_lines": 1}
    mem = _memory(tmp_path)
    for tool in TOOLS:
        props = tool["function"]["parameters"]["properties"]
        assert set(props) <= set(sample), tool["function"]["name"]  # a new argument needs a sample here
        reply = mem.call(tool["function"]["name"], {k: v for k, v in sample.items() if k in props})
        assert not reply.startswith(("error: unknown tool", "error: unexpected argument", "error: missing",
                                     "error: the arguments")), (tool["function"]["name"], reply)
        assert tool["function"]["description"]
    assert (tmp_path / "state" / "a.md").read_text(encoding="utf-8") == "t\nt"  # the write, then an append
    assert mem.call("memory_nope", {}).startswith("error: unknown tool")
