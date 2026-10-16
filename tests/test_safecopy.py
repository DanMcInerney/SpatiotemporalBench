"""The snapshot copy: what it takes from a state directory written by an untrusted process, and what it refuses."""

from __future__ import annotations

import errno
import os
import shutil
import stat
from pathlib import Path

import pytest

from links import dir_link, junction
from stbench.safecopy import UnsafeTree, copy_tree, remove_tree, tree_bytes

# Extended-length paths: this test makes and reads trees too deep for the default limit without using the code under
# test to do it.
PREFIX = "\\\\?\\" if os.name == "nt" else ""


def _walk(root: Path) -> tuple[set[str], dict[str, bytes]]:
    """The folders (relative, POSIX) and the file bytes under `root`."""
    top = PREFIX + str(root)
    dirs, files = set(), {}
    for here, names, leaves in os.walk(top):
        rel = Path(os.path.relpath(here, top)).as_posix()
        dirs.update(f"{'' if rel == '.' else rel + '/'}{n}" for n in names)
        for n in leaves:
            with open(os.path.join(here, n), "rb") as f:
                files[f"{'' if rel == '.' else rel + '/'}{n}"] = f.read()
    return dirs, files


def test_copy_tree_copies_folders_and_regular_files_exactly(tmp_path):
    """Contract (V0 §9, VISION rule 6): a snapshot is the same folders, the same file names and the same bytes as
    the state it copies, empty folders, non-ASCII names and paths over 300 characters included, and the return value
    is the bytes copied. Regression caught: a snapshot that drops an empty folder, truncates a deep path, mangles a
    name or alters a byte, which makes a fork answer from a different memory than the live system's."""
    deep = "/".join(["d" * 50] * 6)
    files = {"a.bin": bytes(range(256)) * 3, "sub/ünï-文件.txt": "héllo".encode(), f"{deep}/leaf.txt": b"deep",
             "sub/inner/empty.dat": b""}
    dirs = {"sub", "sub/inner", "empty", "empty/nested", "nothing-here", *(
        "/".join(["d" * 50] * i) for i in range(1, 7))}
    src, dst = tmp_path / "src", tmp_path / "dst"
    try:
        for d in dirs:
            os.makedirs(PREFIX + str(src / d), exist_ok=True)
        for name, data in files.items():
            with open(PREFIX + str(src / name), "wb") as f:
                f.write(data)
        assert len(str(src / f"{deep}/leaf.txt")) > 300

        copied = copy_tree(src, dst)

        assert _walk(dst) == (dirs, files)
        assert copied == sum(len(b) for b in files.values())
    finally:
        for folder in (src, dst):
            shutil.rmtree(PREFIX + str(folder), ignore_errors=True)  # pytest cannot always delete paths this deep


@pytest.mark.parametrize("make_link", [
    dir_link,
    pytest.param(junction, marks=pytest.mark.skipif(os.name != "nt", reason="junctions exist on Windows only")),
], ids=["link", "junction"])
def test_copy_tree_refuses_every_link_and_leaves_nothing_behind(tmp_path, make_link):
    """Contract (VISION rule 6): the copy follows nothing and takes nothing that is not a folder or a regular file.
    A link anywhere in the tree, whatever its depth, is named in one `UnsafeTree` and nothing is left at `dst`; a
    link as the source itself is refused; a tree over `max_bytes` is refused the same way; the link targets are never
    touched; a `dst` that already exists is refused and left as it was. Regression caught: `shutil.copytree`'s link
    following (a container-made relative symlink became an NTFS link on the host, so a state holding one copied the
    folder it points at, the gold included, into the snapshot the fork reads), a half-written snapshot left behind on
    refusal (a read-only file in it included), and a failed copy that removes a folder it did not make."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "gold.jsonl").write_text("secret", encoding="utf-8")
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("fine", encoding="utf-8")
    os.chmod(src / "a.txt", stat.S_IREAD)  # copied read-only, so a refused copy must still be removable
    make_link(src / "in", outside)
    make_link(src / "sub" / "in2", outside)
    expected = ["in", "sub/in2"]
    if hasattr(os, "mkfifo"):  # a special file that is not a link
        os.mkfifo(src / "pipe")
        expected = ["in", "pipe", "sub/in2"]

    with pytest.raises(UnsafeTree) as refused:
        copy_tree(src, tmp_path / "dst")
    assert refused.value.offenders == expected and refused.value.root == src
    assert all(name in str(refused.value) for name in expected)
    assert not (tmp_path / "dst").exists()
    assert (outside / "gold.jsonl").read_text(encoding="utf-8") == "secret"

    with pytest.raises(UnsafeTree):
        copy_tree(src / "in", tmp_path / "dst2")
    assert not (tmp_path / "dst2").exists()

    big = tmp_path / "big"
    big.mkdir()
    (big / "f").write_bytes(b"x" * 100)
    with pytest.raises(UnsafeTree, match="more than 10 bytes"):
        copy_tree(big, tmp_path / "dst3", max_bytes=10)
    assert not (tmp_path / "dst3").exists()
    assert copy_tree(big, tmp_path / "dst4", max_bytes=100) == 100  # at the cap is allowed
    assert (outside / "gold.jsonl").read_text(encoding="utf-8") == "secret"

    keep = tmp_path / "keep"
    keep.mkdir()
    (keep / "mine.txt").write_text("not yours to remove", encoding="utf-8")
    with pytest.raises(FileExistsError):
        copy_tree(big, keep)
    assert (keep / "mine.txt").read_text(encoding="utf-8") == "not yours to remove"


def test_copy_tree_refuses_a_tree_with_more_files_and_folders_than_allowed(tmp_path):
    """Contract (VISION rule 6): `max_entries` bounds the files and folders under `src` together, empty ones included;
    a tree over it is refused as one `UnsafeTree` and leaves nothing at `dst`, and one exactly at it is copied.
    Regression caught: a state of a million empty files, which no byte limit sees, copied at every checkpoint and again
    for every question; an off-by-one that refuses a tree at the limit."""
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    for name in ("a", "b", "c", "sub/d"):
        (src / name).write_bytes(b"")  # four files and a folder: five entries, no bytes
    with pytest.raises(UnsafeTree, match="more than 4 files and folders"):
        copy_tree(src, tmp_path / "dst", max_entries=4, max_bytes=10**9)
    assert not (tmp_path / "dst").exists()
    assert copy_tree(src, tmp_path / "dst2", max_entries=5) == 0
    assert sorted(p.relative_to(tmp_path / "dst2").as_posix() for p in (tmp_path / "dst2").rglob("*")) == [
        "a", "b", "c", "sub", "sub/d"]


@pytest.mark.parametrize("make_link", [
    dir_link,
    pytest.param(junction, marks=pytest.mark.skipif(os.name != "nt", reason="junctions exist on Windows only")),
], ids=["link", "junction"])
def test_tree_bytes_counts_regular_files_in_place_and_follows_no_link(tmp_path, make_link):
    """Contract (plan §4.4, `state_bytes`): `tree_bytes` is the sum of the regular files' sizes under a folder, at any
    depth, with nothing copied; a link or junction counts for nothing and its target is not entered, a link as the root
    itself and a root that is not there count 0 and raise nothing. Hand count: 3 + 10 + 7 = 20 bytes; the 1,000 bytes
    behind the links are not part of it. Regression caught: a size that follows a link out of the state directory (a
    system could report the dataset's bytes as its own, or hang the walk on a cycle), and a count that raises when the
    folder is gone."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "big.bin").write_bytes(b"x" * 1000)
    src = tmp_path / "src"
    (src / "sub" / "deeper").mkdir(parents=True)
    (src / "a").write_bytes(b"abc")
    (src / "sub" / "b").write_bytes(b"x" * 10)
    (src / "sub" / "deeper" / "c").write_bytes(b"y" * 7)
    make_link(src / "in", outside)
    make_link(src / "sub" / "in2", outside)
    assert tree_bytes(src) == 20
    assert tree_bytes(src / "sub") == 17
    assert tree_bytes(src / "in") == 0  # a link as the root
    assert tree_bytes(tmp_path / "missing") == 0 and tree_bytes(src / "a") == 0  # not a folder
    (tmp_path / "empty").mkdir()
    assert tree_bytes(tmp_path / "empty") == 0
    assert (outside / "big.bin").stat().st_size == 1000


def test_remove_tree_removes_what_a_plain_delete_cannot_and_follows_nothing(tmp_path):
    """Contract (VISION rule 6): `remove_tree` is the one way the harness removes a copy that a system's state made.
    It removes a read-only file, names Windows treats specially (`x.`, `y `), a path over the limit and a link (the link
    itself: what it points at is untouched), says nothing about a path that is not there, and raises no OSError even
    for what it cannot remove. Regression caught: a plain `rmtree`, which stops on those names and files and leaves the
    copy (or the run's own error hidden behind the cleanup's), and a removal that follows a link into the folder it
    points at."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "gold.jsonl").write_text("secret", encoding="utf-8")
    top = tmp_path / "copy"
    deep = top.joinpath(*["d" * 60] * 5)
    os.makedirs(PREFIX + str(deep))
    for path in (top / "ro.txt", top / "x.", top / "y ", deep / "leaf"):
        with open(PREFIX + str(path), "wb") as f:
            f.write(b"x")
    os.chmod(top / "ro.txt", stat.S_IREAD)
    dir_link(top / "peek", outside)
    try:
        remove_tree(top)
        assert not os.path.lexists(top)
        assert (outside / "gold.jsonl").read_text(encoding="utf-8") == "secret"
        remove_tree(top)  # not there: quiet
    finally:
        shutil.rmtree(PREFIX + str(top), ignore_errors=True)


def test_remove_tree_never_makes_a_link_writable_when_it_retries_its_removal(tmp_path, monkeypatch):
    """Contract (VISION rule 6): `remove_tree` makes a read-only entry writable and retries, but never a link: `chmod`
    follows a link on POSIX, so retrying a link that would not unlink would change the mode of what it points at (a
    folder of the dataset, say). The first removal of the link is made to fail here, which sends it down the
    retry path; the link is then removed and nothing has been `chmod`ed through it, and the target's mode is as it
    was. Regression caught: a retry handler that `chmod`s whatever failed to be removed."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "gold.jsonl").write_text("secret", encoding="utf-8")
    top = tmp_path / "copy"
    top.mkdir()
    dir_link(top / "peek", outside)
    real_chmod, real_unlink, real_rmdir = os.chmod, os.unlink, os.rmdir
    os.chmod(outside, stat.S_IREAD | stat.S_IEXEC)  # a mode that a chmod to S_IRWXU would visibly change
    before = stat.S_IMODE(os.stat(outside).st_mode)
    failed, chmods = [], []

    def once(real):
        def remove(path, *args, **kw):
            if os.fspath(path).endswith("peek") and not failed:
                failed.append(os.fspath(path))
                raise PermissionError(errno.EACCES, "forced", os.fspath(path))
            return real(path, *args, **kw)

        return remove

    def spy(path, *args, **kw):
        chmods.append(os.fspath(path))
        return real_chmod(path, *args, **kw)

    monkeypatch.setattr(os, "unlink", once(real_unlink))
    monkeypatch.setattr(os, "rmdir", once(real_rmdir))
    monkeypatch.setattr(os, "chmod", spy)
    try:
        remove_tree(top)
        assert failed, "the removal of the link was not made to fail, so the retry path did not run"
        assert not os.path.lexists(top)
        assert not [c for c in chmods if c.endswith("peek")]
        assert stat.S_IMODE(os.stat(outside).st_mode) == before
        assert (outside / "gold.jsonl").read_text(encoding="utf-8") == "secret"
    finally:
        real_chmod(outside, stat.S_IRWXU)
