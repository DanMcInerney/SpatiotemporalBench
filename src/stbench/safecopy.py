r"""Copying a folder that an untrusted process wrote, following nothing in it (VISION rule 6).

A system's state directory is written by a child in a container. A relative symlink it makes there becomes a real NTFS
symlink on the host (Docker Desktop), which `shutil.copytree` follows: a link to the home folder would copy the
repository and the gold into the snapshot the fork reads. This copy takes only folders and regular files and refuses
everything else: symlinks, junctions, any reparse point (WSL symlinks, fifos and sockets show up as reparse points
that look like files), devices, fifos, sockets. It uses `\\?\` paths on Windows, so a deep or oddly named entry is
neither truncated nor re-parsed. It does not stop the writer: the caller holds it still (`System.paused`).

`remove_tree` is the one owner of removal, with the same paths: a name Windows treats specially (`x.`, `y `), a path
over the limit or a read-only file in a copy cannot leave it behind. `tree_bytes` counts a folder in place with the
same rules, so a state's size needs no copy.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import stat
from pathlib import Path


class UnsafeTree(RuntimeError):
    """A tree the harness will not copy: links or special files in it, or more bytes than allowed."""

    def __init__(self, root: Path, offenders: list[str], reason: str = "holds links or special files"):
        self.root, self.offenders = Path(root), list(offenders)  # offenders: POSIX paths relative to root
        shown = ", ".join(self.offenders[:10]) + (f" and {len(self.offenders) - 10} more"
                                                   if len(self.offenders) > 10 else "")
        super().__init__(f"{root} {reason}, which the harness does not copy" + (f": {shown}" if shown else ""))


def long_path(path: str | os.PathLike) -> str:
    """`path` absolute and, on Windows, in extended-length form (`\\\\?\\`, or `\\\\?\\UNC\\` for a share)."""
    p = os.fspath(path)
    if os.name != "nt" or p.startswith("\\\\?\\"):
        return p
    p = os.path.abspath(p)
    return "\\\\?\\UNC\\" + p[2:] if p.startswith("\\\\") else "\\\\?\\" + p


def remove_tree(path: str | os.PathLike) -> None:
    """Remove the folder `path` and everything under it, best effort: a read-only file or folder is made writable and
    retried, a link or junction is unlinked and never followed (nor is it made writable: `chmod` follows a link on
    POSIX, so that would change the mode of what the link points at), a name or depth that Windows cannot address
    in the ordinary form is removed through the extended-length form (`long_path`), and a path that is not there is
    quiet. What still cannot be removed stays; no OSError is raised, so a cleanup in a `finally` never hides the error
    that brought it there."""

    def writable(func, p, _exc):
        with contextlib.suppress(OSError):
            if _kind(os.lstat(p)) is not None:  # a plain folder or file: never a link, reparse point or special file
                os.chmod(p, stat.S_IRWXU)  # on Windows this clears the read-only attribute
            func(p)

    shutil.rmtree(long_path(path), onexc=writable)


def _kind(st: os.stat_result) -> str | None:
    """"dir" or "file" for a plain folder or regular file; None for anything else, any reparse point included."""
    if getattr(st, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        return None
    if stat.S_ISDIR(st.st_mode):
        return "dir"
    return "file" if stat.S_ISREG(st.st_mode) else None


def tree_bytes(root: str | os.PathLike) -> int:
    """The bytes of the regular files under the folder `root`, counted in place: nothing is copied, and a link,
    junction, reparse point or special file counts for nothing and is never entered, so no byte outside `root` is
    counted. A `root` that is not a plain folder, and an entry that vanishes while it is counted (its writer is still
    running), count for 0 and are skipped: no OSError is raised."""
    top = long_path(root)
    with contextlib.suppress(OSError):
        if _kind(os.lstat(top)) != "dir":
            return 0
    total, pending = 0, [top]
    while pending:
        with contextlib.suppress(OSError), os.scandir(pending.pop()) as entries:
            for entry in entries:
                with contextlib.suppress(OSError):
                    st = entry.stat(follow_symlinks=False)
                    kind = _kind(st)
                    if kind == "dir":
                        pending.append(entry.path)
                    elif kind == "file":
                        total += st.st_size
    return total


def _copy(root: Path, src: str, dst: str, max_bytes: int | None, max_entries: int | None) -> int:
    offenders: list[str] = []
    total = seen = 0
    pending = [(src, dst, "")]
    while pending:
        s_dir, d_dir, rel = pending.pop()
        with os.scandir(s_dir) as entries:
            for entry in entries:
                seen += 1
                if max_entries is not None and seen > max_entries:
                    raise UnsafeTree(root, [], f"holds more than {max_entries} files and folders")
                name = rel + entry.name
                st = entry.stat(follow_symlinks=False)
                kind = _kind(st)
                if kind is None:
                    offenders.append(name)
                    continue
                target = os.path.join(d_dir, entry.name)
                if kind == "dir":
                    os.mkdir(target)
                    pending.append((entry.path, target, name + "/"))
                    continue
                total += st.st_size
                if max_bytes is not None and total > max_bytes:
                    raise UnsafeTree(root, [], f"holds more than {max_bytes} bytes")
                shutil.copy2(entry.path, target, follow_symlinks=False)
    if offenders:
        raise UnsafeTree(root, sorted(offenders))
    return total


def copy_tree(src: Path, dst: Path, *, max_bytes: int | None = None, max_entries: int | None = None) -> int:
    """Copy the folders and regular files under `src` into `dst`, which must not exist (its parent must), and return
    the bytes copied. `src` itself must be a plain folder (`UnsafeTree` naming "." if it is a link). Every offending
    entry anywhere under `src` is collected, not copied, and then raised as one `UnsafeTree`; so is a tree over
    `max_bytes` bytes or `max_entries` files and folders together (the second stops a million empty files that the
    first never sees). On any error `dst` is removed again."""
    s_root, d_root = long_path(src), long_path(dst)
    if _kind(os.lstat(s_root)) != "dir":
        raise UnsafeTree(Path(src), ["."], "is not a plain folder")
    os.mkdir(d_root)  # outside the try: a `dst` that already exists is not ours to remove
    try:
        return _copy(Path(src), s_root, d_root, max_bytes, max_entries)
    except BaseException:
        remove_tree(d_root)
        raise
