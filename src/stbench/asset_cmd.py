"""`stbench assets`: look at, download and credit the photoreal assets.

    stbench assets status [--include-alternates]       sources, licences, sizes and what is cached (downloads nothing)
    stbench assets fetch [--only IDS] --yes            download the files that are missing, and unpack the zips
    stbench assets credits [--out DATASET/CREDITS.md]  the attribution file (refused in scripts/, episodes/, private/)

`fetch` prints what it would download, from where and under which licence, and without `--yes` does nothing else:
downloads need `--yes`. Other modules add their own subcommands to the group through `group`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path, PurePath

from . import asset_fetch as F
from .asset_manifest import AssetError, Manifest, cache_dir, load, packaged_path, validate


def group(sub):
    """The `stbench assets` subcommand group, created on first use; `asset_render` adds `library` and `preview`."""
    parser = sub.choices.get("assets")
    if parser is None:
        parser = sub.add_parser("assets", help="photoreal assets: status, fetch, credits")
        parser.assets_commands = parser.add_subparsers(dest="assets_command", required=True)
    return parser.assets_commands


def _guarded(handler: Callable[[argparse.Namespace], int]) -> Callable[[argparse.Namespace], int]:
    def run(args: argparse.Namespace) -> int:
        try:
            return handler(args)
        except AssetError as e:
            print(f"assets: {e}", file=sys.stderr)
            return 2

    return run


def _manifest(args: argparse.Namespace) -> Manifest:
    m = load(args.manifest)
    if problems := validate(m):
        raise AssetError("the manifest has problems:\n  " + "\n  ".join(problems))
    return m


def _mb(size: int) -> str:
    return f"~{size / 1e6:.1f}"


def _table(rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in rows)


def _sources(m: Manifest) -> list[str]:
    return [f"  {k}: {s.name}, {s.licence}, {s.licence_url or 'our own code'}, hosts {', '.join(s.hosts) or 'none'}"
            for k, s in m.sources.items()]


def _plan_text(m: Manifest, tasks: list[F.FileTask]) -> str:
    """What a download would do: per asset, then the sources, the hosts and the total."""
    rows = [["asset", "source", "licence", "host", "files", "MB"]]
    for asset_id in dict.fromkeys(t.asset.id for t in tasks):
        mine = [t for t in tasks if t.asset.id == asset_id]
        first = mine[0]
        rows.append([asset_id, first.asset.source, first.asset.licence, first.host, str(len(mine)),
                     _mb(F.plan_bytes(mine))])
    hosts = sorted({t.host for t in tasks})
    return "\n".join([_table(rows), "sources:", *_sources(m),
                      f"total: {len(tasks)} files, {_mb(F.plan_bytes(tasks))} MB from {', '.join(hosts)}"])


def _status(args: argparse.Namespace) -> int:
    m, cache = _manifest(args), cache_dir()
    print(f"manifest {args.manifest or packaged_path()}\ncache {cache}\nsources:")
    print("\n".join(_sources(m)))
    rows = [["asset", "for", "source", "licence", "cached", "MB"]]
    for a in m.assets:
        use = (a.kind or ", ".join(a.uses)) + (f" (alternate for {a.alternate_for})" if a.alternate_for else "")
        done = sum(F.file_path(a, f, cache).is_file() for f in a.files)
        rows.append([a.id, use, a.source, a.licence, f"{done}/{len(a.files)}" if a.files else "n/a",
                     _mb(a.download_size())])
    print(_table(rows))
    tasks = F.plan(m, include_alternates=args.include_alternates, cache=cache)
    print(f"to download: {len(tasks)} files, {_mb(F.plan_bytes(tasks))} MB")
    alternates = [a for a in m.assets if a.alternate_for]
    if alternates and not args.include_alternates:
        extra = sum(a.download_size() for a in alternates)
        print(f"not included: {len(alternates)} alternates, {_mb(extra)} MB (--include-alternates)")
    return 0


def _fetch(args: argparse.Namespace) -> int:
    m = _manifest(args)
    tasks = F.plan(m, args.only, args.include_alternates, cache_dir())
    if not tasks:
        print("nothing to fetch")
        return 0
    print(_plan_text(m, tasks))
    if not args.yes:
        print("nothing was downloaded; add --yes to download")
        return 0
    print(f"downloaded {F.fetch(tasks)} files into {cache_dir()}")
    return 0


def _through_dataset_folder(out: PurePath, platform: str = sys.platform) -> bool:
    """Whether the resolved `out` has a folder named scripts, episodes or private in it. macOS keeps /tmp and /var under
    /private, the system's own folder, so a leading /private there is not a dataset's."""
    parts = out.parts[1:]
    if platform == "darwin" and parts[:1] == ("private",):
        parts = parts[1:]
    return any(part.lower() in ("scripts", "episodes", "private") for part in parts)


def _credits(args: argparse.Namespace) -> int:
    text = F.credits(_manifest(args))
    if args.out is None:
        print(text, end="")
        return 0
    try:
        out = Path(args.out).resolve()
        if _through_dataset_folder(out):
            raise AssetError(f"{out} has a folder named scripts, episodes or private in it; CREDITS.md goes at the "
                             "dataset root, where no system can read it")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8", newline="\n")
    except OSError as e:
        raise AssetError(f"cannot write the credits to {args.out}: {e}") from e
    print(f"credits -> {out}")
    return 0


def register(sub) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", type=Path, help="a manifest file (default: the packaged one)")
    pick = argparse.ArgumentParser(add_help=False)
    pick.add_argument("--only", type=lambda s: [x for x in s.split(",") if x], help="asset ids, comma separated")
    pick.add_argument("--include-alternates", action="store_true", help="also the alternates (fallbacks)")
    commands = group(sub)
    s = commands.add_parser("status", parents=[common], help="sources, licences, sizes and cache state")
    s.add_argument("--include-alternates", action="store_true")
    s.set_defaults(func=_guarded(_status))
    f = commands.add_parser("fetch", parents=[common, pick], help="download the files that are missing")
    f.add_argument("--yes", action="store_true", help="really download (without it: print the plan and stop)")
    f.set_defaults(func=_guarded(_fetch))
    c = commands.add_parser("credits", parents=[common], help="write the attribution file (CREDITS.md)")
    c.add_argument("--out", type=Path, help="DATASET/CREDITS.md (default: print it)")
    c.set_defaults(func=_guarded(_credits))
