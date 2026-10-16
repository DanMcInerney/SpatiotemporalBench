"""The visibility log as the compiler reads it: per entity, the keyframes where it was seen and where only glimpsed.

`private/<id>/visibility.jsonl` lists every entity with at least one pixel (`visibility.py`). Here a pixel count becomes
"seen" (>= `rules.gold_seen_px`, 269 px at 640x480), "band" (`rules.glimpse_px` up to that, 1..268) or "unseen" (under
`glimpse_px`: 0, or a glimpse too small to recognise). That is stricter than the oracle channel's `visibility.status`
(215 px), on purpose (DECISIONS: a sighting decides a gold only from 269 px): a gold then holds for a system that
detects anywhere from 1 to 269 px, because whatever hinges on a 1..268 px record is skipped.

The bounds are the entity's kind's (`seen_table`, through `kind_of`): with no kind, or a kind with no entry of its own,
the global ones above. With the identity table every kind has the global ones.

`look(entity, k)` also gives the pixel count and the mean luminance (`visibility.py`'s `lum`) a counted sighting had at
a keyframe. Recall never reads it; the Perceive set's recognisability floor does (`perceive_world`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from .camera import Camera
from .rules import glimpse_px, gold_seen_px
from .timeline import Timeline
from .visibility import read

KindOf = Callable[[str], str | None]


def kinds_of(tl: Timeline) -> KindOf:
    """`kind_of` for a timeline: an entity id to its kind (None for an id the timeline does not hold)."""
    return {e.id: e.kind for e in tl.entities}.get


class Sightings:
    """`seen(e)` and `band(e)` are sorted keyframe lists; an entity absent from a keyframe's record had 0 px there, and
    one under its kind's glimpse bound is in neither list. `kind_of(entity id)` gives the kind whose bounds apply (None,
    the default: the global bounds for every entity)."""

    def __init__(self, rows: Iterable[dict], cam: Camera, kind_of: KindOf | None = None):
        self._seen: dict[str, list[int]] = {}
        self._band: dict[str, list[int]] = {}
        self._look: dict[str, dict[int, tuple[int, float | None]]] = {}  # entity -> keyframe -> (px, lum) of a sighting
        self.frames: set[int] = set()
        bounds: dict[str | None, tuple[int, int]] = {}  # kind -> (glimpse_px, gold_seen_px)
        for row in sorted(rows, key=lambda r: r["k"]):
            self.frames.add(row["k"])
            for rec in row["entities"]:
                kind = None if kind_of is None else kind_of(rec["id"])
                if kind not in bounds:
                    bounds[kind] = (glimpse_px(cam, kind), gold_seen_px(cam, kind))
                lo, sure = bounds[kind]
                px = rec["px"]
                if px >= lo:  # lo is at least 1, so 0 px is never a sighting
                    (self._seen if px >= sure else self._band).setdefault(rec["id"], []).append(row["k"])
                    self._look.setdefault(rec["id"], {})[row["k"]] = (px, rec.get("lum"))

    @classmethod
    def load(cls, path: str | Path, cam: Camera, kind_of: KindOf | None = None) -> Sightings:
        return cls(read(path), cam, kind_of)

    def seen(self, entity: str, before: int | None = None) -> list[int]:
        """Keyframes (< `before` when given) where the entity covered its kind's `rules.gold_seen_px` pixels or more."""
        ks = self._seen.get(entity, [])
        return ks if before is None else [k for k in ks if k < before]

    def band(self, entity: str, before: int | None = None) -> list[int]:
        """Keyframes where it covered its kind's `rules.glimpse_px` pixels or more (1 on the global bound) but under
        `rules.gold_seen_px` (the dead band)."""
        ks = self._band.get(entity, [])
        return ks if before is None else [k for k in ks if k < before]

    def look(self, entity: str, k: int) -> tuple[int, float | None] | None:
        """`(px, lum)` of the entity's sighting at keyframe k: its pixel count and the mean luminance of its pixels
        (None when the log's row has no `lum`); None when it was under its kind's glimpse bound there (no sighting)."""
        return self._look.get(entity, {}).get(k)
