"""`map_reader`: the reference of a structure family (0.8.0, FD-task-questions review §2.3, §6), a system that holds the
place's layout and so answers how far the robot's body could move, or which way has room, without the sensors.

It is a reference and never a shortcut (`gates.MAP_READER`): on a group whose oracle rows all name it, the shortcut
gate and `separation` leave it, and `other_case`, out on every layout, and the check reports its score beside the
family's, as `kind_range`'s is beside `ego_now`'s. It says how much of the family a layout-memorising agent already
has, which only a held-out place takes away (VISION rule 5). `path_log`, which stays gated, answers such a family from
the robot's own log alone (`baselines_path_clear`, `baselines_path_heading`).

`MapReader` is `path_log`'s system (it keeps the log of what it was sent, reads no picture, and holds the public map,
the job's events and other scripts' records) answering a structure family's templates and `unknown` on every other.
Since 0.9.0 the families it answers are those of the modules of `MAP_MODULES` (`baselines_map_<group>`, paired with the
structure families), each defining `MAP_ANSWERERS: dict[str, MapAnswerer]`, found by `answer_registry.collect`: an
answerer is `(reader, question, rec, step) -> answer`, `rec` the record of the ask (`PathLog.ask_record`) and `step`
the log's entry at the ask's step. `baselines_map_clear` answers `clear_path` (the map's raycast from the moved start,
answered as it is; the pose-keyed pool where there is no map), `baselines_map_heading` `open_heading` (the best heading
of the map's raycast profile). On a held-out layout it holds no map (`Baseline.models_agent`).
"""

from __future__ import annotations

from collections.abc import Callable

from . import answer_registry
from .answer_candidates import UNKNOWN
from .baselines_path import PathLog

MAP_MODULES = ("baselines_map_clear", "baselines_map_heading")
MapAnswerer = Callable[["MapReader", dict, dict, dict], dict]   # (the reader, a question, its ask record, its step)


def map_registry(modules: tuple[str, ...] = MAP_MODULES) -> dict[str, MapAnswerer]:
    """The answerers of the modules among `modules` that exist, template by template; a template answered twice
    raises, naming it."""
    return answer_registry.collect(modules, "MAP_ANSWERERS")


MAP_ANSWERERS: dict[str, MapAnswerer] = map_registry()


class MapReader(PathLog):
    """The reference of a structure family: the public map's answer (the module docstring).
    Options as `PathLog`'s: `map` (or `kinds` on a held-out layout), `events` and `prior`."""

    def ask(self, when: str, now: dict, questions: list[dict]) -> dict[str, dict]:
        out = {}
        for q in questions:
            if q["template"] not in MAP_ANSWERERS:
                out[q["qid"]] = dict(UNKNOWN)
            else:
                out[q["qid"]] = MAP_ANSWERERS[q["template"]](
                    self, q, self.ask_record(q, now, when=when), self.log.steps[now["k"]])
        return out
