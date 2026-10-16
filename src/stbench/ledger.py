"""Accumulates placements and states as the generator moves the world forward in time."""

from __future__ import annotations

from .geometry import q, quat_z
from .timeline import Placement, State


class Ledger:
    """Open intervals close when the next one starts; those still open at the end run to the end (t1=None). One that
    would close at its own start is dropped instead of kept as an empty interval (which `validate` rejects): a script
    may re-place an entity, or set a state again, at the instant the world first did."""

    def __init__(self) -> None:
        self._placements: list[Placement] = []
        self._states: list[State] = []
        self._open_p: dict[str, Placement] = {}
        self._open_s: dict[tuple[str, str], State] = {}

    def place(self, entity: str, t: float, pos: tuple[float, float, float], yaw: float, on: str | None) -> None:
        """Start (or replace) the entity's placement: the previous one closes at t, or is dropped if it began at t.
        `pos` is the base centre."""
        self.remove(entity, t)
        self._open_p[entity] = Placement(entity, q(t), None, (q(pos[0]), q(pos[1]), q(pos[2])), quat_z(yaw), on)

    def remove(self, entity: str, t: float) -> None:
        """End the entity's current placement at t (dropping it if it began at t); the entity no longer exists unless
        placed again."""
        cur = self._open_p.pop(entity, None)
        if cur is not None and cur.t0 != q(t):
            self._placements.append(Placement(cur.entity, cur.t0, q(t), cur.pos, cur.rot, cur.on))

    def set_state(self, entity: str, key: str, t: float, value: object) -> None:
        """Set the state from t on: the previous value ends at t, or is dropped if it was set at t (the last wins)."""
        cur = self._open_s.pop((entity, key), None)
        if cur is not None and cur.t0 != q(t):
            self._states.append(State(cur.entity, cur.key, cur.t0, q(t), cur.value))
        self._open_s[(entity, key)] = State(entity, key, q(t), None, value)

    def finish(self) -> tuple[list[Placement], list[State]]:
        """Placements and states sorted by entity id then start time."""
        placements = self._placements + list(self._open_p.values())
        states = self._states + list(self._open_s.values())
        return (
            sorted(placements, key=lambda p: (p.entity, p.t0)),
            sorted(states, key=lambda s: (s.entity, s.key, s.t0)),
        )
