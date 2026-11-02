"""Typed, sequenced events. Every decision the robot makes is recorded with its reason, so the
Inspector and the evaluation read the same record and nothing on screen is invented."""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = '1.0'

TYPES = {
    'MISSION_ACCEPTED', 'OBSERVED', 'BELIEF_UPDATED', 'OBSERVE_REQUESTED', 'PICK_PLANNED',
    'GRIP_OK', 'GRIP_FAILED', 'LIFTED', 'PLACED', 'HANDOVER', 'PILE_CHANGED', 'PLACEMENT_CHECKED',
    'SAFETY_STOP', 'SAFETY_CLEARED', 'MOVE_REJECTED', 'DROP_DETECTED', 'DECISION_DEFERRED',
    'MISSION_COMPLETE', 'MISSION_INCOMPLETE', 'PLACEMENT_REPAIRED', 'LOAD_MEASURED', 'REORIENTED', 'PLACEMENT_SETTLED', 'AIMED', 'REGRIPPED',
    'FORK_IN', 'FORK_SKIPPED', 'FORK_LANDED', 'FORK_UNLOADING', 'FORK_RETRACTING', 'FORK_RETRIED', 'FORK_OUT', 'RELEASED', 'NEAR_MISS', 'VACUUM_LOST', 'CONTACT_LOAD', 'DAMAGE_RISK', 'COLLISION',
}


@dataclass
class Event:
    seq: int
    t: float                      # simulation seconds
    type: str
    data: dict = field(default_factory=dict)
    episode: str = ''
    schema: str = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=_jsonable)


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
    except ImportError:
        pass
    return str(o)


class EventLog:
    """Append-only, thread-safe; optional JSONL file and live subscribers (the Inspector)."""

    def __init__(self, episode: str, path: Path | None = None):
        self.episode = episode
        self.events: list[Event] = []
        self._lock = threading.Lock()
        self._subs = []
        self._fh = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open('w', encoding='utf-8')

    def emit(self, t: float, type_: str, **data) -> Event:
        if type_ not in TYPES:
            raise ValueError(f'unknown event type {type_}')
        with self._lock:
            e = Event(len(self.events), round(float(t), 4), type_, data, self.episode)
            self.events.append(e)
            if self._fh:
                self._fh.write(e.to_json() + '\n')
                self._fh.flush()
            subs = list(self._subs)
        for s in subs:
            try:
                s(e)
            except Exception:   # a broken viewer must never stop the robot
                pass
        return e

    def subscribe(self, fn):
        with self._lock:
            self._subs.append(fn)

    def of(self, type_: str):
        return [e for e in self.events if e.type == type_]

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None
