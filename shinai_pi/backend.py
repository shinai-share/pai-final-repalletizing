"""What the cognitive core may ask of a cell, and nothing more.

Implementations: `shinai_pi.sim.isaac_backend.IsaacBackend` (Isaac Sim physics and rendering)
and `shinai_pi.fake_backend.FakeBackend` (analytic, for fast logic tests). Neither exposes
ground truth through this interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .perception.geometry import CameraFrame
from .safety import Decision, TickState


@dataclass
class MoveResult:
    completed: bool
    reason: str = ''
    seconds: float = 0.0
    contact: bool = False          # stopped early because the axis met resistance (placing)
    guard: object = None           # what stopped a guarded move (the guard's own report)
    info: dict | None = None       # forces of the contact that ended the move: kind, static_N, peak_N (or raw)


Monitor = Callable[[TickState], Decision]
# A guard is looked at every few control ticks of a move: (stop?, report). The executive's guards
# use the downward camera to keep the path of the cups clear (requirements v05 R4).
Guard = Callable[[], tuple]


class CellBackend(Protocol):
    def now(self) -> float: ...
    def gantry_q(self) -> np.ndarray: ...
    def move(self, goal, monitor: Monitor, carrying: str | None = None, speed: float = 1.0,
             stop_on_contact: bool = False, guard=None, fine: bool = False) -> MoveResult: ...
    def idle(self, seconds: float, monitor: Monitor, carrying: str | None = None) -> MoveResult: ...
    def capture(self, camera: str) -> CameraFrame: ...
    def grip(self, close: bool) -> None: ...
    def gripped(self) -> list: ...
    def curtain_clear(self) -> bool: ...
    def operator_reset_requested(self) -> str | None: ...
    # Under-supports (requirements v05). Tool frame; extension is the tine top below the cup face.
    def fork_insert(self, extension: float, insert: float, monitor: Monitor) -> MoveResult: ...
    def fork_press(self, caps) -> tuple: ...
    def fork_loads(self) -> tuple: ...
    def fork_unload(self, monitor: Monitor) -> tuple: ...
    def fork_retract(self, monitor: Monitor) -> MoveResult: ...
