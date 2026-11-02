"""Safety supervisor: independent of the cognitive core, checked every control tick.

It never asks the AI. It stops on: light-curtain breach (latched until an operator resets with
the curtain clear), loss of the carried case, over-speed. Before a move it rejects targets
outside joint limits, plans built on stale observations, overweight cases, and traverses
without clearance. A software supervisor in simulation is not a certified safety function."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .config import Cell

AXES = ('x', 'y', 'z', 'yaw', 'pitch', 'roll')
SPEED_MARGIN = 1.25            # monitoring limit above the command limit (tracking overshoot is normal)
OVERSPEED_TICKS = 3             # consecutive control ticks over the limit before stopping
SPEED_WINDOW_S = 0.1           # the speed that stops is the distance moved over this window: a light wrist springing
                               # back 0.08 rad when the cups let go reads 2.5 rad/s for two physics steps (S3 seed 103)


VACUUM_LOST = '吸着が外れた（爪が支えている）'


class Action(Enum):
    CONTINUE = 'continue'
    STOP = 'stop'


@dataclass
class TickState:
    t: float
    q: np.ndarray
    v: np.ndarray
    carrying: str | None
    gripped: list
    curtain_clear: bool
    supported: bool = False        # the tines are under the carried case and take its weight


@dataclass
class Decision:
    action: Action
    reason: str = ''


class SafetySupervisor:
    def __init__(self, cell: Cell):
        self.cell = cell
        self.limits = np.array(cell.gantry.limits, dtype=float)
        self.v_max = np.array(cell.gantry.v_max, dtype=float)
        self.latched: str | None = None
        self.last_world_change = -np.inf
        self.stops: list = []
        self.vacuum_losses: list = []
        self._overspeed = 0
        self._recent = []              # (t, q) of the last SPEED_WINDOW_S

    # ---------------------------------------------------------------- every tick
    def check(self, s: TickState) -> Decision:
        if self.latched:
            return Decision(Action.STOP, self.latched)
        reason = None
        if not s.curtain_clear:
            reason = '作業区域に人が入った（光線式安全装置）'
        elif s.carrying is not None and s.carrying not in s.gripped:
            if s.supported:
                # The cups let go but the tines hold the case (design 9): stop and hold, not a latched
                # emergency; the executive seals again and goes on slowly.
                self.vacuum_losses.append((s.t, s.carrying))
                return Decision(Action.STOP, VACUUM_LOST)
            reason = f'搬送中のケース {s.carrying} の保持を失った'
        elif self._over_speed(s.t, s.q, s.v):
            k = int(np.argmax(np.abs(s.v) / self.v_max[:len(s.v)]))
            unit = 'm/s' if k < 3 else 'rad/s'
            reason = f'{AXES[k]} 軸の速度が上限を超えた（{abs(float(s.v[k])):.2f} {unit}）'
        if reason:
            self.latched = reason
            self.stops.append((s.t, reason))
            return Decision(Action.STOP, reason)
        return Decision(Action.CONTINUE)

    def _over_speed(self, t, q, v) -> bool:
        """Over the limit by the joint velocity for a few ticks in a row, and by the distance moved over
        the last SPEED_WINDOW_S: a runaway is both, a spring-back or a contact impulse only the first."""
        v, q = np.asarray(v, float), np.asarray(q, float)
        limit = self.v_max[:len(v)] * SPEED_MARGIN
        self._overspeed = self._overspeed + 1 if np.any(np.abs(v) > limit) else 0
        self._recent = [(tt, qq) for tt, qq in self._recent if t - tt <= SPEED_WINDOW_S + 1e-9] + [(t, q)]
        t0, q0 = self._recent[0]
        if self._overspeed < OVERSPEED_TICKS or t - t0 < SPEED_WINDOW_S - 1e-9:
            return False
        moved = np.abs(q[:len(v)] - q0[:len(v)]) / (t - t0)
        return bool(np.any(moved > limit))

    def reset(self, operator: str, curtain_clear: bool) -> bool:
        """Only an identified operator, with the area clear, may release a latched stop."""
        if not operator or operator == 'unknown' or not curtain_clear:
            return False
        self.latched = None
        return True

    # ---------------------------------------------------------------- before a move / pick
    def validate_target(self, q) -> tuple:
        q = np.asarray(q, dtype=float)
        if q.shape not in ((4,), (len(self.limits),)) or not np.all(np.isfinite(q)):
            return False, '目標が数値として不正'
        for i, (lo, hi) in enumerate(self.limits[:len(q)]):     # a gantry move names four axes
            if not lo - 1e-9 <= q[i] <= hi + 1e-9:
                return False, f'{AXES[i]} 軸の可動範囲外'
        return True, ''

    def note_world_change(self, t: float):
        self.last_world_change = max(self.last_world_change, float(t))

    def validate_plan(self, plan_stamp: float) -> tuple:
        if plan_stamp < self.last_world_change:
            return False, 'stale: 計画が最新の観測より古い'
        return True, ''

    def validate_payload(self, mass: float) -> tuple:
        if mass > self.cell.gripper.rated_payload:
            return False, f'可搬重量 {self.cell.gripper.rated_payload} kg を超える'
        return True, ''

    def validate_clearance(self, load_bottom_z: float, obstacle_top_z: float) -> tuple:
        if load_bottom_z - obstacle_top_z < self.cell.gantry.travel_clearance:
            return False, '搬送経路の下の余裕が足りない'
        return True, ''
