"""Synchronised trapezoidal joint-space moves for the gantry (x, y, z, yaw) and its wrist (pitch, roll).

Every axis shares the same accel / cruise / decel phase times, so the tool moves along
a straight line in joint space. This keeps the carried case on a predictable path that
the safety supervisor can check before the move starts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Profile:
    start: np.ndarray
    goal: np.ndarray
    t_acc: float
    t_cruise: float

    @property
    def duration(self) -> float:
        return 2 * self.t_acc + self.t_cruise

    def _s(self, t: float) -> float:
        """Normalised path parameter in [0, 1]."""
        T = self.duration
        if T == 0 or t >= T:
            return 1.0
        if t <= 0:
            return 0.0
        ta, tc = self.t_acc, self.t_cruise
        peak = 1.0 / (ta + tc)            # normalised cruise speed (area under v-profile is 1)
        acc = peak / ta if ta > 0 else 0.0
        if t < ta:
            return 0.5 * acc * t * t
        if t < ta + tc:
            return 0.5 * acc * ta * ta + peak * (t - ta)
        r = T - t
        return 1.0 - 0.5 * acc * r * r

    def position(self, t: float) -> np.ndarray:
        return self.start + (self.goal - self.start) * self._s(t)


def _check(x, name, n=None):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or (n is not None and x.shape != (n,)) or not np.all(np.isfinite(x)):
        raise ValueError(f'{name} must be {n or "a vector of"} finite numbers')
    return x


def tool_rotation(q):
    """Tool orientation for joints q (columns: tool x, y, z in the world): yaw, then the wrist's pitch
    (tool y) and roll (tool x)."""
    cy, sy = np.cos(q[3]), np.sin(q[3])
    cp, sp = np.cos(q[4]), np.sin(q[4])
    cr, sr = np.cos(q[5]), np.sin(q[5])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def limits_for(gantry, slow: bool, touch: bool = False, fine: bool = False):
    """Normal moves use the gantry limits; slow moves cap linear speed and acceleration; a
    touch-down search caps linear speed further, so contact is felt before it presses hard."""
    v, a = np.array(gantry.v_max, float), np.array(gantry.a_max, float)
    if slow or touch:
        v = np.minimum(v, np.concatenate([[gantry.slow_speed] * 3, v[3:] * 0.25]))
        a = np.minimum(a, np.concatenate([[gantry.slow_accel] * 3, a[3:] * 0.25]))
    if touch:
        v[:3] = np.minimum(v[:3], gantry.fine_touch_speed if fine else gantry.touch_speed)
    return v, a


def plan_move(start, goal, v_max, a_max) -> Profile:
    start = _check(start, 'start')
    n = len(start)
    goal = _check(goal, 'goal', n)
    v_max, a_max = _check(v_max, 'v_max', n), _check(a_max, 'a_max', n)
    if np.any(v_max <= 0) or np.any(a_max <= 0):
        raise ValueError('limits must be positive')
    d = np.abs(goal - start)
    if not np.any(d > 1e-12):
        return Profile(start, goal, 0.0, 0.0)
    # Plan on the normalised path s in [0, 1]: axis i moves d_i * s, so its speed and
    # acceleration are d_i times those of s. Bounding s by the tightest axis guarantees
    # every axis stays within its own limits.
    moving = d > 1e-12
    vs = float(np.min(v_max[moving] / d[moving]))
    as_ = float(np.min(a_max[moving] / d[moving]))
    if vs * vs / as_ <= 1.0:           # reaches cruise speed
        t_acc = vs / as_
        t_cruise = 1.0 / vs - t_acc
    else:                              # triangular
        t_acc = float(np.sqrt(1.0 / as_))
        t_cruise = 0.0
    return Profile(start, goal, float(t_acc), float(max(t_cruise, 0.0)))
