"""Transport event that collapses the load, expressed in the truck-deck frame.

In the deck frame a translational acceleration a(t) of the truck is physically identical to
an extra gravity term -a(t) (equivalence principle). So braking, sway and road bumps are
reproduced exactly by a time-varying gravity vector while the pallet stays put.
Only the top layers whose stretch film failed are free to move.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

G = 9.81
RAMP = 0.15            # s, rise/fall of the braking pulse (smooth to avoid an artificial impulse)


@dataclass(frozen=True)
class TransportProfile:
    seed: int
    heading: float           # rad, direction of travel in the deck frame
    brake_start: float
    brake_hold: float
    brake_decel: float
    lateral_amp: float
    lateral_freq: float
    vertical_amp: float
    vertical_freq: float
    phase: float
    duration: float
    loose_layers: tuple

    def _brake(self, t: float) -> float:
        t0, t1 = self.brake_start, self.brake_start + RAMP
        t2, t3 = t1 + self.brake_hold, t1 + self.brake_hold + RAMP
        if t <= t0 or t >= t3:
            return 0.0
        if t < t1:
            return 0.5 - 0.5 * math.cos(math.pi * (t - t0) / RAMP)
        if t <= t2:
            return 1.0
        return 0.5 + 0.5 * math.cos(math.pi * (t - t2) / RAMP)

    def truck_acceleration(self, t: float) -> np.ndarray:
        if t < 0 or t > self.duration:
            return np.zeros(3)
        fwd = np.array([math.cos(self.heading), math.sin(self.heading), 0.0])
        side = np.array([-fwd[1], fwd[0], 0.0])
        w_l = 2 * math.pi * self.lateral_freq
        w_v = 2 * math.pi * self.vertical_freq
        a = -self.brake_decel * self._brake(t) * fwd
        a = a + side * self.lateral_amp * w_l * w_l * math.sin(w_l * t + self.phase)
        a = a + np.array([0, 0, self.vertical_amp * w_v * w_v * math.sin(w_v * t + 2 * self.phase)])
        return a

    def effective_gravity(self, t: float) -> np.ndarray:
        return np.array([0.0, 0.0, -G]) - self.truck_acceleration(t)


def sample_transport(cfg, layers: int, seed: int) -> TransportProfile:
    rng = np.random.default_rng(seed)
    u = lambda r: float(rng.uniform(r[0], r[1]))  # noqa: E731
    n_loose = int(rng.integers(cfg.wrap_failure_layers[0], cfg.wrap_failure_layers[1] + 1))
    hold = 0.5
    start = u((0.4, 1.0))
    p = TransportProfile(
        seed=seed,
        heading=float(rng.choice([0.0, math.pi / 2, math.pi, -math.pi / 2]) + rng.normal(0, 0.15)),
        brake_start=start,
        brake_hold=hold,
        brake_decel=u(cfg.brake_decel),
        lateral_amp=u(cfg.lateral_amp),
        lateral_freq=u(cfg.lateral_freq),
        vertical_amp=u(cfg.vertical_amp),
        vertical_freq=u(cfg.vertical_freq),
        phase=u((0, 2 * math.pi)),
        duration=max(cfg.duration, start + hold + 2 * RAMP + 0.3),
        loose_layers=tuple(range(layers - n_loose, layers)),
    )
    # Clamp sway/bump amplitudes so the load is never lifted off (vertical gravity stays < -G/2)
    w_v = 2 * math.pi * p.vertical_freq
    if p.vertical_amp * w_v * w_v > 0.5 * G:
        p = TransportProfile(**{**p.__dict__, 'vertical_amp': 0.45 * G / (w_v * w_v)})
    return p
